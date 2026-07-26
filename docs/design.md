# ClawDPO v1 设计规范

> 状态：基础工具和单轮 workflow 已实现，模型晋级由 Test Set 机器评测自动决定。本文件是 v1 行为的唯一权威规范；术语定义见 [CONTEXT.md](../CONTEXT.md)，设计原因见 [ADR](adr/)。

## 1. 范围与硬边界

- 全流程遵守 Minimum-Disruption Alignment：从当前 Behavior Policy 采样，只训练
  当前策略可达且偏好差异明确的数据，并用 Test Set 检查业务提升和能力回归。
- ClawDPO 交付候选模型和评测报告，不负责发布、部署或修改线上模型别名。
- v1 只训练单轮回复：一条训练数据由同一 prompt/session 下的 chosen 和 rejected 构成。
- 数据库读取、脱敏、快速机器评测、vLLM rollout 和训练均由 owner 提供固定
  工具；需要细查上下文和调用工具的局部分叉验证由 Codex-as-Critic 完成。
- 系统只维护 train/test。测试集可以被反复观察，但其原始样本永远不能进入训练数据。
- 每个自动迭代任务必须提供正整数 `max_iterations`；它限制 Training Iteration
  总数，不限制单个 Base Stage 必须成功。缺失或非法时拒绝启动。

## 2. 总体架构

```text
脱敏线上对话
    → curl1（md1）事实性初筛并定位 badcase
    → Codex 选择值得训练的 prompt
    → Prompt Pool 调度节点
         v1：全量直通
         未来：冷却、淘汰或优先级调度（TODO）
    → 当前 Behavior Policy 对每个 prompt rollout 256 条
    → Correctness Gate + likelihood 分区
    → Candidate Slice + 全部 Chosen History
    → Codex-as-Critic 定位 high_fail 的候选分叉
    → 固定原 token 与唯一替代 token，各续写 16 次
    → Codex-as-Critic 精评两组的事实性、任务完成情况和质量
    → Codex 按 md1 查事实、按 md2 比质量，构造 Preference Pair
    → 冻结 Dataset Revision
    → DPO 训练 Candidate Model
    → Candidate Model 与 Best Model 在同一 Test Set 上比较
         通过：结束当前 Base Stage，Candidate 成为新 Base
         未通过：保留原 Base，Codex 先诊断数据、再诊断训练动态
    → Test badcase + 数据检索 prompt 生成 WHERE 正则
    → 数据库拉取独立对话 + Test Set messages 序列排除
    → 下一次 Training Iteration
```

同一模型版本内，rollout、机器评测和候选筛选可以并行或异步执行；Dataset Revision 冻结后才允许启动训练。

ClawDPO 最重要的产出是多个不可变的 Training Triple：
`(Behavior Policy / Base Model, Dataset Revision, Candidate Model Path)`。模型
晋级只决定下一轮使用哪个 Behavior Policy，不影响本轮三元组的永久存档。多个
Training Triple 串联成 Data Flywheel：模型生产自己的高质量数据，再由数据训练
下一模型。每一轮都以当轮 Behavior Policy 的生成分布为参照，避免把旧数据的
训练价值不加检查地沿用到新 policy。

一个 Base Stage 固定使用同一个 Best Model 作为 Base Model，可以连续产生多个
Training Triple。只有某个 Candidate 通过晋级条件才算该 Base Stage 成功结束；
失败 Candidate 留档并接受诊断，下一次 Training Iteration 仍从原 Base 开始。
达到最大轮次、零 pair 或命令失败可以终止任务，但不能把未晋级的 Base Stage
记录成成功。

代码只实现可审计的单轮工具链，不支持中断恢复。多轮不再实现第二套 Python
调度器；启动 Codex session 时发送 `prompt/迭代编排.md`，由该 session 根据
`max_iterations` 持续调用单轮入口、读取中间产物并推进下一轮。

## 3. v1 固定参数

| 参数 | v1 规则 |
|---|---|
| 每个 prompt 的 rollout 数 | 固定 256，不提前停止 |
| `temperature` | `1` |
| `top_p` | `1` |
| top-k | 关闭 |
| 额外 logit penalty | 不使用 |
| vLLM logprob 语义 | `raw_logprobs` |
| likelihood 排名分数 | raw 平均 token logprob |
| 单 prompt 当前候选上限 | 32 |
| 每个分叉 token 的续写数 | 固定 16；两组共 32 条 |
| Chosen History 上限 | 无，全部附带 |
| `policy_lag` | `0`，历史回复必须由当前 policy 重评分 |
| Prompt Pool 调度 | v1 每轮全量重跑 |
| 最大循环轮次 | `max_iterations` 必填，无默认值 |

每次 rollout 只更换模型路径和输入 Prompt Pool；采样脚本中的数量和采样参数保持固定。

## 4. 三个信号严格分离

| 信号 | 回答的问题 | 负责人 |
|---|---|---|
| Correctness | 有没有幻觉、事实错误等底线问题 | `curl1` 按 `md1` 评测；构造 pair 时由 Codex 兜底 |
| Response Quality | 在事实可靠的前提下，哪条回复更好 | `curl2` 按 `md2` 比较；构造 pair 时由 Codex 终判 |
| Policy Likelihood | 当前模型多容易生成这条回复 | vLLM raw logprob 与候选筛选脚本 |

通过 Correctness Gate 只表示回复没有底线错误，不表示质量高。高 likelihood 也不代表正确或优质。

owner 提供两份唯一评测 prompt：

- `md1`：事实性检测 prompt。输入原始问题/session 和一条待评回复，判断它是否出现幻觉、事实错误或其他事实性底线问题。
- `md2`：回复好坏比较 prompt。输入相同问题/session 下的两条回复，判断哪条回复回答得更好。

`curl1` 使用 `md1`，`curl2` 使用 `md2`。构造训练 pair 时，Codex 使用唯一入口
`prompt/codex/训练对构造.md`：一个 subagent 接收一个 prompt/session 的完整
Candidate Packet，在组内同时复核事实性、比较回复质量并直接产出 Preference
Pair。外层编排持续拉满可用 subagent 槽位。训练后的模型晋级只汇总 curl 机评结果，不再
经过 Codex。高概率错误回复的局部分叉验证链另外使用两次 Codex-as-Critic，但
这些结果只证明数据价值，不参与 Candidate Model 的晋级判定。

## 5. Rollout 与 likelihood

### 5.1 完整记录

每条新生成回复必须保存：

- 生成它的 Behavior Policy 版本。
- 逐 token raw logprob。
- raw cumulative logprob。
- raw 平均 token logprob。
- token 数量和结束原因。

历史回复通过 vLLM `prompt_logprobs` 在当前 Behavior Policy 下 teacher-force 重评分。新旧回复使用相同的 raw 语义；极小概率只保存 logprob，不转换为容易下溢的浮点概率。

### 5.2 prompt 内分区

只在同一 prompt 的 256 条回复内部，根据 raw 平均 token logprob 排名：

| 区域 | 分位 | 用途 |
|---|---:|---|
| High Likelihood | Top 25% | 优先寻找值得压低的高概率 rejected，也保留高概率优质回复 |
| Middle | 25%–75% | 完整存档，默认不占候选预算 |
| Supported Tail | 5%–25% | 优先寻找少见但可达的高质量 chosen |
| Extreme Tail | Bottom 5% | 只存档，不进入当轮 pair |

raw cumulative logprob 只用于审计，不参与分区，避免长回复因为 token 更多而天然被判为低概率。不做跨 prompt 全局阈值、logprob 断层检测、语义邻域聚类或额外 sampler-reachable 检查。

Bottom 5% 即 v1 的可达性排除规则。极端稀有的好回复先存档，本轮从 Supported Tail 选择可达的桥接回复，让模型逐轮把更好行为抬高。

## 6. Candidate Slice

完整 256 条 rollout 都运行 Correctness Gate 并存档，但 Codex 只查看以下当前候选：

| 来源 | 上限 | 排序与用途 |
|---|---:|---|
| Supported Tail + pass | 16 | 低概率、正确的 chosen 候选 |
| High Likelihood + fail | 8 | likelihood 最高的 Absolute Reject 候选 |
| High Likelihood + pass | 8 | likelihood 最高的优质或 Relative Reject 候选 |

Correctness pass/fail 与 High Likelihood/Supported Tail 共同形成四个 Retrieval Quadrant；其中 Supported Tail + fail 默认只存档，不占 Codex 候选预算。四格只是检索索引，不是 pair 类型或封闭训练规则。

某个候选区域不足时有多少取多少，不从 Middle 区补齐。当前 rollout 只做“移除空白后文本完全一致”的精确去重，不做语义去重。

因此 256 只是上游 rollout 数量，不是 Codex 输入数量。Codex 不读取完整
`evaluated-rollouts.jsonl`；likelihood 边际裁剪后，每组最多输入 32 条当前回复。

除最多 32 条当前回复外，同一 prompt 的全部 Chosen History 必须全部附带，不能截断。每条历史 chosen 都包含当前 Behavior Policy 的重评分结果和当前分位。Codex 一次收到该组完整 session、likelihood 分布、全部机筛 responses 与 Chosen History，在同一个任务里完成事实性复核、质量比较和 pair 构造；其余 rollout 原文只存档，不进入上下文。

## 7. Preference Pair 构造

### 7.1 准入条件

每个 pair 必须同时满足：

1. chosen 与 rejected 来自同一 prompt/session。
2. 两端都通过当前 policy 的可达性排除规则，不能位于 Extreme Tail。
3. chosen 先通过 `curl1`，再由 Codex 在单组任务中确认没有事实性问题。
4. 同一个 Codex 任务必须明确判断 chosen 优于 rejected；拿不准就放弃该 pair。
5. rejected 优先是当前 policy 高概率生成、确实值得压低的行为。
6. 历史回复参与前必须由当前 Behavior Policy 重评分，不能沿用旧 likelihood。

禁止手工编造当前模型根本采不到的“超级坏回复”。普通 DPO loss 不会因为 rejected 实际上几乎不可能出现就忽略它，喂入这种数据仍会产生有害梯度。

### 7.2 常见来源

pair 来源不是封闭白名单，常见的高价值关系包括：

1. **修事实错误**：高概率幻觉或事实错误作为 rejected，事实可靠且更好的回复作为 chosen。
2. **抬高回答质量**：Supported Tail 中正确、高质量的回复作为 chosen，高概率但质量普通的正确回复作为 rejected。
3. **超越历史 chosen**：当前 rollout 出现更好回复时，只要某条历史 chosen 在当前 policy 下仍是高 likelihood，就可以把历史 chosen 作为 rejected。
4. **其他清晰排序**：只要满足全部准入条件，并能明确教会当前 policy“更应该回答什么、更不应该回答什么”，Codex 可以构造其他 pair。

## 8. 历史数据与版本

- Prompt Pool 持续增长；v1 每轮对全部历史 prompt 和新增 prompt 重新 rollout 256 条。
- 历史上积累的是 prompt、完整 Chosen History 和版本化结果，不是永远重复训练的旧 pair。
- 某个历史 prompt 本轮找不到有效的新 pair，就不进入本轮 Dataset Revision，但下轮仍会重新 rollout。
- 每版 prompt、完整 rollout、Candidate Slice、Preference Pair、模型和评测结果都必须不可变存档，不能只覆盖 `latest.jsonl`。
- Dataset Revision 冻结后才能训练，并必须记录它对应的 Behavior Policy 和 Training Iteration。
- 每次完成训练都必须保存 Behavior Policy、Dataset Revision 路径和 Candidate
  Model 路径组成的 Training Triple；Candidate 未晋级也不能删除该三元组。
- 同一个 Base Stage 内的失败不会改变 Behavior Policy；下一次 Training
  Iteration 仍从该 Base Model 生成数据并启动训练。
- Candidate 或最终模型效果不好，不会使历史 Dataset Revision 失效。后续可以
  回到可靠的 Base Model，按各数据的来源模型记录重新选择和组合 Dataset
  Revision，再启动一次训练。

## 9. 一轮完整流程

1. Codex 调用数据库工具，读取已经脱敏的线上对话。
2. `curl1` 批量扫描事实性底线问题，Codex 精读筛出的 session。
3. Codex 选择值得训练的单轮 prompt，加入 Prompt Pool。
4. Prompt Pool 调度节点在 v1 全量选中历史与新增 prompt。
5. 固定 vLLM 脚本让当前 Behavior Policy 为每个 prompt rollout 256 条。
6. Correctness Gate 检查全部回复；脚本计算并保存 raw likelihood 分布。
7. 候选筛选脚本构造最多 32 条当前 Candidate Slice，并重评分全部 Chosen History。
8. Branch Localization Critic 为每条 `high_fail` 定位最早语义分叉；脚本校验
   原文字符范围，并用 Behavior Policy tokenizer 映射到原 response token。
9. 对每个可定位点固定原 token 和唯一替代 token，各续写 16 次；一个 Branch
   Outcome Critic 同时读取两组共 32 条匿名回复，逐条判断事实性和任务完成情况，
   再判断哪一组整体质量更好。
10. 每个 Codex subagent 读取一个完整 Candidate Packet，在同一次任务中复核
   事实性、比较回复质量并构造 Preference Pair；外层编排持续拉满可用 subagent
   槽位，但不会拆分一个 prompt 组。
11. 没有有效 pair 的 prompt 跳过本轮训练；其余数据冻结为新的 Dataset Revision。
   如果所有 prompt 都没有有效 pair，则立即结束整个循环，不训练、不评测、不拉
   新数据。
12. Codex 调用固定命令启动 DPO 训练，得到 Candidate Model，并保存本轮
    Training Triple。
13. Candidate Model 与 Best Model 在相同 Test Set prompt 上分别生成回复。
14. `curl1` 按 `md1` 检查两版回复的事实性，`curl2` 按 `md2` 逐题比较回复质量。
15. Candidate 的事实性失败数不高于 Best 且质量胜场高于 Best 时自动替换 Best，
    当前 Base Stage 成功结束。
16. Candidate 未晋级时，Base 和 Candidate 分别重评分本轮 Dataset Revision；
    固定脚本生成 Diagnosis Packet，Codex 先检查数据、再检查训练动态，并只规定
    下一次尝试的一项改动。失败 Candidate 不能成为下一次训练起点。
17. Codex 读取 Test Set 机评 badcase 和 `prompt/训练数据检索.md`，编写 PostgreSQL `WHERE` 正则。
18. 数据库拉回相似对话后，workflow 删除包含完整 Test Set messages 序列的原始样本，只把其余独立数据交给下一轮 Prompt Pool。
19. 晋级后以新 Best 开始下一个 Base Stage；未晋级时以原 Base 开始下一次
    Training Iteration。达到 `max_iterations` 仍未达标时停止。

测试 badcase 只用于决定“去哪里找新的训练数据”，测试原始对话不能进入 Prompt Pool 或 Dataset Revision。

## 10. 工具边界

外部能力通过薄脚本调用。当前入口包括：

- `infra/cli/curl.sh <request.json>`：调用 OpenAI-compatible 评测模型。
- `infra/cli/database.sh <where.sql>`：从固定数据表拉取对话。
- `infra/cli/dpo.sh <dataset-path>`：使用固定 ms-swift recipe 启动 DPO 训练。
- `infra/inference/rollout.py`：为每条 messages 固定生成 256 条回复。
- `infra/inference/rescore.py`：用当前 policy 重评分 Swift DPO pair。
- `infra/inference/branch_rollout.py`：从同一个 token prefix 重新生成
  Branch Trial。
- `workflow/select_candidates.py`：完成事实性初筛与 16/8/8 Candidate Slice。
- `workflow/locate_branch_points.py`：生成 Branch Localization Critic 任务，
  校验 Codex 结果并映射到原始 response token。
- `workflow/evaluate_branch_points.py`：生成 Branch Outcome Critic 匿名配对任务，
  校验 Codex 结果并筛出 Verified Branch Point。
- `workflow/build_pairs.py`：校验 Codex 选出的 pair 并冻结 Swift DPO 数据。
- `workflow/prepare_data.py`：汇总历史 chosen，并确定性地维护 Prompt Pool。
- `workflow/evaluate_test.py`：生成两版 Test Set 回复并汇总 curl1/curl2 结果。
- `workflow/build_diagnosis_packet.py`：汇总失败模型的 pair 重评分、Test badcase
  和训练日志。
- `workflow/run_iteration.py`：按阶段保存训练、机评晋级与 badcase 拉数产物。
- `workflow/run_report.py`：持续生成跨轮运行报告。
- `prompt/迭代编排.md`：Codex session 的多轮外层控制 prompt。
- `prompt/codex/训练对构造.md`：按 prompt 组并行完成事实复核、质量比较和黄金
  pair 构造。
- `prompt/codex/错误分叉定位.md`：Codex-as-Critic 定位最早语义分叉。
- `prompt/codex/分叉结果评测.md`：Codex-as-Critic 精评两组分叉回复。
- `prompt/codex/训练失败诊断.md`：Candidate 未晋级后确定主因和下一次单一改动。

前三个脚本位于 `infra/cli/`；训练 rollout 与重评分脚本位于
`infra/inference/`，`evaluate_test.py infer` 也直接调用 vLLM Python API。

### 10.1 机器评测

事实性检测和回复质量比较共用同一个 `curl.sh`。`curl1`、`curl2` 只是流程中的两种逻辑用途，不是两份脚本。endpoint 和鉴权分别通过 `OPENAI_API_URL`、`OPENAI_API_KEY` 传入；Codex 每次只准备一个请求文件，脚本内部用 `-d @<request-file>` 提交。

请求文件是一个完整的 OpenAI-compatible 请求体，至少承载以下内容：

- 要调用的评测模型。
- `messages`：包含 `prompt/事实性检测.md` 或 `prompt/回复竞对.md`，以及本次待评内容。
- owner 要求的 vLLM/OpenAI-compatible 推理参数。

两类请求的区别只在评测内容：

| 调用 | request file 中的评测 prompt | 本次待评内容 | 目的 |
|---|---|---|---|
| `infra/cli/curl.sh factuality-request.json` | `prompt/事实性检测.md` | 原始问题/session + 一条回复 | 判断是否存在事实性问题 |
| `infra/cli/curl.sh quality-request.json` | `prompt/回复竞对.md` | 原始问题/session + 两条候选回复 | 判断哪条回复更好 |

一份 request file 对应一次模型请求。批量初筛就是批量生成并提交这些文件，再汇总返回结果。

### 10.2 数据库

`database.sh` 固定执行以下查询头：

```sql
select create_time, conversation_detail
from openai_log_proxy
```

Codex 每次只写一个以 `WHERE` 开头、以分号结束的 SQL 文件，可以在其中使用时间条件和 PostgreSQL 正则。脚本拼接完整 SQL 后，以 `text/plain` POST 到 `DATABASE_API_URL`，要求服务返回 CSV；鉴权只通过 `DATABASE_API_KEY` 环境变量提供。

Codex 使用 `prompt/训练数据检索.md`，把 Test Set badcase 抽象成可复用的错误模式和检索正则，不能用完整测试原题做正向精确匹配。查询结果还必须经过 workflow 的 Test Set messages 序列排除；即使数据库记录在测试 prompt 后还带有原回复，也会被删除。prompt 约束和硬过滤共同保证测试原始样本不进入训练池。

仓库中的薄脚本不保存密钥。API key、数据库凭据和后端权限必须由运行环境隔离；如果脚本本身也必须不可读，再把 `infra/cli/` 替换成 executable-only mount。

### 10.3 vLLM 推理

rollout 输入每行是一条待采样对话：

```json
{"trace_id":"trace-1","messages":[{"role":"user","content":"问题"}]}
```

以下命令为每条输入固定生成 256 条回复：

```bash
python infra/inference/rollout.py prompts.jsonl rollouts.jsonl --model /path/to/model
```

输出仍以 trace 为一行。顶层 `prompt_token_ids` 直接取自 vLLM
`RequestOutput`，不是事后重新套 chat template；`engine` 保存 vLLM 版本、并行数
和 raw logprob 模式。`rollouts` 保存每条回复的文本、token IDs、逐 token raw
logprob、累计 raw logprob、平均 raw token logprob、token 数和结束原因。采样
固定使用 `temperature=1`、`top_p=1`、关闭 top-k 和所有额外 penalty；seed
默认为 0，可通过 `--seed` 修改。seed、最大生成长度和完整采样参数都会随结果
保存，并继续进入 Candidate Packet。

重评分输入直接采用 ms-swift DPO 格式，并额外保留 `trace_id`。chosen 是 `messages` 最后一条 assistant，rejected 位于 `rejected_response`：

```json
{"trace_id":"trace-1","messages":[{"role":"user","content":"问题"},{"role":"assistant","content":"chosen"}],"rejected_response":"rejected"}
```

```bash
python infra/inference/rescore.py pairs.jsonl rescored.jsonl --model /path/to/model
```

脚本用模型 chat template 确定 response token 边界，通过 vLLM `prompt_logprobs` teacher-force 重算 chosen 和 rejected。模板追加的 EOT/EOS 不计入 response 分数，结果写入 `policy_likelihood`。rollout 与重评分默认使用 8 卡 tensor parallel，且显式使用 `raw_logprobs`。

### 10.4 一轮 workflow

`run_iteration.py` 把一次 Training Iteration 的产物保存在独立目录：

```bash
python workflow/run_iteration.py rollout runs/iteration-001 prompts.jsonl --model /path/to/model
python workflow/run_iteration.py select runs/iteration-001 --request-template factuality-request.json
python workflow/run_iteration.py branch-prepare runs/iteration-001
python workflow/run_iteration.py branch-rollout runs/iteration-001 codex-locations.jsonl
python workflow/run_iteration.py branch-evaluate runs/iteration-001 codex-outcomes.jsonl
python workflow/run_iteration.py freeze runs/iteration-001 codex-draft-pairs.jsonl
python workflow/run_iteration.py train runs/iteration-001
python workflow/run_iteration.py evaluate runs/iteration-001 test-set.jsonl test-results.jsonl
python workflow/run_iteration.py diagnose runs/iteration-001 diagnosis.json
python workflow/run_iteration.py mine runs/iteration-001 badcase-where.sql
```

`diagnose` 只用于未晋级 Candidate；成功晋级时跳过。`evaluate` 总是把 manifest
中的 Behavior Policy 当作 Base Model，不再接收第二个可能冲突的模型路径。

`select` 会把 256 条回复全部送入 `curl.sh`，完整结果写入 `evaluated-rollouts.jsonl`，只把 Supported Tail + pass 16 条、High + fail 8 条、High + pass 8 条写入 `candidate-packets.jsonl`。已有机评结果时可用 `--factuality factuality.jsonl` 重放，格式为：

```json
{"trace_id":"trace-1","sample_index":0,"judge_model":"judge-model-version","judge_input_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","pass":true,"reason":""}
```

`judge_input_sha256` 绑定规范 JSON `{"messages":...,"response":...}`。脚本会重新
计算并严格核对，防止只凭 `trace_id + sample_index` 把旧机评套到新回复。同一
文件中的 `judge_model` 必须是同一个非空版本标识，并会写入门禁来源记录。

历史 chosen 先通过 `infra/inference/rescore.py` 用当前 policy 重评分，再用
`select --history rescored-history.jsonl` 全量附加。Codex 按
`prompt/codex/训练对构造.md` 给每个 subagent 一条完整 Candidate Packet，由它
同时复核事实性、比较候选质量并输出 Swift DPO draft pair。`build_pairs.py`
校验上下文和候选来源，要求当前 rollout chosen 通过 Correctness Gate、rejected
位于 High Likelihood、两端都不在 Extreme Tail；脚本不代替 Codex 判断回复质量。

每轮开始前，固定命令准备历史和 Prompt Pool：

```bash
python workflow/prepare_data.py history runs runs/iteration-001/history-source.jsonl
python infra/inference/rescore.py runs/iteration-001/history-source.jsonl runs/iteration-001/chosen-history.jsonl --model /path/to/current-best
python workflow/prepare_data.py prompts next-prompts.jsonl --test-set test-set.jsonl --pool prompts.jsonl --mined previous/mined-train.csv
```

`history` 扫描全部历史 Dataset Revision，不截断 chosen；`prompts` 从数据库结果
选择最后一个由 user 直接触发的 assistant turn，移除原回复和后续消息，并完成
稳定 trace ID、精确去重和 Test Set 序列排除。历史文件为空时跳过重评分。

`evaluate` 接收 curl1/curl2 汇总后的逐题机评：

```json
{"trace_id":"test-1","best_model":"/path/to/base","candidate_model":"/path/to/candidate","best_factuality_pass":true,"candidate_factuality_pass":true,"quality_winner":"candidate"}
```

`best_model` 和 `candidate_model` 必须分别与 manifest 的 Behavior Policy 和
Candidate Model 完全一致；`quality_winner` 只能是 `candidate`、`best`、`tie`
或 `uncertain`。结果必须恰好覆盖完整 Test Set。Candidate 事实性失败数不增加且
质量胜场更高时，workflow 自动把它记为下一轮 Best，并把决定写入
`promotion.json`；该动作不发布模型。Candidate 出现事实性失败或没有赢下质量
比较的题目会连同机评详情写入 `test-badcases.jsonl`，供数据检索 prompt 使用。

Test Set 回复和机评结果由固定命令生成：

```bash
python workflow/evaluate_test.py infer test-set.jsonl best-responses.jsonl --model /path/to/best
python workflow/evaluate_test.py infer test-set.jsonl candidate-responses.jsonl --model /path/to/candidate
python workflow/evaluate_test.py judge test-set.jsonl best-responses.jsonl candidate-responses.jsonl test-results.jsonl --factuality-template factuality-request.json --quality-template quality-request.json
```

`infer` 固定使用贪心解码和 seed 0，两版模型分进程加载；`judge` 为每条 Test Set
样本执行两次事实性检测和一次质量比较，并保留两版回复及机评原因。

Candidate 未晋级时，固定命令生成诊断证据：

```bash
python infra/inference/rescore.py runs/iteration-001/dataset.jsonl runs/iteration-001/base-pair-scores.jsonl --model /path/to/base
python infra/inference/rescore.py runs/iteration-001/dataset.jsonl runs/iteration-001/candidate-pair-scores.jsonl --model /path/to/candidate
python workflow/build_diagnosis_packet.py runs/iteration-001
```

Codex 按 `prompt/codex/训练失败诊断.md` 读取 packet 和其中列出的完整证据文件，
把结论写成固定 JSON，再调用 `run_iteration.py diagnose`。该命令只保存结论并
刷新报告，不改变机评结果或 Base Model。

`mine` 使用 Codex 读取 `test-badcases.jsonl` 并按 `prompt/训练数据检索.md` 写出的 `WHERE` 文件调用数据库，把原始返回存为 `database-response.csv`，再输出剔除 Test Set messages 序列后的 `mined-train.csv`。测试或复跑时可用 `--database-result result.csv` 跳过真实请求。排除数量记录在 `manifest.json`，测试原文只留在隔离的 Test Set 快照中，不写入训练文件。

每轮目录中的 `manifest.json` 记录当前阶段；Dataset Revision 冻结后不允许再次执行 `freeze`。
每次 manifest 更新都会刷新 `<runs_dir>/report.md`，其中持续记录阶段、prompt
数量、pair 数、跳过数量、机评、失败诊断、晋级结果和新拉数据量。报告顶部单独
列出每轮的 Base Model、High-quality Dataset Revision 和 Trained Model Path；需要时可用
`python workflow/run_report.py <runs_dir>` 重建。

多轮执行不是另一个代码入口。Codex 以 `prompt/迭代编排.md` 作为 session
启动消息，读取每轮 manifest 和 `promotion.json`：晋级时更新 Base Model，未
晋级时保持 Base Model 并执行诊断，然后持续调用上述固定命令，直到达到目标或
`max_iterations`。Codex 可以用 shell、
`jq` 或短小临时脚本做一次性查看与筛选，但不应重复实现这些每轮都会运行的大型
转换逻辑。

### 10.5 DPO 训练

`dpo.sh` 基于 ms-swift 官方 full DPO recipe。每次调用只传本轮 Dataset Revision 的本地路径：

```bash
infra/cli/dpo.sh /path/to/dataset.jsonl
```

训练 recipe 固定保存在脚本内。`MODEL_PATH` 指定本轮起始模型，默认是
`Qwen/Qwen3-30B-A3B`；`OUTPUT_DIR` 默认是 `output`。脚本默认使用 8 卡、单卡
batch size 1 和两步梯度累积，全局 batch size 为 16；GPU 列表仍可通过
`NPROC_PER_NODE` 与 `CUDA_VISIBLE_DEVICES` 覆盖。训练输出同时写入
`training.log`。失败诊断明确要求单变量重试时，才可覆盖
`DPO_LEARNING_RATE`、`DPO_BETA`、`DPO_NUM_TRAIN_EPOCHS` 或
`DPO_RPO_ALPHA`，一次只改一项。full DPO 还需要 reference model，实际能否装下
取决于单卡显存，启动前必须在目标机器验证。

本轮 Behavior Policy 必须同时作为 Candidate 的初始化模型和 DPO reference
model。训练以 pair 的相对偏好为主，并由固定 beta 约束相对 reference 的更新；
不得换用无关 reference 或绕开 Dataset Revision 直接对离线 chosen 做全量 SFT。
这是 Minimum-Disruption Alignment 在训练阶段的约束。

### 10.6 高概率错误回复的 token 分叉验证

token 分叉验证读取 `candidate-packets.jsonl` 中的 `high_fail` Absolute Reject，
不构造局部 SFT 或 Preference Pair。它借用 actor–critic 的角色划分：Behavior
Policy 负责生成，Codex-as-Critic 负责定位和结果判断。这里的 critic 是带项目
上下文与工具的外部评测者，不是 PPO 的可训练 value model，也不能替换成一次裸
GPT/API 请求。

流程中有两次 Codex-as-Critic：

1. `locate_branch_points.py prepare` 为每条 `high_fail` 生成一个
   Branch Localization Critic task。外层 Codex session 按
   `prompt/codex/错误分叉定位.md` 每条启动一个 subagent。subagent 可以使用搜索、
   代码和文档工具核实，只返回最早语义分叉的原文字符范围，或明确不可定位。
2. `locate_branch_points.py finalize` 从原 Candidate Packet 重算 `task_id`，
   拒绝缺失、重复、额外或过期结果，校验
   `response[start_char:end_char] == error_span`，再用 Behavior Policy tokenizer
   和原始 `token_ids` 映射出 Branch Point Candidate。Codex 不提供 token ID。
3. `branch_rollout.py` 使用首次 rollout 保存的 `prompt_token_ids`，加上
   `response_token_ids[:pivot_index]`，不重新套 chat template。脚本只排除一次
   原 token，采样出唯一替代 token，然后生成两组：

   ```text
   prefix + original_token    + 后缀 × 16
   prefix + alternative_token + 后缀 × 16
   ```

   同一个 replica 的两组使用相同续写 seed。原 bad response 不计入原 token 组。
   如果替代 token 本身直接结束生成，该点不能形成 16 个独立后缀，第一版拒绝这
   项实验。
4. `evaluate_branch_points.py prepare` 为每个分叉点生成一个匿名整组 task，不
   暴露 branch、token 或 critic 理由。一个 task 包含两组各 16 条回复。Branch
   Outcome Critic 按 `prompt/codex/分叉结果评测.md`，并遵守 `md1`、`md2`，
   逐条判断事实性和任务完成情况，再比较两组的稳定性与整体质量。一个 task 只能
   由同一个 Codex subagent 完整处理，不能拆成 16 个两两比较。
5. `evaluate_branch_points.py finalize` 重算全部 task/response ID，严格校验
   Codex 结果，再汇总两组的事实性通过率、任务完成率、严格通过率和质量胜负。
   `overall_pass = factuality_pass && task_pass`。

迭代编排中的标准命令为：

```bash
python workflow/run_iteration.py branch-prepare runs/iteration-001
# Codex subagents 处理 branch-localization-tasks.jsonl
python workflow/run_iteration.py branch-rollout \
  runs/iteration-001 codex-location-results.jsonl
# Codex subagents 处理 branch-outcome-tasks.jsonl
python workflow/run_iteration.py branch-evaluate \
  runs/iteration-001 codex-outcome-results.jsonl
```

`task_id` 由脚本根据实际输入和对应 prompt 内容生成，subagent 只负责原样复制。
finalize 不信任 task 文件或结果文件中的身份字段，而是从原 Candidate Packet 或
Branch Trial 重新计算。定位结果记录 Codex prompt 与结果文件哈希；结果精评还
记录 Codex prompt、`md1`、`md2` 和结果文件哈希。

迭代编排固定每组 16 条，`temperature=1`、`top_p=1`、关闭 top-k 和额外 penalty。
Verified Branch Point 必须同时满足：

- 两组都达到最小试验数；
- 替代 token 组的 `overall_pass` 率至少为 `0.6`；
- 替代组相对原 token 组的 `overall_pass` 率至少提高 `0.2`；
- 替代组事实性通过率不低于原 token 组；
- 匿名整组比较中，Codex-as-Critic 明确选择替代 token 组。

结果按相同 replica 记录“两边都失败、只替代组通过、只原组通过、两边都通过”。
16 次只是工程筛选下限，不能写成严格的统计显著性证明。

替代 token 的 `branch_token_raw_logprob` 使用 vLLM `raw_logprobs`，表示排除发生
前 Behavior Policy 对该 token 的原始 logprob，不是排除原 token 后重新归一化的
条件概率。

vLLM 公开接口不能复制原 rollout 在 pivot 处的 KV cache。这里验证的是“重新提交
相同 token prefix 后改变下一个 token”的结果，不能写成恢复原 decoder 状态后
原地分叉。

`--model` 必须指向不可变的 checkpoint 路径或版本 ID。脚本会校验来源模型和
分叉模型字符串一致，并保存 vLLM 版本与并行数，但不会为整套模型权重计算哈希；
不能用会被覆盖的 `latest` 别名冒充可复现实验。

第一版只定位回复中已经出现的局部事实错误，不把“过早停止导致内容缺失”映射成
EOS 分叉；Branch Localization Critic 对这类回复返回 `localizable=false`。
Verified Branch Point 只证明这个固定 token 改动在当前采样与 Codex 评测口径下
实测更有价值；它仍不是 good token、局部 SFT 样本或训练收益证明。

## 11. 模型晋级规则

Candidate Model 和 Best Model 必须在完全相同的 Test Set 上生成回复，逐条结果必须完整且唯一。晋级完全由机器评测汇总，不经过 Codex，也不使用加权总分，必须同时满足：

1. `curl1` 结果中，Candidate 的事实性失败数不高于 Best。
2. `curl2` 结果中，Candidate 的质量胜场严格高于 Best；`tie` 和 `uncertain` 不计入任一方胜场。

任一条件不满足都保留原 Best Model。接受 Candidate 只改变下一轮的 Best Model，不代表生产发布。

Test Set 必须同时覆盖目标业务偏好和项目要求保留的基模能力。上述晋级规则只能
证明 Candidate 在该 Test Set 的覆盖范围内没有观察到能力回退，不能证明基模的
全部能力都被完整保留。

未晋级不是 Base Stage 的成功终点。Codex 必须先对本轮 Training Triple 生成
Training Failure Diagnosis：存在直接工程错误时先修错误；否则优先检查 Dataset
Revision 的 chosen 质量、排序强度和样本分布，再检查 DPO margin、chosen 绝对
raw mean logprob、训练 loss、gradient norm 和 reward 指标。跨模型 raw
likelihood margin 与训练器的 DPO reward margin 分开记录，不能互相替代。诊断
只选择一个主因和下一次单一改动。新的 Training Iteration 继续使用原 Best
Model，直到某个 Candidate 晋级或整个任务因最大轮次、零 pair 或命令错误停止。

## 12. 后续 TODO

以下问题已明确保留，但不阻塞 v1：

1. **Prompt Pool 规模化调度**：实际达到何种规模后不再适合每轮全量 rollout，以及届时选择冷却、淘汰还是优先级调度。
