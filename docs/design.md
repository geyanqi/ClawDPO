# ClawDPO v1 设计规范

> 状态：架构已确认，尚未开始实现。本文件是 v1 行为的唯一权威规范；术语定义见 [CONTEXT.md](../CONTEXT.md)，设计原因见 [ADR](adr/)。

## 1. 范围与硬边界

- ClawDPO 交付候选模型和评测报告，不负责发布、部署或修改线上模型别名。
- v1 只训练单轮回复：一条训练数据由同一 prompt/session 下的 chosen 和 rejected 构成。
- 数据库读取、脱敏、机器评测、vLLM rollout 和训练均由 owner 提供固定工具；Codex 只调用接口。
- 系统只维护 train/test。测试集可以被反复观察，但其原始样本永远不能进入训练数据。
- 每个自动迭代任务必须提供正整数 `max_iterations`；缺失或非法时拒绝启动。

## 2. 总体架构

```text
脱敏线上对话
    → curl1 正确性初筛并定位 badcase
    → Codex 选择值得训练的 prompt
    → Prompt Pool 调度节点
         v1：全量直通
         未来：冷却、淘汰或优先级调度（TODO）
    → 当前 Behavior Policy 对每个 prompt rollout 256 条
    → Correctness Gate + likelihood 分区
    → Candidate Slice + 全部 Chosen History
    → Codex 按 md1/md2 构造 Preference Pair
    → 冻结 Dataset Revision
    → DPO 训练 Candidate Model
    → Candidate Model 与 Best Model 在同一 Test Set 上比较
    → 明确提升则替换 Best Model，否则保留旧 Best Model
```

同一模型版本内，rollout、机器评测和候选筛选可以并行或异步执行；Dataset Revision 冻结后才允许启动训练。

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
| Chosen History 上限 | 无，全部附带 |
| `policy_lag` | `0`，历史回复必须由当前 policy 重评分 |
| Prompt Pool 调度 | v1 每轮全量重跑 |
| 最大循环轮次 | `max_iterations` 必填，无默认值 |

每次 rollout 只更换模型路径和输入 Prompt Pool；采样脚本中的数量和采样参数保持固定。

## 4. 三个信号严格分离

| 信号 | 回答的问题 | 负责人 |
|---|---|---|
| Correctness | 有没有幻觉、事实错误等底线问题 | `curl1` 按 `md2` 初筛；Codex 按同一 `md2` 兜底 |
| Response Quality | 在事实可靠的前提下，哪条回复更好 | `curl2` 按 `md1` 初筛；Codex 按同一 `md1` 终判 |
| Policy Likelihood | 当前模型多容易生成这条回复 | vLLM raw logprob 与候选筛选脚本 |

通过 Correctness Gate 只表示回复没有底线错误，不表示质量高。高 likelihood 也不代表正确或优质。

owner 提供两份唯一评测口径：

- `md1`：判断两个回复哪个更好。
- `md2`：判断回复是否存在幻觉或事实错误。

curl 脚本使用它们做快速初筛，Codex 使用相同文件做更强的最终判断。ClawDPO 不另外创建 rubric 或第三套评测服务。

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

除最多 32 条当前回复外，同一 prompt 的全部 Chosen History 必须全部附带，不能截断。每条历史 chosen 都包含当前 Behavior Policy 的重评分结果和当前分位。Codex 还会收到完整 session、likelihood 分布摘要、`md1` 和 `md2`；其余 rollout 原文只存档，不进入上下文。

## 7. Preference Pair 构造

### 7.1 准入条件

每个 pair 必须同时满足：

1. chosen 与 rejected 来自同一 prompt/session。
2. 两端都通过当前 policy 的可达性排除规则，不能位于 Extreme Tail。
3. chosen 先通过 `curl1`，再由 Codex 按 `md2` 确认没有事实性问题。
4. Codex 按 `md1` 明确判断 chosen 优于 rejected；拿不准就放弃该 pair。
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

## 9. 一轮完整流程

1. Codex 调用数据库工具，读取已经脱敏的线上对话。
2. `curl1` 批量扫描事实性底线问题，Codex 精读筛出的 session。
3. Codex 选择值得训练的单轮 prompt，加入 Prompt Pool。
4. Prompt Pool 调度节点在 v1 全量选中历史与新增 prompt。
5. 固定 vLLM 脚本让当前 Behavior Policy 为每个 prompt rollout 256 条。
6. Correctness Gate 检查全部回复；脚本计算并保存 raw likelihood 分布。
7. 候选筛选脚本构造最多 32 条当前 Candidate Slice，并重评分全部 Chosen History。
8. Codex 读取 session、分布摘要、Candidate Slice、Chosen History、`md1`、`md2`，构造有价值的 Preference Pair。
9. 没有有效 pair 的 prompt 跳过本轮训练；其余数据冻结为新的 Dataset Revision。
10. Codex 调用固定命令启动 DPO 训练，得到 Candidate Model。
11. Candidate Model 与 Best Model 在相同 Test Set prompt 上分别生成回复。
12. `curl1` 按 `md2` 检查正确性，`curl2` 按 `md1` 成对比较质量；Codex 使用相同口径终判。
13. Candidate 事实性底线不退步且 Codex 明确判断整体更好时，才替换 Best Model；并列、退步或不确定都保留旧 Best Model。
14. 未达目标时，根据 Test Set badcase 编写 SQL 规则，从数据库检索相似但不同的训练数据，进入下一轮。
15. 达到目标时输出模型与报告；达到 `max_iterations` 仍未达标时停止。

测试 badcase 只用于决定“去哪里找新的训练数据”，测试原始对话不能进入 Prompt Pool 或 Dataset Revision。

## 10. 工具边界

Codex 只通过固定工具访问外部能力。预期能力如下，名称与 JSON 字段在实现时以 owner 的真实脚本为准：

- `fetch_conversations`
- `sanitize_conversations`（可选）
- `evaluate_correctness_curl1`
- `compare_quality_curl2`
- `rollout_vllm`
- `score_response_likelihood`
- `build_candidate_slice`
- `archive_dataset`
- `train_dpo`
- `evaluate_test`

数据库工具默认返回可直接处理的脱敏对话；如仍需额外处理，由 owner 提供独立脱敏工具。

如果要求 Codex 无法查看脚本源码、curl 内容和凭据，必须通过独立进程、容器或不同系统用户强制隔离，只暴露调用接口；prompt 中写“不要读取”不构成安全边界。

## 11. 模型晋级规则

Candidate Model 和 Best Model 必须在完全相同的 Test Set 上生成回复。晋级不使用加权总分，必须同时满足：

1. `curl1` 初筛和 Codex 终判均表明事实性底线没有退步。
2. 在 `curl2` 初筛基础上，Codex 按 `md1` 明确判断 Candidate 整体更好。

任一条件不满足、比较并列或 Codex 无法确定时，都保留原 Best Model。接受 Candidate 只改变下一轮的 Best Model，不代表生产发布。

## 12. 后续 TODO

以下问题已明确保留，但不阻塞 v1：

1. **Prompt Pool 规模化调度**：实际达到何种规模后不再适合每轮全量 rollout，以及届时选择冷却、淘汰还是优先级调度。
2. **工具适配格式**：开始实现每个工具时，先向 owner 索要真实脚本及其输入输出格式，再记录具体 JSON Schema 和调用示例；当前不预设字段。
