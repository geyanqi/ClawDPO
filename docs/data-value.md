# ClawDPO 数据价值证据

本文说明 ClawDPO 保存的 rollout 和 Preference Pair 为什么值得用于后续训练，
以及应该用哪些记录支撑这个判断。具体采样参数、候选分区和 pair 准入条件仍以
[设计规范](design.md)为准。

这里证明的是两件事：

1. 训练前，这些数据来自当前模型真实会生成的行为，并且好坏差异有明确证据，因而
   训练信号不需要脱离当前 policy 去重写模型行为。
2. 训练后，一次 Training Triple 产出的 Candidate 是否提升，由固定且不直接进入
   训练的 Test Set 验证，同时检查其覆盖范围内的能力回归。

单条数据无法只凭概率证明一定有效；模型晋级验证的是数据和训练配置共同作用后
得到的 Candidate，不能把结果单独归因给某条 pair 或整个 Dataset Revision。

本文暂不讨论 token 级局部 SFT 的训练方式。

## 1. 每条 response 保存什么

每条 rollout response 至少保留：

| 记录 | 作用 |
|---|---|
| `trace_id`、`sample_index` | 找回它所属的 prompt 和原始 rollout |
| `behavior_policy` | 明确是哪一版模型生成了它 |
| `prompt_token_ids`、`response`、`token_ids` | 保存实际送入 vLLM 的 prompt token 和回复 token，不靠事后重新分词 |
| `raw_token_logprobs` | 保存每一步实际生成 token 的原始 logprob |
| `raw_cumulative_logprob` | 审计整条回复的累计 logprob |
| `raw_mean_token_logprob` | 衡量当前模型生成这条回复的相对难易 |
| `num_tokens`、`finish_reason`、`stop_reason` | 解释长度差异、停止条件和异常截断 |
| vLLM 版本、并行数、logprob 模式 | 记录生成数据时的推理环境 |
| seed 和完整采样参数 | 说明这一批候选是在什么随机采样条件下生成的 |
| Correctness Gate 结果及输入哈希 | 判断回复是否存在事实性底线问题，并防止旧判定误套到新回复 |
| 评测模型、prompt 和 request template 哈希 | 记录实际使用的机器评测口径 |
| prompt 内分位和 likelihood 区域 | 判断回复位于 High Likelihood、Supported Tail 还是 Extreme Tail |

项目里所说的“平均 token 概率”，落盘时统一使用：

```text
raw_mean_token_logprob
    = 所有 response token 的 raw logprob 之和 / response token 数
```

如果报告必须展示成概率，可以计算：

```text
mean_token_probability = exp(raw_mean_token_logprob)
```

它是几何平均意义下的 token 概率。原始记录仍保留 logprob，避免极小概率出现
数值下溢，也避免把 token 概率直接做算术平均后得到难以解释的数字。

Policy Likelihood 只回答“当前模型多容易生成这条回复”，不回答回复是否正确或
是否优质。不同 prompt 的上下文和答案空间不同，因此 likelihood 只在同一
prompt 的 Rollout Set 内比较，不做跨 prompt 的绝对排名。

## 2. 两种后续复用价值

这两项不是 ClawDPO v1 当前执行的训练流程，只说明完整 rollout 还可以支持哪些
后续用途。

### 2.1 适合后续 GRPO 的 prompt

同一 prompt 的多次 rollout 中，正确回复和错误回复都稳定出现，说明当前模型在
这个问题上还没有完全学会，同时又具备解决它的能力。这样的 prompt 才可能为
后续 GRPO 提供有效反馈差异。

这里的正确/错误由机器评测得到：当前项目按照 owner 提供的 `md1`，让
Correctness Gate 对 Rollout Set 中的每条 response 给出 pass/fail，再按 prompt
汇总。它不是根据 token logprob 推测的，也不是使用 `md2` 的成对质量比较结果。
报告应同时保存评测口径和评测模型版本，保证后续能够复现。

每个 prompt 只需额外汇总：

```text
rollout 总数
Correctness Gate 通过数
Correctness Gate 失败数
通过率
通过和失败两组 response 的 raw_mean_token_logprob 分布
```

平均 token logprob 用来证明这些回复确实处于当前模型会生成的范围；真正说明
prompt 适合 GRPO 的，是同组 rollout 同时存在足够数量的正确和错误结果。只有
一次偶然正确或偶然错误，不能作为充分证据。

### 2.2 可作为整条回复 SFT 候选的数据

如果一条 response：

1. 通过 Correctness Gate；
2. 被确认是同组中的高质量回复；
3. 位于当前 Behavior Policy 的 Supported Tail；

那么它说明模型已经能够生成这种好回答，但生成得还不够稳定。这类完整回复可以
作为后续整条回复 SFT 的候选。

这里的重点仍然是同时保存“回复确实好”和“当前模型不容易生成”两类证据，不能
用低 likelihood 代替质量判断。Extreme Tail 只代表极其少见，不能自动视为更有
训练价值。

## 3. Preference Pair 是主要训练数据

ClawDPO 当前真正送入训练的是 Preference Pair，而不是孤立的 chosen 或
rejected。一个 pair 的价值来自同一 prompt 下明确的训练方向：

```text
chosen：事实可靠，而且回答得更好
rejected：当前模型确实容易生成，但事实错误或质量明显较差
```

每条用于训练的 Preference Pair 必须能够回答以下问题：

| 问题 | 必须保存的证据 |
|---|---|
| 模型当时为什么会犯这个错误？ | Behavior Policy 版本，以及 rejected 的 raw 平均 token logprob 和 prompt 内分位 |
| chosen 为什么值得提高概率？ | Correctness Gate 结果、成对质量判断和具体原因 |
| rejected 为什么值得降低概率？ | 事实错误证据，或它相对 chosen 的明确质量缺陷 |
| 两端是不是模型实际能生成的？ | 两端的来源记录和 likelihood 区域；两端都不得位于 Extreme Tail |
| 训练时到底用了哪两条原文？ | 冻结 Dataset Revision 中的 chosen 和 rejected 原文 |
| 这条 pair 属于哪次训练？ | Dataset Revision、Behavior Policy 和 Training Iteration |

历史 chosen 如果参与新一轮 pair，必须先由当前 Behavior Policy 重新计算
likelihood。旧模型下的概率不能拿来证明它在当前模型下仍有训练价值。

### 3.1 Pair 类型

当前项目记录四类清晰关系：

| `pair_type` | 数据价值 |
|---|---|
| `factual_repair` | 压低当前模型高概率生成的事实错误或幻觉 |
| `quality_lift` | 抬高少见但正确、质量更好的回答 |
| `history_upgrade` | 当前模型已经能超过历史 chosen，继续提高质量上限 |
| `other_clear_ordering` | 不属于前三类，但 chosen 与 rejected 的优劣仍然明确 |

其中 rejected 可以是：

- **Absolute Reject**：未通过 Correctness Gate，存在事实性底线问题。
- **Relative Reject**：事实可靠，但质量明确低于 chosen。

这两种 rejected 的含义不能混在一起。前者教模型避免错误，后者教模型在正确的
基础上回答得更好。

### 3.2 每条 Pair 的证据视图

训练文件继续使用 Swift DPO 所需的精简格式；数据价值报告可以把已有产物连接成
下面的证据视图：

```json
{
  "trace_id": "trace-1",
  "behavior_policy": "/path/to/base-model",
  "pair_type": "quality_lift",
  "reason": "chosen 更完整地回答了用户问题，且没有增加未经证实的断言",
  "chosen": {
    "response": "chosen 原文",
    "source": "rollout",
    "sample_index": 37,
    "correctness_pass": true,
    "raw_mean_token_logprob": -1.42,
    "likelihood_region": "supported_tail"
  },
  "rejected": {
    "response": "rejected 原文",
    "source": "rollout",
    "sample_index": 5,
    "correctness_pass": true,
    "raw_mean_token_logprob": -0.61,
    "likelihood_region": "high"
  },
  "dataset_revision": "runs/iteration-001/dataset.jsonl"
}
```

这不是新的训练格式，而是对现有文件的联合展示：

- `evaluated-rollouts.jsonl` 保存完整 response、likelihood 和正确性结果；
- `candidate-packets.jsonl` 保存进入精审的候选和历史 chosen；
- `chosen-history.jsonl` 保存历史 chosen 的来源迭代、来源模型和当前模型重评分；
- `draft-pairs.jsonl` 保存 `pair_type`、排序原因和 chosen/rejected 原文；
- `dataset.jsonl` 保存最终真正参与训练的 Preference Pair；
- `manifest.json` 保存 Behavior Policy、Dataset Revision 和 Candidate Model。

当前 rollout 产生的 pair 端点可以凭 `sample_index` 回到具体 rollout；历史 chosen
可以回到来源迭代、来源模型和当前重评分记录，但不保证能定位到当时唯一的原始
rollout 样本。最终仍可确认报告展示的 chosen/rejected 是否真正进入了训练，而
不是只展示若干未被使用的例子。

## 4. 一版数据应该汇总什么

每个 Dataset Revision 建议展示下面四组数字。

### 4.1 Rollout 覆盖

```text
prompt 总数
response 总数
每个 prompt 的固定 rollout 数
Correctness Gate 通过数和失败数
同时出现正确与错误 response 的 prompt 数
```

### 4.2 可复用数据

```text
适合后续 GRPO 研究的 prompt 数
Supported Tail 中正确且高质量的完整 response 数
Extreme Tail 排除数
```

这部分只说明数据具备后续用途，不把候选数量写成已经完成的训练收益。

### 4.3 DPO 数据

```text
最终训练 Preference Pair 总数
覆盖的 prompt 数
四种 pair_type 的数量
Absolute Reject 和 Relative Reject 的数量
chosen/rejected 的 prompt 内 likelihood 区域分布
因排序不明确、chosen 不可靠或超出可达范围而放弃的数量
```

还可以记录同一 pair 两端的：

```text
policy_likelihood_gap
    = chosen.raw_mean_token_logprob
    - rejected.raw_mean_token_logprob
```

当这个值小于零时，表示当前模型平均而言更容易生成 rejected，而不是 chosen。
这能直观说明训练方向的必要性，但它不是质量分，也不能跨 prompt 比较大小。

### 4.4 高概率错误回复的局部分叉证据

对 `high_fail` response 完成 token 分叉验证后，再汇总：

```text
Branch Localization Critic 能定位候选分叉点的 response 数
完成两组 Branch Trial 的候选点数
Verified Branch Point 数
原 token 组与固定替代 token 组的事实性、任务完成和严格通过率
严格通过率提升幅度、实际试验次数和质量胜负
固定替代 token 及其原始 logprob
```

这部分证据说明：某条高概率错误回复不只是“整体不好”，而且能够找到一个具体
生成位置；从首次 rollout 保存的同一组 prompt token 和 response prefix 重新
生成时，只改变下一个 token 的选择，在多次试验中会改变 Codex-as-Critic 判断的
实测通过率和回复质量。

Branch Localization Critic 只负责提供语义位置，不能单独证明分叉点有效。原
bad response 也不能直接算作原 token 组的一次失败，因为它本来就是按失败结果
选入的。脚本先排除一次原 token、确定唯一替代 token，再分别固定两个 token
重新 rollout。同一个 replica 的两条回复使用相同续写 seed。匿名的 Branch
Outcome Critic 一次读取两组共 32 条回复，逐条判断事实性和任务完成情况，再判断
哪一组整体更好。只有试验数、替代组严格通过率、提升幅度、事实性不退化，并且
替代组被明确判为整体更好时，才能写入 Verified Branch Point。

定位、生成和复评三段都保存内容绑定的 task ID、Codex prompt 与评测口径哈希、
Codex 结果文件哈希、vLLM 版本和实际采样参数。finalize 会从原始输入重算 ID，
不能只凭 `trace_id` 和序号套用旧结果。

替代组保存的 `branch_token_raw_logprob` 是禁止原 token 之前，Behavior Policy
对实际替代 token 给出的原始 logprob；它不是排除原 token 后重新归一化的条件
概率。

Codex-as-Critic 同时检查事实性、任务完成情况和整组质量，拒答、空泛或没有真正
回答问题的回复不能只靠“没有事实错误”过关。Verified Branch Point 也不是 good
token 或训练 pair，本文仍不讨论局部 SFT。
第一版也不把“过早结束导致内容缺失”硬映射成 EOS 分叉；这类问题继续标记为
不可局部定位。第一版的 16 次只是工程筛选下限，结论应写成“实测通过率达到
门槛”，不能写成已经获得严格的统计显著性。

### 4.5 训练后的整体验证

如果该 Dataset Revision 已经完成训练，再附上：

```text
Candidate 与 Best 的事实性失败数
Candidate / Best / tie / uncertain 的质量比较数量
Candidate 是否晋级
```

这是对该次 Training Triple 所产出 Candidate 的验收结果，不是对 Dataset
Revision 的单独归因。Candidate 没有晋级时，也应保留完整 Training Triple 和
失败诊断；结论应该是“本版数据与训练配置没有带来足够提升”，而不是在没有逐条
消融实验的情况下宣布所有 pair 都无效。

## 5. 可以对外成立的结论

ClawDPO 的数据价值不来自 rollout 数量本身，而来自一条完整证据链：

1. 每条 response 都能追溯到确切的 Behavior Policy 和原始 Rollout Set；
2. 每条 response 都保存模型生成它的 raw token logprob；
3. 正确性、回复质量和生成概率分别判断，互不代替；
4. Preference Pair 两端来自同一 prompt，并且都处于当前模型实际会生成的范围；
5. chosen/rejected 的排序理由和真正参与训练的原文都被永久保存；
6. 高概率错误回复可以额外生成经过多次反事实 rollout 验证的局部分叉证据；
7. 每次 Training Triple 产出的 Candidate 都由固定且不直接进入训练的 Test Set
   判断是否提升。

因此，这批数据不是静态收集的一堆“好答案”和“坏答案”，而是带有当前模型生成
概率、明确改进方向、完整来源记录和训练结果验证的版本化偏好数据。它的价值还在
于把训练约束在当前 policy 已有行为附近，只对有证据的业务偏好差异施加学习信号，
减少对无关能力的扰动。
