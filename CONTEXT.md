# ClawDPO

ClawDPO 负责把观察到的模型问题转化为偏好训练数据，并迭代出经过评测的候选模型；它的职责在生产发布之前结束。

## 模型与迭代

**Behavior Policy**:
本轮生成回复并为训练候选计算 Policy Likelihood 的确切模型版本。
_Avoid_: 最新模型、未注明版本的模型

**Candidate Model**:
一次 Training Iteration 产出的新模型，必须与 Best Model 完成比较后才能被接受。
_Avoid_: 新 Best、上线模型

**Best Model**:
当前保留的最强模型版本；比较结果并列或不确定时继续保留它。
_Avoid_: 最新 checkpoint、production model

**Accepted Candidate**:
通过 Test Set 机器评测的既定正确性与质量条件、可以替代 Best Model 的 Candidate Model；接受不代表发布。
_Avoid_: Released model、production model

**Training Iteration**:
从当前模型重新生成 Preference Pair、完成 DPO 训练并比较新旧模型的一次循环。
_Avoid_: Production rollout

**Asynchronous Preference Improvement**:
Behavior Policy 生成回复、外部评测与 Codex 构造 Preference Pair、DPO 学习下一版策略的 RL-like 循环；它不是 trajectory-level RL。
_Avoid_: 静态离线 DPO、完整 RL

**Policy Lag**:
提供候选准入 likelihood 的 policy 与当前待更新 policy 之间的版本距离。
_Avoid_: Stale likelihood、未重评分的历史回复

## 数据与历史

**Prompt Pool**:
持续积累、可由后续模型重新生成回复的新旧 prompt 集合；进入池子不代表每轮都会形成训练 pair。
_Avoid_: 训练 pair、当前数据集

**Rollout Set**:
一个 Behavior Policy 在固定采样配置下为同一 prompt 生成的完整回复集合。
_Avoid_: Candidate Slice、部分 rollout

**Candidate Slice**:
从当前 Rollout Set 中挑给 Codex 精审的有限候选集合；它不包含完整 rollout，也不是最终训练 pair。
_Avoid_: Full rollout dump、Preference Pair

**Chosen History**:
同一 prompt 历史上所有曾被选为 chosen 的回复，并保留各自来源模型版本。
_Avoid_: 只有最新 chosen、永久标准答案

**Quality Frontier**:
Chosen History 中当前质量最好的一条回复；它是比较基准，但不会覆盖其他历史 chosen。
_Avoid_: Ground truth、永久 chosen

**Preference Pair**:
Codex 为同一 prompt 选择的一组 chosen/rejected 排序关系；chosen 必须事实可靠且明确更好，两端都必须属于当前模型的 Practical Policy Support。
_Avoid_: 两条带分数的样本、Correctness Gate 输出

**Dataset Revision**:
一次训练实际使用并永久存档的不可变偏好数据版本。
_Avoid_: latest.jsonl、不断覆盖的数据目录

**Test Set**:
每次训练后用于比较模型的固定评测集合；其失败可以指导检索独立训练数据，但测试样本本身不进入训练。
_Avoid_: Training Set、一次性保密考试

## 评测与采样信号

**Evaluation Specs**:
owner 提供的两份 Markdown 评测口径：`md1` 定义事实性检测，`md2` 定义回复好坏比较；快速评测与 Codex 共用它们。
_Avoid_: ClawDPO 自创 rubric、另一套 Codex 标准

**Correctness Gate**:
按照 `md1` 检测幻觉、事实错误等底线问题的快速二元初筛；失败回复永远不能成为 chosen。
_Avoid_: 质量排序器、最终 pair selector

**Pairwise Quality Judge**:
按照 `md2` 比较同一 prompt 下两条回复好坏；训练 pair 由 Codex 终判，模型晋级直接汇总机评胜场。
_Avoid_: Correctness evaluator、scalar reward

**Response Quality**:
在事实可靠的前提下，一条回复把问题回答得多好的相对次序。
_Avoid_: 正确性、机评通过率

**Policy Likelihood**:
当前 Behavior Policy 产生某条回复的相对可能性；它不代表回复质量或事实正确性。
_Avoid_: Quality score、correctness confidence

**Likelihood Distribution**:
同一 prompt 的完整 Rollout Set 内，各回复 Policy Likelihood 形成的分布。
_Avoid_: 全局概率阈值、跨 prompt 排序

**Practical Policy Support**:
当前 Behavior Policy 在标准 rollout 下实际有机会产生的回复范围，而不是所有数学上非零概率的序列。
_Avoid_: 任意可构造回复、人工编造负样本

**Supported Tail**:
Likelihood Distribution 中较难出现但仍属于 Practical Policy Support 的低概率区域。
_Avoid_: 最低概率回复、off-policy answer

**Extreme Tail**:
Likelihood Distribution 中稀有到不适合直接进入当前 Preference Pair 的极端区域。
_Avoid_: Supported Tail、理想 chosen

**Retrieval Quadrant**:
由 Correctness Gate 结果与 Policy Likelihood 区域交叉形成的候选检索索引；它不是封闭的 pair 类型表。
_Avoid_: Sample Class、training role

**Absolute Reject**:
未通过 Correctness Gate、永久失去 chosen 资格的回复。
_Avoid_: Relative Reject

**Relative Reject**:
事实可靠但质量被另一条回复明确超过、因而可作为 rejected 的高概率回复。
_Avoid_: Correctness failure、Absolute Reject
