# ClawDPO

ClawDPO 是一套自动数据飞轮：它把观察到的模型问题转化为贴近当前策略、可验证、可复用的训练数据，并用受控训练和评测验收数据版本；它的职责在生产发布之前结束。

## 模型与迭代

**Behavior Policy**:
本轮生成回复、计算 Policy Likelihood 并作为 DPO 训练起点的确切模型版本；它也是 Training Triple 中的 Base Model。
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
从固定 Base Model 重新生成 Preference Pair、完成一次 DPO 训练并比较新旧模型的一次尝试；失败后可以在同一个 Base Stage 内开始新的 Training Iteration。
_Avoid_: Production rollout

**Base Stage**:
以同一个 Best Model 作为 Base Model 的连续训练阶段；只有某个 Candidate Model 通过 Test Set 晋级才算成功结束。失败的 Training Iteration 会被诊断和存档，但不会改变 Base Model。
_Avoid_: 一次训练命令、失败即换 Base、无限重试

**Training Triple**:
一次 Training Iteration 永久关联的 Behavior Policy、Dataset Revision 和 Candidate Model；它是 ClawDPO 最重要的版本化产物。
_Avoid_: 只有最终模型、只有最新数据集

**Training Failure Diagnosis**:
Candidate Model 未晋级后，对本轮 Training Triple 的失败原因分析；先排除明确的工程错误，再优先检查 Dataset Revision，最后检查训练动态，并为同一 Base Stage 的下一次 Training Iteration 提出一个可验证的单一改动。
_Avoid_: Codex 改判机评、同时改多个变量、失败后直接更换 Base Model

**Data Flywheel**:
从问题发现、当前策略采样、数据筛选、训练到验收自动循环形成的迭代链；晋级模型继续为自己生产下一版 Dataset Revision。
_Avoid_: 固定数据集反复训练、只有 checkpoint 的模型迭代

**Minimum-Disruption Alignment**:
以当前 Behavior Policy 的实际生成分布为参照，只学习有证据支持的必要偏好差异，并用 Test Set 检查回归；目标是在保留其已有能力的基础上完成业务偏好对齐。
_Avoid_: 脱离当前策略灌入答案、无约束改写模型分布、从头教授基模能力

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

**Promoted Dataset Revision**:
在固定 Base Model、训练配置和 Test Set 下产出 Accepted Candidate 的 Dataset Revision；模型晋级是该数据版本通过当前验收的操作性标准。
_Avoid_: 跨模型通用数据、单条样本的因果证明

**Test Set**:
每次训练后用于比较模型的固定评测集合；其失败可以指导检索独立训练数据，但测试样本本身不进入训练。
_Avoid_: Training Set、一次性保密考试

## 评测与采样信号

**Evaluation Specs**:
owner 提供的两份 Markdown 评测口径：`md1` 定义事实性检测，`md2` 定义回复好坏比较；机器评测负责规模化初筛和模型晋级，Codex 在精细数据判断中遵守同一口径。
_Avoid_: ClawDPO 自创 rubric、另一套 Codex 标准

**Correctness Gate**:
按照 `md1` 检测幻觉、事实错误等底线问题的快速二元初筛；失败回复永远不能成为 chosen。
_Avoid_: 质量排序器、最终 pair selector

**Pairwise Quality Judge**:
按照 `md2` 比较同一 prompt 下两条回复好坏；训练 pair 由 Codex 终判，模型晋级直接汇总机评胜场。
_Avoid_: Correctness evaluator、scalar reward

**Codex-as-Critic**:
让带有项目上下文、评测口径和工具的 Codex 对 rollout 提供判断信号；它借用 actor–critic 的角色划分，但不是 PPO 的可训练 value model。
_Avoid_: 裸模型调用、GPT API critic、PPO value network

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

**Branch Point Candidate**:
Codex-as-Critic 在高概率 Absolute Reject 中定位出的候选 token 位置；它只是语义线索，尚未证明该位置改变了最终结果。
_Avoid_: 第一个错误 token、Verified Branch Point

**Branch Trial**:
从首次 rollout 保存的同一组 prompt token 和 response token prefix 出发，固定原分叉 token 或唯一替代 token 后生成的一条完整回复。
_Avoid_: 原始 bad response、Preference Pair

**Verified Branch Point**:
两组固定 token 的 Branch Trial 达到样本量、事实性和严格通过率门槛，并由 Codex-as-Critic 明确判定替代组整体更好的 Branch Point Candidate；它不是统计显著性的代名词，也不是训练 pair。
_Avoid_: 单次成功 rollout、chosen token、Preference Pair
