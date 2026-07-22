# ClawDPO 当前设计（人话版）

> 2026-07-23 根据 owner 反馈更新。下面“已经决定”的内容不再继续争论；最后一节才是后续需要讨论的事项。

## 核心架构

ClawDPO 本质上是在用 DPO 模拟一个异步 RL 风格的策略改进循环：

```text
当前模型（Behavior Policy）
    → Prompt Pool 调度节点（v1 直通全量；规模化规则待讨论）
    → 对本轮 Prompt Set 固定 rollout 256 条
    → Correctness Gate 做事实性初筛
    → likelihood 分布筛候选
    → Codex 构建多样 pair，并兜底检查 chosen 的事实性
    → 形成 Preference Pair
    → DPO 更新下一版模型
    → 下一版模型重新 rollout
```

| RL 里的概念 | ClawDPO 中的对应物 |
|---|---|
| state / replay buffer | 完整 session 上下文与不断增长的 Prompt Pool |
| actor / behavior policy | 当前模型和固定 vLLM rollout 脚本 |
| action samples | 每个 prompt 的 256 条 response |
| hard reward / constraint | Correctness Gate 的二元正确性结果 |
| preference signal | Codex 对通过正确性门槛的回复做质量排序 |
| learner | DPO trainer |
| policy iteration | 新模型重新 rollout 历史与新增 prompt，再生成新 pair |

它不是完整 RL：第一版没有多步环境 trajectory、累计 return 或策略梯度信用分配。它追求的是 actor、评测、pair 生成和 learner 解耦，以及每版模型重新产出样本所带来的近似 on-policy 效果。同一模型版本内可以异步跑 rollout、机评和候选筛选，但冻结 Dataset Revision 后才启动训练。

## 已经决定

1. **不上线**：ClawDPO 最后只交付新模型和评测结果，不负责发布。
2. **只训单轮回复**：一个样本就是同一个问题下的一条好回复和一条坏回复，不搞多步 agent trajectory。
3. **rollout 是固定脚本**：本地启动 vLLM server，每次只替换模型路径；脚本对每个 prompt 固定生成 256 个回复，不做提前停止，也不让 Codex 自己循环请求。采样直接模仿 RL 中的标准 policy rollout：`temperature=1`、`top_p=1`、top-k 关闭，不加额外 logit penalty。
4. **机器评测是正确性初筛，不是质量排序器**：它专门检查幻觉、事实错误等底线问题。通过只代表可以继续参与 pair 审查；不通过就是 Absolute Reject，永远不能成为 chosen。
5. **Codex 才负责精挑 pair**：系统先根据当前 prompt 的 256 条 likelihood 分布和正确性结果切出少量当前候选，再把完整 session、分布摘要、候选回复及该 prompt 的全部历史 chosen 交给 Codex，由它决定最终 chosen/rejected；Codex 不看全部 256 条原文。同一个 pair-selection prompt 里再加一条简单要求：输出前检查 chosen 没有事实性问题。这是最后兜底，不是新的评测服务。
6. **只保留 train/test 两套数据**：每次训练后都跑 test；Codex 可以观察 test badcase，再写 SQL 规则去数据库寻找相似但不同的训练数据。test 原始对话绝不能混进 train。
7. **每轮训练后都比较**：Candidate Model 与 Best Model 在同一 test 上生成回复，`curl1`、`curl2` 先批量初筛，Codex 再按同一 `md1`、`md2` 做最终判断。只有事实性底线不退步且 Codex 明确判断整体更好，Candidate 才成为新 Best Model；并列或拿不准时保留旧 Best。
8. **自动循环只限制最大轮次**：没达到目标可以继续找数据、训练和测试，直到达到 `max_iterations`。`max_iterations` 是每次任务启动时的必填参数，不设默认值；没传就拒绝启动。
9. **外部能力由 owner 包好**：数据库脚本、脱敏、curl 机评、vLLM rollout 和训练命令都做成固定输入格式的工具。Codex 只能调用，不能读取内部实现或凭据。
10. **数据库输出已经脱敏**：ClawDPO 默认拿到的是可直接处理的脱敏对话；如果需要额外脱敏，就预留一个 owner 提供的插件工具。
11. **每版训练数据必须存档**：不能只有一个不断覆盖的 `latest.jsonl`；每次训练明确记录使用的是哪一版 prompt、rollout 和 pair。
12. **历史上长期积累的是 prompt，不是固定 pair**：新模型会重新 rollout 历史 prompt，再为当前模型生成一套新的 pair；旧 chosen/rejected 不原样重复训练。
13. **任意历史 chosen 都可以成为后续 rejected**：每轮都用当前 policy 对该 prompt 的全部历史 chosen 重新计算 likelihood；只要某条仍处于高 likelihood 区域，并且找到一个质量更好的回复，它就可以成为 Relative Reject。新 chosen 的 likelihood 比它高或低都可以。
14. **不是所有历史 prompt 每轮都训练**：v1 仍会对全部历史 prompt 每轮重新 rollout；如果找不到更好的有效 pair，这个 prompt 只是不进入本轮 Dataset Revision，本轮优先使用新拉取的数据。因此 Prompt Pool 越来越大，但每轮实际 pair 只是它的一个子集。
15. **“容易/不容易采样”是 prompt 内相对概念**：每个 prompt 都用自己的 256 条 response likelihood 分布划分高、低区域，不拿一个全局固定概率跨 prompt 判断。
16. **完整 rollout 存档，Codex 只看候选切片**：256 条全部完成正确性检查和 likelihood 计算并保存，但进入 Codex 上下文的只有最有训练价值的一小部分。
17. **每份 rollout 必须记录 Behavior Policy**：即使 rollout、机评、Codex 选 pair 和训练异步完成，也必须知道这些 response 和 likelihood 到底由哪一版模型产生。
18. **第一版 `policy_lag = 0` 指评分 policy 不滞后**：从 v3 训练 v4 时，所有 pair eligibility likelihood 都必须由 v3 计算。历史 response 可以复用，但不能沿用旧概率，必须由 v3 teacher-force 重新打分。
19. **vLLM 直接提供 response likelihood**：新生成回复读取 sampled-token logprobs 与 cumulative logprob；任意历史回复通过 `prompt_logprobs` teacher forcing 在当前模型下重新打分。保存 logprob，不把极小概率直接 `exp` 成浮点数。
20. **`curl1` 与 `curl2` 分工固定**：`curl1` 根据 `md2` 快速初筛幻觉/事实错误；`curl2` 根据 `md1` 快速比较两个回复哪个更好。它们是便宜的初筛器，不是另一套评测标准。
21. **likelihood 使用 raw 语义**：vLLM 固定 `logprobs_mode=raw_logprobs`，使用 temperature、top-p 等采样变换之前的模型原始概率；新 rollout 与历史 response 重评分必须保持同一语义。
22. **pair 来源是开放集合**：四个样本分区只用来找候选，不把训练 pair 锁死为几个模板。当前 policy 的事实错误、当前 rollout 内的质量差、以及新 rollout 对历史 chosen 的超越都可以产生 pair；只要对当前 policy 构成有价值的排序信号就行。
23. **所有 chosen 都有双层事实性保障**：先由 `curl1` 按 `md2` 初筛，再把同一份 `md2` 作为 context 交给 Codex 做事实性兜底。这里不增加新标准、agent、curl 或 pipeline 阶段。
24. **pair 两端都必须是当前 policy 实际采得到的回复**：好回复可以位于低概率尾部，坏回复优先选高概率区域，但任何一端都不能脱离当前 policy 的实际支持集。在 256 条中出现过是 support 证据，但如果它位于 Bottom 5% 的 Extreme Tail，仍不能直接入 pair。历史回复必须先用当前 policy 重评分。禁止手工编造一条模型根本不会产生的“超级坏回复”当 rejected。
25. **低概率不是越低越好**：真正要找的是 Supported Tail，也就是“很少出现，但当前 policy 还能实际走到”的高质量回复。如果某条回复虽然很好，但它在分布中稀有到极点，就先存档而不直接当 chosen；本轮先选一条更可达、仍然明显更好的桥接回复，让下一版 policy 把高质量行为逐步抬进可达区域。
26. **rollout 直接从 raw policy 采样**：固定 `temperature=1`、`top_p=1`、top-k 关闭且无额外 logit penalty，不改写模型原始分布；因此 raw logprob 既是 likelihood 排序语义，也与实际 rollout policy 保持一致。不再引入 processed logprob 或单独的 sampler-reachable 检查。
27. **likelihood 分区用简单固定规则**：每条回复同时保存 raw cumulative logprob 和 raw 平均 token logprob，但只用平均 token logprob 在同一 prompt 的 256 条内排序。Top 25% 是高概率区，5%–25% 是 Supported Tail，Bottom 5% 是 Extreme Tail，中间 50% 默认不优先。不做 logprob 断层检测、语义邻域聚类或其他 support-floor 模型。
28. **Candidate Slice 固定最多 32 条当前回复，但历史 chosen 全带**：Supported Tail/pass 最多 16 条，Top 25%/fail 最多 8 条，Top 25%/pass 最多 8 条；某区不足就有多少带多少，不从中间 50% 补。当前 rollout 只做去空格后的完全重复去重，不做语义去重。该 prompt 历史上有多少条 chosen 就全部附带，每条都包含当前 policy 重评分结果和当前分位，不受 32 条上限限制。
29. **评测口径只有 owner 提供的两个 Markdown 文件**：`md1` 定义“哪个回复更好”，`md2` 定义“是否存在事实性问题”。curl 脚本和 Codex 使用完全相同的文件；curl 负责快速初筛，Codex 依靠更强能力做最终判断。ClawDPO 不另外发明质量 rubric 或事实性 rubric。
30. **模型晋级不计加权总分**：`curl1`、`curl2` 在完整 test 上批量初筛，Codex 结合初筛结果、相关回复和同一 `md1`、`md2` 终判。Candidate 必须同时满足“事实性底线不退步”和“Codex 明确判断整体更好”；任一不满足、并列或无法判断，都保留旧 Best Model。
31. **`max_iterations` 没有默认值**：它是启动每个自动迭代任务时的必填正整数；缺失或非法时任务不启动。
32. **v1 每轮全量重跑 Prompt Pool**：Prompt Pool 调度节点在 v1 是直通节点，不做冷却、淘汰或优先级调度；池中所有 prompt 每轮都固定 rollout 256 条。这个节点显式保留，等实际遇到规模瓶颈后再讨论如何调度，但不阻塞 v1。
33. **工具格式等实现时再要**：现在不凭空设计各工具的 JSON 字段。实现某个工具适配时，Codex 先向 owner 索要真实脚本及其输入输出格式，再补齐该工具的调用约定和示例。

## 三个信号必须分开

| 信号 | 它回答的问题 | 用法 |
|---|---|---|
| Correctness Gate | 有没有幻觉、事实错误等底线问题？ | `curl1` 用 `md2` 初筛，Codex 用同一 `md2` 兜底；fail 必然是 Absolute Reject |
| Response Quality | 在正确的前提下，哪个回复更好？ | `curl2` 用 `md1` 初筛，Codex 用同一 `md1` 做更强的最终判断 |
| Policy Likelihood | 当前模型多容易生成这类回复？ | 先判断是否在当前 policy 的实际支持集内，再寻找“低概率的更好回复”和“高概率的较差回复” |

这三个值不能相互替代：正确但普通的回复可以质量较低；高概率回复也可以正确；文风很好但出现幻觉的回复仍然直接失败。

likelihood 不只是用来排优先级，它还是 pair 的准入证据。普通 DPO loss 不会因为某条 rejected 在当前 policy 下几乎不可能出现，就自动忽略这条 pair；只要喂进训练，它仍会产生梯度。因此“这条回复很坏”不等于“这条 rejected 有训练价值”。

## 低概率不是越低越好

对每个 prompt 的 256 条回复，用 raw 平均 token logprob 排序：

| 区域 | 含义 | 处理 |
|---|---|---|
| 高概率区 | Top 25% | 优先寻找需要被压低的 rejected |
| 中间区 | 25%–75% | 默认不优先，完整存档 |
| Supported Tail | 5%–25% | 优先寻找正确且高质量的 chosen |
| Extreme Tail | Bottom 5% | 只存档，不进本轮 pair |

raw cumulative logprob 仍然存档，但不参与分区，避免长回复因 token 更多而天然被判成低概率。Extreme Tail 里即使有高质量回复，也只存档；本轮改用 Supported Tail 中质量最高的桥接回复。

## 四个候选索引分区

先排除 raw 平均 token logprob 位于 Bottom 5% 的 Extreme Tail，再把剩余回复的正确性 `pass/fail` 与 likelihood `high/supported low` 交叉，用四个分区给候选建索引：

| 类别 | likelihood | 正确性 | 是否给 Codex | 可能用途 |
|---|---|---|---|---|
| A | supported low | pass | 是 | 探索型 chosen 候选：不容易出现但仍然可达，是否正确且更好由机评和 Codex 判断 |
| B | high | fail | 是 | Absolute Reject 候选：模型经常产生的幻觉或事实错误 |
| C | high | pass | 是 | 可能是 chosen，也可能是 Relative Reject；完全取决于回复质量比较 |
| D | supported low | fail | 低优先级 | 虽然错误，但模型本来就很少产生；默认不占用宝贵的 Codex 审查预算 |

历史 chosen 不是第五类。全部 Chosen History 都用当前 policy 重算 raw likelihood 并标注当前分位，然后原样附带给 Codex；它们不受当前 rollout 的 32 条 Candidate Slice 上限限制。只有当某条历史 chosen 当前仍是高 likelihood，并且存在质量更好的回复时，它才成为 Relative Reject。

这四格只是找当前候选的索引，不是 pair 类型，更不是封闭的训练规则。Codex 看到完整 session、likelihood 分布摘要、最多 32 条当前候选原文和全部 Chosen History；完整 rollout 仍全部存档。

上表中的 A/B/C/D 都是通过 Bottom 5% 排除规则后的候选，区别是训练价值和优先级。一条回复即使在本轮碰巧采到了一次，但如果仍然落在 Extreme Tail，也只存档。历史或外部回复更不能因为“内容很好”或“内容很坏”就直接入 pair；必须先证明当前 policy 实际有机会产生它。

## 训练 pair 的常见来源（不封闭）

1. **修当前 policy 的事实错误**：当前 rollout 里高概率产生的幻觉或事实错误做 rejected，找一条正确且更好的回复做 chosen。
2. **抬高当前 policy 的回复上限**：同一批 rollout 里，低概率但高质量的正确回复做 chosen，高概率但质量较低的正确回复做 rejected。
3. **超越上一轮的上限**：当前 policy 采到比历史 chosen 更好的 rollout 时，若历史 chosen 经当前 policy 重评分后仍是高 likelihood，就把新回复做 chosen、历史 chosen 做 rejected。
4. **其他有价值的排序**：上面是高频例子，不是白名单。Codex 可以构建其他 pair，只要它明确教当前 policy “更应该回答什么、更不应该回答什么”。

来源可以多样，但统一约束不变：两条回复对应同一 prompt/session；两端都通过 Bottom 5% 排除规则；chosen 明确更好，先通过 Correctness Gate，再经 Codex 的简单事实性兜底；rejected 优先选当前 policy 下高 likelihood 的行为。chosen 可以比 rejected 更难采到，但必须位于 Supported Tail 或更可达的区域，不能落入 Extreme Tail。历史回复用作任意一端前都必须先由当前 policy 重评分。

## 实际运行流程

1. Codex 调用数据库工具，拉取已经脱敏的线上对话。
2. Codex 调用机器评测工具，筛出可能有问题的 session。
3. Codex 精读这些 session，挑出值得训练的单轮 prompt。
4. Prompt Pool 调度节点在 v1 直接选中“全部历史 prompt + 本轮新 prompt”；固定脚本用当前模型路径启动 vLLM，并对每个 prompt 重新 rollout 256 次。该节点未来可以加入冷却、淘汰或优先级调度，但 v1 不实现。
5. Correctness Gate 先检查幻觉、事实错误等底线问题：fail 是 Absolute Reject；pass 只表示回复没有底线错误。
6. 对每个 prompt 的 256 条回复按 raw 平均 token logprob 排序：Bottom 5% 只存档，再从 5%–25%/pass、Top 25%/fail、Top 25%/pass 三个区域各取少量候选。
7. 用当前 policy 对该 prompt 的全部 Chosen History 做 teacher-forced raw likelihood 评分并标注当前分位。
8. 把 `md1`、`md2`、完整 session、分布摘要、最多 32 条当前 Candidate Slice 和全部 Chosen History 一起交给 Codex；Codex 不看其余 rollout 原文，按 `md1` 精选 pair，并按 `md2` 确认 chosen 没有幻觉或事实错误。不通过就换 chosen 或放弃该 pair。
9. 历史 prompt 如果找不到比上一版 frontier 更好的训练关系，本轮跳过；能找到才生成新 pair。
10. 保存一版不可修改的 prompt、完整 rollout、候选切片和 pair 数据，然后调用固定训练命令。
11. 训练结束后，让 Candidate Model 与 Best Model 在相同 test prompt 上分别生成回复；`curl1` 按 `md2` 检查正确性，`curl2` 按 `md1` 做成对质量比较。
12. Codex 读取同一 `md1`、`md2`、curl 初筛结果和相关回复后终判。事实性不退步且整体明确更好才替换 Best Model；否则保留旧模型，并针对 test badcase 重新找训练数据。
13. 达标时输出候选模型和报告；达到本次任务必填的 `max_iterations` 仍未达标就停止。

## Codex 怎么接入

Codex 不直接写 SQL、拼 curl 或手动请求 vLLM。它只需要看到一组固定工具：

- `fetch_conversations(request.json) -> conversations.jsonl`
- `sanitize_conversations(request.json) -> sanitized.jsonl`（可选）
- `evaluate_correctness_curl1(request.json) -> correctness.jsonl`
- `compare_quality_curl2(request.json) -> pairwise_quality.jsonl`
- `rollout_vllm(request.json) -> rollouts.jsonl`
- `score_response_likelihood(request.json) -> likelihood.jsonl`
- `build_candidate_slice(request.json) -> candidate_slice.json`
- `archive_dataset(request.json) -> dataset_revision.json`
- `train_dpo(request.json) -> model_revision.json`
- `evaluate_test(request.json) -> comparison.json`

以 `rollout_vllm` 为例，rollout 数量和 sampling 参数都固化在脚本里，采样直接来自 raw policy，并固定 `logprobs_mode=raw_logprobs`。每一轮只替换模型路径和输入 Prompt Pool；脚本负责启动/切换本地 vLLM server、生成 256 条回复，并把逐 token raw logprob、累计 logprob、平均 token logprob、token 数和结束原因写入完整结果文件。`score_response_likelihood` 使用当前模型的 `prompt_logprobs` 对全部 Chosen History 做同语义的 teacher-forced raw 打分。`build_candidate_slice` 按 raw 平均 token logprob 分位和正确性结果抽取最多 32 条当前候选，再附带不设上限的全部 Chosen History；Codex 不接管 vLLM server 的细节。

Codex 端直接把 owner 提供的 `md1` 和 `md2` 作为 context：按 `md1` 比较候选回复并输出有价值的 chosen/rejected，按 `md2` 确认 chosen 没有事实性问题。这不需要 ClawDPO 另写 rubric、另起一个 Codex agent 或多加一个评测工具。

如果要求 Codex **真的无法查看**脚本和凭据，这些工具不能以源码形式放在同一个可读工作区里；应当运行在独立进程、容器或不同系统用户下，只向 Codex 暴露调用接口。

## 已确认保留的后续 TODO

1. **保留的规模化扩展节点（不阻塞 v1）**：Prompt Pool 多大时不再适合每轮全量重跑，以及届时采用冷却、淘汰还是优先级调度。
2. **工具适配节点（实现时再要）**：每开始实现一个工具，先向 owner 索要真实脚本及其输入输出格式，再记录具体调用方式；当前不预设 JSON 字段。
