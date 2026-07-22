---
status: accepted
---

# 用 DPO 模拟异步 RL 的策略改进循环

ClawDPO 不把 DPO 当作消费固定离线数据的一次性训练，而把当前模型视为 Behavior Policy：它持续对 Prompt Pool 做标准 raw-policy rollout，Correctness Gate 与 Codex 将反馈转成精炼的 Preference Pair，DPO 再学习下一版 policy。rollout、评测和选 pair 可以在同一模型版本内异步执行，但 v1 固定 `policy_lag = 0`：pair 的 eligibility likelihood 必须由本轮 Behavior Policy 计算，且 chosen/rejected 两端都必须在它的实际支持集内。每个 prompt 的 256 条只用 raw 平均 token logprob 排序，Bottom 5% 不入 pair；该 prompt 的全部 Chosen History 每轮都由当前模型做 raw 重评分并按同一分位规则定位。普通 DPO loss 不会按当前 policy 的实际采样频率自动降低一条 pair 的权重，所以禁止使用当前模型根本采不到的人工 rejected，即使它的内容再糟糕。Dataset Revision 冻结后才启动训练。该架构模拟异步 actor–evaluator–learner 与近似 on-policy 的策略迭代，同时保留 DPO 的稳定性；它不声称是完整 RL，因为 v1 只有单轮回复、没有环境 trajectory、return 或策略梯度信用分配。
