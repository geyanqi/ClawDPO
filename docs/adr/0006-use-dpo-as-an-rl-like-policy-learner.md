---
status: accepted
---

# 用 DPO 模拟异步 RL 的策略改进循环

ClawDPO 把当前模型视为 Behavior Policy：模型采样，Correctness Gate 与 Codex
把反馈转成 Preference Pair，DPO 再更新策略。pair 的 likelihood 必须由本轮
Behavior Policy 计算，两端也必须处于它的实际支持集；否则 DPO 仍会对模型几乎
不可能生成的 rejected 施加无价值梯度。

这形成近似 on-policy 的 actor–evaluator–learner 循环，但不是完整 RL：v1 没有
多步环境 trajectory、return 或策略梯度信用分配。具体采样和训练规则以
`docs/design.md` 为准。
