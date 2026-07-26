---
status: accepted
---

# 用 DPO 模拟异步 RL 的策略改进循环

ClawDPO 把当前模型视为 Behavior Policy：模型采样，Correctness Gate 与 Codex
把反馈转成 Preference Pair，DPO 再更新策略。pair 的 likelihood 必须由本轮
Behavior Policy 计算，两端也必须处于它的实际支持集；否则 DPO 仍会对模型几乎
不可能生成的 rejected 施加无价值梯度。

本轮 Behavior Policy 同时作为 Candidate 的初始化模型和 DPO reference model；
训练主要学习 chosen 相对 rejected 的偏好差异，并用 beta 约束相对 reference
的更新。选择这种更新方式，是为了减少把 chosen 当标准答案重新模仿时对其他模型
行为造成的无关扰动。它不能单独保证保留全部基模能力，能力回归仍由 Test Set 在
其覆盖范围内验收。

这形成近似 on-policy 的 actor–evaluator–learner 循环，但不是完整 RL：v1 没有
多步环境 trajectory、return 或策略梯度信用分配。具体采样和训练规则以
`docs/design.md` 为准。
