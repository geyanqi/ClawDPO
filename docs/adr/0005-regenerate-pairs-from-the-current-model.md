---
status: accepted
---

# 每版模型都从 Prompt Pool 重新生成 Preference Pair

ClawDPO 长期积累 Prompt Pool 和完整 Chosen History，不把旧 Preference Pair 原样
重复训练。每个 Behavior Policy 都重新生成候选，并重新计算历史 chosen 在当前
策略下的 likelihood；只有本轮仍能形成清晰排序的 prompt 才进入 Dataset
Revision。这样训练数据尽量来自当前策略，同时保留全部历史版本供审计和重组。
它避免新 policy 不加检查地重复学习旧分布上的 pair，是 Minimum-Disruption
Alignment 在数据更新上的约束。具体 rollout、分位和候选数量以
`docs/design.md` 为准。
