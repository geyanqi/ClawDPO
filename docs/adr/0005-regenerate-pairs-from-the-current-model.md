---
status: accepted
---

# 每版模型都从 Prompt Pool 重新生成 Preference Pair

ClawDPO 长期积累的是 Prompt Pool 和完整 Chosen History，而不是把旧 Preference Pair 原样重复训练。v1 的每个 Behavior Policy 都对整个 Prompt Pool 固定 rollout 256 次；系统依据每个 prompt 自己的 likelihood 分布和 Correctness Gate 结果生成最多 32 条当前 Candidate Slice，Codex 同时查看完整 session、分布摘要和不设上限的全部 Chosen History，完整 rollout 另行存档。rollout 固定 `temperature=1`、`top_p=1`、top-k 关闭且无额外 logit penalty，直接从 raw policy 采样。每个 prompt 只用 raw 平均 token logprob 排序：Top 25% 是高概率区，5%–25% 是 Supported Tail，Bottom 5% 只存档。pair 两端都必须处在当前 policy 的实际支持集内；全部历史 chosen 每轮都由当前 Behavior Policy 做 raw 重评分并按同一分位规则定位。历史 prompt 只有在能形成新的有效 pair 时才进入当轮 Dataset Revision，否则跳过并优先使用新拉取的 prompt；这样每轮训练回复尽量来自当前策略，近似 on-policy，同时所有历史 pair 仍按版本存档。
