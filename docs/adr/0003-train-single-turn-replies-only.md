---
status: accepted
---

# 第一版只训练单轮回复

ClawDPO 第一版的 DPO 样本是同一个 prompt 下的一条 chosen 回复和一条 rejected 回复，不训练多步 agent trajectory，也不负责工具 action 的模拟或副作用控制。工具执行属于外部 harness 的责任，不进入本项目的训练数据模型。
