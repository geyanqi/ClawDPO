---
status: accepted
---

# 只使用训练集和可反复观察的测试集

ClawDPO 只维护 train/test 两类数据：每次训练后都在同一 test 上比较模型，并允许 Codex 根据 test badcase 和 `prompt/训练数据检索.md` 编写 SQL 规则，从数据库中寻找新的、独立的训练对话。数据库结果进入 Prompt Pool 前必须删除包含完整 test messages 序列的原始样本。这个 test 是会影响后续训练方向的回归基准，而不是一次性保密考试，因此系统只声称模型在该基准上的迭代结果，不额外维护第三套 hidden acceptance set。
