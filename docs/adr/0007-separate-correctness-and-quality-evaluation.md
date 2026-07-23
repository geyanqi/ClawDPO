---
status: accepted
---

# 分离正确性门槛与成对质量评测

事实正确只代表回复越过底线，不代表它回答得好。ClawDPO 因此分别使用事实性
门禁和成对质量比较；两者可以共用同一个请求脚本，但不能混成一个分数。构造
Preference Pair 时 Codex 同时复核两个信号，模型晋级则只汇总固定机器评测，
不由 Codex 改判。具体 prompt、并行度和晋级条件以 `docs/design.md` 为准。
