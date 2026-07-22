---
status: accepted
---

# 分离正确性门槛与成对质量评测

ClawDPO 只接受 owner 提供的两份评测口径：`md1` 判断两个回复哪个更好，`md2` 判断是否存在幻觉、事实错误等底线问题。`curl1` 按 `md2` 做快速正确性初筛，`curl2` 按 `md1` 做快速成对质量初筛；Codex 直接接收同样的 `md1` 和 `md2` 作为 context，依靠更强能力做最终判断。这同时用于构造训练 pair 和训练后的模型晋级：Candidate 的事实性底线不能退步，且 Codex 必须明确判断它在同一 Test Set 上整体更好；并列或拿不准时保留旧 Best Model。正确性与回复质量仍是两个独立信号，但 curl 与 Codex 不各自发明一套 rubric；ClawDPO 也不新增第三个评测服务。
