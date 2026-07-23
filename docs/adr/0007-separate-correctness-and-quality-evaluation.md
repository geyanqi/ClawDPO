---
status: accepted
---

# 分离正确性门槛与成对质量评测

ClawDPO 只接受两份评测 prompt：`prompt/md1.md` 检测幻觉、事实错误等事实性问题，`prompt/md2.md` 判断同一问题下两条回复哪条更好。两种评测共用 `infra/cli/curl.sh`，区别只在 OpenAI-compatible request file 的内容；Codex 直接接收相同 Markdown 文件作为 context，依靠更强能力做最终判断。这同时用于构造 Preference Pair 和训练后的模型晋级：Candidate Model 的事实性底线不能退步，且 Codex 必须明确判断它在同一 Test Set 上整体优于 Best Model；并列或拿不准时保留旧 Best Model。正确性与回复质量仍是两个独立信号，curl 只是请求模型的传输方式，不定义另一套 rubric。
