---
status: accepted
---

# 分离正确性门槛与成对质量评测

ClawDPO 只接受两份评测 prompt：`prompt/事实性检测.md` 检测幻觉、事实错误等事实性问题，`prompt/回复竞对.md` 判断同一问题下两条回复哪条更好。两种评测共用 `infra/cli/curl.sh`，区别只在 OpenAI-compatible request file 的内容。构造 Preference Pair 时，Codex 直接接收相同 Markdown 文件作为 context，依靠更强能力做最终判断；训练后的模型晋级不再经过 Codex。Candidate Model 在完整 Test Set 上的事实性失败数不高于 Best 且质量胜场严格更高时自动晋升，否则保留旧 Best Model。正确性与回复质量仍是两个独立信号，curl 只是请求模型的传输方式，不定义另一套 rubric。
