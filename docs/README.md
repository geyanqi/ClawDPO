# 文档索引

ClawDPO 的文档按职责分层，避免同一条规则在多个文件里重复维护。

| 文档 | 职责 |
|---|---|
| [项目 README](../README.md) | 项目入口、边界和阅读导航 |
| [项目动机](motivation.md) | 问题背景、目标、非目标和选择 DPO 的原因 |
| [设计规范](design.md) | v1 的唯一行为规范：架构、阈值、流程和工具边界 |
| [领域术语](../CONTEXT.md) | 项目统一语言，只定义概念，不记录实现参数 |
| [架构决策](adr/) | 难以逆转的重要取舍及其原因 |

## 架构决策索引

| ADR | 决定 |
|---|---|
| [0001](adr/0001-stop-before-production-release.md) | 自治边界在生产发布前停止 |
| [0002](adr/0002-use-an-adaptive-test-set.md) | 只维护 train/test，并允许测试 badcase 指导后续检索 |
| [0003](adr/0003-train-single-turn-replies-only.md) | v1 只训练单轮回复 |
| [0004](adr/0004-use-opaque-tools-for-external-access.md) | Codex 只通过不透明固定工具访问外部能力 |
| [0005](adr/0005-regenerate-pairs-from-the-current-model.md) | 每版模型都从 Prompt Pool 重新生成 pair |
| [0006](adr/0006-use-dpo-as-an-rl-like-policy-learner.md) | 用 DPO 模拟异步 RL 风格的策略改进 |
| [0007](adr/0007-separate-correctness-and-quality-evaluation.md) | 正确性门槛与回复质量评测分离 |

## 维护规则

- 行为、阈值或流程发生变化，只修改 `design.md`。
- 领域概念发生变化，只修改 `CONTEXT.md`，并保持定义简短。
- 只有难以逆转、原因不明显且存在真实取舍的决定才新增 ADR。
- 尚未完成的设计统一放在 `design.md` 的“后续 TODO”，不散落在其他文档。
- 原始讨论稿不继续维护；需要追溯时查看 Git 历史或 `codex/clawdpo-architecture` 分支。
