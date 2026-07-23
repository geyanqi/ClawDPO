# ClawDPO

ClawDPO 是一个由 Codex 编排的 DPO 迭代系统：它从脱敏业务对话中发现问题，让当前模型重新生成候选回复，构造高价值偏好对，启动训练，并用测试集机评决定是否保留新模型。

```text
线上 badcase → 当前模型 rollout → 正确性初筛 → Codex 精选 pair
             → DPO 训练 → 新旧模型机评晋级 → badcase 拉取独立训练数据 → 下一轮
```

项目边界止于候选模型和评测报告，不负责发布或部署。v1 只训练单轮回复，不做多步 agent trajectory。

## 文档

- [项目动机](docs/motivation.md)：为什么要做、希望解决什么问题。
- [设计规范](docs/design.md)：v1 的完整架构、数据筛选规则和运行流程；这是实现行为的唯一权威文档。
- [领域术语](CONTEXT.md)：项目内统一使用的概念和名称。
- [架构决策](docs/adr/)：重要取舍及其原因。
- [文档索引](docs/README.md)：文档职责和维护规则。

当前状态：单轮 rollout、筛选、pair 冻结、训练、机评晋级和防 Test Set 污染的 badcase 拉数入口已完成。

## 开发入口

```bash
python infra/inference/rollout.py --help
python infra/inference/rescore.py --help
python workflow/run_iteration.py --help
uv run python -m unittest discover -s tests
```
