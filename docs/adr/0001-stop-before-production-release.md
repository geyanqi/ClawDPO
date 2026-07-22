---
status: accepted
---

# ClawDPO 在生产发布前停止

ClawDPO 的自治边界止于产出通过验收的候选推理包及其证据；它不发布、不部署，也不修改任何 production、serving 或 release alias。训练验收与线上发布具有不同的授权、安全和回滚责任，因此后者必须由独立的发布流程接手。
