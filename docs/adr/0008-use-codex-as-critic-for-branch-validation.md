---
status: accepted
---

# 用 Codex-as-Critic 验证局部分叉

局部分叉定位和两组结果精评需要理解完整上下文，并可能调用搜索、代码或文档工具
核实事实。ClawDPO 因此由外层 Codex session 分派 subagent 完成这两次判断，而不
直接请求一个裸模型接口；代价是评测更慢、更贵，换来的是带 harness 的可核查判断。
这里借用 actor–critic 的角色划分，但 Codex critic 不是 PPO 的可训练 value model。
