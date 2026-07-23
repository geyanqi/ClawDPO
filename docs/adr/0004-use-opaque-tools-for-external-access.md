---
status: accepted
---

# Codex 通过固定薄 CLI 访问外部能力

数据库读取、机器评测、vLLM rollout 和训练启动都通过 `infra/cli/` 下的固定薄脚本调用；脚本本身可以进入仓库，但不得包含 API key、数据库凭据或其他秘密。Codex 只提供 request file 或 `WHERE` 文件，敏感信息与后端权限由环境变量和操作系统权限隔离；需要隐藏脚本实现时，再把该目录替换成 executable-only mount。
