# Owner tools

This directory is the single entry point for owner-provided commands:

- database retrieval
- `curl1` factuality evaluation
- `curl2` response-quality comparison
- vLLM rollout
- DPO training

Keep the directory flat. ClawDPO executes these commands through their declared
interfaces; it does not need separate database, evaluation, or training
subdirectories.

Real commands and credentials stay local and are ignored by Git. If Codex must
not read their implementations, mount an external executable-only directory at
this path instead.
