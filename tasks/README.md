# Training tasks

Each concrete campaign uses one directory:

```text
tasks/<task-name>/
├── md1.md
├── md2.md
├── task.json
└── iterations/
```

`md1.md` checks factuality, `md2.md` compares response quality, and
`iterations/` stores immutable per-round artifacts. The exact `task.json`
fields will be defined when the owner tools are connected.

Task inputs and generated artifacts stay local and are ignored by Git.
