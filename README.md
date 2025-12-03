# myagent

A lightweight, self-contained agent framework for building stateful LLM workflows.

## Motivation

LangChain and LangGraph are powerful but heavyweight - they pull in many dependencies and abstractions
that are often unnecessary for focused agent applications. **myagent** aims to provide a minimal,
API-compatible subset of LangChain/LangGraph that you can vendor, fork, or extend without inheriting
the full dependency tree.

The goal is simple: if your code works with myagent, it should work with LangChain/LangGraph too
(and vice versa for the supported surface area).

## Environment Setup

```shell
uv sync
```

## License

This repository is released under the [MIT License](LICENSE).
