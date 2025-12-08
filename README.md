# myagent

A lightweight, self-contained agent framework for building stateful LLM workflows.

## Motivation

LangChain and LangGraph are powerful but heavyweight - they pull in many dependencies and abstractions
that are often unnecessary for focused agent applications. **myagent** aims to provide a minimal,
API-compatible subset of LangChain/LangGraph that you can vendor, fork, or extend without inheriting
the full dependency tree.

The goal is simple: if your code works with myagent, it should work with LangChain/LangGraph too
(and vice versa for the supported surface area).

## Overview

The project is organized as a uv workspace with two packages:

- **myagent-core**: Zero-dependency foundation providing the `Runnable` abstraction - the universal
  execution interface shared by every component.
- **myagent**: The graph engine built on top of myagent-core.

## Environment Setup

```shell
uv sync
```

## Running Tests

```shell
uv run pytest tests/ -v
```

## License

This repository is released under the [MIT License](LICENSE).
