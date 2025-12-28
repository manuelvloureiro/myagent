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
- **myagent**: The graph engine built on top of myagent-core, providing `StateGraph`, channels,
  checkpointing, and the Pregel execution engine.

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Running Tests](#running-tests)
3. [Architecture](#architecture)
4. [License](#license)

## Environment Setup

```shell
uv sync
```

## Running Tests

```shell
# Full suite with coverage
uv run pytest tests/ --cov --cov-report=term-missing --cov-fail-under=80

# Single test file
uv run pytest tests/test_core/test_runnable.py -v
```

## Architecture

### myagent-core

The core package provides the foundational `Runnable[Input, Output]` interface:

- **Runnable**: Abstract base with `invoke`, `ainvoke`, `stream`, `astream`, `batch`, `abatch`
- **RunnableConfig**: TypedDict carrying tags, metadata, callbacks, configurable dict, recursion limit
- **RunnableLambda**: Wraps plain functions as Runnables with automatic config injection
- **JsonPlusSerializer**: Extended JSON serialization for datetime, UUID, Decimal, bytes, set, frozenset

### Channels

Channels are typed state containers used by the graph engine:

- **LastValue**: Default channel - stores the last written value
- **BinaryOperatorAggregate**: Folds updates via a reducer function (e.g., `operator.add` for lists)
- **EphemeralValue**: Temporary channel cleared each superstep

### Checkpointing

- **BaseCheckpointSaver**: Abstract interface for persisting graph state
- **InMemorySaver**: Dict-backed storage keyed by thread_id

## License

This repository is released under the [MIT License](LICENSE).
