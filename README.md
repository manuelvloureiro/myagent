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
4. [Usage](#usage)
5. [License](#license)

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

# Compat tests (verify API compatibility with LangGraph)
uv run pytest tests/test_compat/ -v

# Run compat tests against real LangGraph
uv run pytest tests/test_compat/ --langgraph -v
```

## Architecture

### myagent-core

The core package provides the foundational `Runnable[Input, Output]` interface:

- **Runnable**: Abstract base with `invoke`, `ainvoke`, `stream`, `astream`, `batch`, `abatch`
- **RunnableConfig**: TypedDict carrying tags, metadata, callbacks, configurable dict, recursion limit
- **RunnableLambda**: Wraps plain functions as Runnables with automatic config injection
- **JsonPlusSerializer**: Extended JSON serialization for datetime, UUID, Decimal, bytes, set, frozenset

### Graph Engine (myagent)

- **StateGraph**: Builder that introspects TypedDict schemas to create typed channels.
  Supports `Annotated[T, reducer]` for custom field reducers.
- **Pregel**: Superstep execution engine. Each step: plan eligible nodes -> check interrupts ->
  snapshot state -> execute nodes concurrently -> apply updates via channel reducers -> stream -> checkpoint.
- **Channels**: `LastValue` (default), `BinaryOperatorAggregate` (reducer-based), `EphemeralValue` (cleared each superstep)
- **Checkpointing**: `BaseCheckpointSaver` -> `InMemorySaver`. Multi-turn via `thread_id` in config.
- **MessagesState**: Pre-built state with `add_messages` reducer for chat-style applications.

### Error Handling

- `GraphRecursionError`: Exceeds recursion limit
- `EmptyChannelError`: Reading from uninitialized channel
- `InvalidUpdateError`: Channel update invalid for type
- `NodeInterrupt`: Execution interrupted

## Usage

```python
from myagent import StateGraph, START, END
from typing import TypedDict

class State(TypedDict):
    count: int

graph = StateGraph(State)
graph.add_node("increment", lambda s: {"count": s["count"] + 1})
graph.add_edge(START, "increment")
graph.add_edge("increment", END)
app = graph.compile()

result = app.invoke({"count": 0})
print(result)  # {"count": 1}
```

## License

This repository is released under the [MIT License](LICENSE).
