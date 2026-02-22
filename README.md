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

- **myagent-core**: Zero-dependency foundation providing the `Runnable` abstraction, composition
  primitives, message types, prompt templates, chat model interface, tools, and output parsers.
- **myagent**: The graph engine built on top of myagent-core, providing `StateGraph`, channels,
  checkpointing, and the Pregel execution engine.

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Running Tests](#running-tests)
3. [Architecture](#architecture)
4. [Usage](#usage)
5. [Changelog](#changelog)
6. [License](#license)

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

# Linting and formatting
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Type checking
uv run pyright
```

## Architecture

### myagent-core

The core package provides the foundational abstractions:

**Runnable Interface:**
- **Runnable**: Abstract base with `invoke`, `ainvoke`, `stream`, `astream`, `batch`, `abatch`
- **RunnableConfig**: TypedDict carrying tags, metadata, callbacks, configurable dict, recursion limit
- **RunnableLambda**: Wraps plain functions as Runnables with automatic config injection

**Composition Primitives:**
- **RunnableSequence**: Chains Runnables in series - powers the `|` pipe operator
- **RunnableParallel**: Runs multiple Runnables concurrently, returns a dict
- **RunnablePassthrough**: Identity passthrough with `.assign()` for adding computed keys
- **RunnableBranch**: Conditional routing based on input predicates

**Messages & Prompts:**
- **Message types**: `HumanMessage`, `AIMessage`, `SystemMessage`, `ToolMessage`, `ToolCall`
- **ChatPromptTemplate**: Template with variable substitution producing message lists
- **PromptTemplate**: Simple string formatting

**Chat Models:**
- **BaseChatModel**: Abstract Runnable for chat model integration
- **FakeListChatModel** / **FakeToolCallingModel**: Test doubles

**Tools & Parsers:**
- **BaseTool** / `@tool` decorator: Wraps functions with name/description/schema
- **StrOutputParser**, **JsonOutputParser**, **PydanticOutputParser**

### Graph Engine (myagent)

- **StateGraph**: Builder that introspects TypedDict schemas to create typed channels.
  Supports `Annotated[T, reducer]` for custom field reducers.
- **Pregel**: Superstep execution engine. Each step: plan eligible nodes -> check interrupts ->
  snapshot state -> execute nodes concurrently -> apply updates via channel reducers -> stream -> checkpoint.
- **Channels**: `LastValue` (default), `BinaryOperatorAggregate` (reducer-based), `EphemeralValue` (cleared each superstep)
- **Checkpointing**: `BaseCheckpointSaver` -> `InMemorySaver`. Multi-turn via `thread_id` in config.
- **MessagesState**: Pre-built state with `add_messages` reducer for chat-style applications.

### Intentionally Excluded

The following LangChain/LangGraph features are deliberately out of scope:

- `Send()` dynamic fan-out, `Command(resume=...)`
- Stream modes beyond `values` / `updates`
- Subgraph nested checkpointing, `get_state_history`
- Visualization (`get_graph`, Mermaid)
- Document loaders, text splitters, embeddings, vector stores, retrievers
- Real LLM provider integrations (users wrap their own)

## Usage

### Simple Graph

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

### Chain with Pipe Operator

```python
from myagent_core import ChatPromptTemplate, StrOutputParser, FakeListChatModel

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}."),
    ("human", "{question}"),
])
model = FakeListChatModel(responses=["42"])
parser = StrOutputParser()

chain = prompt | model | parser
result = chain.invoke({"role": "math tutor", "question": "What is 6*7?"})
print(result)  # "42"
```

### Tools

```python
from myagent_core import tool

@tool
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

print(search.invoke({"query": "myagent"}))
```

## Changelog

### 0.1.0

Initial release with LangGraph-compatible graph engine and LangChain-compatible
composition primitives, messages, prompts, chat models, tools, and output parsers.

## License

This repository is released under the [MIT License](LICENSE).
