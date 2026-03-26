"""Tests for async graph execution."""

import asyncio
import operator
import threading
from typing import Annotated, TypedDict

import pytest
from myagent.checkpoint.memory import InMemorySaver
from myagent.graph import END, START, StateGraph
from myagent_core.runnable.config import RunnableConfig


class AsyncState(TypedDict):
    value: str


class AsyncListState(TypedDict):
    items: Annotated[list, operator.add]


class TestAsyncGraph:
    def test_invoke_with_async_node(self):
        """Sync invoke should reject async-only node functions."""

        async def async_node(state: AsyncState) -> dict:
            return {"value": state["value"] + " truly_async"}

        graph = StateGraph(AsyncState)
        graph.add_node("a", async_node)
        graph.add_edge(START, "a")
        graph.add_edge("a", END)
        app = graph.compile()

        with pytest.raises(TypeError, match="ainvoke"):
            app.invoke({"value": "test"})

    def test_stream_with_async_node(self):
        """Sync stream should reject async-only node functions."""

        async def async_node(state: AsyncState) -> dict:
            return {"value": state["value"] + " streamed_async"}

        graph = StateGraph(AsyncState)
        graph.add_node("a", async_node)
        graph.add_edge(START, "a")
        graph.add_edge("a", END)
        app = graph.compile()

        with pytest.raises(TypeError, match="astream"):
            list(app.stream({"value": "test"}))

    @pytest.mark.asyncio
    async def test_ainvoke_simple(self):
        """Test basic async invocation."""

        def node_a(state: AsyncState) -> dict:
            return {"value": state["value"] + " async"}

        graph = StateGraph(AsyncState)
        graph.add_node("a", node_a)
        graph.add_edge(START, "a")
        graph.add_edge("a", END)
        app = graph.compile()

        result = await app.ainvoke({"value": "hello"})
        assert result["value"] == "hello async"

    @pytest.mark.asyncio
    async def test_ainvoke_with_async_node(self):
        """Test async node function."""

        async def async_node(state: AsyncState) -> dict:
            return {"value": state["value"] + " truly_async"}

        graph = StateGraph(AsyncState)
        graph.add_node("a", async_node)
        graph.add_edge(START, "a")
        graph.add_edge("a", END)
        app = graph.compile()

        result = await app.ainvoke({"value": "test"})
        assert result["value"] == "test truly_async"

    @pytest.mark.asyncio
    async def test_astream_values(self):
        """Test async streaming in values mode."""

        def node_a(state: AsyncState) -> dict:
            return {"value": state["value"] + " streamed"}

        graph = StateGraph(AsyncState)
        graph.add_node("a", node_a)
        graph.add_edge(START, "a")
        graph.add_edge("a", END)
        app = graph.compile()

        chunks = []
        async for chunk in app.astream({"value": "hi"}):
            chunks.append(chunk)

        assert len(chunks) >= 1
        assert chunks[-1]["value"] == "hi streamed"

    @pytest.mark.asyncio
    async def test_astream_updates(self):
        """Test async streaming in updates mode."""

        def node_a(state: AsyncState) -> dict:
            return {"value": "updated"}

        graph = StateGraph(AsyncState)
        graph.add_node("a", node_a)
        graph.add_edge(START, "a")
        graph.add_edge("a", END)
        app = graph.compile()

        chunks = []
        async for chunk in app.astream({"value": "hi"}, stream_mode="updates"):
            chunks.append(chunk)

        assert len(chunks) >= 1
        found = any("a" in chunk for chunk in chunks)
        assert found

    @pytest.mark.asyncio
    async def test_ainvoke_multi_node(self):
        """Test async with multiple nodes."""

        def node_a(state: AsyncState) -> dict:
            return {"value": state["value"] + " A"}

        def node_b(state: AsyncState) -> dict:
            return {"value": state["value"] + " B"}

        graph = StateGraph(AsyncState)
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", END)
        app = graph.compile()

        result = await app.ainvoke({"value": "x"})
        assert result["value"] == "x A B"

    @pytest.mark.asyncio
    async def test_ainvoke_with_checkpointer(self):
        """Test async multi-turn with checkpointer."""

        def echo(state: AsyncListState) -> dict:
            last = state["items"][-1]
            return {"items": [f"echo:{last}"]}

        graph = StateGraph(AsyncListState)
        graph.add_node("echo", echo)
        graph.add_edge(START, "echo")
        graph.add_edge("echo", END)
        saver = InMemorySaver()
        app = graph.compile(checkpointer=saver)

        config: RunnableConfig = {"configurable": {"thread_id": "async-t1"}}

        result1 = await app.ainvoke({"items": ["hello"]}, config)
        assert "echo:hello" in result1["items"]

        result2 = await app.ainvoke({"items": ["world"]}, config)
        assert len(result2["items"]) > len(result1["items"])

    def test_invoke_respects_max_concurrency(self):
        class ParallelState(TypedDict):
            items: Annotated[list[str], operator.add]

        lock = threading.Lock()
        active = 0
        peak = 0

        def make_node(name: str):
            def node(state: ParallelState) -> dict:
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                try:
                    import time

                    time.sleep(0.05)
                    return {"items": [name]}
                finally:
                    with lock:
                        active -= 1

            return node

        graph = StateGraph(ParallelState)
        graph.add_node("a", make_node("a"))
        graph.add_node("b", make_node("b"))
        graph.add_node("c", make_node("c"))
        graph.add_edge(START, "a")
        graph.add_edge(START, "b")
        graph.add_edge(START, "c")
        graph.add_edge(["a", "b", "c"], END)
        app = graph.compile()

        result = app.invoke({"items": []}, {"max_concurrency": 1})
        assert sorted(result["items"]) == ["a", "b", "c"]
        assert peak == 1

    @pytest.mark.asyncio
    async def test_ainvoke_respects_max_concurrency(self):
        class ParallelState(TypedDict):
            items: Annotated[list[str], operator.add]

        active = 0
        peak = 0
        lock = asyncio.Lock()

        def make_node(name: str):
            async def node(state: ParallelState) -> dict:
                nonlocal active, peak
                async with lock:
                    active += 1
                    peak = max(peak, active)
                try:
                    await asyncio.sleep(0.05)
                    return {"items": [name]}
                finally:
                    async with lock:
                        active -= 1

            return node

        graph = StateGraph(ParallelState)
        graph.add_node("a", make_node("a"))
        graph.add_node("b", make_node("b"))
        graph.add_node("c", make_node("c"))
        graph.add_edge(START, "a")
        graph.add_edge(START, "b")
        graph.add_edge(START, "c")
        graph.add_edge(["a", "b", "c"], END)
        app = graph.compile()

        result = await app.ainvoke({"items": []}, {"max_concurrency": 1})
        assert sorted(result["items"]) == ["a", "b", "c"]
        assert peak == 1
