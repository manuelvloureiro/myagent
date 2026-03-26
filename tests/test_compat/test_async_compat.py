"""Import-swappable: async tests."""

import operator
from typing import Annotated, TypedDict

import pytest


class AsyncState(TypedDict):
    value: str


class AsyncListState(TypedDict):
    items: Annotated[list, operator.add]


class TestAsyncCompat:
    def test_invoke_with_async_node(self, StateGraph, START, END):
        async def async_node(state: AsyncState) -> dict:
            return {"value": state["value"] + " truly_async"}

        graph = StateGraph(AsyncState)
        graph.add_node("a", async_node)
        graph.add_edge(START, "a")
        graph.add_edge("a", END)
        app = graph.compile()

        with pytest.raises(TypeError):
            app.invoke({"value": "test"})

    def test_stream_with_async_node(self, StateGraph, START, END):
        async def async_node(state: AsyncState) -> dict:
            return {"value": state["value"] + " streamed_async"}

        graph = StateGraph(AsyncState)
        graph.add_node("a", async_node)
        graph.add_edge(START, "a")
        graph.add_edge("a", END)
        app = graph.compile()

        with pytest.raises(TypeError):
            list(app.stream({"value": "test"}))

    @pytest.mark.asyncio
    async def test_ainvoke(self, StateGraph, START, END):
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
    async def test_ainvoke_with_async_node(self, StateGraph, START, END):
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
    async def test_astream_values(self, StateGraph, START, END):
        def node_a(state: AsyncState) -> dict:
            return {"value": state["value"] + " streamed"}

        graph = StateGraph(AsyncState)
        graph.add_node("a", node_a)
        graph.add_edge(START, "a")
        graph.add_edge("a", END)
        app = graph.compile()

        chunks = []
        async for chunk in app.astream({"value": "hi"}, stream_mode="values"):
            chunks.append(chunk)

        assert len(chunks) >= 1
        assert chunks[-1]["value"] == "hi streamed"

    @pytest.mark.asyncio
    async def test_ainvoke_multi_turn(self, StateGraph, START, END, InMemorySaver):
        def echo(state: AsyncListState) -> dict:
            last = state["items"][-1]
            return {"items": [f"echo:{last}"]}

        graph = StateGraph(AsyncListState)
        graph.add_node("echo", echo)
        graph.add_edge(START, "echo")
        graph.add_edge("echo", END)
        saver = InMemorySaver()
        app = graph.compile(checkpointer=saver)

        config = {"configurable": {"thread_id": "async-t1"}}

        result1 = await app.ainvoke({"items": ["hello"]}, config)
        assert "echo:hello" in result1["items"]

        result2 = await app.ainvoke({"items": ["world"]}, config)
        assert len(result2["items"]) > len(result1["items"])
