"""Tests for recursion limit and GraphRecursionError."""

from typing import TypedDict

import pytest
from myagent.errors import GraphRecursionError
from myagent.graph import END, START, StateGraph


class LoopState(TypedDict):
    counter: int


class TestRecursionLimit:
    def test_infinite_loop_raises(self):
        """A graph that never reaches END should raise GraphRecursionError."""

        def increment(state: LoopState) -> dict:
            return {"counter": state["counter"] + 1}

        def always_loop(state: LoopState) -> str:
            return "increment"

        graph = StateGraph(LoopState)
        graph.add_node("increment", increment)
        graph.add_edge(START, "increment")
        graph.add_conditional_edges("increment", always_loop, {"increment": "increment"})
        app = graph.compile()

        with pytest.raises(GraphRecursionError):
            app.invoke({"counter": 0}, {"recursion_limit": 5})

    def test_loop_exits_before_limit(self):
        """A loop that exits within limit should succeed."""

        def increment(state: LoopState) -> dict:
            return {"counter": state["counter"] + 1}

        def check(state: LoopState) -> str:
            if state["counter"] >= 3:
                return END
            return "increment"

        graph = StateGraph(LoopState)
        graph.add_node("increment", increment)
        graph.add_edge(START, "increment")
        graph.add_conditional_edges("increment", check, {END: END, "increment": "increment"})
        app = graph.compile()

        result = app.invoke({"counter": 0}, {"recursion_limit": 25})
        assert result["counter"] >= 3

    def test_custom_recursion_limit(self):
        """Recursion limit from config should be respected."""

        def increment(state: LoopState) -> dict:
            return {"counter": state["counter"] + 1}

        def always_loop(state: LoopState) -> str:
            return "increment"

        graph = StateGraph(LoopState)
        graph.add_node("increment", increment)
        graph.add_edge(START, "increment")
        graph.add_conditional_edges("increment", always_loop, {"increment": "increment"})
        app = graph.compile()

        with pytest.raises(GraphRecursionError):
            app.invoke({"counter": 0}, {"recursion_limit": 3})


class TestRecursionLimitAsync:
    @pytest.mark.asyncio
    async def test_async_infinite_loop_raises(self):
        def increment(state: LoopState) -> dict:
            return {"counter": state["counter"] + 1}

        def always_loop(state: LoopState) -> str:
            return "increment"

        graph = StateGraph(LoopState)
        graph.add_node("increment", increment)
        graph.add_edge(START, "increment")
        graph.add_conditional_edges("increment", always_loop, {"increment": "increment"})
        app = graph.compile()

        with pytest.raises(GraphRecursionError):
            await app.ainvoke({"counter": 0}, {"recursion_limit": 5})
