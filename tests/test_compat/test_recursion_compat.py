"""Import-swappable: recursion limit tests.

Verifies that both implementations raise GraphRecursionError when
the recursion limit is exceeded, and that loops exiting within
the limit succeed normally.
"""

from typing import TypedDict


class LoopState(TypedDict):
    counter: int


class TestRecursionCompat:
    def test_infinite_loop_raises(self, StateGraph, START, END, GraphRecursionError):
        """A graph that never reaches END raises GraphRecursionError."""

        def increment(state: LoopState) -> dict:
            return {"counter": state["counter"] + 1}

        def always_loop(state: LoopState) -> str:
            return "increment"

        graph = StateGraph(LoopState)
        graph.add_node("increment", increment)
        graph.add_edge(START, "increment")
        graph.add_conditional_edges("increment", always_loop, {"increment": "increment"})
        app = graph.compile()

        import pytest

        with pytest.raises(GraphRecursionError):
            app.invoke({"counter": 0}, {"recursion_limit": 5})

    def test_loop_exits_before_limit(self, StateGraph, START, END):
        """A conditional loop that exits within the limit succeeds."""

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

    def test_custom_recursion_limit_respected(self, StateGraph, START, END, GraphRecursionError):
        """The recursion_limit from config is honored."""

        def increment(state: LoopState) -> dict:
            return {"counter": state["counter"] + 1}

        def always_loop(state: LoopState) -> str:
            return "increment"

        graph = StateGraph(LoopState)
        graph.add_node("increment", increment)
        graph.add_edge(START, "increment")
        graph.add_conditional_edges("increment", always_loop, {"increment": "increment"})
        app = graph.compile()

        import pytest

        with pytest.raises(GraphRecursionError):
            app.invoke({"counter": 0}, {"recursion_limit": 3})
