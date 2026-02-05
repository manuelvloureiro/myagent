"""Import-swappable: custom reducer tests.

Verifies that Annotated[T, custom_fn] reducers work identically,
not just operator.add.
"""

from typing import Annotated, TypedDict


def max_reducer(a: int, b: int) -> int:
    return max(a, b)


def concat_with_sep(a: str, b: str) -> str:
    return f"{a} | {b}" if a else b


class MaxState(TypedDict):
    score: Annotated[int, max_reducer]


class ConcatState(TypedDict):
    log: Annotated[str, concat_with_sep]


class TestCustomReducersCompat:
    def test_max_reducer(self, StateGraph, START, END):
        """Custom max reducer keeps the highest value across nodes."""

        def node_a(state: MaxState) -> dict:
            return {"score": 10}

        def node_b(state: MaxState) -> dict:
            return {"score": 5}

        graph = StateGraph(MaxState)
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", END)
        app = graph.compile()

        result = app.invoke({"score": 0})
        assert result["score"] == 10

    def test_string_concat_reducer(self, StateGraph, START, END):
        """Custom string reducer concatenates with separator."""

        def step1(state: ConcatState) -> dict:
            return {"log": "step1"}

        def step2(state: ConcatState) -> dict:
            return {"log": "step2"}

        graph = StateGraph(ConcatState)
        graph.add_node("step1", step1)
        graph.add_node("step2", step2)
        graph.add_edge(START, "step1")
        graph.add_edge("step1", "step2")
        graph.add_edge("step2", END)
        app = graph.compile()

        result = app.invoke({"log": ""})
        assert "step1" in result["log"]
        assert "step2" in result["log"]
