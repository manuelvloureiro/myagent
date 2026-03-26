"""Tests for subgraph composition via add_subgraph()."""

from typing import Annotated, TypedDict

import pytest
from myagent.graph import END, START, StateGraph


def _append(left: list, right: list) -> list:
    return left + right


class ParentState(TypedDict):
    value: str
    child_result: str
    log: Annotated[list[str], _append]


class ChildState(TypedDict):
    input_value: str
    output_value: str


class InnerState(TypedDict):
    x: int
    y: int


class OuterState(TypedDict):
    a: int
    b: int
    result: int


class CounterState(TypedDict):
    items: Annotated[list[str], _append]


def _build_child_graph() -> StateGraph:
    def process(state: ChildState) -> dict:
        return {"output_value": state["input_value"] + " child"}

    child = StateGraph(ChildState)
    child.add_node("process", process)
    child.add_edge(START, "process")
    child.add_edge("process", END)
    return child


def _build_math_child() -> StateGraph:
    def multiply(state: InnerState) -> dict:
        return {"y": state["x"] * state["y"]}

    child = StateGraph(InnerState)
    child.add_node("multiply", multiply)
    child.add_edge(START, "multiply")
    child.add_edge("multiply", END)
    return child


def _build_conditional_child() -> StateGraph:
    def check(state: ChildState) -> dict:
        return {}

    def upper(state: ChildState) -> dict:
        return {"output_value": state["input_value"].upper()}

    def lower(state: ChildState) -> dict:
        return {"output_value": state["input_value"].lower()}

    def route(state: ChildState) -> str:
        if state.get("input_value", "").startswith("UP:"):
            return "upper"
        return "lower"

    child = StateGraph(ChildState)
    child.add_node("check", check)
    child.add_node("upper", upper)
    child.add_node("lower", lower)
    child.add_edge(START, "check")
    child.add_conditional_edges("check", route, {"upper": "upper", "lower": "lower"})
    child.add_edge("upper", END)
    child.add_edge("lower", END)
    return child


class TestSubgraphBasic:
    def test_simple_subgraph(self):
        child = _build_child_graph().compile()

        parent = StateGraph(ParentState)

        def before(state: ParentState) -> dict:
            return {"value": state["value"] + " before"}

        parent.add_node("before", before)
        parent.add_subgraph(
            "child",
            child,
            input_map=lambda s: {"input_value": s["value"]},
            output_map=lambda result, parent: {"child_result": result["output_value"]},
        )

        parent.add_edge(START, "before")
        parent.add_edge("before", "child")
        parent.add_edge("child", END)

        app = parent.compile()
        result = app.invoke({"value": "hello"})

        assert result["value"] == "hello before"
        assert result["child_result"] == "hello before child"

    def test_subgraph_with_post_processing(self):
        child = _build_child_graph().compile()

        def after(state: ParentState) -> dict:
            return {"value": state["child_result"] + " after"}

        parent = StateGraph(ParentState)
        parent.add_subgraph(
            "child",
            child,
            input_map=lambda s: {"input_value": s.get("value", "")},
            output_map=lambda r, p: {"child_result": r["output_value"]},
        )
        parent.add_node("after", after)

        parent.add_edge(START, "child")
        parent.add_edge("child", "after")
        parent.add_edge("after", END)

        app = parent.compile()
        result = app.invoke({"value": "start"})

        assert result["child_result"] == "start child"
        assert result["value"] == "start child after"

    def test_subgraph_with_math(self):
        child = _build_math_child().compile()

        parent = StateGraph(OuterState)
        parent.add_subgraph(
            "compute",
            child,
            input_map=lambda s: {"x": s["a"], "y": s["b"]},
            output_map=lambda r, p: {"result": r["y"]},
        )
        parent.add_edge(START, "compute")
        parent.add_edge("compute", END)

        app = parent.compile()
        result = app.invoke({"a": 7, "b": 6})

        assert result["result"] == 42


class TestSubgraphConditional:
    def test_conditional_routing_to_subgraph(self):
        child = _build_child_graph().compile()

        def router(state: ParentState) -> str:
            if state.get("value", "").startswith("use_child"):
                return "child"
            return "skip"

        def skip(state: ParentState) -> dict:
            return {"child_result": "skipped"}

        parent = StateGraph(ParentState)
        parent.add_subgraph(
            "child",
            child,
            input_map=lambda s: {"input_value": s["value"]},
            output_map=lambda r, p: {"child_result": r["output_value"]},
        )
        parent.add_node("skip", skip)

        parent.add_conditional_edges(START, router, {"child": "child", "skip": "skip"})
        parent.add_edge("child", END)
        parent.add_edge("skip", END)

        app = parent.compile()

        result = app.invoke({"value": "use_child please"})
        assert result["child_result"] == "use_child please child"

        result = app.invoke({"value": "nope"})
        assert result["child_result"] == "skipped"

    def test_subgraph_with_internal_conditional(self):
        child = _build_conditional_child().compile()

        parent = StateGraph(ParentState)
        parent.add_subgraph(
            "child",
            child,
            input_map=lambda s: {"input_value": s["value"]},
            output_map=lambda r, p: {"child_result": r["output_value"]},
        )
        parent.add_edge(START, "child")
        parent.add_edge("child", END)

        app = parent.compile()

        result = app.invoke({"value": "UP:hello"})
        assert result["child_result"] == "UP:HELLO"

        result = app.invoke({"value": "hello"})
        assert result["child_result"] == "hello"


class TestSubgraphReducers:
    def test_output_map_with_reducer(self):
        child = _build_child_graph().compile()

        parent = StateGraph(CounterState)
        parent.add_subgraph(
            "child",
            child,
            input_map=lambda s: {"input_value": "item"},
            output_map=lambda r, p: {"items": [r["output_value"]]},
        )
        parent.add_edge(START, "child")
        parent.add_edge("child", END)

        app = parent.compile()
        result = app.invoke({"items": ["existing"]})

        assert result["items"] == ["existing", "item child"]


class TestSubgraphMultiple:
    def test_two_subgraphs_sequential(self):
        child_a = _build_child_graph().compile()
        child_b = _build_child_graph().compile()

        parent = StateGraph(ParentState)
        parent.add_subgraph(
            "child_a",
            child_a,
            input_map=lambda s: {"input_value": s.get("value", "")},
            output_map=lambda r, p: {"value": r["output_value"]},
        )
        parent.add_subgraph(
            "child_b",
            child_b,
            input_map=lambda s: {"input_value": s.get("value", "")},
            output_map=lambda r, p: {"child_result": r["output_value"]},
        )

        parent.add_edge(START, "child_a")
        parent.add_edge("child_a", "child_b")
        parent.add_edge("child_b", END)

        app = parent.compile()
        result = app.invoke({"value": "start"})

        assert result["value"] == "start child"
        assert result["child_result"] == "start child child"

    def test_subgraph_and_regular_node_mixed(self):
        child = _build_child_graph().compile()

        def step_a(state: ParentState) -> dict:
            return {"value": state.get("value", "") + "_A"}

        def step_c(state: ParentState) -> dict:
            return {"value": state.get("child_result", "") + "_C"}

        parent = StateGraph(ParentState)
        parent.add_node("step_a", step_a)
        parent.add_subgraph(
            "child",
            child,
            input_map=lambda s: {"input_value": s["value"]},
            output_map=lambda r, p: {"child_result": r["output_value"]},
        )
        parent.add_node("step_c", step_c)

        parent.add_edge(START, "step_a")
        parent.add_edge("step_a", "child")
        parent.add_edge("child", "step_c")
        parent.add_edge("step_c", END)

        app = parent.compile()
        result = app.invoke({"value": "x"})

        assert result["value"] == "x_A child_C"
        assert result["child_result"] == "x_A child"


class TestSubgraphAsync:
    @pytest.mark.asyncio
    async def test_async_subgraph(self):
        child = _build_child_graph().compile()

        parent = StateGraph(ParentState)
        parent.add_subgraph(
            "child",
            child,
            input_map=lambda s: {"input_value": s.get("value", "")},
            output_map=lambda r, p: {"child_result": r["output_value"]},
        )
        parent.add_edge(START, "child")
        parent.add_edge("child", END)

        app = parent.compile()
        result = await app.ainvoke({"value": "async_test"})

        assert result["child_result"] == "async_test child"

    @pytest.mark.asyncio
    async def test_async_subgraph_with_async_child_nodes(self):
        async def async_process(state: ChildState) -> dict:
            return {"output_value": state["input_value"] + " async_child"}

        child = StateGraph(ChildState)
        child.add_node("process", async_process)
        child.add_edge(START, "process")
        child.add_edge("process", END)
        compiled_child = child.compile()

        parent = StateGraph(ParentState)
        parent.add_subgraph(
            "child",
            compiled_child,
            input_map=lambda s: {"input_value": s.get("value", "")},
            output_map=lambda r, p: {"child_result": r["output_value"]},
        )
        parent.add_edge(START, "child")
        parent.add_edge("child", END)

        app = parent.compile()
        result = await app.ainvoke({"value": "test"})

        assert result["child_result"] == "test async_child"


class TestSubgraphStream:
    def test_stream_values_with_subgraph(self):
        child = _build_child_graph().compile()

        parent = StateGraph(ParentState)
        parent.add_subgraph(
            "child",
            child,
            input_map=lambda s: {"input_value": s.get("value", "")},
            output_map=lambda r, p: {"child_result": r["output_value"]},
        )
        parent.add_edge(START, "child")
        parent.add_edge("child", END)

        app = parent.compile()
        chunks = list(app.stream({"value": "hello"}, stream_mode="values"))

        assert len(chunks) >= 2
        assert chunks[-1]["child_result"] == "hello child"

    def test_stream_updates_with_subgraph(self):
        child = _build_child_graph().compile()

        parent = StateGraph(ParentState)
        parent.add_subgraph(
            "child",
            child,
            input_map=lambda s: {"input_value": s.get("value", "")},
            output_map=lambda r, p: {"child_result": r["output_value"]},
        )
        parent.add_edge(START, "child")
        parent.add_edge("child", END)

        app = parent.compile()
        chunks = list(app.stream({"value": "hello"}, stream_mode="updates"))

        assert len(chunks) >= 1
        assert "child" in chunks[0]
        assert chunks[0]["child"]["child_result"] == "hello child"
