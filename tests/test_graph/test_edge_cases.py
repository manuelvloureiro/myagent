"""Tests for edge cases and error handling."""

import operator
from typing import Annotated, TypedDict

import pytest
from myagent.graph import END, START, StateGraph


class SimpleState(TypedDict):
    value: str


class MultiFieldState(TypedDict):
    a: str
    b: int
    c: Annotated[list, operator.add]


class TestNodeReturnValues:
    def test_node_returns_partial_state(self):
        """Nodes can return a subset of state keys."""

        def node(state: MultiFieldState) -> dict:
            return {"a": "updated"}

        graph = StateGraph(MultiFieldState)
        graph.add_node("node", node)
        graph.add_edge(START, "node")
        graph.add_edge("node", END)
        app = graph.compile()

        result = app.invoke({"a": "old", "b": 42, "c": []})
        assert result["a"] == "updated"
        assert result["b"] == 42

    def test_node_returns_none(self):
        """Node returning None should not crash."""

        def node(state: SimpleState):
            return None

        graph = StateGraph(SimpleState)
        graph.add_node("node", node)
        graph.add_edge(START, "node")
        graph.add_edge("node", END)
        app = graph.compile()

        result = app.invoke({"value": "hello"})
        assert result["value"] == "hello"

    def test_node_returns_empty_dict(self):
        """Node returning empty dict should not change state."""

        def node(state: SimpleState) -> dict:
            return {}

        graph = StateGraph(SimpleState)
        graph.add_node("node", node)
        graph.add_edge(START, "node")
        graph.add_edge("node", END)
        app = graph.compile()

        result = app.invoke({"value": "unchanged"})
        assert result["value"] == "unchanged"


class TestMultipleFields:
    def test_mixed_channels(self):
        """State with both LastValue and reducer channels."""

        def node(state: MultiFieldState) -> dict:
            return {"a": "new_a", "b": state["b"] + 1, "c": ["item"]}

        graph = StateGraph(MultiFieldState)
        graph.add_node("node", node)
        graph.add_edge(START, "node")
        graph.add_edge("node", END)
        app = graph.compile()

        result = app.invoke({"a": "old", "b": 0, "c": ["existing"]})
        assert result["a"] == "new_a"
        assert result["b"] == 1
        assert "existing" in result["c"]
        assert "item" in result["c"]


class TestStreamModes:
    def test_stream_values_yields_full_state(self):
        def node_a(state: SimpleState) -> dict:
            return {"value": state["value"] + "_a"}

        def node_b(state: SimpleState) -> dict:
            return {"value": state["value"] + "_b"}

        graph = StateGraph(SimpleState)
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", END)
        app = graph.compile()

        chunks = list(app.stream({"value": "x"}, stream_mode="values"))
        # Each chunk should be full state dict
        for chunk in chunks:
            assert "value" in chunk
        assert chunks[-1]["value"] == "x_a_b"

    def test_stream_updates_yields_node_outputs(self):
        def node_a(state: SimpleState) -> dict:
            return {"value": "from_a"}

        graph = StateGraph(SimpleState)
        graph.add_node("a", node_a)
        graph.add_edge(START, "a")
        graph.add_edge("a", END)
        app = graph.compile()

        chunks = list(app.stream({"value": "x"}, stream_mode="updates"))
        assert any("a" in chunk for chunk in chunks)


class TestWithoutCheckpointer:
    def test_invoke_no_checkpointer(self):
        """Graph works fine without checkpointer."""

        def node(state: SimpleState) -> dict:
            return {"value": "done"}

        graph = StateGraph(SimpleState)
        graph.add_node("node", node)
        graph.add_edge(START, "node")
        graph.add_edge("node", END)
        app = graph.compile()

        result = app.invoke({"value": "start"})
        assert result["value"] == "done"

    def test_get_state_without_checkpointer_raises(self):
        def node(state: SimpleState) -> dict:
            return {"value": "done"}

        graph = StateGraph(SimpleState)
        graph.add_node("node", node)
        graph.add_edge(START, "node")
        graph.add_edge("node", END)
        app = graph.compile()

        with pytest.raises(ValueError, match="No checkpointer"):
            app.get_state({"configurable": {"thread_id": "t1"}})

    def test_update_state_without_checkpointer_raises(self):
        def node(state: SimpleState) -> dict:
            return {"value": "done"}

        graph = StateGraph(SimpleState)
        graph.add_node("node", node)
        graph.add_edge(START, "node")
        graph.add_edge("node", END)
        app = graph.compile()

        with pytest.raises(ValueError, match="No checkpointer"):
            app.update_state({"configurable": {"thread_id": "t1"}}, {"value": "x"})


class TestInvokeWithThreadButNoCheckpointer:
    def test_thread_id_ignored_without_checkpointer(self):
        """thread_id in config should be fine even without checkpointer."""

        def node(state: SimpleState) -> dict:
            return {"value": state["value"] + " done"}

        graph = StateGraph(SimpleState)
        graph.add_node("node", node)
        graph.add_edge(START, "node")
        graph.add_edge("node", END)
        app = graph.compile()

        result = app.invoke(
            {"value": "start"},
            {"configurable": {"thread_id": "t1"}},
        )
        assert result["value"] == "start done"
