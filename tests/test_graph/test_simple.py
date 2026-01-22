"""Tests for simple linear graphs."""

from typing import TypedDict

from myagent.graph import END, START, StateGraph


class SimpleState(TypedDict):
    value: str


class TestLinearGraph:
    def test_single_node(self):
        """START -> A -> END"""

        def node_a(state: SimpleState) -> dict:
            return {"value": state["value"] + " processed"}

        graph = StateGraph(SimpleState)
        graph.add_node("a", node_a)
        graph.add_edge(START, "a")
        graph.add_edge("a", END)
        app = graph.compile()

        result = app.invoke({"value": "hello"})
        assert result["value"] == "hello processed"

    def test_two_nodes(self):
        """START -> A -> B -> END"""

        def node_a(state: SimpleState) -> dict:
            return {"value": state["value"] + " A"}

        def node_b(state: SimpleState) -> dict:
            return {"value": state["value"] + " B"}

        graph = StateGraph(SimpleState)
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", END)
        app = graph.compile()

        result = app.invoke({"value": "start"})
        assert result["value"] == "start A B"

    def test_three_nodes(self):
        """START -> A -> B -> C -> END"""

        def node_a(state: SimpleState) -> dict:
            return {"value": state["value"] + "1"}

        def node_b(state: SimpleState) -> dict:
            return {"value": state["value"] + "2"}

        def node_c(state: SimpleState) -> dict:
            return {"value": state["value"] + "3"}

        graph = StateGraph(SimpleState)
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_node("c", node_c)
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")
        graph.add_edge("c", END)
        app = graph.compile()

        result = app.invoke({"value": "x"})
        assert result["value"] == "x123"

    def test_stream_values(self):
        """Test streaming in values mode."""

        def node_a(state: SimpleState) -> dict:
            return {"value": state["value"] + " A"}

        graph = StateGraph(SimpleState)
        graph.add_node("a", node_a)
        graph.add_edge(START, "a")
        graph.add_edge("a", END)
        app = graph.compile()

        chunks = list(app.stream({"value": "hi"}))
        # values mode: initial state + after each step
        assert len(chunks) >= 1
        assert chunks[-1]["value"] == "hi A"

    def test_stream_updates(self):
        """Test streaming in updates mode."""

        def node_a(state: SimpleState) -> dict:
            return {"value": state["value"] + " updated"}

        graph = StateGraph(SimpleState)
        graph.add_node("a", node_a)
        graph.add_edge(START, "a")
        graph.add_edge("a", END)
        app = graph.compile()

        chunks = list(app.stream({"value": "hi"}, stream_mode="updates"))
        assert len(chunks) >= 1
        # Updates mode yields {node_name: update_dict}
        found = False
        for chunk in chunks:
            if "a" in chunk:
                found = True
                assert "value" in chunk["a"]
        assert found

    def test_inferred_node_name(self):
        """Test add_node with name inferred from function."""

        def my_node(state: SimpleState) -> dict:
            return {"value": state["value"] + " inferred"}

        graph = StateGraph(SimpleState)
        graph.add_node(my_node)
        graph.add_edge(START, "my_node")
        graph.add_edge("my_node", END)
        app = graph.compile()

        result = app.invoke({"value": "test"})
        assert result["value"] == "test inferred"
