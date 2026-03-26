"""Tests for conditional edges and routing."""

from typing import TypedDict

import pytest
from myagent.graph import END, START, StateGraph


class RouterState(TypedDict):
    value: str
    route: str


class TestConditionalEdges:
    def test_basic_routing(self):
        """Route to different nodes based on state."""

        def router(state: RouterState) -> str:
            return state["route"]

        def node_a(state: RouterState) -> dict:
            return {"value": state["value"] + " went_a"}

        def node_b(state: RouterState) -> dict:
            return {"value": state["value"] + " went_b"}

        graph = StateGraph(RouterState)
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_edge(START, "router_node")
        graph.add_node("router_node", lambda state: state)
        graph.add_conditional_edges("router_node", router, {"a": "a", "b": "b"})
        graph.add_edge("a", END)
        graph.add_edge("b", END)
        app = graph.compile()

        result_a = app.invoke({"value": "test", "route": "a"})
        assert result_a["value"] == "test went_a"

        result_b = app.invoke({"value": "test", "route": "b"})
        assert result_b["value"] == "test went_b"

    def test_routing_to_end(self):
        """Route directly to END."""

        def should_continue(state: RouterState) -> str:
            if state["value"] == "stop":
                return END
            return "process"

        def process(state: RouterState) -> dict:
            return {"value": "processed"}

        graph = StateGraph(RouterState)
        graph.add_node("check", lambda state: state)
        graph.add_node("process", process)
        graph.add_edge(START, "check")
        graph.add_conditional_edges("check", should_continue, {END: END, "process": "process"})
        graph.add_edge("process", END)
        app = graph.compile()

        # Should stop
        result = app.invoke({"value": "stop", "route": ""})
        assert result["value"] == "stop"

        # Should process
        result = app.invoke({"value": "go", "route": ""})
        assert result["value"] == "processed"

    def test_conditional_without_map(self):
        """Conditional edges returning node names directly."""

        def router(state: RouterState) -> str:
            return "a" if state["route"] == "a" else "b"

        def node_a(state: RouterState) -> dict:
            return {"value": "A"}

        def node_b(state: RouterState) -> dict:
            return {"value": "B"}

        graph = StateGraph(RouterState)
        graph.add_node("start_node", lambda s: s)
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_edge(START, "start_node")
        graph.add_conditional_edges("start_node", router)
        graph.add_edge("a", END)
        graph.add_edge("b", END)
        app = graph.compile()

        assert app.invoke({"value": "", "route": "a"})["value"] == "A"
        assert app.invoke({"value": "", "route": "b"})["value"] == "B"

    def test_router_errors_are_raised(self):
        """Router exceptions should not be swallowed."""

        def router(state: RouterState) -> str:
            raise RuntimeError("boom")

        graph = StateGraph(RouterState)
        graph.add_node("start_node", lambda s: s)
        graph.add_node("a", lambda s: {"value": "A"})
        graph.add_edge(START, "start_node")
        graph.add_conditional_edges("start_node", router, {"a": "a"})
        graph.add_edge("a", END)
        app = graph.compile()

        with pytest.raises(RuntimeError, match="boom"):
            app.invoke({"value": "", "route": "a"})

    def test_then_is_explicitly_unsupported(self):
        graph = StateGraph(RouterState)
        graph.add_node("start_node", lambda s: s)

        with pytest.raises(NotImplementedError, match="then"):
            graph.add_conditional_edges("start_node", lambda s: "a", {"a": "a"}, then="next")
