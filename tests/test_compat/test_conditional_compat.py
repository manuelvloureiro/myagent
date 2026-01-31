"""Import-swappable: conditional edge tests."""

from typing import TypedDict

import pytest


class RouterState(TypedDict):
    value: str
    route: str


class TestConditionalCompat:
    def test_basic_routing(self, StateGraph, START, END):
        def router(state: RouterState) -> str:
            return state["route"]

        def node_a(state: RouterState) -> dict:
            return {"value": state["value"] + " went_a"}

        def node_b(state: RouterState) -> dict:
            return {"value": state["value"] + " went_b"}

        graph = StateGraph(RouterState)
        graph.add_node("router_node", lambda state: state)
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_edge(START, "router_node")
        graph.add_conditional_edges("router_node", router, {"a": "a", "b": "b"})
        graph.add_edge("a", END)
        graph.add_edge("b", END)
        app = graph.compile()

        assert app.invoke({"value": "test", "route": "a"})["value"] == "test went_a"
        assert app.invoke({"value": "test", "route": "b"})["value"] == "test went_b"

    def test_routing_to_end(self, StateGraph, START, END):
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

        assert app.invoke({"value": "stop", "route": ""})["value"] == "stop"
        assert app.invoke({"value": "go", "route": ""})["value"] == "processed"

    def test_router_errors_are_raised(self, StateGraph, START, END):
        def router(state: RouterState) -> str:
            raise RuntimeError("boom")

        graph = StateGraph(RouterState)
        graph.add_node("router_node", lambda state: state)
        graph.add_node("a", lambda state: {"value": "A"})
        graph.add_edge(START, "router_node")
        graph.add_conditional_edges("router_node", router, {"a": "a"})
        graph.add_edge("a", END)
        app = graph.compile()

        with pytest.raises(RuntimeError, match="boom"):
            app.invoke({"value": "", "route": "a"})
