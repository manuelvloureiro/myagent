"""Import-swappable: update_state tests.

Verifies that external state updates via update_state are visible
in subsequent get_state calls.
"""

import operator
from typing import Annotated, TypedDict


class UpdateState(TypedDict):
    messages: Annotated[list, operator.add]


class TestUpdateStateCompat:
    def test_update_state_modifies_checkpoint(self, StateGraph, START, END, InMemorySaver):
        """update_state writes to the checkpoint and get_state reflects it."""

        def node(state: UpdateState) -> dict:
            return {"messages": ["response"]}

        graph = StateGraph(UpdateState)
        graph.add_node("node", node)
        graph.add_edge(START, "node")
        graph.add_edge("node", END)
        saver = InMemorySaver()
        app = graph.compile(checkpointer=saver)

        config = {"configurable": {"thread_id": "t-update"}}
        app.invoke({"messages": ["initial"]}, config)

        app.update_state(config, {"messages": ["injected"]}, as_node="node")

        state = app.get_state(config)
        assert "injected" in state.values["messages"]

    def test_update_state_uses_reducer(self, StateGraph, START, END, InMemorySaver):
        """update_state applies through the reducer, not raw overwrite."""

        def node(state: UpdateState) -> dict:
            return {"messages": ["from_node"]}

        graph = StateGraph(UpdateState)
        graph.add_node("node", node)
        graph.add_edge(START, "node")
        graph.add_edge("node", END)
        saver = InMemorySaver()
        app = graph.compile(checkpointer=saver)

        config = {"configurable": {"thread_id": "t-update-reducer"}}
        app.invoke({"messages": ["first"]}, config)

        app.update_state(config, {"messages": ["second"]}, as_node="node")

        state = app.get_state(config)
        # Both should be present because operator.add reducer accumulates
        assert "first" in state.values["messages"]
        assert "second" in state.values["messages"]
