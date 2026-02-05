"""Import-swappable: checkpoint / multi-turn tests."""

import operator
from typing import Annotated, TypedDict


class ConversationState(TypedDict):
    messages: Annotated[list, operator.add]


class TestCheckpointCompat:
    def test_multi_turn_persistence(self, StateGraph, START, END, InMemorySaver):
        def echo(state: ConversationState) -> dict:
            last = state["messages"][-1]
            return {"messages": [f"echo: {last}"]}

        graph = StateGraph(ConversationState)
        graph.add_node("echo", echo)
        graph.add_edge(START, "echo")
        graph.add_edge("echo", END)
        saver = InMemorySaver()
        app = graph.compile(checkpointer=saver)

        config = {"configurable": {"thread_id": "t1"}}

        result1 = app.invoke({"messages": ["hello"]}, config)
        assert "echo: hello" in result1["messages"]

        result2 = app.invoke({"messages": ["world"]}, config)
        assert len(result2["messages"]) > len(result1["messages"])
        assert "echo: world" in result2["messages"]

    def test_separate_threads(self, StateGraph, START, END, InMemorySaver):
        def process(state: ConversationState) -> dict:
            return {"messages": [f"processed {len(state['messages'])}"]}

        graph = StateGraph(ConversationState)
        graph.add_node("process", process)
        graph.add_edge(START, "process")
        graph.add_edge("process", END)
        saver = InMemorySaver()
        app = graph.compile(checkpointer=saver)

        config1 = {"configurable": {"thread_id": "thread-1"}}
        config2 = {"configurable": {"thread_id": "thread-2"}}

        app.invoke({"messages": ["a"]}, config1)
        app.invoke({"messages": ["b"]}, config2)

        state1 = app.get_state(config1)
        state2 = app.get_state(config2)

        assert state1.values != state2.values

    def test_get_state(self, StateGraph, START, END, InMemorySaver):
        def node(state: ConversationState) -> dict:
            return {"messages": ["done"]}

        graph = StateGraph(ConversationState)
        graph.add_node("node", node)
        graph.add_edge(START, "node")
        graph.add_edge("node", END)
        saver = InMemorySaver()
        app = graph.compile(checkpointer=saver)

        config = {"configurable": {"thread_id": "t1"}}
        app.invoke({"messages": ["start"]}, config)

        state = app.get_state(config)
        assert "start" in state.values["messages"]
        assert "done" in state.values["messages"]
