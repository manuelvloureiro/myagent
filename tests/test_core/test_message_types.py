"""Tests for message types and prompt templates."""

import pytest
from myagent_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    _message_from_tuple,
)
from myagent_core.prompts import ChatPromptTemplate, PromptTemplate


class TestMessages:
    def test_human_message(self):
        m = HumanMessage(content="Hello")
        assert m.content == "Hello"
        assert m.type == "human"
        assert m.id  # auto-generated

    def test_ai_message(self):
        m = AIMessage(content="Hi there")
        assert m.content == "Hi there"
        assert m.type == "ai"
        assert m.tool_calls == []

    def test_ai_message_with_tool_calls(self):
        tc = ToolCall(name="search", args={"q": "test"})
        m = AIMessage(content="", tool_calls=[tc])
        assert len(m.tool_calls) == 1
        assert m.tool_calls[0].name == "search"
        assert m.tool_calls[0].args == {"q": "test"}

    def test_system_message(self):
        m = SystemMessage(content="You are helpful.")
        assert m.type == "system"

    def test_tool_message(self):
        m = ToolMessage(content="result data", tool_call_id="tc_123")
        assert m.type == "tool"
        assert m.tool_call_id == "tc_123"

    def test_message_equality(self):
        m1 = HumanMessage(content="Hello", id="1")
        m2 = HumanMessage(content="Hello", id="1")
        assert m1 == m2

    def test_message_inequality(self):
        m1 = HumanMessage(content="Hello", id="1")
        m2 = HumanMessage(content="Hello", id="2")
        assert m1 != m2

    def test_message_dict(self):
        m = HumanMessage(content="Hello", id="test-id")
        d = m.dict()
        assert d["type"] == "human"
        assert d["content"] == "Hello"
        assert d["id"] == "test-id"

    def test_ai_message_dict_with_tool_calls(self):
        tc = ToolCall(name="search", args={"q": "test"}, id="tc1")
        m = AIMessage(content="", tool_calls=[tc])
        d = m.dict()
        assert "tool_calls" in d
        assert d["tool_calls"][0]["name"] == "search"

    def test_tool_message_dict(self):
        m = ToolMessage(content="result", tool_call_id="tc_123")
        d = m.dict()
        assert d["tool_call_id"] == "tc_123"

    def test_message_name(self):
        m = HumanMessage(content="Hi", name="Alice")
        assert m.name == "Alice"
        d = m.dict()
        assert d["name"] == "Alice"

    def test_message_repr(self):
        m = HumanMessage(content="Hello")
        assert "HumanMessage" in repr(m)
        assert "Hello" in repr(m)

    def test_message_hash(self):
        m1 = HumanMessage(content="Hello", id="1")
        m2 = HumanMessage(content="Hello", id="1")
        assert hash(m1) == hash(m2)

    def test_message_additional_kwargs(self):
        m = AIMessage(content="Hi", additional_kwargs={"custom": "data"})
        assert m.additional_kwargs == {"custom": "data"}

    def test_tool_call_dict(self):
        tc = ToolCall(name="search", args={"q": "test"}, id="tc1")
        d = tc.dict()
        assert d == {"name": "search", "args": {"q": "test"}, "id": "tc1", "type": "tool_call"}

    def test_tool_call_repr(self):
        tc = ToolCall(name="search", args={"q": "test"}, id="tc1")
        assert "search" in repr(tc)

    def test_ai_message_usage_metadata(self):
        m = AIMessage(content="Hi", usage_metadata={"input_tokens": 10, "output_tokens": 5})
        assert m.usage_metadata == {"input_tokens": 10, "output_tokens": 5}
        d = m.dict()
        assert d["usage_metadata"] == {"input_tokens": 10, "output_tokens": 5}


class TestMessageFromTuple:
    def test_human(self):
        m = _message_from_tuple("human", "Hello")
        assert isinstance(m, HumanMessage)

    def test_user(self):
        m = _message_from_tuple("user", "Hello")
        assert isinstance(m, HumanMessage)

    def test_ai(self):
        m = _message_from_tuple("ai", "Hi")
        assert isinstance(m, AIMessage)

    def test_assistant(self):
        m = _message_from_tuple("assistant", "Hi")
        assert isinstance(m, AIMessage)

    def test_system(self):
        m = _message_from_tuple("system", "You are helpful.")
        assert isinstance(m, SystemMessage)

    def test_unknown_role(self):
        with pytest.raises(ValueError, match="Unknown message role"):
            _message_from_tuple("unknown", "content")


class TestPromptTemplate:
    def test_basic(self):
        pt = PromptTemplate(template="Hello, {name}!")
        assert pt.invoke({"name": "world"}) == "Hello, world!"

    def test_from_template(self):
        pt = PromptTemplate.from_template("Hello, {name}!")
        assert pt.format(name="world") == "Hello, world!"

    def test_input_variables(self):
        pt = PromptTemplate(template="{a} and {b}")
        assert set(pt.input_variables) == {"a", "b"}

    async def test_ainvoke(self):
        pt = PromptTemplate(template="Hello, {name}!")
        assert await pt.ainvoke({"name": "async"}) == "Hello, async!"

    def test_in_chain(self):
        pt = PromptTemplate(template="Hello, {name}!")
        chain = pt | (lambda x: x.upper())
        assert chain.invoke({"name": "world"}) == "HELLO, WORLD!"


class TestChatPromptTemplate:
    def test_from_messages(self):
        ct = ChatPromptTemplate.from_messages(
            [
                ("system", "You are {role}."),
                ("human", "{question}"),
            ]
        )
        msgs = ct.invoke({"role": "helpful", "question": "Hi!"})
        assert len(msgs) == 2
        assert msgs[0].type == "system"
        assert msgs[0].content == "You are helpful."
        assert msgs[1].type == "human"
        assert msgs[1].content == "Hi!"

    def test_from_template(self):
        ct = ChatPromptTemplate.from_template("Hello, {name}!")
        msgs = ct.invoke({"name": "world"})
        assert len(msgs) == 1
        assert msgs[0].type == "human"
        assert msgs[0].content == "Hello, world!"

    def test_input_variables(self):
        ct = ChatPromptTemplate.from_messages(
            [
                ("system", "You are {role}."),
                ("human", "{question}"),
            ]
        )
        assert set(ct.input_variables) == {"role", "question"}

    def test_with_base_message(self):
        sys_msg = SystemMessage(content="Fixed system message")
        ct = ChatPromptTemplate(
            messages=[
                sys_msg,
                ("human", "{question}"),
            ]
        )
        msgs = ct.invoke({"question": "Hi!"})
        assert len(msgs) == 2
        assert msgs[0] is sys_msg
        assert msgs[1].content == "Hi!"

    async def test_ainvoke(self):
        ct = ChatPromptTemplate.from_messages([("human", "{q}")])
        msgs = await ct.ainvoke({"q": "test"})
        assert msgs[0].content == "test"

    def test_format_messages(self):
        ct = ChatPromptTemplate.from_messages([("human", "{text}")])
        msgs = ct.format_messages(text="hello")
        assert msgs[0].content == "hello"

    def test_in_chain(self):
        ct = ChatPromptTemplate.from_messages([("human", "{q}")])
        chain = ct | (lambda msgs: len(msgs))
        assert chain.invoke({"q": "test"}) == 1
