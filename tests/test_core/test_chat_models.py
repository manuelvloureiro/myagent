"""Tests for chat model interface."""

from myagent_core.chat_models import FakeListChatModel, FakeToolCallingModel
from myagent_core.messages import AIMessage, HumanMessage, SystemMessage, ToolCall


class TestFakeListChatModel:
    def test_basic_invoke(self):
        model = FakeListChatModel(responses=["Hello!", "Goodbye!"])
        result = model.invoke([HumanMessage(content="Hi")])
        assert isinstance(result, AIMessage)
        assert result.content == "Hello!"

    def test_cycles_through_responses(self):
        model = FakeListChatModel(responses=["A", "B", "C"])
        assert model.invoke("first").content == "A"
        assert model.invoke("second").content == "B"
        assert model.invoke("third").content == "C"
        assert model.invoke("fourth").content == "A"  # wraps around

    def test_string_input_coercion(self):
        model = FakeListChatModel(responses=["response"])
        result = model.invoke("plain string")
        assert result.content == "response"

    def test_stop_sequences(self):
        model = FakeListChatModel(responses=["Hello world! How are you?"])
        result = model.invoke("hi", stop=["!"])
        assert result.content == "Hello world"

    def test_stream(self):
        model = FakeListChatModel(responses=["chunk"])
        chunks = list(model.stream([HumanMessage(content="Hi")]))
        assert len(chunks) == 1
        assert chunks[0].content == "chunk"

    async def test_ainvoke(self):
        model = FakeListChatModel(responses=["async response"])
        result = await model.ainvoke([HumanMessage(content="Hi")])
        assert result.content == "async response"

    async def test_astream(self):
        model = FakeListChatModel(responses=["async chunk"])
        chunks = []
        async for chunk in model.astream("Hi"):
            chunks.append(chunk)
        assert len(chunks) == 1
        assert chunks[0].content == "async chunk"

    def test_in_chain(self):
        from myagent_core.output_parsers import StrOutputParser
        from myagent_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([("human", "{question}")])
        model = FakeListChatModel(responses=["42"])
        parser = StrOutputParser()

        chain = prompt | model | parser
        result = chain.invoke({"question": "What is the answer?"})
        assert result == "42"


class TestFakeToolCallingModel:
    def test_returns_tool_calls(self):
        tc = ToolCall(name="search", args={"q": "test"})
        model = FakeToolCallingModel(tool_calls=[tc], content="Let me search")
        result = model.invoke([HumanMessage(content="search")])
        assert result.content == "Let me search"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"

    def test_multiple_tool_calls(self):
        tcs = [
            ToolCall(name="search", args={"q": "a"}),
            ToolCall(name="calc", args={"expr": "2+2"}),
        ]
        model = FakeToolCallingModel(tool_calls=tcs)
        result = model.invoke("do stuff")
        assert len(result.tool_calls) == 2


class TestBindTools:
    def test_bind_tools_returns_copy(self):
        model = FakeListChatModel(responses=["hi"])
        bound = model.bind_tools(["tool1", "tool2"])
        assert len(bound._bound_tools) == 2
        assert len(model._bound_tools) == 0  # original unchanged

    def test_bound_model_still_works(self):
        model = FakeListChatModel(responses=["hi"])
        bound = model.bind_tools(["tool1"])
        result = bound.invoke("hello")
        assert result.content == "hi"


class TestInputCoercion:
    def test_string(self):
        model = FakeListChatModel(responses=["ok"])
        result = model.invoke("plain text")
        assert result.content == "ok"

    def test_single_message(self):
        model = FakeListChatModel(responses=["ok"])
        result = model.invoke(HumanMessage(content="hi"))
        assert result.content == "ok"

    def test_list_of_messages(self):
        model = FakeListChatModel(responses=["ok"])
        result = model.invoke([SystemMessage(content="sys"), HumanMessage(content="hi")])
        assert result.content == "ok"

    def test_list_of_strings(self):
        model = FakeListChatModel(responses=["ok"])
        result = model.invoke(["hello", "world"])
        assert result.content == "ok"

    def test_list_of_dicts(self):
        model = FakeListChatModel(responses=["ok"])
        result = model.invoke([{"content": "hello"}])
        assert result.content == "ok"
