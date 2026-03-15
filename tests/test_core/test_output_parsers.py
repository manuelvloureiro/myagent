"""Tests for output parsers."""

import pytest
from myagent_core.messages import AIMessage
from myagent_core.output_parsers import JsonOutputParser, PydanticOutputParser, StrOutputParser


class TestStrOutputParser:
    def test_from_string(self):
        parser = StrOutputParser()
        assert parser.invoke("hello") == "hello"

    def test_from_ai_message(self):
        parser = StrOutputParser()
        msg = AIMessage(content="the answer is 42")
        assert parser.invoke(msg) == "the answer is 42"

    def test_from_dict(self):
        parser = StrOutputParser()
        assert parser.invoke({"content": "hello"}) == "hello"

    async def test_ainvoke(self):
        parser = StrOutputParser()
        assert await parser.ainvoke("async") == "async"

    def test_in_chain(self):
        from myagent_core import RunnableLambda

        chain = RunnableLambda(lambda x: AIMessage(content=f"Result: {x}")) | StrOutputParser()
        assert chain.invoke("test") == "Result: test"


class TestJsonOutputParser:
    def test_basic_json(self):
        parser = JsonOutputParser()
        result = parser.invoke('{"key": "value", "num": 42}')
        assert result == {"key": "value", "num": 42}

    def test_json_with_code_fence(self):
        parser = JsonOutputParser()
        text = '```json\n{"key": "value"}\n```'
        result = parser.invoke(text)
        assert result == {"key": "value"}

    def test_json_with_bare_fence(self):
        parser = JsonOutputParser()
        text = '```\n{"key": "value"}\n```'
        result = parser.invoke(text)
        assert result == {"key": "value"}

    def test_from_ai_message(self):
        parser = JsonOutputParser()
        msg = AIMessage(content='{"answer": 42}')
        result = parser.invoke(msg)
        assert result == {"answer": 42}

    def test_invalid_json(self):
        parser = JsonOutputParser()
        with pytest.raises(Exception):
            parser.invoke("not json")

    async def test_ainvoke(self):
        parser = JsonOutputParser()
        result = await parser.ainvoke('{"a": 1}')
        assert result == {"a": 1}

    def test_json_array(self):
        parser = JsonOutputParser()
        result = parser.invoke("[1, 2, 3]")
        assert result == [1, 2, 3]


class TestPydanticOutputParser:
    def test_basic(self):
        from pydantic import BaseModel

        class Answer(BaseModel):
            text: str
            confidence: float

        parser = PydanticOutputParser(pydantic_object=Answer)
        result = parser.invoke('{"text": "hello", "confidence": 0.95}')
        assert isinstance(result, Answer)
        assert result.text == "hello"
        assert result.confidence == 0.95

    def test_with_code_fence(self):
        from pydantic import BaseModel

        class Item(BaseModel):
            name: str
            count: int

        parser = PydanticOutputParser(pydantic_object=Item)
        result = parser.invoke('```json\n{"name": "apple", "count": 3}\n```')
        assert result.name == "apple"
        assert result.count == 3

    def test_validation_error(self):
        from pydantic import BaseModel

        class Strict(BaseModel):
            value: int

        parser = PydanticOutputParser(pydantic_object=Strict)
        with pytest.raises(Exception):
            parser.invoke('{"value": "not_an_int"}')

    async def test_ainvoke(self):
        from pydantic import BaseModel

        class Simple(BaseModel):
            x: int

        parser = PydanticOutputParser(pydantic_object=Simple)
        result = await parser.ainvoke('{"x": 42}')
        assert result.x == 42
