"""Tests for tool abstractions."""

from myagent_core.tools import tool


class TestToolDecorator:
    def test_basic(self):
        @tool
        def search(query: str) -> str:
            """Search the web."""
            return f"Results for: {query}"

        assert search.name == "search"
        assert search.description == "Search the web."
        assert search.invoke({"query": "langchain"}) == "Results for: langchain"

    def test_custom_name_and_description(self):
        @tool(name="my_search", description="Custom search")
        def search(query: str) -> str:
            return f"Results for: {query}"

        assert search.name == "my_search"
        assert search.description == "Custom search"

    def test_args_schema(self):
        @tool
        def calc(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        schema = calc.args_schema
        assert schema["type"] == "object"
        assert "a" in schema["properties"]
        assert "b" in schema["properties"]
        assert schema["properties"]["a"]["type"] == "integer"
        assert set(schema["required"]) == {"a", "b"}

    def test_invoke_with_dict(self):
        @tool
        def add(a: int, b: int) -> int:
            """Add."""
            return a + b

        assert add.invoke({"a": 3, "b": 4}) == 7

    def test_invoke_with_string(self):
        @tool
        def echo(text: str) -> str:
            """Echo."""
            return text

        assert echo.invoke("hello") == "hello"

    def test_to_dict(self):
        @tool
        def search(query: str) -> str:
            """Search the web."""
            return f"Results for: {query}"

        d = search.to_dict()
        assert d["type"] == "function"
        assert d["function"]["name"] == "search"
        assert d["function"]["description"] == "Search the web."
        assert "query" in d["function"]["parameters"]["properties"]

    def test_optional_param(self):
        @tool
        def greet(name: str, greeting: str = "Hello") -> str:
            """Greet someone."""
            return f"{greeting}, {name}!"

        assert greet.invoke({"name": "World"}) == "Hello, World!"
        assert greet.invoke({"name": "World", "greeting": "Hi"}) == "Hi, World!"

        schema = greet.args_schema
        assert "name" in schema["required"]
        assert "greeting" not in schema.get("required", [])

    async def test_ainvoke(self):
        @tool
        def double(n: int) -> int:
            """Double it."""
            return n * 2

        result = await double.ainvoke({"n": 5})
        assert result == 10

    def test_no_docstring(self):
        @tool
        def nodoc(x: str) -> str:
            return x

        assert nodoc.description == ""

    def test_float_type(self):
        @tool
        def scale(factor: float) -> float:
            """Scale."""
            return factor * 2

        assert scale.args_schema["properties"]["factor"]["type"] == "number"

    def test_bool_type(self):
        @tool
        def toggle(flag: bool) -> bool:
            """Toggle."""
            return not flag

        assert toggle.args_schema["properties"]["flag"]["type"] == "boolean"
