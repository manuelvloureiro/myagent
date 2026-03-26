"""Tests for myagent_core runnable primitives."""

import pytest
from myagent_core.runnable.config import (
    RunnableConfig,
    ensure_config,
    merge_configs,
)
from myagent_core.runnable.lambda_ import RunnableLambda
from myagent_core.serde import JsonPlusSerializer


class TestRunnableConfig:
    def test_ensure_config_defaults(self):
        config = ensure_config()
        assert config.get("tags") == []
        assert config.get("metadata") == {}
        assert config.get("recursion_limit") == 25
        assert config.get("run_id")  # non-empty

    def test_ensure_config_preserves(self):
        config = ensure_config({"tags": ["a"], "recursion_limit": 50})
        assert config.get("tags") == ["a"]
        assert config.get("recursion_limit") == 50

    def test_merge_configs(self):
        c1: RunnableConfig = {"tags": ["a"], "metadata": {"k": "v1"}}
        c2: RunnableConfig = {"tags": ["b"], "metadata": {"k2": "v2"}}
        merged = merge_configs(c1, c2)
        assert "a" in merged.get("tags", [])
        assert "b" in merged.get("tags", [])
        assert merged.get("metadata", {})["k"] == "v1"
        assert merged.get("metadata", {})["k2"] == "v2"

    def test_merge_configs_none(self):
        merged = merge_configs(None, {"tags": ["x"]})
        assert "x" in merged.get("tags", [])


class TestRunnableLambda:
    def test_basic_invoke(self):
        r = RunnableLambda(lambda x: x * 2)
        assert r.invoke(5) == 10

    def test_invoke_with_config(self):
        def fn(x, config):
            return x + len(config.get("tags", []))

        r = RunnableLambda(fn)
        result = r.invoke(10, {"tags": ["a", "b"]})
        assert result == 12

    def test_name_inferred(self):
        def my_func(x):
            return x

        r = RunnableLambda(my_func)
        assert r.name == "my_func"

    def test_stream(self):
        r = RunnableLambda(lambda x: x + 1)
        chunks = list(r.stream(5))
        assert chunks == [6]

    def test_batch(self):
        r = RunnableLambda(lambda x: x * 2)
        results = r.batch([1, 2, 3])
        assert results == [2, 4, 6]

    def test_batch_accepts_config_sequence(self):
        def fn(x, config):
            return x + config.get("metadata", {}).get("offset", 0)

        r = RunnableLambda(fn)
        results = r.batch([1, 2], ({"metadata": {"offset": 10}}, {"metadata": {"offset": 20}}))
        assert results == [11, 22]

    @pytest.mark.asyncio
    async def test_ainvoke(self):
        r = RunnableLambda(lambda x: x * 3)
        result = await r.ainvoke(4)
        assert result == 12

    @pytest.mark.asyncio
    async def test_ainvoke_with_async_func(self):
        async def afn(x):
            return x * 5

        r = RunnableLambda(lambda x: x, afunc=afn)
        result = await r.ainvoke(3)
        assert result == 15

    @pytest.mark.asyncio
    async def test_astream(self):
        r = RunnableLambda(lambda x: x + 10)
        chunks = []
        async for chunk in r.astream(5):
            chunks.append(chunk)
        assert chunks == [15]

    @pytest.mark.asyncio
    async def test_abatch(self):
        r = RunnableLambda(lambda x: x * 2)
        results = await r.abatch([1, 2, 3])
        assert results == [2, 4, 6]


class TestJsonPlusSerializer:
    def setup_method(self):
        self.serde = JsonPlusSerializer()

    def test_basic_roundtrip(self):
        data = {"key": "value", "num": 42}
        assert self.serde.loads(self.serde.dumps(data)) == data

    def test_datetime(self):
        from datetime import datetime, timezone

        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        result = self.serde.loads(self.serde.dumps(dt))
        assert result == dt

    def test_uuid(self):
        import uuid

        u = uuid.uuid4()
        result = self.serde.loads(self.serde.dumps(u))
        assert result == u

    def test_decimal(self):
        from decimal import Decimal

        d = Decimal("3.14")
        result = self.serde.loads(self.serde.dumps(d))
        assert result == d

    def test_bytes(self):
        b = b"hello"
        result = self.serde.loads(self.serde.dumps(b))
        assert result == b

    def test_set(self):
        s = {1, 2, 3}
        result = self.serde.loads(self.serde.dumps(s))
        assert result == s
