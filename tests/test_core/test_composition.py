"""Tests for Runnable composition primitives: sequence, parallel, passthrough, branch."""

import pytest

from myagent_core import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableSequence,
    coerce_to_runnable,
)


class TestRunnableSequence:
    def test_pipe_operator(self):
        add1 = RunnableLambda(lambda x: x + 1)
        mul2 = RunnableLambda(lambda x: x * 2)
        chain = add1 | mul2
        assert isinstance(chain, RunnableSequence)
        assert chain.invoke(3) == 8  # (3+1)*2

    def test_multi_pipe(self):
        chain = RunnableLambda(lambda x: x + 1) | RunnableLambda(lambda x: x * 2) | RunnableLambda(lambda x: x - 3)
        assert chain.invoke(5) == 9  # (5+1)*2 - 3

    def test_pipe_with_callable(self):
        chain = RunnableLambda(lambda x: x + 1) | (lambda x: x * 10)
        assert chain.invoke(2) == 30

    def test_pipe_with_dict(self):
        add1 = RunnableLambda(lambda x: x + 1)
        chain = add1 | {"doubled": lambda x: x * 2, "tripled": lambda x: x * 3}
        result = chain.invoke(4)
        assert result == {"doubled": 10, "tripled": 15}

    def test_first_middle_last(self):
        a = RunnableLambda(lambda x: x + 1)
        b = RunnableLambda(lambda x: x * 2)
        c = RunnableLambda(lambda x: x - 1)
        chain = a | b | c
        assert isinstance(chain, RunnableSequence)
        assert chain.first is a
        assert chain.middle == [b]
        assert chain.last is c

    def test_pipe_method(self):
        a = RunnableLambda(lambda x: x + 1)
        chain = a.pipe(lambda x: x * 2, lambda x: x + 100)
        assert chain.invoke(3) == 108  # (3+1)*2 + 100

    def test_stream(self):
        chain = RunnableLambda(lambda x: x + 1) | RunnableLambda(lambda x: x * 2)
        chunks = list(chain.stream(5))
        assert chunks == [12]

    async def test_ainvoke(self):
        chain = RunnableLambda(lambda x: x + 1) | RunnableLambda(lambda x: x * 2)
        result = await chain.ainvoke(5)
        assert result == 12

    async def test_astream(self):
        chain = RunnableLambda(lambda x: x + 1) | RunnableLambda(lambda x: x * 2)
        chunks = []
        async for chunk in chain.astream(5):
            chunks.append(chunk)
        assert chunks == [12]

    def test_flatten_nested_sequences(self):
        a = RunnableLambda(lambda x: x + 1)
        b = RunnableLambda(lambda x: x * 2)
        c = RunnableLambda(lambda x: x - 1)
        chain1 = a | b
        chain2 = chain1 | c
        assert isinstance(chain2, RunnableSequence)
        assert len(chain2.steps) == 3

    def test_ror_with_callable(self):
        """Test reverse pipe: callable | Runnable."""
        r = RunnableLambda(lambda x: x * 2)
        chain = (lambda x: x + 1) | r
        assert chain.invoke(3) == 8  # (3+1)*2


class TestRunnableParallel:
    def test_basic(self):
        par = RunnableParallel(
            added=RunnableLambda(lambda x: x + 1),
            doubled=RunnableLambda(lambda x: x * 2),
        )
        result = par.invoke(5)
        assert result == {"added": 6, "doubled": 10}

    def test_with_dict_arg(self):
        par = RunnableParallel(steps={"a": lambda x: x + 1, "b": lambda x: x * 2})
        result = par.invoke(3)
        assert result == {"a": 4, "b": 6}

    def test_with_callables(self):
        par = RunnableParallel(upper=lambda x: x.upper(), lower=lambda x: x.lower())
        result = par.invoke("Hello")
        assert result == {"upper": "HELLO", "lower": "hello"}

    async def test_ainvoke(self):
        par = RunnableParallel(a=lambda x: x + 1, b=lambda x: x * 2)
        result = await par.ainvoke(5)
        assert result == {"a": 6, "b": 10}

    def test_stream(self):
        par = RunnableParallel(a=lambda x: x + 1)
        chunks = list(par.stream(5))
        assert chunks == [{"a": 6}]

    async def test_astream(self):
        par = RunnableParallel(a=lambda x: x + 1)
        chunks = []
        async for chunk in par.astream(5):
            chunks.append(chunk)
        assert chunks == [{"a": 6}]


class TestRunnablePassthrough:
    def test_basic(self):
        pt = RunnablePassthrough()
        assert pt.invoke("hello") == "hello"
        assert pt.invoke(42) == 42
        assert pt.invoke({"key": "val"}) == {"key": "val"}

    async def test_ainvoke(self):
        pt = RunnablePassthrough()
        assert await pt.ainvoke("async") == "async"

    def test_assign(self):
        assign = RunnablePassthrough.assign(
            upper=lambda x: x["text"].upper(),
            length=lambda x: len(x["text"]),
        )
        result = assign.invoke({"text": "hello"})
        assert result == {"text": "hello", "upper": "HELLO", "length": 5}

    async def test_assign_ainvoke(self):
        assign = RunnablePassthrough.assign(doubled=lambda x: x["val"] * 2)
        result = await assign.ainvoke({"val": 5})
        assert result == {"val": 5, "doubled": 10}

    def test_assign_in_chain(self):
        chain = RunnablePassthrough.assign(upper=lambda x: x["text"].upper()) | (lambda x: x["upper"])
        assert chain.invoke({"text": "hello"}) == "HELLO"


class TestRunnableBranch:
    def test_basic_routing(self):
        branch = RunnableBranch(
            (lambda x: x > 0, lambda x: f"positive: {x}"),
            (lambda x: x < 0, lambda x: f"negative: {x}"),
            lambda x: f"zero: {x}",
        )
        assert branch.invoke(5) == "positive: 5"
        assert branch.invoke(-3) == "negative: -3"
        assert branch.invoke(0) == "zero: 0"

    async def test_ainvoke(self):
        branch = RunnableBranch(
            (lambda x: x > 0, lambda x: "pos"),
            lambda x: "non-pos",
        )
        assert await branch.ainvoke(1) == "pos"
        assert await branch.ainvoke(-1) == "non-pos"

    def test_requires_default(self):
        with pytest.raises(ValueError):
            RunnableBranch((lambda x: True, lambda x: x))

    def test_invalid_branch_tuple(self):
        with pytest.raises(ValueError, match="Expected"):
            RunnableBranch("not_a_tuple", lambda x: x)

    def test_with_runnable_branches(self):
        branch = RunnableBranch(
            (lambda x: x["type"] == "upper", RunnableLambda(lambda x: x["text"].upper())),
            RunnableLambda(lambda x: x["text"].lower()),
        )
        assert branch.invoke({"type": "upper", "text": "Hello"}) == "HELLO"
        assert branch.invoke({"type": "lower", "text": "Hello"}) == "hello"


class TestCoerceToRunnable:
    def test_runnable_passthrough(self):
        r = RunnableLambda(lambda x: x)
        assert coerce_to_runnable(r) is r

    def test_callable(self):
        r = coerce_to_runnable(lambda x: x + 1)
        assert r.invoke(5) == 6

    def test_dict(self):
        r = coerce_to_runnable({"a": lambda x: x + 1})
        assert isinstance(r, RunnableParallel)
        assert r.invoke(5) == {"a": 6}

    def test_invalid(self):
        with pytest.raises(TypeError):
            coerce_to_runnable(42)
