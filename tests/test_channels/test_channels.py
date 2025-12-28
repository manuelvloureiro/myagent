"""Tests for channel implementations."""

import operator

import pytest
from myagent.channels.binop import BinaryOperatorAggregate
from myagent.channels.ephemeral import EphemeralValue
from myagent.channels.last_value import LastValue
from myagent.errors import EmptyChannelError, InvalidUpdateError


class TestLastValue:
    def test_empty_raises(self):
        ch = LastValue(int)
        with pytest.raises(EmptyChannelError):
            ch.get()

    def test_update_single(self):
        ch = LastValue(int)
        changed = ch.update([42])
        assert changed
        assert ch.get() == 42

    def test_update_multiple_raises(self):
        ch = LastValue(int)
        with pytest.raises(InvalidUpdateError):
            ch.update([1, 2])

    def test_update_empty(self):
        ch = LastValue(int)
        changed = ch.update([])
        assert not changed

    def test_checkpoint_roundtrip(self):
        ch = LastValue(str)
        ch.update(["hello"])
        cp = ch.checkpoint()
        new_ch = ch.from_checkpoint(cp)
        assert new_ch.get() == "hello"

    def test_from_checkpoint_none(self):
        ch = LastValue(int)
        new_ch = ch.from_checkpoint(None)
        with pytest.raises(EmptyChannelError):
            new_ch.get()


class TestBinaryOperatorAggregate:
    def test_add_reducer(self):
        ch = BinaryOperatorAggregate(list, operator.add)
        ch.update([[1, 2]])
        ch.update([[3]])
        assert ch.get() == [1, 2, 3]

    def test_empty_raises(self):
        ch = BinaryOperatorAggregate(int, operator.add)
        with pytest.raises(EmptyChannelError):
            ch.get()

    def test_multiple_updates_in_one_call(self):
        ch = BinaryOperatorAggregate(int, operator.add)
        ch.update([1, 2, 3])
        assert ch.get() == 6

    def test_checkpoint_roundtrip(self):
        ch = BinaryOperatorAggregate(list, operator.add)
        ch.update([[1, 2]])
        cp = ch.checkpoint()
        new_ch = ch.from_checkpoint(cp)
        assert new_ch.get() == [1, 2]
        new_ch.update([[3]])
        assert new_ch.get() == [1, 2, 3]


class TestEphemeralValue:
    def test_empty_raises(self):
        ch = EphemeralValue()
        with pytest.raises(EmptyChannelError):
            ch.get()

    def test_update_and_clear(self):
        ch = EphemeralValue()
        ch.update(["val"])
        assert ch.get() == "val"
        ch.clear()
        with pytest.raises(EmptyChannelError):
            ch.get()

    def test_from_checkpoint_always_empty(self):
        ch = EphemeralValue()
        ch.update(["val"])
        new_ch = ch.from_checkpoint(None)
        with pytest.raises(EmptyChannelError):
            new_ch.get()
