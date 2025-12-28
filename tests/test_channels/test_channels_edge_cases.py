"""Edge case tests for channels."""

import operator

import pytest
from myagent.channels.binop import BinaryOperatorAggregate
from myagent.channels.ephemeral import EphemeralValue
from myagent.channels.last_value import LastValue
from myagent.errors import EmptyChannelError, InvalidUpdateError


class TestLastValueEdgeCases:
    def test_overwrite(self):
        ch = LastValue(str)
        ch.update(["first"])
        ch.update(["second"])
        assert ch.get() == "second"

    def test_none_value(self):
        ch = LastValue(type(None))
        ch.update([None])
        assert ch.get() is None

    def test_complex_value(self):
        ch = LastValue(dict)
        ch.update([{"nested": [1, 2, 3]}])
        assert ch.get() == {"nested": [1, 2, 3]}

    def test_checkpoint_empty_channel(self):
        ch = LastValue(int)
        # Checkpoint of empty channel should not crash
        cp = ch.checkpoint()
        new_ch = ch.from_checkpoint(cp)
        with pytest.raises(EmptyChannelError):
            new_ch.get()


class TestBinaryOperatorAggregateEdgeCases:
    def test_string_concat(self):
        ch = BinaryOperatorAggregate(str, operator.add)
        ch.update(["hello"])
        ch.update([" world"])
        assert ch.get() == "hello world"

    def test_int_addition(self):
        ch = BinaryOperatorAggregate(int, operator.add)
        ch.update([1])
        ch.update([2])
        ch.update([3])
        assert ch.get() == 6

    def test_empty_update_no_change(self):
        ch = BinaryOperatorAggregate(int, operator.add)
        ch.update([10])
        changed = ch.update([])
        assert not changed
        assert ch.get() == 10

    def test_checkpoint_preserves_accumulated(self):
        ch = BinaryOperatorAggregate(int, operator.add)
        ch.update([1])
        ch.update([2])
        cp = ch.checkpoint()
        new_ch = ch.from_checkpoint(cp)
        assert new_ch.get() == 3
        new_ch.update([4])
        assert new_ch.get() == 7


class TestEphemeralValueEdgeCases:
    def test_guard_rejects_multiple(self):
        ch = EphemeralValue(guard=True)
        with pytest.raises(InvalidUpdateError):
            ch.update([1, 2])

    def test_no_guard_takes_last(self):
        ch = EphemeralValue(guard=False)
        ch.update([1, 2, 3])
        assert ch.get() == 3

    def test_repeated_clear(self):
        ch = EphemeralValue()
        ch.update(["val"])
        ch.clear()
        ch.clear()  # double clear should be fine
        with pytest.raises(EmptyChannelError):
            ch.get()
