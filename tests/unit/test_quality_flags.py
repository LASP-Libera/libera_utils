"""Test coverage for the libera_utils.quality_flags module"""
# Installed
import pytest
# Local
import libera_utils.quality_flags as qf


def test_quality_flag_metaclass():
    """
    Test to ensure that the QualityFlagMeta metaclass successfully
    prevents adding entries to a class created with the metaclass
    """
    class TestFlag(qf.QualityFlag, metaclass=qf.FrozenFlagMeta):
        BIT_0 = 0b001
        BIT_1 = 0b010
        BIT_2 = 0b100
        BIT_1_AND_2 = 0b110

    with pytest.raises(AttributeError):
        TestFlag.FOOBAR = 0b111  # Can't add a new element after definition

    with pytest.raises(AttributeError):
        TestFlag.BIT_0 = 99  # Can't reassign element after definition

    assert TestFlag.BIT_2 | TestFlag.BIT_1 == TestFlag.BIT_1_AND_2


def test_quality_flag():
    """Test our ability to create summary messages from a quality flag"""
    @qf.with_all_none
    class TestFlag(qf.QualityFlag, metaclass=qf.FrozenFlagMeta):
        A = qf.FlagBit(0b1, message="Bit 0 - A")
        B = qf.FlagBit(0b10, message="Bit 1 - B")
        C = qf.FlagBit(0b100, message="Bit 2 - C")
        D = qf.FlagBit(0b100000000, message="D")

    assert (TestFlag.A | TestFlag.B).decompose() == ([TestFlag.B, TestFlag.A], 0)

    assert TestFlag.A & TestFlag.B == TestFlag.NONE

    assert bool(TestFlag.NONE) is False

    assert bool(TestFlag.A) is True

    f0 = TestFlag.A | TestFlag.B
    f1 = TestFlag.A | TestFlag.B | TestFlag.C
    assert f1 & f0 == TestFlag.A | TestFlag.B
    assert f1 & TestFlag.A & TestFlag.B == TestFlag.NONE

    f2 = TestFlag.A | TestFlag.C
    assert f2 & f0 == TestFlag.A

    assert TestFlag.D.decompose() == ([TestFlag.D], 0)

    f = TestFlag.A | TestFlag.B
    assert set(f.summary[1]) == {"Bit 0 - A", "Bit 1 - B"}

    all = TestFlag.ALL
    assert set(all.summary[1]) == {"Bit 0 - A", "Bit 1 - B", "Bit 2 - C", "D"}


def test_strict_quality_flags():
    """Test that quality flags are STRICT and raise errors for invalid values"""
    class TestFlag(qf.QualityFlag, metaclass=qf.FrozenFlagMeta):
        BIT_0 = 0b001
        BIT_2 = 0b100
        # Note: there is no way to represent the number 2 or 3 with this quality flag since it is missing bit 1

    TestFlag(1)  # bit 0
    TestFlag(4)  # bit 2
    TestFlag(5)  # bit 0 and 2
    with pytest.raises(ValueError):
        TestFlag(2)
    with pytest.raises(ValueError):
        TestFlag(3)
    with pytest.raises(ValueError):
        TestFlag(6)

