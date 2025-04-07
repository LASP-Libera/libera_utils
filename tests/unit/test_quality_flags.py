"""Test coverage for the libera_utils.quality_flags module"""
from enum import NAMED_FLAGS, UNIQUE, verify

import pytest

import libera_utils.quality_flags as qf


def test_quality_flag_verify_unique():
    """
    Test to ensure that all bit masks are named uniquely
    """

    with pytest.raises(ValueError, match="aliases found"):
        @verify(UNIQUE)
        class TestFlag(qf.LiberaFlag):
            BIT_0 = 0b001
            BIT_1 = 0b010
            BIT_2 = 0b100
            BIT_1_AND_2 = 0b110
            ALIAS = 0b100  # NOT unique


def test_quality_flag_verify_named_flags():
    """
    Test to ensure that all individual bits are named in the enum before being referenced in
    combined flag bit masks
    """
    with pytest.raises(ValueError, match="invalid Flag 'TestFlag': alias BIT_1_AND_2 is missing value 0x4"):
        @verify(NAMED_FLAGS)
        class TestFlag(qf.LiberaFlag):
            BIT_0 = 0b001
            BIT_1 = 0b010
            BIT_1_AND_2 = 0b110  # BIT 2 not named


def test_quality_flag_member_immutability():
    """Test that you can't change a member of a quality flag class later (you can still add members though)"""
    @verify(UNIQUE, NAMED_FLAGS)
    class TestFlag(qf.LiberaFlag):
        BIT_0 = 0b001
        BIT_1 = 0b010
        BIT_2 = 0b100
        BIT_1_AND_2 = 0b110

    with pytest.raises(ValueError, match="invalid value 99"):
        TestFlag(99)  # Raises because 99 is out of range (violates STRICT boundary)

    with pytest.raises(AttributeError):
        TestFlag.BIT_0 = 0b111  # Can't reassign element after definition

    TestFlag.FOOBAR = 0b001

    assert TestFlag.BIT_2 | TestFlag.BIT_1 == TestFlag.BIT_1_AND_2


def test_quality_flag():
    """Test our ability to create summary messages from a quality flag"""
    class TestFlag(qf.LiberaFlag):
        A = qf.FlagBit(0b1, message="Bit 0 - A")
        B = qf.FlagBit(0b10, message="Bit 1 - B")
        C = qf.FlagBit(0b100, message="Bit 2 - C")
        D = qf.FlagBit(0b100000000, message="D")

    assert (TestFlag.A | TestFlag.B).decompose() == ([TestFlag.B, TestFlag.A], 0)

    assert TestFlag.A & TestFlag.B == TestFlag(0)

    assert bool(TestFlag(0)) is False

    assert bool(TestFlag.A) is True

    f0 = TestFlag.A | TestFlag.B
    f1 = TestFlag.A | TestFlag.B | TestFlag.C
    assert f1 & f0 == TestFlag.A | TestFlag.B
    assert f1 & TestFlag.A & TestFlag.B == TestFlag(0)

    f2 = TestFlag.A | TestFlag.C
    assert f2 & f0 == TestFlag.A

    assert TestFlag.D.decompose() == ([TestFlag.D], 0)

    f = TestFlag.A | TestFlag.B
    assert set(f.summary[1]) == {"Bit 0 - A", "Bit 1 - B"}

    all_ = ~TestFlag(0)
    assert set(all_.summary[1]) == {"Bit 0 - A", "Bit 1 - B", "Bit 2 - C", "D"}


def test_strict_quality_flags():
    """Test that quality flags are STRICT and raise errors for invalid values"""
    class TestFlag(qf.LiberaFlag):
        BIT_0 = 0b001
        BIT_2 = 0b100
        # Note: there is no way to represent the number 2 or 3 with this quality flag since it is missing bit 1

    TestFlag(1)  # bit 0
    TestFlag(4)  # bit 2
    TestFlag(5)  # bit 0 and 2
    with pytest.raises(ValueError, match="invalid value"):
        TestFlag(2)
    with pytest.raises(ValueError, match="invalid value"):
        TestFlag(3)
    with pytest.raises(ValueError, match="invalid value"):
        TestFlag(6)

