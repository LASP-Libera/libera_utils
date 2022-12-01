"""Test coverage for the emmemuspy.utils.quality_flags module"""

import pytest
import libera_utils.quality_flags as qf
from enum import Flag


def test_quality_flag_metaclass():
    """
    Test to ensure that the QualityFlagMeta metaclass successfully
    prevents adding entries to a class created with the metaclass
    """
    class TestFlag(Flag, metaclass=qf.FrozenFlagMeta):
        BIT_0 = 0b00001
        BIT_1 = 0b00010
        BIT_2 = 0b00100
        BIT_3 = 0b01000
        BIT_4 = 0b10000
        BIT_1_AND_2 = 0b00110
        BIT_3_AND_4 = 0b11000

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


# def test_quality_flag_decompose_patch():
#     """Test patches made to the enum._decompose method and relevant Flag methods"""
#     x = qf.L1ImageQualityFlag(131073)
#     print(str(x))


def test_L1QualityFlag():
    """Test behavior of the L1QualityFlag class"""
    for f in qf.L1QualityFlag:
        assert f.value
        assert f.value.message
        assert f.summary

    assert len(qf.L1QualityFlag.ALL.summary[1]) == 5


def test_L1ImageQualityFlag():
    for f in qf.L1ImageQualityFlag:
        assert f.value
        assert f.value.message
        assert f.summary

    assert len(qf.L1ImageQualityFlag.ALL.summary[1]) == 13


def test_WavelengthHduQualityFlag():
    for f in qf.WavelengthHduQualityFlag:
        assert f.value
        assert f.value.message
        assert f.summary

    assert len(qf.WavelengthHduQualityFlag.ALL.summary[1]) == 5