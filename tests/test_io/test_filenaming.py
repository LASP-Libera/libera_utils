"""Tests for filenaming module"""
import pytest

from libera_sdp.io import filenaming


@pytest.mark.parametrize(
    ('isot', 'expected'),
    [
        ('2027-05-06T11:22:33.456789', '20270506t112233'),
        ('2027-05-06T11:22:33.4567891234567890123456789', '20270506t112233'),
        ('2027-05-06t11:22:33', '20270506t112233'),
        ('2027-05-06t11:22:33.456789', '20270506t112233'),
    ]
)
def test_isot_printable(isot, expected):
    """Test creation of filename-compatible timestamps from iso strings"""
    result = filenaming.isot_printable(isot)
    assert result == expected
