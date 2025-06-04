"""Tests for time module"""

from collections.abc import Iterable
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from libera_utils import time


# TODO: Prevent these tests from retrieving their LSK from NAIF by
#  providing a mocked metakernel and asserting that it gets furnished by checking numbers of calls
@pytest.mark.parametrize(
    ("et", "expected"),
    [
        (0, datetime.strptime("2000-01-01T11:58:55.816073", "%Y-%m-%dT%H:%M:%S.%f")),
        (378651667, datetime.strptime("2012-01-01T01:00:00.816089", "%Y-%m-%dT%H:%M:%S.%f")),
        (
            [378651667, 378651668],
            [
                datetime.strptime("2012-01-01T01:00:00.816089", "%Y-%m-%dT%H:%M:%S.%f"),
                datetime.strptime("2012-01-01T01:00:01.816089", "%Y-%m-%dT%H:%M:%S.%f"),
            ],
        ),
    ],
)
def test_et_2_datetime(et, expected):
    """Test et_2_datetime wrapper function"""
    assert np.array_equal(time.et_2_datetime(et), expected)


@pytest.mark.parametrize(
    ("et", "fmt", "expected"),
    [
        (0, "%Y%m%dt%H%M%S", "20000101t115855"),
        (378651667, "%Y%m%dt%H%M%S.%f", "20120101t010000.816089"),
        ([378651667, 378651668], "%Y/%j %b %d %H:%M:%S", ["2012/001 Jan 01 01:00:00", "2012/001 Jan 01 01:00:01"]),
    ],
)
def test_et_2_timestamp(et, fmt, expected):
    """Test et_2_timestamp wrapper function"""
    result = time.et_2_timestamp(et, fmt)
    assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    ("sclk_str", "expected"),
    [
        ("15340:43200000:000", 0.000000),
        ("23109:50398930:938", 671248798.930938),
        (["23109:50398930:938", "23109:50398930:938"], np.array([671248798.930938, 671248798.930938])),
        (np.array(["23109:50398930:938", "23109:50398930:938"]), np.array([671248798.930938, 671248798.930938])),
    ],
)
def test_scs2e_wrapper(sclk_str, expected):
    """Test conversion of SCLK strings to ET"""
    result = time.scs2e_wrapper(sclk_str)
    if isinstance(sclk_str, Iterable):
        assert np.array_equal(result, expected)
    else:
        assert result == expected


@pytest.mark.parametrize(
    ("et", "expected"),
    [
        (0.000000, "1/15340:43200000:000"),  # Ephemeris epoch in SCLK time
        (671248798.9309382, "1/23109:50398930:938"),
        ([671248798.9309382, 671248798.9309382], np.array(["1/23109:50398930:938", "1/23109:50398930:938"])),
        (np.array([671248798.9309382, 671248798.9309382]), np.array(["1/23109:50398930:938", "1/23109:50398930:938"])),
    ],
)
def test_sce2s_wrapper(et, expected):
    """Test conversion of ET to SCLK strings"""
    result = time.sce2s_wrapper(et)
    if isinstance(et, Iterable):
        assert np.array_equal(result, expected)
    else:
        assert result == expected


@pytest.mark.parametrize(
    ("et", "expected"),
    [
        (671248798.9309382, "2021-04-09T13:58:49.745289445"),
        (
            [671248798.9309382, 671248798.9309382],
            np.array(["2021-04-09T13:58:49.745289445", "2021-04-09T13:58:49.745289445"]),
        ),
        (
            np.array([671248798.9309382, 671248798.9309382]),
            np.array(["2021-04-09T13:58:49.745289445", "2021-04-09T13:58:49.745289445"]),
        ),
    ],
)
def test_et2utc_wrapper(et, expected):
    """Test conversion of ET to UTC strings"""
    prec = 9
    result = time.et2utc_wrapper(et, "ISOC", prec)
    if isinstance(et, Iterable):
        assert np.array_equal(result, expected)
    else:
        assert result == expected


@pytest.mark.parametrize(
    ("utc_str", "expected"),
    [
        ("2021-04-09T13:58:49.745289445", 671248798.9309382),
        (
            ["2021-04-09T13:58:49.745289445", "2021-04-09T13:58:49.745289445"],
            np.array([671248798.9309382, 671248798.9309382]),
        ),
        (
            np.array(["2021-04-09T13:58:49.745289445", "2021-04-09T13:58:49.745289445"]),
            np.array([671248798.9309382, 671248798.9309382]),
        ),
    ],
)
def test_utc2et_wrapper(utc_str, expected):
    """Test conversion of ET to UTC strings"""
    result = time.utc2et_wrapper(utc_str)
    if isinstance(utc_str, Iterable):
        assert np.array_equal(result, expected)
    else:
        assert result == expected


@pytest.mark.parametrize(
    ("data", "epoch", "expected"),
    [
        ([(0, 0, 0)], None, ["1958-01-01"]),
        (
            [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
            None,
            ["1958-01-02T00:00:00.000000", "1958-01-01T00:00:00.001000", "1958-01-01T00:00:00.000001"],
        ),
        ([(23109, 43206030, 922)], None, ["2021-04-09 12:00:06.030922"]),
        ([(0, 0, 0)], "2000-01-01", ["2000-01-01"]),
    ],
)
def test_multipart_to_dt64(data, epoch, expected):
    """Test multipart_to_dt64 wrapper function"""
    data = pd.DataFrame(data, columns=["a", "b", "c"])
    result = time.multipart_to_dt64(data, "a", "b", "c", **({} if epoch is None else dict(epoch=epoch)))
    assert np.array_equal(result, pd.to_datetime(expected))
