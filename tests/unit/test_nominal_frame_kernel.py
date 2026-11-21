"""Tests for ``write_nominal_frame_kernel`` in ``tests/plugins/spice_fixtures.py``.

The misalignment A/B geolocation test depends on this copy of the production frame kernel holding
the ideal axes and boresight and nothing else changed. A layout change in the production kernel
must stop the copy being made rather than leave a measured value in place.
"""

import re
from pathlib import Path

import pytest

from libera_utils.config import config
from tests.plugins.spice_fixtures import _NOMINAL_FRAME_VALUES, write_nominal_frame_kernel


def test_the_production_frame_kernel_gets_every_ideal_value(tmp_path):
    """Each misalignment keyword in the production kernel is replaced by its ideal value, once."""
    source = Path(config.get("LIBERA_KERNEL_FRAME"))
    nominal = write_nominal_frame_kernel(source, tmp_path / "nominal.tf").read_text()

    for keyword, ideal in _NOMINAL_FRAME_VALUES.items():
        definitions = re.findall(rf"^\s*{re.escape(keyword)}\s*=\s*(\([^)]*\))", nominal, flags=re.MULTILINE)
        assert definitions == [ideal], keyword
    assert len(nominal.splitlines()) == len(source.read_text().splitlines())


@pytest.mark.parametrize(("copies", "expected_count"), [(0, 0), (2, 2)], ids=["missing", "duplicated"])
def test_a_keyword_defined_other_than_once_raises(tmp_path, copies, expected_count):
    """A keyword missing from, or repeated in, the frame kernel stops the copy being written."""
    source_text = Path(config.get("LIBERA_KERNEL_FRAME")).read_text()
    keyword = "LIBERA_EL_AOR_IN_STAND"
    (definition,) = re.findall(rf"^\s*{re.escape(keyword)}\s*=\s*\([^)]*\)\n", source_text, flags=re.MULTILINE)
    altered = tmp_path / "altered.tf"
    altered.write_text(source_text.replace(definition, definition * copies))
    destination = tmp_path / "nominal.tf"

    with pytest.raises(AssertionError, match=f"{keyword} found {expected_count} times"):
        write_nominal_frame_kernel(altered, destination)
    assert not destination.exists()
