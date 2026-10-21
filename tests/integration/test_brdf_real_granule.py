"""Integration tests for VIIRSBRDFReader on a real VJ143C1 granule.

Runs only when a granule is staged under the repository's
``external_data/VIIRS/VJ143C1/`` (untracked; ``external_data/`` is gitignored). A
granule is about 360 MB and loads about 0.9 GB. Only the global grid is checked here;
values, fill and box queries run in the unit tests on a cut of a real granule
(``TestVIIRSBRDFReaderRealSubset``).

Real VJ143C1 files can be downloaded from:
    LP DAAC: https://e4ftl01.cr.usgs.gov/VIIRS/VJ143C1.002/
    EarthData login required: https://urs.earthdata.nasa.gov/
"""

from __future__ import annotations

from pathlib import Path

import pytest

from libera_utils.footprint_matching.readers.brdf import VIIRSBRDFReader

pytestmark = pytest.mark.integration

_STAGE_ROOT = Path(__file__).resolve().parents[2] / "external_data" / "VIIRS" / "VJ143C1"
_GRANULES = sorted(_STAGE_ROOT.glob("VJ143C1.*.h5")) if _STAGE_ROOT.is_dir() else []


@pytest.fixture(scope="module")
def native():
    """The first staged granule, loaded once for the module (about 0.9 GB)."""
    return VIIRSBRDFReader(_GRANULES[0])._native


@pytest.mark.skipif(not _GRANULES, reason="no real VJ143C1 granule staged under external_data/VIIRS/VJ143C1/")
class TestVIIRSBRDFReaderRealGranule:
    def test_axes_are_the_global_cmg_cell_centres(self, native):
        assert native.lat_axis.size == 3600
        assert native.lon_axis.size == 7200
        assert native.lat_axis[0] == pytest.approx(-89.975)
        assert native.lat_axis[-1] == pytest.approx(89.975)
        assert native.lon_axis[0] == pytest.approx(-179.975)
        assert native.lon_axis[-1] == pytest.approx(179.975)
