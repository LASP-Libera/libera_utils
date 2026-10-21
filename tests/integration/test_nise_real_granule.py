"""Integration tests for NISEReader on a real NISE granule.

Runs only when a granule is staged under the repository's ``external_data/NSIDC/``
(untracked; ``external_data/`` is gitignored). The operational product is NISE_A2
(AMSR-2); the SSM/I-SSMIS granule is a format-identical fallback.

Real NISE files can be downloaded from:
    NSIDC HTTPS: https://n5eil01u.ecs.nsidc.org/NISE/
    Earthdata login required: https://urs.earthdata.nasa.gov/
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from libera_utils.footprint_matching.readers.nsidc import _HEMISPHERE_LABEL_NORTH, _HEMISPHERE_LABEL_SOUTH, NISEReader
from libera_utils.footprint_matching.types import BoundingBox

pytestmark = pytest.mark.integration

_T = np.datetime64("2026-01-01T00:00:00", "ns")
_REPO_ROOT = Path(__file__).resolve().parents[2]
_NISE_STAGE_ROOTS = (_REPO_ROOT / "external_data" / "NSIDC", _REPO_ROOT / "external_data" / "external_data" / "NSIDC")
_NISE_GRANULE_NAMES = ("NISE_AMSR2_20260610.HDFEOS", "NISE_SSMISF18_20260111.HDFEOS")


def _find_real_nise() -> Path:
    """First staged NISE granule found (AMSR-2 preferred), or a non-existent sentinel."""
    for name in _NISE_GRANULE_NAMES:
        for root in _NISE_STAGE_ROOTS:
            candidate = root / name
            if candidate.exists():
                return candidate
    return _NISE_STAGE_ROOTS[0] / _NISE_GRANULE_NAMES[0]


_REAL_NISE = _find_real_nise()


@pytest.mark.skipif(not _REAL_NISE.exists(), reason="real NISE granule not staged under external_data/NSIDC/")
class TestNISEReaderRealGranule:
    """End-to-end checks against a real staged NISE granule (both hemispheres flow through)."""

    def test_reads_both_extent_sds_by_hemisphere(self):
        reader = NISEReader(_REAL_NISE)
        north = reader._read_extent_sds(_HEMISPHERE_LABEL_NORTH)
        south = reader._read_extent_sds(_HEMISPHERE_LABEL_SOUTH)
        assert north.shape == south.shape == (721, 721)
        # Two genuinely different grids, not the same SDS resolved twice.
        assert not np.array_equal(north, south)

    def test_northern_and_southern_boxes_both_covered(self):
        reader = NISEReader(_REAL_NISE)
        north_box = reader.samples_in(BoundingBox(70.0, 80.0, -10.0, 10.0), _T).values
        south_box = reader.samples_in(BoundingBox(-80.0, -70.0, -10.0, 10.0), _T).values
        assert np.isfinite(north_box).any()
        assert np.isfinite(south_box).any()

    def test_point_cloud_spans_both_hemispheres(self):
        lats = NISEReader(_REAL_NISE)._load_native().lats
        finite = lats[np.isfinite(lats)]
        assert finite.max() > 60.0
        assert finite.min() < -60.0
