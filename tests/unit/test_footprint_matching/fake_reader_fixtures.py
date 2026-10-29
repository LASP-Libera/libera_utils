"""Fake readers, a fake footprint and the readers dict over them, shared by the product tests.

The fake readers register themselves in ``ReaderRegistry`` when this module is imported, so they are
defined once here and imported by every test module that needs them. They serve *uniform* grids, so
an aggregated value is independent of the PSF weight kernel and can be asserted exactly:

* a weighted mean of a constant field is that constant (weights cancel);
* a weighted mode of a single-class field is that class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from libera_utils.footprint_matching.readers.base import AncillaryReader, StaticReader
from libera_utils.footprint_matching.types import (
    BoundingBox,
    GridSamples,
    OperationalMode,
    TimeMatch,
    VariableSpec,
)

# A small uniform grid served by every fake reader, sitting inside the footprint
# bounding boxes the tests use so the weighers give its samples positive weight. Its
# 0.2 degree spacing (~22 km) is within the 25 km native cell the fakes declare, so the
# grid counts as complete coverage.
GRID_LATS = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float64)
GRID_LONS = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float64)

# The footprint time the fake footprints carry; the fake readers are DAILY and ignore it.
FAKE_TIME = np.datetime64("2026-01-01T00:00:00", "ns")

# Every product carries the fake variables, so the fakes serve whichever mode a test runs.
_ALL_MODES: frozenset[OperationalMode] = frozenset(OperationalMode)


class FakeMeanReader(StaticReader):
    """Multi-variable reader: two continuous fields with constant values 5 and 2."""

    READER_KEY = "_fake_agg_mean"
    INSTRUMENT = "TEST"
    RESOLUTION_KM = 25.0
    TIME_MATCH = TimeMatch.DAILY
    VARIABLES = (
        VariableSpec(name="field_a", dtype="float32", aggregation="weighted_mean", modes=_ALL_MODES),
        VariableSpec(name="field_b", dtype="float32", aggregation="weighted_mean", modes=_ALL_MODES),
    )

    def _load_native(self) -> GridSamples:
        # (n_var, n_lat, n_lon): plane 0 == 5.0 everywhere, plane 1 == 2.0.
        data = np.stack(
            [
                np.full((GRID_LATS.size, GRID_LONS.size), 5.0, dtype=np.float32),
                np.full((GRID_LATS.size, GRID_LONS.size), 2.0, dtype=np.float32),
            ]
        )
        return GridSamples(lat_axis=GRID_LATS, lon_axis=GRID_LONS, data=data)


class FakeCatReader(StaticReader):
    """Single-variable categorical reader: every sample is class 3, plus ranked extras."""

    READER_KEY = "_fake_agg_cat"
    INSTRUMENT = "TEST"
    RESOLUTION_KM = 25.0
    TIME_MATCH = TimeMatch.DAILY
    VARIABLES = (
        VariableSpec(
            name="surface",
            dtype="int16",
            aggregation="weighted_mode",
            modes=_ALL_MODES,
            n_categories=6,
        ),
    )
    ADDITIONAL_PRODUCT_VARIABLES = (
        VariableSpec(
            name="surface_primary",
            dtype="int16",
            aggregation="weighted_mode_primary",
            modes=_ALL_MODES,
            n_categories=6,
        ),
        VariableSpec(
            name="surface_secondary",
            dtype="int16",
            aggregation="weighted_mode_secondary",
            modes=_ALL_MODES,
            n_categories=6,
        ),
    )

    def _load_native(self) -> GridSamples:
        data = np.full((1, GRID_LATS.size, GRID_LONS.size), 3.0, dtype=np.float32)
        return GridSamples(lat_axis=GRID_LATS, lon_axis=GRID_LONS, data=data)


@dataclass
class FakeFootprint:
    """Minimal footprint exposing what the weigher and the readers need."""

    bbox: BoundingBox | None
    latitude: float
    longitude: float
    # Optional geometry consumed by the angular-frame weigher (ignored by RadialWeigher).
    subsatellite_latitude: float | None = None
    subsatellite_longitude: float | None = None
    satellite_radius_km: float | None = None
    cone_angle_rate: float | None = None
    viewing_zenith_angle: float | None = 0.0
    off_limb: bool = False
    time: np.datetime64 = field(default_factory=lambda: FAKE_TIME)


def make_sources(tmp_path: Path) -> dict[str, AncillaryReader]:
    """Return the two fake readers keyed by registry key, each over one staged (empty) granule."""
    mean_path, cat_path = tmp_path / "mean.bin", tmp_path / "cat.bin"
    mean_path.touch()
    cat_path.touch()
    return {
        FakeMeanReader.READER_KEY: FakeMeanReader([mean_path]),
        FakeCatReader.READER_KEY: FakeCatReader([cat_path]),
    }
