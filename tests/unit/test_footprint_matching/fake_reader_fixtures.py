"""Fake gridded readers, a fake footprint and a TileManager over them, shared by the product tests.

The fake readers register themselves in ``ReaderRegistry`` when this module is imported, so they are
defined once here and imported by every test module that needs them. They return *uniform* grids, so
an aggregated value is independent of the PSF weight kernel and can be asserted exactly:

* a weighted mean of a constant field is that constant (weights cancel);
* a weighted mode of a single-class field is that class.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from libera_utils.footprint_matching.readers.base import GriddedDataReader
from libera_utils.footprint_matching.tiling import TileManager
from libera_utils.footprint_matching.types import BoundingBox, OperationalMode, VariableSpec

# A small uniform grid returned by every fake reader, sitting inside the footprint
# bounding boxes the tests use so the weighers give its cells positive weight.
GRID_LATS = np.array([0.3, 0.5, 0.7], dtype=np.float64)
GRID_LONS = np.array([0.3, 0.5, 0.7], dtype=np.float64)


class FakeMeanReader(GriddedDataReader):
    """Multi-variable reader: two continuous fields with constant values 5 and 2."""

    READER_KEY = "_fake_agg_mean"
    INSTRUMENT = "TEST"
    RESOLUTION_KM = 10.0
    REQUIRED_MODE = OperationalMode.CAM
    VARIABLES = (
        VariableSpec(name="field_a", dtype="float32", aggregation="weighted_mean", required_mode=OperationalMode.CAM),
        VariableSpec(name="field_b", dtype="float32", aggregation="weighted_mean", required_mode=OperationalMode.CAM),
    )

    def _load_spatial_region(self, bbox: BoundingBox):
        # 3-D (n_var, n_lat, n_lon): plane 0 == 5.0 everywhere, plane 1 == 2.0.
        data = np.stack(
            [
                np.full((GRID_LATS.size, GRID_LONS.size), 5.0, dtype=np.float32),
                np.full((GRID_LATS.size, GRID_LONS.size), 2.0, dtype=np.float32),
            ]
        )
        return data, GRID_LATS, GRID_LONS


class FakeCatReader(GriddedDataReader):
    """Single-variable categorical reader: every cell is class 3, plus ranked extras."""

    READER_KEY = "_fake_agg_cat"
    INSTRUMENT = "TEST"
    RESOLUTION_KM = 10.0
    REQUIRED_MODE = OperationalMode.CAM
    VARIABLES = (
        VariableSpec(
            name="surface",
            dtype="int16",
            aggregation="weighted_mode",
            required_mode=OperationalMode.CAM,
            n_categories=6,
        ),
    )
    ADDITIONAL_PRODUCT_VARIABLES = (
        VariableSpec(
            name="surface_primary",
            dtype="int16",
            aggregation="weighted_mode_primary",
            required_mode=OperationalMode.CAM,
            n_categories=6,
        ),
        VariableSpec(
            name="surface_secondary",
            dtype="int16",
            aggregation="weighted_mode_secondary",
            required_mode=OperationalMode.CAM,
            n_categories=6,
        ),
    )

    def _load_spatial_region(self, bbox: BoundingBox):
        # 2-D single-variable grid, all class 3.
        data = np.full((GRID_LATS.size, GRID_LONS.size), 3.0, dtype=np.float32)
        return data, GRID_LATS, GRID_LONS


@dataclass
class FakeFootprint:
    """Minimal footprint exposing what the weigher and TileManager need."""

    bbox: BoundingBox
    latitude: float
    longitude: float
    # Optional geometry consumed by the angular-frame weigher (ignored by RadialWeigher).
    subsatellite_latitude: float | None = None
    subsatellite_longitude: float | None = None
    satellite_radius_km: float | None = None
    cone_angle_rate: float | None = None
    viewing_zenith_angle: float | None = 0.0
    off_limb: bool = False


def make_manager(tmp_path: Path) -> TileManager:
    """Return a TileManager holding just the two fake readers, in CAM mode."""
    readers = {
        FakeMeanReader.READER_KEY: FakeMeanReader(tmp_path / "mean.bin"),
        FakeCatReader.READER_KEY: FakeCatReader(tmp_path / "cat.bin"),
    }
    return TileManager(readers, OperationalMode.CAM)
