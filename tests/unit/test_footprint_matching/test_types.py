"""Unit tests for the shared FMATCH types (libera_utils.footprint_matching.types)."""

from __future__ import annotations

import numpy as np
import pytest

from libera_utils.footprint_matching.types import (
    BoundingBox,
    OperationalMode,
    RadiometerFootprint,
    VariableSpec,
    with_standard_deviation_companions,
)

_BOX = BoundingBox(lat_min=9.0, lat_max=11.0, lon_min=19.0, lon_max=21.0)


def _footprint(**overrides) -> RadiometerFootprint:
    kwargs = {
        "bbox": _BOX,
        "time": np.datetime64("2026-07-12T00:00:00"),
        "latitude": 10.0,
        "longitude": 20.0,
        "viewing_zenith_angle": 30.0,
        "subsatellite_latitude": 12.0,
        "subsatellite_longitude": 20.0,
        "satellite_radius_km": 7213.0,
    }
    kwargs.update(overrides)
    return RadiometerFootprint(**kwargs)


class TestRadiometerFootprintBox:
    """``bbox`` is ``None`` exactly when the footprint is ``off_limb``."""

    def test_on_earth_footprint_carries_a_box(self):
        assert _footprint().bbox == _BOX

    def test_off_limb_footprint_has_no_box(self):
        footprint = _footprint(bbox=None, off_limb=True)
        assert footprint.bbox is None

    def test_missing_box_on_earth_raises(self):
        with pytest.raises(ValueError, match="no bbox but off_limb is False"):
            _footprint(bbox=None)

    def test_box_on_off_limb_footprint_raises(self):
        with pytest.raises(ValueError, match="off_limb but was given a bbox"):
            _footprint(off_limb=True)


class TestVariableSpec:
    """``modes`` is a non-empty frozenset of ``OperationalMode`` naming every product that carries the variable."""

    def test_valid_spec_is_carried_by_exactly_its_modes(self):
        spec = VariableSpec(
            name="v", dtype="float32", aggregation="weighted_mean", modes=frozenset({OperationalMode.CAM})
        )
        assert OperationalMode.CAM in spec.modes
        assert OperationalMode.IMAGER not in spec.modes
        assert spec in {spec}

    def test_missing_modes_raises(self):
        with pytest.raises(TypeError, match="modes"):
            VariableSpec(name="v", dtype="float32", aggregation="weighted_mean")

    def test_empty_modes_raises(self):
        with pytest.raises(ValueError, match="VariableSpec 'v' has empty modes: name at least one product"):
            VariableSpec(name="v", dtype="float32", aggregation="weighted_mean", modes=frozenset())

    @pytest.mark.parametrize(
        "modes",
        [
            pytest.param((OperationalMode.CAM,), id="tuple"),
            pytest.param({OperationalMode.CAM}, id="set"),
            pytest.param(frozenset({"FMATCH-CAM"}), id="string-member"),
        ],
    )
    def test_modes_that_are_not_a_frozenset_of_modes_raise(self, modes):
        with pytest.raises(TypeError, match="VariableSpec 'v' has modes=.*pass a frozenset of OperationalMode"):
            VariableSpec(name="v", dtype="float32", aggregation="weighted_mean", modes=modes)

    def test_standard_deviation_companion_has_its_parents_modes(self):
        modes = frozenset({OperationalMode.IMAGER, OperationalMode.IMAGER_CAMTIME})
        parent = VariableSpec(name="v", dtype="float32", aggregation="weighted_mean", modes=modes)
        _, companion = with_standard_deviation_companions((parent,))
        assert companion.name == "v_standard_deviation"
        assert companion.modes == modes
