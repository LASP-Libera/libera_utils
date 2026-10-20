"""Unit tests for FMATCH product assembly and writing (product.py).

Covers the assembly entry points (``assemble_fmatch_dataset`` / ``write_fmatch_product``) for both
timescales, their failure paths, the integer fill, the derived sun glint angle, the coverage
QA bits, and the small helpers whose edge cases decide a product value (longitude wrapping, the
viewing zenith angle read).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import xarray as xr

from libera_utils.constants import DataProductIdentifier
from libera_utils.footprint_matching import product
from libera_utils.footprint_matching.product import (
    _assemble_camtime_dataset,
    _assemble_radiometer_dataset,
    _fill_value_for,
    _footprint_geometry,
    _merge_coverage_qa,
    _normalize_longitude,
    assemble_fmatch_dataset,
    compute_derived_viewing_geometry,
    load_fmatch_definition,
    write_fmatch_product,
)
from libera_utils.footprint_matching.types import BoundingBox, FmatchCoverageFlag, OperationalMode
from libera_utils.footprint_matching.weighting import AngularPSFWeigher
from libera_utils.io.product_definition import LiberaVariableDefinition
from tests.unit.test_footprint_matching.fake_reader_fixtures import FakeFootprint, make_sources

_PARTIAL = int(FmatchCoverageFlag.PARTIAL_COVERAGE)
_INSUFFICIENT = int(FmatchCoverageFlag.INSUFFICIENT_COVERAGE)
_TRUNCATED = int(FmatchCoverageFlag.LIMB_TRUNCATED)
_OFF_LIMB = int(FmatchCoverageFlag.OFF_LIMB)


def _l1b_inputs(n: int = 3) -> dict[str, np.ndarray]:
    """L1B pass-through inputs for ``n`` nadir footprints over the fake readers' grid near (0.5, 0.5)."""
    return {
        "RADIOMETER_TIME": np.datetime64("2028-01-01T00:00:00", "ns") + np.arange(n) * np.timedelta64(1, "s"),
        "latitude": np.full(n, 0.5),
        "longitude": np.full(n, 0.5),
        "altitude": np.zeros(n),
        "solar_zenith_angle": np.full(n, 30.0),
        "viewing_zenith_angle": np.zeros(n),
        "relative_azimuth_angle": np.full(n, 90.0),
        "subsatellite_latitude": np.full(n, 0.5),
        "subsatellite_longitude": np.full(n, 0.5),
        "satellite_radius_km": np.full(n, 7213.0),
        "cone_angle_rate": np.full(n, -1.0),
    }


@dataclass
class _FakePseudoFootprint:
    """The attributes the camera-timescale assembly reads off a camera pseudo-footprint."""

    time: np.datetime64
    bbox: BoundingBox
    center_ix: int
    center_iy: int
    slice_x: slice
    slice_y: slice
    latitude: float = 0.5
    longitude: float = 0.5
    altitude: float = 0.0
    solar_zenith_angle: float = 30.0
    viewing_zenith_angle: float = 10.0
    relative_azimuth_angle: float = 90.0
    q_flags: int = 0


def _pseudo_footprints() -> list[_FakePseudoFootprint]:
    """Two images: the first segmented into two subsections, the second into one (one padded cell)."""
    first = np.datetime64("2028-01-01T00:00:00", "ns")
    second = np.datetime64("2028-01-01T00:00:10", "ns")
    box = BoundingBox(0.2, 0.8, 0.2, 0.8)
    return [
        _FakePseudoFootprint(first, box, center_ix=0, center_iy=0, slice_x=slice(0, 4), slice_y=slice(0, 4)),
        _FakePseudoFootprint(first, box, center_ix=6, center_iy=2, slice_x=slice(4, 8), slice_y=slice(0, 4)),
        _FakePseudoFootprint(second, box, center_ix=1, center_iy=1, slice_x=slice(0, 4), slice_y=slice(0, 4)),
    ]


class TestRadiometerAssemblyRaises:
    """_assemble_radiometer_dataset rejects inputs it cannot build a product from."""

    @staticmethod
    def _without(name: str) -> dict[str, np.ndarray]:
        inputs = _l1b_inputs()
        del inputs[name]
        return inputs

    @staticmethod
    def _short(name: str) -> dict[str, np.ndarray]:
        inputs = _l1b_inputs()
        inputs[name] = inputs[name][:-1]
        return inputs

    @pytest.mark.parametrize(
        ("inputs", "match"),
        [
            pytest.param(lambda: TestRadiometerAssemblyRaises._without("relative_azimuth_angle"), "missing", id="key"),
            pytest.param(lambda: _l1b_inputs(0), "zero footprints", id="empty"),
            pytest.param(lambda: TestRadiometerAssemblyRaises._short("latitude"), "inconsistent", id="length"),
        ],
    )
    def test_bad_inputs_raise(self, inputs, match):
        with pytest.raises(ValueError, match=match):
            assemble_fmatch_dataset(OperationalMode.CAM, inputs(), allow_placeholders=True)

    def test_camera_timescale_mode_raises(self):
        with pytest.raises(ValueError, match="camera-timescale mode"):
            _assemble_radiometer_dataset(_l1b_inputs(), mode=OperationalMode.CAM_CAMTIME)


class TestCamtimeAssemblyRaises:
    """_assemble_camtime_dataset rejects a radiometer mode and an empty footprint list."""

    def test_zero_pseudo_footprints_raise(self):
        with pytest.raises(ValueError, match="zero pseudo-footprints"):
            assemble_fmatch_dataset(OperationalMode.CAM_CAMTIME, [])

    def test_radiometer_mode_raises(self):
        with pytest.raises(ValueError, match="not a camera-timescale mode"):
            _assemble_camtime_dataset(_pseudo_footprints(), mode=OperationalMode.CAM)


class TestAncillaryInputs:
    """Without ancillary inputs, assembly raises unless placeholders are asked for."""

    def test_no_ancillary_inputs_raise(self):
        with pytest.raises(ValueError, match="allow_placeholders"):
            assemble_fmatch_dataset(OperationalMode.CAM, _l1b_inputs())

    def test_writer_raises_without_ancillary_inputs(self, tmp_path):
        with pytest.raises(ValueError, match="allow_placeholders"):
            write_fmatch_product(
                OperationalMode.CAM, _l1b_inputs(), tmp_path, algorithm_version="1.0.0", input_files="x"
            )

    def test_allow_placeholders_writes_fills(self):
        dataset = assemble_fmatch_dataset(OperationalMode.CAM, _l1b_inputs(), allow_placeholders=True)
        # Integer categories take the -1 fill, not 0 (a real category); q_flags 0 means no flag set.
        np.testing.assert_array_equal(dataset["igbp_surface_type"].values, [-1, -1, -1])
        np.testing.assert_array_equal(dataset["q_flags"].values, [0, 0, 0])
        assert np.isnan(dataset["psf_coverage_fraction"].values).all()


class TestFillValueFor:
    """_fill_value_for gives integers -1 (0 for a bitmask) and floats NaN."""

    @staticmethod
    def _definition(dtype: str, attributes: dict) -> LiberaVariableDefinition:
        return LiberaVariableDefinition(dtype=dtype, dimensions=["RADIOMETER_TIME"], attributes=attributes)

    def test_declared_fill_wins(self):
        assert _fill_value_for(self._definition("int16", {"long_name": "x", "_FillValue": -999}))[1] == -999

    def test_integer_takes_minus_one(self):
        dtype, fill = _fill_value_for(self._definition("int16", {"long_name": "surface type"}))
        assert dtype == np.int16
        assert fill == -1

    def test_bitmask_takes_zero(self):
        assert _fill_value_for(self._definition("int32", {"long_name": "flags", "flag_masks": [1, 4]}))[1] == 0

    def test_float_takes_nan(self):
        assert np.isnan(_fill_value_for(self._definition("float32", {"long_name": "x"}))[1])


class TestNormalizeLongitude:
    """Only longitudes above 180 wrap, so a full-span box keeps its 180 bound."""

    @pytest.mark.parametrize(
        ("lon_min", "lon_max", "expected"),
        [
            (-180.0, 180.0, (-180.0, 180.0)),
            (170.0, 180.0, (170.0, 180.0)),
            (179.9, 180.2, (179.9, -179.8)),
        ],
    )
    def test_box_bounds(self, lon_min, lon_max, expected):
        assert (_normalize_longitude(lon_min), _normalize_longitude(lon_max)) == pytest.approx(expected)


class TestFootprintGeometry:
    @pytest.mark.parametrize("viewing_zenith", [None, float("nan")])
    def test_missing_viewing_zenith_raises(self, viewing_zenith):
        footprint = FakeFootprint(BoundingBox(0.2, 0.8, 0.2, 0.8), 0.5, 0.5, viewing_zenith_angle=viewing_zenith)
        with pytest.raises(ValueError, match="viewing zenith angle"):
            _footprint_geometry(footprint)


class TestSunglint:
    def test_specular_geometry_is_zero(self):
        glint = compute_derived_viewing_geometry(np.array([30.0]), np.array([30.0]), np.array([180.0]))
        assert glint["sunglint_angle"][0] == pytest.approx(0.0, abs=1e-6)

    def test_nadir_view_equals_solar_zenith(self):
        glint = compute_derived_viewing_geometry(np.array([30.0]), np.array([0.0]), np.array([0.0]))
        assert glint["sunglint_angle"][0] == pytest.approx(30.0)

    def test_nan_input_propagates(self):
        glint = compute_derived_viewing_geometry(np.array([np.nan]), np.array([0.0]), np.array([0.0]))
        assert np.isnan(glint["sunglint_angle"][0])


class TestMergeCoverageQa:
    """Coverage below 75% is insufficient, 75-95% partial; truncation and off-limb add their own bits."""

    @staticmethod
    def _footprint(*, truncated: bool = False, off_limb: bool = False) -> FakeFootprint:
        bbox = BoundingBox(0.2, 0.8, 0.2, 0.8, truncated=truncated)
        return FakeFootprint(bbox, 0.5, 0.5, off_limb=off_limb)

    def test_coverage_bits(self):
        definition = load_fmatch_definition(OperationalMode.CAM)
        footprints = [
            self._footprint(),
            self._footprint(),
            self._footprint(),
            self._footprint(truncated=True),
            self._footprint(off_limb=True),
        ]
        data: dict[str, np.ndarray] = {}
        _merge_coverage_qa(data, definition, np.array([0.5, 0.8, 1.0, 1.0, 0.0]), footprints)
        np.testing.assert_array_equal(
            data["q_flags"], [_INSUFFICIENT, _PARTIAL, 0, _TRUNCATED, _OFF_LIMB | _INSUFFICIENT]
        )
        np.testing.assert_allclose(data["psf_coverage_fraction"], [0.5, 0.8, 1.0, 1.0, 0.0])

    def test_existing_flags_are_kept(self):
        definition = load_fmatch_definition(OperationalMode.CAM)
        data = {"q_flags": np.array([_TRUNCATED], dtype=np.int32)}
        _merge_coverage_qa(data, definition, np.array([0.8]), [self._footprint()])
        assert data["q_flags"][0] == _TRUNCATED | _PARTIAL


class TestCamtimeAssembly:
    def test_grid_padding_takes_fills(self):
        cloud_fraction = np.array([0.1, 0.2, 0.3])
        dataset = assemble_fmatch_dataset(
            OperationalMode.CAM_CAMTIME,
            _pseudo_footprints(),
            cloud_fraction_camera=cloud_fraction,
            allow_placeholders=True,
        )
        assert dataset["latitude"].shape == (2, 2)
        # Row 1 holds the second image's single subsection; its second cell is padding.
        np.testing.assert_array_equal(dataset["center_pixel_x"].values, [[0, 6], [1, -1]])
        np.testing.assert_array_equal(dataset["camera_pixel_x_max"].values, [[3, 7], [3, -1]])
        assert np.isnan(dataset["latitude"].values[1, 1])
        np.testing.assert_allclose(dataset["cloud_fraction_camera"].values, [[0.1, 0.2], [0.3, np.nan]])
        np.testing.assert_array_equal(dataset["igbp_surface_type"].values, np.full((2, 2), -1))


class TestIntegerAggregatePlacement:
    """An aggregated integer column takes the -1 fill where a footprint had no usable data."""

    @staticmethod
    def _aggregate_with_one_gap(monkeypatch):
        def fake_aggregate(mode, footprints, sources=None, **kwargs):
            values = np.full(len(footprints), 3.0)
            values[0] = np.nan
            return {"igbp_surface_type": values}, np.ones(len(footprints))

        monkeypatch.setattr(product, "aggregate_external_variables", fake_aggregate)

    def test_radiometer_column(self, tmp_path, monkeypatch):
        self._aggregate_with_one_gap(monkeypatch)
        dataset = assemble_fmatch_dataset(OperationalMode.CAM, _l1b_inputs(), sources=make_sources(tmp_path))
        assert dataset["igbp_surface_type"].dtype == np.int16
        np.testing.assert_array_equal(dataset["igbp_surface_type"].values, [-1, 3, 3])

    def test_camtime_grid(self, tmp_path, monkeypatch):
        self._aggregate_with_one_gap(monkeypatch)
        dataset = assemble_fmatch_dataset(
            OperationalMode.CAM_CAMTIME, _pseudo_footprints(), sources=make_sources(tmp_path)
        )
        # The first footprint's gap and the padded cell both take -1.
        np.testing.assert_array_equal(dataset["igbp_surface_type"].values, [[-1, 3], [3, -1]])


class TestRadiometerAggregation:
    def test_defaults_to_angular_weigher(self, tmp_path, monkeypatch):
        calls = []
        original = AngularPSFWeigher.weight_field

        def spy(self, *args, **kwargs):
            calls.append(type(self))
            return original(self, *args, **kwargs)

        monkeypatch.setattr(AngularPSFWeigher, "weight_field", spy)
        assemble_fmatch_dataset(OperationalMode.CAM, _l1b_inputs(), sources=make_sources(tmp_path))
        assert calls
        assert set(calls) == {AngularPSFWeigher}


class TestWriteFmatchProduct:
    def test_round_trip(self, tmp_path):
        sources = make_sources(tmp_path)
        filename = write_fmatch_product(
            OperationalMode.CAM,
            _l1b_inputs(),
            tmp_path,
            algorithm_version="1.0.0",
            input_files="x",
            sources=sources,
        )
        assert filename.data_product_id == DataProductIdentifier.aux_fmatch_cam

        with xr.open_dataset(filename.path, mask_and_scale=False) as written:
            coverage = written["psf_coverage_fraction"].values
            assert np.all((coverage >= 0.0) & (coverage <= 1.0))
            expected = np.where(
                coverage < product.ACCEPT_COVERAGE_THRESHOLD,
                _INSUFFICIENT,
                np.where(coverage < product.PARTIAL_COVERAGE_THRESHOLD, _PARTIAL, 0),
            )
            np.testing.assert_array_equal(written["q_flags"].values, expected)
            # The fake readers feed no IGBP data, so the column is the -1 integer fill on disk.
            np.testing.assert_array_equal(written["igbp_surface_type"].values, [-1, -1, -1])
            assert "_FillValue" not in written["igbp_surface_type"].attrs
