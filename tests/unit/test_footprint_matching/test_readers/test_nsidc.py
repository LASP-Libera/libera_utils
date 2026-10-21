"""Unit tests for NISEReader.

The tests write small NISE HDF-EOS4 granules with pyhdf (both hemispheres, the
duplicate ``Extent``/``Age`` SDS names and the hemisphere dimension names of a real
granule) and read them back on a coarse 4 x 4 EASE-Grid, so the hemisphere
selection, the pyproj EPSG:3408/3409 -> EPSG:4326 reprojection and the code-to-layer
split all run for real. ``tests/integration/test_nise_real_granule.py`` runs the
reader on a real granule when one is staged.

Real NISE files can be downloaded from:
    NSIDC HTTPS: https://n5eil01u.ecs.nsidc.org/NISE/
    Earthdata login required: https://urs.earthdata.nasa.gov/
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pyhdf.error import HDF4Error

from libera_utils.footprint_matching.readers.nsidc import (
    _HEMISPHERE_LABEL_NORTH,
    _HEMISPHERE_LABEL_SOUTH,
    NISEReader,
    _HemisphereGrid,
)
from libera_utils.footprint_matching.types import BoundingBox, SampleSet, TimeMatch
from tests.test_data.footprint_matching.fixtures import make_nise_hdfeos_fixture

_T = np.datetime64("2026-01-01T00:00:00", "ns")

# Expected output variables, in the canonical order the reader stacks them
# (axis 0 of the returned data array). Kept here so the tests assert the
# ordering contract independently of the reader's internal constants.
_EXPECTED_VARIABLES = (
    "sea_ice_concentration",
    "snow_free_land",
    "permanent_ice",
    "snow_on_land",
    "open_ocean",
)


def _layer_index(name: str) -> int:
    """Return the axis-0 index of variable ``name`` in the reader output."""
    return _EXPECTED_VARIABLES.index(name)


# A coarse 4 x 4 EASE-Grid: 500 km cells give it a usable extent around each pole.
_TEST_ROWS = 4
_TEST_COLS = 4
_TEST_RESOLUTION_M = 500_000.0
_TEST_X_ORIGIN = -1_000_000.0  # meters
_TEST_Y_ORIGIN = 1_000_000.0  # meters


def _small_hemi(epsg: int, label: str) -> _HemisphereGrid:
    """A ``_HemisphereGrid`` on the coarse 4 x 4 test grid."""
    return _HemisphereGrid(
        epsg=epsg,
        hemisphere_label=label,
        grid_rows=_TEST_ROWS,
        grid_cols=_TEST_COLS,
        resolution_m=_TEST_RESOLUTION_M,
        x_origin=_TEST_X_ORIGIN,
        y_origin=_TEST_Y_ORIGIN,
    )


_TEST_HEMI_NORTH = _small_hemi(3408, _HEMISPHERE_LABEL_NORTH)
_TEST_HEMI_SOUTH = _small_hemi(3409, _HEMISPHERE_LABEL_SOUTH)


def _filled(code: int) -> np.ndarray:
    return np.full((_TEST_ROWS, _TEST_COLS), code, dtype=np.uint8)


def _granule(tmp_path: Path, north: np.ndarray | None = None, south: np.ndarray | None = None) -> Path:
    """Write a 4 x 4 NISE granule; a hemisphere not given is all ocean (255)."""
    return make_nise_hdfeos_fixture(
        tmp_path / "NISE_AMSR2_20260101.HDFEOS",
        north=_filled(255) if north is None else north,
        south=_filled(255) if south is None else south,
    )


def _make_reader(tmp_path: Path, north: np.ndarray | None = None) -> NISEReader:
    """A Northern-hemisphere NISEReader over a written 4 x 4 granule."""
    return NISEReader(_granule(tmp_path, north=north), hemispheres=(_TEST_HEMI_NORTH,))


def _whole_grid_box(reader: NISEReader) -> BoundingBox:
    lats_2d, lons_2d = reader._compute_latlon_grid(reader._hemispheres[0])
    return BoundingBox(
        float(lats_2d.min()) - 0.1,
        float(lats_2d.max()) + 0.1,
        float(lons_2d.min()) - 0.1,
        float(lons_2d.max()) + 0.1,
    )


class TestNISEReaderClassAttributes:
    def test_reader_key(self):
        assert NISEReader.READER_KEY == "nise"

    def test_instrument_is_amsr2(self):
        # Operationally the NISE_A2 (AMSR-2) product is ingested; provenance reads AMSR2.
        assert NISEReader.INSTRUMENT == "AMSR2"

    def test_resolution_km(self):
        assert NISEReader.RESOLUTION_KM == 25.0

    def test_time_match_is_daily(self):
        # The daily granule is the match for the whole staged day.
        assert NISEReader.TIME_MATCH is TimeMatch.DAILY

    def test_variable_names_and_order(self):
        assert tuple(v.name for v in NISEReader.VARIABLES) == _EXPECTED_VARIABLES

    def test_all_variables_are_float32_weighted_mean_fractions(self):
        assert all(v.dtype == "float32" for v in NISEReader.VARIABLES)
        assert all(v.aggregation == "weighted_mean" for v in NISEReader.VARIABLES)
        assert all(v.n_categories is None for v in NISEReader.VARIABLES)


class TestNISEReaderConstructor:
    def test_reads_both_hemispheres_by_default(self, tmp_path):
        reader = NISEReader(tmp_path / "NISE.HDFEOS")
        assert [h.hemisphere_label for h in reader._hemispheres] == [_HEMISPHERE_LABEL_NORTH, _HEMISPHERE_LABEL_SOUTH]
        assert [h.epsg for h in reader._hemispheres] == [3408, 3409]


class TestNISEReaderLayerMapping:
    """Verify each NISE Extent code routes to the correct float32 coverage layer."""

    @staticmethod
    def _run(tmp_path: Path, code: int) -> np.ndarray:
        """Return the ``(5, n_pixels)`` value rows served for a grid of one code."""
        reader = _make_reader(tmp_path, north=_filled(code))
        values = reader.samples_in(_whole_grid_box(reader), _T).values
        assert values.shape == (len(_EXPECTED_VARIABLES), _TEST_ROWS * _TEST_COLS)
        return values

    def test_code_50_is_half_sea_ice(self, tmp_path):
        stack = self._run(tmp_path, 50)
        assert np.allclose(stack[_layer_index("sea_ice_concentration")], 0.5, atol=1e-5)
        for name in _EXPECTED_VARIABLES:
            if name != "sea_ice_concentration":
                assert np.all(stack[_layer_index(name)] == 0.0)

    @pytest.mark.parametrize(("code", "fraction"), [(1, 0.01), (100, 1.0)])
    def test_sea_ice_concentration_bounds(self, tmp_path, code, fraction):
        stack = self._run(tmp_path, code)
        assert np.allclose(stack[_layer_index("sea_ice_concentration")], fraction, atol=1e-5)

    @pytest.mark.parametrize(
        ("code", "layer"),
        [
            (0, "snow_free_land"),
            (101, "permanent_ice"),
            (103, "snow_on_land"),
            (104, "snow_on_land"),
            (255, "open_ocean"),
        ],
    )
    def test_class_codes_set_their_indicator_layer(self, tmp_path, code, layer):
        stack = self._run(tmp_path, code)
        assert np.all(stack[_layer_index(layer)] == 1.0)
        assert np.all(stack[_layer_index("sea_ice_concentration")] == 0.0)

    @pytest.mark.parametrize("code", [102, 105, 251, 252, 253])
    def test_unused_and_quality_codes_are_missing_in_every_layer(self, tmp_path, code):
        # Codes with no surface class — 102 and 105–251 ("not used"), 252/253 (quality
        # flags) — are served as NaN in every layer, so the weigher counts them as missing.
        stack = self._run(tmp_path, code)
        assert np.all(np.isnan(stack)), f"code {code} leaked into a layer"


class TestNISEReaderLatLonGrid:
    def test_lat_lon_grid_shape_and_range(self):
        lats_2d, lons_2d = NISEReader("unused.HDFEOS")._compute_latlon_grid(_TEST_HEMI_NORTH)
        assert lats_2d.shape == lons_2d.shape == (_TEST_ROWS, _TEST_COLS)
        assert np.all((lats_2d >= -90) & (lats_2d <= 90))
        assert np.all((lons_2d >= -180) & (lons_2d <= 180))

    def test_northern_grid_is_in_the_northern_hemisphere(self):
        lats_2d, _ = NISEReader("unused.HDFEOS")._compute_latlon_grid(_TEST_HEMI_NORTH)
        assert np.all(lats_2d > 0)

    def test_southern_grid_is_in_the_southern_hemisphere(self):
        lats_2d, _ = NISEReader("unused.HDFEOS")._compute_latlon_grid(_TEST_HEMI_SOUTH)
        assert np.all(lats_2d < 0)


class TestNISEReaderSamples:
    def test_returns_value_rows_and_1d_coords(self, tmp_path):
        reader = _make_reader(tmp_path, north=_filled(50))
        samples = reader.samples_in(_whole_grid_box(reader), _T)
        assert isinstance(samples, SampleSet)
        assert samples.source == "nise"
        assert samples.timestamp_source is None
        assert samples.values.dtype == np.float32
        assert samples.values.shape == (len(NISEReader.VARIABLES), _TEST_ROWS * _TEST_COLS)
        assert samples.lats.shape == samples.lons.shape == (_TEST_ROWS * _TEST_COLS,)

    def test_empty_result_outside_bbox(self, tmp_path):
        reader = _make_reader(tmp_path, north=_filled(50))
        samples = reader.samples_in(BoundingBox(-60.0, -58.0, 170.0, 172.0), _T)
        assert samples.values.shape == (len(NISEReader.VARIABLES), 0)

    def test_data_values_in_0_to_1_range(self, tmp_path):
        mixed = np.array([[0, 50, 100, 101], [0, 25, 75, 103], [0, 1, 99, 101], [0, 0, 0, 0]], dtype=np.uint8)
        reader = _make_reader(tmp_path, north=mixed)
        values = reader.samples_in(_whole_grid_box(reader), _T).values
        assert values.size > 0
        assert np.all((values >= 0.0) & (values <= 1.0))


class TestNISEExtentToCategoryMasks:
    """Directly exercise the Extent-code → five-layer split helper."""

    def test_stack_shape_and_dtype(self):
        masks = NISEReader("unused.HDFEOS")._extent_to_category_masks(np.zeros((_TEST_ROWS, _TEST_COLS), np.uint8))
        assert masks.shape == (len(NISEReader.VARIABLES), _TEST_ROWS, _TEST_COLS)
        assert masks.dtype == np.float32

    def test_each_code_lands_in_expected_layer(self):
        # One pixel per code group, laid out across a 2×3 grid:
        #   60  -> sea ice 0.60      0   -> snow_free_land
        #   101 -> permanent ice     103 -> snow on land (dry)
        #   255 -> open ocean        102 -> missing in every layer
        raw = np.array([[60, 0, 101], [103, 255, 102]], dtype=np.uint8)
        masks = NISEReader("unused.HDFEOS")._extent_to_category_masks(raw)
        assert np.isclose(masks[_layer_index("sea_ice_concentration"), 0, 0], 0.60, atol=1e-5)
        assert masks[_layer_index("snow_free_land"), 0, 1] == 1.0
        assert masks[_layer_index("permanent_ice"), 0, 2] == 1.0
        assert masks[_layer_index("snow_on_land"), 1, 0] == 1.0
        assert masks[_layer_index("open_ocean"), 1, 1] == 1.0
        assert np.all(np.isnan(masks[:, 1, 2]))

    def test_indicator_layers_are_mutually_exclusive_per_pixel(self):
        # At most one 0/1 indicator layer is 1.0 at any pixel (sea ice is a fraction).
        raw = np.array([[0, 101, 104], [255, 102, 0]], dtype=np.uint8)
        masks = NISEReader("unused.HDFEOS")._extent_to_category_masks(raw)
        names = ("snow_free_land", "permanent_ice", "snow_on_land", "open_ocean")
        assert np.all(np.nansum([masks[_layer_index(n)] for n in names], axis=0) <= 1.0)


class TestNISEReadExtentSds:
    """Hemisphere selection on a written granule whose two grids share SDS names."""

    def test_each_label_returns_its_own_grid(self, tmp_path):
        north, south = _filled(50), _filled(101)
        reader = NISEReader(_granule(tmp_path, north=north, south=south))
        np.testing.assert_array_equal(reader._read_extent_sds(_HEMISPHERE_LABEL_NORTH), north)
        np.testing.assert_array_equal(reader._read_extent_sds(_HEMISPHERE_LABEL_SOUTH), south)

    def test_unknown_label_raises(self, tmp_path):
        reader = NISEReader(_granule(tmp_path))
        with pytest.raises(KeyError, match="NISE or NISE_A2 HDF-EOS granule"):
            reader._read_extent_sds("Equatorial")

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(HDF4Error):
            NISEReader(tmp_path / "absent.HDFEOS")._read_extent_sds(_HEMISPHERE_LABEL_NORTH)

    def test_extent_not_matching_the_grid_raises(self, tmp_path):
        # The default 721 x 721 grid declared for a 4 x 4 Extent SDS.
        reader = NISEReader(_granule(tmp_path), hemispheres=(_HemisphereGrid(3408, _HEMISPHERE_LABEL_NORTH),))
        with pytest.raises(ValueError, match=r"Extent SDS has shape \(4, 4\).*721 x 721"):
            reader._load_native()


class TestNISEReaderBothHemispheres:
    """The reader reads and merges both the Northern and Southern EASE-Grids."""

    def test_both_hemispheres_are_concatenated(self, tmp_path):
        path = _granule(tmp_path, north=_filled(50), south=_filled(50))
        points = NISEReader(path, hemispheres=(_TEST_HEMI_NORTH, _TEST_HEMI_SOUTH))._load_native()
        # The point cloud spans both hemispheres and keeps every pixel (no code 254 to drop).
        assert np.any(points.lats > 0)
        assert np.any(points.lats < 0)
        expected = 2 * _TEST_ROWS * _TEST_COLS
        assert points.lats.size == points.lons.size == expected
        assert points.values.shape == (len(NISEReader.VARIABLES), expected)

    def test_southern_box_is_served_from_the_southern_grid(self, tmp_path):
        path = _granule(tmp_path, north=_filled(255), south=_filled(50))
        reader = NISEReader(path, hemispheres=(_TEST_HEMI_SOUTH,))
        values = reader.samples_in(_whole_grid_box(reader), _T).values
        assert values.shape[1] == _TEST_ROWS * _TEST_COLS
        assert np.allclose(values[_layer_index("sea_ice_concentration")], 0.5, atol=1e-5)

    def test_off_earth_code_254_is_dropped_not_zeroed(self, tmp_path):
        # Off-Earth corner pixels (254) are removed from the point cloud entirely, not
        # carried as all-zero layers (which would dilute the opposite hemisphere).
        raw = _filled(50)
        raw[0, 0] = 254
        raw[-1, -1] = 254
        points = _make_reader(tmp_path, north=raw)._load_native()
        expected = _TEST_ROWS * _TEST_COLS - 2
        assert points.lats.size == points.lons.size == expected
        assert points.values.shape == (len(NISEReader.VARIABLES), expected)


class TestNISEReaderRealSubset:
    """A real NISE_A2 (AMSR-2) granule for 2026-06-10, with both full hemisphere grids.

    The reader fixes the 721 x 721 EASE-Grid geometry, so the grids are not cut; only
    the ``Age`` fields are dropped. Northern ``Extent`` stays the first SDS and Southern
    the second, so the selection by dimension label runs as on the distributed file.
    Pinned values are the source granule's own.
    """

    _TIME = np.datetime64("2026-06-10T00:00:00", "ns")

    @pytest.fixture
    def reader(self, fmatch_real_data_path):
        return NISEReader(fmatch_real_data_path / "NISE_AMSR2_20260610_subset.HDFEOS")

    def test_reads_both_extent_sds_by_hemisphere(self, reader):
        north = reader._read_extent_sds(_HEMISPHERE_LABEL_NORTH)
        south = reader._read_extent_sds(_HEMISPHERE_LABEL_SOUTH)
        assert north.shape == south.shape == (721, 721)
        # Two genuinely different grids, not the same SDS resolved twice.
        assert not np.array_equal(north, south)

    def test_point_cloud_spans_both_hemispheres_without_corner_fill(self, reader):
        points = reader._load_native()
        # 2 x 721 x 721 pixels less the off-Earth corners (code 254).
        assert points.lats.size == 811786
        assert points.lats.max() > 89.9
        assert points.lats.min() < -89.9

    def test_surface_classes_at_known_places(self, reader):
        layer = {name: _layer_index(name) for name in _EXPECTED_VARIABLES}
        south_pole = reader.samples_in(BoundingBox(-90.0, -89.5, -180.0, 180.0), self._TIME).values
        assert np.all(south_pole[layer["permanent_ice"]] == 1.0)
        arctic = reader.samples_in(BoundingBox(84.9, 85.1, -0.3, 0.3), self._TIME).values
        assert np.allclose(arctic[layer["sea_ice_concentration"]], 0.93)
        boulder = reader.samples_in(BoundingBox(39.9, 40.1, -105.3, -104.7), self._TIME).values
        assert np.all(boulder[layer["snow_free_land"]] == 1.0)

    def test_suspect_ice_at_the_north_pole_is_missing_in_every_layer(self, reader):
        # The AMSR-2 pole hole is coded 253 (suspect ice value), which belongs to no class.
        north_pole = reader.samples_in(BoundingBox(89.5, 90.0, -180.0, 180.0), self._TIME).values
        assert north_pole.size > 0
        assert np.isnan(north_pole).all()
