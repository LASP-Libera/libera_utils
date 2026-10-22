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

from pathlib import Path

import numpy as np
import pytest
from pyhdf.error import HDF4Error

from libera_utils.footprint_matching.readers.nsidc import (
    _DEFAULT_HEMISPHERES,
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
        """``NISEReader.READER_KEY`` is ``"nise"``.

        The key is the name the reader registers under and the ``source`` tag on every ``SampleSet`` it serves, so a
        changed key would break pipeline configuration that selects the reader and output naming built from it.

        The class attribute is compared with the literal string.
        """
        assert NISEReader.READER_KEY == "nise"

    def test_instrument_is_amsr2(self):
        """``NISEReader.INSTRUMENT`` is ``"AMSR2"``.

        The instrument is recorded in each output variable's ``long_name``, and the pipeline ingests the AMSR-2 based
        NISE_A2 product, so a different label would misstate the provenance of the published sea ice and snow layers.

        The class attribute is compared with the literal string.
        """
        # Operationally the NISE_A2 (AMSR-2) product is ingested; provenance reads AMSR2.
        assert NISEReader.INSTRUMENT == "AMSR2"

    def test_resolution_km(self):
        """``NISEReader.RESOLUTION_KM`` is 25 km, the nominal NISE EASE-Grid cell size.

        The reader base class pads every query box by half this cell size and reports it as the samples' resolution, so
        a wrong value would serve too few or too many pixels at a footprint's rim.

        The class attribute is compared with 25.0.
        """
        assert NISEReader.RESOLUTION_KM == 25.0

    def test_time_match_is_daily(self):
        """``NISEReader.TIME_MATCH`` is ``TimeMatch.DAILY``.

        A NISE granule is a daily map with no time axis, so every pixel in a box must be served whatever the footprint
        time; a time-selecting policy would demand per-sample times the granule does not carry.

        The class attribute is checked for identity with the enum member.
        """
        # The daily granule is the match for the whole staged day.
        assert NISEReader.TIME_MATCH is TimeMatch.DAILY

    def test_variable_names_and_order(self):
        """``NISEReader.VARIABLES`` lists the five coverage layers in the order the reader stacks them.

        The aggregation engine indexes each layer by its position on axis 0 of the served values, so a renamed or
        reordered entry would attach one surface class's numbers to another class's name.

        The variable names are compared, in order, with an expected tuple kept in the test module independently of the
        reader's internal constants.
        """
        assert tuple(v.name for v in NISEReader.VARIABLES) == _EXPECTED_VARIABLES

    def test_all_variables_are_float32_weighted_mean_fractions(self):
        """Every ``NISEReader`` variable is a float32 fraction aggregated by ``weighted_mean``.

        The layers are per-pixel fractions or 0/1 indicators that the footprint average turns into class fractions; a
        categorical setting (``n_categories``) or another aggregation would produce a different and wrong product.

        Each ``VariableSpec`` is checked for ``dtype == "float32"``, ``aggregation == "weighted_mean"`` and
        ``n_categories is None``.
        """
        assert all(v.dtype == "float32" for v in NISEReader.VARIABLES)
        assert all(v.aggregation == "weighted_mean" for v in NISEReader.VARIABLES)
        assert all(v.n_categories is None for v in NISEReader.VARIABLES)


class TestNISEReaderConstructor:
    def test_reads_both_hemispheres_by_default(self, tmp_path):
        """A ``NISEReader`` built without ``hemispheres`` reads the Northern and then the Southern EASE-Grid.

        A NISE granule carries both polar grids, and a reader that defaulted to one would leave every footprint over the
        other pole with no sea ice or snow data.

        A reader is constructed on a path that is never opened, and its hemisphere labels and EPSG codes (3408, 3409)
        are checked in order.
        """
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
        """Extent code 50 is served as 0.5 sea ice concentration and 0.0 in the four indicator layers.

        Codes 1-100 are percent sea ice concentration, so a mid-range code checks both the percent-to-fraction scaling
        and that a sea ice pixel is not also counted as land, permanent ice, snow or open ocean.

        A Northern grid filled with code 50 is written and queried over its whole extent; the sea ice layer must be 0.5
        and every other layer 0.0.
        """
        stack = self._run(tmp_path, 50)
        assert np.allclose(stack[_layer_index("sea_ice_concentration")], 0.5, atol=1e-5)
        for name in _EXPECTED_VARIABLES:
            if name != "sea_ice_concentration":
                assert np.all(stack[_layer_index(name)] == 0.0)

    @pytest.mark.parametrize(("code", "fraction"), [(1, 0.01), (100, 1.0)])
    def test_sea_ice_concentration_bounds(self, tmp_path, code, fraction):
        """The end codes of the sea ice range, 1 and 100, are served as concentrations 0.01 and 1.0.

        An off-by-one at either end of the 1-100 range would drop the lowest or highest concentrations from the sea ice
        layer or scale them wrongly.

        The parameters vary the code filled into the Northern grid and the expected fraction in the sea ice layer.
        """
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
        """Each surface-class code sets its own indicator layer to 1.0 and leaves sea ice concentration at 0.0.

        The footprint fraction of snow-free land, permanent ice, snow on land and open ocean comes from these
        indicators, so a code routed to the wrong layer would misreport the surface under every footprint containing it.

        The parameters pair each class code (0, 101, 103, 104, 255) with its layer; a grid filled with the code is
        queried, and that layer must be 1.0 everywhere and the sea ice layer 0.0.
        """
        stack = self._run(tmp_path, code)
        assert np.all(stack[_layer_index(layer)] == 1.0)
        assert np.all(stack[_layer_index("sea_ice_concentration")] == 0.0)

    @pytest.mark.parametrize("code", [102, 105, 251, 252, 253])
    def test_unused_and_quality_codes_are_missing_in_every_layer(self, tmp_path, code):
        """Codes that belong to no surface class are served as NaN in all five layers.

        The weigher counts a NaN pixel as missing, which lowers a footprint's coverage; a 0.0 instead would pull every
        class fraction toward zero for pixels the product itself marks as unused or unreliable.

        The parameters vary the code over 102, both ends of the unused range 105-251, and the 252 and 253 quality flags;
        a grid filled with the code is queried and every served value must be NaN.
        """
        # Codes with no surface class — 102 and 105–251 ("not used"), 252/253 (quality
        # flags) — are served as NaN in every layer, so the weigher counts them as missing.
        stack = self._run(tmp_path, code)
        assert np.all(np.isnan(stack)), f"code {code} leaked into a layer"


class TestNISEReaderLatLonGrid:
    def test_lat_lon_grid_shape_and_range(self):
        """``_compute_latlon_grid`` returns one latitude and longitude per pixel, within valid geographic bounds.

        The reader serves every pixel at its own reprojected center, so a misshapen or out-of-range coordinate array
        would misplace the data or misalign it with the values.

        The coarse 4 x 4 Northern test grid is reprojected and both arrays are checked for shape ``(4, 4)``, latitudes
        in [-90, 90] and longitudes in [-180, 180].
        """
        lats_2d, lons_2d = NISEReader("unused.HDFEOS")._compute_latlon_grid(_TEST_HEMI_NORTH)
        assert lats_2d.shape == lons_2d.shape == (_TEST_ROWS, _TEST_COLS)
        assert np.all((lats_2d >= -90) & (lats_2d <= 90))
        assert np.all((lons_2d >= -180) & (lons_2d <= 180))

    def test_northern_grid_is_in_the_northern_hemisphere(self):
        """The Northern EASE-Grid (EPSG:3408) reprojects to positive latitudes.

        Each hemisphere is reprojected with its own CRS; applying the wrong one would place Arctic data over Antarctica.

        The coarse 4 x 4 Northern test grid, which lies near the pole, is reprojected and every latitude must be above
        0.
        """
        lats_2d, _ = NISEReader("unused.HDFEOS")._compute_latlon_grid(_TEST_HEMI_NORTH)
        assert np.all(lats_2d > 0)

    def test_southern_grid_is_in_the_southern_hemisphere(self):
        """The Southern EASE-Grid (EPSG:3409) reprojects to negative latitudes.

        Each hemisphere is reprojected with its own CRS; applying the wrong one would place Antarctic data over the
        Arctic.

        The coarse 4 x 4 Southern test grid, which lies near the pole, is reprojected and every latitude must be below
        0.
        """
        lats_2d, _ = NISEReader("unused.HDFEOS")._compute_latlon_grid(_TEST_HEMI_SOUTH)
        assert np.all(lats_2d < 0)

    @pytest.mark.parametrize("hemi", _DEFAULT_HEMISPHERES, ids=lambda h: h.hemisphere_label)
    def test_full_grid_has_twelve_non_finite_corner_pixels(self, hemi):
        """On the full 721 x 721 grid, exactly 12 corner pixels reproject to non-finite coordinates.

        Pixels with a non-finite coordinate are dropped by the reader base class's point index; this pins how many there
        are, so a change to the grid geometry (origin, cell size or half-cell offset) that moved pixels into or out of
        the projection domain is caught.

        The parameters run each default hemisphere grid; the test counts non-finite latitudes and checks that latitude
        and longitude are non-finite at the same pixels.
        """
        # Of the 721 x 721 EASE-Grid, only 12 corner pixels lie outside the projection's domain.
        lats_2d, lons_2d = NISEReader("unused.HDFEOS")._compute_latlon_grid(hemi)
        assert (~np.isfinite(lats_2d)).sum() == 12
        assert np.array_equal(np.isfinite(lats_2d), np.isfinite(lons_2d))


class TestNISEReaderSamples:
    def test_returns_value_rows_and_1d_coords(self, tmp_path):
        """``NISEReader.samples_in`` returns a ``SampleSet`` of float32 value rows per variable and 1-D coordinates.

        Downstream weighting expects this layout: a ``(5, n_pixels)`` value array matched to flat per-pixel coordinates,
        tagged with the reader's key and no timestamp source, as for any surface product.

        A 4 x 4 Northern grid of code 50 is queried over its whole extent, and the result's type, ``source``,
        ``timestamp_source``, value dtype and shape, and coordinate shapes are checked.
        """
        reader = _make_reader(tmp_path, north=_filled(50))
        samples = reader.samples_in(_whole_grid_box(reader), _T)
        assert isinstance(samples, SampleSet)
        assert samples.source == "nise"
        assert samples.timestamp_source is None
        assert samples.values.dtype == np.float32
        assert samples.values.shape == (len(NISEReader.VARIABLES), _TEST_ROWS * _TEST_COLS)
        assert samples.lats.shape == samples.lons.shape == (_TEST_ROWS * _TEST_COLS,)

    def test_empty_result_outside_bbox(self, tmp_path):
        """A query box with no NISE pixels in it returns zero samples, keeping the five-variable first axis.

        A footprint outside the grid must get an empty result of the usual layout rather than an error or misshaped
        array, so the pipeline can treat it as no coverage.

        A Northern-only reader is queried with a small box at 58-60 degrees south, and the values must have shape ``(5,
        0)``.
        """
        reader = _make_reader(tmp_path, north=_filled(50))
        samples = reader.samples_in(BoundingBox(-60.0, -58.0, 170.0, 172.0), _T)
        assert samples.values.shape == (len(NISEReader.VARIABLES), 0)

    def test_data_values_in_0_to_1_range(self, tmp_path):
        """Every served value lies in [0, 1] for a grid mixing class codes and sea ice concentrations.

        Each layer is declared as a fraction from 0.0 to 1.0, and a value outside that range would yield impossible
        footprint fractions.

        A 4 x 4 Northern grid of snow-free land, permanent ice, snow and several concentrations is queried over its
        whole extent; the values must be non-empty and all within [0, 1].
        """
        mixed = np.array([[0, 50, 100, 101], [0, 25, 75, 103], [0, 1, 99, 101], [0, 0, 0, 0]], dtype=np.uint8)
        reader = _make_reader(tmp_path, north=mixed)
        values = reader.samples_in(_whole_grid_box(reader), _T).values
        assert values.size > 0
        assert np.all((values >= 0.0) & (values <= 1.0))


class TestNISEExtentToCategoryMasks:
    """Directly exercise the Extent-code → five-layer split helper."""

    def test_stack_shape_and_dtype(self):
        """``_extent_to_category_masks`` returns a float32 stack of shape ``(5, n_rows, n_cols)``.

        The reader flattens this stack into the served values, so the variable axis must come first and match the number
        of declared variables.

        A 4 x 4 array of code 0 is split, and the shape and dtype of the result are checked.
        """
        masks = NISEReader("unused.HDFEOS")._extent_to_category_masks(np.zeros((_TEST_ROWS, _TEST_COLS), np.uint8))
        assert masks.shape == (len(NISEReader.VARIABLES), _TEST_ROWS, _TEST_COLS)
        assert masks.dtype == np.float32

    def test_each_code_lands_in_expected_layer(self):
        """``_extent_to_category_masks`` routes one pixel of each code group to its layer in a single array.

        Checking the codes side by side catches a layer mapping that holds for a uniform grid but mixes up pixels within
        one array.

        A 2 x 3 array holds codes 60, 0, 101, 103, 255 and 102; the test checks the sea ice value 0.60, the 1.0 in each
        class's indicator layer at its pixel, and NaN in every layer at the code 102 pixel.
        """
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
        """At most one of the four 0/1 indicator layers is 1.0 at any pixel.

        Each NISE code names one surface class, so a pixel counted in two indicator layers would inflate both class
        fractions for a footprint.

        A 2 x 3 array of class codes and code 102 is split, and the NaN-ignoring sum of the four indicator layers must
        not exceed 1.0 at any pixel.
        """
        # At most one 0/1 indicator layer is 1.0 at any pixel (sea ice is a fraction).
        raw = np.array([[0, 101, 104], [255, 102, 0]], dtype=np.uint8)
        masks = NISEReader("unused.HDFEOS")._extent_to_category_masks(raw)
        names = ("snow_free_land", "permanent_ice", "snow_on_land", "open_ocean")
        assert np.all(np.nansum([masks[_layer_index(n)] for n in names], axis=0) <= 1.0)


class TestNISEReadExtentSds:
    """Hemisphere selection on a written granule whose two grids share SDS names."""

    def test_each_label_returns_its_own_grid(self, tmp_path):
        """``_read_extent_sds`` returns the ``Extent`` SDS of the hemisphere whose label it is given.

        Both hemispheres name their field ``Extent``, so selecting by name always returns the Northern grid; the
        Southern data would then be silently replaced by the Northern.

        A granule with different codes in each hemisphere is written, and each label must return that hemisphere's
        array.
        """
        north, south = _filled(50), _filled(101)
        reader = NISEReader(_granule(tmp_path, north=north, south=south))
        np.testing.assert_array_equal(reader._read_extent_sds(_HEMISPHERE_LABEL_NORTH), north)
        np.testing.assert_array_equal(reader._read_extent_sds(_HEMISPHERE_LABEL_SOUTH), south)

    def test_unknown_label_raises(self, tmp_path):
        """``_read_extent_sds`` raises ``KeyError`` when no ``Extent`` SDS matches the hemisphere label.

        A staged file without the expected grid must stop the run with a message saying what file to stage, rather than
        serving no data or another hemisphere's data.

        A valid granule is read with the label ``"Equatorial"``, and the error message is checked for its staging
        advice.
        """
        reader = NISEReader(_granule(tmp_path))
        with pytest.raises(KeyError, match="NISE or NISE_A2 HDF-EOS granule"):
            reader._read_extent_sds("Equatorial")

    def test_missing_file_raises(self, tmp_path):
        """``_read_extent_sds`` raises pyhdf's ``HDF4Error`` when the file does not exist.

        A missing input must fail the run rather than read as an empty grid.

        A reader is pointed at a path that was never written, and reading the Northern ``Extent`` must raise
        ``HDF4Error``.
        """
        with pytest.raises(HDF4Error):
            NISEReader(tmp_path / "absent.HDFEOS")._read_extent_sds(_HEMISPHERE_LABEL_NORTH)

    def test_extent_not_matching_the_grid_raises(self, tmp_path):
        """``NISEReader._load_native`` raises ``ValueError`` when an ``Extent`` SDS is not its grid's declared shape.

        Pixel coordinates are computed from the declared grid, so an SDS of another shape cannot be geolocated and would
        otherwise misplace every pixel or fail obscurely.

        A 4 x 4 granule is read with the default 721 x 721 Northern grid, and the error message must name both shapes.
        """
        # The default 721 x 721 grid declared for a 4 x 4 Extent SDS.
        reader = NISEReader(_granule(tmp_path), hemispheres=(_HemisphereGrid(3408, _HEMISPHERE_LABEL_NORTH),))
        with pytest.raises(ValueError, match=r"Extent SDS has shape \(4, 4\).*721 x 721"):
            reader._load_native()


class TestNISEReaderBothHemispheres:
    """The reader reads and merges both the Northern and Southern EASE-Grids."""

    def test_both_hemispheres_are_concatenated(self, tmp_path):
        """``NISEReader._load_native`` merges both hemispheres' pixels into one point cloud.

        Footprints over either pole draw from the same cloud, so a hemisphere left out of the merge would leave its
        footprints without data.

        A granule with code 50 in both 4 x 4 grids is loaded; the cloud must hold latitudes on both sides of the
        equator, 32 points and a ``(5, 32)`` value array.
        """
        path = _granule(tmp_path, north=_filled(50), south=_filled(50))
        points = NISEReader(path, hemispheres=(_TEST_HEMI_NORTH, _TEST_HEMI_SOUTH))._load_native()
        # The point cloud spans both hemispheres and keeps every pixel (no code 254 to drop).
        assert np.any(points.lats > 0)
        assert np.any(points.lats < 0)
        expected = 2 * _TEST_ROWS * _TEST_COLS
        assert points.lats.size == points.lons.size == expected
        assert points.values.shape == (len(NISEReader.VARIABLES), expected)

    def test_southern_box_is_served_from_the_southern_grid(self, tmp_path):
        """A box over the Southern grid is served the Southern hemisphere's values, not the Northern's.

        This guards the end-to-end path for the hemisphere that name-based SDS selection would miss: a Southern
        footprint must see Antarctic data.

        A granule with open ocean in the North and code 50 in the South is read with only the Southern grid and queried
        over that grid; all 16 pixels must come back with 0.5 sea ice concentration.
        """
        path = _granule(tmp_path, north=_filled(255), south=_filled(50))
        reader = NISEReader(path, hemispheres=(_TEST_HEMI_SOUTH,))
        values = reader.samples_in(_whole_grid_box(reader), _T).values
        assert values.shape[1] == _TEST_ROWS * _TEST_COLS
        assert np.allclose(values[_layer_index("sea_ice_concentration")], 0.5, atol=1e-5)

    def test_off_earth_code_254_is_dropped_not_zeroed(self, tmp_path):
        """``NISEReader._load_native`` removes pixels with Extent code 254 from the point cloud.

        Code 254 marks off-Earth grid corners that reproject into the opposite hemisphere; kept as all-zero layers they
        would dilute the class fractions of that hemisphere's footprints.

        Two corners of a 4 x 4 Northern grid of code 50 are set to 254, and the loaded cloud must hold 14 points and 14
        value columns.
        """
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
        """On the real granule, ``_read_extent_sds`` returns a different 721 x 721 grid for each hemisphere label.

        This confirms hemisphere selection by dimension label works on the layout NSIDC distributes, not only on the
        written fixture.

        Both labels are read; the arrays must have the full grid shape and must differ.
        """
        north = reader._read_extent_sds(_HEMISPHERE_LABEL_NORTH)
        south = reader._read_extent_sds(_HEMISPHERE_LABEL_SOUTH)
        assert north.shape == south.shape == (721, 721)
        # Two genuinely different grids, not the same SDS resolved twice.
        assert not np.array_equal(north, south)

    def test_point_cloud_spans_both_hemispheres_without_corner_fill(self, reader):
        """On the real granule, ``NISEReader._load_native`` yields both hemispheres less the off-Earth corners.

        The point count pins the number of code 254 corner pixels dropped from the two full grids, and the latitude
        extent shows that both polar regions are covered.

        The real granule is loaded; the cloud must hold 811,786 points with a maximum latitude above 89.9 and a minimum
        below -89.9.
        """
        points = reader._load_native()
        # 2 x 721 x 721 pixels less the off-Earth corners (code 254).
        assert points.lats.size == 811786
        assert points.lats.max() > 89.9
        assert points.lats.min() < -89.9

    @pytest.mark.parametrize("hemi", _DEFAULT_HEMISPHERES, ids=lambda h: h.hemisphere_label)
    def test_every_non_finite_pixel_is_off_earth(self, reader, hemi):
        """On the real granule, every pixel that reprojects to a non-finite coordinate carries code 254.

        The reader drops corner pixels by their code, so this confirms that the code-based drop also covers the pixels
        the projection cannot place.

        The parameters run each default hemisphere grid; the Extent codes at the non-finite latitudes must all be 254.
        """
        # The reader drops off-Earth pixels by their code, which covers every non-finite one.
        raw = reader._read_extent_sds(hemi.hemisphere_label)
        lats_2d, _ = reader._compute_latlon_grid(hemi)
        assert np.all(raw[~np.isfinite(lats_2d)] == 254)

    def test_surface_classes_at_known_places(self, reader):
        """On the real granule, queries at three known places return the expected surface classes.

        This ties the code-to-layer split and the reprojection together on real data: a wrong mapping or misplaced grid
        would put the wrong class at a recognizable location.

        Boxes at the South Pole, 85 degrees north on the prime meridian and near Boulder, Colorado must return permanent
        ice, 0.93 sea ice concentration and snow-free land respectively; the expected values are the source granule's
        own.
        """
        layer = {name: _layer_index(name) for name in _EXPECTED_VARIABLES}
        south_pole = reader.samples_in(BoundingBox(-90.0, -89.5, -180.0, 180.0), self._TIME).values
        assert np.all(south_pole[layer["permanent_ice"]] == 1.0)
        arctic = reader.samples_in(BoundingBox(84.9, 85.1, -0.3, 0.3), self._TIME).values
        assert np.allclose(arctic[layer["sea_ice_concentration"]], 0.93)
        boulder = reader.samples_in(BoundingBox(39.9, 40.1, -105.3, -104.7), self._TIME).values
        assert np.all(boulder[layer["snow_free_land"]] == 1.0)

    def test_suspect_ice_at_the_north_pole_is_missing_in_every_layer(self, reader):
        """On the real granule, the North Pole pixels are served, but as NaN in every layer.

        The AMSR-2 pole hole is coded 253 (suspect ice value), which belongs to no class; serving it as zeros would
        report open conditions at the pole instead of missing data.

        A box from 89.5 degrees north to the pole is queried; it must return values, and all of them must be NaN.
        """
        # The AMSR-2 pole hole is coded 253 (suspect ice value), which belongs to no class.
        north_pole = reader.samples_in(BoundingBox(89.5, 90.0, -180.0, 180.0), self._TIME).values
        assert north_pole.size > 0
        assert np.isnan(north_pole).all()
