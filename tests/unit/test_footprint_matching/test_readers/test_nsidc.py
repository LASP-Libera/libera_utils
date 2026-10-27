"""Unit tests for NISEReader.

The tests write small NISE HDF-EOS4 granules with pyhdf (both hemispheres, the
duplicate ``Extent``/``Age`` SDS names and the hemisphere dimension names of a real
granule) and read them back on a coarse 4 x 4 EASE-Grid, so the hemisphere
selection, the pyproj EPSG:3408/3409 -> EPSG:4326 reprojection and the code-to-layer
split all run for real. ``TestNISEReaderRealSubset`` runs the reader on a real NISE_A2
granule.

Real NISE files can be downloaded from:
    NSIDC HTTPS: https://n5eil01u.ecs.nsidc.org/NISE/
    Earthdata login required: https://urs.earthdata.nasa.gov/
"""

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
from libera_utils.footprint_matching.types import BoundingBox, SampleSet
from tests.plugins.footprint_matching_fixtures import make_nise_hdfeos_fixture

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
# The test grid's corners, as its granule's StructMetadata.0 states them.
_TEST_UPPER_LEFT_M = (_TEST_X_ORIGIN, _TEST_Y_ORIGIN)
_TEST_LOWER_RIGHT_M = (
    _TEST_X_ORIGIN + _TEST_COLS * _TEST_RESOLUTION_M,
    _TEST_Y_ORIGIN - _TEST_ROWS * _TEST_RESOLUTION_M,
)


def _filled(code: int) -> np.ndarray:
    return np.full((_TEST_ROWS, _TEST_COLS), code, dtype=np.uint8)


def _granule(tmp_path: Path, north: np.ndarray | None = None, south: np.ndarray | None = None, **metadata) -> Path:
    """Write a 4 x 4 NISE granule on the test grid; a hemisphere not given is all ocean (255).

    ``metadata`` overrides the ``StructMetadata.0`` keywords of ``make_nise_hdfeos_fixture``.
    """
    corners = {"upper_left_m": _TEST_UPPER_LEFT_M, "lower_right_m": _TEST_LOWER_RIGHT_M}
    return make_nise_hdfeos_fixture(
        tmp_path / "NISE_AMSR2_20260101.HDFEOS",
        north=_filled(255) if north is None else north,
        south=_filled(255) if south is None else south,
        **{**corners, **metadata},
    )


@pytest.fixture
def north_test_grid(monkeypatch):
    """Read only the coarse 4 x 4 Northern test grid in place of the two 721 x 721 EASE-Grids."""
    monkeypatch.setattr(NISEReader, "_HEMISPHERES", (_TEST_HEMI_NORTH,))


def _make_reader(tmp_path: Path, north: np.ndarray | None = None) -> NISEReader:
    """A NISEReader over a written 4 x 4 granule (use with ``north_test_grid``)."""
    return NISEReader(_granule(tmp_path, north=north))


def _whole_grid_box(reader: NISEReader) -> BoundingBox:
    lats_2d, lons_2d = reader._compute_latlon_grid(reader._HEMISPHERES[0])
    return BoundingBox(
        float(lats_2d.min()) - 0.1,
        float(lats_2d.max()) + 0.1,
        float(lons_2d.min()) - 0.1,
        float(lons_2d.max()) + 0.1,
    )


class TestNISEReaderLatLonGrid:
    @pytest.mark.parametrize(
        ("hemi", "sign"),
        [pytest.param(_TEST_HEMI_NORTH, 1.0, id="north"), pytest.param(_TEST_HEMI_SOUTH, -1.0, id="south")],
    )
    def test_grid_reprojects_into_its_hemisphere(self, hemi, sign):
        """``_compute_latlon_grid`` returns one in-range latitude and longitude per pixel, in the grid's own hemisphere.

        The reader serves every pixel at its own reprojected center, so a misshapen or out-of-range coordinate array
        would misplace the data; and each hemisphere is reprojected with its own CRS (EPSG:3408 north, 3409 south), so
        the wrong one would place Arctic data over Antarctica or the reverse.

        Parametrized over the coarse 4 x 4 Northern and Southern test grids, which lie near their poles; each is
        reprojected and both arrays must have shape ``(4, 4)``, longitudes in [-180, 180], and every latitude on the
        grid's side of the equator, within [-90, 90].
        """
        lats_2d, lons_2d = NISEReader("unused.HDFEOS")._compute_latlon_grid(hemi)
        assert lats_2d.shape == lons_2d.shape == (_TEST_ROWS, _TEST_COLS)
        assert np.all((sign * lats_2d > 0) & (np.abs(lats_2d) <= 90))
        assert np.all((lons_2d >= -180) & (lons_2d <= 180))

    @pytest.mark.parametrize("hemi", NISEReader._HEMISPHERES, ids=lambda h: h.hemisphere_label)
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


@pytest.mark.usefixtures("north_test_grid")
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


class TestNISEExtentToCategoryMasks:
    # Every Extent code group (the granule's data_grid_key, reproduced in the nsidc module docstring), in one row.
    _CODES = (0, 1, 50, 100, 101, 102, 103, 104, 105, 251, 252, 253, 255)
    # One column per code: sea ice concentration, snow-free land, permanent ice, snow on land, open ocean.
    _EXPECTED = np.array(
        [
            [0.0, 0.01, 0.5, 1.0, 0.0, np.nan, 0.0, 0.0, np.nan, np.nan, np.nan, np.nan, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, np.nan, 0.0, 0.0, np.nan, np.nan, np.nan, np.nan, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, np.nan, 0.0, 0.0, np.nan, np.nan, np.nan, np.nan, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, np.nan, 1.0, 1.0, np.nan, np.nan, np.nan, np.nan, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, np.nan, 0.0, 0.0, np.nan, np.nan, np.nan, np.nan, 1.0],
        ]
    )

    def test_each_code_sets_exactly_its_layers(self):
        """``_extent_to_category_masks`` splits every Extent code group into its five float32 layers and nothing else.

        Codes 1-100 are percent sea ice concentration, scaled to a fraction; 0, 101, 103/104 and 255 each set their own
        0/1 indicator (snow-free land, permanent ice, snow on land, open ocean) and leave every other layer 0, so a
        pixel is counted in one class only; and the unused codes 102 and 105-251 and the 252/253 quality flags are NaN
        in every layer, so the weigher counts them as missing rather than pulling every class fraction toward zero. A
        code routed to the wrong layer would misreport the surface under every footprint containing it.

        Splits one row holding a code from each group, including both ends of the concentration and unused ranges, and
        compares the float32 ``(5, 1, 13)`` stack with the expected table, NaN for NaN.
        """
        masks = NISEReader("unused.HDFEOS")._extent_to_category_masks(np.array([self._CODES], dtype=np.uint8))
        assert masks.dtype == np.float32
        assert masks.shape == (len(_EXPECTED_VARIABLES), 1, len(self._CODES))
        np.testing.assert_allclose(masks[:, 0, :], self._EXPECTED, rtol=0, atol=1e-6, equal_nan=True)


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

        A 4 x 4 granule is read with the reader's 721 x 721 grids, and the error message must name both shapes.
        """
        with pytest.raises(ValueError, match=r"Extent SDS has shape \(4, 4\).*721 x 721"):
            NISEReader(_granule(tmp_path))._load_native()


@pytest.mark.usefixtures("north_test_grid")
class TestNISEGridGeometry:
    """Each hemisphere's ``StructMetadata.0`` grid is checked against the grid it is geolocated on."""

    @pytest.mark.parametrize(
        "metadata",
        [
            pytest.param({"upper_left_m": (-1_100_000.0, 1_000_000.0)}, id="upper-left"),
            pytest.param({"lower_right_m": (1_000_000.0, -1_050_000.0)}, id="lower-right"),
        ],
    )
    def test_other_corners_raise(self, tmp_path, metadata):
        """``NISEReader._load_native`` raises ``ValueError`` when ``StructMetadata.0`` gives other grid corners.

        Pixel centers are computed from the reader's fixed grid, so a granule of the same shape on another extent would
        be geolocated wrongly with no error.

        The parameters move the upper-left or the lower-right corner of an otherwise matching 4 x 4 granule; the load
        must raise a message naming the grid, both sets of corners and what to stage.
        """
        with pytest.raises(ValueError, match=r"grid 'Northern Hemisphere' YDim=4, XDim=4, corners .*stage the 25 km"):
            NISEReader(_granule(tmp_path, **metadata))._load_native()

    def test_other_projection_raises(self, tmp_path):
        """``NISEReader._load_native`` raises ``ValueError`` when a grid names a projection other than ``GCTP_LAMAZ``.

        The pixel centers are reprojected from the EASE-Grid's Lambert azimuthal equal-area CRS, which is wrong for a
        grid in any other projection.

        A granule whose grids name ``GCTP_GEO`` is loaded, and the error must name both projections.
        """
        with pytest.raises(ValueError, match="Projection=GCTP_GEO .* not the EASE-Grid GCTP_LAMAZ"):
            NISEReader(_granule(tmp_path, projection="GCTP_GEO"))._load_native()

    def test_missing_struct_metadata_raises(self, tmp_path):
        """``NISEReader._load_native`` raises ``ValueError`` when the granule has no ``StructMetadata.0`` grid.

        Without the attribute the reader cannot confirm the grid it geolocates on, and a distributed NISE granule always
        carries it.

        A granule written without ``StructMetadata.0`` is loaded, and the error must name the missing grid.
        """
        with pytest.raises(ValueError, match="has no grid named 'Northern Hemisphere'"):
            NISEReader(_granule(tmp_path, struct_metadata=False))._load_native()


class TestNISEReaderBothHemispheres:
    """The reader reads and merges both the Northern and Southern EASE-Grids."""

    def test_both_hemispheres_are_concatenated(self, tmp_path, monkeypatch):
        """``NISEReader._load_native`` merges both hemispheres' pixels into one point cloud.

        Footprints over either pole draw from the same cloud, so a hemisphere left out of the merge would leave its
        footprints without data.

        A granule with code 50 in both 4 x 4 grids is loaded; the cloud must hold latitudes on both sides of the
        equator, 32 points and a ``(5, 32)`` value array.
        """
        monkeypatch.setattr(NISEReader, "_HEMISPHERES", (_TEST_HEMI_NORTH, _TEST_HEMI_SOUTH))
        points = NISEReader(_granule(tmp_path, north=_filled(50), south=_filled(50)))._load_native()
        # The point cloud spans both hemispheres and keeps every pixel (no code 254 to drop).
        assert np.any(points.lats > 0)
        assert np.any(points.lats < 0)
        expected = 2 * _TEST_ROWS * _TEST_COLS
        assert points.lats.size == points.lons.size == expected
        assert points.values.shape == (len(NISEReader.VARIABLES), expected)

    def test_southern_box_is_served_from_the_southern_grid(self, tmp_path, monkeypatch):
        """A box over the Southern grid is served the Southern hemisphere's values, not the Northern's.

        This guards the end-to-end path for the hemisphere that name-based SDS selection would miss: a Southern
        footprint must see Antarctic data.

        A granule with open ocean in the North and code 50 in the South is read with only the Southern grid and queried
        over that grid; all 16 pixels must come back with 0.5 sea ice concentration.
        """
        monkeypatch.setattr(NISEReader, "_HEMISPHERES", (_TEST_HEMI_SOUTH,))
        reader = NISEReader(_granule(tmp_path, north=_filled(255), south=_filled(50)))
        values = reader.samples_in(_whole_grid_box(reader), _T).values
        assert values.shape[1] == _TEST_ROWS * _TEST_COLS
        assert np.allclose(values[_layer_index("sea_ice_concentration")], 0.5, atol=1e-5)

    @pytest.mark.usefixtures("north_test_grid")
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

    @pytest.mark.parametrize("hemi", NISEReader._HEMISPHERES, ids=lambda h: h.hemisphere_label)
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
