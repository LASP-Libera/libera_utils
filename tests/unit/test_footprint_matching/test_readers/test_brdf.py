"""Unit tests for VIIRSBRDFReader and the VJ143C1 HDF5 read.

Uses synthetic VJ143C1 HDF5 fixture files created by
``make_viirs_brdf_hdf5_fixture``, laid out like a real granule: cell-edge
``lat``/``lon`` axes, an HDF-EOS5 ``StructMetadata.0`` grid definition and the
per-field fill, valid-range, scale and offset attributes.

Real VJ143C1 files can be downloaded from:
    LP DAAC: https://e4ftl01.cr.usgs.gov/VIIRS/VJ143C1.002/
    EarthData login required: https://urs.earthdata.nasa.gov/
    Example: VJ143C1.A2026153.002.2026161161054.h5
"""

import tracemalloc

import h5py
import numpy as np
import pytest

from libera_utils.footprint_matching.readers._hdf5_io import _parse_he5_grid_metadata
from libera_utils.footprint_matching.readers.brdf import VIIRSBRDFReader
from libera_utils.footprint_matching.types import BoundingBox
from tests.test_data.footprint_matching.fixtures import make_viirs_brdf_hdf5_fixture
from tests.unit.test_footprint_matching.test_readers.grid_helpers import region as _region
from tests.unit.test_footprint_matching.test_readers.grid_helpers import sample_at

_T = np.datetime64("2026-01-01T00:00:00", "ns")

# Default fixture grid: a tiny region of 0.05 degree cells, given by its outer cell edges.
_N_LAT = 4
_N_LON = 8
_LAT_MIN = 0.0
_LAT_MAX = 0.20
_LON_MIN = 10.0
_LON_MAX = 10.40
_HALF_CELL = 0.025
_PARAM_FILL = 200  # raw int16 → 0.200 after scale_factor=0.001
_FILL_SENTINEL = 32767  # last pixel; should become NaN
_FIELDS = "HDFEOS/GRIDS/VIIRS_CMG_BRDF/Data Fields"


def _fixture(tmp_path, **kwargs):
    return make_viirs_brdf_hdf5_fixture(
        tmp_path,
        n_lat=_N_LAT,
        n_lon=_N_LON,
        lat_min=_LAT_MIN,
        lat_max=_LAT_MAX,
        lon_min=_LON_MIN,
        lon_max=_LON_MAX,
        param_fill=_PARAM_FILL,
        fill_sentinel=_FILL_SENTINEL,
        **kwargs,
    )


def _make_reader(tmp_path, **kwargs) -> VIIRSBRDFReader:
    return VIIRSBRDFReader(_fixture(tmp_path, **kwargs))


def _full_bbox() -> BoundingBox:
    return BoundingBox(_LAT_MIN - 0.01, _LAT_MAX + 0.01, _LON_MIN - 0.01, _LON_MAX + 0.01)


class TestVIIRSBRDFReaderClassAttributes:
    def test_reader_key(self):
        """``VIIRSBRDFReader.READER_KEY`` is ``"viirs_brdf"``.

        The key is how the reader registry finds this reader, and it names the source on every served sample set and
        prefixes every output variable name, so a changed key breaks lookups and renames the product's variables.

        Compares the class attribute with the literal string.
        """
        assert VIIRSBRDFReader.READER_KEY == "viirs_brdf"

    def test_resolution_km(self):
        """``VIIRSBRDFReader.RESOLUTION_KM`` is 5.6 km, the spacing of the 0.05 degree CMG at the equator.

        The declared resolution sets how far ``samples_in`` widens each query box and how the weigher sizes each
        sample's coverage, so a wrong value over- or under-credits BRDF samples in a footprint.

        Compares the class attribute with 5.6.
        """
        assert VIIRSBRDFReader.RESOLUTION_KM == 5.6

    def test_variables_count_is_nine(self):
        """``VIIRSBRDFReader.VARIABLES`` declares nine variables, three RTLS kernel weights for each of three bands.

        The variable count must match the nine planes the reader loads; the base class rejects a load whose plane count
        differs from ``VARIABLES``.

        Checks the length of the class attribute.
        """
        assert len(VIIRSBRDFReader.VARIABLES) == 9

    def test_variable_names_cover_three_bands_three_params(self):
        """``VIIRSBRDFReader.VARIABLES`` has a ``brdf_{band}_{param}`` name for every band and kernel-weight pair.

        Downstream configuration and the product definition refer to these variables by name, so a missing or misspelled
        name drops that BRDF parameter from the product.

        Loops over the bands ``shortwave``, ``vis`` and ``nir`` and the weights ``fiso``, ``fvol`` and ``fgeo`` and
        asserts each expected name is declared.
        """
        names = {v.name for v in VIIRSBRDFReader.VARIABLES}
        for band in ("shortwave", "vis", "nir"):
            for param in ("fiso", "fvol", "fgeo"):
                assert f"brdf_{band}_{param}" in names

    def test_all_variables_are_float32_weighted_mean(self):
        """Every ``VIIRSBRDFReader`` variable is a continuous float32 field aggregated by ``weighted_mean``.

        BRDF kernel weights are continuous quantities, so they are averaged over a footprint; declaring them categorical
        or giving them another strategy would aggregate them as the wrong kind of quantity.

        Asserts ``dtype``, ``aggregation`` and ``n_categories is None`` on each declared variable.
        """
        for v in VIIRSBRDFReader.VARIABLES:
            assert v.dtype == "float32"
            assert v.aggregation == "weighted_mean"
            assert v.n_categories is None


class TestVIIRSBRDFReaderScaleAndFill:
    """Verify scale_factor application and fill → NaN conversion."""

    def test_scale_factor_applied_to_param_fill(self, tmp_path):
        """The reader multiplies each raw int16 count by the field's ``scale_factor``.

        The file stores kernel weights as scaled integers; serving the raw counts would make every weight a thousand
        times too large.

        The fixture writes raw count 200 with ``scale_factor`` 0.001 everywhere but one fill pixel; every non-NaN value
        served for the whole region must be 0.200.
        """
        # param_fill=200 with scale_factor=0.001 → 0.2
        reader = _make_reader(tmp_path)
        data, _, _ = _region(reader, _full_bbox(), _T)
        # All non-fill pixels should equal 0.200
        non_fill = data[~np.isnan(data)]
        assert non_fill.size > 0
        assert np.allclose(non_fill, 0.200, atol=1e-4)

    def test_fill_sentinel_becomes_nan(self, tmp_path):
        """A pixel holding the field's ``_FillValue`` is served as NaN, at its position after the latitude flip.

        A fill value left in place would be scaled into a plausible-looking weight of 32.767 and averaged into
        footprints; landing it in the wrong row would mask a valid cell and keep the fill.

        The fixture writes the fill to the last (southernmost) row and column of each field in the descending file; the
        served region must be NaN at output row 0, last column, for every variable, and nowhere else.
        """
        # The fixture writes fill_sentinel to file position [-1, -1] — the lat_min row of
        # the DESCENDING file. The reader flips lats to ascending, moving that row to
        # output row 0, so the fill lands at [:, 0, -1] (one fill pixel per band).
        reader = _make_reader(tmp_path)
        data, lats, lons = _region(reader, _full_bbox(), _T)
        assert np.isnan(data[:, 0, -1]).all()
        assert np.isnan(data).sum() == data.shape[0]

    def test_non_fill_pixels_are_not_nan(self, tmp_path):
        """Fill handling in ``VIIRSBRDFReader`` leaves valid pixels as numbers.

        A fill or valid-range test that masked everything would publish an all-fill BRDF product with no error.

        Loads the default fixture's full region and asserts at least one value is not NaN.
        """
        reader = _make_reader(tmp_path)
        data, _, _ = _region(reader, _full_bbox(), _T)
        # At least one pixel (all except [-1, -1]) should be non-NaN.
        assert not np.all(np.isnan(data))


class TestVIIRSBRDFReaderLoadSpatialRegion:
    def test_returns_3d_data_array(self, tmp_path):
        """A region served by ``VIIRSBRDFReader`` folds back to a ``(9, n_lat, n_lon)`` array, one plane per variable.

        Consumers index the variable axis by ``VARIABLES`` order, so a missing or extra plane misattributes every BRDF
        parameter.

        Queries the full fixture region through ``samples_in`` and checks the folded array's rank and first dimension.
        """
        reader = _make_reader(tmp_path)
        data, lats, lons = _region(reader, _full_bbox(), _T)
        assert data.ndim == 3
        assert data.shape[0] == 9

    def test_data_dtype_is_float32(self, tmp_path):
        """``VIIRSBRDFReader`` serves its values as float32, as ``VARIABLES`` declares.

        The nine global fields are held in memory at once, so float32 keeps the load near 0.9 GB, and the served type
        must agree with the declared dtype.

        Queries the full fixture region and checks the dtype of the values.
        """
        reader = _make_reader(tmp_path)
        data, _, _ = _region(reader, _full_bbox(), _T)
        assert data.dtype == np.float32

    def test_lat_is_ascending(self, tmp_path):
        """``VIIRSBRDFReader`` serves latitude ascending, though the file stores it descending.

        Every reader serves latitude ascending; a reader that kept the file's north-to-south order would pair each value
        with the wrong row of cells.

        Queries the full fixture region and asserts the latitudes increase and the first one is the southernmost cell
        center, half a cell above the region's southern edge.
        """
        # The fixture stores lats in DESCENDING order; the reader must flip them
        # to ascending, so output row 0 is the southernmost cell center.
        reader = _make_reader(tmp_path)
        _, lats, _ = _region(reader, _full_bbox(), _T)
        assert lats.size > 1
        assert np.all(np.diff(lats) >= 0), f"lats not ascending: {lats}"
        assert np.isclose(lats[0], _LAT_MIN + _HALF_CELL), f"row 0 should be the southern cell center: {lats}"

    def test_spatial_shape_matches_fixture(self, tmp_path):
        """A box covering the whole fixture grid returns every one of its cells.

        A box that covers the grid must not drop rows or columns at the edges, or footprints near a cell boundary lose
        samples.

        Queries a box just wider than the fixture's 4 x 8 grid and checks the folded array has 4 latitude rows and 8
        longitude columns.
        """
        reader = _make_reader(tmp_path)
        data, lats, lons = _region(reader, _full_bbox(), _T)
        assert data.shape[1] == _N_LAT
        assert data.shape[2] == _N_LON

    def test_empty_result_outside_bbox(self, tmp_path):
        """A box that misses the grid returns no samples but keeps the nine-variable axis.

        A footprint outside the BRDF coverage must yield an empty result rather than an error, and its values array
        still needs the variable axis so it combines with other results.

        Queries a box far from the fixture region and asserts the folded array is empty with first dimension 9.
        """
        reader = _make_reader(tmp_path)
        bbox = BoundingBox(-60.0, -58.0, 170.0, 172.0)
        data, lats, lons = _region(reader, bbox, _T)
        assert data.size == 0
        assert data.shape[0] == 9

    def test_partial_bbox_subsets_correctly(self, tmp_path):
        """A box covering part of the grid returns exactly the cells whose centers fall in the padded box.

        Serving cells beyond the box would average distant surfaces into the footprint's BRDF, and dropping cells
        inside it would lose part of the footprint.

        Queries every row but only longitudes 10.11 to 10.24, padded by about 0.025 degree, and asserts all four
        latitude rows and exactly the three cell-center columns 10.125, 10.175 and 10.225 come back, so both
        longitude edges are cut where the padding ends.
        """
        reader = _make_reader(tmp_path)
        bbox = BoundingBox(_LAT_MIN - 0.01, _LAT_MAX + 0.01, 10.11, 10.24)
        data, lats, lons = _region(reader, bbox, _T)
        centers_lon = _LON_MIN + _HALF_CELL + 2 * _HALF_CELL * np.arange(_N_LON)
        centers_lat = _LAT_MIN + _HALF_CELL + 2 * _HALF_CELL * np.arange(_N_LAT)
        np.testing.assert_allclose(lons, centers_lon[2:5])
        np.testing.assert_allclose(lats, centers_lat)

    def test_load_peak_memory_is_the_held_fields(self, tmp_path):
        """Loading the nine BRDF fields peaks below one and a half times the size of the array held afterwards.

        A global VJ143C1 load holds about 0.9 GB; reading each field into its own array and stacking them would at least
        double the peak and can exhaust the memory of the machine running it.

        Writes a 400 x 800 fixture, traces allocations while ``_native`` loads it, and compares the peak with the loaded
        data's byte count.
        """
        path = make_viirs_brdf_hdf5_fixture(
            tmp_path, n_lat=400, n_lon=800, lat_min=0.0, lat_max=20.0, lon_min=10.0, lon_max=50.0
        )
        reader = VIIRSBRDFReader(path)
        tracemalloc.start()
        try:
            native = reader._native
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        # The nine fields are read into one preallocated array; stacking per-field arrays would
        # hold at least twice the result.
        assert peak < 1.5 * native.data.nbytes


class TestVIIRSBRDFReaderSamples:
    def test_samples_carry_source_and_no_timestamp_source(self, tmp_path):
        """Samples from ``VIIRSBRDFReader`` name ``viirs_brdf`` as their source and carry no timestamp source.

        BRDF is a static surface product, so it has no radiometer or camera time to record, and the source name is what
        attributes the samples to this reader.

        Queries the full fixture region and checks ``source``, ``timestamp_source`` and the nine-variable values axis.
        """
        # BRDF is a static surface property product; no instrument timestamp source.
        reader = _make_reader(tmp_path)
        samples = reader.samples_in(_full_bbox(), _T)
        assert samples.source == "viirs_brdf"
        assert samples.timestamp_source is None
        assert samples.values.shape[0] == 9

    def test_granule_is_read_once_for_every_box(self, tmp_path):
        """``VIIRSBRDFReader`` reads the granule on the first query and serves every later box from memory.

        A global granule is about 360 MB; reopening it for each footprint box would repeat that read for every
        footprint.

        Queries one box, deletes the file, queries a second box, and asserts the second box still returns the fixture's
        0.2 values and the cached native object is unchanged.
        """
        path = _fixture(tmp_path)
        reader = VIIRSBRDFReader(path)
        first = reader.samples_in(BoundingBox(0.0, 0.1, 10.0, 10.1), _T)
        native = reader._native
        # With the file gone, a second box can only be served from what the first load holds.
        path.unlink()
        second = reader.samples_in(BoundingBox(0.1, 0.2, 10.2, 10.4), _T)
        assert first.n_samples > 0
        assert second.n_samples > 0
        assert np.allclose(second.values[np.isfinite(second.values)], 0.2)
        assert reader._native is native


class TestVIIRSBRDFCellCenters:
    """The file's lat/lon are cell edges; samples are served at cell centers."""

    def test_axes_are_cell_centers(self, tmp_path):
        """The reader turns the file's cell-edge ``lat``/``lon`` axes into cell centers.

        The file gives each cell's upper-left edge; serving edges as sample locations would shift every BRDF value half
        a cell (about 2.8 km) north and west.

        Loads the default fixture, whose edges start at 0.20 N and 10.0 E, and asserts the served axes are the centers
        0.025 to 0.175 N and 10.025 E onward in 0.05 degree steps.
        """
        native = _make_reader(tmp_path)._native
        np.testing.assert_allclose(native.lat_axis, [0.025, 0.075, 0.125, 0.175])
        np.testing.assert_allclose(native.lon_axis, 10.025 + 0.05 * np.arange(_N_LON))

    def test_global_cmg_edges_become_the_expected_centers(self, tmp_path):
        """A global grid stored as the real CMG stores it is served as ascending cell centers.

        The real file's ``lat`` runs from 90 down and its ``lon`` from -180, so its axes start on the pole and the
        antimeridian; mishandling the direction or the half-cell offset there would put the first and last rows outside
        the globe or on the wrong side of the dateline.

        Writes a 1 degree global fixture, confirms its stored axes start at 90 and -180, and asserts the served axes run
        -89.5 to 89.5 and -179.5 to 179.5.
        """
        # A coarse global CMG laid out as the real file stores it: lat 90 -> -89, lon -180 -> 179.
        path = make_viirs_brdf_hdf5_fixture(
            tmp_path, n_lat=180, n_lon=360, lat_min=-90.0, lat_max=90.0, lon_min=-180.0, lon_max=180.0
        )
        with h5py.File(path, "r") as f:
            assert f["HDFEOS/GRIDS/VIIRS_CMG_BRDF/Data Fields/lat"][0] == 90.0
            assert f["HDFEOS/GRIDS/VIIRS_CMG_BRDF/Data Fields/lon"][0] == -180.0
        native = VIIRSBRDFReader(path)._native
        assert native.lat_axis[0] == pytest.approx(-89.5)
        assert native.lat_axis[-1] == pytest.approx(89.5)
        assert native.lon_axis[0] == pytest.approx(-179.5)
        assert native.lon_axis[-1] == pytest.approx(179.5)


class TestVIIRSBRDFFieldAttributes:
    @pytest.mark.parametrize("attribute", ["_FillValue", "valid_range", "scale_factor", "add_offset"])
    def test_missing_field_attribute_raises(self, tmp_path, attribute):
        """A BRDF field missing one of its four decoding attributes raises ``ValueError`` on load.

        Every VJ143C1 field carries ``_FillValue``, ``valid_range``, ``scale_factor`` and ``add_offset``; without any
        one the reader cannot tell fill from data or turn counts into weights; guessing a default would publish wrong
        values silently.

        Parametrized over those four attributes; each case writes a fixture without that attribute and asserts the load
        raises with a message naming it.
        """
        reader = _make_reader(tmp_path, omit_attributes=(attribute,))
        with pytest.raises(ValueError, match=f"has no {attribute} attribute"):
            _ = reader._native

    @pytest.mark.parametrize("valid_range", [(0, 100), (300, 400)])
    def test_values_outside_the_valid_range_are_nan(self, tmp_path, valid_range):
        """Raw values outside a field's ``valid_range`` are served as NaN.

        Counts outside the product's valid range are not usable kernel weights, so they must be masked rather than
        scaled into numbers that look real.

        Parametrized over a range entirely below and one entirely above the fixture's raw value 200; in both cases every
        loaded value must be NaN.
        """
        # The fixture's raw value 200 lies above the first range and below the second.
        reader = _make_reader(tmp_path, valid_range=valid_range)
        assert np.all(np.isnan(reader._native.data))

    def test_add_offset_is_applied(self, tmp_path):
        """The reader adds each field's ``add_offset`` after scaling, and only to that field.

        VJ143C1 writes an offset of 0, but a field with a nonzero offset would otherwise be served shifted by that
        offset.

        Sets ``add_offset`` to 0.5 on the first shortwave field only and asserts that field reads 0.7 and the next field
        still reads 0.2.
        """
        path = _fixture(tmp_path)
        with h5py.File(path, "r+") as f:
            f["HDFEOS/GRIDS/VIIRS_CMG_BRDF/Data Fields/BRDF_Albedo_Parameter1_shortwave"].attrs["add_offset"] = [0.5]
        data = VIIRSBRDFReader(path)._native.data
        assert np.allclose(data[0][np.isfinite(data[0])], 0.7)
        assert np.allclose(data[1][np.isfinite(data[1])], 0.2)

    def test_each_field_uses_its_own_attribute(self, tmp_path):
        """Each BRDF field is decoded with its own ``_FillValue`` and ``scale_factor``, not those of another field.

        The product gives every field its own attributes; reusing one field's attributes for all nine would mis-scale or
        mis-mask any field that differs.

        Sets the first field's ``_FillValue`` to the fixture's data value and the second field's ``scale_factor`` to
        0.01; the first field must be all NaN, the second read 2.0, and the third keep 0.2.
        """
        path = _fixture(tmp_path)
        with h5py.File(path, "r+") as f:
            f[f"{_FIELDS}/BRDF_Albedo_Parameter1_shortwave"].attrs["_FillValue"] = np.array([_PARAM_FILL], np.int16)
            f[f"{_FIELDS}/BRDF_Albedo_Parameter2_shortwave"].attrs["scale_factor"] = [0.01]
        data = VIIRSBRDFReader(path)._native.data
        assert np.isnan(data[0]).all()
        assert np.allclose(data[1][np.isfinite(data[1])], 2.0)
        assert np.allclose(data[2][np.isfinite(data[2])], 0.2)


class TestVIIRSBRDFGridMetadata:
    def test_real_cmg_corners_decode_from_packed_dms(self):
        """``_parse_he5_grid_metadata`` decodes the real VJ143C1 grid definition's packed-degree corners and dimensions.

        The reader checks its axes against these corners, so a mis-decoded corner would reject every real granule or
        accept a misplaced grid.

        Parses ``StructMetadata.0`` text with the real CMG's 7200 x 3600 size and packed ``DDDMMMSSS.SS`` corners and
        asserts the dimensions and the corners -180, 90, 180, -90 degrees.
        """
        text = (
            'GROUP=GridStructure\n\tGROUP=GRID_1\n\t\tGridName="VIIRS_CMG_BRDF"\n\t\tXDim=7200\n\t\tYDim=3600\n'
            "\t\tUpperLeftPointMtrs=(-180000000.000000,90000000.000000)\n"
            "\t\tLowerRightMtrs=(180000000.000000,-90000000.000000)\n"
            "\t\tProjection=HE5_GCTP_GEO\n\tEND_GROUP=GRID_1\nEND_GROUP=GridStructure\n"
        )
        grid = _parse_he5_grid_metadata(text, "VIIRS_CMG_BRDF", "granule.h5")
        assert (grid["xdim"], grid["ydim"]) == (7200, 3600)
        assert (grid["ul_lon"], grid["ul_lat"], grid["lr_lon"], grid["lr_lat"]) == (-180.0, 90.0, 180.0, -90.0)

    def test_missing_grid_raises(self):
        """``_parse_he5_grid_metadata`` raises ``ValueError`` when ``StructMetadata.0`` has no grid of the given name.

        A file without the ``VIIRS_CMG_BRDF`` grid is not a VJ143C1 BRDF granule, and the reader has no grid definition
        to check its axes against.

        Parses a ``GridStructure`` group with no grids and checks the message names the missing grid.
        """
        with pytest.raises(ValueError, match="has no grid named 'VIIRS_CMG_BRDF'"):
            _parse_he5_grid_metadata("GROUP=GridStructure\nEND_GROUP=GridStructure\n", "VIIRS_CMG_BRDF", "g.h5")

    def test_missing_field_raises(self):
        """``_parse_he5_grid_metadata`` raises ``ValueError`` when the grid's definition lacks a required field.

        Without ``YDim`` the reader cannot confirm the latitude axis has the grid's size.

        Parses a grid block that has ``XDim`` and ``Projection`` but no ``YDim`` and checks the message names ``YDim``.
        """
        text = 'GridName="VIIRS_CMG_BRDF"\n\t\tXDim=8\n\t\tProjection=HE5_GCTP_GEO\n\tEND_GROUP=GRID_1\n'
        with pytest.raises(ValueError, match="has no YDim field"):
            _parse_he5_grid_metadata(text, "VIIRS_CMG_BRDF", "g.h5")

    def test_non_geographic_projection_raises(self, tmp_path):
        """Loading a grid whose ``Projection`` is not ``HE5_GCTP_GEO`` raises ``ValueError``.

        The reader serves the file's ``lat``/``lon`` as a regular latitude/longitude grid, which holds only for a
        geographic projection; a projected grid would be served at the wrong locations.

        Writes a fixture declaring the sinusoidal ``HE5_GCTP_SNSOID`` projection and asserts the load raises.
        """
        reader = _make_reader(tmp_path, projection="HE5_GCTP_SNSOID")
        with pytest.raises(ValueError, match="not the geographic HE5_GCTP_GEO"):
            _ = reader._native

    def test_axes_not_spanning_the_corners_raise(self, tmp_path):
        """Load raises ``ValueError`` when the ``lat``/``lon`` cell edges do not span the ``StructMetadata.0`` corners.

        The edge-to-center conversion is right only if the axes are the cell edges the corners describe; a file whose
        axes are offset from its corners, for example axes written as cell centers, would be served half a cell off.

        Writes corners half a cell west and north of the fixture's edges, as if its axes held centers, and asserts the
        load raises.
        """
        # Axes written as cell centers (the corners say the region starts half a cell earlier).
        reader = _make_reader(
            tmp_path, metadata_corners=(_LON_MIN - _HALF_CELL, _LAT_MAX + _HALF_CELL, _LON_MAX, _LAT_MIN)
        )
        with pytest.raises(ValueError, match="does not span the StructMetadata corners"):
            _ = reader._native

    def test_non_uniform_axis_raises(self, tmp_path):
        """Loading raises ``ValueError`` when a coordinate axis is not uniformly spaced.

        The reader serves a regular grid and derives each cell center from one cell size; an irregular axis means the
        file is not the CMG grid and its centers would be wrong.

        Moves one latitude edge by 0.01 degree in the fixture and asserts the load raises naming the ``lat`` axis.
        """
        path = _fixture(tmp_path)
        with h5py.File(path, "r+") as f:
            lat = f["HDFEOS/GRIDS/VIIRS_CMG_BRDF/Data Fields/lat"]
            lat[1] = lat[1] + 0.01
        with pytest.raises(ValueError, match="lat axis is not uniformly spaced"):
            _ = VIIRSBRDFReader(path)._native

    def test_axis_length_not_matching_the_grid_raises(self, tmp_path):
        """Loading raises ``ValueError`` when an axis length differs from the grid's ``XDim`` or ``YDim``.

        A mismatch means the axes and the grid definition describe different grids, so the axis check against the
        corners cannot be trusted.

        Rewrites ``StructMetadata.0`` to give ``XDim=9`` for the fixture's 8 longitudes and asserts the message quotes
        ``XDim=9``.
        """
        path = _fixture(tmp_path)
        with h5py.File(path, "r+") as f:
            meta = f["HDFEOS INFORMATION/StructMetadata.0"][()].decode().replace("XDim=8", "XDim=9")
            del f["HDFEOS INFORMATION/StructMetadata.0"]
            f.create_dataset("HDFEOS INFORMATION/StructMetadata.0", data=np.bytes_(meta))
        with pytest.raises(ValueError, match="XDim=9"):
            _ = VIIRSBRDFReader(path)._native

    def test_non_string_struct_metadata_raises(self, tmp_path):
        """Loading raises ``ValueError`` when ``StructMetadata.0`` is not a fixed-length string.

        HDF-EOS5 stores the grid definition as text; any other type means the file is not a distributed granule, and
        decoding it would fail with an unhelpful error.

        Replaces ``StructMetadata.0`` with an integer array and asserts the load raises with the fixed-length string
        message.
        """
        path = _fixture(tmp_path)
        with h5py.File(path, "r+") as f:
            del f["HDFEOS INFORMATION/StructMetadata.0"]
            f.create_dataset("HDFEOS INFORMATION/StructMetadata.0", data=np.arange(3))
        with pytest.raises(ValueError, match="not the fixed-length string"):
            _ = VIIRSBRDFReader(path)._native


class TestVIIRSBRDFFileErrors:
    """h5py's own errors reach the caller.

    A missing file raises ``FileNotFoundError`` (an ``OSError``) and a missing object raises ``KeyError``.
    """

    def test_missing_file_raises_file_not_found_error(self, tmp_path):
        """Loading a path that does not exist raises ``FileNotFoundError``.

        A granule missing from staging must stop the run with an error that says the file is absent rather than produce
        an empty BRDF product.

        Builds a reader on a path under ``tmp_path`` that was never written and asserts the load raises.
        """
        with pytest.raises(FileNotFoundError):
            _ = VIIRSBRDFReader(tmp_path / "absent.h5")._native

    @pytest.mark.parametrize(
        "missing",
        [
            "HDFEOS INFORMATION/StructMetadata.0",
            _FIELDS,
            f"{_FIELDS}/lat",
            f"{_FIELDS}/BRDF_Albedo_Parameter3_nir",
        ],
    )
    def test_missing_dataset_raises_key_error(self, tmp_path, missing):
        """Loading a file missing an object the reader needs raises ``KeyError``.

        A granule without its grid definition, its fields group, a coordinate axis or a BRDF field cannot be served, and
        the run must stop rather than serve partial data.

        Parametrized over ``StructMetadata.0``, the ``Data Fields`` group, the ``lat`` dataset and one BRDF field; each
        case deletes that object from the fixture and asserts the load raises.
        """
        path = _fixture(tmp_path)
        with h5py.File(path, "r+") as f:
            del f[missing]
        with pytest.raises(KeyError):
            _ = VIIRSBRDFReader(path)._native


class TestVIIRSBRDFReaderRealSubset:
    """A real VJ143C1 granule (2026-06-02) cut to 37N..35N, 123W..121W, the Monterey Bay coast.

    The cut keeps the nine fields the reader reads and rewrites the grid's
    ``StructMetadata.0`` corners and dimensions to the window, so the reader's grid
    checks run on it unchanged. About 69% of the window is ocean fill. Pinned values are
    the source granule's own.
    """

    @pytest.fixture
    def reader(self, fmatch_real_data_path):
        return VIIRSBRDFReader(fmatch_real_data_path / "VJ143C1.A2026153.002.2026161161054_subset.h5")

    def test_axes_are_the_window_cell_centers(self, reader):
        """A real VJ143C1 cut loads as a 40 x 40 grid of nine variables with the window's cell centers as axes.

        This exercises the edge-to-center conversion, the latitude flip and the ``StructMetadata.0`` checks on the
        format as distributed rather than on a synthetic layout.

        Loads the Monterey Bay cut and asserts the data shape and the first and last cell centers, 35.025 to 36.975 N
        and -122.975 to -121.025 E.
        """
        native = reader._native
        assert native.data.shape == (len(VIIRSBRDFReader.VARIABLES), 40, 40)
        assert native.lat_axis[0] == pytest.approx(35.025)
        assert native.lat_axis[-1] == pytest.approx(36.975)
        assert native.lon_axis[0] == pytest.approx(-122.975)
        assert native.lon_axis[-1] == pytest.approx(-121.025)

    def test_land_is_scaled_and_ocean_fill_is_nan(self, reader):
        """In a real granule, ocean fill is served as NaN and land values are scaled into the kernel-weight range.

        The real file marks ocean with fill; if fill survived it would appear as 32.767 and corrupt any coastal
        footprint's BRDF.

        Loads the cut's shortwave ``fiso`` plane and asserts the cut's NaN fraction of 0.694375, then that the finite
        values are non-negative and below the scaled fill 32.767.
        """
        fiso = reader._native.data[0]
        assert np.isnan(fiso).mean() == pytest.approx(0.694375)
        finite = fiso[np.isfinite(fiso)]
        # Scaled by 0.001 from int16 counts in valid_range 0..32766; no 32.767 fill survives.
        assert finite.min() >= 0.0
        assert finite.max() < 32.767

    def test_values_at_a_land_and_an_ocean_cell_are_the_granules(self, reader):
        """On a real granule, ``samples_in`` returns the source's own values at a land cell and NaN at an ocean cell.

        Pinned values from the source granule catch a mis-scaled field, a misplaced row or column, or a swapped variable
        order that the synthetic fixtures, with one value everywhere, cannot.

        Queries a small box over a land cell and asserts its shortwave and visible ``fiso`` equal 0.069 and 0.062, then
        queries an ocean cell and asserts all nine values are NaN.
        """
        land = reader.samples_in(BoundingBox(36.4, 36.45, -121.95, -121.9), _T)
        values = sample_at(land, 36.425, -121.925)
        assert values[0] == pytest.approx(0.069)  # brdf_shortwave_fiso
        assert values[3] == pytest.approx(0.062)  # brdf_vis_fiso
        ocean = reader.samples_in(BoundingBox(36.95, 37.0, -123.0, -122.95), _T)
        assert np.isnan(sample_at(ocean, 36.975, -122.975)).all()
