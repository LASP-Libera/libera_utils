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
        # Targets the reader's registry key; asserts READER_KEY equals the expected "viirs_brdf".
        assert VIIRSBRDFReader.READER_KEY == "viirs_brdf"

    def test_resolution_km(self):
        # Targets the declared native resolution; asserts RESOLUTION_KM equals the VJ143C1 5.6 km value.
        assert VIIRSBRDFReader.RESOLUTION_KM == 5.6

    def test_variables_count_is_nine(self):
        # Targets the variable inventory size; asserts VARIABLES has exactly 9 entries (3 bands x 3 params).
        assert len(VIIRSBRDFReader.VARIABLES) == 9

    def test_variable_names_cover_three_bands_three_params(self):
        # Targets variable naming coverage; asserts every band/param pair yields a "brdf_{band}_{param}" name.
        names = {v.name for v in VIIRSBRDFReader.VARIABLES}
        for band in ("shortwave", "vis", "nir"):
            for param in ("fiso", "fvol", "fgeo"):
                assert f"brdf_{band}_{param}" in names

    def test_all_variables_are_float32_weighted_mean(self):
        # Targets per-variable metadata; asserts each var is float32, weighted_mean aggregation, non-categorical.
        for v in VIIRSBRDFReader.VARIABLES:
            assert v.dtype == "float32"
            assert v.aggregation == "weighted_mean"
            assert v.n_categories is None


class TestVIIRSBRDFReaderScaleAndFill:
    """Verify scale_factor application and fill → NaN conversion."""

    def test_scale_factor_applied_to_param_fill(self, tmp_path):
        # param_fill=200 with scale_factor=0.001 → 0.2
        reader = _make_reader(tmp_path)
        data, _, _ = _region(reader, _full_bbox(), _T)
        # All non-fill pixels should equal 0.200
        non_fill = data[~np.isnan(data)]
        assert non_fill.size > 0
        assert np.allclose(non_fill, 0.200, atol=1e-4)

    def test_fill_sentinel_becomes_nan(self, tmp_path):
        # The fixture writes fill_sentinel to file position [-1, -1] — the lat_min row of
        # the DESCENDING file. The reader flips lats to ascending, moving that row to
        # output row 0, so the fill lands at [:, 0, -1] (one fill pixel per band).
        reader = _make_reader(tmp_path)
        data, lats, lons = _region(reader, _full_bbox(), _T)
        assert np.isnan(data[:, 0, -1]).all()
        assert np.isnan(data).sum() == data.shape[0]

    def test_non_fill_pixels_are_not_nan(self, tmp_path):
        # Targets that valid data survives fill handling; asserts not every pixel is NaN after loading.
        reader = _make_reader(tmp_path)
        data, _, _ = _region(reader, _full_bbox(), _T)
        # At least one pixel (all except [-1, -1]) should be non-NaN.
        assert not np.all(np.isnan(data))


class TestVIIRSBRDFReaderLoadSpatialRegion:
    def test_returns_3d_data_array(self, tmp_path):
        # Targets output array shape; asserts samples_in returns a 3-D array with 9 variable planes.
        reader = _make_reader(tmp_path)
        data, lats, lons = _region(reader, _full_bbox(), _T)
        assert data.ndim == 3
        assert data.shape[0] == 9

    def test_data_dtype_is_float32(self, tmp_path):
        # Targets output dtype; asserts the loaded region array is float32.
        reader = _make_reader(tmp_path)
        data, _, _ = _region(reader, _full_bbox(), _T)
        assert data.dtype == np.float32

    def test_lat_is_ascending(self, tmp_path):
        # The fixture stores lats in DESCENDING order; the reader must flip them
        # to ascending, so output row 0 is the southernmost cell center.
        reader = _make_reader(tmp_path)
        _, lats, _ = _region(reader, _full_bbox(), _T)
        assert lats.size > 1
        assert np.all(np.diff(lats) >= 0), f"lats not ascending: {lats}"
        assert np.isclose(lats[0], _LAT_MIN + _HALF_CELL), f"row 0 should be the southern cell center: {lats}"

    def test_spatial_shape_matches_fixture(self, tmp_path):
        # Targets spatial dimensions of the load; asserts lat/lon axes match the fixture's _N_LAT x _N_LON grid.
        reader = _make_reader(tmp_path)
        data, lats, lons = _region(reader, _full_bbox(), _T)
        assert data.shape[1] == _N_LAT
        assert data.shape[2] == _N_LON

    def test_empty_result_outside_bbox(self, tmp_path):
        # Targets bbox filtering; asserts a bbox with no overlap yields an empty array that keeps the 9-var axis.
        reader = _make_reader(tmp_path)
        bbox = BoundingBox(-60.0, -58.0, 170.0, 172.0)
        data, lats, lons = _region(reader, bbox, _T)
        assert data.size == 0
        assert data.shape[0] == 9

    def test_partial_bbox_subsets_correctly(self, tmp_path):
        # Targets partial bbox subsetting; asserts a half-width lon request returns fewer lon columns than _N_LON.
        reader = _make_reader(tmp_path)
        # Request only the first half of the lon range.
        lon_mid = (_LON_MIN + _LON_MAX) / 2
        bbox = BoundingBox(_LAT_MIN - 0.01, _LAT_MAX + 0.01, _LON_MIN - 0.01, lon_mid)
        data, lats, lons = _region(reader, bbox, _T)
        assert lons.size < _N_LON

    def test_load_peak_memory_is_the_held_fields(self, tmp_path):
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
        # BRDF is a static surface property product; no instrument timestamp source.
        reader = _make_reader(tmp_path)
        samples = reader.samples_in(_full_bbox(), _T)
        assert samples.source == "viirs_brdf"
        assert samples.timestamp_source is None
        assert samples.values.shape[0] == 9

    def test_granule_is_read_once_for_every_box(self, tmp_path):
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
        native = _make_reader(tmp_path)._native
        np.testing.assert_allclose(native.lat_axis, [0.025, 0.075, 0.125, 0.175])
        np.testing.assert_allclose(native.lon_axis, 10.025 + 0.05 * np.arange(_N_LON))

    def test_global_cmg_edges_become_the_expected_centers(self, tmp_path):
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
        reader = _make_reader(tmp_path, omit_attributes=(attribute,))
        with pytest.raises(ValueError, match=f"has no {attribute} attribute"):
            _ = reader._native

    @pytest.mark.parametrize("valid_range", [(0, 100), (300, 400)])
    def test_values_outside_the_valid_range_are_nan(self, tmp_path, valid_range):
        # The fixture's raw value 200 lies above the first range and below the second.
        reader = _make_reader(tmp_path, valid_range=valid_range)
        assert np.all(np.isnan(reader._native.data))

    def test_add_offset_is_applied(self, tmp_path):
        path = _fixture(tmp_path)
        with h5py.File(path, "r+") as f:
            f["HDFEOS/GRIDS/VIIRS_CMG_BRDF/Data Fields/BRDF_Albedo_Parameter1_shortwave"].attrs["add_offset"] = [0.5]
        data = VIIRSBRDFReader(path)._native.data
        assert np.allclose(data[0][np.isfinite(data[0])], 0.7)
        assert np.allclose(data[1][np.isfinite(data[1])], 0.2)

    def test_each_field_uses_its_own_attribute(self, tmp_path):
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
        with pytest.raises(ValueError, match="has no grid named 'VIIRS_CMG_BRDF'"):
            _parse_he5_grid_metadata("GROUP=GridStructure\nEND_GROUP=GridStructure\n", "VIIRS_CMG_BRDF", "g.h5")

    def test_missing_field_raises(self):
        text = 'GridName="VIIRS_CMG_BRDF"\n\t\tXDim=8\n\t\tProjection=HE5_GCTP_GEO\n\tEND_GROUP=GRID_1\n'
        with pytest.raises(ValueError, match="has no YDim field"):
            _parse_he5_grid_metadata(text, "VIIRS_CMG_BRDF", "g.h5")

    def test_non_geographic_projection_raises(self, tmp_path):
        reader = _make_reader(tmp_path, projection="HE5_GCTP_SNSOID")
        with pytest.raises(ValueError, match="not the geographic HE5_GCTP_GEO"):
            _ = reader._native

    def test_axes_not_spanning_the_corners_raise(self, tmp_path):
        # Axes written as cell centers (the corners say the region starts half a cell earlier).
        reader = _make_reader(
            tmp_path, metadata_corners=(_LON_MIN - _HALF_CELL, _LAT_MAX + _HALF_CELL, _LON_MAX, _LAT_MIN)
        )
        with pytest.raises(ValueError, match="does not span the StructMetadata corners"):
            _ = reader._native

    def test_non_uniform_axis_raises(self, tmp_path):
        path = _fixture(tmp_path)
        with h5py.File(path, "r+") as f:
            lat = f["HDFEOS/GRIDS/VIIRS_CMG_BRDF/Data Fields/lat"]
            lat[1] = lat[1] + 0.01
        with pytest.raises(ValueError, match="lat axis is not uniformly spaced"):
            _ = VIIRSBRDFReader(path)._native

    def test_axis_length_not_matching_the_grid_raises(self, tmp_path):
        path = _fixture(tmp_path)
        with h5py.File(path, "r+") as f:
            meta = f["HDFEOS INFORMATION/StructMetadata.0"][()].decode().replace("XDim=8", "XDim=9")
            del f["HDFEOS INFORMATION/StructMetadata.0"]
            f.create_dataset("HDFEOS INFORMATION/StructMetadata.0", data=np.bytes_(meta))
        with pytest.raises(ValueError, match="XDim=9"):
            _ = VIIRSBRDFReader(path)._native

    def test_non_string_struct_metadata_raises(self, tmp_path):
        path = _fixture(tmp_path)
        with h5py.File(path, "r+") as f:
            del f["HDFEOS INFORMATION/StructMetadata.0"]
            f.create_dataset("HDFEOS INFORMATION/StructMetadata.0", data=np.arange(3))
        with pytest.raises(ValueError, match="not the fixed-length string"):
            _ = VIIRSBRDFReader(path)._native


class TestVIIRSBRDFFileErrors:
    """h5py's own errors reach the caller: FileNotFoundError (an OSError) for a missing file, KeyError for a missing object."""

    def test_missing_file_raises_file_not_found_error(self, tmp_path):
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
        native = reader._native
        assert native.data.shape == (len(VIIRSBRDFReader.VARIABLES), 40, 40)
        assert native.lat_axis[0] == pytest.approx(35.025)
        assert native.lat_axis[-1] == pytest.approx(36.975)
        assert native.lon_axis[0] == pytest.approx(-122.975)
        assert native.lon_axis[-1] == pytest.approx(-121.025)

    def test_land_is_scaled_and_ocean_fill_is_nan(self, reader):
        fiso = reader._native.data[0]
        assert np.isnan(fiso).mean() == pytest.approx(0.694375)
        finite = fiso[np.isfinite(fiso)]
        # Scaled by 0.001 from int16 counts in valid_range 0..32766; no 32.767 fill survives.
        assert finite.min() >= 0.0
        assert finite.max() < 32.767

    def test_values_at_a_land_and_an_ocean_cell_are_the_granules(self, reader):
        land = reader.samples_in(BoundingBox(36.4, 36.45, -121.95, -121.9), _T)
        values = sample_at(land, 36.425, -121.925)
        assert values[0] == pytest.approx(0.069)  # brdf_shortwave_fiso
        assert values[3] == pytest.approx(0.062)  # brdf_vis_fiso
        ocean = reader.samples_in(BoundingBox(36.95, 37.0, -123.0, -122.95), _T)
        assert np.isnan(sample_at(ocean, 36.975, -122.975)).all()
