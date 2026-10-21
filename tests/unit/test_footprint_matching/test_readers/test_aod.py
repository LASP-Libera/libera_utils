"""Unit tests for VIIRSAODReader.

Uses synthetic AERDB_D3_VIIRS_NOAA20 Deep Blue aerosol NetCDF4 fixtures created
by ``make_aod_noaa20_fixture``.

Real files come from the NASA Deep Blue single-sensor Level-3 daily product, e.g.
``AERDB_D3_VIIRS_NOAA20.A2026001.002.2026005001030.nc``; the reader reads the
root-level ``Aerosol_Optical_Thickness_550_Land_Ocean_Mean`` (AOD) and
``Aerosol_Type_Land_Ocean_Mode`` (aerosol type) fields.
"""

from __future__ import annotations

import numpy as np
import pytest

from libera_utils.footprint_matching.readers.aod import VIIRSAODReader
from libera_utils.footprint_matching.types import BoundingBox
from tests.test_data.footprint_matching.fixtures import make_aod_noaa20_fixture
from tests.unit.test_footprint_matching.test_readers.grid_helpers import region as _region
from tests.unit.test_footprint_matching.test_readers.grid_helpers import sample_at

_T = np.datetime64("2026-01-01T00:00:00", "ns")

_N_LAT = 4
_N_LON = 8
_LAT_MIN = 0.5
_LAT_MAX = 3.5
_LON_MIN = 10.5
_LON_MAX = 17.5
_AOD_FILL = 0.2
_AEROSOL_TYPE = 1  # smoke

# Axis-0 indices into the stacked (n_var, n_lat, n_lon) array, in VARIABLES order.
_AOD_AXIS = 0
_TYPE_AXIS = 1


def _make_reader(tmp_path, **kwargs) -> VIIRSAODReader:
    kwargs.setdefault("aod_fill", _AOD_FILL)
    kwargs.setdefault("aerosol_type_value", _AEROSOL_TYPE)
    fixture_path = make_aod_noaa20_fixture(
        tmp_path,
        n_lat=_N_LAT,
        n_lon=_N_LON,
        lat_min=_LAT_MIN,
        lat_max=_LAT_MAX,
        lon_min=_LON_MIN,
        lon_max=_LON_MAX,
        **kwargs,
    )
    return VIIRSAODReader(fixture_path)


def _full_bbox() -> BoundingBox:
    return BoundingBox(_LAT_MIN - 0.1, _LAT_MAX + 0.1, _LON_MIN - 0.1, _LON_MAX + 0.1)


class TestVIIRSAODReaderClassAttributes:
    def test_reader_key(self):
        # Targets the reader's registry key; asserts READER_KEY equals the expected "viirs_aod".
        assert VIIRSAODReader.READER_KEY == "viirs_aod"

    def test_resolution_km(self):
        # Targets the declared grid resolution; asserts RESOLUTION_KM equals 111.0 km.
        assert VIIRSAODReader.RESOLUTION_KM == 111.0

    def test_reads_aod_and_aerosol_type(self):
        # Targets the VARIABLES spec pair; asserts each spec's name/dtype/aggregation/n_categories match AOD and type.
        assert len(VIIRSAODReader.VARIABLES) == 2
        aod, aerosol_type = VIIRSAODReader.VARIABLES
        assert aod.name == "aod_550"
        assert aod.dtype == "float32"
        assert aod.aggregation == "weighted_log_mean"
        assert aod.n_categories is None
        assert aerosol_type.name == "aerosol_type"
        assert aerosol_type.dtype == "int16"
        assert aerosol_type.aggregation == "weighted_mode"
        assert aerosol_type.n_categories == 8

    def test_ranked_aerosol_type_companions(self):
        # The ranked types are derived product outputs, not read variables; asserts the ADDITIONAL_PRODUCT_VARIABLES
        # names plus each spec's aggregation/dtype/n_categories.
        additional = {spec.name: spec for spec in VIIRSAODReader.ADDITIONAL_PRODUCT_VARIABLES}
        assert set(additional) == {
            "aerosol_type_primary",
            "aerosol_type_secondary",
            "aerosol_type_tertiary",
        }
        for name, agg in (
            ("aerosol_type_primary", "weighted_mode_primary"),
            ("aerosol_type_secondary", "weighted_mode_secondary"),
            ("aerosol_type_tertiary", "weighted_mode_tertiary"),
        ):
            assert additional[name].aggregation == agg
            assert additional[name].dtype == "int16"
            assert additional[name].n_categories == 8

    def test_categorical_variables_get_no_standard_deviation_companion(self):
        # aod_550 is continuous so it gains a std-dev companion while weighted_mode aerosol-type vars do not; asserts
        # product_variable_specs() names include aod_550_standard_deviation but exclude categorical *_standard_deviation.
        names = {spec.name for spec in VIIRSAODReader.product_variable_specs()}
        assert "aod_550_standard_deviation" in names
        for categorical in (
            "aerosol_type",
            "aerosol_type_primary",
            "aerosol_type_secondary",
            "aerosol_type_tertiary",
        ):
            assert f"{categorical}_standard_deviation" not in names


class TestVIIRSAODReaderLoadSpatialRegion:
    def test_returns_3d_data_stacked_in_variables_order(self, tmp_path):
        # Targets samples_in output shape; asserts a 3-D float32 array stacked as (2 vars, n_lat, n_lon).
        reader = _make_reader(tmp_path, include_fill_pixel=False)
        data, lats, lons = _region(reader, _full_bbox(), _T)
        assert data.ndim == 3
        assert data.shape == (2, _N_LAT, _N_LON)
        assert data.dtype == np.float32

    def test_lat_is_ascending(self, tmp_path):
        # Targets latitude ordering in the loaded region; asserts np.diff(lats) stays non-negative (ascending).
        reader = _make_reader(tmp_path)
        _, lats, _ = _region(reader, _full_bbox(), _T)
        assert np.all(np.diff(lats) >= 0)

    def test_non_fill_aod_values_preserved(self, tmp_path):
        # Targets AOD passthrough for valid pixels; asserts the AOD axis equals the fixture fill value _AOD_FILL.
        reader = _make_reader(tmp_path, include_fill_pixel=False)
        data, _, _ = _region(reader, _full_bbox(), _T)
        assert np.allclose(data[_AOD_AXIS], _AOD_FILL, atol=1e-5)

    def test_non_fill_aerosol_type_values_preserved(self, tmp_path):
        # Targets aerosol-type passthrough for valid pixels; asserts the type axis equals the fixture _AEROSOL_TYPE.
        reader = _make_reader(tmp_path, include_fill_pixel=False)
        data, _, _ = _region(reader, _full_bbox(), _T)
        assert np.allclose(data[_TYPE_AXIS], _AEROSOL_TYPE, atol=1e-5)

    def test_fill_pixel_becomes_nan_in_both_fields(self, tmp_path):
        # Targets fill masking; asserts pixel (0,0) is NaN in both fields while all other pixels stay finite.
        reader = _make_reader(tmp_path, include_fill_pixel=True)
        data, _, _ = _region(reader, _full_bbox(), _T)
        assert np.isnan(data[_AOD_AXIS, 0, 0])
        assert np.isnan(data[_TYPE_AXIS, 0, 0])
        # Every other pixel of each field remains valid.
        assert np.isfinite(data[_AOD_AXIS]).sum() == _N_LAT * _N_LON - 1
        assert np.isfinite(data[_TYPE_AXIS]).sum() == _N_LAT * _N_LON - 1

    def test_aod_out_of_range_becomes_nan(self, tmp_path):
        # Targets AOD valid-range masking; asserts an out-of-range 99.0 AOD becomes all-NaN while type is preserved.
        reader = _make_reader(tmp_path, aod_fill=99.0, include_fill_pixel=False)
        data, _, _ = _region(reader, _full_bbox(), _T)
        # The file's AOD valid_range is [0, 10], so 99.0 must be masked; aerosol type is unaffected.
        assert np.all(np.isnan(data[_AOD_AXIS]))
        assert np.allclose(data[_TYPE_AXIS], _AEROSOL_TYPE, atol=1e-5)

    def test_aod_inside_the_files_valid_range_is_served(self, tmp_path):
        # Heavy-smoke AOD between 5 and 10 lies inside the file's valid_range and is kept.
        reader = _make_reader(tmp_path, aod_fill=7.0, include_fill_pixel=False)
        data, _, _ = _region(reader, _full_bbox(), _T)
        assert np.allclose(data[_AOD_AXIS], 7.0, atol=1e-5)

    def test_valid_range_comes_from_the_file(self, tmp_path):
        # A narrower valid_range in the file masks a value the default range keeps.
        reader = _make_reader(tmp_path, aod_fill=3.0, include_fill_pixel=False, aod_valid_range=(0.0, 2.0))
        data, _, _ = _region(reader, _full_bbox(), _T)
        assert np.all(np.isnan(data[_AOD_AXIS]))

    @pytest.mark.parametrize("field", ["aod_valid_range", "aerosol_type_valid_range"])
    def test_field_without_valid_range_raises(self, tmp_path, field):
        reader = _make_reader(tmp_path, **{field: None})
        with pytest.raises(ValueError, match="has no valid_range attribute"):
            reader.samples_in(_full_bbox(), _T)

    def test_aerosol_type_range_beyond_the_categories_raises(self, tmp_path):
        reader = _make_reader(tmp_path, aerosol_type_valid_range=(0, 9))
        with pytest.raises(ValueError, match=r"valid_range is \[0, 9\].*aerosol types 0..7"):
            reader.samples_in(_full_bbox(), _T)

    def test_out_of_range_aerosol_type_becomes_nan(self, tmp_path):
        # Category 99 is outside the valid [0, 7] range -> NaN; AOD is unaffected. Asserts the type axis is all NaN
        # while the AOD axis still equals _AOD_FILL.
        reader = _make_reader(tmp_path, aerosol_type_value=99, include_fill_pixel=False)
        data, _, _ = _region(reader, _full_bbox(), _T)
        assert np.all(np.isnan(data[_TYPE_AXIS]))
        assert np.allclose(data[_AOD_AXIS], _AOD_FILL, atol=1e-5)

    def test_empty_result_outside_bbox(self, tmp_path):
        # Targets querying a bbox with no data; asserts the returned array has shape (2, 0, 0) and zero size.
        reader = _make_reader(tmp_path)
        data, lats, lons = _region(reader, BoundingBox(-60.0, -58.0, 170.0, 172.0), _T)
        assert data.shape == (2, 0, 0)
        assert data.size == 0


class TestVIIRSAODReaderSamples:
    def test_samples_carry_source_and_no_timestamp_source(self, tmp_path):
        reader = _make_reader(tmp_path)
        samples = reader.samples_in(_full_bbox(), _T)
        assert samples.source == "viirs_aod"
        assert samples.timestamp_source is None
        assert samples.values.shape[0] == 2


class TestVIIRSAODReaderRealSubset:
    """The full 1 degree grid of a real AERDB_D3_VIIRS_NOAA20 granule (2026-09-27).

    Only the coordinates and the two fields the reader reads are kept. Pinned values
    are the source granule's own.
    """

    @pytest.fixture
    def reader(self, fmatch_real_data_path):
        return VIIRSAODReader(fmatch_real_data_path / "AERDB_D3_VIIRS_NOAA20.A2026270.021.2026274001051_subset.nc")

    def test_axes_are_the_global_one_degree_cell_centres(self, reader):
        native = reader._native
        assert (native.lat_axis.size, native.lon_axis.size) == (180, 360)
        assert (native.lat_axis[0], native.lat_axis[-1]) == (-89.5, 89.5)
        assert (native.lon_axis[0], native.lon_axis[-1]) == (-179.5, 179.5)

    def test_fill_and_out_of_range_cells_are_nan(self, reader):
        aod, aerosol_type = reader._native.data[_AOD_AXIS], reader._native.data[_TYPE_AXIS]
        # 26 960 of the 64 800 cells hold an AOD inside valid_range 0..10; the rest are -999 fill.
        assert np.isfinite(aod).sum() == 26960
        assert np.nanmin(aod) >= 0.0
        assert set(np.unique(aerosol_type[np.isfinite(aerosol_type)])) <= set(range(8))

    def test_values_at_a_cell_are_the_granules(self, reader):
        samples = reader.samples_in(BoundingBox(-16.0, -15.0, 21.0, 22.0), _T)
        aod, aerosol_type = sample_at(samples, -15.5, 21.5)
        assert aod == pytest.approx(0.69888, abs=1e-5)
        assert aerosol_type == 4
