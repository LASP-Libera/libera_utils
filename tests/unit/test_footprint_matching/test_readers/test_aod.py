"""Unit tests for VIIRSAODReader.

Uses synthetic AERDB_D3_VIIRS_NOAA20 Deep Blue aerosol NetCDF4 fixtures created
by ``make_aod_noaa20_fixture``.

Real files come from the NASA Deep Blue single-sensor Level-3 daily product, e.g.
``AERDB_D3_VIIRS_NOAA20.A2026001.002.2026005001030.nc``; the reader reads the
root-level ``Aerosol_Optical_Thickness_550_Land_Ocean_Mean`` (AOD) and
``Aerosol_Type_Land_Ocean_Mode`` (aerosol type) fields.
"""

import numpy as np
import pytest

from libera_utils.footprint_matching.readers.aod import VIIRSAODReader
from libera_utils.footprint_matching.types import BoundingBox
from tests.plugins.footprint_matching_fixtures import make_aod_noaa20_fixture
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
    return VIIRSAODReader([fixture_path])


def _full_bbox() -> BoundingBox:
    return BoundingBox(_LAT_MIN - 0.1, _LAT_MAX + 0.1, _LON_MIN - 0.1, _LON_MAX + 0.1)


class TestVIIRSAODReaderLoadSpatialRegion:
    def test_full_box_serves_both_fields(self, tmp_path):
        """A query enclosing the whole grid serves both fields' values unchanged, as float32, for every cell.

        Aggregation indexes the served values by variable on axis 0, so each cell must come back with one value per
        declared variable; each value is transformed only once, by aggregation, so the reader must not alter in-range
        AOD or change an aerosol-type code, which would assign a footprint to the wrong category. The source key
        attributes the samples to this reader, and a daily aerosol composite carries no timestamp source.

        Builds a 4 x 8 fixture with AOD 0.2 and aerosol type 1 (smoke) in every cell and no fill pixel, queries a box
        enclosing it, and asserts a float32 ``(2, 4, 8)`` grid whose AOD plane is 0.2 and aerosol-type plane is 1, with
        source ``"viirs_aod"`` and no timestamp source.
        """
        reader = _make_reader(tmp_path, include_fill_pixel=False)
        data, _, _ = _region(reader, _full_bbox(), _T)
        assert data.shape == (2, _N_LAT, _N_LON)
        assert data.dtype == np.float32
        assert np.allclose(data[_AOD_AXIS], _AOD_FILL, atol=1e-5)
        assert np.allclose(data[_TYPE_AXIS], _AEROSOL_TYPE, atol=1e-5)
        samples = reader.samples_in(_full_bbox(), _T)
        assert samples.source == "viirs_aod"
        assert samples.timestamp_source is None

    def test_fill_pixel_becomes_nan_in_both_fields(self, tmp_path):
        """A cell holding the file's -999 ``_FillValue`` is served as NaN in both the AOD and the aerosol-type plane.

        A fill sentinel left as a number would be averaged into the footprint's AOD or counted as an aerosol category,
        corrupting the product.

        Writes the fill value into cell (0, 0) of both fields and asserts that cell is NaN in each plane while the other
        31 cells of each plane stay finite.
        """
        reader = _make_reader(tmp_path, include_fill_pixel=True)
        data, _, _ = _region(reader, _full_bbox(), _T)
        assert np.isnan(data[_AOD_AXIS, 0, 0])
        assert np.isnan(data[_TYPE_AXIS, 0, 0])
        # Every other pixel of each field remains valid.
        assert np.isfinite(data[_AOD_AXIS]).sum() == _N_LAT * _N_LON - 1
        assert np.isfinite(data[_TYPE_AXIS]).sum() == _N_LAT * _N_LON - 1

    @pytest.mark.parametrize(
        ("kwargs", "aod", "aerosol_type"),
        [
            pytest.param({"aod_fill": 99.0}, np.nan, _AEROSOL_TYPE, id="aod-above-range"),
            pytest.param({"aod_fill": 7.0}, 7.0, _AEROSOL_TYPE, id="aod-high-in-range"),
            pytest.param(
                {"aod_fill": 3.0, "aod_valid_range": (0.0, 2.0)}, np.nan, _AEROSOL_TYPE, id="aod-outside-file-range"
            ),
            pytest.param({"aerosol_type_value": 99}, _AOD_FILL, np.nan, id="type-out-of-range"),
        ],
    )
    def test_values_outside_the_fields_valid_range_are_nan(self, tmp_path, kwargs, aod, aerosol_type):
        """A value outside its own field's ``valid_range`` is served as NaN, and the other field's value is kept.

        An out-of-range AOD is not a valid retrieval and would skew the footprint mean, and an aerosol-type code with no
        category has no bin in the mode aggregation; masking one field must not discard the valid other field in the
        same cell. Heavy-smoke scenes produce AOD between 5 and 10, so values high but in range must be kept, and the
        file, not a default, is the authority on which values are valid.

        Writes one value to every cell: AOD 99 (outside the file's [0, 10]), AOD 7 (inside it), AOD 3 with the file's
        AOD ``valid_range`` narrowed to [0, 2], or aerosol type 99 (outside [0, 7]); asserts each plane is NaN or the
        value written.
        """
        reader = _make_reader(tmp_path, include_fill_pixel=False, **kwargs)
        data, _, _ = _region(reader, _full_bbox(), _T)
        np.testing.assert_allclose(data[_AOD_AXIS], np.full((_N_LAT, _N_LON), aod), atol=1e-5)
        np.testing.assert_allclose(data[_TYPE_AXIS], np.full((_N_LAT, _N_LON), aerosol_type), atol=1e-5)

    @pytest.mark.parametrize(
        ("fill_values", "masked_axis", "kept_axis"),
        [
            ({"aod_fill_value": _AOD_FILL}, _AOD_AXIS, _TYPE_AXIS),
            ({"aerosol_type_fill_value": _AEROSOL_TYPE}, _TYPE_AXIS, _AOD_AXIS),
        ],
    )
    def test_fill_value_comes_from_each_field(self, tmp_path, fill_values, masked_axis, kept_axis):
        """Each field is masked with its own ``_FillValue`` attribute, independently of the other field's.

        The two fields declare their sentinels separately, so applying one field's fill to the other would keep fill
        values in one or discard valid data in the other.

        Parametrized over which field gets a ``_FillValue`` equal to the constant the fixture writes in every cell (AOD
        0.2 or aerosol type 1); asserts that field's plane is all NaN and the other field's plane is all finite.
        """
        # A field whose own _FillValue equals the fixture's constant is masked; the other field is not.
        reader = _make_reader(tmp_path, include_fill_pixel=False, **fill_values)
        data, _, _ = _region(reader, _full_bbox(), _T)
        assert np.all(np.isnan(data[masked_axis]))
        assert np.all(np.isfinite(data[kept_axis]))

    @pytest.mark.parametrize(
        ("field", "match"),
        [
            ("aod_fill_value", "has no _FillValue attribute"),
            ("aerosol_type_fill_value", "has no _FillValue attribute"),
            ("aod_valid_range", "has no valid_range attribute"),
            ("aerosol_type_valid_range", "has no valid_range attribute"),
        ],
    )
    def test_field_without_fill_or_range_attribute_raises(self, tmp_path, field, match):
        """``VIIRSAODReader`` raises ``ValueError`` when either field lacks its ``_FillValue`` or ``valid_range``.

        Without ``_FillValue`` the reader cannot tell fill from data, and it masks out-of-range values only by the
        file's own ``valid_range``; serving sentinels or invalid values as data would silently corrupt the product, so
        the read stops instead.

        Parametrized over the two attributes of each field; writes the fixture without that attribute and asserts
        ``samples_in`` raises with a message naming the missing attribute.
        """
        reader = _make_reader(tmp_path, include_fill_pixel=False, **{field: None})
        with pytest.raises(ValueError, match=match):
            reader.samples_in(_full_bbox(), _T)

    def test_aerosol_type_range_beyond_the_categories_raises(self, tmp_path):
        """``VIIRSAODReader`` raises ``ValueError`` when the aerosol-type ``valid_range`` admits codes outside 0..7.

        The modal and ranked aerosol-type aggregations have one bin per known category, so a file whose valid range
        admits more codes means the product's category set has changed and the reader must be updated before reading it.

        Writes an aerosol-type ``valid_range`` of [0, 9] and asserts ``samples_in`` raises with a message quoting that
        range and the 0..7 the reader maps.
        """
        reader = _make_reader(tmp_path, aerosol_type_valid_range=(0, 9))
        with pytest.raises(ValueError, match=r"valid_range is \[0, 9\].*aerosol types 0..7"):
            reader.samples_in(_full_bbox(), _T)


class TestVIIRSAODReaderRealSubset:
    """The full 1 degree grid of a real AERDB_D3_VIIRS_NOAA20 granule (2026-09-27).

    Only the coordinates and the two fields the reader reads are kept. Pinned values
    are the source granule's own.
    """

    @pytest.fixture
    def reader(self, fmatch_real_data_path):
        return VIIRSAODReader([fmatch_real_data_path / "AERDB_D3_VIIRS_NOAA20.A2026270.021.2026274001051_subset.nc"])

    def test_axes_are_the_global_one_degree_cell_centers(self, reader):
        """On a real granule, ``VIIRSAODReader`` loads the global 180 x 360 grid of 1 degree cell centers.

        Each cell is served at its axis coordinate, so a shifted, truncated or reversed axis would place every AOD and
        aerosol type at the wrong location.

        Loads the native grid of the real granule subset and asserts the axis sizes and that latitude runs from -89.5 to
        89.5 and longitude from -179.5 to 179.5.
        """
        native = reader._native
        assert (native.lat_axis.size, native.lon_axis.size) == (180, 360)
        assert (native.lat_axis[0], native.lat_axis[-1]) == (-89.5, 89.5)
        assert (native.lon_axis[0], native.lon_axis[-1]) == (-179.5, 179.5)

    def test_fill_and_out_of_range_cells_are_nan(self, reader):
        """On a real granule, fill cells are NaN and every remaining value lies in its field's valid range.

        This checks the masking against the provider's own attributes rather than the synthetic fixture's: an unmasked
        -999 fill would show up as a negative AOD or an unknown aerosol code.

        Asserts the native AOD plane has exactly 26960 finite cells (the source granule's count inside the valid
        range), no AOD below 0, and only aerosol-type codes 0 to 7.
        """
        aod, aerosol_type = reader._native.data[_AOD_AXIS], reader._native.data[_TYPE_AXIS]
        # 26 960 of the 64 800 cells hold an AOD inside valid_range 0..10; the rest are -999 fill.
        assert np.isfinite(aod).sum() == 26960
        assert np.nanmin(aod) >= 0.0
        assert set(np.unique(aerosol_type[np.isfinite(aerosol_type)])) <= set(range(8))

    def test_values_at_a_cell_are_the_granules(self, reader):
        """On a real granule, the sample served at one cell carries that cell's AOD and aerosol type from the file.

        This checks the whole read path, from the file's arrays through the box query to one served sample, against
        values pinned from the source granule, so a misaligned axis or swapped field would serve another cell's values.

        Queries a 1 degree box around 15.5 S, 21.5 E and asserts the sample there has AOD 0.69888 and aerosol type 4.
        """
        samples = reader.samples_in(BoundingBox(-16.0, -15.0, 21.0, 22.0), _T)
        aod, aerosol_type = sample_at(samples, -15.5, 21.5)
        assert aod == pytest.approx(0.69888, abs=1e-5)
        assert aerosol_type == 4
