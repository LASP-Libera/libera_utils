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
        """``VIIRSAODReader`` registers under the key ``"viirs_aod"``.

        The key is how a caller looks the reader up in ``ReaderRegistry`` and the prefix of every output variable
        name, so a different key breaks lookups and renames the product's variables.

        Asserts the ``READER_KEY`` class attribute equals ``"viirs_aod"``.
        """
        assert VIIRSAODReader.READER_KEY == "viirs_aod"

    def test_resolution_km(self):
        """``VIIRSAODReader`` declares a native resolution of 111 km, the size of a 1 degree cell at the equator.

        The PSF weigher sizes each sample's coverage from this value and every query box is padded by half a cell of
        it, so a wrong value misstates footprint coverage and can drop cells at a box's edge.

        Asserts the ``RESOLUTION_KM`` class attribute equals ``111.0``.
        """
        assert VIIRSAODReader.RESOLUTION_KM == 111.0

    def test_reads_aod_and_aerosol_type(self):
        """``VIIRSAODReader.VARIABLES`` declares ``aod_550`` then ``aerosol_type``, with their dtypes and aggregations.

        The order fixes axis 0 of every served ``values`` array, and the aggregation tells the PSF aggregation engine
        how to reduce each variable: a log mean for the continuous AOD and a mode over 8 categories for the aerosol
        type. A mismatch would mislabel or mis-aggregate a product variable.

        Unpacks the two specs and asserts each one's ``name``, ``dtype``, ``aggregation`` and ``n_categories``.
        """
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
        """``VIIRSAODReader.ADDITIONAL_PRODUCT_VARIABLES`` declares the primary, secondary and tertiary aerosol types.

        These ranked types are computed during PSF aggregation rather than read from the file, so this declaration is
        the only place they enter the product definition; a missing or misconfigured spec drops a product variable or
        allocates the wrong number of category bins.

        Asserts the names are exactly the three ranked types and that each is ``int16`` with 8 categories and its own
        ``weighted_mode_primary``, ``weighted_mode_secondary`` or ``weighted_mode_tertiary`` aggregation.
        """
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
        """``VIIRSAODReader.product_variable_specs`` gives AOD a standard-deviation companion and no aerosol type one.

        A standard deviation of a most-common category is not physically meaningful, so the product carries
        ``aod_550_standard_deviation`` and no ``*_standard_deviation`` variable for any mode-aggregated aerosol type.

        Collects the names from ``product_variable_specs()`` and asserts the AOD companion is present and the companions
        of the four categorical variables are absent.
        """
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
    def test_full_box_serves_both_variables_for_every_cell(self, tmp_path):
        """A query enclosing the whole fixture grid serves both variables, as float32, for every grid cell.

        Aggregation indexes the served values by variable on axis 0, so each cell inside the box must come back with
        one value per declared variable. Which plane holds which variable is pinned by the value tests below.

        Builds a 4 x 8 fixture with no fill pixel, queries a box enclosing it through the ``region`` helper (which folds
        the samples back into a grid), and asserts a 3-D float32 array of shape ``(2, 4, 8)``.
        """
        reader = _make_reader(tmp_path, include_fill_pixel=False)
        data, lats, lons = _region(reader, _full_bbox(), _T)
        assert data.ndim == 3
        assert data.shape == (2, _N_LAT, _N_LON)
        assert data.dtype == np.float32

    def test_non_fill_aod_values_preserved(self, tmp_path):
        """Valid AOD values pass through ``VIIRSAODReader`` unchanged.

        Each value is meant to be transformed only once, by aggregation, on its way into the product, so the reader
        must not rescale or alter in-range AOD.

        Writes a fixture whose every AOD cell is 0.2 with no fill pixel and asserts the served AOD plane equals 0.2.
        """
        reader = _make_reader(tmp_path, include_fill_pixel=False)
        data, _, _ = _region(reader, _full_bbox(), _T)
        assert np.allclose(data[_AOD_AXIS], _AOD_FILL, atol=1e-5)

    def test_non_fill_aerosol_type_values_preserved(self, tmp_path):
        """Valid aerosol-type codes pass through ``VIIRSAODReader`` unchanged, carried as float.

        A changed code would assign a footprint to the wrong aerosol category.

        Writes a fixture whose every aerosol-type cell is 1 (smoke) with no fill pixel and asserts the served
        aerosol-type plane equals 1.
        """
        reader = _make_reader(tmp_path, include_fill_pixel=False)
        data, _, _ = _region(reader, _full_bbox(), _T)
        assert np.allclose(data[_TYPE_AXIS], _AEROSOL_TYPE, atol=1e-5)

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

    def test_aod_out_of_range_becomes_nan(self, tmp_path):
        """AOD outside the file's ``valid_range`` of [0, 10] is served as NaN, leaving the aerosol type untouched.

        An out-of-range AOD is not a valid retrieval and would skew the footprint mean; masking it must not also discard
        the valid aerosol type in the same cell.

        Writes AOD 99.0 to every cell and asserts the AOD plane is all NaN while the aerosol-type plane still equals 1.
        """
        reader = _make_reader(tmp_path, aod_fill=99.0, include_fill_pixel=False)
        data, _, _ = _region(reader, _full_bbox(), _T)
        # The file's AOD valid_range is [0, 10], so 99.0 must be masked; aerosol type is unaffected.
        assert np.all(np.isnan(data[_AOD_AXIS]))
        assert np.allclose(data[_TYPE_AXIS], _AEROSOL_TYPE, atol=1e-5)

    def test_aod_inside_the_files_valid_range_is_served(self, tmp_path):
        """An AOD of 7.0, high but inside the file's ``valid_range`` of [0, 10], is served unchanged.

        Heavy-smoke scenes produce AOD between 5 and 10, and masking them would discard valid retrievals.

        Writes AOD 7.0 to every cell and asserts the AOD plane equals 7.0.
        """
        # Heavy-smoke AOD between 5 and 10 lies inside the file's valid_range and is kept.
        reader = _make_reader(tmp_path, aod_fill=7.0, include_fill_pixel=False)
        data, _, _ = _region(reader, _full_bbox(), _T)
        assert np.allclose(data[_AOD_AXIS], 7.0, atol=1e-5)

    def test_valid_range_comes_from_the_file(self, tmp_path):
        """``VIIRSAODReader`` takes the AOD valid range from the field's own ``valid_range``, not a fixed default.

        The file is the authority on which values are valid; a reader applying its own range would keep values the
        provider flags as invalid.

        Writes AOD 3.0 with a ``valid_range`` of [0, 2], so 3.0 is valid under the default [0, 10] but not under the
        file's range, and asserts the AOD plane is all NaN.
        """
        # A narrower valid_range in the file masks a value the default range keeps.
        reader = _make_reader(tmp_path, aod_fill=3.0, include_fill_pixel=False, aod_valid_range=(0.0, 2.0))
        data, _, _ = _region(reader, _full_bbox(), _T)
        assert np.all(np.isnan(data[_AOD_AXIS]))

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

    @pytest.mark.parametrize("field", ["aod_fill_value", "aerosol_type_fill_value"])
    def test_field_without_fill_value_raises(self, tmp_path, field):
        """``VIIRSAODReader`` raises ``ValueError`` when either field lacks a ``_FillValue`` attribute.

        Without the attribute the reader cannot tell fill from data, and serving a sentinel as a real value would
        silently corrupt the product, so the read stops instead.

        Parametrized over the AOD and aerosol-type fields; writes the fixture without that field's ``_FillValue`` and
        asserts ``samples_in`` raises with a message naming the missing attribute.
        """
        reader = _make_reader(tmp_path, include_fill_pixel=False, **{field: None})
        with pytest.raises(ValueError, match="has no _FillValue attribute"):
            reader.samples_in(_full_bbox(), _T)

    @pytest.mark.parametrize("field", ["aod_valid_range", "aerosol_type_valid_range"])
    def test_field_without_valid_range_raises(self, tmp_path, field):
        """``VIIRSAODReader`` raises ``ValueError`` when either field lacks a ``valid_range`` attribute.

        The reader masks out-of-range values only by the file's own ``valid_range``, so without it invalid values would
        be served as data; the read stops instead.

        Parametrized over the AOD and aerosol-type fields; writes the fixture without that field's ``valid_range`` and
        asserts ``samples_in`` raises with a message naming the missing attribute.
        """
        reader = _make_reader(tmp_path, **{field: None})
        with pytest.raises(ValueError, match="has no valid_range attribute"):
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

    def test_out_of_range_aerosol_type_becomes_nan(self, tmp_path):
        """An aerosol-type code outside the file's ``valid_range`` of [0, 7] is served as NaN, leaving AOD untouched.

        A code with no category has no bin in the mode aggregation; masking it must not also discard the valid AOD in
        the same cell.

        Writes code 99 to every cell and asserts the aerosol-type plane is all NaN while the AOD plane still equals 0.2.
        """
        # Category 99 is outside the valid [0, 7] range -> NaN; AOD is unaffected. Asserts the type axis is all NaN
        # while the AOD axis still equals _AOD_FILL.
        reader = _make_reader(tmp_path, aerosol_type_value=99, include_fill_pixel=False)
        data, _, _ = _region(reader, _full_bbox(), _T)
        assert np.all(np.isnan(data[_TYPE_AXIS]))
        assert np.allclose(data[_AOD_AXIS], _AOD_FILL, atol=1e-5)

    def test_empty_result_outside_bbox(self, tmp_path):
        """A query box that holds no grid cells returns an empty result rather than an error.

        A footprint can lie outside the grid a reader holds; the query must then serve nothing, still shaped for both
        variables, rather than fail or serve cells from outside the box.

        Queries a box at 58 to 60 degrees south, 170 to 172 degrees east, far from the fixture grid, and asserts the
        folded result has shape ``(2, 0, 0)`` and no elements.
        """
        reader = _make_reader(tmp_path)
        data, lats, lons = _region(reader, BoundingBox(-60.0, -58.0, 170.0, 172.0), _T)
        assert data.shape == (2, 0, 0)
        assert data.size == 0


class TestVIIRSAODReaderSamples:
    def test_samples_carry_source_and_no_timestamp_source(self, tmp_path):
        """Samples served by ``VIIRSAODReader`` are labeled with source ``"viirs_aod"`` and carry no timestamp source.

        The source key attributes each sample set to its reader, and ``timestamp_source`` is set only for cloud-property
        products whose radiometer or camera time governs the data, which a daily aerosol composite is not.

        Queries the full fixture grid and asserts the ``source``, a ``None`` ``timestamp_source``, and two rows of
        values.
        """
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
        """On a real granule, ``VIIRSAODReader`` serves fill as NaN, leaving no negative AOD and only type codes 0 to 7.

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
