"""Unit tests for VIIRSCldpropD3Reader.

Uses synthetic CLDPROP_D3 NetCDF4 fixture files created by
``make_viirs_cldprop_d3_fixture`` so that the group navigation, fill handling,
and (lon, lat) → (lat, lon) transpose logic can be exercised without real data.

Real CLDPROP_D3 files can be downloaded from:
    NCEI CDR: https://www.ncei.noaa.gov/data/cloud-properties-viirs/access/
    Example: CLDPROP_D3_VIIRS_NOAA20.A2026147.011.2026151000710.nc
"""

import numpy as np
import pytest

from libera_utils.footprint_matching.readers.viirs import VIIRSCldpropD3Reader
from libera_utils.footprint_matching.types import BoundingBox
from tests.plugins.footprint_matching_fixtures import make_viirs_cldprop_d3_fixture
from tests.unit.test_footprint_matching.test_readers.grid_helpers import region as _region
from tests.unit.test_footprint_matching.test_readers.grid_helpers import sample_at

_T = np.datetime64("2026-01-01T00:00:00", "ns")

# Grid dimensions used in most tests.
_N_LAT = 4
_N_LON = 8
_LAT_MIN = 0.5
_LAT_MAX = 3.5
_LON_MIN = 10.5
_LON_MAX = 17.5


def _make_reader(tmp_path, **kwargs) -> tuple[VIIRSCldpropD3Reader, object]:
    """Return (reader, fixture_path) using default or overridden fixture params."""
    fixture_path = make_viirs_cldprop_d3_fixture(
        tmp_path,
        n_lat=_N_LAT,
        n_lon=_N_LON,
        lat_min=_LAT_MIN,
        lat_max=_LAT_MAX,
        lon_min=_LON_MIN,
        lon_max=_LON_MAX,
        **kwargs,
    )
    return VIIRSCldpropD3Reader([fixture_path]), fixture_path


def _full_bbox() -> BoundingBox:
    return BoundingBox(_LAT_MIN - 0.1, _LAT_MAX + 0.1, _LON_MIN - 0.1, _LON_MAX + 0.1)


def _position_code(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """A value unique to each fixture cell: ``100 * latitude + longitude``."""
    return 100.0 * lats + lons


def _write_position_coded_means(fixture_path) -> None:
    """Overwrite each group's ``Mean`` so every cell holds :func:`_position_code` of its own location.

    The fields are written in the file's (longitude, latitude) order, so a reader that skips the transpose or slices
    the wrong axis serves a value that does not match the location it is served at.
    """
    import netCDF4

    with netCDF4.Dataset(str(fixture_path), "r+") as ds:
        lats = ds["latitude"][:].astype(np.float64)
        lons = ds["longitude"][:].astype(np.float64)
        coded = _position_code(lats[None, :], lons[:, None]).astype(np.float32)  # (n_lon, n_lat)
        for group in ds.groups.values():
            group["Mean"][:] = coded


class TestVIIRSCldpropD3ReaderLoadSpatialRegion:
    def test_full_box_serves_both_fields_in_variable_order(self, tmp_path):
        """``VIIRSCldpropD3Reader.samples_in`` serves both cloud fields as float32 rows, in ``VARIABLES`` order.

        Each row is written to the product variable at the same position, so reading the wrong group or stacking out of
        order would publish one cloud property under another's name, and a dropped row or column would leave part of a
        footprint without cloud data. A daily composite has no observation time, so its samples carry no timestamp
        source.

        Builds a fixture whose cloud fraction, optical thickness and top pressure groups hold 0.6, 4.0 and 700.0,
        queries a box just larger than the 4 x 8 grid and asserts a float32 ``(2, 4, 8)`` grid with row 0 at 4.0 and
        row 1 at 700.0, so the unread cloud fraction group appears nowhere, the source ``"viirs_cldprop_d3"`` and no
        timestamp source.
        """
        reader, _ = _make_reader(tmp_path, cf_fill=0.6, cot_fill=4.0, ctp_fill=700.0)
        data, _, _ = _region(reader, _full_bbox(), _T)
        assert data.shape == (2, _N_LAT, _N_LON)
        assert data.dtype == np.float32
        assert np.allclose(data[0], 4.0, atol=1e-4)  # cloud_optical_thickness
        assert np.allclose(data[1], 700.0, atol=1e-3)  # cloud_top_pressure
        samples = reader.samples_in(_full_bbox(), _T)
        assert samples.source == "viirs_cldprop_d3"
        assert samples.timestamp_source is None

    def test_fill_value_comes_from_the_field(self, tmp_path):
        """``VIIRSCldpropD3Reader.samples_in`` masks the fill value each field declares, not a hard-coded -9999.0.

        A file that declares a different sentinel would otherwise have its fill averaged in as data, while a genuine
        value that happens to equal the usual sentinel would be discarded.

        Builds a fixture declaring ``_FillValue`` -999.0 with optical thickness all -9999.0 and top pressure all -999.0,
        and asserts optical thickness is served as -9999.0 and top pressure as NaN.
        """
        reader, _ = _make_reader(tmp_path, mean_fill_value=-999.0, cot_fill=-9999.0, ctp_fill=-999.0)
        data, _, _ = _region(reader, _full_bbox(), _T)
        np.testing.assert_array_equal(data[0], -9999.0)
        assert np.all(np.isnan(data[1]))

    def test_field_without_fill_value_raises(self, tmp_path):
        """``VIIRSCldpropD3Reader.samples_in`` raises ``ValueError`` when a field declares no ``_FillValue``.

        Without a declared sentinel the reader cannot tell fill from data, and guessing one could publish fill as cloud
        values, so it stops instead.

        Builds a fixture whose ``Mean`` fields carry no ``_FillValue`` and asserts the query raises ``ValueError``
        matching "has no _FillValue attribute".
        """
        reader, _ = _make_reader(tmp_path, mean_fill_value=None)
        with pytest.raises(ValueError, match="has no _FillValue attribute"):
            _region(reader, _full_bbox(), _T)


class TestVIIRSCldpropD3ReaderTranspose:
    """Verify the (lon, lat) → (lat, lon) dimension transpose is applied."""

    def test_spatial_shape_after_transpose(self, tmp_path):
        """``VIIRSCldpropD3Reader`` transposes the file's (longitude, latitude) fields to (latitude, longitude).

        CLDPROP_D3 stores its fields longitude first; without the transpose the grid would not match its latitude and
        longitude axes and values would be placed at the wrong locations.

        Stores the 4 lat x 8 lon fixture as (8, 4) with every cell holding ``100 * latitude + longitude``, queries the
        full grid and asserts all 32 cells are served and every sample's values equal the code of the location it is
        served at.
        """
        reader, path = _make_reader(tmp_path)
        _write_position_coded_means(path)
        samples = reader.samples_in(_full_bbox(), _T)
        assert samples.n_samples == _N_LAT * _N_LON
        expected = _position_code(samples.lats, samples.lons)
        assert np.allclose(samples.values, expected[None, :], atol=1e-3)

    def test_partial_lat_subset_selects_correct_rows(self, tmp_path):
        """``VIIRSCldpropD3Reader.samples_in`` serves the rows inside a box over part of the latitudes, at every column.

        A footprint should be served only the cells near it, with their own values; slicing the wrong axis of the
        transposed grid would cut columns instead of rows or serve another row's values.

        Writes ``100 * latitude + longitude`` into every cell, queries a box over the lower half of the fixture's
        latitudes and the full longitude range, and asserts the top row (3.5°, beyond the box and its one-cell padding)
        is not served while all 8 longitudes are, each sample holding the code of its own location.
        """
        reader, path = _make_reader(tmp_path)
        _write_position_coded_means(path)
        half_lat = (_LAT_MIN + _LAT_MAX) / 2
        bbox = BoundingBox(_LAT_MIN - 0.1, half_lat, _LON_MIN - 0.1, _LON_MAX + 0.1)
        data, lats, lons = _region(reader, bbox, _T)
        assert lats.size < _N_LAT
        assert _LAT_MAX not in lats
        assert lons.size == _N_LON
        samples = reader.samples_in(bbox, _T)
        assert np.allclose(samples.values, _position_code(samples.lats, samples.lons)[None, :], atol=1e-3)


class TestVIIRSCldpropD3ReaderRealSubset:
    """The full 1 degree grid of a real CLDPROP_D3_VIIRS_NOAA20 granule (2026-05-27).

    Only the coordinates and the two Mean fields the reader reads are kept, stored as
    in the product with dimensions (longitude, latitude). Pinned values are the source
    granule's own.
    """

    @pytest.fixture
    def reader(self, fmatch_real_data_path):
        return VIIRSCldpropD3Reader(
            [fmatch_real_data_path / "CLDPROP_D3_VIIRS_NOAA20.A2026147.011.2026151000710_subset.nc"]
        )

    def test_grid_is_transposed_to_latitude_by_longitude(self, reader):
        """``VIIRSCldpropD3Reader._load_native`` loads a real granule as a 180 x 360 latitude-by-longitude grid.

        The distributed product stores fields longitude first; the real granule confirms the transpose and the 1 degree
        bin centers, so values land at the right locations.

        Loads the real subset granule and asserts the data shape is (2, 180, 360), latitudes run -89.5 to 89.5 and
        longitudes -179.5 to 179.5.
        """
        native = reader._load_native()
        assert native.data.shape == (2, 180, 360)
        assert (native.lat_axis[0], native.lat_axis[-1]) == (-89.5, 89.5)
        assert (native.lon_axis[0], native.lon_axis[-1]) == (-179.5, 179.5)

    def test_fill_cells_are_nan(self, reader):
        """``VIIRSCldpropD3Reader._load_native`` turns every fill cell of a real granule into NaN.

        Cells with no retrieval carry -9999 in the product; any left as numbers would be averaged into footprints as
        cloud values.

        Loads the real subset granule and asserts the NaN counts equal the granule's fill counts: 16035 for optical
        thickness and 885 for top pressure.
        """
        cot, ctp = reader._load_native().data
        # -9999 fill: 16 035 of the 64 800 cells have no optical thickness, 885 no top pressure.
        assert np.isnan(cot).sum() == 16035
        assert np.isnan(ctp).sum() == 885

    def test_values_at_cells_are_the_granules(self, reader):
        """``VIIRSCldpropD3Reader.samples_in`` serves a real granule's values at the cell they belong to.

        A misplaced or rescaled value would attach another location's cloud properties to a footprint.

        Queries two 1 degree boxes and asserts the cell at (-29.5, 120.5) holds the granule's optical thickness and top
        pressure, and the cell at (30.5, 20.5) has NaN optical thickness and the granule's top pressure.
        """
        samples = reader.samples_in(BoundingBox(-30.0, -29.0, 120.0, 121.0), _T)
        cot, ctp = sample_at(samples, -29.5, 120.5)
        assert cot == pytest.approx(9.74419, abs=1e-4)
        assert ctp == pytest.approx(332.3532, abs=1e-3)
        samples = reader.samples_in(BoundingBox(30.0, 31.0, 20.0, 21.0), _T)
        cot, ctp = sample_at(samples, 30.5, 20.5)
        assert np.isnan(cot)
        assert ctp == pytest.approx(849.7842, abs=1e-3)
