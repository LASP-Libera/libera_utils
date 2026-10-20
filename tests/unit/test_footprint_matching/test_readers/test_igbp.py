"""Unit tests for IGBPReader.

pyhdf requires a system HDF4 C library which is unavailable in this container.
These tests mock ``read_hdf4_lat_lon_grid`` to return synthetic numpy arrays,
allowing the reader's spatial subsetting logic to be tested without HDF4.

Real MCD12Q1 files can be downloaded from:
    LP DAAC AppEEARS: https://appeears.earthdatacloud.nasa.gov/
    LP DAAC Data Pool: https://e4ftl01.cr.usgs.gov/MOTA/MCD12Q1.061/
"""

from __future__ import annotations

import numpy as np

from libera_utils.footprint_matching.readers.igbp import IGBPReader
from libera_utils.footprint_matching.types import BoundingBox, SampleSet, TimeMatch

_T = np.datetime64("2026-01-01T00:00:00", "ns")

# ---------------------------------------------------------------------------
# Synthetic HDF4 data used in all IGBPReader tests
# ---------------------------------------------------------------------------


def _make_synthetic_igbp_data():
    """Return (data, lats_2d, lons_2d) mimicking a small MCD12Q1 tile.

    ``read_modis_sinusoidal_hdf4`` returns the data and **2-D per-pixel**
    coordinate grids (the sinusoidal projection has no 1-D lat/lon axes), so the
    synthetic fixture must do the same for the reader's flatten-to-points path.

    Grid: 6 rows × 6 cols, lats 0–3°, lons 10–13°. Category values cycle through
    IGBP classes 0–5 along each row for easy assertion.
    """
    lats_1d = np.linspace(0.0, 3.0, 6)  # ascending latitudes
    lons_1d = np.linspace(10.0, 13.0, 6)
    lons_2d, lats_2d = np.meshgrid(lons_1d, lats_1d)  # both (6, 6)
    # 6 × 6 grid; values cycle through IGBP classes 0–5
    data = np.tile(np.arange(6, dtype=np.float32), (6, 1))
    return data, lats_2d, lons_2d


class TestIGBPReaderClassAttributes:
    def test_reader_key(self):
        # Targets the reader's registry key; asserts READER_KEY equals the expected "igbp".
        assert IGBPReader.READER_KEY == "igbp"

    def test_resolution_km(self):
        # Targets the declared source resolution; asserts RESOLUTION_KM is 1.0 km (MCD12Q1).
        assert IGBPReader.RESOLUTION_KM == 1.0

    def test_variables_has_one_entry(self):
        # Targets the reader's variable declaration; asserts VARIABLES holds exactly one entry.
        assert len(IGBPReader.VARIABLES) == 1

    def test_variable_name_is_surface_type(self):
        # Targets the single variable's identity; asserts its name is "surface_type".
        assert IGBPReader.VARIABLES[0].name == "surface_type"

    def test_n_categories_is_20(self):
        # Targets the categorical variable's class count; asserts n_categories is 20 (IGBP classes).
        assert IGBPReader.VARIABLES[0].n_categories == 20

    def test_time_match_is_daily(self):
        # The yearly land-cover map is the match for the whole staged period.
        assert IGBPReader.TIME_MATCH is TimeMatch.DAILY


class TestIGBPReaderSamples:
    @staticmethod
    def _patch(monkeypatch, data, lats, lons):
        monkeypatch.setattr(
            "libera_utils.footprint_matching.readers.igbp.read_modis_sinusoidal_hdf4",
            lambda **kwargs: (data, lats, lons),
        )

    def test_returns_samples_with_correct_source(self, tmp_path, monkeypatch):
        data, lats, lons = _make_synthetic_igbp_data()
        self._patch(monkeypatch, data, lats, lons)
        reader = IGBPReader(tmp_path / "MCD12Q1.hdf")
        samples = reader.samples_in(BoundingBox(0.5, 2.5, 10.5, 12.5), _T)
        assert isinstance(samples, SampleSet)
        assert samples.source == "igbp"
        assert samples.timestamp_source is None
        # Single-variable contract: one value row, every pixel in the box served.
        assert samples.values.shape == (1, samples.n_samples)
        assert samples.n_samples > 0
        assert np.isfinite(samples.values).any()

    def test_every_pixel_in_the_box_is_served_at_its_own_position(self, tmp_path, monkeypatch):
        data, lats, lons = _make_synthetic_igbp_data()
        self._patch(monkeypatch, data, lats, lons)
        reader = IGBPReader(tmp_path / "MCD12Q1.hdf")
        bbox = BoundingBox(0.5, 2.5, 10.5, 12.5)
        samples = reader.samples_in(bbox, _T)
        inside = (lats >= 0.5) & (lats <= 2.5) & (lons >= 10.5) & (lons <= 12.5)
        assert samples.n_samples == int(inside.sum())
        assert set(zip(samples.lats.tolist(), samples.lons.tolist())) == set(
            zip(lats[inside].tolist(), lons[inside].tolist())
        )
        # Pixel values travel with their positions (no per-cell mode).
        expected = {(la, lo): v for la, lo, v in zip(lats[inside], lons[inside], data[inside])}
        for la, lo, v in zip(samples.lats, samples.lons, samples.values[0]):
            assert v == expected[(la, lo)]

    def test_empty_result_when_no_pixels_in_bbox(self, tmp_path, monkeypatch):
        data, lats, lons = _make_synthetic_igbp_data()
        self._patch(monkeypatch, data, lats, lons)
        reader = IGBPReader(tmp_path / "MCD12Q1.hdf")
        samples = reader.samples_in(BoundingBox(50.0, 52.0, 50.0, 52.0), _T)
        assert samples.n_samples == 0
        assert samples.values.shape == (1, 0)

    def test_fill_value_is_not_a_category(self, tmp_path, monkeypatch):
        # A tile that is entirely fill (255) must serve only NaN values, never 255 as a
        # class -- fill is masked to NaN at load.
        data, lats, lons = _make_synthetic_igbp_data()
        data = np.full_like(data, 255.0)
        self._patch(monkeypatch, data, lats, lons)
        reader = IGBPReader(tmp_path / "MCD12Q1.hdf")
        samples = reader.samples_in(BoundingBox(0.5, 2.5, 10.5, 12.5), _T)
        assert samples.n_samples > 0
        assert np.all(np.isnan(samples.values))

    def test_out_of_domain_codes_are_masked_to_nan(self, tmp_path, monkeypatch):
        # Regression: MCD12Q1 LC_Type1 valid IGBP codes are 1..17 (17 = Water Bodies),
        # which align 1:1 with the low IGBPSurfaceType members. Codes outside that domain
        # -- a stray 0 (which calculate_trmm_surface_type rejects) or an 18/255 -- must be
        # masked to NaN so the modal class fed downstream is always enum-consistent.
        _, lats, lons = _make_synthetic_igbp_data()
        # Row-constant codes: 0 (invalid), 17 (valid, water), 18 (invalid), 5 (valid),
        # 255 (fill), 1 (valid). Shape (6, 6) to match the coordinate grids.
        rows = np.array([0.0, 17.0, 18.0, 5.0, 255.0, 1.0], dtype=np.float32)
        data = np.repeat(rows[:, np.newaxis], 6, axis=1)
        self._patch(monkeypatch, data, lats, lons)
        reader = IGBPReader(tmp_path / "MCD12Q1.hdf")
        values = reader._load_native().values
        finite = values[np.isfinite(values)]
        # Only the in-domain codes survive; 0, 18, and 255 are gone.
        assert set(np.unique(finite).tolist()) == {1.0, 5.0, 17.0}

    def test_values_dtype_is_float32(self, tmp_path, monkeypatch):
        data, lats, lons = _make_synthetic_igbp_data()
        self._patch(monkeypatch, data, lats, lons)
        reader = IGBPReader(tmp_path / "MCD12Q1.hdf")
        samples = reader.samples_in(BoundingBox(0.5, 2.5, 10.5, 12.5), _T)
        assert samples.values.dtype == np.float32

    def test_several_tiles_are_concatenated(self, tmp_path, monkeypatch):
        # Two staged MODIS tiles (one file each) form one point cloud.
        data, lats, lons = _make_synthetic_igbp_data()
        self._patch(monkeypatch, data, lats, lons)
        reader = IGBPReader([tmp_path / "h09v05.hdf", tmp_path / "h10v05.hdf"])
        assert reader._load_native().lats.size == 2 * lats.size
