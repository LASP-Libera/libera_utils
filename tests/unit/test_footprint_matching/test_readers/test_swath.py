"""Unit tests for the shared swath → grid rasterization helpers.

These cover the format-agnostic pieces in
:mod:`libera_utils.footprint_matching.readers._swath` used by the SSF and
CLDPIX readers: longitude normalization, fill / valid-range masking, and the
point-to-grid rasterizer (mean, geometric mean, and mode aggregation).
"""

from __future__ import annotations

import numpy as np
import pytest

from libera_utils.footprint_matching.readers._swath import (
    AGG_LOG_MEAN,
    AGG_MEAN,
    AGG_MODE,
    apply_fill_and_valid_range,
    normalize_longitude,
    rasterize_points_to_grid,
)


class TestNormalizeLongitude:
    def test_0_360_maps_to_minus180_180(self):
        lon = np.array([0.0, 90.0, 180.0, 270.0, 350.0, 359.9])
        out = normalize_longitude(lon)
        assert np.allclose(out, [0.0, 90.0, -180.0, -90.0, -10.0, -0.1], atol=1e-6)

    def test_already_in_range_is_unchanged(self):
        lon = np.array([-179.0, -10.0, 0.0, 45.0, 179.0])
        out = normalize_longitude(lon)
        assert np.allclose(out, lon, atol=1e-6)


class TestApplyFillAndValidRange:
    def test_fill_value_becomes_nan(self):
        raw = np.array([1.0, 2.0, -999.0, 3.0])
        out = apply_fill_and_valid_range(raw, fill_value=-999.0)
        assert np.isnan(out[2])
        assert np.allclose(out[[0, 1, 3]], [1.0, 2.0, 3.0])

    def test_out_of_valid_range_becomes_nan(self):
        raw = np.array([-1.0, 0.0, 5.0, 6.0])
        out = apply_fill_and_valid_range(raw, valid_range=(0.0, 5.0))
        assert np.isnan(out[0])
        assert np.isnan(out[3])
        assert np.allclose(out[[1, 2]], [0.0, 5.0])

    def test_descending_valid_range_is_normalized(self):
        # The CLDPIX pressure field stores valid_range as [1100, 10]; values in
        # between must be kept, not masked.
        raw = np.array([800.0, 50.0, 5.0, 2000.0])
        out = apply_fill_and_valid_range(raw, valid_range=(1100.0, 10.0))
        assert np.allclose(out[[0, 1]], [800.0, 50.0])
        assert np.isnan(out[2])
        assert np.isnan(out[3])

    def test_integer_fill_matches_exactly(self):
        raw = np.array([1, 2, 127, 3], dtype=np.int8)
        out = apply_fill_and_valid_range(raw, fill_value=127)
        assert np.isnan(out[2])

    def test_masked_array_input(self):
        raw = np.ma.array([1.0, 2.0, 3.0], mask=[False, True, False])
        out = apply_fill_and_valid_range(raw)
        assert np.isnan(out[1])
        assert np.allclose(out[[0, 2]], [1.0, 3.0])


class TestRasterizePointsToGrid:
    def _bbox(self):
        return (0.0, 2.0, 0.0, 2.0)

    def test_output_grid_shape_and_coords(self):
        lats = np.array([0.5, 1.5])
        lons = np.array([0.5, 1.5])
        vals = np.array([[1.0, 2.0]])
        data, out_lats, out_lons = rasterize_points_to_grid(
            lats, lons, vals, self._bbox(), cell_size_deg=1.0, aggregations=[AGG_MEAN]
        )
        assert data.shape == (1, 2, 2)
        assert np.allclose(out_lats, [0.5, 1.5])
        assert np.allclose(out_lons, [0.5, 1.5])
        assert data.dtype == np.float32

    def test_mean_aggregation_averages_points_in_a_cell(self):
        # Two points in the same (lower-left) cell average together.
        lats = np.array([0.25, 0.75])
        lons = np.array([0.25, 0.75])
        vals = np.array([[10.0, 20.0]])
        data, _, _ = rasterize_points_to_grid(
            lats, lons, vals, self._bbox(), cell_size_deg=1.0, aggregations=[AGG_MEAN]
        )
        assert data[0, 0, 0] == pytest.approx(15.0)

    def test_empty_cells_are_nan(self):
        lats = np.array([0.5])
        lons = np.array([0.5])
        vals = np.array([[7.0]])
        data, _, _ = rasterize_points_to_grid(
            lats, lons, vals, self._bbox(), cell_size_deg=1.0, aggregations=[AGG_MEAN]
        )
        # Only the lower-left cell is populated; the other three are NaN.
        assert data[0, 0, 0] == pytest.approx(7.0)
        assert np.isnan(data[0, 0, 1])
        assert np.isnan(data[0, 1, 0])
        assert np.isnan(data[0, 1, 1])

    def test_log_mean_is_geometric_mean(self):
        # Geometric mean of 1 and 100 is 10.
        lats = np.array([0.25, 0.75])
        lons = np.array([0.25, 0.75])
        vals = np.array([[1.0, 100.0]])
        data, _, _ = rasterize_points_to_grid(
            lats, lons, vals, self._bbox(), cell_size_deg=1.0, aggregations=[AGG_LOG_MEAN]
        )
        assert data[0, 0, 0] == pytest.approx(10.0)

    def test_log_mean_retains_valid_zero(self):
        # A valid zero (e.g. AOD == 0, "perfectly clear") must NOT be dropped: it is
        # floored to the default 1e-4, so the geometric mean of {0, 100} is
        # exp((log(1e-4) + log(100)) / 2) == 0.1 — far below the drop-the-zero value
        # of 100. This is the core of the #7 fix (no upward bias, cell not NaN).
        lats = np.array([0.25, 0.75])
        lons = np.array([0.25, 0.75])
        vals = np.array([[0.0, 100.0]])
        data, _, _ = rasterize_points_to_grid(
            lats, lons, vals, self._bbox(), cell_size_deg=1.0, aggregations=[AGG_LOG_MEAN]
        )
        assert data[0, 0, 0] == pytest.approx(0.1)

    def test_log_mean_all_zero_cell_is_floor_not_nan(self):
        # An all-zero cell reports ~0 (the floor), not "no coverage" (NaN).
        lats = np.array([0.25, 0.75])
        lons = np.array([0.25, 0.75])
        vals = np.array([[0.0, 0.0]])
        data, _, _ = rasterize_points_to_grid(
            lats, lons, vals, self._bbox(), cell_size_deg=1.0, aggregations=[AGG_LOG_MEAN]
        )
        assert not np.isnan(data[0, 0, 0])
        assert data[0, 0, 0] == pytest.approx(1e-4)

    def test_log_mean_rejects_negatives(self):
        # Negatives are physically invalid and still excluded; the positive survives,
        # and a cell whose only value is negative becomes NaN.
        lats = np.array([0.25, 0.75, 1.5])
        lons = np.array([0.25, 0.75, 1.5])
        vals = np.array([[-5.0, 100.0, -1.0]])
        data, _, _ = rasterize_points_to_grid(
            lats, lons, vals, self._bbox(), cell_size_deg=1.0, aggregations=[AGG_LOG_MEAN]
        )
        assert data[0, 0, 0] == pytest.approx(100.0)  # -5 dropped, 100 kept
        assert np.isnan(data[0, 1, 1])  # only a negative point → no valid coverage

    def test_log_mean_per_variable_floor_override(self):
        # A per-variable log_floor (e.g. a detection limit) changes how a zero is
        # floored: with floor=1.0, geometric mean of {0, 100} is exp(log(100)/2) == 10.
        lats = np.array([0.25, 0.75])
        lons = np.array([0.25, 0.75])
        vals = np.array([[0.0, 100.0]])
        data, _, _ = rasterize_points_to_grid(
            lats, lons, vals, self._bbox(), cell_size_deg=1.0, aggregations=[AGG_LOG_MEAN], log_floors=[1.0]
        )
        assert data[0, 0, 0] == pytest.approx(10.0)

    def test_mode_aggregation_picks_most_common(self):
        # Three points in the lower-left cell: codes [5, 5, 7] → mode 5.
        lats = np.array([0.1, 0.2, 0.3])
        lons = np.array([0.1, 0.2, 0.3])
        vals = np.array([[5.0, 5.0, 7.0]])
        data, _, _ = rasterize_points_to_grid(
            lats, lons, vals, self._bbox(), cell_size_deg=1.0, aggregations=[AGG_MODE]
        )
        assert data[0, 0, 0] == pytest.approx(5.0)

    def test_nan_values_are_ignored(self):
        lats = np.array([0.25, 0.75])
        lons = np.array([0.25, 0.75])
        vals = np.array([[np.nan, 20.0]])
        data, _, _ = rasterize_points_to_grid(
            lats, lons, vals, self._bbox(), cell_size_deg=1.0, aggregations=[AGG_MEAN]
        )
        assert data[0, 0, 0] == pytest.approx(20.0)

    def test_points_outside_bbox_excluded(self):
        lats = np.array([0.5, 50.0])
        lons = np.array([0.5, 50.0])
        vals = np.array([[1.0, 999.0]])
        data, _, _ = rasterize_points_to_grid(
            lats, lons, vals, self._bbox(), cell_size_deg=1.0, aggregations=[AGG_MEAN]
        )
        assert data[0, 0, 0] == pytest.approx(1.0)
        assert not np.any(data == 999.0)

    def test_all_points_outside_returns_all_nan(self):
        lats = np.array([50.0, 60.0])
        lons = np.array([50.0, 60.0])
        vals = np.array([[1.0, 2.0]])
        data, out_lats, out_lons = rasterize_points_to_grid(
            lats, lons, vals, self._bbox(), cell_size_deg=1.0, aggregations=[AGG_MEAN]
        )
        assert data.shape == (1, 2, 2)
        assert np.all(np.isnan(data))

    def test_unknown_aggregation_raises(self):
        lats = np.array([0.5])
        lons = np.array([0.5])
        vals = np.array([[1.0]])
        with pytest.raises(ValueError, match="Unknown aggregation"):
            rasterize_points_to_grid(lats, lons, vals, self._bbox(), cell_size_deg=1.0, aggregations=["bogus"])
