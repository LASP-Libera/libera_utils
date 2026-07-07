"""Unit tests for camera pseudo-footprint segmentation and FMATCH-CAM-CAMTIME assembly.

Coverage:
- helper units: fill detection, block sizing from ground-sampling distance, block
  tiling, corner indices, and centre-pixel (nearest-valid) selection;
- the public ``segment_l1b_camera`` entry point: footprint counts, corner-derived
  bounding boxes, centre-pixel reduction, and the off-Earth policy (partial-coverage
  flag, all-corner-fill drop, centre-pixel substitution, all-fill image);
- end-to-end assembly of a conformant FMATCH-CAM-CAMTIME dataset.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import xarray as xr

from libera_utils.footprint_matching import camera_segmentation as seg
from libera_utils.footprint_matching.camera_segmentation import (
    CAMERA_TIME_NAME,
    PIXEL_X_DIM,
    PIXEL_Y_DIM,
    TARGET_FOOTPRINT_DIAMETER_KM,
    CameraFootprintQualityFlag,
    segment_l1b_camera,
)
from libera_utils.footprint_matching.geometry import L1B_FILL_VALUE
from libera_utils.footprint_matching.product import (
    _assemble_camtime_dataset,
    load_fmatch_definition,
)
from libera_utils.footprint_matching.types import OperationalMode


def _make_l1b_camera(
    lat: np.ndarray,
    lon: np.ndarray,
    *,
    altitude: float = 835_000.0,
    sza: float = 30.0,
    vza: float = 10.0,
    raa: float = 120.0,
) -> xr.Dataset:
    """Build a minimal L1B Daily Camera dataset from (time, x, y) lat/lon grids.

    The non-geolocation angle fields are filled with constants (their exact values do
    not matter for segmentation; they are just carried through to the centre pixel).
    """
    n_time, nx, ny = lat.shape
    dims = (CAMERA_TIME_NAME, PIXEL_X_DIM, PIXEL_Y_DIM)
    times = np.datetime64("2028-02-12T00:00:00", "ns") + np.arange(n_time) * np.timedelta64(10, "ms")

    def const_grid(value: float) -> np.ndarray:
        return np.full((n_time, nx, ny), value, dtype=float)

    return xr.Dataset(
        {
            seg.LATITUDE_NAME: (dims, lat.astype(float)),
            seg.LONGITUDE_NAME: (dims, lon.astype(float)),
            seg.ALTITUDE_NAME: (dims, const_grid(altitude)),
            seg.SOLAR_ZENITH_NAME: (dims, const_grid(sza)),
            seg.VIEWING_ZENITH_NAME: (dims, const_grid(vza)),
            seg.RELATIVE_AZIMUTH_NAME: (dims, const_grid(raa)),
        },
        coords={CAMERA_TIME_NAME: times},
    )


def _grid_image(
    nx: int, ny: int, *, lat0: float, lon0: float, dlat: float, dlon: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return (lat, lon) 2-D grids where pixel [x, y] = (lat0 + x*dlat, lon0 + y*dlon)."""
    xs = lat0 + np.arange(nx) * dlat
    ys = lon0 + np.arange(ny) * dlon
    lat = np.repeat(xs[:, None], ny, axis=1)
    lon = np.repeat(ys[None, :], nx, axis=0)
    return lat, lon


class TestHelpers:
    def test_is_valid_rejects_fill_and_nonfinite(self):
        assert seg._is_valid(10.0)
        assert not seg._is_valid(L1B_FILL_VALUE)
        assert not seg._is_valid(math.nan)
        assert not seg._is_valid(math.inf)

    def test_target_diameter_matches_fov_derivation(self):
        # Documents the derivation so a change to the constants is caught.
        from libera_utils.footprint_matching.geometry import NOMINAL_ALTITUDE_KM
        from libera_utils.footprint_matching.psf import LIBERA_FOV_HALFANGLE_DEG

        expected = 2.0 * NOMINAL_ALTITUDE_KM * math.tan(math.radians(LIBERA_FOV_HALFANGLE_DEG))
        assert TARGET_FOOTPRINT_DIAMETER_KM == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("gsd_km", "expected"),
        [
            (TARGET_FOOTPRINT_DIAMETER_KM, 1),  # one pixel spans the whole footprint
            (TARGET_FOOTPRINT_DIAMETER_KM / 3.0, 3),  # three pixels per side
            (1e9, 1),  # absurdly coarse -> clamped to at least 1
        ],
    )
    def test_block_size_pixels(self, gsd_km, expected):
        assert seg._block_size_pixels(gsd_km) == expected

    def test_iter_blocks_tiles_grid_with_partial_edges(self):
        blocks = seg._iter_blocks(nx=5, ny=4, block=2)
        # 5 -> [0:2, 2:4, 4:5], 4 -> [0:2, 2:4]  => 3 * 2 = 6 blocks
        assert len(blocks) == 6
        assert (slice(0, 2), slice(0, 2)) in blocks
        assert (slice(4, 5), slice(2, 4)) in blocks  # partial edge block

    def test_corner_indices_dedup_for_thin_block(self):
        # A 1-pixel-wide block collapses its four corners to a single pixel.
        assert seg._corner_indices(slice(3, 4), slice(2, 3)) == [(3, 2)]
        assert set(seg._corner_indices(slice(0, 3), slice(0, 3))) == {(0, 0), (0, 2), (2, 0), (2, 2)}

    def test_select_center_pixel_prefers_geometric_center(self):
        valid = np.ones((3, 3), dtype=bool)
        assert seg._select_center_pixel(slice(0, 3), slice(0, 3), valid) == (1, 1, False)

    def test_select_center_pixel_substitutes_nearest_valid(self):
        valid = np.ones((3, 3), dtype=bool)
        valid[1, 1] = False  # centre is fill
        ix, iy, substituted = seg._select_center_pixel(slice(0, 3), slice(0, 3), valid)
        assert substituted
        assert max(abs(ix - 1), abs(iy - 1)) == 1  # a nearest neighbour

    def test_select_center_pixel_none_when_all_fill(self):
        valid = np.zeros((2, 2), dtype=bool)
        assert seg._select_center_pixel(slice(0, 2), slice(0, 2), valid) is None


class TestSegmentation:
    def test_single_pixel_blocks_when_coarse(self):
        # Coarse spacing (~0.5 deg ~ 55 km/pixel) forces block size 1: one footprint
        # per pixel, each bounding box degenerate to that pixel's location.
        lat, lon = _grid_image(3, 4, lat0=0.0, lon0=0.0, dlat=0.5, dlon=0.5)
        ds = _make_l1b_camera(lat[None], lon[None])
        footprints = segment_l1b_camera(ds)

        assert len(footprints) == 3 * 4
        f = footprints[0]
        assert f.q_flags == CameraFootprintQualityFlag(0)
        # Single-pixel block: bbox collapses to the pixel, centre == the pixel.
        assert f.bbox.lat_min == pytest.approx(f.bbox.lat_max)
        assert f.latitude == pytest.approx(lat[f.center_ix, f.center_iy])
        assert f.longitude == pytest.approx(lon[f.center_ix, f.center_iy])

    def test_multi_pixel_blocks_and_bbox_encloses_corners(self):
        # ~0.09 deg spacing at the equator (~10 km/pixel) gives block size 3.
        nx = ny = 7
        lat, lon = _grid_image(nx, ny, lat0=0.0, lon0=0.0, dlat=0.09, dlon=0.09)
        ds = _make_l1b_camera(lat[None], lon[None])
        footprints = segment_l1b_camera(ds)

        # ceil(7/3) == 3 blocks per dimension.
        assert len(footprints) == 3 * 3
        first = footprints[0]
        block_lat = lat[first.slice_x, first.slice_y]
        block_lon = lon[first.slice_x, first.slice_y]
        # Corner-derived box must enclose the block's extent.
        assert first.bbox.lat_min == pytest.approx(block_lat.min())
        assert first.bbox.lat_max == pytest.approx(block_lat.max())
        assert first.bbox.lon_min == pytest.approx(block_lon.min())
        assert first.bbox.lon_max == pytest.approx(block_lon.max())

    def test_partial_coverage_flag_and_shrunk_box(self):
        nx = ny = 7  # block size 3 as above; first block spans pixels [0:3, 0:3]
        lat, lon = _grid_image(nx, ny, lat0=0.0, lon0=0.0, dlat=0.09, dlon=0.09)
        # Fill BOTH corners on the min-latitude edge of the first block (x == 0:
        # pixels [0, 0] and [0, 2]). Each extreme of a rectangle is shared by two
        # corners, so we must remove both to actually shrink the box.
        for iy in (0, 2):
            lat[0, iy] = L1B_FILL_VALUE
            lon[0, iy] = L1B_FILL_VALUE
        ds = _make_l1b_camera(lat[None], lon[None])
        footprints = segment_l1b_camera(ds)

        first = footprints[0]
        assert first.q_flags & CameraFootprintQualityFlag.PARTIAL_COVERAGE
        # Only the x == 2 corners remain, so the box's southern edge moves north.
        assert first.bbox.lat_min == pytest.approx(lat[2, 1])
        assert first.bbox.lat_min > 0.0

    def test_all_corner_fill_block_is_dropped(self):
        nx = ny = 7  # block size 3
        lat, lon = _grid_image(nx, ny, lat0=0.0, lon0=0.0, dlat=0.09, dlon=0.09)
        # Fill all four corners of the first block (0:3, 0:3) but keep its interior.
        for ix, iy in [(0, 0), (0, 2), (2, 0), (2, 2)]:
            lat[ix, iy] = L1B_FILL_VALUE
            lon[ix, iy] = L1B_FILL_VALUE
        ds = _make_l1b_camera(lat[None], lon[None])
        footprints = segment_l1b_camera(ds)

        # One fewer footprint than the clean 3x3 case.
        assert len(footprints) == 3 * 3 - 1

    def test_center_pixel_substituted_flag(self):
        nx = ny = 7  # block size 3
        lat, lon = _grid_image(nx, ny, lat0=0.0, lon0=0.0, dlat=0.09, dlon=0.09)
        # Fill the geometric centre of the first block (pixel [1, 1]); corners stay valid.
        lat[1, 1] = L1B_FILL_VALUE
        lon[1, 1] = L1B_FILL_VALUE
        ds = _make_l1b_camera(lat[None], lon[None])
        footprints = segment_l1b_camera(ds)

        first = footprints[0]
        assert first.q_flags & CameraFootprintQualityFlag.CENTER_PIXEL_SUBSTITUTED
        assert (first.center_ix, first.center_iy) != (1, 1)

    def test_all_fill_image_yields_no_footprints(self):
        lat = np.full((1, 4, 4), L1B_FILL_VALUE)
        lon = np.full((1, 4, 4), L1B_FILL_VALUE)
        ds = _make_l1b_camera(lat, lon)
        assert segment_l1b_camera(ds) == []

    def test_footprints_share_image_time(self):
        lat, lon = _grid_image(3, 3, lat0=0.0, lon0=0.0, dlat=0.5, dlon=0.5)
        # Two images, coarse spacing -> 9 single-pixel footprints per image.
        lat3 = np.stack([lat, lat])
        lon3 = np.stack([lon, lon])
        ds = _make_l1b_camera(lat3, lon3)
        footprints = segment_l1b_camera(ds)

        times = ds[CAMERA_TIME_NAME].values
        # All footprints from image 0 carry the first timestamp (duplicates are expected).
        first_image = [f for f in footprints if f.time == times[0]]
        assert len(first_image) == 9


class TestCamtimeAssembly:
    def test_assemble_conforms_and_carries_real_values(self):
        lat, lon = _grid_image(3, 4, lat0=10.0, lon0=20.0, dlat=0.5, dlon=0.5)
        ds = _make_l1b_camera(lat[None], lon[None], altitude=830_000.0, sza=42.0, vza=8.0, raa=95.0)
        footprints = segment_l1b_camera(ds)

        definition = load_fmatch_definition(OperationalMode.CAM_CAMTIME)
        dataset = _assemble_camtime_dataset(
            footprints,
            definition=definition,
            algorithm_version="0.1.0",
            input_files="synthetic_l1b_cam.nc",
        )

        # Conformance: dtypes/dims/attributes all satisfy the product definition.
        assert definition.check_dataset_conformance(dataset, strict=False) == []

        # Real, segmentation-derived columns match the footprints.
        assert dataset.sizes[CAMERA_TIME_NAME] == len(footprints)
        np.testing.assert_allclose(dataset["latitude"].values, [f.latitude for f in footprints], rtol=1e-4)
        np.testing.assert_allclose(dataset["viewing_zenith_angle"].values, 8.0, rtol=1e-4)
        np.testing.assert_array_equal(dataset["q_flags"].values, [int(f.q_flags) for f in footprints])

        # A placeholder (aggregation-owned) variable is filled with NaN, not real data.
        assert np.all(np.isnan(dataset["sunglint_angle"].values))

    def test_assemble_empty_footprints_raises(self):
        with pytest.raises(ValueError, match="zero pseudo-footprints"):
            _assemble_camtime_dataset([], algorithm_version="0.1.0", input_files="x.nc")
