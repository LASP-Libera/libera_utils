"""Unit tests for IGBPReader and the MODIS sinusoidal HDF4 read.

The tests write small MODIS sinusoidal tiles with pyhdf (real tile corners, coarse
pixels) and read them back, so the HDF-EOS2 ``StructMetadata.0`` parsing, the tile
holder and its per-query geolocation all run on real HDF4 files. Served samples are
checked against every pixel centre computed in the test from the sinusoidal formula.

Real MCD12Q1 files can be downloaded from:
    LP DAAC AppEEARS: https://appeears.earthdatacloud.nasa.gov/
    LP DAAC Data Pool: https://e4ftl01.cr.usgs.gov/MOTA/MCD12Q1.061/
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest
from pyhdf.error import HDF4Error

from libera_utils.footprint_matching.readers._hdf4_io import read_modis_sinusoidal_hdf4
from libera_utils.footprint_matching.readers.igbp import IGBPReader
from libera_utils.footprint_matching.types import BoundingBox, SampleSet, TimeMatch
from tests.test_data.footprint_matching.fixtures import (
    MODIS_SPHERE_RADIUS_M,
    MODIS_TILE_SIZE_M,
    make_modis_sinusoidal_hdf4_fixture,
    modis_tile_corners,
)

_T = np.datetime64("2026-01-01T00:00:00", "ns")
# Pixels per tile side in the synthetic tiles: about 18.5 km pixels, so a footprint-sized
# box selects a handful and a whole tile is small.
_N = 60


def _codes(seed: int, n: int = _N) -> np.ndarray:
    """IGBP-like uint8 codes: mostly 1..17, with some 0, 18 and 255."""
    rng = np.random.default_rng(seed)
    return rng.choice(np.array([0, 1, 5, 9, 12, 17, 18, 255], dtype=np.uint8), size=(n, n))


def _tile(tmp_path, h: int, v: int, codes: np.ndarray | None = None, **kwargs):
    ul, lr = modis_tile_corners(h, v)
    data = _codes(h * 100 + v) if codes is None else codes
    path = make_modis_sinusoidal_hdf4_fixture(
        tmp_path / f"MCD12Q1.h{h:02d}v{v:02d}.hdf", data, ul_m=ul, lr_m=lr, **kwargs
    )
    return path, data


def _brute_force(tiles, bbox: BoundingBox) -> np.ndarray:
    """Every pixel centre inside ``bbox``, as sorted ``(lat, lon, code)`` rows."""
    rows = []
    for (h, v), data in tiles.items():
        (x_ul, y_ul), (x_lr, y_lr) = modis_tile_corners(h, v)
        n_rows, n_cols = data.shape
        dx, dy = (x_lr - x_ul) / n_cols, (y_lr - y_ul) / n_rows
        lat_rad = (y_ul + (np.arange(n_rows) + 0.5) * dy) / MODIS_SPHERE_RADIUS_M
        x = x_ul + (np.arange(n_cols) + 0.5) * dx
        with np.errstate(divide="ignore", invalid="ignore"):
            lon = np.degrees(x[np.newaxis, :] / (MODIS_SPHERE_RADIUS_M * np.cos(lat_rad)[:, np.newaxis]))
        lat = np.broadcast_to(np.degrees(lat_rad)[:, np.newaxis], lon.shape)
        inside = (lat >= bbox.lat_min) & (lat <= bbox.lat_max) & (lon >= bbox.lon_min) & (lon <= bbox.lon_max)
        rows.append(np.column_stack([lat[inside], lon[inside], data[inside].astype(np.float64)]))
    out = np.concatenate(rows) if rows else np.empty((0, 3))
    return out[np.lexsort((out[:, 2], out[:, 1], out[:, 0]))]


def _served(lats, lons, values) -> np.ndarray:
    """Served samples as sorted ``(lat, lon, code)`` rows, NaN codes compared as their raw class."""
    out = np.column_stack([lats, lons, values[0].astype(np.float64)])
    return out[np.lexsort((out[:, 2], out[:, 1], out[:, 0]))]


def _assert_matches(reader: IGBPReader, tiles, bbox: BoundingBox) -> None:
    lats, lons, values, times = reader._native.query_box(bbox)
    assert times is None
    expected = _brute_force(tiles, bbox)
    # The reader masks codes outside 1..17 to NaN; do the same to the expected codes.
    codes = expected[:, 2]
    expected[:, 2] = np.where((codes >= 1) & (codes <= 17), codes, np.nan)
    got = _served(lats, lons, values)
    assert got.shape == expected.shape
    np.testing.assert_allclose(got[:, :2], expected[:, :2], rtol=0, atol=1e-9)
    np.testing.assert_array_equal(np.sort(got[:, 2]), np.sort(expected[:, 2]))


class TestIGBPReaderClassAttributes:
    def test_reader_key(self):
        assert IGBPReader.READER_KEY == "igbp"

    def test_resolution_km(self):
        # The MCD12Q1 sinusoidal grid cell, 463.3 m.
        assert IGBPReader.RESOLUTION_KM == 0.463

    def test_one_categorical_variable(self):
        assert len(IGBPReader.VARIABLES) == 1
        assert IGBPReader.VARIABLES[0].name == "surface_type"
        assert IGBPReader.VARIABLES[0].n_categories == 20

    def test_time_match_is_daily(self):
        # The yearly land-cover map is the match for the whole staged period.
        assert IGBPReader.TIME_MATCH is TimeMatch.DAILY


class TestReadModisSinusoidal:
    def test_geometry_is_parsed_from_struct_metadata(self, tmp_path):
        path, data = _tile(tmp_path, 18, 4)
        codes, geometry = read_modis_sinusoidal_hdf4(str(path), "LC_Type1")
        np.testing.assert_array_equal(codes, data)
        assert codes.dtype == np.uint8
        (x_ul, y_ul), _ = modis_tile_corners(18, 4)
        # Corners are written to the micrometre, as MODIS StructMetadata stores them.
        assert geometry.x_ul == pytest.approx(x_ul, abs=1e-5)
        assert geometry.y_ul == pytest.approx(y_ul, abs=1e-5)
        assert geometry.dx == pytest.approx(MODIS_TILE_SIZE_M / _N)
        assert geometry.dy == pytest.approx(-MODIS_TILE_SIZE_M / _N)
        assert geometry.radius_m == pytest.approx(MODIS_SPHERE_RADIUS_M)

    @pytest.mark.parametrize("field", ["XDim", "YDim", "UpperLeftPointMtrs", "LowerRightMtrs", "Projection"])
    def test_missing_field_raises(self, tmp_path, field):
        path, _ = _tile(tmp_path, 18, 4, omit=(field,))
        with pytest.raises(ValueError, match=f"has no {field} field"):
            read_modis_sinusoidal_hdf4(str(path), "LC_Type1")

    def test_missing_proj_params_raises(self, tmp_path):
        path, _ = _tile(tmp_path, 18, 4, omit=("ProjParams",))
        with pytest.raises(ValueError, match="has no ProjParams field"):
            read_modis_sinusoidal_hdf4(str(path), "LC_Type1")

    def test_error_quotes_struct_metadata(self, tmp_path):
        path, _ = _tile(tmp_path, 18, 4, omit=("XDim",))
        with pytest.raises(ValueError, match="StructMetadata.0 begins: .*GridStructure"):
            read_modis_sinusoidal_hdf4(str(path), "LC_Type1")

    def test_non_sinusoidal_projection_raises(self, tmp_path):
        path, _ = _tile(tmp_path, 18, 4, projection="GCTP_GEO")
        with pytest.raises(ValueError, match="Projection=GCTP_GEO, not the sinusoidal GCTP_SNSOID"):
            read_modis_sinusoidal_hdf4(str(path), "LC_Type1")

    @pytest.mark.parametrize("radius", ["0", "-6371007.181"])
    def test_non_positive_radius_raises(self, tmp_path, radius):
        path, _ = _tile(tmp_path, 18, 4, proj_params=f"{radius},0,0,0,0,0,0,0,0,0,0,0,0")
        with pytest.raises(ValueError, match="not a positive length"):
            read_modis_sinusoidal_hdf4(str(path), "LC_Type1")

    def test_dimension_mismatch_raises(self, tmp_path):
        path, _ = _tile(tmp_path, 18, 4, xdim=_N + 1)
        with pytest.raises(ValueError, match=f"XDim={_N + 1} but the LC_Type1 SDS has shape"):
            read_modis_sinusoidal_hdf4(str(path), "LC_Type1")

    def test_missing_sds_raises_hdf4_error(self, tmp_path):
        path, _ = _tile(tmp_path, 18, 4)
        with pytest.raises(HDF4Error):
            read_modis_sinusoidal_hdf4(str(path), "LC_Type2")


class TestIGBPTiles:
    @staticmethod
    def _reader(tmp_path, tile_ids):
        tiles = {}
        paths = []
        for h, v in tile_ids:
            path, data = _tile(tmp_path, h, v)
            tiles[(h, v)] = data
            paths.append(path)
        return IGBPReader(paths), tiles

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_random_boxes_match_every_pixel_centre(self, tmp_path, seed):
        reader, tiles = self._reader(tmp_path, [(17, 3), (18, 3), (19, 3), (17, 4), (18, 4), (19, 4)])
        rng = np.random.default_rng(seed)
        for _ in range(20):
            lat0 = rng.uniform(31.0, 58.0)
            lon0 = rng.uniform(-25.0, 30.0)
            _assert_matches(reader, tiles, BoundingBox(lat0, lat0 + rng.uniform(0.1, 3.0), lon0, lon0 + 2.0))

    @pytest.mark.parametrize(
        "bbox",
        [
            # Wholly east of the prime meridian, straddling the h18/h19 edge near the
            # poleward rows: x reaches lowest at the pole-most row.
            BoundingBox(60.0, 62.0, 20.0, 22.0),
            # Wholly west, straddling the h16/h17 edge: x reaches highest at the pole-most row.
            BoundingBox(60.0, 62.0, -22.0, -20.0),
        ],
    )
    def test_high_latitude_boxes_on_one_side_of_the_prime_meridian(self, tmp_path, bbox):
        reader, tiles = self._reader(tmp_path, [(15, 2), (16, 2), (17, 2), (18, 2), (19, 2), (20, 2)])
        _assert_matches(reader, tiles, bbox)
        assert reader._native.query_box(bbox)[0].size > 0

    def test_box_straddling_the_equator(self, tmp_path):
        reader, tiles = self._reader(tmp_path, [(18, 8), (19, 8), (18, 9), (19, 9)])
        bbox = BoundingBox(-1.0, 1.0, -0.5, 1.5)
        _assert_matches(reader, tiles, bbox)
        lats = reader._native.query_box(bbox)[0]
        assert lats.min() < 0.0 < lats.max()

    def test_dateline_box_serves_h35_and_h00(self, tmp_path):
        reader, tiles = self._reader(tmp_path, [(0, 8), (35, 8)])
        samples = reader.samples_in(BoundingBox(4.0, 6.0, 179.0, 181.0, wraps_dateline=True), _T)
        assert samples.lons.max() > 179.0
        assert samples.lons.min() < -179.0
        for half in (BoundingBox(4.0, 6.0, 179.0, 180.0), BoundingBox(4.0, 6.0, -180.0, -179.0)):
            _assert_matches(reader, tiles, half)

    def test_polar_box(self, tmp_path):
        reader, tiles = self._reader(tmp_path, [(17, 0), (18, 0)])
        bbox = BoundingBox(85.0, 90.0, -180.0, 180.0, is_polar=True)
        _assert_matches(reader, tiles, bbox)
        assert reader._native.query_box(bbox)[0].size > 0

    def test_no_off_globe_pixel_is_served(self, tmp_path):
        # Tiles h06/h29 at 50-60 deg straddle the edge of the sinusoidal globe: part of each
        # maps beyond +-180 deg longitude.
        reader, tiles = self._reader(tmp_path, [(6, 3), (29, 3)])
        lats, lons, _, _ = reader._native.query_box(BoundingBox(50.0, 60.0, -180.0, 180.0))
        assert lons.size > 0
        assert np.all(np.abs(lons) <= 180.0)
        _assert_matches(reader, tiles, BoundingBox(50.0, 60.0, -180.0, 180.0))

    def test_tile_outside_the_box_contributes_nothing(self, tmp_path):
        reader, tiles = self._reader(tmp_path, [(18, 4)])
        assert reader._native.query_box(BoundingBox(-10.0, -8.0, 100.0, 102.0))[0].size == 0
        assert reader.samples_in(BoundingBox(-10.0, -8.0, 100.0, 102.0), _T).values.shape == (1, 0)

    def test_samples_in_serves_the_padded_box(self, tmp_path):
        reader, _ = self._reader(tmp_path, [(18, 4)])
        samples = reader.samples_in(BoundingBox(44.0, 45.0, 5.0, 6.0), _T)
        assert isinstance(samples, SampleSet)
        assert samples.source == "igbp"
        assert samples.values.dtype == np.float32
        assert samples.values.shape == (1, samples.n_samples)
        assert samples.n_samples > 0
        assert samples.resolution_km == 0.463


class TestIGBPCodes:
    def test_codes_outside_the_igbp_domain_are_nan(self, tmp_path):
        # MCD12Q1 LC_Type1 valid IGBP codes are 1..17 (17 = Water Bodies), aligned 1:1 with
        # the low IGBPSurfaceType members; 0, 18 and the 255 fill are masked to NaN.
        rows = np.array([0, 17, 18, 5, 255, 1], dtype=np.uint8)
        codes = np.repeat(rows, _N // rows.size)[:, np.newaxis].repeat(_N, axis=1)
        path, _ = _tile(tmp_path, 18, 4, codes=codes)
        samples = IGBPReader(path).samples_in(BoundingBox(40.0, 50.0, 0.0, 14.0), _T)
        finite = samples.values[np.isfinite(samples.values)]
        assert set(np.unique(finite).tolist()) == {1.0, 5.0, 17.0}
        assert np.isnan(samples.values).any()

    def test_all_fill_tile_serves_only_nan(self, tmp_path):
        path, _ = _tile(tmp_path, 18, 4, codes=np.full((_N, _N), 255, dtype=np.uint8))
        samples = IGBPReader(path).samples_in(BoundingBox(44.0, 45.0, 5.0, 6.0), _T)
        assert samples.n_samples > 0
        assert np.all(np.isnan(samples.values))


class TestIGBPTileHolder:
    def test_holds_codes_as_uint8_and_one_dimensional_geometry(self, tmp_path):
        paths = [_tile(tmp_path, h, 4)[0] for h in (17, 18, 19)]
        native = IGBPReader(paths)._native
        assert native.data.dtype == np.uint8
        assert native.data.shape == (3, _N, _N)
        for name in ("x_ul", "y_ul", "dx", "dy"):
            assert getattr(native, name).shape == (3,)
        assert isinstance(native.radius_m, float)

    def test_load_memory_is_about_one_byte_per_pixel(self, tmp_path):
        n = 480
        paths = [_tile(tmp_path, h, 4, codes=np.ones((n, n), dtype=np.uint8))[0] for h in (17, 18)]
        reader = IGBPReader(paths)
        tracemalloc.start()
        try:
            native = reader._native
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        n_pixels = native.data.size
        # The held codes plus one tile read in flight; a per-pixel float64 lat/lon cloud
        # would be about 20 bytes per pixel.
        assert peak < 3 * n_pixels

    def test_tiles_with_different_radii_raise(self, tmp_path):
        first, _ = _tile(tmp_path, 17, 4)
        second, _ = _tile(tmp_path, 18, 4, proj_params="6378137.0,0,0,0,0,0,0,0,0,0,0,0,0")
        with pytest.raises(ValueError, match="sphere radius .* differs from the first staged tile"):
            _ = IGBPReader([first, second])._native

    def test_tiles_with_different_shapes_raise(self, tmp_path):
        first, _ = _tile(tmp_path, 17, 4)
        second, _ = _tile(tmp_path, 18, 4, codes=np.ones((_N // 2, _N // 2), dtype=np.uint8))
        with pytest.raises(ValueError, match="tile shape .* differs from the first staged tile"):
            _ = IGBPReader([first, second])._native

    def test_non_uint8_codes_raise(self, tmp_path):
        path, _ = _tile(tmp_path, 18, 4, codes=np.ones((_N, _N), dtype=np.int16))
        with pytest.raises(ValueError, match="LC_Type1 is int16"):
            _ = IGBPReader(path)._native
