"""Unit tests for IGBPReader and the MODIS sinusoidal HDF4 read.

The tests write small MODIS sinusoidal tiles with pyhdf (real tile corners, coarse
pixels) and read them back, so the HDF-EOS2 ``StructMetadata.0`` parsing, the tile
holder and its per-query geolocation all run on real HDF4 files. Served samples are
checked against every pixel center computed in the test from the sinusoidal formula.

Real MCD12Q1 files can be downloaded from:
    LP DAAC AppEEARS: https://appeears.earthdatacloud.nasa.gov/
    LP DAAC Data Pool: https://e4ftl01.cr.usgs.gov/MOTA/MCD12Q1.061/
"""

import tracemalloc

import numpy as np
import pyhdf.SD as sd
import pytest
from pyhdf.error import HDF4Error

from libera_utils.footprint_matching.readers._hdf4_io import read_modis_sinusoidal_hdf4
from libera_utils.footprint_matching.readers.igbp import IGBPReader
from libera_utils.footprint_matching.types import BoundingBox, SampleSet, TimeMatch
from tests.plugins.footprint_matching_fixtures import (
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
    """Every pixel center inside ``bbox``, as sorted ``(lat, lon, code)`` rows."""
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
        """``IGBPReader`` registers under the reader key ``"igbp"``.

        The key names the reader in the registry and is stamped as the ``source`` of every sample set it serves, so
        a different value would break lookups of the reader and attribution of its samples.

        Asserts the class attribute ``READER_KEY`` equals ``"igbp"``.
        """
        assert IGBPReader.READER_KEY == "igbp"

    def test_resolution_km(self):
        """``IGBPReader.RESOLUTION_KM`` is 0.463 km, the MCD12Q1 sinusoidal grid cell of 463.3 m.

        The coverage measure credits each served pixel with a cell of this side, and the query box is padded by half
        of it; a wrong value misstates how much of a footprint the land-cover pixels cover.

        Asserts the class attribute equals 0.463.
        """
        # The MCD12Q1 sinusoidal grid cell, 463.3 m.
        assert IGBPReader.RESOLUTION_KM == 0.463

    def test_one_categorical_variable(self):
        """``IGBPReader`` declares one variable, ``surface_type``, with 20 categories.

        The PSF aggregation histograms the class codes into ``n_categories`` bins sized to the 20-class
        ``IGBPSurfaceType`` domain, so a different count would misalign the bins with the surface-type enum.

        Asserts ``VARIABLES`` has length one, its name is ``surface_type`` and its ``n_categories`` is 20.
        """
        assert len(IGBPReader.VARIABLES) == 1
        assert IGBPReader.VARIABLES[0].name == "surface_type"
        assert IGBPReader.VARIABLES[0].n_categories == 20

    def test_time_match_is_daily(self):
        """``IGBPReader`` matches in time as ``TimeMatch.DAILY``.

        MCD12Q1 is a yearly product with no time axis, so every staged tile applies to every footprint; selecting by
        time would discard land cover that does apply.

        Asserts ``TIME_MATCH`` is ``TimeMatch.DAILY``.
        """
        # The yearly land-cover map is the match for the whole staged period.
        assert IGBPReader.TIME_MATCH is TimeMatch.DAILY


class TestReadModisSinusoidal:
    def test_geometry_is_parsed_from_struct_metadata(self, tmp_path):
        """``read_modis_sinusoidal_hdf4`` returns the stored codes and the tile geometry from ``StructMetadata.0``.

        MODIS tiles store no latitude or longitude arrays; every pixel position is derived from the corners, pixel
        size and sphere radius parsed here, so a parsing error misplaces every land-cover pixel.

        Writes a synthetic h18v04 tile and asserts the codes come back unchanged as uint8, the upper-left corner
        matches the tile's corner, the pixel sizes are the tile size over the pixel count (``dy`` negative), and the
        radius is the MODIS sphere radius.
        """
        path, data = _tile(tmp_path, 18, 4)
        codes, geometry = read_modis_sinusoidal_hdf4(str(path), "LC_Type1")
        np.testing.assert_array_equal(codes, data)
        assert codes.dtype == np.uint8
        (x_ul, y_ul), _ = modis_tile_corners(18, 4)
        # Corners are written to the micrometer, as MODIS StructMetadata stores them.
        assert geometry.x_ul == pytest.approx(x_ul, abs=1e-5)
        assert geometry.y_ul == pytest.approx(y_ul, abs=1e-5)
        assert geometry.dx == pytest.approx(MODIS_TILE_SIZE_M / _N)
        assert geometry.dy == pytest.approx(-MODIS_TILE_SIZE_M / _N)
        assert geometry.radius_m == pytest.approx(MODIS_SPHERE_RADIUS_M)

    @pytest.mark.parametrize("field", ["XDim", "YDim", "UpperLeftPointMtrs", "LowerRightMtrs", "Projection"])
    def test_missing_field_raises(self, tmp_path, field):
        """``read_modis_sinusoidal_hdf4`` raises ``ValueError`` naming a geometry field not in ``StructMetadata.0``.

        Without any one of these fields the pixels cannot be located, so the read must stop with a message saying
        which field is absent rather than fail obscurely or serve misplaced pixels.

        Parametrized over ``XDim``, ``YDim``, ``UpperLeftPointMtrs``, ``LowerRightMtrs`` and ``Projection``: each run
        writes a tile without that field and asserts the error says it has no such field.
        """
        path, _ = _tile(tmp_path, 18, 4, omit=(field,))
        with pytest.raises(ValueError, match=f"has no {field} field"):
            read_modis_sinusoidal_hdf4(str(path), "LC_Type1")

    def test_missing_proj_params_raises(self, tmp_path):
        """``read_modis_sinusoidal_hdf4`` raises ``ValueError`` when ``StructMetadata.0`` has no ``ProjParams``.

        ``ProjParams`` carries the sphere radius that converts sinusoidal meters to degrees, so without it no pixel
        can be geolocated.

        Writes a tile without ``ProjParams`` and asserts the error says it has no ``ProjParams`` field.
        """
        path, _ = _tile(tmp_path, 18, 4, omit=("ProjParams",))
        with pytest.raises(ValueError, match="has no ProjParams field"):
            read_modis_sinusoidal_hdf4(str(path), "LC_Type1")

    def test_error_quotes_struct_metadata(self, tmp_path):
        """A ``read_modis_sinusoidal_hdf4`` geometry error quotes the start of ``StructMetadata.0``.

        Quoting what the file does carry lets a user see why the geometry could not be parsed without opening the
        HDF4 file themselves.

        Writes a tile without ``XDim`` and asserts the message contains ``StructMetadata.0 begins:`` followed by the
        attribute's ``GridStructure`` text.
        """
        path, _ = _tile(tmp_path, 18, 4, omit=("XDim",))
        with pytest.raises(ValueError, match="StructMetadata.0 begins: .*GridStructure"):
            read_modis_sinusoidal_hdf4(str(path), "LC_Type1")

    def test_error_quotes_only_the_first_400_characters(self, tmp_path):
        """A ``read_modis_sinusoidal_hdf4`` geometry error quotes at most 400 characters of ``StructMetadata.0``.

        The attribute runs to tens of kilobytes in real tiles; quoting all of it would bury the error under
        metadata.

        Overwrites a tile's ``StructMetadata.0`` with long text lacking ``XDim`` and asserts the message contains the
        first 400 characters but not the first 401.
        """
        path, _ = _tile(tmp_path, 18, 4)
        long_text = "GROUP=GridStructure\n" + "\tSphereCode=-1\n" * 100
        hdf = sd.SD(str(path), sd.SDC.WRITE)
        try:
            hdf.attr("StructMetadata.0").set(sd.SDC.CHAR8, long_text)
        finally:
            hdf.end()
        with pytest.raises(ValueError, match="has no XDim field") as excinfo:
            read_modis_sinusoidal_hdf4(str(path), "LC_Type1")
        assert repr(long_text[:400]) in str(excinfo.value)
        assert repr(long_text[:401]) not in str(excinfo.value)

    def test_non_sinusoidal_projection_raises(self, tmp_path):
        """``read_modis_sinusoidal_hdf4`` raises ``ValueError`` for a projection other than ``GCTP_SNSOID``.

        The reader geolocates pixels only with the sinusoidal formulas, so a tile in another projection would be
        served at wrong positions if it were accepted.

        Writes a tile declaring ``GCTP_GEO`` and asserts the error names that projection and the expected one.
        """
        path, _ = _tile(tmp_path, 18, 4, projection="GCTP_GEO")
        with pytest.raises(ValueError, match="Projection=GCTP_GEO, not the sinusoidal GCTP_SNSOID"):
            read_modis_sinusoidal_hdf4(str(path), "LC_Type1")

    @pytest.mark.parametrize("radius", ["0", "-6371007.181"])
    def test_non_positive_radius_raises(self, tmp_path, radius):
        """``read_modis_sinusoidal_hdf4`` raises ``ValueError`` when the ``ProjParams`` sphere radius is not positive.

        Latitude is ``y / R``, so a zero radius divides by zero and a negative one flips the hemispheres; neither
        gives usable positions.

        Parametrized over a zero and a negative radius: each run writes a tile with that radius and asserts the
        error says it is not a positive length.
        """
        path, _ = _tile(tmp_path, 18, 4, proj_params=f"{radius},0,0,0,0,0,0,0,0,0,0,0,0")
        with pytest.raises(ValueError, match="not a positive length"):
            read_modis_sinusoidal_hdf4(str(path), "LC_Type1")

    def test_dimension_mismatch_raises(self, tmp_path):
        """``read_modis_sinusoidal_hdf4`` raises ``ValueError`` when ``XDim`` disagrees with the SDS shape.

        The pixel size is the corner span divided by ``XDim``/``YDim``, so a mismatch would give every pixel a wrong
        size and position.

        Writes a tile whose ``XDim`` is one more than the column count and asserts the error quotes both values.
        """
        path, _ = _tile(tmp_path, 18, 4, xdim=_N + 1)
        with pytest.raises(ValueError, match=f"XDim={_N + 1} but the LC_Type1 SDS has shape"):
            read_modis_sinusoidal_hdf4(str(path), "LC_Type1")

    def test_missing_sds_raises_hdf4_error(self, tmp_path):
        """``read_modis_sinusoidal_hdf4`` raises ``HDF4Error`` when the requested SDS is not in the file.

        Asking for the wrong dataset must fail rather than return other data.

        Writes a tile holding ``LC_Type1``, requests ``LC_Type2`` and asserts ``HDF4Error`` is raised.
        """
        path, _ = _tile(tmp_path, 18, 4)
        with pytest.raises(HDF4Error):
            read_modis_sinusoidal_hdf4(str(path), "LC_Type2")

    def test_missing_file_raises_hdf4_error(self, tmp_path):
        """``read_modis_sinusoidal_hdf4`` raises ``HDF4Error`` for a path with no file.

        A missing staged tile must stop the read rather than yield an empty land-cover map.

        Reads a path that does not exist and asserts ``HDF4Error`` is raised.
        """
        with pytest.raises(HDF4Error):
            read_modis_sinusoidal_hdf4(str(tmp_path / "absent.hdf"), "LC_Type1")


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
    def test_random_boxes_match_every_pixel_center(self, tmp_path, seed):
        """The tile holder's ``query_box`` serves exactly the pixel centers inside a box, across tiles.

        The holder prunes tiles, rows and columns before geolocating; an over-tight prune drops land-cover pixels
        from a footprint and an over-loose mask serves pixels outside it.

        Parametrized over three random seeds: each run stages six mid-latitude tiles and, for 20 random boxes,
        asserts the served positions and codes equal a brute-force computation over every pixel center of every
        tile.
        """
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
        """``query_box`` serves every pixel of a high-latitude box wholly east or wholly west of the prime meridian.

        At high latitude the box's sinusoidal x-range depends on which latitude edge is used: east of the meridian it
        is lowest at the pole-most row, west of it highest there, and using the wrong edge would cut off pixels
        across a tile boundary.

        Parametrized over a box at 60-62 degrees straddling the h18/h19 edge and its mirror straddling h16/h17: each
        asserts the served pixels equal the brute-force set and that some are served.
        """
        reader, tiles = self._reader(tmp_path, [(15, 2), (16, 2), (17, 2), (18, 2), (19, 2), (20, 2)])
        _assert_matches(reader, tiles, bbox)
        assert reader._native.query_box(bbox)[0].size > 0

    def test_box_straddling_the_equator(self, tmp_path):
        """``query_box`` serves pixels on both sides of the equator for a box that crosses it.

        A box crossing the equator spans the widest point of each row and two vertical tile rows; losing either
        hemisphere would leave half a footprint without land cover.

        Stages the four tiles around the equator at the prime meridian, asserts the served pixels equal the
        brute-force set, and that latitudes both below and above zero are served.
        """
        reader, tiles = self._reader(tmp_path, [(18, 8), (19, 8), (18, 9), (19, 9)])
        bbox = BoundingBox(-1.0, 1.0, -0.5, 1.5)
        _assert_matches(reader, tiles, bbox)
        lats = reader._native.query_box(bbox)[0]
        assert lats.min() < 0.0 < lats.max()

    def test_dateline_box_serves_h35_and_h00(self, tmp_path):
        """``samples_in`` serves a box wrapping the antimeridian from the tiles on both sides of it.

        A footprint crossing 180 degrees is split into two halves; if either half were dropped, the footprint would
        get land cover from only one side.

        Stages tiles h00v08 and h35v08, queries a wrapping box from 179 to 181 degrees and asserts longitudes past
        179 and below -179 are both served, then asserts each half box matches the brute-force set.
        """
        reader, tiles = self._reader(tmp_path, [(0, 8), (35, 8)])
        samples = reader.samples_in(BoundingBox(4.0, 6.0, 179.0, 181.0, wraps_dateline=True), _T)
        assert samples.lons.max() > 179.0
        assert samples.lons.min() < -179.0
        for half in (BoundingBox(4.0, 6.0, 179.0, 180.0), BoundingBox(4.0, 6.0, -180.0, -179.0)):
            _assert_matches(reader, tiles, half)

    def test_polar_box(self, tmp_path):
        """``query_box`` serves the pixels of a polar box spanning all longitudes.

        Near the pole the sinusoidal rows shrink to a few pixels that span every longitude, so a polar footprint
        depends on the full-longitude box reaching them.

        Stages tiles h17v00 and h18v00, queries 85-90 degrees north over -180 to 180, and asserts the served pixels
        equal the brute-force set and are not empty.
        """
        reader, tiles = self._reader(tmp_path, [(17, 0), (18, 0)])
        bbox = BoundingBox(85.0, 90.0, -180.0, 180.0, is_polar=True)
        _assert_matches(reader, tiles, bbox)
        assert reader._native.query_box(bbox)[0].size > 0

    def test_no_off_globe_pixel_is_served(self, tmp_path):
        """``query_box`` serves no pixel lying off the sinusoidal globe, beyond 180 degrees longitude.

        Edge tiles contain pixels whose computed longitude exceeds 180 degrees; they are not on Earth, and serving
        them would add bogus land cover to footprints.

        Stages tiles h06v03 and h29v03, which straddle the globe's edge at 50-60 degrees, queries a full-longitude
        band, and asserts every served longitude is within 180 degrees and the result matches the brute-force
        set.
        """
        # Tiles h06/h29 at 50-60 deg straddle the edge of the sinusoidal globe: part of each
        # maps beyond +-180 deg longitude.
        reader, tiles = self._reader(tmp_path, [(6, 3), (29, 3)])
        lats, lons, _, _ = reader._native.query_box(BoundingBox(50.0, 60.0, -180.0, 180.0))
        assert lons.size > 0
        assert np.all(np.abs(lons) <= 180.0)
        _assert_matches(reader, tiles, BoundingBox(50.0, 60.0, -180.0, 180.0))

    def test_tile_outside_the_box_contributes_nothing(self, tmp_path):
        """A staged tile far from the query box serves no samples.

        A footprint must get land cover only from pixels inside its box; a tile elsewhere on the globe must not leak
        pixels into it.

        Stages h18v04 (Europe), queries a box in the southern Indian Ocean, and asserts ``query_box`` returns no
        pixels and ``samples_in`` returns values of shape ``(1, 0)``.
        """
        reader, tiles = self._reader(tmp_path, [(18, 4)])
        assert reader._native.query_box(BoundingBox(-10.0, -8.0, 100.0, 102.0))[0].size == 0
        assert reader.samples_in(BoundingBox(-10.0, -8.0, 100.0, 102.0), _T).values.shape == (1, 0)

    def test_samples_in_serves_the_padded_box(self, tmp_path):
        """``IGBPReader.samples_in`` widens the box by half a 0.463 km pixel and describes what it serves.

        A pixel whose center lies just outside a footprint's box still covers part of it, so the box is padded by
        half a pixel before the tile holder is queried; the footprint pipeline then relies on the sample set's
        source, dtype, shape and resolution to aggregate and credit coverage.

        Places the box's top edge 0.001 degree below one row of pixel centers, inside the 0.0021 degree padding, and
        its bottom edge 0.003 degree above another, beyond it; asserts the first row is served and the second is not,
        then that the result is a ``SampleSet`` with source ``"igbp"``, float32 values of shape ``(1, n_samples)``
        and ``resolution_km`` 0.463.
        """
        reader, tiles = self._reader(tmp_path, [(18, 4)])
        row_lats = np.unique(_brute_force(tiles, BoundingBox(44.0, 46.0, 5.0, 6.0))[:, 0])
        mid = len(row_lats) // 2
        inside_pad_lat, beyond_pad_lat = row_lats[mid], row_lats[mid - 3]
        samples = reader.samples_in(BoundingBox(beyond_pad_lat + 0.003, inside_pad_lat - 0.001, 5.0, 6.0), _T)
        assert np.isclose(samples.lats, inside_pad_lat).any()
        assert not np.isclose(samples.lats, beyond_pad_lat).any()
        assert isinstance(samples, SampleSet)
        assert samples.source == "igbp"
        assert samples.values.dtype == np.float32
        assert samples.values.shape == (1, samples.n_samples)
        assert samples.n_samples > 0
        assert samples.resolution_km == 0.463


class TestIGBPCodes:
    def test_codes_outside_the_igbp_domain_are_nan(self, tmp_path):
        """``IGBPReader`` serves codes outside 1..17 as NaN and passes 1..17 through unchanged.

        MCD12Q1 ``LC_Type1`` classes 1..17 map one to one onto ``IGBPSurfaceType``; 0, 18 and the 255 fill are not
        classes, and serving them as numbers would put invalid surface types into the modal class.

        Writes a tile with row bands of 0, 17, 18, 5, 255 and 1, serves a box over it, and asserts the finite
        values are exactly 1, 5 and 17 and that some values are NaN.
        """
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
        """A tile that is all fill (255) serves its pixels, every one as NaN.

        Fill pixels still occupy the footprint, but they carry no class, so they must not be counted as land cover.

        Writes a tile of 255 everywhere, serves a box over it, and asserts samples are served and all are NaN.
        """
        path, _ = _tile(tmp_path, 18, 4, codes=np.full((_N, _N), 255, dtype=np.uint8))
        samples = IGBPReader(path).samples_in(BoundingBox(44.0, 45.0, 5.0, 6.0), _T)
        assert samples.n_samples > 0
        assert np.all(np.isnan(samples.values))


class TestIGBPTileHolder:
    def test_holds_codes_as_uint8_and_one_dimensional_geometry(self, tmp_path):
        """``IGBPReader`` holds the staged tiles as uint8 codes plus one geometry value per tile.

        Holding the codes at one byte each and geolocating per query keeps memory small; a per-pixel geometry would
        multiply it.

        Stages three tiles and asserts the held data is uint8 of shape ``(3, n, n)``, each of ``x_ul``, ``y_ul``,
        ``dx`` and ``dy`` has shape ``(3,)``, and ``radius_m`` is a single float.
        """
        paths = [_tile(tmp_path, h, 4)[0] for h in (17, 18, 19)]
        native = IGBPReader(paths)._native
        assert native.data.dtype == np.uint8
        assert native.data.shape == (3, _N, _N)
        for name in ("x_ul", "y_ul", "dx", "dy"):
            assert getattr(native, name).shape == (3,)
        assert isinstance(native.radius_m, float)

    def test_load_memory_is_about_one_byte_per_pixel(self, tmp_path):
        """Loading the staged tiles peaks below 3 bytes per pixel.

        A float64 latitude and longitude for every pixel would take about 20 bytes per pixel, which is untenable for
        full-size 2400 by 2400 tiles.

        Stages two 480 by 480 tiles, traces memory while the holder loads, and asserts the peak is below three
        times the pixel count.
        """
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
        """``IGBPReader`` raises ``ValueError`` when the staged tiles use different sphere radii.

        The holder geolocates every tile with one shared radius, so a tile on another sphere would be served at
        shifted positions.

        Stages one tile with the MODIS radius and one with the WGS84 equatorial radius and asserts loading raises
        an error naming the radius mismatch.
        """
        first, _ = _tile(tmp_path, 17, 4)
        second, _ = _tile(tmp_path, 18, 4, proj_params="6378137.0,0,0,0,0,0,0,0,0,0,0,0,0")
        with pytest.raises(ValueError, match="sphere radius .* differs from the first staged tile"):
            _ = IGBPReader([first, second])._native

    def test_tiles_with_different_shapes_raise(self, tmp_path):
        """``IGBPReader`` raises ``ValueError`` when the staged tiles differ in pixel count.

        All tiles are held in one ``(n_tiles, n_rows, n_cols)`` array, which cannot take tiles of mixed size, and
        mixed sizes mean the tiles are not from one MCD12Q1 product.

        Stages one full-size and one half-size tile and asserts loading raises an error naming the shape
        mismatch.
        """
        first, _ = _tile(tmp_path, 17, 4)
        second, _ = _tile(tmp_path, 18, 4, codes=np.ones((_N // 2, _N // 2), dtype=np.uint8))
        with pytest.raises(ValueError, match="tile shape .* differs from the first staged tile"):
            _ = IGBPReader([first, second])._native

    def test_non_uint8_codes_raise(self, tmp_path):
        """``IGBPReader`` raises ``ValueError`` when ``LC_Type1`` is not stored as uint8.

        MCD12Q1 distributes IGBP classes as uint8; another dtype means the file is not the product as distributed,
        and its codes cannot be trusted as IGBP classes.

        Writes a tile with int16 codes and asserts loading raises an error naming the dtype.
        """
        path, _ = _tile(tmp_path, 18, 4, codes=np.ones((_N, _N), dtype=np.int16))
        with pytest.raises(ValueError, match="LC_Type1 is int16"):
            _ = IGBPReader(path)._native


class TestIGBPReaderRealSubset:
    """A real MCD12Q1 h21v07 tile (2024) cut to 120 x 120 pixels in the Ethiopian highlands, near 12N, 38E.

    The cut keeps ``LC_Type1`` and rewrites the tile's ``StructMetadata.0`` corners and
    dimensions to the window, so the sinusoidal geolocation runs on it unchanged. The
    window holds eleven IGBP classes, mostly croplands (12), with a few water pixels (17).
    """

    @pytest.fixture
    def reader(self, fmatch_real_data_path):
        return IGBPReader(fmatch_real_data_path / "MCD12Q1.A2024001.h21v07.061.2025205222320_subset.hdf")

    def test_window_codes_are_the_granules(self, reader):
        """``IGBPReader`` holds the real MCD12Q1 subset's ``LC_Type1`` codes exactly as stored.

        The real tile checks that reading a distributed MCD12Q1 file keeps every pixel's class; a lost or altered
        pixel would change the land cover served.

        Loads the subset and asserts the count of each class code equals the known histogram of the window.
        """
        codes, counts = np.unique(reader._native.data, return_counts=True)
        assert dict(zip(codes.tolist(), counts.tolist(), strict=True)) == {
            1: 2,
            2: 8,
            5: 1,
            9: 26,
            10: 1292,
            11: 27,
            12: 13022,
            13: 12,
            14: 3,
            16: 1,
            17: 6,
        }

    def test_a_box_in_the_window_is_geolocated_and_served(self, reader):
        """``samples_in`` geolocates a real MCD12Q1 tile's pixels into a box inside the window.

        The real tile's ``StructMetadata.0`` exercises the sinusoidal geolocation as distributed; a wrong position
        would assign land cover to the wrong footprints.

        Queries a 0.2 degree box near 12N, 38E and asserts samples are served, all lie within 0.05 degree of the box,
        all codes are classes present in the window, and the most common is croplands (12).
        """
        bbox = BoundingBox(11.9, 12.1, 38.0, 38.2)
        samples = reader.samples_in(bbox, _T)
        assert samples.n_samples > 0
        assert np.all((samples.lats >= bbox.lat_min - 0.05) & (samples.lats <= bbox.lat_max + 0.05))
        assert np.all((samples.lons >= bbox.lon_min - 0.05) & (samples.lons <= bbox.lon_max + 0.05))
        served, counts = np.unique(samples.values[0], return_counts=True)
        assert set(served.tolist()) <= {1, 2, 5, 9, 10, 11, 12, 13, 14, 16, 17}
        assert served[np.argmax(counts)] == 12
