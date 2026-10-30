"""Regression tests pinning the native-sample path against per-pixel reference statistics.

A reader serves every native pixel inside a footprint's box, the weigher weights each
pixel where it lies, and the aggregation engine reduces the weighted pixels once. These
tests build a synthetic 1 km field through a real point reader and check, against the
same statistics computed directly on the pixels, that:

* the within-footprint standard deviation is the pixel standard deviation (an
  intermediate per-cell average would report only the spread between cell means), and
* a class that is a large minority everywhere survives as the secondary scene (an
  intermediate per-cell mode would drop it before ranking).
"""

from __future__ import annotations

import numpy as np
import pytest

from libera_utils.footprint_matching.product import aggregate_external_variables
from libera_utils.footprint_matching.readers.base import StaticReader
from libera_utils.footprint_matching.types import (
    BoundingBox,
    OperationalMode,
    PointSamples,
    TimeMatch,
    VariableSpec,
)
from libera_utils.footprint_matching.weighting import RadialWeigher
from tests.plugins.footprint_matching_fixtures import FakeFootprint

_LAT0, _LON0 = 40.0, -100.0
_RNG_SEED = 20261020


def _pixel_field() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A 1 km pixel field over a 1 x 1 degree box: a textured continuous field and a 60/40 class mix."""
    rng = np.random.default_rng(_RNG_SEED)
    lats = np.arange(_LAT0, _LAT0 + 1.0, 0.009)
    lons = np.arange(_LON0, _LON0 + 1.0, 0.0117)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    texture = rng.normal(280.0, 5.0, lon_grid.shape)
    classes = np.where(rng.random(lon_grid.shape) < 0.4, 17.0, 1.0)  # 40 % water in forest
    return lat_grid.ravel(), lon_grid.ravel(), texture.ravel(), classes.ravel()


class _PixelReader(StaticReader):
    """Serves the synthetic pixel field: one continuous and one categorical variable."""

    READER_KEY = "_native_stats"
    INSTRUMENT = "TEST"
    RESOLUTION_KM = 1.0
    TIME_MATCH = TimeMatch.DAILY
    VARIABLES = (
        VariableSpec(name="texture", dtype="float32", aggregation="weighted_mean", modes=frozenset(OperationalMode)),
        VariableSpec(
            name="scene", dtype="int16", aggregation="weighted_mode", modes=frozenset(OperationalMode), n_categories=20
        ),
    )
    ADDITIONAL_PRODUCT_VARIABLES = (
        VariableSpec(
            name="scene_primary",
            dtype="int16",
            aggregation="weighted_mode_primary",
            modes=frozenset(OperationalMode),
            n_categories=20,
        ),
        VariableSpec(
            name="scene_secondary",
            dtype="int16",
            aggregation="weighted_mode_secondary",
            modes=frozenset(OperationalMode),
            n_categories=20,
        ),
    )

    def _load_native(self) -> PointSamples:
        lats, lons, texture, classes = _pixel_field()
        return PointSamples(lats=lats, lons=lons, values=np.stack([texture, classes]).astype(np.float32))


@pytest.fixture
def pixel_run(tmp_path):
    """Aggregate one footprint in the middle of the field and return (result, weigher, footprint)."""
    footprint = FakeFootprint(BoundingBox(_LAT0 + 0.2, _LAT0 + 0.8, _LON0 + 0.2, _LON0 + 0.8), _LAT0 + 0.5, _LON0 + 0.5)
    granule = tmp_path / "pixels.bin"
    granule.touch()
    sources = {_PixelReader.READER_KEY: _PixelReader([granule])}
    weigher = RadialWeigher()
    result, coverage = aggregate_external_variables(
        OperationalMode.CAM, [footprint], sources, weigher=weigher, return_coverage=True
    )
    return result, coverage, weigher, footprint, sources


def _reference(weigher, footprint, sources):
    """The same statistics computed directly on the pixels with the same weights."""
    samples = sources[_PixelReader.READER_KEY].samples_in(footprint.bbox, footprint.time)
    weights = weigher.weight_field(
        samples,
        footprint.latitude,
        footprint.longitude,
        subsatellite_lat_deg=footprint.latitude,
        subsatellite_lon_deg=footprint.longitude,
        satellite_radius_km=7213.0,
        sample_resolution_km=1.0,
    ).weights
    texture, classes = samples.values
    use = weights > 0
    w, x = weights[use], texture[use]
    mean = np.sum(w * x) / np.sum(w)
    std = np.sqrt(np.sum(w * (x - mean) ** 2) / np.sum(w))
    class_weight = {int(c): float(np.sum(w[classes[use] == c])) for c in np.unique(classes[use])}
    ranked = sorted(class_weight, key=class_weight.get, reverse=True)
    return mean, std, ranked


def test_standard_deviation_is_the_pixel_spread(pixel_run):
    result, _, weigher, footprint, sources = pixel_run
    mean, std, _ = _reference(weigher, footprint, sources)
    assert result["_native_stats_texture"][0] == pytest.approx(mean, rel=1e-5)
    # The pixel field has a 5 K spread; the product must report it, not a fraction of it.
    assert result["_native_stats_texture_standard_deviation"][0] == pytest.approx(std, rel=1e-4)
    assert 4.0 < result["_native_stats_texture_standard_deviation"][0] < 6.0


def test_minority_class_survives_as_secondary_scene(pixel_run):
    result, _, weigher, footprint, sources = pixel_run
    _, _, ranked = _reference(weigher, footprint, sources)
    assert ranked == [1, 17]
    assert result["_native_stats_scene"][0] == 1
    assert result["_native_stats_scene_primary"][0] == 1
    # 40 % of every square kilometer is water: a per-cell mode would never show it.
    assert result["_native_stats_scene_secondary"][0] == 17


def test_complete_pixel_field_is_fully_covered(pixel_run):
    _, coverage, _, _, _ = pixel_run
    assert coverage[0] == pytest.approx(1.0, abs=0.02)
