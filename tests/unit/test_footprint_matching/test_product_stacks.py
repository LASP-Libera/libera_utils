"""Tests that FMATCH assembly writes a reader's ProductStacks as stacked product variables.

FMATCH-IMAGER declares each ERA5 pressure-level field as one ``(RADIOMETER_TIME, PRESSURE_LEVEL)``
variable, while the reader aggregates every (field, level) pair on its own; assembly collects the
levels back into the declared variables and writes the ``PRESSURE_LEVEL`` coordinate.
"""

from __future__ import annotations

import numpy as np

from libera_utils.footprint_matching.product import _fill_value_for, assemble_fmatch_dataset, load_fmatch_definition
from libera_utils.footprint_matching.readers.era5_pressure import (
    _ERA5_PRESSURE_LEVEL_VARIABLES,
    _ERA5_PRESSURE_LEVELS,
    ERA5PressureLevelReader,
)
from libera_utils.footprint_matching.types import OperationalMode
from tests.plugins.footprint_matching_fixtures import (
    ERA5_FIXTURE_TIME,
    era5_pressure_fixture_value,
    make_era5_pressure_netcdf_fixture,
)


def _l1b_inputs(n: int = 3) -> dict[str, np.ndarray]:
    """L1B pass-through inputs for ``n`` nadir footprints at (0.5, 0.5), one second apart."""
    return {
        "RADIOMETER_TIME": np.datetime64("2028-01-01T00:00:00", "ns") + np.arange(n) * np.timedelta64(1, "s"),
        "latitude": np.full(n, 0.5),
        "longitude": np.full(n, 0.5),
        "altitude": np.zeros(n),
        "solar_zenith_angle": np.full(n, 30.0),
        "viewing_zenith_angle": np.zeros(n),
        "relative_azimuth_angle": np.full(n, 90.0),
        "subsatellite_latitude": np.full(n, 0.5),
        "subsatellite_longitude": np.full(n, 0.5),
        "satellite_radius_km": np.full(n, 7213.0),
        "cone_angle_rate": np.full(n, -1.0),
    }


class TestStackedProductVariables:
    def test_era5_pressure_levels_fill_the_stacked_imager_variables(self, tmp_path):
        """FMATCH-IMAGER carries each ERA5 pressure-level field as one variable on ``PRESSURE_LEVEL``.

        The reader aggregates every (field, level) pair on its own, and the product definition declares one
        ``(RADIOMETER_TIME, PRESSURE_LEVEL)`` variable per field. If assembly published the per-level columns under
        their own names, the definition would drop them and write the stacked variables as fill, with no error.

        Assembles three nadir footprints inside a dense ERA5 pressure-level fixture whose every layer is a known
        constant, so the PSF-weighted mean is that constant. Asserts every stacked mean holds its layers' constants in
        level order, every stacked standard deviation is zero, and the ``PRESSURE_LEVEL`` coordinate holds the levels.
        """
        era5_path = make_era5_pressure_netcdf_fixture(tmp_path, n_lat=41, n_lon=41)
        l1b = _l1b_inputs()
        l1b["RADIOMETER_TIME"] = ERA5_FIXTURE_TIME + np.arange(3) * np.timedelta64(1, "s")
        for name, value in [("latitude", 1.0), ("subsatellite_latitude", 1.0)]:
            l1b[name] = np.full(3, value)
        for name in ("longitude", "subsatellite_longitude"):
            l1b[name] = np.full(3, 11.0)

        dataset = assemble_fmatch_dataset(
            OperationalMode.IMAGER, l1b, sources={"era5_pressure": ERA5PressureLevelReader([era5_path])}
        )

        np.testing.assert_array_equal(dataset["PRESSURE_LEVEL"].values, _ERA5_PRESSURE_LEVELS)
        for var_index, (field_name, _) in enumerate(_ERA5_PRESSURE_LEVEL_VARIABLES):
            mean = dataset[f"era5_pressure_{field_name}"]
            assert mean.dims == ("RADIOMETER_TIME", "PRESSURE_LEVEL")
            expected = [era5_pressure_fixture_value(var_index, level) for level in _ERA5_PRESSURE_LEVELS]
            np.testing.assert_allclose(mean.values, np.tile(expected, (3, 1)), rtol=1e-6)
            spread = dataset[f"era5_pressure_{field_name}_standard_deviation"]
            np.testing.assert_allclose(spread.values, 0.0, atol=1e-6 * np.max(np.abs(expected)))

    def test_unstaged_era5_pressure_leaves_stacked_placeholders(self):
        """Without ERA5 pressure-level inputs, FMATCH-IMAGER still carries the levels and level-shaped fill.

        The ``PRESSURE_LEVEL`` coordinate comes from the reader's stacks, not its inputs, and a stacked variable no
        input produced must still have its declared ``(RADIOMETER_TIME, PRESSURE_LEVEL)`` shape to conform.

        Assembles three footprints with ``allow_placeholders=True`` and no ancillary inputs, and asserts the coordinate
        holds the configured levels and a stacked variable is all fill at shape ``(3, n_levels)``.
        """
        dataset = assemble_fmatch_dataset(OperationalMode.IMAGER, _l1b_inputs(), allow_placeholders=True)

        np.testing.assert_array_equal(dataset["PRESSURE_LEVEL"].values, _ERA5_PRESSURE_LEVELS)
        temperature = dataset["era5_pressure_temperature"]
        assert temperature.shape == (3, len(_ERA5_PRESSURE_LEVELS))
        _, fill_value = _fill_value_for(
            load_fmatch_definition(OperationalMode.IMAGER).variables["era5_pressure_temperature"]
        )
        np.testing.assert_array_equal(temperature.values, fill_value)
