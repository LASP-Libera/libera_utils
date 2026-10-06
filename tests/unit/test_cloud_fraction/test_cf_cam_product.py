"""CF-CAM (radiometer-timescale) product-write test.

Guards that ``cf_cam.yml`` can be written under ``strict=True`` conformance on the RADIOMETER_TIME
axis and re-opens with only its declared variables.
"""

import numpy as np
import xarray as xr

from libera_utils.cloud_fraction.cf_cam import PRODUCT_DEFINITION_PATH
from libera_utils.cloud_fraction.cloud_fraction import generate_placeholder_cloud_fraction_radiometer
from libera_utils.io.netcdf import write_libera_data_product
from libera_utils.io.product_definition import LiberaDataProductDefinition


def _radiometer_times(n: int = 8) -> np.ndarray:
    base = np.datetime64("2026-06-11T00:00:00", "ns")
    return base + np.arange(n, dtype="int64") * np.timedelta64(10_000_000, "ns")


def test_write_is_conformant_and_reopens(tmp_path):
    definition = LiberaDataProductDefinition.from_yaml(PRODUCT_DEFINITION_PATH)
    data = generate_placeholder_cloud_fraction_radiometer(_radiometer_times(8))

    output_file = write_libera_data_product(
        data_product_definition=definition,
        data=data,
        output_path=tmp_path,
        time_variable="RADIOMETER_TIME",
        dynamic_product_attributes={"algorithm_version": "9.9.9", "InputGranules": "l1b_rad.nc"},
        strict=True,
    )

    assert output_file.path.exists()
    reopened = xr.open_dataset(output_file.path, mask_and_scale=False)
    declared = set(definition.coordinates) | set(definition.variables)
    assert [name for name in reopened.variables if name not in declared] == []
    assert reopened["cloud_fraction"].dims == ("RADIOMETER_TIME",)
    assert reopened.attrs["InputGranules"] == "l1b_rad.nc"
