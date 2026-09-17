"""CF-CAM-CAMTIME product-write and FMATCH-CAM-CAMTIME alignment tests.

The write test is the guard that ``cf_cam_camtime.yml`` can be written under ``strict=True``
conformance. The parity test is the operational meaning of "records align 1:1": a CF-CAM-CAMTIME
product and the FMATCH-CAM-CAMTIME product built from the *same* segmentation must carry the
identical ``CAMERA_TIME`` / ``PSEUDOFOOTPRINT`` axes and ``camera_pixel_{x,y}_{min,max}`` provenance.
"""

import numpy as np
import xarray as xr

from libera_utils.cloud_fraction.cf_cam_camtime import PRODUCT_DEFINITION_PATH
from libera_utils.cloud_fraction.cloud_fraction import generate_placeholder_cloud_fraction_camtime
from libera_utils.footprint_matching._runner import load_l1b_camera_dataset
from libera_utils.footprint_matching.camera_segmentation import segment_l1b_camera
from libera_utils.footprint_matching.product import assemble_fmatch_dataset
from libera_utils.footprint_matching.types import OperationalMode
from libera_utils.io.netcdf import write_libera_data_product
from libera_utils.io.product_definition import LiberaDataProductDefinition
from tests.test_data.footprint_matching.fixtures import make_l1b_camera_fixture

_PROVENANCE_COORDINATES = (
    "CAMERA_TIME",
    "PSEUDOFOOTPRINT",
    "camera_pixel_x_min",
    "camera_pixel_x_max",
    "camera_pixel_y_min",
    "camera_pixel_y_max",
)


def _footprints(tmp_path):
    l1b_file = make_l1b_camera_fixture(tmp_path, n_images=2, n_pixels_x=4, n_pixels_y=4)
    return segment_l1b_camera(load_l1b_camera_dataset(l1b_file))


def test_write_is_conformant_and_reopens(tmp_path):
    """A full build + write succeeds under strict conformance and re-opens with only declared variables."""
    definition = LiberaDataProductDefinition.from_yaml(PRODUCT_DEFINITION_PATH)
    footprints = _footprints(tmp_path)
    data = generate_placeholder_cloud_fraction_camtime(footprints, definition)

    output_file = write_libera_data_product(
        data_product_definition=definition,
        data=data,
        output_path=tmp_path,
        time_variable="CAMERA_TIME",
        dynamic_product_attributes={"algorithm_version": "9.9.9", "InputGranules": "l1b_cam.nc"},
        strict=True,
    )

    assert output_file.path.exists()
    reopened = xr.open_dataset(output_file.path, mask_and_scale=False)
    declared = set(definition.coordinates) | set(definition.variables)
    assert [name for name in reopened.variables if name not in declared] == []
    assert reopened.attrs["InputGranules"] == "l1b_cam.nc"


def test_provenance_aligns_with_fmatch_cam_camtime(tmp_path):
    """CF-CAM-CAMTIME and FMATCH-CAM-CAMTIME built from the same footprints share record axis + provenance."""
    definition = LiberaDataProductDefinition.from_yaml(PRODUCT_DEFINITION_PATH)
    footprints = _footprints(tmp_path)

    cf_data = generate_placeholder_cloud_fraction_camtime(footprints, definition)
    fmatch_dataset = assemble_fmatch_dataset(OperationalMode.CAM_CAMTIME, footprints)

    for name in _PROVENANCE_COORDINATES:
        np.testing.assert_array_equal(
            np.asarray(cf_data[name]),
            fmatch_dataset[name].to_numpy(),
            err_msg=f"{name} differs between CF-CAM-CAMTIME and FMATCH-CAM-CAMTIME",
        )
