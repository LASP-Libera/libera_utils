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
from libera_utils.footprint_matching._runner_common import load_l1b_camera_dataset
from libera_utils.footprint_matching.camera_segmentation import segment_l1b_camera
from libera_utils.footprint_matching.product import assemble_fmatch_dataset, pseudofootprints_from_camtime_dataset
from libera_utils.footprint_matching.types import (
    BoundingBox,
    CameraFootprintQualityFlag,
    OperationalMode,
    PseudoFootprint,
)
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


# ---------------------------------------------------------------------------
# Round-trip: CF-CAM-CAMTIME carries the COMPLETE pseudo-footprint record, so FMATCH can reconstruct
# each footprint exactly instead of re-segmenting. These guard the two fidelity traps from the field
# audit (doc/fmatch_cf_cam_camtime_field_audit.md): (A) the raw/un-normalized bbox longitudes + the
# wraps_dateline/is_polar/truncated flags, and (B) the base (segmentation) q_flags.
# ---------------------------------------------------------------------------

_T0 = np.datetime64("2024-03-01T00:00:00", "ns")
_T1 = np.datetime64("2024-03-01T00:00:05", "ns")


def _synthetic_footprints() -> list[PseudoFootprint]:
    """Footprints spanning two images (ragged: 3 then 1, to exercise grid padding) and every edge case:

    a dateline-wrapping box (raw lon > 180), a polar box, and a limb-truncated box with nonzero
    segmentation q_flags. Values are chosen exactly representable in float32 so the round-trip is exact.
    """
    return [
        # image 0, footprint 0 -- ordinary
        PseudoFootprint(
            time=_T0, slice_x=slice(0, 4), slice_y=slice(0, 4), center_ix=2, center_iy=2,
            latitude=45.5, longitude=-120.25, altitude=1024.0,
            solar_zenith_angle=30.5, viewing_zenith_angle=12.25, relative_azimuth_angle=88.0,
            bbox=BoundingBox(44.5, 46.5, -121.5, -119.0, False, False, False),
            q_flags=CameraFootprintQualityFlag(0),
        ),
        # image 0, footprint 1 -- dateline-wrapping box (raw lon_max > 180) + partial-coverage flag
        PseudoFootprint(
            time=_T0, slice_x=slice(4, 8), slice_y=slice(0, 4), center_ix=6, center_iy=2,
            latitude=-10.0, longitude=179.5, altitude=0.0,
            solar_zenith_angle=50.0, viewing_zenith_angle=20.0, relative_azimuth_angle=170.0,
            bbox=BoundingBox(-11.0, -9.0, 170.0, 190.0, True, False, False),
            q_flags=CameraFootprintQualityFlag.PARTIAL_COVERAGE,
        ),
        # image 0, footprint 2 -- polar box + limb-truncated + both q_flags bits set
        PseudoFootprint(
            time=_T0, slice_x=slice(0, 4), slice_y=slice(4, 8), center_ix=2, center_iy=6,
            latitude=87.5, longitude=10.0, altitude=250.0,
            solar_zenith_angle=80.0, viewing_zenith_angle=60.0, relative_azimuth_angle=15.0,
            bbox=BoundingBox(86.0, 89.0, -20.0, 40.0, False, True, True),
            q_flags=CameraFootprintQualityFlag.PARTIAL_COVERAGE | CameraFootprintQualityFlag.CENTER_PIXEL_SUBSTITUTED,
        ),
        # image 1, footprint 0 -- ordinary (narrower image -> the grid pads image 1's trailing columns)
        PseudoFootprint(
            time=_T1, slice_x=slice(8, 12), slice_y=slice(8, 12), center_ix=10, center_iy=10,
            latitude=0.5, longitude=0.25, altitude=500.0,
            solar_zenith_angle=10.0, viewing_zenith_angle=5.0, relative_azimuth_angle=45.0,
            bbox=BoundingBox(-0.5, 1.5, -1.0, 1.5, False, False, False),
            q_flags=CameraFootprintQualityFlag(0),
        ),
    ]


def _write_and_reopen_cf_product(footprints, definition, tmp_path):
    data = generate_placeholder_cloud_fraction_camtime(footprints, definition)
    output_file = write_libera_data_product(
        data_product_definition=definition,
        data=data,
        output_path=tmp_path,
        time_variable="CAMERA_TIME",
        dynamic_product_attributes={"algorithm_version": "9.9.9", "InputGranules": "l1b_cam.nc"},
        strict=True,
    )
    return xr.open_dataset(output_file.path)


def test_pseudofootprints_round_trip_through_cf_cam_camtime(tmp_path):
    """footprints -> CF-CAM-CAMTIME (write) -> reconstruct reproduces every field, incl. the edge cases."""
    definition = LiberaDataProductDefinition.from_yaml(PRODUCT_DEFINITION_PATH)
    originals = _synthetic_footprints()

    reopened = _write_and_reopen_cf_product(originals, definition, tmp_path)
    reconstructed = pseudofootprints_from_camtime_dataset(reopened)

    assert len(reconstructed) == len(originals)
    for original, got in zip(originals, reconstructed, strict=True):
        assert got.time == original.time
        assert got.slice_x == original.slice_x
        assert got.slice_y == original.slice_y
        assert got.center_ix == original.center_ix
        assert got.center_iy == original.center_iy
        # Floats chosen float32-exact, so equality holds after the float32 storage round-trip.
        assert got.latitude == np.float32(original.latitude)
        assert got.longitude == np.float32(original.longitude)
        assert got.altitude == np.float32(original.altitude)
        assert got.solar_zenith_angle == np.float32(original.solar_zenith_angle)
        assert got.viewing_zenith_angle == np.float32(original.viewing_zenith_angle)
        assert got.relative_azimuth_angle == np.float32(original.relative_azimuth_angle)
        # Trap A: raw (un-normalized) bbox longitudes + the three flags must reconstruct exactly.
        assert got.bbox.lat_min == np.float32(original.bbox.lat_min)
        assert got.bbox.lat_max == np.float32(original.bbox.lat_max)
        assert got.bbox.lon_min == np.float32(original.bbox.lon_min)
        assert got.bbox.lon_max == np.float32(original.bbox.lon_max)
        assert got.bbox.wraps_dateline == original.bbox.wraps_dateline
        assert got.bbox.is_polar == original.bbox.is_polar
        assert got.bbox.truncated == original.bbox.truncated
        # Trap B: base segmentation q_flags (NOT coverage-augmented) preserved exactly.
        assert got.q_flags == original.q_flags


def test_dateline_box_longitude_is_stored_raw_not_normalized(tmp_path):
    """The dateline footprint's raw lon_max (190) must survive; a normalized store would corrupt it."""
    definition = LiberaDataProductDefinition.from_yaml(PRODUCT_DEFINITION_PATH)
    originals = _synthetic_footprints()
    reopened = _write_and_reopen_cf_product(originals, definition, tmp_path)
    reconstructed = pseudofootprints_from_camtime_dataset(reopened)

    dateline = next(f for f in reconstructed if f.bbox.wraps_dateline)
    assert dateline.bbox.lon_max == np.float32(190.0)  # raw, > 180 -- normalization would have lost this
    assert dateline.bbox.lon_min == np.float32(170.0)


def test_fmatch_cam_camtime_output_identical_via_cf_product(tmp_path):
    """FMATCH-CAM-CAMTIME output is unchanged whether footprints come straight from segmentation or are
    reconstructed from CF-CAM-CAMTIME -- the source swap must not alter a single value."""
    cf_definition = LiberaDataProductDefinition.from_yaml(PRODUCT_DEFINITION_PATH)
    footprints = _footprints(tmp_path)

    # Old path: FMATCH assembled straight from the segmented footprints.
    from_segmentation = assemble_fmatch_dataset(OperationalMode.CAM_CAMTIME, footprints)

    # New path: same footprints -> CF-CAM-CAMTIME -> reconstruct -> FMATCH assembled from those.
    reopened = _write_and_reopen_cf_product(footprints, cf_definition, tmp_path)
    reconstructed = pseudofootprints_from_camtime_dataset(reopened)
    from_cf_product = assemble_fmatch_dataset(OperationalMode.CAM_CAMTIME, reconstructed)

    assert set(from_cf_product.variables) == set(from_segmentation.variables)
    for name in from_segmentation.variables:
        got = from_cf_product[name].to_numpy()
        expected = from_segmentation[name].to_numpy()
        message = f"FMATCH-CAM-CAMTIME variable {name} changed after the segmentation->CF-product source swap"
        if np.issubdtype(expected.dtype, np.floating):
            # The stored segmentation columns are float32-exact; a derived column (sunglint_angle) is
            # computed from angles that are now float32-quantized by the CF round-trip rather than the
            # segmenter's float64 -- a sub-float32-ULP difference that is physically meaningless (the L1B
            # angles are float32). Guard to float32 precision.
            np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6, equal_nan=True, err_msg=message)
        else:
            np.testing.assert_array_equal(got, expected, err_msg=message)
