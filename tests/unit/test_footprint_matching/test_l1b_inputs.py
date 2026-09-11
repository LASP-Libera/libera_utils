"""Unit tests for reading the L1B Daily inputs that FMATCH is built on.

These cover the pass-through contract read by
``libera_utils.footprint_matching._runner.load_l1b_radiometer_inputs``: which variables are read,
the dtypes they are cast to, and the dropping of footprints whose boresight had no valid Earth
intersection.
"""

import numpy as np
import pytest
import xarray as xr

from libera_utils.footprint_matching._runner import (
    FMATCH_RADIOMETER_TIME_COORDINATE,
    load_l1b_camera_dataset,
    load_l1b_radiometer_inputs,
)
from libera_utils.footprint_matching.product import (
    FMATCH_CONE_ANGLE_RATE_KEY,
    L1B_PASSTHROUGH_VARIABLES,
    L1B_SCAN_REFERENCE_VARIABLES,
)
from tests.test_data.footprint_matching.fixtures import (
    make_l1b_camera_fixture,
    make_l1b_radiometer_fixture,
)


class TestRadiometerPassthrough:
    """The radiometer reader returns exactly the product's L1B-derived columns."""

    def test_returns_time_coordinate_and_every_passthrough_variable(self, tmp_path):
        l1b_file = make_l1b_radiometer_fixture(tmp_path, n_footprints=8)

        inputs = load_l1b_radiometer_inputs(l1b_file)

        assert set(inputs) == {
            FMATCH_RADIOMETER_TIME_COORDINATE,
            *L1B_PASSTHROUGH_VARIABLES,
            *L1B_SCAN_REFERENCE_VARIABLES,
            FMATCH_CONE_ANGLE_RATE_KEY,
        }

    def test_dtypes_match_the_product_definition(self, tmp_path):
        """Time decodes to datetime64[ns] and every pass-through variable is float32, as declared."""
        l1b_file = make_l1b_radiometer_fixture(tmp_path, n_footprints=8)

        inputs = load_l1b_radiometer_inputs(l1b_file)

        assert inputs[FMATCH_RADIOMETER_TIME_COORDINATE].dtype == np.dtype("datetime64[ns]")
        for name in L1B_PASSTHROUGH_VARIABLES:
            assert inputs[name].dtype == np.float32, name

    def test_scan_reference_geometry_is_read_at_float64(self, tmp_path):
        """The subsatellite point and cone-angle rate feed the ray-trace, so they stay float64."""
        l1b_file = make_l1b_radiometer_fixture(tmp_path, n_footprints=8)

        inputs = load_l1b_radiometer_inputs(l1b_file)

        for name in (*L1B_SCAN_REFERENCE_VARIABLES, FMATCH_CONE_ANGLE_RATE_KEY):
            assert inputs[name].dtype == np.float64, name
            assert len(inputs[name]) == len(inputs[FMATCH_RADIOMETER_TIME_COORDINATE]), name

    def test_a_fill_cone_angle_rate_does_not_drop_the_footprint(self, tmp_path):
        """Cone-angle rate is optional geometry: a NaN there must not discard an Earth-viewing footprint."""
        l1b_file = make_l1b_radiometer_fixture(tmp_path, n_footprints=6)
        with xr.open_dataset(l1b_file) as dataset:
            modified = dataset.load()
        modified["Cone_Angle_Rate"].values[2] = np.nan
        modified.to_netcdf(tmp_path / "modified.nc")

        inputs = load_l1b_radiometer_inputs(tmp_path / "modified.nc")

        # All six footprints survive; the fill cone-angle rate is carried through as NaN.
        assert len(inputs[FMATCH_RADIOMETER_TIME_COORDINATE]) == 6
        assert np.isnan(inputs[FMATCH_CONE_ANGLE_RATE_KEY][2])

    def test_all_arrays_share_one_length(self, tmp_path):
        l1b_file = make_l1b_radiometer_fixture(tmp_path, n_footprints=8)

        inputs = load_l1b_radiometer_inputs(l1b_file)

        assert {len(values) for values in inputs.values()} == {8}

    def test_time_is_monotonically_increasing(self, tmp_path):
        l1b_file = make_l1b_radiometer_fixture(tmp_path, n_footprints=8)

        inputs = load_l1b_radiometer_inputs(l1b_file)

        times = inputs[FMATCH_RADIOMETER_TIME_COORDINATE]
        assert np.all(np.diff(times) > np.timedelta64(0, "ns"))


class TestNonFiniteFootprints:
    """Footprints whose boresight misses the Earth carry NaN and must be dropped."""

    def test_non_finite_footprints_are_dropped(self, tmp_path):
        l1b_file = make_l1b_radiometer_fixture(tmp_path, n_footprints=10, n_invalid=3)

        inputs = load_l1b_radiometer_inputs(l1b_file)

        assert len(inputs[FMATCH_RADIOMETER_TIME_COORDINATE]) == 7
        for name in L1B_PASSTHROUGH_VARIABLES:
            assert np.all(np.isfinite(inputs[name])), name

    def test_a_nan_in_any_single_variable_drops_the_footprint(self, tmp_path):
        """The finite mask is an AND across variables, so one bad column is enough to drop a row."""
        l1b_file = make_l1b_radiometer_fixture(tmp_path, n_footprints=6)
        with xr.open_dataset(l1b_file) as dataset:
            modified = dataset.load()
        # Corrupt only the viewing zenith angle; geolocation stays finite.
        values = modified[L1B_PASSTHROUGH_VARIABLES["viewing_zenith_angle"]].values
        values[2] = np.nan
        modified.to_netcdf(tmp_path / "modified.nc")

        inputs = load_l1b_radiometer_inputs(tmp_path / "modified.nc")

        assert len(inputs[FMATCH_RADIOMETER_TIME_COORDINATE]) == 5

    def test_a_nan_subsatellite_point_drops_the_footprint(self, tmp_path):
        """The subsatellite point is required for a real box, so a NaN there drops the row too."""
        l1b_file = make_l1b_radiometer_fixture(tmp_path, n_footprints=6)
        with xr.open_dataset(l1b_file) as dataset:
            modified = dataset.load()
        # Corrupt only the subsatellite latitude; the footprint geolocation stays finite.
        modified["Subsatellite_Latitude"].values[4] = np.nan
        modified.to_netcdf(tmp_path / "modified.nc")

        inputs = load_l1b_radiometer_inputs(tmp_path / "modified.nc")

        assert len(inputs[FMATCH_RADIOMETER_TIME_COORDINATE]) == 5

    def test_all_non_finite_raises(self, tmp_path):
        """A file with nothing usable cannot produce a product, so it fails loudly."""
        l1b_file = make_l1b_radiometer_fixture(tmp_path, n_footprints=4, n_invalid=4)

        with pytest.raises(ValueError, match="No usable footprints"):
            load_l1b_radiometer_inputs(l1b_file)


class TestCameraDataset:
    """The camera reader hands segmentation a fully-loaded dataset."""

    def test_contains_the_segmentation_grids(self, tmp_path):
        from libera_utils.footprint_matching import camera_segmentation as seg

        l1b_file = make_l1b_camera_fixture(tmp_path, n_images=2, n_pixels_x=4, n_pixels_y=4)

        dataset = load_l1b_camera_dataset(l1b_file)

        for name in (
            seg.LATITUDE_NAME,
            seg.LONGITUDE_NAME,
            seg.ALTITUDE_NAME,
            seg.SOLAR_ZENITH_NAME,
            seg.VIEWING_ZENITH_NAME,
            seg.RELATIVE_AZIMUTH_NAME,
        ):
            assert name in dataset, name
        assert dataset.sizes[seg.CAMERA_TIME_NAME] == 2

    def test_usable_after_the_source_file_is_deleted(self, tmp_path):
        """The dataset is loaded eagerly, so an S3 input materialized into a temp dir stays readable."""
        from libera_utils.footprint_matching import camera_segmentation as seg

        l1b_file = make_l1b_camera_fixture(tmp_path, n_images=1, n_pixels_x=3, n_pixels_y=3)
        dataset = load_l1b_camera_dataset(l1b_file)
        l1b_file.unlink()

        assert np.isfinite(dataset[seg.LATITUDE_NAME].values).all()
