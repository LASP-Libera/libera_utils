"""Unit tests for ``footprint_match_algorithm``: the LIBERA_FMATCH_VARIANT dispatch and the camera-timescale inputs."""

import numpy as np
import pytest
import xarray as xr

from libera_utils.constants import DataProductIdentifier
from libera_utils.footprint_matching.footprint_match_algorithm import (
    FMATCH_VARIANT_ENV,
    RUNNER_CONFIGS,
    _load_cam_camtime_footprints,
    _load_imager_camtime_footprints,
    create_and_write_data_product,
    resolve_runner_config_from_env,
)
from libera_utils.footprint_matching.product import MissingCamtimeVariablesError
from tests.plugins.footprint_matching_fixtures import make_cf_cam_camtime_dataset, make_cf_cam_camtime_fixture


class TestResolveRunnerConfigFromEnv:
    """LIBERA_FMATCH_VARIANT selects the FMATCH runner config that the CLI handler runs."""

    @pytest.mark.parametrize("variant", ["cam", "cam-camtime", "imager", "imager-camtime", "imager-flash"])
    def test_variant_selects_its_config(self, monkeypatch, variant):
        monkeypatch.setenv(FMATCH_VARIANT_ENV, variant)
        assert resolve_runner_config_from_env() is RUNNER_CONFIGS[variant]

    @pytest.mark.parametrize("value", [None, "", "  "])
    def test_unset_variant_raises(self, monkeypatch, value):
        if value is None:
            monkeypatch.delenv(FMATCH_VARIANT_ENV, raising=False)
        else:
            monkeypatch.setenv(FMATCH_VARIANT_ENV, value)
        with pytest.raises(ValueError, match="not set"):
            resolve_runner_config_from_env()

    def test_unknown_variant_raises(self, monkeypatch):
        monkeypatch.setenv(FMATCH_VARIANT_ENV, "scene-id")
        with pytest.raises(ValueError, match="not a FMATCH variant"):
            resolve_runner_config_from_env()


class TestCameraTimescaleInputs:
    """Both camera-timescale runners read their pseudo-footprints from CF-CAM-CAMTIME."""

    @pytest.mark.parametrize("variant", ["cam-camtime", "imager-camtime"])
    def test_input_product_is_cf_cam_camtime(self, variant):
        assert RUNNER_CONFIGS[variant].input_product_id is DataProductIdentifier.l2_cf_cam_camtime

    def test_cam_camtime_loader_aligns_cloud_fraction_with_footprints(self, tmp_path):
        cf_file = make_cf_cam_camtime_fixture(tmp_path, n_images=2, n_pixels_x=2, n_pixels_y=2, n_padded=3)

        footprints, cloud_fraction = _load_cam_camtime_footprints(cf_file)

        assert len(footprints) == 5
        expected = make_cf_cam_camtime_dataset(n_images=2, n_pixels_x=2, n_pixels_y=2)["cloud_fraction"].values
        np.testing.assert_allclose(cloud_fraction, expected.ravel()[:5])

    def test_imager_camtime_loader_reads_no_cloud_fraction(self, tmp_path):
        cf_file = make_cf_cam_camtime_fixture(tmp_path, n_images=1)

        footprints, cloud_fraction = _load_imager_camtime_footprints(cf_file)

        assert len(footprints) == 9
        assert cloud_fraction is None

    def test_imager_camtime_writes_a_product_from_cf_cam_camtime(self, tmp_path):
        cf_file = make_cf_cam_camtime_fixture(tmp_path, n_images=2)
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        written = create_and_write_data_product(cf_file, output_dir, RUNNER_CONFIGS["imager-camtime"])

        with xr.open_dataset(written.path) as product:
            assert product.sizes["CAMERA_TIME"] == 2
            assert product.sizes["PSEUDOFOOTPRINT"] == 9
            assert "cloud_fraction_camera" not in product
            assert cf_file.name in product.attrs["InputGranules"]

    def test_input_missing_pseudofootprint_variables_fails_the_run(self, tmp_path):
        dataset = make_cf_cam_camtime_dataset().drop_vars("psf_bbox_flags")
        cf_file = tmp_path / "cf.nc"
        dataset.to_netcdf(cf_file)

        with pytest.raises(MissingCamtimeVariablesError, match="psf_bbox_flags"):
            create_and_write_data_product(cf_file, tmp_path, RUNNER_CONFIGS["cam-camtime"])
