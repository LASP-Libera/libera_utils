"""Integration tests for the SCENE-ID CAM-family runner and product write path.

These exercise the manifest/dropbox plumbing in
``libera_utils.scene_identification._runner`` and the concrete CAM runner, including the actual product write (which
is not covered by the algorithm-level tests in ``test_scene_id.py``). The happy-path test in particular is the guard
that the SCENE-ID product definitions can be written under ``strict=True`` conformance.
"""

from datetime import UTC, datetime

import pytest
import xarray as xr

from libera_utils.constants import DataProductIdentifier
from libera_utils.io.filenaming import LiberaDataProductFilename
from libera_utils.io.manifest import Manifest, ManifestFileRecord, ManifestType
from libera_utils.io.product_definition import LiberaDataProductDefinition
from libera_utils.scene_identification import FootprintData
from libera_utils.scene_identification.cam.scene_id_cam import (
    PRODUCT_DEFINITION_PATH,
    collect_ssf_input_files,
    create_and_write_data_product_cam,
    run_scene_identification_cam,
)

SSF_INPUT_NAME = "CER_SSF_NOAA20-FM6-VIIRS_Edition1C_101103.2023010100.nc"


def _libera_product_name(product_id: DataProductIdentifier) -> str:
    """Build a valid Libera data-product filename string for the given product id."""
    return LiberaDataProductFilename.from_filename_parts(
        product_name=product_id,
        version="V1-0-0",
        utc_start=datetime(2023, 1, 1, tzinfo=UTC),
        utc_end=datetime(2023, 1, 1, 23, 59, 59, tzinfo=UTC),
    ).path.name


class TestSceneIdCamWrite:
    """The CAM runner must produce a conformant SCENE-ID-CAM product with only declared variables."""

    def test_write_data_product_is_conformant(self, test_scene_id, tmp_path):
        """A full run + write succeeds under strict conformance and re-opens."""
        input_path = test_scene_id / SSF_INPUT_NAME
        footprint_data = run_scene_identification_cam(input_path)

        # create_and_write_data_product_cam writes with strict=True; if the product definition and dataset are not
        # conformant this raises. Reaching the assertions below is itself the strict-conformance guarantee.
        output_file = create_and_write_data_product_cam(footprint_data, input_path.name, tmp_path)

        assert output_file.path.exists()
        reopened = xr.open_dataset(output_file.path)
        # Provenance attributes set by the runner survive the round trip.
        assert reopened.attrs["input_files"] == input_path.name
        assert reopened.attrs["algorithm_version"] == "0.1.0"

    def test_written_product_has_no_undeclared_variables(self, test_scene_id, tmp_path):
        """Intermediate FootprintData inputs must not leak into the written product."""
        input_path = test_scene_id / SSF_INPUT_NAME
        footprint_data = run_scene_identification_cam(input_path)
        output_file = create_and_write_data_product_cam(footprint_data, input_path.name, tmp_path)

        definition = LiberaDataProductDefinition.from_yaml(PRODUCT_DEFINITION_PATH)
        declared = set(definition.coordinates) | set(definition.variables)

        # Read without CF mask/scale so integer variables are not upcast and encoding is preserved as written.
        reopened = xr.open_dataset(output_file.path, mask_and_scale=False)
        undeclared = [name for name in reopened.variables if name not in declared]
        assert undeclared == []
        # And the intermediate scene-property inputs specifically must be gone.
        for leaked in ("surface_wind_u", "surface_wind_v", "optical_depth_lower", "cloud_phase_lower"):
            assert leaked not in reopened.variables


class TestCollectInputFiles:
    """collect_input_files selects the right manifest entries in placeholder vs product mode."""

    # Manifest records must be absolute paths; the runner keys off the filename (basename) when parsing.
    _INPUT_DIR = "/dropbox/inputs"

    def _manifest(self, *filenames: str) -> Manifest:
        return Manifest(
            manifest_type=ManifestType.INPUT,
            files=[ManifestFileRecord(filename=f"{self._INPUT_DIR}/{name}", checksum="0") for name in filenames],
        )

    def test_placeholder_mode_keeps_non_libera_files(self):
        """The CAM runner runs in placeholder mode: keep the CERES SSF (non-Libera) file, skip Libera products."""
        libera_name = _libera_product_name(DataProductIdentifier.aux_fmatch_cam_camtime)
        manifest = self._manifest(SSF_INPUT_NAME, libera_name)

        selected = collect_ssf_input_files(manifest)

        assert selected == [f"{self._INPUT_DIR}/{SSF_INPUT_NAME}"]

    def test_product_mode_keeps_only_matching_product(self):
        """In Libera-product mode only files with the configured product id are kept."""
        from libera_utils.scene_identification._runner import collect_input_files

        wanted = _libera_product_name(DataProductIdentifier.aux_fmatch_cam_camtime)
        other = _libera_product_name(DataProductIdentifier.l1b_rad)
        manifest = self._manifest(wanted, other, SSF_INPUT_NAME)

        selected = collect_input_files(manifest, DataProductIdentifier.aux_fmatch_cam_camtime)

        assert selected == [f"{self._INPUT_DIR}/{wanted}"]


class TestToTimeProduct:
    """FootprintData.to_time_product prepares the dataset for writing on its time axis."""

    def test_promotes_time_and_adds_quality_flag(self, test_scene_id):
        footprint_data = run_scene_identification_cam(test_scene_id / SSF_INPUT_NAME)
        product = footprint_data.to_time_product("radiometer_time")

        assert "radiometer_time" in product.coords
        assert "Quality_Flag" in product.data_vars

    def test_missing_time_variable_raises(self):
        footprint_data = FootprintData(xr.Dataset({"cloud_fraction": ("RADIOMETER_TIME", [1.0, 2.0])}))
        with pytest.raises(ValueError, match="radiometer_time"):
            footprint_data.to_time_product("radiometer_time")


class TestFmatchReaders:
    """The operational FMATCH readers are not implemented yet."""

    def test_from_fmatch_cam_not_implemented(self, tmp_path):
        with pytest.raises(NotImplementedError):
            FootprintData.from_fmatch_cam(tmp_path / "fmatch.nc")

    def test_from_fmatch_cam_camtime_not_implemented(self, tmp_path):
        with pytest.raises(NotImplementedError):
            FootprintData.from_fmatch_cam_camtime(tmp_path / "fmatch.nc")
