"""Integration tests for the CF-CAM (Camera Cloud Fraction) manifest-driven runners.

Exercise the full manifest/dropbox plumbing in ``libera_utils.cloud_fraction._runner`` and both
concrete runners end-to-end: a real L1B input fixture in, a conformant CF product plus an output
manifest out. The happy-path write is the guard that the CF product definitions can be written under
``strict=True`` conformance from actual runner output.
"""

import pytest
import xarray as xr

from libera_utils.cloud_fraction.cf_cam import RUNNER_CONFIG as CF_CAM_CONFIG
from libera_utils.cloud_fraction.cf_cam import algorithm as cf_cam_algorithm
from libera_utils.cloud_fraction.cf_cam_camtime import RUNNER_CONFIG as CF_CAM_CAMTIME_CONFIG
from libera_utils.cloud_fraction.cf_cam_camtime import algorithm as cf_cam_camtime_algorithm
from libera_utils.io.manifest import Manifest, ManifestType
from tests.test_data.footprint_matching.fixtures import (
    make_l1b_camera_fixture,
    make_l1b_radiometer_fixture,
)

pytestmark = pytest.mark.integration


def _write_input_manifest(input_file, directory) -> str:
    """Write an INPUT manifest referencing ``input_file`` and return its path."""
    manifest = Manifest(
        manifest_type=ManifestType.INPUT,
        files=[{"filename": str(input_file), "checksum": "fakesum"}],
    )
    return str(manifest.write(directory))


def _run(algorithm, input_file, tmp_path, monkeypatch):
    """Set up the dropbox + manifest and run one CF algorithm, returning the written CF product path."""
    dropbox = tmp_path / "dropbox"
    dropbox.mkdir()
    monkeypatch.setenv("PROCESSING_PATH", str(dropbox))

    manifest_path = _write_input_manifest(input_file, tmp_path)
    output_manifest_path = algorithm(manifest_path)

    assert output_manifest_path.exists()
    output_manifest = Manifest.from_file(output_manifest_path)
    product_files = [record.filename for record in output_manifest.files]
    assert len(product_files) == 1
    return product_files[0]


def test_cf_cam_camtime_runner_writes_conformant_product(tmp_path, monkeypatch):
    """The CF-CAM-CAMTIME runner segments an L1B CAM file and writes a conformant product."""
    l1b_file = make_l1b_camera_fixture(tmp_path, n_images=2, n_pixels_x=4, n_pixels_y=4)
    product_path = _run(cf_cam_camtime_algorithm, l1b_file, tmp_path, monkeypatch)

    product = xr.open_dataset(product_path)
    assert product.attrs["ProductID"] == CF_CAM_CAMTIME_CONFIG.output_product_id.value == "CF-CAM-CAMTIME"
    assert product["cloud_fraction"].dims == ("CAMERA_TIME", "PSEUDOFOOTPRINT")
    assert "cloud_fraction_standard_deviation" in product.variables
    assert product.attrs["InputGranules"] == l1b_file.name


def test_cf_cam_runner_writes_conformant_product(tmp_path, monkeypatch):
    """The CF-CAM runner reads an L1B RAD-4CH file and writes a conformant radiometer-timescale product."""
    l1b_file = make_l1b_radiometer_fixture(tmp_path, n_footprints=8)
    product_path = _run(cf_cam_algorithm, l1b_file, tmp_path, monkeypatch)

    product = xr.open_dataset(product_path)
    assert product.attrs["ProductID"] == CF_CAM_CONFIG.output_product_id.value == "CF-CAM"
    assert product["cloud_fraction"].dims == ("RADIOMETER_TIME",)
    assert "cloud_fraction_standard_deviation" in product.variables
    assert product.attrs["InputGranules"] == l1b_file.name
