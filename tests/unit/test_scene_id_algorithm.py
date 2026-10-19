"""Unit tests for SCENE-ID runner helper logic in ``libera_utils.scene_identification.scene_id_algorithm``.

These are pure-logic tests that need no real data or product write — e.g. the manifest input-selection rules in
:func:`collect_input_files` and the ``LIBERA_SCENE_ID_VARIANT`` dispatch in :func:`resolve_runner_config_from_env`.
The full manifest-in / product-out behavior is covered by the integration tests in ``tests/integration/test_scene_id.py``.
"""

from datetime import UTC, datetime

import pytest

from libera_utils.constants import DataProductIdentifier
from libera_utils.io.filenaming import LiberaDataProductFilename
from libera_utils.io.manifest import Manifest, ManifestType
from libera_utils.scene_identification.scene_id_algorithm import (
    RUNNER_CONFIGS,
    SCENE_ID_VARIANT_ENV,
    collect_input_files,
    resolve_runner_config_from_env,
)

# A CERES SSF filename: not a Libera product name, so it stands in for the placeholder-mode input.
SSF_INPUT_NAME = "CER_SSF_NOAA20-FM6-VIIRS_Edition1C_101103.2023010100.nc"


def _libera_product_name(product_id: DataProductIdentifier) -> str:
    """Build a valid Libera data-product filename string for the given product id."""
    return LiberaDataProductFilename.from_filename_parts(
        product_name=product_id,
        version="V1-0-0",
        utc_start=datetime(2023, 1, 1, tzinfo=UTC),
        utc_end=datetime(2023, 1, 1, 23, 59, 59, tzinfo=UTC),
    ).path.name


class TestCollectInputFiles:
    """collect_input_files selects the right manifest entries by product id."""

    # Manifest records must be absolute paths; the runner keys off the filename (basename) when parsing.
    _INPUT_DIR = "/dropbox/inputs"

    def _manifest(self, *filenames: str) -> Manifest:
        # collect_input_files selects purely by filename, so the listed files need not exist
        return Manifest(manifest_type=ManifestType.INPUT, files=[f"{self._INPUT_DIR}/{name}" for name in filenames])

    def test_product_mode_keeps_only_matching_product(self):
        """In Libera-product mode only files with the configured product id are kept."""
        wanted = _libera_product_name(DataProductIdentifier.aux_fmatch_cam_camtime)
        other = _libera_product_name(DataProductIdentifier.l1b_rad)
        manifest = self._manifest(wanted, other, SSF_INPUT_NAME)

        selected = collect_input_files(manifest, DataProductIdentifier.aux_fmatch_cam_camtime)

        assert selected == [f"{self._INPUT_DIR}/{wanted}"]


class TestRunnerDispatch:
    """LIBERA_SCENE_ID_VARIANT selects the SCENE-ID runner config that the CLI handler runs."""

    @pytest.mark.parametrize("variant", sorted(RUNNER_CONFIGS))
    def test_variant_selects_its_config(self, monkeypatch, variant):
        """Each registry key in the environment variable selects that registry entry."""
        monkeypatch.setenv(SCENE_ID_VARIANT_ENV, variant)

        assert resolve_runner_config_from_env() is RUNNER_CONFIGS[variant]

    @pytest.mark.parametrize("value", [None, "", "  "])
    def test_unset_variant_raises(self, monkeypatch, value):
        """A missing or blank variable is an error rather than a silent default variant."""
        if value is None:
            monkeypatch.delenv(SCENE_ID_VARIANT_ENV, raising=False)
        else:
            monkeypatch.setenv(SCENE_ID_VARIANT_ENV, value)

        with pytest.raises(ValueError, match="is not set"):
            resolve_runner_config_from_env()

    def test_unknown_variant_raises(self, monkeypatch):
        """A value naming no registry entry is rejected, listing the valid variants."""
        monkeypatch.setenv(SCENE_ID_VARIANT_ENV, "not-a-variant")

        with pytest.raises(ValueError, match="'not-a-variant' is not a SCENE-ID variant"):
            resolve_runner_config_from_env()
