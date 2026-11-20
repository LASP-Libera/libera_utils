"""Unit tests for SCENE-ID runner helper logic in ``libera_utils.scene_identification.scene_id_algorithm``.

These are pure-logic tests that need no real data or product write — e.g. the manifest input-selection rules in
:func:`collect_input_files` and the ``processing_step_id`` dispatch in :func:`resolve_runner_config`.
The full manifest-in / product-out behavior is covered by the integration tests in ``tests/integration/test_scene_id.py``.
"""

from datetime import UTC, datetime

import numpy as np
import pytest
import xarray as xr

from libera_utils.constants import DataProductIdentifier, ProcessingStepIdentifier
from libera_utils.io.filenaming import LiberaDataProductFilename
from libera_utils.io.manifest import (
    PROCESSING_STEP_ID_CONFIGURATION_KEY,
    Manifest,
    ManifestType,
    ProcessingStepIdMismatchError,
)
from libera_utils.io.product_definition import LiberaDataProductDefinition
from libera_utils.scene_identification.scene_id_algorithm import (
    RUNNER_CONFIGS,
    _fill_missing_integer_values,
    collect_input_files,
    resolve_runner_config,
)

CAM = ProcessingStepIdentifier.aux_scene_id_cam
IMAGER = ProcessingStepIdentifier.aux_scene_id_imager

# A CERES SSF filename: not a Libera product name, so it is skipped.
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

    # Built once: a Libera filename carries its creation time as the revision, so a second call names another file.
    _WANTED = _libera_product_name(DataProductIdentifier.aux_fmatch_cam_camtime)
    _OTHER = _libera_product_name(DataProductIdentifier.l1b_rad)

    @pytest.mark.parametrize(
        ("filenames", "expected"),
        [
            pytest.param([_WANTED, _OTHER, SSF_INPUT_NAME], [_WANTED], id="mixed"),
            pytest.param([_OTHER], [], id="another-libera-product-only"),
            pytest.param([SSF_INPUT_NAME, "not_a_libera_file.txt"], [], id="unparsable-names-only"),
        ],
    )
    def test_keeps_only_matching_product(self, filenames, expected):
        """Only files with the configured product id are kept, and unparsable names are skipped."""
        manifest = self._manifest(*filenames)

        selected = collect_input_files(manifest, DataProductIdentifier.aux_fmatch_cam_camtime)

        assert selected == [f"{self._INPUT_DIR}/{name}" for name in expected]


def _manifest_naming(step: str | None) -> Manifest:
    """An input manifest whose configuration names ``step``, or names no step when ``step`` is ``None``."""
    configuration = {} if step is None else {PROCESSING_STEP_ID_CONFIGURATION_KEY: str(step)}
    return Manifest(manifest_type=ManifestType.INPUT, configuration=configuration)


class TestResolveRunnerConfig:
    """The input manifest's configuration, or the command line, names the processing step whose config runs."""

    @pytest.mark.parametrize("step", list(RUNNER_CONFIGS))
    def test_manifest_step_selects_its_config(self, step):
        assert resolve_runner_config(_manifest_naming(step)) is RUNNER_CONFIGS[step]

    @pytest.mark.parametrize("step", [IMAGER, str(IMAGER)])
    def test_command_line_step_selects_its_config(self, step):
        assert resolve_runner_config(_manifest_naming(None), step) is RUNNER_CONFIGS[IMAGER]

    def test_matching_steps_select_the_config(self):
        assert resolve_runner_config(_manifest_naming(CAM), str(CAM)) is RUNNER_CONFIGS[CAM]

    def test_different_steps_raise(self):
        with pytest.raises(ProcessingStepIdMismatchError, match="aux-scene-id-cam.*aux-scene-id-imager"):
            resolve_runner_config(_manifest_naming(CAM), IMAGER)

    def test_no_step_raises(self):
        """A manifest naming no step is an error rather than a silent default variant."""
        with pytest.raises(ValueError, match="No SCENE-ID processing step was given"):
            resolve_runner_config(_manifest_naming(None))

    @pytest.mark.parametrize("value", [str(ProcessingStepIdentifier.aux_fmatch_cam), "cam"])
    def test_step_that_is_not_scene_id_raises(self, value):
        """A step naming no registry entry, including an old variant name, is rejected."""
        with pytest.raises(ValueError, match="is not a SCENE-ID processing step"):
            resolve_runner_config(_manifest_naming(value))


class TestFillMissingIntegerValues:
    """_fill_missing_integer_values writes -1 into a NaN integer cell and refuses a dtype that cannot hold it."""

    @staticmethod
    def _definition(surface_type_dtype: str) -> LiberaDataProductDefinition:
        """The shipped SCENE-ID-IMAGER definition, with Surface_Type declared as ``surface_type_dtype``."""
        definition = LiberaDataProductDefinition.from_yaml(RUNNER_CONFIGS[IMAGER].product_definition_path)
        surface_type = definition.variables["Surface_Type"].model_copy(update={"dtype": surface_type_dtype})
        return definition.model_copy(update={"variables": {"Surface_Type": surface_type}})

    def test_nan_becomes_minus_one_in_the_declared_dtype(self):
        dataset = xr.Dataset({"Surface_Type": ("RADIOMETER_TIME", np.array([2.0, np.nan], dtype=np.float32))})

        filled = _fill_missing_integer_values(dataset, self._definition("int8"))

        assert filled["Surface_Type"].dtype == np.int8
        np.testing.assert_array_equal(filled["Surface_Type"].values, [2, -1])

    def test_unsigned_dtype_with_a_missing_value_raises(self):
        dataset = xr.Dataset({"Surface_Type": ("RADIOMETER_TIME", np.array([2.0, np.nan], dtype=np.float32))})

        with pytest.raises(ValueError, match="Surface_Type has missing values.*signed integer dtype"):
            _fill_missing_integer_values(dataset, self._definition("uint8"))
