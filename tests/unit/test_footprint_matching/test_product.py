"""Unit tests for the FMATCH-CAM data product definition and its loader.

These tests confirm that:
- ``FMATCH-CAM`` is registered as an ancillary (ANC) DataProductIdentifier.
- The product definition YAML loads and validates via LiberaDataProductDefinition.
- The schema declares the expected geolocation, derived-geometry, coverage/QA,
  and PSF-aggregated external variables.
- The external-variable names/dtypes in the YAML stay in sync with the reader
  plugins' VariableSpec definitions (guards against future drift).
- A small dummy dataset round-trips through create/enforce/check conformance.
"""

from __future__ import annotations

import numpy as np
import pytest

from libera_utils.constants import DataLevel, DataProductIdentifier
from libera_utils.footprint_matching.product import (
    FMATCH_CAM_TIME_VARIABLE,
    load_fmatch_cam_definition,
)

# Import the concrete production readers directly rather than reading the global
# ReaderRegistry: other tests register throwaway readers (e.g. _FakeReader in
# test_base.py) into that shared registry, so iterating it would make this test
# order-dependent. Pinning to the real reader classes keeps the cross-check stable.
from libera_utils.footprint_matching.readers.brdf import VIIRSBRDFReader
from libera_utils.footprint_matching.readers.era5 import ERA5Reader
from libera_utils.footprint_matching.readers.igbp import IGBPReader
from libera_utils.footprint_matching.readers.nsidc import NISEReader
from libera_utils.footprint_matching.readers.viirs import VIIRSCloudReader
from libera_utils.footprint_matching.types import OperationalMode
from libera_utils.io.product_definition import LiberaDataProductDefinition

# The production readers active in FMATCH-CAM (OperationalMode.CAM).
_FMATCH_CAM_READERS = (
    ERA5Reader,
    IGBPReader,
    NISEReader,
    VIIRSBRDFReader,
    VIIRSCloudReader,
)

# Geolocation inputs and derived viewing-geometry variables the definition must
# expose (these have no reader; they come from L1B / are computed).
GEOLOCATION_VARIABLES = (
    "latitude",
    "longitude",
    "altitude",
    "solar_zenith_angle",
    "viewing_zenith_angle",
    "relative_azimuth_angle",
)
DERIVED_GEOMETRY_VARIABLES = ("scattering_angle", "sunglint_angle")
COVERAGE_QA_VARIABLES = ("psf_coverage_fraction", "q_flags")


def _expected_external_variables() -> dict[str, str]:
    """Return {variable_name: dtype} for every CAM-mode production reader variable."""
    expected: dict[str, str] = {}
    for cls in _FMATCH_CAM_READERS:
        # Defensive: every reader pinned here should indeed be active in CAM mode.
        assert cls.REQUIRED_MODE.rank <= OperationalMode.CAM.rank
        for spec in cls.VARIABLES:
            expected[spec.name] = spec.dtype
    return expected


@pytest.fixture(scope="module")
def fmatch_cam_definition() -> LiberaDataProductDefinition:
    """The loaded FMATCH-CAM product definition."""
    return load_fmatch_cam_definition()


class TestFmatchCamIdentifier:
    """The product ID must exist in the enum as an ANC product."""

    def test_fmatch_cam_in_enum(self):
        product = DataProductIdentifier("FMATCH-CAM")
        assert product is DataProductIdentifier.anc_fmatch_cam
        assert product.data_level is DataLevel.ANC

    def test_product_id_matches_operational_mode(self):
        # The product ID string must stay identical to the operational mode value.
        assert str(DataProductIdentifier.anc_fmatch_cam) == OperationalMode.CAM.value


class TestFmatchCamDefinition:
    """The YAML loads and declares the expected structure."""

    def test_definition_loads(self, fmatch_cam_definition):
        assert isinstance(fmatch_cam_definition, LiberaDataProductDefinition)
        assert fmatch_cam_definition.attributes["ProductID"] == "FMATCH-CAM"

    def test_time_coordinate_present(self, fmatch_cam_definition):
        assert FMATCH_CAM_TIME_VARIABLE in fmatch_cam_definition.coordinates
        time_def = fmatch_cam_definition.coordinates[FMATCH_CAM_TIME_VARIABLE]
        assert time_def.dtype == "datetime64[ns]"
        assert time_def.dimensions == [FMATCH_CAM_TIME_VARIABLE]

    @pytest.mark.parametrize(
        "variable_name",
        GEOLOCATION_VARIABLES + DERIVED_GEOMETRY_VARIABLES + COVERAGE_QA_VARIABLES,
    )
    def test_non_reader_variables_present(self, fmatch_cam_definition, variable_name):
        assert variable_name in fmatch_cam_definition.variables

    def test_external_variables_match_readers(self, fmatch_cam_definition):
        """Every active reader variable must be present with a matching dtype."""
        expected = _expected_external_variables()
        for name, dtype in expected.items():
            assert name in fmatch_cam_definition.variables, f"missing external variable {name}"
            assert fmatch_cam_definition.variables[name].dtype == dtype, (
                f"dtype drift for {name}: definition has "
                f"{fmatch_cam_definition.variables[name].dtype}, reader has {dtype}"
            )

    def test_all_variables_use_radiometer_time_dimension(self, fmatch_cam_definition):
        for name, var_def in fmatch_cam_definition.variables.items():
            assert var_def.dimensions == [FMATCH_CAM_TIME_VARIABLE], (
                f"{name} should be indexed by {FMATCH_CAM_TIME_VARIABLE}"
            )


class TestFmatchCamConformance:
    """A dummy dataset must round-trip through create/enforce/check."""

    def test_roundtrip(self, fmatch_cam_definition):
        n_footprints = 4
        # One datetime64 value per footprint identifies each record.
        times = np.array(
            ["2026-06-11T00:00:00", "2026-06-11T00:00:01", "2026-06-11T00:00:02", "2026-06-11T00:00:03"],
            dtype="datetime64[ns]",
        )

        data: dict[str, np.ndarray] = {FMATCH_CAM_TIME_VARIABLE: times}
        # Fill every declared variable with dummy data of the correct dtype.
        for name, var_def in fmatch_cam_definition.variables.items():
            data[name] = np.zeros(n_footprints, dtype=var_def.dtype)

        dynamic_attrs = {
            "algorithm_version": "1.0.0",
            "date_created": "2026-06-11T00:00:00Z",
            "input_files": "dummy_l1b.nc",
        }
        dataset = fmatch_cam_definition.create_product_dataset(data, dynamic_product_attributes=dynamic_attrs)
        dataset = fmatch_cam_definition.enforce_dataset_conformance(dataset)

        # strict=True raises on any nonconformance; an empty list means success.
        errors = fmatch_cam_definition.check_dataset_conformance(dataset, strict=True)
        assert errors == []
