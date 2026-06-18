"""Unit tests for libera_utils.footprint_matching.product.

Tests confirm:
- PRODUCT_FOR_MODE covers all five operational modes
- Every mode maps to a distinct DataProductIdentifier
- Product ID strings match the DPI v87 Table of Data Product Identifiers
- get_product_identifier() returns the correct identifier for each mode
"""

from __future__ import annotations

import pytest

from libera_utils.constants import DataProductIdentifier
from libera_utils.footprint_matching.product import PRODUCT_FOR_MODE, get_product_identifier
from libera_utils.footprint_matching.types import OperationalMode


class TestProductForMode:
    def test_all_five_modes_are_covered(self):
        assert set(PRODUCT_FOR_MODE.keys()) == set(OperationalMode)

    def test_all_identifiers_are_distinct(self):
        # Each mode must map to a unique product, not share an identifier.
        values = list(PRODUCT_FOR_MODE.values())
        assert len(values) == len(set(values))

    def test_all_values_are_data_product_identifiers(self):
        for mode, product_id in PRODUCT_FOR_MODE.items():
            assert isinstance(product_id, DataProductIdentifier), (
                f"Mode {mode} maps to {product_id!r}, expected DataProductIdentifier"
            )

    # --- Check each product ID string against the DPI v87 ---
    # Product ID strings must match the DPI v87 Table of Data Product Identifiers
    # exactly. These are the canonical identifiers used in output filenames,
    # NetCDF global attributes, and AWS pipeline routing.

    @pytest.mark.parametrize(
        ("mode", "expected_id_str"),
        [
            (OperationalMode.CAM, "FMATCHCAM"),
            (OperationalMode.CAM_CAMTIME, "FMATCHCAMCAMTIME"),
            (OperationalMode.IMAGER_FLASH, "FMATCHIMAGERFLASH"),
            (OperationalMode.IMAGER, "FMATCHIMAGER"),
            (OperationalMode.IMAGER_CAMTIME, "FMATCHIMAGERCAMTIME"),
        ],
    )
    def test_product_id_string_matches_dpi(self, mode: OperationalMode, expected_id_str: str):
        product_id = PRODUCT_FOR_MODE[mode]
        assert str(product_id) == expected_id_str


class TestGetProductIdentifier:
    def test_returns_correct_identifier_for_cam(self):
        result = get_product_identifier(OperationalMode.CAM)
        assert result == DataProductIdentifier.fmatch_cam
        assert str(result) == "FMATCHCAM"

    def test_returns_correct_identifier_for_imager(self):
        result = get_product_identifier(OperationalMode.IMAGER)
        assert result == DataProductIdentifier.fmatch_imager
        assert str(result) == "FMATCHIMAGER"

    def test_returns_correct_identifier_for_imager_flash(self):
        result = get_product_identifier(OperationalMode.IMAGER_FLASH)
        assert result == DataProductIdentifier.fmatch_imager_flash
        assert str(result) == "FMATCHIMAGERFLASH"

    def test_returns_correct_identifier_for_cam_camtime(self):
        result = get_product_identifier(OperationalMode.CAM_CAMTIME)
        assert result == DataProductIdentifier.fmatch_cam_camtime
        assert str(result) == "FMATCHCAMCAMTIME"

    def test_returns_correct_identifier_for_imager_camtime(self):
        result = get_product_identifier(OperationalMode.IMAGER_CAMTIME)
        assert result == DataProductIdentifier.fmatch_imager_camtime
        assert str(result) == "FMATCHIMAGERCAMTIME"

    def test_raises_key_error_for_unknown_mode(self):
        # PRODUCT_FOR_MODE is a plain dict; passing a non-OperationalMode key
        # should raise KeyError (no silent fallback).
        with pytest.raises((KeyError, TypeError)):
            get_product_identifier("NOT_A_MODE")  # type: ignore[arg-type]
