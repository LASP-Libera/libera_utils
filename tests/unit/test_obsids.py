"""Unit tests for libera_utils.obsids registry."""

import pytest

from libera_utils.constants import DataProductIdentifier, LiberaApid
from libera_utils.obsids import (
    OBSID_REGISTRY,
    NomHkObsidSource,
    ObsIdKind,
    get_obsid_spec,
    iter_trim_eligible,
)


class TestObsidRegistry:
    """Tests for OBSID_REGISTRY completeness and consistency."""

    def test_keys_match_source_and_obsid(self):
        """Registry keys must equal (spec.source, spec.obsid)."""
        for key, spec in OBSID_REGISTRY.items():
            assert key == (spec.source, spec.obsid)

    def test_rad_and_wfov_256_differ(self):
        """RAD and WFOV ObsID 256 map to different products."""
        rad = get_obsid_spec(NomHkObsidSource.RAD, 256)
        wfov = get_obsid_spec(NomHkObsidSource.WFOV, 256)
        assert rad.cal_product is DataProductIdentifier.cal_swc_365nm
        assert wfov.cal_product is DataProductIdentifier.cal_darks_of_darks
        assert rad.trimmed_product is not wfov.trimmed_product

    def test_trim_eligible_have_both_products(self):
        """Every trim-eligible entry has trimmed and cal ProductIDs."""
        for spec in iter_trim_eligible():
            assert spec.trimmed_product is not None
            assert spec.cal_product is not None
            assert spec.kind in (ObsIdKind.RAD_CAL, ObsIdKind.CAM_CAL)

    def test_science_entries_have_no_products(self):
        """Science/scan modes are catalog-only."""
        science = [s for s in OBSID_REGISTRY.values() if s.kind is ObsIdKind.SCIENCE]
        assert len(science) == 14  # 7 ObsIDs × 2 sources
        for spec in science:
            assert spec.trimmed_product is None
            assert spec.cal_product is None

    def test_science_modes_on_both_sources(self):
        """Science ObsIDs 128, 132, and 136–140 are registered under RAD and WFOV."""
        for obsid in (128, 132, *range(136, 141)):
            get_obsid_spec(NomHkObsidSource.RAD, obsid)
            get_obsid_spec(NomHkObsidSource.WFOV, obsid)

    def test_cross_track_and_rap_science_modes(self):
        """ObsIDs 128 and 132 are Cross Track and RAP scan science modes."""
        cross = get_obsid_spec(NomHkObsidSource.RAD, 128)
        rap = get_obsid_spec(NomHkObsidSource.WFOV, 132)
        assert cross.name == "Cross Track"
        assert cross.description == "Cross Track Scan Mode"
        assert rap.name == "RAP Scan"
        assert rap.description == "RAP Scan Mode"

    def test_trimmed_products_associate_with_nom_hk_apid(self):
        """All TRIMMED DPIs associate with the NOM-HK APID."""
        for spec in iter_trim_eligible():
            assert spec.trimmed_product is not None
            assert spec.trimmed_product.associated_apid is LiberaApid.icie_nom_hk

    def test_camera_dpi_string_values(self):
        """Camera TRIMMED/CAL ProductID strings match the catalog table."""
        expected = {
            129: ("NOM-HK-CT-VIDEO-6MIN-TRIMMED", "CT-VIDEO-6MIN"),
            130: ("NOM-HK-CT-VIDEO-12MIN-TRIMMED", "CT-VIDEO-12MIN"),
            131: ("NOM-HK-CT-VIDEO-18MIN-TRIMMED", "CT-VIDEO-18MIN"),
            133: ("NOM-HK-RAPS-VIDEO-6MIN-TRIMMED", "RAPS-VIDEO-6MIN"),
            134: ("NOM-HK-RAPS-VIDEO-12MIN-TRIMMED", "RAPS-VIDEO-12MIN"),
            135: ("NOM-HK-RAPS-VIDEO-18MIN-TRIMMED", "RAPS-VIDEO-18MIN"),
            256: ("NOM-HK-DARKS-OF-DARKS-TRIMMED", "DARKS-OF-DARKS"),
            257: ("NOM-HK-LED-OF-DARK-TRIMMED", "LED-OF-DARK"),
            258: ("NOM-HK-NOMINAL-DARKS-TRIMMED", "NOMINAL-DARKS"),
            513: ("NOM-HK-VIIRS-LUNAR-CAL-TRIMMED", "VIIRS-LUNAR-CAL"),
        }
        for obsid, (trimmed_val, cal_val) in expected.items():
            spec = get_obsid_spec(NomHkObsidSource.WFOV, obsid)
            assert spec.trimmed_product is not None
            assert spec.cal_product is not None
            assert spec.trimmed_product.value == trimmed_val
            assert spec.cal_product.value == cal_val

    def test_rad_cal_count(self):
        """Twenty-five radiometer cal ObsIDs (22 gain/SWC/LWC/solar + 2 lunar + VIIRS lunar)."""
        rad_cal = [s for s in OBSID_REGISTRY.values() if s.kind is ObsIdKind.RAD_CAL]
        assert len(rad_cal) == 25

    def test_lunar_cal_obsids(self):
        """Radiometer lunar ObsIDs 448/449 map to LUNAR-CAL1/2 products."""
        lunar1 = get_obsid_spec(NomHkObsidSource.RAD, 448)
        lunar2 = get_obsid_spec(NomHkObsidSource.RAD, 449)
        assert lunar1.trimmed_product is DataProductIdentifier.l1a_icie_nom_hk_lunar_cal1_trimmed
        assert lunar1.cal_product is DataProductIdentifier.cal_lunar_cal1
        assert lunar2.trimmed_product is DataProductIdentifier.l1a_icie_nom_hk_lunar_cal2_trimmed
        assert lunar2.cal_product is DataProductIdentifier.cal_lunar_cal2
        assert "Monthly" in lunar1.description
        assert "Quarterly" in lunar2.description

    def test_rad_viirs_lunar_cal_obsid(self):
        """RAD ObsID 513 shares VIIRS-LUNAR-CAL products with WFOV ObsID 513."""
        rad = get_obsid_spec(NomHkObsidSource.RAD, 513)
        wfov = get_obsid_spec(NomHkObsidSource.WFOV, 513)
        assert rad.trimmed_product is DataProductIdentifier.l1a_icie_nom_hk_viirs_lunar_cal_trimmed
        assert rad.cal_product is DataProductIdentifier.cal_viirs_lunar_cal
        assert rad.trimmed_product is wfov.trimmed_product
        assert rad.cal_product is wfov.cal_product
        assert rad.kind is ObsIdKind.RAD_CAL
        assert wfov.kind is ObsIdKind.CAM_CAL

    def test_cam_cal_count(self):
        """Ten camera cal ObsIDs are registered."""
        cam_cal = [s for s in OBSID_REGISTRY.values() if s.kind is ObsIdKind.CAM_CAL]
        assert len(cam_cal) == 10

    def test_iter_trim_eligible_filters_by_source(self):
        """Source filter restricts iter_trim_eligible."""
        rad = list(iter_trim_eligible(NomHkObsidSource.RAD))
        wfov = list(iter_trim_eligible(NomHkObsidSource.WFOV))
        assert len(rad) == 25
        assert len(wfov) == 10
        assert all(s.source is NomHkObsidSource.RAD for s in rad)
        assert all(s.source is NomHkObsidSource.WFOV for s in wfov)

    def test_get_obsid_spec_unknown_raises(self):
        """Unknown ObsID raises KeyError."""
        with pytest.raises(KeyError, match="99999"):
            get_obsid_spec(NomHkObsidSource.RAD, 99999)
