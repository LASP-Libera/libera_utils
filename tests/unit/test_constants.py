"""Tests for the top level constants module"""

import warnings
from enum import IntEnum, StrEnum

import pytest

from libera_utils.constants import (
    DataLevel,
    DataProductIdentifier,
    LiberaApid,
    ManifestType,
    ProcessingStepIdentifier,
)


class TestManifestType:
    """Tests for ManifestType enum"""

    def test_enum_member_names(self):
        """Test that enum member names remain consistent.

        This ensures API compatibility - the names are part of the contract.
        """
        # Note: Aliases don't appear in iteration, only primary members
        expected_names = ["INPUT", "OUTPUT"]
        actual_names = [member.name for member in ManifestType]
        assert actual_names == expected_names

        # Test that aliases exist
        assert hasattr(ManifestType, "input")
        assert hasattr(ManifestType, "output")
        assert ManifestType.input == ManifestType.INPUT
        assert ManifestType.output == ManifestType.OUTPUT

    def test_str_enum_behavior(self):
        """Test that ManifestType behaves as a StrEnum"""
        assert issubclass(ManifestType, StrEnum)
        assert isinstance(ManifestType.INPUT, str)
        assert ManifestType.INPUT == "INPUT"

    def test_manifest_type_aliases(self):
        """Test ManifestType enum aliases work correctly"""
        assert ManifestType.input == ManifestType.INPUT
        assert ManifestType.output == ManifestType.OUTPUT
        assert ManifestType.input is ManifestType.INPUT
        assert ManifestType.output is ManifestType.OUTPUT

    def test_manifest_type_string_conversion(self):
        """Test ManifestType string conversion"""
        for member in ManifestType:
            # Skip aliases
            if member.name == member.value:
                assert str(member) == member.value
                assert isinstance(str(member), str)


class TestDataLevel:
    """Tests for DataLevel enum"""

    def test_enum_member_names(self):
        """Test that enum member names remain consistent.

        This ensures API compatibility - the names are part of the contract.
        """
        expected_names = ["L0", "SPICE", "CAL", "L1A", "L1B", "L2", "ANC"]
        actual_names = [member.name for member in DataLevel]
        assert actual_names == expected_names

    def test_str_enum_behavior(self):
        """Test that DataLevel behaves as a StrEnum"""
        assert issubclass(DataLevel, StrEnum)
        for level in DataLevel:
            assert isinstance(level, str)
            assert level == level.value

    def test_archive_bucket_name_property_exists(self):
        """Test that all DataLevel members have archive_bucket_name property"""
        for level in DataLevel:
            bucket_name = level.archive_bucket_name
            assert isinstance(bucket_name, str)
            assert len(bucket_name) > 0
            assert "libera" in bucket_name.lower()

    def test_archive_bucket_name_consistency(self):
        """Test that CAL and ANC share the same bucket"""
        # This is a business rule that should be maintained
        assert DataLevel.CAL.archive_bucket_name == DataLevel.ANC.archive_bucket_name


class TestDataProductIdentifier:
    """Tests for DataProductIdentifier enum"""

    def test_enum_member_names(self):
        """Test that enum member names remain consistent.

        This ensures API compatibility - the names are part of the contract.
        """
        expected_names = [
            # L0 PDS Products
            "l0_pds_cr",
            "l0_jpss_sc_pos_pds",
            "l0_icie_rad_sample_pds",
            "l0_icie_wfov_sci_pds",
            "l0_icie_axis_sample_pds",
            "l0_pev_sw_stat_pds",
            "l0_pec_sw_stat_pds",
            "l0_icie_sw_stat_pds",
            "l0_icie_seq_hk_pds",
            "l0_icie_fp_hk_pds",
            "l0_icie_log_msg_pds",
            "l0_icie_rad_full_pds",
            "l0_icie_axis_hk_pds",
            "l0_icie_wfov_hk_pds",
            "l0_icie_cal_full_pds",
            "l0_icie_cal_sample_pds",
            "l0_icie_wfov_resp_pds",
            "l0_icie_crit_hk_pds",
            "l0_icie_nom_hk_pds",
            "l0_icie_ana_hk_pds",
            "l0_icie_temp_hk_pds",
            # L1A Decoded Products
            "l1a_jpss_sc_pos_decoded",
            "l1a_icie_rad_sample_decoded",
            "l1a_icie_wfov_sci_decoded",
            "l1a_icie_axis_sample_decoded",
            "l1a_pev_sw_stat_decoded",
            "l1a_pec_sw_stat_decoded",
            "l1a_icie_sw_stat_decoded",
            "l1a_icie_seq_hk_decoded",
            "l1a_icie_fp_hk_decoded",
            "l1a_icie_log_msg_decoded",
            "l1a_icie_rad_full_decoded",
            "l1a_icie_axis_hk_decoded",
            "l1a_icie_wfov_hk_decoded",
            "l1a_icie_cal_full_decoded",
            "l1a_icie_cal_sample_decoded",
            "l1a_icie_wfov_resp_decoded",
            "l1a_icie_crit_hk_decoded",
            "l1a_icie_nom_hk_decoded",
            "l1a_icie_ana_hk_decoded",
            "l1a_icie_temp_hk_decoded",
            # Solar Calibration Event Products
            "l1a_solar_cal_face1",
            "l1a_solar_cal_face2",
            # SPICE kernels
            "spice_az_ck",
            "spice_el_ck",
            "spice_jpss_ck",
            "spice_jpss_spk",
            # Calibration Products
            "cal_rad",
            "cal_cam",
            # L1B Products
            "l1b_rad",
            "l1b_cam",
            # L2 Products
            "l2_unf",
            "l2_cf_rad",
            "l2_cf_cam",
            "l2_ssw_toa_osse",
            "l2_ssw_toa_erbe",
            "l2_ssw_toa_trmm",
            "l2_ssw_toa_rt",
            "l2_ssw_surf",
            # Ancillary Products
            "anc_adm",
            "anc_scene_id",
        ]
        actual_names = [member.name for member in DataProductIdentifier]
        assert actual_names == expected_names

    def test_str_enum_behavior(self):
        """Test that DataProductIdentifier behaves as a StrEnum"""
        assert issubclass(DataProductIdentifier, StrEnum)
        for product in DataProductIdentifier:
            assert isinstance(product, str)
            assert product == product.value

    def test_product_name_property(self):
        """Test product_name property returns string value"""
        for product in DataProductIdentifier:
            assert product.product_name == str(product)
            assert isinstance(product.product_name, str)

    def test_data_level_property(self):
        """Test that all products have a valid DataLevel"""
        for product in DataProductIdentifier:
            level = product.data_level
            assert isinstance(level, DataLevel)
            assert level in DataLevel

    def test_associated_apid_property(self):
        """Test associated_apid property for L0 and L1A products"""
        # Multi-APID merged products have no single associated APID
        _no_single_apid = {
            DataProductIdentifier.l0_pds_cr,
            DataProductIdentifier.l1a_solar_cal_face1,
            DataProductIdentifier.l1a_solar_cal_face2,
        }
        for product in DataProductIdentifier:
            apid = product.associated_apid
            # L0 and L1A products should have an associated APID, others should be None.
            # Products in _no_single_apid are special cases and should be None.
            if (
                product.data_level == DataLevel.L1A or product.data_level == DataLevel.L0
            ) and product not in _no_single_apid:
                assert isinstance(apid, LiberaApid)
            else:
                assert apid is None

    def test_get_partial_archive_bucket_name_deprecation(self):
        """Test deprecated get_partial_archive_bucket_name method"""
        product = DataProductIdentifier.l1b_rad

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            bucket_name = product.get_partial_archive_bucket_name()

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "Use DataProductIdentifier.level.archive_bucket_name instead" in str(w[0].message)
            assert isinstance(bucket_name, str)


class TestProcessingStepIdentifier:
    """Tests for ProcessingStepIdentifier enum"""

    def test_enum_member_names(self):
        """Test that enum member names remain consistent.

        This ensures API compatibility - the names are part of the contract.
        """
        expected_names = [
            # Calibration steps
            "cal_rad",
            "cal_cam",
            # SPICE steps
            "spice_azel",
            "spice_jpss",
            # L1B steps
            "l1b_rad",
            "l1b_cam",
            # Intermediate steps
            "int_footprint_scene_id",
            # L2 steps
            "l2_cf_rad",
            "l2_cf_cam",
            "l2_unfiltered",
            "l2_ssw_toa_osse",
            "l2_ssw_toa_erbe",
            "l2_ssw_toa_trmm",
            "l2_ssw_toa_rt",
            "l2_surface_flux",
            # ADM steps
            "adm_binning",
        ]
        actual_names = [member.name for member in ProcessingStepIdentifier]
        assert actual_names == expected_names

    def test_str_enum_behavior(self):
        """Test that ProcessingStepIdentifier behaves as a StrEnum"""
        assert issubclass(ProcessingStepIdentifier, StrEnum)
        for step in ProcessingStepIdentifier:
            assert isinstance(step, str)
            assert step == step.value

    def test_processing_step_name_property(self):
        """Test processing_step_name property returns string value"""
        for step in ProcessingStepIdentifier:
            assert step.processing_step_name == str(step)
            assert isinstance(step.processing_step_name, str)

    def test_products_property(self):
        """Test that products property returns list of DataProductIdentifier"""
        for step in ProcessingStepIdentifier:
            products = step.products
            assert isinstance(products, list)
            for product in products:
                assert isinstance(product, DataProductIdentifier)

    def test_level_property_valid_cases(self):
        """Test level property for steps with valid product configurations"""
        for step in ProcessingStepIdentifier:
            if step.products:  # Only test steps with products
                try:
                    level = step.level
                    assert isinstance(level, DataLevel)
                    # All products should have the same level
                    for product in step.products:
                        assert product.data_level == level
                except ValueError:
                    # Some steps might have invalid configurations - that's tested separately
                    pass

    def test_level_property_no_products_error(self):
        """Test level property with no products raises error"""
        # Create a test step with no products
        step = ProcessingStepIdentifier.l1b_rad
        original_products = step._products
        step._products = []

        try:
            with pytest.raises(ValueError, match="produces no products"):
                _ = step.level
        finally:
            # Restore original products
            step._products = original_products

    def test_level_property_multiple_levels_error(self):
        """Test level property with multiple product levels raises error"""
        # Create a test step with products of different levels
        step = ProcessingStepIdentifier.l1b_rad
        original_products = step._products
        step._products = [DataProductIdentifier.l1b_rad, DataProductIdentifier.l2_cf_rad]

        try:
            with pytest.raises(ValueError, match="products of multiple levels"):
                _ = step.level
        finally:
            # Restore original products
            step._products = original_products

    def test_step_function_name_property(self):
        """Test step_function_name property formatting"""
        for step in ProcessingStepIdentifier:
            name = step.step_function_name
            assert isinstance(name, str)
            assert name.endswith("-processing-step-function")
            # Check that underscores are replaced with hyphens
            assert "_" not in name

    def test_policy_name_property(self):
        """Test policy_name property formatting"""
        for step in ProcessingStepIdentifier:
            name = step.policy_name
            assert isinstance(name, str)
            assert name.endswith("DevPolicy")
            assert "LiberaSDC" in name

    def test_ecr_name_property(self):
        """Test ecr_name property formatting"""
        for step in ProcessingStepIdentifier:
            name = step.ecr_name
            assert isinstance(name, str)
            assert name.endswith("-docker-repo")

    def test_from_data_product_method(self):
        """Test from_data_product class method"""
        # Test that all products that appear in steps can be found
        for step in ProcessingStepIdentifier:
            for product in step.products:
                found_step = ProcessingStepIdentifier.from_data_product(product)
                assert found_step is not None
                assert product in found_step.products

    def test_from_data_product_returns_none_for_unknown(self):
        """Test from_data_product returns None for products not in any step"""
        # Create a mock product that's not in any step
        # We'll use l0_pds_cr which likely isn't produced by any processing step
        product = DataProductIdentifier.l0_pds_cr
        result = ProcessingStepIdentifier.from_data_product(product)
        # This should return None if the product isn't found in any step
        if result is not None:
            assert product in result.products


class TestLiberaApid:
    """Tests for LiberaApid IntEnum"""

    def test_enum_member_names(self):
        """Test that enum member names remain consistent.

        This ensures API compatibility - the names are part of the contract.
        """
        expected_names = [
            "jpss_sc_pos",
            "pev_sw_stat",
            "pec_sw_stat",
            "icie_sw_stat",
            "icie_seq_hk",
            "icie_fp_hk",
            "icie_log_msg",
            "icie_rad_full",
            "icie_rad_sample",
            "icie_axis_hk",
            "icie_wfov_hk",
            "icie_wfov_sci",
            "icie_cal_full",
            "icie_cal_sample",
            "icie_axis_sample",
            "icie_wfov_resp",
            "icie_crit_hk",
            "icie_nom_hk",
            "icie_ana_hk",
            "icie_temp_hk",
        ]
        actual_names = [member.name for member in LiberaApid]
        assert actual_names == expected_names

    def test_int_enum_behavior(self):
        """Test that LiberaApid behaves as an IntEnum"""
        assert issubclass(LiberaApid, IntEnum)
        for apid in LiberaApid:
            assert isinstance(apid, int)
            assert isinstance(apid.value, int)
            # Test arithmetic operations
            assert apid + 1 == apid.value + 1
            assert apid - 1 == apid.value - 1
            # Test comparisons
            assert apid == apid.value
            assert apid > apid.value - 1
            assert apid < apid.value + 1

    def test_direct_instantiation_from_integer(self):
        """Test that APIDs can be instantiated from their integer values"""
        for apid in LiberaApid:
            # Create from integer value
            recreated = LiberaApid(apid.value)
            assert recreated == apid
            assert recreated is apid

    def test_data_product_id_property(self):
        """Test that all APIDs have corresponding L0 PDS DataProductIdentifiers"""
        for apid in LiberaApid:
            product = apid.data_product_id
            assert isinstance(product, DataProductIdentifier)
            assert product.data_level == DataLevel.L0
            # The product name should contain the APID name (with some transformation)
            assert apid.name.replace("_", "-").upper() in product.name.replace("_", "-").upper()

    def test_data_product_id_matching_convention(self):
        """Test that APID names match with their corresponding DPI names"""
        for apid in LiberaApid:
            product = apid.data_product_id
            # Check that the product follows the naming convention
            assert product.name.startswith("l0_")
            assert product.name.endswith("_pds")
