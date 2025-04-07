"""Tests for constants file"""
import pytest

from libera_utils.aws import constants
from libera_utils.aws.constants import DataProductIdentifier, ProcessingStepIdentifier


@pytest.mark.parametrize(
    'algorithm_name',
    [
        'l2-cloud-fraction', 'l2-ssw-toa', 'adms-libera', 'l2-ssw-surface-flux', 'l2-unfiltered',
        'spice-azel', 'spice-jpss', 'l1b-rad', 'l1b-cam', 'l0-jpss', 'l0-azel', 'l0-rad', 'l0-cam', 'l0-cr'
    ]
)
def test_algorithm_names(algorithm_name):
    """Test algorithm enum class to ensure we get the correct names"""
    processing_step_id = constants.ProcessingStepIdentifier(algorithm_name)
    algo_name = processing_step_id
    values = [member for member in constants.ProcessingStepIdentifier]
    assert algo_name in values

    bucket_name = processing_step_id.get_archive_bucket_name()
    bucket_names = [f"{member.get_archive_bucket_name()}" for member in constants.ProcessingStepIdentifier]
    assert bucket_name in bucket_names

    if "l0" in algorithm_name:
        # All L0 pstep IDs do not have associated ECRs
        assert processing_step_id.ecr_name is None
    else:
        # All others do.
        assert processing_step_id.ecr_name


def test_product_dump():
    product_name = DataProductIdentifier.l0_rad_pds.dump(chunk_number=0)
    assert product_name == "l0-rad-pds-0"

    product_name = DataProductIdentifier.spice_az_ck.dump()
    assert product_name == "spice-az-ck"


def test_product_validate():
    prod_enum, chunk = DataProductIdentifier.validate("l0-rad-pds")
    assert prod_enum == DataProductIdentifier.l0_rad_pds
    assert chunk is None

    prod_enum, chunk = DataProductIdentifier.validate("l0-rad-pds-0")
    assert prod_enum == DataProductIdentifier.l0_rad_pds
    assert chunk == 0

    prod_enum, chunk = DataProductIdentifier.validate("l0-cam-pds-11")
    assert prod_enum == DataProductIdentifier.l0_cam_pds
    assert chunk == 11

    with pytest.raises(ValueError, match="is not a valid DataProductIdentifier"):
        _, _ = DataProductIdentifier.validate("l0-rad-pds-bad")

    with pytest.raises(ValueError, match="is not a valid DataProductIdentifier"):
        _, _ = DataProductIdentifier.validate("spice_az_ck")


def test_step_dump():
    step_name = ProcessingStepIdentifier.l0_rad_pds.dump(chunk_number=0)
    assert step_name == "l0-rad-0"

    step_name = ProcessingStepIdentifier.spice_jpss.dump()
    assert step_name == "spice-jpss"


def test_step_validate():
    prod_enum, chunk = ProcessingStepIdentifier.validate("l0-rad")
    assert prod_enum == ProcessingStepIdentifier.l0_rad_pds
    assert chunk is None

    prod_enum, chunk = ProcessingStepIdentifier.validate("l0-rad-0")
    assert prod_enum == ProcessingStepIdentifier.l0_rad_pds
    assert chunk == 0

    prod_enum, chunk = ProcessingStepIdentifier.validate("l0-rad-11")
    assert prod_enum == ProcessingStepIdentifier.l0_rad_pds
    assert chunk == 11

    with pytest.raises(ValueError, match="is not a valid DataProductIdentifier"):
        _, _ = DataProductIdentifier.validate("l0-rad-bad")

    with pytest.raises(ValueError, match="is not a valid DataProductIdentifier"):
        _, _ = DataProductIdentifier.validate("spice_jpss")
