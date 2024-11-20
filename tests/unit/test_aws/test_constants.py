"""Tests for constants file"""
# Installed
import pytest
# Local
from libera_utils.aws import constants


@pytest.mark.parametrize(
    'algorithm_name',
    [
        'l2-cloud-fraction', 'l2-ssw-toa', 'libera-adms', 'l2-ssw-surface-flux', 'l2-far-ir-toa-flux', 'l1c-unfiltered',
        'spice-azel', 'spice-jpss', 'l1b-rad', 'l1b-cam', 'l0-jpss', 'l0-azel', 'l0-rad', 'l0-cam', 'l0-cr'
    ]
)
def test_algorithm_names(algorithm_name):
    """Test algorithm enum class to ensure we get the correct names"""
    pstep_id = constants.ProcessingStepIdentifier(algorithm_name)
    algo_name = pstep_id.value
    values = [member.value for member in constants.ProcessingStepIdentifier]
    assert algo_name in values

    if "l0" in algorithm_name:
        # All L0 pstep IDs do not have associated ECRs
        assert pstep_id.ecr_name is None
    else:
        # All others do.
        assert pstep_id.ecr_name
