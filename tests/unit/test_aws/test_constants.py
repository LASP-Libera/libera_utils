"""Tests for constants file"""
# Installed
import pytest
# Local
from libera_utils.aws import constants


@pytest.mark.parametrize(
    'algorithm_name',
    [
        'l2cf', 'l2_stf', 'adms', 'l2_surface_flux', 'l2_firf',
        'unfilt', 'spice_az', 'spice_el', 'spice_jpss', 'pds_ingest'
    ]
)
def test_algorithm_names(algorithm_name):
    """Test algorithm enum class to ensure we get the correct names"""
    algo_name = constants.ProcessingStepIdentifier[algorithm_name].value
    values = [member.value for member in constants.ProcessingStepIdentifier]
    assert algo_name in values
