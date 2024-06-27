"""File for testing processing step function trigger upload module"""
# Standard
import argparse
# Installed
from moto import mock_aws
# Local
from libera_utils.aws import processing_step_function_trigger as psfn


@mock_aws()
def test_processing_step_function_trigger(make_step_function):
    """Test that step function trigger uploads to AWS correctly"""
    args = argparse.Namespace(
        algorithm_name="fake_algorithm",
        day_of_interest="fake_day",
        wait_for_finish=False,
        verbose=False
    )
    make_step_function("fake_algorithm")
    resp = psfn.step_function_trigger(args)
    print("RESP IS ", resp)
