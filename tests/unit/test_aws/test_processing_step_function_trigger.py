"""File for testing processing step function trigger upload module"""

import argparse
from datetime import datetime
from unittest.mock import patch

import pytest
from moto import mock_aws

# Local
from libera_utils.aws import processing_step_function_trigger as psfn
from libera_utils.aws.constants import ProcessingStepIdentifier


@pytest.mark.parametrize(
    ("algorithm_name", "applicable_day", "wait_time"), [("l1b-cam", "2025-01-01", 0), ("l1b-rad", "2025-01-01", 5)]
)
@patch("libera_utils.aws.processing_step_function_trigger.step_function_trigger")
def test_step_function_trigger_cli_handler(mock_step_function_trigger, algorithm_name, applicable_day, wait_time):
    """Test the ECR upload CLI handler for file upload."""
    # Make the input namespace object
    args = argparse.Namespace(
        func=psfn.step_function_trigger_cli_handler,
        algorithm_name=algorithm_name,
        applicable_day=applicable_day,
        wait_time=wait_time,
    )

    psfn.step_function_trigger_cli_handler(args)

    expected_algorithm = ProcessingStepIdentifier(algorithm_name)
    expected_date = datetime.fromisoformat(applicable_day)
    mock_step_function_trigger.assert_called_once_with(expected_algorithm, expected_date, wait_time=wait_time)


@mock_aws()
@pytest.mark.parametrize(
    ("algorithm_name", "applicable_day", "wait_for_finish", "response", "expected_response"),
    [
        ("l1b-cam", "2030-01-01", False, None, "RUNNING"),
        (ProcessingStepIdentifier.l1b_cam, datetime.fromisoformat("2030-01-01"), False, "FAILED", "FAILED"),
        ("l1b-rad", "2030-01-01", True, None, "RUNNING"),
        (ProcessingStepIdentifier.l1b_rad, datetime.fromisoformat("2030-01-01"), True, "FAILED", "FAILED"),
    ],
)
def test_processing_step_function_trigger(
    make_step_function, algorithm_name, applicable_day, wait_for_finish, response, expected_response
):
    """Test that step function trigger uploads to AWS correctly"""
    if type(algorithm_name) is str:
        algorithm_name = ProcessingStepIdentifier(algorithm_name)

    # Mock the step function and pass in the expected response that the step function will return. If None is passed
    # then the step function will return "RUNNING" forever, if "FAILED" is passed then the step function will return
    # "FAILED" after the first call. There doesn't seem to be a way with Moto to get the SUCCEEDED response.
    make_step_function(algorithm_name.step_function_name, status=response)

    # Run the step function with a custom shortened wait time for testing
    resp = psfn.step_function_trigger(algorithm_name, applicable_day, wait_time=1)

    # If we got the response back then the triggering was correctly formatted and sent to the step function
    assert resp == expected_response
