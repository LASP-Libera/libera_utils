"""Module for manually triggering a step function"""

import argparse
import json
import logging
import time
from datetime import UTC, datetime

import boto3
from botocore.exceptions import ClientError
from ulid import ULID

from libera_utils.aws import utils
from libera_utils.constants import ProcessingStepIdentifier
from libera_utils.logutil import configure_task_logging

logger = logging.getLogger(__name__)


def step_function_trigger_cli_handler(parsed_args: argparse.Namespace):
    """CLI handler function for the step function trigger CLI subcommand.

    Parameters
    ----------
    parsed_args : argparse.Namespace
        The parsed object of CLI arguments

    """
    now = datetime.now(UTC)
    configure_task_logging(
        f"processing_step_function_trigger_{now}", limit_debug_loggers="libera_utils", console_log_level=logging.DEBUG
    )
    logger.debug(f"CLI args: {parsed_args}")

    algorithm_name = ProcessingStepIdentifier(parsed_args.algorithm_name)
    applicable_day = datetime.fromisoformat(parsed_args.applicable_day)

    step_function_trigger(algorithm_name, applicable_day, wait_time=parsed_args.wait_time)


def step_function_trigger(
    algorithm_name: str | ProcessingStepIdentifier, applicable_day: str | datetime, wait_time: int = 0
) -> str:
    """Start a stepfunction to process a certain days data
    Parameters
    ----------
    algorithm_name : Union[str, ProcessingStepIdentifier]
        The name of the algorithm to run as identified by the processing step identifier
    applicable_day : Union[str, datetime]
        The day to process data for
    wait_time : int, optional
        The time to wait between checking the status of the step function, default is 5 seconds for a maximum of
        ~30 second total wait time
    Returns
    -------
    str
        The resulting status of execution. Returns N/A if n
    """
    if isinstance(applicable_day, str):
        applicable_day = datetime.fromisoformat(applicable_day)
    if isinstance(algorithm_name, str):
        algorithm_name = ProcessingStepIdentifier(algorithm_name)

    region_name = "us-west-2"

    account_id = utils.get_aws_account_number()
    logger.debug(f"Account id is : {account_id}")

    state_machine_arn = f"arn:aws:states:{region_name}:{account_id}:stateMachine:{algorithm_name.step_function_name}"
    logger.debug(f"State machine is: {state_machine_arn}")

    ulid_timestamp = ULID.from_datetime(datetime.now(UTC))

    step_function_client = boto3.client("stepfunctions", region_name)
    input_object = json.dumps(
        {
            "detail": {
                "job_id": f"cli-manual-trigger-{ulid_timestamp}",
                "node_id": algorithm_name,
                "applicable_date": applicable_day.date().strftime("%Y-%m-%d"),
            }
        }
    )
    logger.debug(f"Input object to the state machine is : {input_object}")

    try:
        response = step_function_client.start_execution(stateMachineArn=state_machine_arn, input=input_object)
        execution_response = step_function_client.describe_execution(executionArn=response["executionArn"])
        if execution_response["status"] == "RUNNING":
            logger.debug("Execution is running.")
            if wait_time > 0:
                logger.debug("Execution is still running. Waiting...")
                # TODO[LIBSDC-615]: Let's reevaluate the amount of time needed to wait for the step function to finish
                # TODO[LIBSDC-615]: once we have a better understanding of how the L2 algorithms will take to run
                time.sleep(wait_time)
                execution_response = step_function_client.describe_execution(executionArn=response["executionArn"])

    except ClientError as err:
        logger.error(
            f"Couldn't start state machine {state_machine_arn}. Here's why: {err.response['Error']['Code']}: "
            f"{err.response['Error']['Message']}"
        )
        raise

    if execution_response["status"] == "SUCCEEDED":
        logger.info("Execution of Step Function Succeeded")
    elif execution_response["status"] == "FAILED":
        logger.info("Execution of Step Function Failed")
    else:
        logger.info(f"Function complete step function status: {execution_response['status']}")
        logger.info("See AWS console for full details on Step Function execution.\n\n")
        logger.info("" + get_stepfunction_execution_url(execution_response["executionArn"]))
        return execution_response["status"]
    logger.debug(f"Final response status was {execution_response}")
    return execution_response["status"]


def get_stepfunction_execution_url(execution_arn) -> str:
    """
    Generate AWS Console URL for Step Function execution
    """
    # Extract state machine ARN and execution name from execution ARN
    # execution_arn format: arn:aws:states:region:account:execution:stateMachineName:executionName
    parts = execution_arn.split(":")
    region = parts[3]

    # Construct the console URL
    console_url = (
        f"https://{region}.console.aws.amazon.com/states/home?region={region}#/executions/details/{execution_arn}"
    )

    return console_url
