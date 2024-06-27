"""Module for manually triggering a step function"""
# Standard
import argparse
from datetime import datetime, timezone
import json
import time
import logging
# Installed
import boto3
from botocore.exceptions import ClientError
# Local
from libera_utils.logutil import configure_task_logging
from libera_utils.aws import utils

logger = logging.getLogger(__name__)


def step_function_trigger(parsed_args: argparse.Namespace):
    """Start a stepfunction to process a certain days data
        Parameters
        ----------
        parsed_args : argparse.Namespace
            Namespace of parsed CLI arguments

        region_name : str
            string of the AWS region name

        Returns
        -------
        None
    """
    now = datetime.now(timezone.utc)
    configure_task_logging(f'processing_step_function_trigger_{now}',
                           limit_debug_loggers='libera_utils',
                           console_log_level=logging.DEBUG)

    logger.debug(f"CLI args: {parsed_args}")

    region_name = 'us-west-2'

    account_id = utils.get_aws_account_number()
    logger.debug(f"Account id is : {account_id}")

    state_machine_arn = f"arn:aws:states:{region_name}:{account_id}:stateMachine:{parsed_args.algorithm_name}"
    logger.debug(f"State machine is: {state_machine_arn}")

    step_function_client = boto3.client("stepfunctions", region_name)
    input_object = json.dumps({
        "TriggerSource": "Manual",
        "RetryCount": 1,
        "RetryHistory": "NA",
        "DayOfInterest": parsed_args.day_of_interest})
    logger.debug(f"Input object to the state machine is : {input_object}")

    try:
        response = step_function_client.start_execution(stateMachineArn=state_machine_arn, input=input_object)
        if parsed_args.verbose:
            logger.debug("Execution Started")
        execution_response = step_function_client.describe_execution(executionArn=response['executionArn'])
        if execution_response['status'] == "RUNNING":
            if parsed_args.verbose:
                logger.debug("Waiting for execution to complete")
            if parsed_args.wait_for_finish:
                while execution_response['status'] == "RUNNING":
                    time.sleep(5)
                    execution_response = step_function_client.describe_execution(executionArn=response['executionArn'])

    except ClientError as err:
        logger.error(
            f"Couldn't start state machine {state_machine_arn}. Here's why: {err.response['Error']['Code']}: "
            f"{err.response['Error']['Message']}")
        raise

    if parsed_args.verbose:
        logger.info(f"Reached the try loop else with ARN: {response['executionArn']}")
    if execution_response['status'] == "SUCCEEDED":
        logger.info("Execution of Step Function Succeeded")

    elif execution_response['status'] == "FAILED":
        logger.info("Execution of Step Function Failed")

    else:
        logger.info(f"Function complete step function status: {execution_response['status']}")
        logger.info("See AWS console for full details on Step ")
    if parsed_args.verbose:
        logger.debug(execution_response)
