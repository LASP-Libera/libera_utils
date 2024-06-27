"""Module for uploading docker images to the ECR"""
# Standard
import argparse
from datetime import datetime, timezone
import subprocess  # nosec
import logging
import json
# Installed
import docker
# Local
from libera_utils.logutil import configure_task_logging
from libera_utils.aws import constants, utils

logger = logging.getLogger(__name__)


def login_to_ecr(account_id, region_name):
    """Login to the AWS ECR using commands
    Parameters
    ----------
    account_id : int
        Users AWS account ID

    region_name : string
        String of the region that the users AWS account is in

    Returns
    -------
    result : CompletedProcess
        subproccess object that holds the details of the completed CLI command
    """
    ecr_path = f"{account_id}.dkr.ecr.{region_name}.amazonaws.com"
    logger.debug(f'ECR path is {ecr_path}')

    # Login to ECR using subprocess
    ecr_login_command = f"aws ecr get-login-password --region {region_name} | docker login --username AWS " \
                        f"--password-stdin {ecr_path}"
    result = subprocess.run(ecr_login_command,  # nosec
                            shell=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True, check=False)
    return result


def upload_image_to_ecr(parsed_args: argparse.Namespace):
    """Upload docker image to the correct ECR repository

    Parameters
    ----------
    parsed_args : argparse.Namespace
        Namespace of parsed CLI arguments

    Returns
    -------
    None
    """
    now = datetime.now(timezone.utc)
    configure_task_logging(f'ecr_upload_{now}',
                           limit_debug_loggers='libera_utils',
                           console_log_level=logging.DEBUG)
    # if parsed_args.verbose else None
    docker_client = docker.from_env()
    logger.debug(f"CLI args: {parsed_args}")
    image_name = parsed_args.image_name
    if parsed_args.image_tag:
        image_tag = parsed_args.image_tag
    else:
        docker_image = docker_client.images.get(image_name)
        image_tag = docker_image.tags

    region_name = "us-west-2"
    logger.debug(f'Region set to {region_name}')

    account_id = utils.get_aws_account_number()
    logger.debug(f'Account ID is {account_id}')

    ecr_name = constants.ProcessingStepIdentifier[parsed_args.algorithm_name].value
    logger.debug(f'Algorithm name is {ecr_name}')

    # ECR path
    ecr_path = f"{account_id}.dkr.ecr.{region_name}.amazonaws.com"
    logger.debug(f'ECR path is {ecr_path}')

    # Login to ECR using subprocess
    result = login_to_ecr(account_id=account_id, region_name=region_name)
    if result.returncode == 0:
        logger.info(f"Docker Login successful. STDOUT: {result.stdout}")
    else:
        logger.error(f"STDERR: {result.stderr}")
        raise RuntimeError(f"ECR login command: {result.stderr} failed.")

    # # Tag the latest libera_utils image with the ECR repo name
    docker_client.images.get(f"{image_name}:{image_tag}").tag(f"{ecr_path}/{ecr_name}")
    logger.info("Pushing image to ECR.")
    resp = docker_client.images.push(f"{ecr_path}/{ecr_name}", stream=True, decode=True)
    error_message = []
    for line in resp:
        message = json.loads(json.dumps([line]))[0]
        if parsed_args.verbose:
            logger.debug(message)

        if "error" in message:
            error_message.append(message)

    if error_message:
        logger.error(f"Errors were encountered during image push. Error was: \n{error_message}")
        return
    logger.info("Image pushed to ECR successfully.")
