"""Module for uploading docker images to the ECR"""
# Standard
import argparse
import base64
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Union, Optional
# Installed
import boto3
import docker
from docker import errors as docker_errors
# Local
from libera_utils.logutil import configure_task_logging
from libera_utils.aws import constants, utils

logger = logging.getLogger(__name__)


def get_ecr_docker_client(region_name: Optional[str] = None) -> docker.DockerClient:
    """Perform programmatic docker login to the default ECR for the current AWS credential account (e.g. AWS_PROFILE)
    and return a DockerClient object for interacting with the ECR.

    Parameters
    ----------
    region_name : Optional[str]
        AWS region name. Each region has a separate default ECR. If region_name is None, boto3 uses the default
        region for the configured credentials.

    Returns
    -------
    : docker.DockerClient
        Logged in docker client.
    """
    logger.info("Creating a docker client for ECR")
    docker_client = docker.from_env()
    ecr_client = boto3.client('ecr', region_name=region_name)
    token = ecr_client.get_authorization_token()
    username, password = base64.b64decode(token['authorizationData'][0]['authorizationToken']).decode().split(':')
    registry = token['authorizationData'][0]['proxyEndpoint']
    docker_client.login(username, password, registry=registry)
    logger.info(f"Docker login successful. ECR registry: {registry}")
    return docker_client


def build_docker_image(
        context_dir: Union[str, Path],
        image_name: str,
        tag: str = "latest",
        target: Optional[str]=None,
        platform: str = "linux/amd64"
) -> None:
    """
    Build a Docker image from a specified directory and tag it with a custom name.

    Parameters
    ----------
    context_dir : Union[str, Path]
        The path to the directory containing the Dockerfile and other build context.
    image_name : str
        The name to give the Docker image.
    tag : str, optional
        The tag to apply to the image (default is 'latest').
    target : Optional[str]
        Name of the target to build.
    platform : str
        Default "linux/amd64".

    Raises
    ------
    ValueError
        If the specified directory does not exist or the build fails.
    """
    context_dir = Path(context_dir)
    # Check if the directory exists
    if not context_dir.is_dir():
        raise ValueError(f"Directory {context_dir} does not exist.")

    # Initialize the Docker client
    client = docker.from_env()

    # Build the Docker image
    logger.info(f"Building docker target {target} in context directory {context_dir}")
    try:
        _, logs = client.images.build(
            path=str(context_dir.absolute()),
            target=target,
            tag=f"{image_name}:{tag}",
            platform=platform
        )
        # We process this output as print statements rather than logging messages because it's the direct
        # output from `docker build`
        for log in logs:
            if 'stream' in log:
                print(log['stream'].strip())  # Print build output to console
        print(f"Image {image_name}:{tag} built successfully.")
    except docker_errors.BuildError as e:
        logger.error("Failed to build docker image.")
        logger.exception(e)
        raise
    except docker_errors.APIError as e:
        logger.error("Docker API error.")
        logger.exception(e)
        raise
    logger.info(f"Image built successfully and tagged as {image_name}:{tag}")


def ecr_upload_cli_func(parsed_args: argparse.Namespace) -> None:
    """CLI handler function for ecr-upload CLI subcommand.

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
    logger.debug(f"CLI args: {parsed_args}")
    verbose: bool = parsed_args.verbose
    image_name: str = parsed_args.image_name
    image_tag = parsed_args.image_tag
    algorithm_name = parsed_args.algorithm_name
    push_image_to_ecr(image_name, image_tag, algorithm_name, verbose=verbose)


def push_image_to_ecr(image_name: str,
                      image_tag: str,
                      algorithm_name: Union[str, constants.ProcessingStepIdentifier],
                      region_name: str = "us-west-2",
                      verbose: bool = False) -> None:
    """Programmatically upload a docker image for a science algorithm to an ECR. ECR name is determined based
    on the algorithm name.

    Parameters
    ----------
    image_name : str
        Name of the image
    image_tag : str
        Tag of the image (often latest)
    algorithm_name : Union[str, constants.ProcessingStepIdentifier]
        Processing step ID string or object. Used to infer the ECR repository name.
    region_name : str
        AWS region. Used to infer the ECR name.
    verbose : bool
        Enable debug logging

    Returns
    -------
    None
    """
    logger.info("Preparing to push image to ECR")
    docker_client = get_ecr_docker_client(region_name=region_name)
    logger.debug(f'Region set to {region_name}')

    account_id = utils.get_aws_account_number()
    logger.debug(f'Account ID is {account_id}')

    algorithm_identifier = constants.ProcessingStepIdentifier(algorithm_name)
    ecr_name = algorithm_identifier.ecr_name  # The repostiory name within the ECR
    logger.debug(f'Algorithm name is {ecr_name}')

    # ECR path. This is really just "the registry" URL
    ecr_path = f"{account_id}.dkr.ecr.{region_name}.amazonaws.com"
    logger.debug(f'ECR path is {ecr_path}')

    # Tag the local image with the ECR repo name
    logger.info(f"Tagging {image_name}:{image_tag} into ECR repo {ecr_path}/{ecr_name}")
    docker_client.images.get(f"{image_name}:{image_tag}").tag(f"{ecr_path}/{ecr_name}")

    logger.info("Pushing {ecr_path}/{ecr_name} to ECR.")
    error_messages = []
    try:
        push_logs = docker_client.images.push(f"{ecr_path}/{ecr_name}", stream=True, decode=True)
        # We process these logs as print statements because this is the direct output from docker push, not log
        # messages. We aggregate the errors to report later in an exception.
        for log in push_logs:
            print(log)
            # Print and keep track of any errors in the log
            if 'error' in log:
                print(f"Error: {log['error']}")
                error_messages.append(log['error'])

    except docker_errors.APIError as e:
        logger.error("Docker API error during image push.")
        logger.exception(e)
        raise

    if error_messages:
        raise ValueError(f"Errors encountered during image push: \n{error_messages}")

    logger.info("Image pushed to ECR successfully.")
