"""Module for uploading docker images to the ECR"""

import argparse
import base64
import json
import logging
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import boto3
import docker
from docker import errors as docker_errors

from libera_utils.aws import constants, utils
from libera_utils.logutil import configure_task_logging

logger = logging.getLogger(__name__)


class DockerConfigManager:
    """Context manager object, suitable for use with docker-py DockerClient.login

    If override_default_config is True, dockercfg_path points to a temporary directory
    with a blank config. Otherwise, dockercfg_path is None, which allows DockerClient.login
    to use the default config location.
    """

    _minimal_config_content = {"auths": {}, "HttpHeaders": {}}

    def __init__(self, override_default_config: bool = False):
        if override_default_config:
            self.tempdir = tempfile.TemporaryDirectory(prefix="docker-config-")  # pylint: disable=consider-using-with
            self.dockercfg_path = self.tempdir.name
            config_file_path = Path(self.dockercfg_path) / "config.json"
            logger.info(f"Overriding default docker config location with minimal config: {config_file_path}")
            with config_file_path.open("w") as f:
                json_str = json.dumps(self._minimal_config_content, indent=4)
                f.write(json_str)
        else:
            self.tempdir = None
            self.dockercfg_path = None

    def __enter__(self):
        # Return self so it can be used as a context manager
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Automatically clean up the file (if it exists) when exiting the context
        if self.tempdir:
            self.tempdir.cleanup()


def get_ecr_docker_client(region_name: str | None = None, dockercfg_path: Path | None = None) -> docker.DockerClient:
    """Perform programmatic docker login to the default ECR for the current AWS credential account (e.g. AWS_PROFILE)
    and return a DockerClient object for interacting with the ECR.

    Parameters
    ----------
    region_name : Optional[str]
        AWS region name. Each region has a separate default ECR. If region_name is None, boto3 uses the default
        region for the configured credentials.
    dockercfg_path : Optional[Path]
        Use a custom path for the Docker config file.
        (default `$HOME/.docker/config.json` if present, otherwise `$HOME/.dockercfg`)

    Returns
    -------
    : docker.DockerClient
        Logged in docker client.
    """
    logger.info("Creating a docker client for ECR")
    docker_client = docker.from_env()
    ecr_client = boto3.client("ecr", region_name=region_name)
    token = ecr_client.get_authorization_token()
    username, password = base64.b64decode(token["authorizationData"][0]["authorizationToken"]).decode().split(":")
    registry = token["authorizationData"][0]["proxyEndpoint"]
    docker_client.login(username, password, registry=registry, reauth=True, dockercfg_path=dockercfg_path)
    logger.info(f"Docker login successful. ECR registry: {registry}")
    return docker_client


def build_docker_image(
    context_dir: str | Path,
    image_name: str,
    tag: str = "latest",
    target: str | None = None,
    platform: str = "linux/amd64",
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
            path=str(context_dir.absolute()), target=target, tag=f"{image_name}:{tag}", platform=platform
        )
        # We process this output as print statements rather than logging messages because it's the direct
        # output from `docker build`
        for log in logs:
            if "stream" in log:
                print(log["stream"].strip())  # Print build output to console
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


def ecr_upload_cli_handler(parsed_args: argparse.Namespace) -> None:
    """CLI handler function for ecr-upload CLI subcommand.

    Parameters
    ----------
    parsed_args : argparse.Namespace
        Namespace of parsed CLI arguments

    Returns
    -------
    None
    """
    now = datetime.now(UTC)
    configure_task_logging(f"ecr_upload_{now}", limit_debug_loggers="libera_utils", console_log_level=logging.DEBUG)
    logger.debug(f"CLI args: {parsed_args}")
    image_name: str = parsed_args.image_name
    image_tag = parsed_args.image_tag
    algorithm_name = constants.ProcessingStepIdentifier(parsed_args.algorithm_name)
    ecr_tags = parsed_args.ecr_tags
    push_image_to_ecr(
        image_name,
        image_tag,
        algorithm_name,
        ecr_image_tags=ecr_tags,
        ignore_docker_config=parsed_args.ignore_docker_config,
    )


def push_image_to_ecr(
    image_name: str,
    image_tag: str,
    processing_step_id: str | constants.ProcessingStepIdentifier,
    *,
    ecr_image_tags: list[str] | None = None,
    region_name: str = "us-west-2",
    ignore_docker_config: bool = False,
) -> None:
    """Programmatically upload a docker image for a science algorithm to an ECR. ECR name is determined based
    on the algorithm name.

    Parameters
    ----------
    image_name : str
        Local name of the image
    image_tag : str
        Local tag of the image (often latest)
    processing_step_id : Union[str, constants.ProcessingStepIdentifier]
        Processing step ID string or object. Used to infer the ECR repository name.
        L0 processing step IDs are not allowed because they have no associated ECR.
    ecr_image_tags : Optional[List[str]]
        List of tags to apply to the pushed image in the ECR (e.g. ["1.3.4", "latest"]). Default None, results
        in pushing only as "latest".
    region_name : str
        AWS region. Used to infer the ECR name.
    ignore_docker_config : bool
        Default False. If True, creates a temporary docker config.json file to prevent using stored credentials.

    Returns
    -------
    None
    """
    if not ecr_image_tags:
        # Default to tagging the remote image as "latest"
        ecr_image_tags = ["latest"]
    if isinstance(processing_step_id, str):
        processing_step_id = constants.ProcessingStepIdentifier(processing_step_id)

    with DockerConfigManager(override_default_config=ignore_docker_config) as docker_config_manager:
        logger.info("Preparing to push image to ECR")
        docker_client = get_ecr_docker_client(
            region_name=region_name, dockercfg_path=docker_config_manager.dockercfg_path
        )
        account_id = utils.get_aws_account_number()
        ecr_name = processing_step_id.ecr_name  # The repository name within the ECR
        if ecr_name is None:
            raise ValueError(
                f"Unable to determine an ECR name for algorithm identifier: {processing_step_id}. "
                f"Note, L0 (`l0-*`) algorithm IDs do not have associated ECRs."
            )
        logger.debug(f"Algorithm name is {ecr_name}")

        # ECR path. This is really just "the registry" URL
        ecr_path = f"{account_id}.dkr.ecr.{region_name}.amazonaws.com"
        logger.debug(f"ECR path is {ecr_path}")

        for remote_tag in ecr_image_tags:
            # Tag the local image with the ECR repo name
            full_ecr_tag = f"{ecr_path}/{ecr_name}:{remote_tag}"
            logger.info(f"Tagging {image_name}:{image_tag} into ECR repo {full_ecr_tag}")
            local_image = docker_client.images.get(f"{image_name}:{image_tag}")
            local_image.tag(full_ecr_tag)

            logger.info(f"Pushing {full_ecr_tag}.")
            error_messages = []
            try:
                push_logs = docker_client.images.push(full_ecr_tag, stream=True, decode=True)
                # We process these logs as print statements because this is the direct output from docker push, not log
                # messages. We aggregate the errors to report later in an exception.
                for log in push_logs:
                    print(log)
                    # Print and keep track of any errors in the log
                    if "error" in log:
                        print(f"Error: {log['error']}")
                        error_messages.append(log["error"])

            except docker_errors.APIError as e:
                logger.error("Docker API error during image push.")
                logger.exception(e)
                raise

            if error_messages:
                raise ValueError(f"Errors encountered during image push: \n{error_messages}")

            logger.info(f"Successfully pushed {full_ecr_tag}.")

        logger.info("All tags pushed to ECR successfully.")
