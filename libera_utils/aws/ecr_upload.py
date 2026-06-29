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

from libera_utils.aws.utils import L2_DEVELOPER_ROLE_PATH, get_l2_team_role_session
from libera_utils.constants import ProcessingStepIdentifier
from libera_utils.logutil import configure_task_logging

logger = logging.getLogger(__name__)


def _resolve_ecr_session(algorithm: ProcessingStepIdentifier, profile_name: str | None) -> boto3.Session:
    """Build the boto3 session to use for an ECR upload of the given algorithm's image.

    L2 algorithms (those with a ``ProcessingStepIdentifier.l2_team_iam_role``) require their team's L2 Team Role to
    push to ECR, so this assumes that role. All other steps (SPICE, L1B, scene-id) use the default or ``--profile``
    session directly.

    Parameters
    ----------
    algorithm : ProcessingStepIdentifier
        The processing step whose image is being uploaded.
    profile_name : str or None
        AWS profile name from the CLI (``--profile``), or None for default resolution.

    Returns
    -------
    boto3.Session
        The session to use for the ECR upload.

    Raises
    ------
    ValueError
        If the algorithm requires an L2 Team Role that the base profile cannot assume.
    """
    team_role = algorithm.l2_team_iam_role
    if team_role is None:
        logger.info(f"{algorithm} is not an L2 algorithm; using the default/--profile session for the ECR upload.")
        return boto3.Session(profile_name=profile_name)

    role_name = f"{L2_DEVELOPER_ROLE_PATH}/{team_role}"
    logger.info(f"{algorithm} is an L2 algorithm; assuming the {role_name} role for the ECR upload.")
    return get_l2_team_role_session(profile_name=profile_name, role_name=role_name)


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


def _push_single_tag(
    docker_client: docker.DockerClient,
    local_image: docker.models.images.Image,
    full_ecr_tag: str,
    region_name: str,
    max_retries: int = 3,
    boto_session: boto3.Session = None,
) -> None:
    """Push a single tagged image to ECR with retry logic and fresh authentication.

    Parameters
    ----------
    docker_client : docker.DockerClient
        Docker client instance
    local_image : docker.models.images.Image
        Local Docker image to push
    full_ecr_tag : str
        Complete ECR tag (registry/repository:tag)
    region_name : str
        AWS region name
    max_retries : int
        Maximum retry attempts
    boto_session : boto3.Session
        Boto3 session used to obtain ECR credentials (already role-assumed if needed)
    """
    for attempt in range(max_retries + 1):
        try:
            # Get fresh ECR credentials for this push attempt
            auth_config = _get_fresh_ecr_auth(region_name, boto_session=boto_session)

            # Tag the local image
            local_image.tag(full_ecr_tag)
            logger.info(f"Tagged local image with: {full_ecr_tag}")

            # Push with explicit authentication
            logger.info(f"Pushing {full_ecr_tag} (attempt {attempt + 1}/{max_retries + 1})")

            push_logs = docker_client.api.push(full_ecr_tag, stream=True, decode=True, auth_config=auth_config)

            error_messages = []
            for log in push_logs:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Push log: {log}")

                if "error" in log:
                    error_message = log["error"]
                    logger.error(f"Push error: {error_message}")
                    error_messages.append(error_message)

            if error_messages:
                raise ValueError(f"Push errors: {error_messages}")

            # Success - break out of retry loop
            break

        except (docker_errors.APIError, ValueError) as e:
            if attempt < max_retries:
                logger.warning(f"Push attempt {attempt + 1} failed, retrying: {e}")
                continue
            else:
                logger.error(f"Push failed after {max_retries + 1} attempts")
                raise


def _get_fresh_ecr_auth(region_name: str, *, boto_session: boto3.Session) -> dict:
    """Get fresh ECR authentication configuration.

    Parameters
    ----------
    region_name : str
        AWS region name
    boto_session : boto3.Session
        Boto3 session used to obtain the ECR authorization token (already role-assumed if needed)

    Returns
    -------
    dict
        Authentication configuration for Docker API
    """
    try:
        ecr_client = boto_session.client("ecr", region_name=region_name)
        token_response = ecr_client.get_authorization_token()

        auth_data = token_response["authorizationData"][0]
        token = auth_data["authorizationToken"]

        # Decode base64 token to get username:password
        username, password = base64.b64decode(token).decode().split(":", 1)

        return {"username": username, "password": password}

    except Exception as e:
        logger.exception(f"Error obtaining ECR authorization token. {e}", stack_info=True)
        raise


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
        logger.exception(f"Failed to build docker image. {e}", stack_info=True)
        raise
    except docker_errors.APIError as e:
        logger.exception(f"Docker API error. {e}", stack_info=True)
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
    algorithm_name = ProcessingStepIdentifier(parsed_args.algorithm_name)
    ecr_tags = parsed_args.ecr_tags
    profile_name = parsed_args.profile

    # L2 algorithms require their team's L2 Team Role to push to ECR; other steps use the default/--profile session.
    try:
        boto_session = _resolve_ecr_session(algorithm_name, profile_name)
    except ValueError:
        # The raised error already names the base role and target role. Add the algorithm-specific remediation: this
        # is the team-membership cause (you are the right base role but not in the L2 Team Role's user list).
        logger.error(
            "Could not assume the %s/%s role required to upload the %s algorithm image. If you are signed in with "
            "the correct L2 Developer base-role profile, contact the SDC Team to be added to the list of users for "
            "that L2 Team Role.",
            L2_DEVELOPER_ROLE_PATH,
            algorithm_name.l2_team_iam_role,
            algorithm_name,
        )
        raise

    push_image_to_ecr(
        image_name,
        image_tag,
        algorithm_name,
        ecr_image_tags=ecr_tags,
        ignore_docker_config=parsed_args.ignore_docker_config,
        boto_session=boto_session,
    )


def push_image_to_ecr(
    image_name: str,
    image_tag: str,
    processing_step_id: str | ProcessingStepIdentifier,
    *,
    ecr_image_tags: list[str] = None,
    region_name: str = "us-west-2",
    ignore_docker_config: bool = False,
    max_retries: int = 1,
    boto_session: boto3.Session | None = None,
) -> None:
    """Push a Docker image to Amazon ECR with robust authentication handling.

    This function handles ECR authentication by obtaining fresh credentials for each
    push operation, preventing authentication token expiration issues during
    multi-tag pushes.

    Parameters
    ----------
    image_name : str
        Local name of the Docker image
    image_tag : str
        Local tag of the Docker image (often 'latest')
    processing_step_id : Union[str, ProcessingStepIdentifier]
        Processing step ID string or object used to determine ECR repository name.
        L0 processing step IDs are not supported as they have no associated ECR.
    ecr_image_tags : Optional[List[str]], default None
        Tags to apply to the pushed image in ECR (e.g., ["1.3.4", "latest"]).
        If None, defaults to ["latest"].
    region_name : str, default "us-west-2"
        AWS region containing the target ECR registry
    ignore_docker_config : bool, default False
        If True, creates a temporary Docker config to prevent using stored credentials
    max_retries : int, default 3
        Maximum number of retry attempts for failed push operations
    boto_session : boto3.Session, optional
        Boto3 session used for ECR operations (already role-assumed if needed). If None, a default session is created
        (so callers that don't need role assumption, e.g. libera_cdk integration tests, can omit it).

    Raises
    ------
    ValueError
        If processing_step_id cannot be mapped to an ECR repository name,
        or if push operations encounter errors after all retries
    docker.errors.APIError
        If Docker API operations fail
    boto3.exceptions.ClientError
        If AWS ECR operations fail

    Returns
    -------
    None
    """
    # Input validation and defaults
    if not ecr_image_tags:
        ecr_image_tags = ["latest"]

    if isinstance(processing_step_id, str):
        processing_step_id = ProcessingStepIdentifier(processing_step_id)

    # Default to a plain session when no (role-assumed) session is provided, so callers that don't need role
    # assumption can omit it.
    if boto_session is None:
        boto_session = boto3.Session()

    with DockerConfigManager(override_default_config=ignore_docker_config):
        logger.info(f"Starting ECR push for image {image_name}:{image_tag}")

        # Get AWS account and ECR repository information. Deriving the account from the session ensures the registry
        # account matches the credentials performing the push.
        account_id = boto_session.client("sts").get_caller_identity()["Account"]
        ecr_name = processing_step_id.ecr_name

        if ecr_name is None:
            raise ValueError(
                f"Unable to determine ECR repository name for processing step: {processing_step_id}. "
                f"Note: L0 processing steps (l0-*) do not have associated ECR repositories."
            )

        ecr_registry = f"{account_id}.dkr.ecr.{region_name}.amazonaws.com"
        logger.info(f"Target ECR registry: {ecr_registry}/{ecr_name}")

        # Verify local image exists before attempting pushes
        docker_client = docker.from_env()
        try:
            local_image = docker_client.images.get(f"{image_name}:{image_tag}")
        except docker.errors.ImageNotFound:
            raise ValueError(f"Local image not found: {image_name}:{image_tag}")

        successful_pushes = []

        for remote_tag in ecr_image_tags:
            full_ecr_tag = f"{ecr_registry}/{ecr_name}:{remote_tag}"

            try:
                _push_single_tag(
                    docker_client=docker_client,
                    local_image=local_image,
                    full_ecr_tag=full_ecr_tag,
                    region_name=region_name,
                    max_retries=max_retries,
                    boto_session=boto_session,
                )
                successful_pushes.append(remote_tag)
                logger.info(f"Successfully pushed tag: {remote_tag}")

            except Exception as e:
                logger.exception(f"Failed to push tag {remote_tag}: {e}", stack_info=True)
                # Clean up any successful pushes on failure (optional)
                if successful_pushes:
                    logger.warning(f"Partial success: pushed tags {successful_pushes} before failure")
                raise

        logger.info(
            f"All {len(ecr_image_tags)} tags pushed successfully to ECR. Remote tags pushed: {successful_pushes}"
        )
