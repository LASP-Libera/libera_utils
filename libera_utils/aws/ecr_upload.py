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
from botocore.exceptions import ClientError
from docker import errors as docker_errors

from libera_utils.aws.algorithm_registration import put_new_algorithm_image_event
from libera_utils.aws.utils import L2_DEVELOPER_ROLE_PATH, _resolve_algorithm_specific_session, _session_region
from libera_utils.constants import ProcessingStepIdentifier
from libera_utils.logutil import configure_task_logging

logger = logging.getLogger(__name__)


class ImageTagAlreadyExistsError(ValueError):
    """A requested version tag already names an image in the ECR repository, found before pushing or refused by ECR."""


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


# Docker push-stream ``status`` values that indicate a layer reached a terminal state (as opposed to the
# high-frequency "Pushing" progress ticks, which we suppress).
_LAYER_PUSHED_STATUS = "Pushed"
_LAYER_EXISTS_STATUS = "Layer already exists"

# The one ECR tag that algorithm repositories leave mutable. It is a moving pointer, never a registered version.
_LATEST_TAG = "latest"

# Text shared by both wordings of Amazon ECR's TAG_INVALID registry error for a tag that already exists in an
# immutable repository: "... cannot be overwritten because the repository is immutable." and, since 2025,
# "... cannot be overwritten because the tag is immutable."
_TAG_ALREADY_EXISTS_MARKER = "cannot be overwritten"


def _tag_in_use_message(tags: list[str], repository: str, console_url: str) -> str:
    """Describe version tags that already exist in an ECR repository and what to do about them.

    Parameters
    ----------
    tags : list[str]
        The version tags already in use, at least one.
    repository : str
        The ECR registry and repository name (``<registry>/<repository>``).
    console_url : str
        AWS console page of the ECR repository.

    Returns
    -------
    str
        The message, naming every tag and the repository and linking the console page.
    """
    quoted = ", ".join(repr(tag) for tag in tags)
    if len(tags) == 1:
        in_use = f"Tag {quoted} is already in use"
        remedy = f"under a new version tag, or, to reuse {quoted}, delete the image it tags"
    else:
        in_use = f"Tags {quoted} are already in use"
        remedy = f"under new version tags, or, to reuse {quoted}, delete the images they tag"
    return (
        f"{in_use} in ECR repository {repository!r}. ecr-upload does not push a version tag that already exists. "
        f"Push the image {remedy} by hand, for example in the AWS console at {console_url}, and push again."
    )


def _process_push_logs(push_logs, full_ecr_tag: str, console_url: str) -> str | None:
    """Consume a decoded Docker push stream, log a concise summary, and return the pushed image digest.

    The raw Docker push stream is extremely chatty: it emits many per-layer ``"Pushing"`` progress events
    (byte counters and terminal progress bars) that are meaningless in a line-based log. This processor
    suppresses those, logs one readable line per layer transition at DEBUG, surfaces the terminal digest and
    a single summary at INFO, and raises on any error the stream reports.

    Parameters
    ----------
    push_logs : iterable of dict
        The decoded events yielded by ``docker_client.api.push(..., stream=True, decode=True)``.
    full_ecr_tag : str
        Complete ECR tag (registry/repository:tag) being pushed, used for log context.
    console_url : str
        AWS console page of the ECR repository being pushed to, linked when ECR refuses an existing tag.

    Returns
    -------
    str or None
        The image digest (``sha256:...``) reported in the stream's ``aux`` event, or None if not present.

    Raises
    ------
    ImageTagAlreadyExistsError
        If ECR refuses the push because the tag already exists in an immutable repository.
    ValueError
        If the stream reports any other error (via the ``error`` or ``errorDetail`` fields).
    """
    pushed_layers = 0
    existing_layers = 0
    digest: str | None = None
    error_messages: list[str] = []

    for event in push_logs:
        # Errors are reported via "error" (a string) and/or "errorDetail" (a dict with a "message").
        if "error" in event or "errorDetail" in event:
            error_message = event.get("error") or event.get("errorDetail", {}).get("message") or str(event)
            logger.error(f"Push error for {full_ecr_tag}: {error_message}")
            error_messages.append(error_message)
            continue

        # The terminal "aux" event carries the digest/size of the pushed manifest.
        if "aux" in event:
            digest = event["aux"].get("Digest")
            continue

        status = event.get("status")
        if status is None:
            continue

        if status == _LAYER_PUSHED_STATUS:
            pushed_layers += 1
            logger.debug(f"Layer pushed ({event.get('id', '?')}) for {full_ecr_tag}")
        elif status == _LAYER_EXISTS_STATUS:
            existing_layers += 1
            logger.debug(f"Layer already exists ({event.get('id', '?')}) for {full_ecr_tag}")
        elif "progressDetail" in event:
            # High-frequency per-layer progress ticks ("Pushing"/"Preparing"/"Waiting"): suppressed entirely
            # to avoid flooding the log with byte counters and progress bars.
            continue
        else:
            # Stream-level status lines without a progressDetail (e.g. the terminal "<tag>: digest: ...").
            logger.debug(f"Push status for {full_ecr_tag}: {status}")

    for error_message in error_messages:
        if _TAG_ALREADY_EXISTS_MARKER in error_message:
            repository, _, tag = full_ecr_tag.rpartition(":")
            raise ImageTagAlreadyExistsError(
                f"{_tag_in_use_message([tag], repository, console_url)} ECR reported: {error_message}"
            )
    if error_messages:
        raise ValueError(f"Push errors: {error_messages}")

    logger.info(
        "Pushed %s: %d layer(s) pushed, %d already existed%s",
        full_ecr_tag,
        pushed_layers,
        existing_layers,
        f" (digest {digest})" if digest else "",
    )
    return digest


def _push_single_tag(
    docker_client: docker.DockerClient,
    local_image: docker.models.images.Image,
    full_ecr_tag: str,
    console_url: str,
    region_name: str,
    max_retries: int = 3,
    boto_session: boto3.Session | None = None,
) -> str | None:
    """Push a single tagged image to ECR with retry logic and fresh authentication.

    Parameters
    ----------
    docker_client : docker.DockerClient
        Docker client instance
    local_image : docker.models.images.Image
        Local Docker image to push
    full_ecr_tag : str
        Complete ECR tag (registry/repository:tag)
    console_url : str
        AWS console page of the ECR repository, linked when ECR refuses an existing tag
    region_name : str
        AWS region name
    max_retries : int
        Maximum retry attempts
    boto_session : boto3.Session
        Boto3 session used to obtain ECR credentials (already role-assumed if needed)

    Returns
    -------
    str or None
        The image digest (``sha256:...``) reported for the pushed image, or None if not available.

    Raises
    ------
    ImageTagAlreadyExistsError
        If ECR refuses the tag because it already exists in an immutable repository. This is not retried.
    ValueError
        If the push stream reports any other error on the last attempt.
    docker.errors.APIError
        If the Docker API fails on the last attempt.
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

            # Success - return the digest reported by the stream
            return _process_push_logs(push_logs, full_ecr_tag, console_url)

        except ImageTagAlreadyExistsError:
            # ECR refuses an existing tag every time, so a retry would only re-upload the layers.
            raise
        except (docker_errors.APIError, ValueError) as e:
            if attempt < max_retries:
                logger.warning(f"Push attempt {attempt + 1} failed, retrying: {e}")
                continue
            else:
                logger.error(f"Push failed after {max_retries + 1} attempts")
                raise
    return None


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


def _existing_version_tags(
    boto_session: boto3.Session, region_name: str, repository_name: str, tags: list[str]
) -> list[str]:
    """Return the requested version tags that already exist in an ECR repository.

    ``latest`` is never checked, since it is the one tag that is pushed again. Each remaining tag takes its own
    ``DescribeImages`` call, because one call naming a missing tag fails as a whole. This answers whether tags exist
    rather than raising when one is missing, and treats a missing repository as an error, which is why it does not
    use ``algorithm_registration._verify_ecr_image_exists``.

    Parameters
    ----------
    boto_session : boto3.Session
        Session used for the ECR calls (already role-assumed if needed); it needs ``ecr:DescribeImages``.
    region_name : str
        AWS region of the ECR repository.
    repository_name : str
        Name of the ECR repository.
    tags : list[str]
        The requested ECR tags.

    Returns
    -------
    list[str]
        The requested tags other than ``latest`` that already exist, in the order requested.

    Raises
    ------
    botocore.exceptions.ClientError
        If ECR cannot answer for any reason other than a missing tag, such as a missing repository or a session
        without ``ecr:DescribeImages``.
    """
    version_tags = [tag for tag in tags if tag != _LATEST_TAG]
    if not version_tags:
        return []

    ecr_client = boto_session.client("ecr", region_name=region_name)
    existing = []
    for tag in version_tags:
        try:
            ecr_client.describe_images(repositoryName=repository_name, imageIds=[{"imageTag": tag}])
        except ClientError as e:
            if e.response["Error"]["Code"] == "ImageNotFoundException":
                continue
            logger.error(
                "Could not check ECR repository %s for existing version tags (%s), so nothing was pushed. If the "
                "repository was not found, check the algorithm name and the AWS account; if access was denied, the "
                "check needs ecr:DescribeImages on the repository, so contact the SDC team.",
                repository_name,
                e.response["Error"]["Code"],
            )
            raise
        existing.append(tag)
    return existing


def _split_local_image_reference(image_reference: str, image_tag: str | None = None) -> tuple[str, str]:
    """Resolve a local Docker image reference, and an optionally separate tag, into a (name, tag) pair.

    Two syntaxes are supported. The preferred one carries the tag in the reference itself
    (``my-image:1.2.3``); the deprecated one passes a bare image name and supplies the tag separately
    (the ``--image-tag`` CLI option). Exactly one of them must carry a tag: ``latest`` is deliberately
    not assumed, because algorithm images are built under an explicit ``docker build -t`` version and a
    local ``latest`` usually does not exist.

    A colon is only read as a tag separator when it appears in the last path component, so a registry
    reference carrying a port (``localhost:5000/my-image``) is not mistaken for a tagged image.

    Parameters
    ----------
    image_reference : str
        Local image reference, either ``image-name`` or ``image-name:tag``.
    image_tag : str, optional
        Tag supplied separately from the reference (the deprecated ``--image-tag`` form). If None
        (the default), the tag is taken from ``image_reference``.

    Returns
    -------
    tuple[str, str]
        The image name (with any tag removed) and the resolved tag.

    Raises
    ------
    ValueError
        If no tag is supplied at all, if ``image_reference`` is malformed (empty name or empty tag), if it
        is a digest reference (``my-image@sha256:...``, which is not supported), or if a tag in
        ``image_reference`` conflicts with a separately supplied ``image_tag``.
    """
    if image_tag is not None and not image_tag:
        raise ValueError(
            "An empty image tag was supplied. Give the tag explicitly as part of the image reference "
            "(e.g. my-image:1.2.3)."
        )

    if "@" in image_reference:
        raise ValueError(
            f"Digest references are not supported for the local image: {image_reference!r}. "
            f"Specify the image by tag instead (e.g. my-image:1.2.3)."
        )

    image_name, separator, tag_from_reference = image_reference.rpartition(":")
    if not separator or "/" in tag_from_reference:
        # Either there is no tag at all, or the colon belongs to a registry host:port such as
        # localhost:5000/my-image, in which case the whole reference is the image name.
        image_name, tag_from_reference = image_reference, None
    elif not tag_from_reference:
        raise ValueError(f"Image reference {image_reference!r} ends in ':' with no tag after it.")

    if not image_name:
        raise ValueError(f"Image reference {image_reference!r} has no image name before the ':'.")

    if image_tag is None:
        if tag_from_reference is None:
            raise ValueError(
                f"No tag given for local image {image_reference!r}. Specify the tag explicitly, e.g. "
                f"{image_name}:1.2.3. 'latest' is not assumed, so pass {image_name}:latest if that is "
                f"really the image you mean."
            )
        return image_name, tag_from_reference

    if tag_from_reference is not None and tag_from_reference != image_tag:
        raise ValueError(
            f"Conflicting local image tags: the image reference {image_reference!r} specifies tag "
            f"{tag_from_reference!r} but --image-tag specifies {image_tag!r}. Specify the tag only once, "
            f"preferably as part of the image reference (e.g. {image_name}:{tag_from_reference})."
        )

    return image_name, image_tag


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
        # Stream the raw `docker build` output through the logger at DEBUG so it goes through the same
        # handlers as everything else (visible with -v) instead of bypassing them via print().
        for log in logs:
            if "stream" in log:
                build_line = log["stream"].strip()
                if build_line:
                    logger.debug(build_line)
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

    Raises
    ------
    ImageTagAlreadyExistsError
        If a requested version tag already exists in the repository, in which case nothing is pushed, or if ECR refuses
        a tag taken by another push since that check, in which case the tags not yet pushed are skipped. No version is
        registered.
    ValueError
        If the local image reference is malformed, the role cannot be assumed, the local image is not found, or the
        push fails for any other reason.
    botocore.exceptions.ClientError
        If the check for existing tags, or another ECR call, fails.
    """
    now = datetime.now(UTC)
    # The Docker push stream is very chatty; default the console to INFO and let -v opt into DEBUG detail.
    console_log_level = logging.DEBUG if parsed_args.verbose else logging.INFO
    configure_task_logging(f"ecr_upload_{now}", limit_debug_loggers="libera_utils", console_log_level=console_log_level)
    logger.debug(f"CLI args: {parsed_args}")
    # Resolve the local image reference up front so a malformed or conflicting tag fails immediately, before
    # the (slower, and separately fallible) role assumption below.
    image_name, image_tag = _split_local_image_reference(parsed_args.image_name, parsed_args.image_tag)
    # Warn using the *resolved* name and tag so the suggested command is correct even for references whose
    # colon is not a tag separator (e.g. localhost:5000/my-image). Guard on the raw argument, since the
    # resolved tag is never None.
    if parsed_args.image_tag is not None:
        logger.warning(
            "--image-tag is deprecated. Specify the tag as part of the image reference instead, e.g. "
            "`libera-utils ecr-upload %s %s:%s`.",
            parsed_args.algorithm_name,
            image_name,
            image_tag,
        )
    algorithm_name = ProcessingStepIdentifier(parsed_args.algorithm_name)
    ecr_tags = parsed_args.ecr_tags
    profile_name = parsed_args.profile

    # L2 algorithms require their team's L2 Team Role to push to ECR; other steps use the ambient/--profile session.
    try:
        boto_session = _resolve_algorithm_specific_session(algorithm_name, profile_name)
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

    pushed_digests = push_image_to_ecr(
        image_name,
        image_tag,
        algorithm_name,
        ecr_image_tags=ecr_tags,
        ignore_docker_config=parsed_args.ignore_docker_config,
        boto_session=boto_session,
    )

    # Uploading an image is always paired with registering its version(s): an unregistered image cannot be run by
    # version. "latest" is a moving pointer, not a concrete algorithm version, so it is never registered.
    versions_to_register = [tag for tag in pushed_digests if tag != _LATEST_TAG]
    if not versions_to_register:
        logger.warning(
            "No concrete (non-'latest') ECR tag was pushed, so there is no algorithm version to register. "
            "Re-run with an explicit version tag (e.g. --ecr-tags latest 1.2.3) to register one."
        )
        return

    # Reuse the session resolved for the push to also emit the registration event(s): L2 algorithms use their
    # per-team L2 Team Role, while non-L2 (SDC-owned) algorithms use the ambient/--profile session. This lets SDC
    # developers -- whose admin credentials cannot assume the LiberaUtils role -- register non-L2 images with their
    # ambient credentials instead of failing on a role-assumption chain they are not part of.
    for version in versions_to_register:
        put_new_algorithm_image_event(
            algorithm_name,
            version,
            boto_session=boto_session,
            image_digest=pushed_digests[version],
            verify=parsed_args.verify,
            timeout=parsed_args.timeout,
        )


def push_image_to_ecr(
    image_name: str,
    image_tag: str | None,
    processing_step_id: str | ProcessingStepIdentifier,
    *,
    ecr_image_tags: list[str] | None = None,
    region_name: str | None = None,
    ignore_docker_config: bool = False,
    max_retries: int = 1,
    boto_session: boto3.Session | None = None,
) -> dict[str, str | None]:
    """Push a Docker image to Amazon ECR with robust authentication handling.

    This function handles ECR authentication by obtaining fresh credentials for each
    push operation, preventing authentication token expiration issues during
    multi-tag pushes.

    Parameters
    ----------
    image_name : str
        Local reference of the Docker image, either a bare name ('my-image') or a name with its tag
        ('my-image:1.2.3'). A tag given here must not conflict with `image_tag`.
    image_tag : str or None
        Local tag of the Docker image, for callers that keep the tag separate from the name. Pass None to
        take the tag from `image_name` instead, which must then carry one ('latest' is not assumed).
    processing_step_id : Union[str, ProcessingStepIdentifier]
        Processing step ID string or object used to determine ECR repository name.
        L0 processing step IDs are not supported as they have no associated ECR.
    ecr_image_tags : list[str] | None, optional
        Tags to apply to the pushed image in ECR (e.g., ["1.3.4", "latest"]).
        If None (the default), defaults to ["latest"]. Every tag other than "latest" is checked in ECR first, and
        none is pushed if any of them already exists. Tags other than "latest" are then pushed first, in the order
        given, so a tag taken by another push since the check leaves "latest" where it was.
    region_name : str, optional
        AWS region containing the target ECR registry. If None (the default), the region is taken from the
        session's AWS configuration (profile / AWS_REGION), so it is not hard-coded.
    ignore_docker_config : bool, optional
        If True, creates a temporary Docker config to prevent using stored credentials. Defaults to False.
    max_retries : int, optional
        Maximum number of retry attempts for failed push operations. Defaults to 1.
    boto_session : boto3.Session, optional
        Boto3 session used for ECR operations (already role-assumed if needed). If None, a default session is created
        (so callers that don't need role assumption, e.g. libera_cdk integration tests, can omit it).

    Raises
    ------
    ImageTagAlreadyExistsError
        If a requested tag other than "latest" already exists in the repository, in which case nothing is pushed, or
        if ECR refuses a tag taken by another push since that check, in which case the tags not yet pushed are skipped.
    ValueError
        If processing_step_id cannot be mapped to an ECR repository name,
        or if push operations encounter errors after all retries
    docker.errors.APIError
        If Docker API operations fail
    botocore.exceptions.ClientError
        If AWS ECR operations fail, including the check for existing tags (which needs ``ecr:DescribeImages``)

    Returns
    -------
    dict[str, str | None]
        Mapping of each pushed ECR tag to the image digest (``sha256:...``) reported for it, or None if the
        digest was not available in the push stream.
    """
    # Input validation and defaults
    image_name, image_tag = _split_local_image_reference(image_name, image_tag)

    if not ecr_image_tags:
        ecr_image_tags = [_LATEST_TAG]
    # Push version tags before "latest": a version tag taken by another push after the check below is refused by
    # ECR, and "latest" must not move to an image whose version was refused.
    ecr_image_tags = sorted(ecr_image_tags, key=lambda tag: tag == _LATEST_TAG)

    if isinstance(processing_step_id, str):
        processing_step_id = ProcessingStepIdentifier(processing_step_id)

    # Default to a plain session when no (role-assumed) session is provided, so callers that don't need role
    # assumption can omit it.
    if boto_session is None:
        boto_session = boto3.Session()

    # Region follows the session's AWS configuration unless explicitly overridden, so it is not hard-coded.
    if region_name is None:
        region_name = _session_region(boto_session)

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
        console_url = (
            f"https://{region_name}.console.aws.amazon.com/ecr/repositories/private/{account_id}/{ecr_name}"
            f"?region={region_name}"
        )
        logger.info(f"Target ECR registry: {ecr_registry}/{ecr_name}")

        # Verify local image exists before attempting pushes
        docker_client = docker.from_env()
        try:
            local_image = docker_client.images.get(f"{image_name}:{image_tag}")
        except docker.errors.ImageNotFound:
            raise ValueError(f"Local image not found: {image_name}:{image_tag}")

        existing_tags = _existing_version_tags(boto_session, region_name, ecr_name, ecr_image_tags)
        if existing_tags:
            raise ImageTagAlreadyExistsError(
                f"{_tag_in_use_message(existing_tags, f'{ecr_registry}/{ecr_name}', console_url)} No tags were pushed."
            )

        pushed_digests: dict[str, str | None] = {}

        for remote_tag in ecr_image_tags:
            full_ecr_tag = f"{ecr_registry}/{ecr_name}:{remote_tag}"

            try:
                digest = _push_single_tag(
                    docker_client=docker_client,
                    local_image=local_image,
                    full_ecr_tag=full_ecr_tag,
                    console_url=console_url,
                    region_name=region_name,
                    max_retries=max_retries,
                    boto_session=boto_session,
                )
                pushed_digests[remote_tag] = digest
                logger.info(f"Successfully pushed tag: {remote_tag}")

            except Exception as e:
                logger.exception(f"Failed to push tag {remote_tag}: {e}", stack_info=True)
                # Clean up any successful pushes on failure (optional)
                if pushed_digests:
                    logger.warning(f"Partial success: pushed tags {list(pushed_digests)} before failure")
                raise

        logger.info(
            f"All {len(ecr_image_tags)} tags pushed successfully to ECR. Remote tags pushed: {list(pushed_digests)}"
        )
        return pushed_digests
