"""Module for registering algorithm versions in the SDC.

Uploading an algorithm Docker image to the SDC ECR only makes the image available; it does not make it
runnable by version. The SDC processing step function resolves a requested algorithm version by searching
the AWS Batch job definitions for one whose container references the matching image. The SDC Registrar
service (a Batch Job Definition Registrar Lambda) creates that versioned job definition in response to a
``NewAlgorithmImage`` EventBridge event: it clones the step's default job definition and swaps in the
uploaded image.

This module emits that event (see :func:`put_new_algorithm_image_event`) and, optionally, verifies that the
Registrar produced the expected job definition (see :func:`verify_algorithm_registration`).
"""

import argparse
import json
import logging
import time
from datetime import UTC, datetime

import boto3
from botocore.exceptions import ClientError

from libera_utils.aws.utils import (
    SDC_EVENT_BUS_PARTIAL_NAME,
    _resolve_algorithm_specific_session,
    find_event_bus_in_account_by_partial_name,
)
from libera_utils.constants import ProcessingStepIdentifier
from libera_utils.logutil import configure_task_logging

logger = logging.getLogger(__name__)

# These values are part of the NewAlgorithmImage event contract and must match exactly what the SDC
# Registrar's EventBridge rule expects.
NEW_ALGORITHM_IMAGE_EVENT_SOURCE = "algorithm-image-publisher"
NEW_ALGORITHM_IMAGE_EVENT_DETAIL_TYPE = "NewAlgorithmImageEventDetail"

# Registration verification (--verify) polling parameters.
DEFAULT_REGISTRATION_TIMEOUT_SECONDS = 300.0  # 5 minutes
REGISTRATION_POLL_INTERVAL_SECONDS = 10.0


def put_new_algorithm_image_event(
    processing_step_id: str | ProcessingStepIdentifier,
    algorithm_version: str,
    *,
    boto_session: boto3.Session,
    image_digest: str | None = None,
    region_name: str = "us-west-2",
    verify: bool = False,
    timeout: float = DEFAULT_REGISTRATION_TIMEOUT_SECONDS,
) -> None:
    """Emit a single NewAlgorithmImage event to the SDC event bus.

    The event announces that a concrete algorithm image (one ECR tag) has been uploaded and that the SDC
    Registrar should register a corresponding versioned AWS Batch job definition for it.

    Parameters
    ----------
    processing_step_id : str or ProcessingStepIdentifier
        The processing step whose algorithm this image implements. Determines the ECR repository name.
    algorithm_version : str
        The concrete ECR image tag that was uploaded, e.g. ``"1.2.3"``.
    boto_session : boto3.Session
        Boto3 session used to discover the event bus, resolve the account id, and put the event.
    image_digest : str or None, optional
        Optional image digest (``sha256:...``) carried for provenance/logging. Not required; the registered
        job definition references the tag, not the digest.
    region_name : str, optional
        AWS region containing the target ECR registry. Default ``"us-west-2"``.
    verify : bool, optional
        If True, block after emitting the event until the corresponding job definition is confirmed
        registered (see :func:`verify_algorithm_registration`). Default False.
    timeout : float, optional
        Maximum number of seconds to wait for registration verification when ``verify`` is set.

    Raises
    ------
    ValueError
        If ``processing_step_id`` has no associated ECR repository (e.g. an L0 step).
    RuntimeError
        If EventBridge reports a failed entry when putting the event.
    """
    if isinstance(processing_step_id, str):
        processing_step_id = ProcessingStepIdentifier(processing_step_id)

    ecr_repository_name = processing_step_id.ecr_name
    if ecr_repository_name is None:
        raise ValueError(
            f"Unable to determine ECR repository name for processing step: {processing_step_id}. "
            f"Note: L0 processing steps do not have associated ECR repositories."
        )

    # The account id is resolved from the passed session so that whatever identity/role the caller assumed
    # is reflected in the ECR registry URI.
    account_id = boto_session.client("sts").get_caller_identity()["Account"]
    ecr_repository_uri = f"{account_id}.dkr.ecr.{region_name}.amazonaws.com/{ecr_repository_name}"

    # Built as a literal dict matching the SDC's NewAlgorithmImageEventDetail schema (field names must match).
    detail = {
        "processing_step_id": str(processing_step_id),
        "algorithm_version": algorithm_version,
        "ecr_repository_name": ecr_repository_name,
        "ecr_repository_uri": ecr_repository_uri,
        "image_digest": image_digest,
    }

    event_bus_name = find_event_bus_in_account_by_partial_name(boto_session, SDC_EVENT_BUS_PARTIAL_NAME)
    events_client = boto_session.client("events")

    response = events_client.put_events(
        Entries=[
            {
                "Source": NEW_ALGORITHM_IMAGE_EVENT_SOURCE,
                "DetailType": NEW_ALGORITHM_IMAGE_EVENT_DETAIL_TYPE,
                "Detail": json.dumps(detail),
                "EventBusName": event_bus_name,
            }
        ]
    )

    if response.get("FailedEntryCount", 0) > 0:
        raise RuntimeError(
            f"Failed to put NewAlgorithmImage event to event bus {event_bus_name}. Response entries: "
            f"{response['Entries']}"
        )

    logger.info(
        "Put NewAlgorithmImage event for %s version %s (%s) to event bus %s",
        processing_step_id,
        algorithm_version,
        ecr_repository_uri,
        event_bus_name,
    )

    if verify:
        logger.info(
            "Verifying registration of job definition for %s version %s (timeout %.0fs)...",
            processing_step_id,
            algorithm_version,
            timeout,
        )
        verify_algorithm_registration(
            ecr_repository_name,
            ecr_repository_uri,
            algorithm_version,
            boto_session=boto_session,
            timeout=timeout,
        )
        logger.info("Verified registration of job definition for %s version %s.", processing_step_id, algorithm_version)


def verify_algorithm_registration(
    ecr_repository_name: str,
    ecr_repository_uri: str,
    algorithm_version: str,
    *,
    boto_session: boto3.Session,
    timeout: float = DEFAULT_REGISTRATION_TIMEOUT_SECONDS,
    poll_interval: float = REGISTRATION_POLL_INTERVAL_SECONDS,
) -> None:
    """Verify the referenced ECR image exists and an ACTIVE Batch job definition references it.

    Two independent checks are performed:

    1. The referenced image (``{ecr_repository_name}:{algorithm_version}``) actually exists in ECR. This is
       checked first and unconditionally -- even if a matching Batch job definition already exists -- because
       registering a job definition for a nonexistent image would produce a job definition that can never run.
    2. Some ACTIVE Batch job definition has ``containerProperties.image`` equal to
       ``{ecr_repository_uri}:{algorithm_version}``. This is naming-agnostic and polled until it appears.

    Parameters
    ----------
    ecr_repository_name : str
        The short ECR repository name (e.g. ``l1b-rad-docker-repo``), used to look the image up in ECR.
    ecr_repository_uri : str
        The full ECR registry URI for the algorithm image (without tag).
    algorithm_version : str
        The concrete ECR image tag that should exist in ECR and be referenced by the registered job definition.
    boto_session : boto3.Session
        Boto3 session used for the (read-only) ECR and Batch describe calls.
    timeout : float, optional
        Maximum number of seconds to wait for the job definition to appear. Default 300 (5 minutes).
    poll_interval : float, optional
        Number of seconds between polling passes. Default 10.

    Raises
    ------
    ValueError
        If the referenced image does not exist in ECR.
    TimeoutError
        If no matching ACTIVE job definition is found before the timeout elapses.
    """
    # 1. The referenced image must actually exist in ECR. Checked unconditionally (even if a matching job
    #    definition already exists), because a job definition referencing a nonexistent image can never run.
    _verify_ecr_image_exists(boto_session, ecr_repository_name, algorithm_version)

    # 2. Poll for an ACTIVE Batch job definition that references the image.
    expected_image = f"{ecr_repository_uri}:{algorithm_version}"
    batch_client = boto_session.client("batch")

    deadline = time.monotonic() + timeout
    while True:
        if _active_job_definition_exists_for_image(batch_client, expected_image):
            logger.info("Found ACTIVE Batch job definition referencing image %s.", expected_image)
            return

        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"No ACTIVE Batch job definition referencing image {expected_image} was found within "
                f"{timeout:.0f} seconds. The SDC Registrar may not have processed the NewAlgorithmImage event yet."
            )

        logger.debug("No job definition for %s yet; retrying in %.0fs.", expected_image, poll_interval)
        time.sleep(poll_interval)


def _verify_ecr_image_exists(boto_session: boto3.Session, repository_name: str, tag: str) -> None:
    """Confirm that ``repository_name:tag`` exists in ECR, raising ValueError if it does not.

    Parameters
    ----------
    boto_session : boto3.Session
        Boto3 session used for the (read-only) ECR describe call.
    repository_name : str
        The short ECR repository name (e.g. ``l1b-rad-docker-repo``).
    tag : str
        The image tag that must be present in the repository.

    Raises
    ------
    ValueError
        If the repository or the tagged image does not exist in ECR.
    """
    ecr_client = boto_session.client("ecr")
    try:
        ecr_client.describe_images(repositoryName=repository_name, imageIds=[{"imageTag": tag}])
    except ClientError as err:
        error_code = err.response.get("Error", {}).get("Code")
        if error_code in ("ImageNotFoundException", "RepositoryNotFoundException"):
            raise ValueError(
                f"ECR image {repository_name}:{tag} was not found in ECR ({error_code}). Cannot register a Batch "
                f"job definition for an image that does not exist; ensure the image has been uploaded to ECR."
            ) from err
        raise
    logger.info("Confirmed ECR image %s:%s exists.", repository_name, tag)


def _active_job_definition_exists_for_image(batch_client, expected_image: str) -> bool:
    """Return whether any ACTIVE Batch job definition references ``expected_image`` as its container image."""
    paginator = batch_client.get_paginator("describe_job_definitions")
    for page in paginator.paginate(status="ACTIVE"):
        for job_definition in page.get("jobDefinitions", []):
            if job_definition.get("containerProperties", {}).get("image") == expected_image:
                return True
    return False


def register_algorithm_image_cli_handler(parsed_args: argparse.Namespace) -> None:
    """CLI handler function for the ``register-algorithm-image`` subcommand.

    Assumes the algorithm image has already been uploaded to ECR and emits a NewAlgorithmImage event so the
    SDC Registrar creates the corresponding versioned Batch job definition. With ``--verify``, blocks until
    that job definition is confirmed registered.
    """
    now = datetime.now(UTC)
    configure_task_logging(
        f"register_algorithm_image_{now}", limit_debug_loggers="libera_utils", console_log_level=logging.DEBUG
    )
    logger.debug(f"CLI args: {parsed_args}")

    processing_step_id = ProcessingStepIdentifier(parsed_args.algorithm_name)
    profile_name = parsed_args.profile

    # The boto session originates here and is passed to the workflow function. It assumes the algorithm's per-team
    # L2 Team Role for L2 algorithms, or uses the ambient/--profile session for non-L2 (SDC-owned) algorithms so
    # SDC developers can register with their own credentials. Keeping session creation in a single place lets tests
    # inject a custom session and call the workflow function directly.
    boto_session = _resolve_algorithm_specific_session(processing_step_id, profile_name)

    put_new_algorithm_image_event(
        processing_step_id,
        parsed_args.algorithm_version,
        boto_session=boto_session,
        image_digest=parsed_args.image_digest,
        verify=parsed_args.verify,
        timeout=parsed_args.timeout,
    )
