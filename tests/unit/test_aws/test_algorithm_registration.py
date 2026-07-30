"""Tests for the algorithm_registration module (NewAlgorithmImage event emission and verification)."""

import argparse
import json
from unittest.mock import patch

import boto3
import pytest

from libera_utils.aws import algorithm_registration
from libera_utils.constants import ProcessingStepIdentifier

# Account id returned by moto's sts.get_caller_identity under the test credentials.
MOTO_ACCOUNT_ID = "123456789012"


def _expected_uri(processing_step_id: str, region_name: str = "us-west-2") -> str:
    ecr_name = ProcessingStepIdentifier(processing_step_id).ecr_name
    return f"{MOTO_ACCOUNT_ID}.dkr.ecr.{region_name}.amazonaws.com/{ecr_name}"


class TestPutNewAlgorithmImageEvent:
    """Tests for put_new_algorithm_image_event."""

    def test_emits_event_with_expected_source_detail_type_and_payload(
        self, make_sdc_event_bus, make_event_capturing_session
    ):
        """A single NewAlgorithmImage event is emitted with the contract source/detail-type and full detail."""
        session, captured = make_event_capturing_session()

        algorithm_registration.put_new_algorithm_image_event(
            "l1b-rad", "1.2.3", boto_session=session, image_digest="sha256:abc123"
        )

        entries = captured["entries"]
        assert len(entries) == 1
        entry = entries[0]
        assert entry["Source"] == "algorithm-image-publisher"
        assert entry["DetailType"] == "NewAlgorithmImageEventDetail"
        assert entry["EventBusName"] == make_sdc_event_bus

        detail = json.loads(entry["Detail"])
        assert detail == {
            "processing_step_id": "l1b-rad",
            "algorithm_version": "1.2.3",
            "ecr_repository_name": "l1b-rad-docker-repo",
            "ecr_repository_uri": _expected_uri("l1b-rad"),
            "image_digest": "sha256:abc123",
        }

    def test_image_digest_defaults_to_none(self, make_sdc_event_bus, make_event_capturing_session):
        """image_digest is optional and serialized as null when not provided."""
        session, captured = make_event_capturing_session()

        algorithm_registration.put_new_algorithm_image_event("l2-cf-cam", "4.5.6", boto_session=session)

        detail = json.loads(captured["entries"][0]["Detail"])
        assert detail["image_digest"] is None

    def test_accepts_processing_step_identifier_instance(self, make_sdc_event_bus, make_event_capturing_session):
        """A ProcessingStepIdentifier instance is accepted as well as a plain string."""
        session, captured = make_event_capturing_session()

        algorithm_registration.put_new_algorithm_image_event(
            ProcessingStepIdentifier("l1b-cam"), "2.0.0", boto_session=session
        )

        detail = json.loads(captured["entries"][0]["Detail"])
        assert detail["processing_step_id"] == "l1b-cam"

    def test_raises_when_step_has_no_ecr(self, make_sdc_event_bus):
        """A processing step without an ECR repository (ecr_name is None) raises ValueError."""
        session = boto3.Session(profile_name="test-profile")
        with (
            patch.object(ProcessingStepIdentifier, "ecr_name", None),
            pytest.raises(ValueError, match="do not have associated ECR repositories"),
        ):
            algorithm_registration.put_new_algorithm_image_event("l1b-rad", "1.2.3", boto_session=session)

    def test_raises_on_failed_entry(self, make_sdc_event_bus):
        """A failed event entry from put_events surfaces as a RuntimeError."""
        session = boto3.Session(profile_name="test-profile")
        real_client = session.client

        def failing_put_client(service_name, *args, **kwargs):
            client = real_client(service_name, *args, **kwargs)
            if service_name == "events":
                client.put_events = lambda **kwargs: {"FailedEntryCount": 1, "Entries": [{"ErrorCode": "Boom"}]}
            return client

        session.client = failing_put_client

        with pytest.raises(RuntimeError, match="Failed to put NewAlgorithmImage event"):
            algorithm_registration.put_new_algorithm_image_event("l1b-rad", "1.2.3", boto_session=session)

    @patch("libera_utils.aws.algorithm_registration.verify_algorithm_registration")
    def test_verify_true_triggers_verification(self, mock_verify, make_sdc_event_bus, make_event_capturing_session):
        """When verify=True, verification runs against the derived image URI after the event is emitted."""
        session, _ = make_event_capturing_session()

        algorithm_registration.put_new_algorithm_image_event(
            "l1b-rad", "1.2.3", boto_session=session, verify=True, timeout=99.0
        )

        mock_verify.assert_called_once_with(
            "l1b-rad-docker-repo", _expected_uri("l1b-rad"), "1.2.3", boto_session=session, timeout=99.0
        )


class TestVerifyAlgorithmRegistration:
    """Tests for verify_algorithm_registration against mocked AWS ECR and Batch backends."""

    ECR_REPO = "l1b-rad-docker-repo"

    @staticmethod
    def _register_job_definition(session: boto3.Session, name: str, image: str) -> None:
        batch_client = session.client("batch", region_name=session.region_name)
        batch_client.register_job_definition(
            jobDefinitionName=name,
            type="container",
            containerProperties={"image": image, "vcpus": 1, "memory": 128},
        )

    @staticmethod
    def _put_ecr_image(session: boto3.Session, repository_name: str, tag: str) -> None:
        """Create the ECR repository (if needed) and push a minimal tagged image manifest into it."""
        ecr_client = session.client("ecr", region_name=session.region_name)
        try:
            ecr_client.create_repository(repositoryName=repository_name)
        except ecr_client.exceptions.RepositoryAlreadyExistsException:
            pass
        manifest = {
            "schemaVersion": 2,
            "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
            "config": {"mediaType": "application/vnd.docker.container.image.v1+json", "size": 7, "digest": "sha256:c"},
            "layers": [],
        }
        ecr_client.put_image(repositoryName=repository_name, imageManifest=json.dumps(manifest), imageTag=tag)

    def test_passes_when_image_and_matching_job_definition_exist(self, mock_s3_context_with_profile):
        """Verification returns cleanly once the ECR image exists and an ACTIVE job definition references it."""
        session = boto3.Session(profile_name="test-profile")
        uri = _expected_uri("l1b-rad")
        self._put_ecr_image(session, self.ECR_REPO, "1.2.3")
        self._register_job_definition(session, "l1b-rad-1-2-3", f"{uri}:1.2.3")

        # Should not raise.
        algorithm_registration.verify_algorithm_registration(
            self.ECR_REPO, uri, "1.2.3", boto_session=session, timeout=5.0
        )

    def test_times_out_when_no_matching_job_definition(self, mock_s3_context_with_profile):
        """Verification raises TimeoutError when the image exists but no job definition references it."""
        session = boto3.Session(profile_name="test-profile")
        uri = _expected_uri("l1b-rad")
        self._put_ecr_image(session, self.ECR_REPO, "1.2.3")
        # A job definition exists, but for a different version.
        self._register_job_definition(session, "l1b-rad-9-9-9", f"{uri}:9.9.9")

        with pytest.raises(TimeoutError, match="No ACTIVE Batch job definition"):
            algorithm_registration.verify_algorithm_registration(
                self.ECR_REPO, uri, "1.2.3", boto_session=session, timeout=0.0, poll_interval=0.0
            )

    def test_raises_when_ecr_image_missing_even_if_job_definition_exists(self, mock_s3_context_with_profile):
        """The ECR image check runs unconditionally: a matching job definition does not excuse a missing image."""
        session = boto3.Session(profile_name="test-profile")
        uri = _expected_uri("l1b-rad")
        # The repository exists and a matching job definition exists, but the image tag was never pushed.
        self._put_ecr_image(session, self.ECR_REPO, "9.9.9")
        self._register_job_definition(session, "l1b-rad-1-2-3", f"{uri}:1.2.3")

        with pytest.raises(ValueError, match="was not found in ECR"):
            algorithm_registration.verify_algorithm_registration(
                self.ECR_REPO, uri, "1.2.3", boto_session=session, timeout=5.0
            )

    def test_raises_when_ecr_repository_missing(self, mock_s3_context_with_profile):
        """A missing ECR repository surfaces as the same 'not found in ECR' ValueError."""
        session = boto3.Session(profile_name="test-profile")
        uri = _expected_uri("l1b-rad")

        with pytest.raises(ValueError, match="was not found in ECR"):
            algorithm_registration.verify_algorithm_registration(
                self.ECR_REPO, uri, "1.2.3", boto_session=session, timeout=5.0
            )


class TestRegisterAlgorithmImageCliHandler:
    """Tests for the register-algorithm-image CLI handler."""

    @pytest.mark.parametrize("profile", [None, "deploy-profile"])
    @patch("libera_utils.aws.algorithm_registration.put_new_algorithm_image_event")
    @patch("libera_utils.aws.algorithm_registration._resolve_algorithm_specific_session")
    def test_handler_builds_session_and_delegates(self, mock_resolve_session, mock_put_event, profile):
        """The handler resolves the algorithm-specific session and delegates to the workflow function."""
        args = argparse.Namespace(
            func=algorithm_registration.register_algorithm_image_cli_handler,
            algorithm_name="l1b-rad",
            algorithm_version="1.2.3",
            image_digest="sha256:abc",
            verify=True,
            timeout=123.0,
            profile=profile,
        )

        algorithm_registration.register_algorithm_image_cli_handler(args)

        expected_algorithm = ProcessingStepIdentifier("l1b-rad")
        mock_resolve_session.assert_called_once_with(expected_algorithm, profile)
        mock_put_event.assert_called_once_with(
            expected_algorithm,
            "1.2.3",
            boto_session=mock_resolve_session.return_value,
            image_digest="sha256:abc",
            verify=True,
            timeout=123.0,
        )
