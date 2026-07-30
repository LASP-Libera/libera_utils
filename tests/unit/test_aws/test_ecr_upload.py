"""File for testing ECR upload module"""

import argparse
import base64
import json
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import docker
import pytest
from moto import mock_aws

import libera_utils.aws.ecr_upload as ecr_upload
from libera_utils.aws.ecr_upload import (
    DockerConfigManager,
    _get_fresh_ecr_auth,
    _process_push_logs,
    _push_single_tag,
    build_docker_image,
    push_image_to_ecr,
)
from libera_utils.constants import ProcessingStepIdentifier


def test_docker_config_manager():
    with DockerConfigManager() as dcfg:
        # When no override, this should just be None
        assert dcfg.dockercfg_path is None

    with DockerConfigManager(override_default_config=True) as dcfg:
        # When overriding, we make a minimal temp config file to pass to the DockerClient.login method
        assert dcfg.dockercfg_path is not None
        assert Path(dcfg.dockercfg_path).exists()
        config_file = Path(dcfg.dockercfg_path) / "config.json"
        assert config_file.is_file()
        with config_file.open("r") as f:
            s = f.read()
            d = json.loads(s)
            assert d == {"auths": {}, "HttpHeaders": {}}

    assert not config_file.exists()  # Ensure temp config gets cleaned up correctly


def test_build_docker_image(test_data_path):
    """Test building a docker image programmatically.
    This mocks the Docker client to avoid needing a real Docker daemon."""
    # Create a mock Docker client
    mock_docker_client = mock.MagicMock(spec=docker.DockerClient)

    # Mock the build logs
    mock_logs = [
        {"stream": "Step 1/3 : FROM python:3.9\n"},
        {"stream": "Step 2/3 : COPY . /app\n"},
        {"stream": 'Step 3/3 : CMD ["python", "app.py"]\n'},
        {"stream": "Successfully built abc123\n"},
    ]

    # Configure the mock client's images.build() method
    mock_docker_client.images.build.return_value = (mock.MagicMock(), mock_logs)

    # Mock docker.from_env to return our mock client
    with mock.patch("docker.from_env", return_value=mock_docker_client):
        build_docker_image(context_dir=test_data_path / "docker_test", image_name="test-image", target="test-target")

        # Verify build was called with expected arguments
        assert mock_docker_client.images.build.called
        call_args = mock_docker_client.images.build.call_args
        assert call_args[1]["target"] == "test-target"
        assert call_args[1]["tag"] == "test-image:latest"
        assert call_args[1]["platform"] == "linux/amd64"


@pytest.mark.parametrize(
    ("algorithm_name", "image_name", "image_tag", "ecr_tags", "ignore_docker_config", "profile"),
    [
        ("l1b-cam", "test-image", "latest", None, True, None),
        ("l1b-rad", "test-image", "latest", ["latest", "v1.0"], False, "test-profile"),
        ("l2-cf-cam", "test-image", "latest", ["latest"], False, "test-profile"),
    ],
)
@mock.patch("libera_utils.aws.ecr_upload.put_new_algorithm_image_event")
@mock.patch("libera_utils.aws.ecr_upload.push_image_to_ecr")
@mock.patch("libera_utils.aws.ecr_upload._resolve_algorithm_specific_session")
def test_ecr_upload_cli_handler(
    mock_resolve_session,
    mock_push_image_to_ecr,
    mock_put_event,
    image_name,
    algorithm_name,
    image_tag,
    ecr_tags,
    ignore_docker_config,
    profile,
):
    """The handler resolves the (possibly role-assumed) session and forwards it to push_image_to_ecr."""
    # The handler always registers after uploading, so push_image_to_ecr must return a {tag: digest} mapping.
    effective_tags = ecr_tags or ["latest"]
    mock_push_image_to_ecr.return_value = {tag: f"sha256:{tag}" for tag in effective_tags}

    # Make the input namespace object
    args = argparse.Namespace(
        func=ecr_upload.ecr_upload_cli_handler,
        algorithm_name=algorithm_name,
        image_name=image_name,
        image_tag=image_tag,
        ecr_tags=ecr_tags,
        ignore_docker_config=ignore_docker_config,
        profile=profile,
        verbose=False,
        verify=False,
        timeout=300.0,
    )

    ecr_upload.ecr_upload_cli_handler(args)

    expected_algorithm = ProcessingStepIdentifier(algorithm_name)
    mock_resolve_session.assert_called_once_with(expected_algorithm, profile)
    mock_push_image_to_ecr.assert_called_once_with(
        image_name,
        image_tag,
        expected_algorithm,
        ecr_image_tags=ecr_tags,
        ignore_docker_config=ignore_docker_config,
        boto_session=mock_resolve_session.return_value,
    )


@mock.patch("libera_utils.aws.ecr_upload.logger")
@mock.patch("libera_utils.aws.ecr_upload.push_image_to_ecr")
@mock.patch("libera_utils.aws.ecr_upload._resolve_algorithm_specific_session")
def test_ecr_upload_cli_handler_reraises_role_assumption_error(
    mock_resolve_session, mock_push_image_to_ecr, mock_logger
):
    """If the L2 Team Role cannot be assumed, the handler logs guidance and re-raises (does not push)."""
    mock_resolve_session.side_effect = ValueError("Could not assume role L2Developer/L2-CloudFraction ...")
    args = argparse.Namespace(
        func=ecr_upload.ecr_upload_cli_handler,
        algorithm_name="l2-cf-cam",
        image_name="test-image",
        image_tag="latest",
        ecr_tags=None,
        ignore_docker_config=False,
        profile=None,
        verbose=False,
    )

    with pytest.raises(ValueError, match="Could not assume role"):
        ecr_upload.ecr_upload_cli_handler(args)

    # The image is not pushed, and algorithm-specific guidance naming the required L2 Team Role is logged.
    mock_push_image_to_ecr.assert_not_called()
    mock_logger.error.assert_called_once()
    log_args = mock_logger.error.call_args.args
    assert "L2-CloudFraction" in log_args


@pytest.mark.parametrize(
    ("ecr_tags", "expected_versions"),
    [
        (["latest", "1.2.3"], ["1.2.3"]),  # single concrete version alongside latest
        (["1.2.3", "1.2.4"], ["1.2.3", "1.2.4"]),  # every non-latest tag gets its own event
        (None, []),  # defaults to ["latest"] -> nothing concrete to register
        (["latest"], []),  # only latest -> nothing concrete to register
    ],
)
@mock.patch("libera_utils.aws.ecr_upload.put_new_algorithm_image_event")
@mock.patch("libera_utils.aws.ecr_upload.push_image_to_ecr")
@mock.patch("libera_utils.aws.ecr_upload._resolve_algorithm_specific_session")
def test_ecr_upload_cli_handler_register(
    mock_resolve_session, mock_push_image_to_ecr, mock_put_event, ecr_tags, expected_versions
):
    """ecr-upload always registers: one NewAlgorithmImage event is emitted per non-'latest' ECR tag pushed."""
    # push_image_to_ecr returns a {tag: digest} mapping; the handler registers each non-'latest' tag and passes
    # its digest through to the event, reusing the session resolved for the push.
    effective_tags = ecr_tags or ["latest"]
    pushed_digests = {tag: f"sha256:{tag}" for tag in effective_tags}
    mock_push_image_to_ecr.return_value = pushed_digests

    args = argparse.Namespace(
        func=ecr_upload.ecr_upload_cli_handler,
        algorithm_name="l1b-rad",
        image_name="my-image",
        image_tag="latest",
        ecr_tags=ecr_tags,
        ignore_docker_config=False,
        profile="test",
        verbose=False,
        verify=True,
        timeout=42.0,
    )

    ecr_upload.ecr_upload_cli_handler(args)

    expected_algorithm = ProcessingStepIdentifier("l1b-rad")
    if not expected_versions:
        # No concrete version -> no event emitted.
        mock_put_event.assert_not_called()
        return

    # The registration event(s) reuse the session that was resolved for the ECR push (per-team role for L2
    # algorithms, or the ambient/--profile session for non-L2 SDC-owned algorithms).
    assert mock_put_event.call_count == len(expected_versions)
    for version, call in zip(expected_versions, mock_put_event.call_args_list, strict=True):
        assert call.args == (expected_algorithm, version)
        assert call.kwargs == {
            "boto_session": mock_resolve_session.return_value,
            "image_digest": pushed_digests[version],
            "verify": True,
            "timeout": 42.0,
        }


class TestGetFreshEcrAuth:
    """Test the _get_fresh_ecr_auth function."""

    def test_get_fresh_ecr_auth_success(self):
        """Test successful ECR authentication token retrieval using the provided session."""
        mock_session = MagicMock()
        mock_ecr_client = MagicMock()
        mock_session.client.return_value = mock_ecr_client

        # Mock the token response
        username = "AWS"
        password = "test-password-123"
        token = base64.b64encode(f"{username}:{password}".encode()).decode()
        mock_ecr_client.get_authorization_token.return_value = {
            "authorizationData": [{"authorizationToken": token, "expiresAt": "2024-01-01T00:00:00Z"}]
        }

        # Run the function
        result = _get_fresh_ecr_auth("us-west-2", boto_session=mock_session)

        assert result == {"username": username, "password": password}

        # Verify the ECR client was created from the provided session for the requested region.
        mock_session.client.assert_called_with("ecr", region_name="us-west-2")

    def test_get_fresh_ecr_auth_failure(self):
        """Test ECR authentication failure handling."""
        mock_session = MagicMock()
        mock_ecr_client = MagicMock()
        mock_session.client.return_value = mock_ecr_client

        # Set the side effect on the client method
        mock_ecr_client.get_authorization_token.side_effect = Exception("ECR error")

        with pytest.raises(Exception, match="ECR error"):
            _get_fresh_ecr_auth("us-west-2", boto_session=mock_session)


class TestPushSingleTag:
    """Test the _push_single_tag function."""

    @mock.patch("libera_utils.aws.ecr_upload._get_fresh_ecr_auth")
    def test_push_single_tag_success(self, mock_get_auth):
        """Test successful single tag push with session propagation."""
        # Mock authentication
        mock_get_auth.return_value = {"username": "AWS", "password": "test-password"}

        mock_docker_client = MagicMock()
        mock_local_image = MagicMock()
        mock_session = MagicMock()
        mock_docker_client.api.push.return_value = [{"status": "Pushed"}]

        _push_single_tag(
            docker_client=mock_docker_client,
            local_image=mock_local_image,
            full_ecr_tag="repo:latest",
            region_name="us-west-2",
            max_retries=3,
            boto_session=mock_session,
        )

        # Verify the session was propagated to the auth helper
        mock_get_auth.assert_called_once_with("us-west-2", boto_session=mock_session)

    @mock.patch("libera_utils.aws.ecr_upload._get_fresh_ecr_auth")
    def test_push_single_tag_with_errors_in_logs(self, mock_get_auth):
        """Test push with error messages in logs."""
        mock_get_auth.return_value = {"username": "AWS", "password": "test-password"}

        mock_docker_client = MagicMock()
        mock_local_image = MagicMock()

        # Mock push logs with errors
        mock_docker_client.api.push.return_value = [{"status": "Pushing"}, {"error": "Authentication failed"}]

        with pytest.raises(ValueError, match="Push errors"):
            _push_single_tag(
                docker_client=mock_docker_client,
                local_image=mock_local_image,
                full_ecr_tag="123456789.dkr.ecr.us-west-2.amazonaws.com/test-repo:latest",
                region_name="us-west-2",
                max_retries=1,
            )

    @mock.patch("libera_utils.aws.ecr_upload._get_fresh_ecr_auth")
    def test_push_single_tag_with_retries(self, mock_get_auth):
        """Test push with retry logic."""
        mock_get_auth.return_value = {"username": "AWS", "password": "test-password"}

        mock_docker_client = MagicMock()
        mock_local_image = MagicMock()

        # First call fails, second succeeds
        mock_docker_client.api.push.side_effect = [docker.errors.APIError("Network error"), [{"status": "Pushed"}]]

        _push_single_tag(
            docker_client=mock_docker_client,
            local_image=mock_local_image,
            full_ecr_tag="123456789.dkr.ecr.us-west-2.amazonaws.com/test-repo:latest",
            region_name="us-west-2",
            max_retries=2,
        )

        # Should be called twice (one failure, one success)
        assert mock_docker_client.api.push.call_count == 2
        assert mock_get_auth.call_count == 2  # Fresh auth for each attempt


class TestProcessPushLogs:
    """Test the _process_push_logs stream processor."""

    def test_returns_digest_and_suppresses_progress(self, caplog):
        """A realistic push stream returns the aux digest without emitting per-progress-tick log lines."""
        # A representative stream: an intro status, many high-frequency progress ticks, layer terminal states,
        # the terminal digest status line, and the aux event carrying the digest.
        push_logs = [
            {"status": "The push refers to repository [123.dkr.ecr.us-west-2.amazonaws.com/repo]"},
            {"status": "Preparing", "progressDetail": {}, "id": "layer1"},
            {"status": "Pushing", "progressDetail": {"current": 1, "total": 10}, "progress": "[==> ]", "id": "layer1"},
            {"status": "Pushing", "progressDetail": {"current": 9, "total": 10}, "progress": "[===>]", "id": "layer1"},
            {"status": "Pushed", "progressDetail": {}, "id": "layer1"},
            {"status": "Layer already exists", "progressDetail": {}, "id": "layer2"},
            {"status": "1.2.3: digest: sha256:deadbeef size: 1234"},
            {"aux": {"Tag": "1.2.3", "Digest": "sha256:deadbeef", "Size": 1234}},
        ]

        with caplog.at_level("DEBUG", logger="libera_utils.aws.ecr_upload"):
            digest = _process_push_logs(iter(push_logs), "repo:1.2.3")

        assert digest == "sha256:deadbeef"

        # No raw "Pushing" progress ticks are logged (the byte counters / progress bars are suppressed).
        assert not any("[==>" in record.message or "progress" in record.message.lower() for record in caplog.records)

        # A single INFO summary reports the layer counts and the digest.
        info_messages = [r.message for r in caplog.records if r.levelname == "INFO"]
        assert any(
            "1 layer(s) pushed" in m and "1 already existed" in m and "sha256:deadbeef" in m for m in info_messages
        )

    def test_returns_none_when_no_aux_digest(self):
        """When the stream carries no aux event, the digest is None (but processing still succeeds)."""
        push_logs = [{"status": "Pushed", "progressDetail": {}, "id": "layer1"}]
        assert _process_push_logs(iter(push_logs), "repo:latest") is None

    def test_raises_on_error_field(self):
        """An ``error`` field in the stream raises ValueError."""
        push_logs = [{"status": "Pushing"}, {"error": "Authentication failed"}]
        with pytest.raises(ValueError, match="Push errors"):
            _process_push_logs(iter(push_logs), "repo:latest")

    def test_raises_on_error_detail_field(self):
        """An ``errorDetail`` field (without a top-level ``error``) also raises ValueError."""
        push_logs = [{"errorDetail": {"message": "denied: not authorized"}}]
        with pytest.raises(ValueError, match="denied: not authorized"):
            _process_push_logs(iter(push_logs), "repo:latest")


@pytest.mark.parametrize("ecr_tags", [None, ["latest"], ["latest", "v1.0"]])
@mock_aws
@mock.patch("docker.from_env")
@mock.patch("libera_utils.aws.ecr_upload._push_single_tag")
def test_push_image_to_ecr(mock_push_single_tag, mock_docker_from_env, ecr_tags):
    """Test the push_image_to_ecr function."""
    # Mock Docker client and image
    mock_docker_client = MagicMock()
    mock_local_image = MagicMock()
    mock_docker_client.images.get.return_value = mock_local_image
    mock_docker_from_env.return_value = mock_docker_client

    # The account id is derived from the session (moto returns 123456789012). No boto_session passed exercises the
    # default-session fallback used by callers like libera_cdk.
    push_image_to_ecr(
        "test-image", "latest", ProcessingStepIdentifier.l1b_rad, ecr_image_tags=ecr_tags, ignore_docker_config=True
    )

    # Verify Docker client setup
    mock_docker_from_env.assert_called_once()
    mock_docker_client.images.get.assert_called_once_with("test-image:latest")

    # Verify push calls
    expected_calls = 1 if ecr_tags is None else len(ecr_tags)
    assert mock_push_single_tag.call_count == expected_calls

    # Verify the tags being pushed
    if ecr_tags is None:
        expected_tags = ["latest"]
    else:
        expected_tags = ecr_tags

    for i, expected_tag in enumerate(expected_tags):
        call_args = mock_push_single_tag.call_args_list[i]
        assert call_args[1]["full_ecr_tag"].endswith(f":{expected_tag}")


@mock_aws
@mock.patch("docker.from_env")
@mock.patch("libera_utils.aws.ecr_upload._push_single_tag")
def test_push_image_to_ecr_returns_digest_mapping(mock_push_single_tag, mock_docker_from_env):
    """push_image_to_ecr returns a {tag: digest} mapping built from each single-tag push."""
    mock_docker_client = MagicMock()
    mock_docker_client.images.get.return_value = MagicMock()
    mock_docker_from_env.return_value = mock_docker_client

    mock_push_single_tag.side_effect = ["sha256:aaa", "sha256:bbb"]

    result = push_image_to_ecr(
        "test-image", "latest", ProcessingStepIdentifier.l1b_rad, ecr_image_tags=["1.2.3", "latest"]
    )

    assert result == {"1.2.3": "sha256:aaa", "latest": "sha256:bbb"}


@mock_aws
@mock.patch("docker.from_env")
def test_push_image_to_ecr_image_not_found(mock_docker_from_env):
    """Test push_image_to_ecr when local image is not found."""
    mock_docker_client = MagicMock()
    mock_docker_client.images.get.side_effect = docker.errors.ImageNotFound("Image not found")
    mock_docker_from_env.return_value = mock_docker_client

    with pytest.raises(ValueError, match="Local image not found: test-image:latest"):
        push_image_to_ecr("test-image", "latest", ProcessingStepIdentifier.l1b_rad)


@mock_aws
def test_push_image_to_ecr_invalid_processing_step():
    """Test push_image_to_ecr with an invalid processing step that has no ECR name."""
    # Create a mock processing step that returns None for ecr_name
    with mock.patch.object(ProcessingStepIdentifier, "ecr_name", None):
        processing_step = ProcessingStepIdentifier.l1b_rad

        with pytest.raises(ValueError, match="Unable to determine ECR repository name"):
            push_image_to_ecr("test-image", "latest", processing_step)


@mock_aws
@mock.patch("docker.from_env")
@mock.patch("libera_utils.aws.ecr_upload._push_single_tag")
def test_push_image_to_ecr_partial_failure(mock_push_single_tag, mock_docker_from_env):
    """Test push_image_to_ecr when one tag succeeds and another fails."""
    mock_docker_client = MagicMock()
    mock_local_image = MagicMock()
    mock_docker_client.images.get.return_value = mock_local_image
    mock_docker_from_env.return_value = mock_docker_client

    # First call succeeds, second fails
    mock_push_single_tag.side_effect = [None, Exception("Push failed")]

    with pytest.raises(Exception, match="Push failed"):
        push_image_to_ecr("test-image", "latest", ProcessingStepIdentifier.l1b_rad, ecr_image_tags=["v1.0", "latest"])

    # Should have attempted both pushes
    assert mock_push_single_tag.call_count == 2
