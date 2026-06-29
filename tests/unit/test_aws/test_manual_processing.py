"""Tests for the manual_processing module (manual processing via the SDC event bus)."""

import argparse
import json
from datetime import datetime
from unittest.mock import patch

import boto3
import pytest
from ulid import ULID

from libera_utils.aws import manual_processing as mp
from libera_utils.constants import ProcessingStepIdentifier


def _detail_from_capture(captured: dict) -> tuple[dict, dict]:
    """Pull the single emitted entry and its parsed Detail payload out of a capturing-session capture dict."""
    entries = captured["entries"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["Source"] == "manual-processing"
    assert entry["DetailType"] == "ManualProcessingEventDetail"
    return entry, json.loads(entry["Detail"])


def _ulid(year=2026, month=6, day=1) -> ULID:
    """Build a deterministic ULID from a fixed datetime for tests."""
    return ULID.from_datetime(datetime(year, month, day))


class TestStepFunctionTrigger:
    """Tests for the single-node ``step-function-trigger`` path."""

    def test_emits_single_node_default_dag_event(self, make_sdc_event_bus, make_event_capturing_session):
        """step_function_trigger emits a ManualProcessing event for one node using the default DAG, no downstream."""
        session, captured = make_event_capturing_session()

        job_ids = mp.step_function_trigger("l1b-rad", "2030-01-01", boto_session=session, verify=False)

        entry, detail = _detail_from_capture(captured)
        assert entry["EventBusName"] == make_sdc_event_bus
        assert detail["applicable_dates"] == ["2030-01-01"]
        assert detail["start_processing_step_ids"] == ["l1b-rad"]
        assert detail["process_downstream"] is False
        assert detail["dag"] is None
        # verify=False so no job ids are minted or sent.
        assert detail["job_ids"] is None
        assert job_ids == []

    def test_accepts_enum_and_datetime(self, make_sdc_event_bus, make_event_capturing_session):
        """The step and date can be passed as enum/datetime; the date is normalized to YYYY-MM-DD."""
        session, captured = make_event_capturing_session()
        mp.step_function_trigger(
            ProcessingStepIdentifier.l1b_cam,
            datetime.fromisoformat("2030-01-01T12:00:00"),
            boto_session=session,
            verify=False,
        )
        _, detail = _detail_from_capture(captured)
        assert detail["start_processing_step_ids"] == ["l1b-cam"]
        assert detail["applicable_dates"] == ["2030-01-01"]


class TestStartManualProcessing:
    """Tests for the general ``manual-processing`` helper."""

    def test_emits_custom_dag_event(self, make_sdc_event_bus, make_event_capturing_session):
        """A custom DAG round-trips into the event Detail with process_downstream defaulting to True."""
        session, captured = make_event_capturing_session()
        dag = {
            "nodes": {
                "l1b-rad": {
                    "description": "L1B radiometer processing (manual test)",
                    "algorithm-version": None,
                    "output-products": ["RAD-4CH"],
                    "input-products": [{"id": "RAD-SAMPLE-DECODED", "version": None}],
                    "upstream-nodes": [],
                }
            }
        }

        mp.start_manual_processing(
            ["2026-06-01", "2026-06-02"], boto_session=session, custom_dag_config=dag, verify=False
        )

        _, detail = _detail_from_capture(captured)
        assert detail["applicable_dates"] == ["2026-06-01", "2026-06-02"]
        assert detail["start_processing_step_ids"] is None
        assert detail["process_downstream"] is True
        assert detail["dag"] == dag

    def test_explicit_job_ids_are_sent(self, make_sdc_event_bus, make_event_capturing_session):
        """Caller-supplied job ids are serialized into the event Detail."""
        session, captured = make_event_capturing_session()
        job_ids = [_ulid()]

        returned = mp.start_manual_processing(["2026-06-01"], boto_session=session, job_ids=job_ids, verify=False)

        _, detail = _detail_from_capture(captured)
        assert detail["job_ids"] == [str(job_ids[0])]
        assert returned == job_ids

    def test_job_ids_length_mismatch_raises(self, make_sdc_event_bus, make_event_capturing_session):
        """A job_ids list whose length does not match applicable_dates is rejected before emitting."""
        session, captured = make_event_capturing_session()
        with pytest.raises(ValueError, match="must equal applicable_dates length"):
            mp.start_manual_processing(
                ["2026-06-01", "2026-06-02"],
                boto_session=session,
                job_ids=[_ulid()],
                verify=False,
            )
        assert "entries" not in captured

    def test_verify_succeeds_when_job_present(
        self, make_sdc_event_bus, make_coordination_table, make_event_capturing_session
    ):
        """With verify=True and explicit job ids seeded in the Coordination Table, the helper completes cleanly."""
        _, seed = make_coordination_table
        session, captured = make_event_capturing_session()
        job_ids = [_ulid()]
        seed(job_ids[0])

        returned = mp.start_manual_processing(
            ["2026-06-01"], boto_session=session, job_ids=job_ids, verify=True, wait_time=5
        )

        _, detail = _detail_from_capture(captured)
        assert detail["job_ids"] == [str(job_ids[0])]
        assert returned == job_ids

    def test_verify_mints_job_ids(self, make_sdc_event_bus, make_coordination_table, make_event_capturing_session):
        """When verifying without supplied job ids, one is minted per date and sent in the event."""
        session, captured = make_event_capturing_session()
        # wait_time=0 so the (unseeded) verification poll returns immediately with a warning, not a hang/raise.
        returned = mp.start_manual_processing(
            ["2026-06-01", "2026-06-02"], boto_session=session, verify=True, wait_time=0
        )

        _, detail = _detail_from_capture(captured)
        assert len(detail["job_ids"]) == 2
        assert [str(j) for j in returned] == detail["job_ids"]

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

        with pytest.raises(RuntimeError, match="Failed to put ManualProcessing event"):
            mp.start_manual_processing(["2026-06-01"], boto_session=session, verify=False)


class TestVerifyJobsCreated:
    """Tests for the Coordination Table verification helper directly."""

    def test_found(self, make_coordination_table):
        session = boto3.Session(profile_name="test-profile")
        _, seed = make_coordination_table
        job_id = _ulid()
        seed(job_id)
        # Should not raise.
        mp._verify_jobs_created([job_id], boto_session=session, wait_time=5)

    def test_timeout_warns_does_not_raise(self, make_coordination_table, caplog):
        session = boto3.Session(profile_name="test-profile")
        job_id = _ulid()
        # Not seeded; wait_time=0 makes the poll loop exit immediately.
        mp._verify_jobs_created([job_id], boto_session=session, wait_time=0)
        assert any("did not appear" in record.message for record in caplog.records)


class TestValidateDagConfig:
    """Tests for the basic custom DAG validation."""

    def _valid_node(self):
        return {
            "description": "test",
            "algorithm-version": None,
            "output-products": ["RAD-4CH"],
            "input-products": [{"id": "RAD-SAMPLE-DECODED", "version": None}],
            "upstream-nodes": [],
        }

    def test_valid(self):
        mp._validate_dag_config({"nodes": {"l1b-rad": self._valid_node()}})

    @pytest.mark.parametrize(
        ("mutator", "match"),
        [
            (lambda d: d.pop("nodes"), "top-level 'nodes' key"),
            (lambda d: d["nodes"].clear(), "non-empty mapping"),
        ],
    )
    def test_structure_errors(self, mutator, match):
        dag = {"nodes": {"l1b-rad": self._valid_node()}}
        mutator(dag)
        with pytest.raises(ValueError, match=match):
            mp._validate_dag_config(dag)

    def test_invalid_node_id(self):
        with pytest.raises(ValueError, match="Invalid processing step id"):
            mp._validate_dag_config({"nodes": {"not-a-step": self._valid_node()}})

    def test_snake_case_key_rejected(self):
        node = self._valid_node()
        node["output_products"] = node.pop("output-products")
        with pytest.raises(ValueError, match="unexpected key"):
            mp._validate_dag_config({"nodes": {"l1b-rad": node}})

    def test_invalid_product_id(self):
        node = self._valid_node()
        node["output-products"] = ["NOT-A-PRODUCT"]
        with pytest.raises(ValueError, match="Invalid data product id"):
            mp._validate_dag_config({"nodes": {"l1b-rad": node}})

    def test_dangling_upstream(self):
        node = self._valid_node()
        node["upstream-nodes"] = ["l1b-cam"]
        with pytest.raises(ValueError, match="not a key in the DAG nodes"):
            mp._validate_dag_config({"nodes": {"l1b-rad": node}})


def test_get_state_machine_console_url():
    """The console URL embeds the deterministic state machine ARN and region."""
    url = mp.get_state_machine_console_url(ProcessingStepIdentifier.l1b_rad, "123456789012", "us-west-2")
    assert "arn:aws:states:us-west-2:123456789012:stateMachine:l1b-rad-processing-step-function" in url
    assert url.startswith("https://us-west-2.console.aws.amazon.com/states/home?region=us-west-2")


@patch("libera_utils.aws.manual_processing.step_function_trigger")
@patch("libera_utils.aws.manual_processing.get_l2_team_role_session")
def test_step_function_trigger_cli_handler(mock_get_session, mock_trigger):
    """The CLI handler builds a session from the profile and delegates to step_function_trigger."""
    args = argparse.Namespace(
        func=mp.step_function_trigger_cli_handler,
        algorithm_name="l1b-rad",
        applicable_day="2030-01-01",
        wait_time=60,
        verify=True,
        profile="test-profile",
    )

    mp.step_function_trigger_cli_handler(args)

    mock_get_session.assert_called_once_with(profile_name="test-profile")
    mock_trigger.assert_called_once_with(
        "l1b-rad",
        "2030-01-01",
        boto_session=mock_get_session.return_value,
        verify=True,
        wait_time=60,
    )


@patch("libera_utils.aws.manual_processing.start_manual_processing")
@patch("libera_utils.aws.manual_processing.get_l2_team_role_session")
def test_manual_processing_cli_handler(mock_get_session, mock_start, tmp_path):
    """The CLI handler reads the DAG file, parses start steps, and delegates to start_manual_processing."""
    dag = {
        "nodes": {
            "l1b-rad": {
                "description": "x",
                "output-products": ["RAD-4CH"],
                "input-products": [{"id": "RAD-SAMPLE-DECODED"}],
                "upstream-nodes": [],
            }
        }
    }
    dag_file = tmp_path / "dag.json"
    dag_file.write_text(json.dumps(dag))

    args = argparse.Namespace(
        func=mp.manual_processing_cli_handler,
        applicable_dates=["2026-06-01", "2026-06-02"],
        dag_config=str(dag_file),
        start_steps=["l1b-rad"],
        process_downstream=False,
        wait_time=60,
        verify=True,
        profile=None,
    )

    mp.manual_processing_cli_handler(args)

    mock_get_session.assert_called_once_with(profile_name=None)
    mock_start.assert_called_once_with(
        ["2026-06-01", "2026-06-02"],
        boto_session=mock_get_session.return_value,
        custom_dag_config=dag,
        start_processing_step_ids=[ProcessingStepIdentifier.l1b_rad],
        process_downstream=False,
        verify=True,
        wait_time=60,
    )
