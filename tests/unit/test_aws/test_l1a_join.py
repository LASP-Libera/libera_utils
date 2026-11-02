"""Tests for the l1a_join module (operator-triggered L1A day assembly via the SDC event bus)."""

import argparse
import json
from datetime import date, datetime
from unittest.mock import patch

import boto3
import pytest

from libera_utils.aws import l1a_join
from libera_utils.constants import LiberaApid


def _detail_from_capture(captured: dict) -> tuple[dict, dict]:
    """Pull the single emitted entry and its parsed Detail payload out of a capture dict."""
    entries = captured["entries"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["Source"] == "l1a-processing-request"
    assert entry["DetailType"] == "L1AProcessingRequestEventDetail"
    return entry, json.loads(entry["Detail"])


def _cli_args(**overrides) -> argparse.Namespace:
    """Namespace matching what the l1a-processing subparser produces."""
    defaults = {
        "apid": 1057,
        "applicable_date": "2026-07-12",
        "force": False,
        "ground_data": False,
        "reason": None,
        "profile": "test-profile",
    }
    return argparse.Namespace(**{**defaults, **overrides})


class TestRequestL1AProcessing:
    def test_emits_gated_event_by_default(self, make_sdc_event_bus, make_event_capturing_session):
        session, captured = make_event_capturing_session()

        applicable_date = l1a_join.request_l1a_processing(1057, "2026-07-12", boto_session=session)

        entry, detail = _detail_from_capture(captured)
        assert entry["EventBusName"] == make_sdc_event_bus
        assert detail["apid"] == 1057
        assert detail["applicable_date"] == "2026-07-12"
        assert detail["force"] is False
        assert detail["ground_data"] is False
        assert detail["reason"] is None
        assert applicable_date == date(2026, 7, 12)

    def test_ground_data_and_reason_carried(self, make_sdc_event_bus, make_event_capturing_session):
        session, captured = make_event_capturing_session()
        l1a_join.request_l1a_processing(
            LiberaApid.icie_rad_full,
            datetime.fromisoformat("2026-07-12T04:00:00"),
            boto_session=session,
            ground_data=True,
            reason="LIBSDC-1 DITL2 backfill",
        )
        _, detail = _detail_from_capture(captured)
        assert detail["apid"] == 1035
        assert detail["applicable_date"] == "2026-07-12"
        assert detail["ground_data"] is True
        assert detail["reason"] == "LIBSDC-1 DITL2 backfill"

    def test_force_skips_the_check(self, make_sdc_event_bus, make_event_capturing_session):
        session, captured = make_event_capturing_session()
        l1a_join.request_l1a_processing(1057, "2026-07-12", boto_session=session, force=True)
        _, detail = _detail_from_capture(captured)
        assert detail["force"] is True

    def test_unknown_apid_rejected(self, make_sdc_event_bus, make_event_capturing_session):
        session, captured = make_event_capturing_session()
        with pytest.raises(ValueError, match="4321 is not a valid LiberaApid"):
            l1a_join.request_l1a_processing(4321, "2026-07-12", boto_session=session)
        assert "entries" not in captured

    def test_unconfigured_apid_rejected(self, make_sdc_event_bus, make_event_capturing_session):
        """icie_sw_stat (1013) is a LiberaApid with no packet configuration, so it cannot be assembled."""
        session, captured = make_event_capturing_session()
        with pytest.raises(ValueError, match=r"APID 1013 \(icie_sw_stat\) has no entry in l1a_processing_configs.yml"):
            l1a_join.request_l1a_processing(1013, "2026-07-12", boto_session=session)
        assert "entries" not in captured

    def test_malformed_date_rejected(self, make_sdc_event_bus, make_event_capturing_session):
        session, captured = make_event_capturing_session()
        with pytest.raises(ValueError, match="Invalid isoformat string"):
            l1a_join.request_l1a_processing(1057, "not-a-date", boto_session=session)
        assert "entries" not in captured

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

        with pytest.raises(RuntimeError, match="Failed to put L1AProcessingRequest event"):
            l1a_join.request_l1a_processing(1057, "2026-07-12", boto_session=session)


@patch("libera_utils.aws.l1a_join.request_l1a_processing")
@patch("libera_utils.aws.l1a_join.boto3.Session")
class TestCliHandler:
    def test_builds_session_and_forwards_every_flag(self, mock_session, mock_request):
        l1a_join.l1a_processing_cli_handler(
            _cli_args(
                apid=1035,
                applicable_date="2026-07-12",
                force=True,
                ground_data=True,
                reason="LIBSDC-1 DITL2 backfill",
                profile="ops-profile",
            )
        )
        mock_session.assert_called_once_with(profile_name="ops-profile")
        mock_request.assert_called_once_with(
            1035,
            "2026-07-12",
            boto_session=mock_session.return_value,
            force=True,
            ground_data=True,
            reason="LIBSDC-1 DITL2 backfill",
        )
