"""Tests for the l1a_combine module (operator-triggered L1A day assembly via the SDC event bus)."""

import argparse
import json
from datetime import date, datetime
from unittest.mock import patch

import pytest

from libera_utils.aws import l1a_combine
from libera_utils.constants import LiberaApid


def _detail_from_capture(captured: dict) -> tuple[dict, dict]:
    """Pull the single emitted entry and its parsed Detail payload out of a capture dict."""
    entries = captured["entries"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["Source"] == "manual-l1a-preprocessing"
    assert entry["DetailType"] == "ManualL1APreprocessingEventDetail"
    return entry, json.loads(entry["Detail"])


def _cli_args(**overrides) -> argparse.Namespace:
    """Namespace matching what the force-l1a-combine subparser produces."""
    defaults = {
        "apid": 1057,
        "applicable_dates": [],
        "start": None,
        "end": None,
        "force": True,
        "ground_data": False,
        "reason": None,
        "profile": "test-profile",
    }
    return argparse.Namespace(**{**defaults, **overrides})


class TestDatesInRange:
    def test_inclusive_of_both_endpoints(self):
        assert l1a_combine.dates_in_range("2026-07-01", "2026-07-03") == [
            date(2026, 7, 1),
            date(2026, 7, 2),
            date(2026, 7, 3),
        ]

    def test_single_day_range(self):
        assert l1a_combine.dates_in_range("2026-07-01", "2026-07-01") == [date(2026, 7, 1)]

    def test_inverted_range_rejected(self):
        with pytest.raises(ValueError, match="precedes start date"):
            l1a_combine.dates_in_range("2026-07-03", "2026-07-01")


class TestForceL1ACombine:
    def test_emits_force_event(self, make_sdc_event_bus, make_event_capturing_session):
        session, captured = make_event_capturing_session()

        dates = l1a_combine.force_l1a_combine(1057, ["2026-07-12", "2026-07-13"], boto_session=session)

        entry, detail = _detail_from_capture(captured)
        assert entry["EventBusName"] == make_sdc_event_bus
        assert detail["apid"] == 1057
        assert detail["applicable_dates"] == ["2026-07-12", "2026-07-13"]
        assert detail["force"] is True
        assert detail["ground_data"] is False
        assert detail["reason"] is None
        assert dates == [date(2026, 7, 12), date(2026, 7, 13)]

    def test_ground_data_and_reason_carried(self, make_sdc_event_bus, make_event_capturing_session):
        session, captured = make_event_capturing_session()
        l1a_combine.force_l1a_combine(
            LiberaApid.icie_rad_full,
            [datetime.fromisoformat("2026-07-12T04:00:00")],
            boto_session=session,
            ground_data=True,
            reason="LIBSDC-1 DITL2 backfill",
        )
        _, detail = _detail_from_capture(captured)
        assert detail["apid"] == 1035
        assert detail["applicable_dates"] == ["2026-07-12"]
        assert detail["ground_data"] is True
        assert detail["reason"] == "LIBSDC-1 DITL2 backfill"

    def test_no_force_requests_a_gated_run(self, make_sdc_event_bus, make_event_capturing_session):
        session, captured = make_event_capturing_session()
        l1a_combine.force_l1a_combine(1057, ["2026-07-12"], boto_session=session, force=False)
        _, detail = _detail_from_capture(captured)
        assert detail["force"] is False

    def test_unknown_apid_rejected(self, make_sdc_event_bus, make_event_capturing_session):
        session, captured = make_event_capturing_session()
        with pytest.raises(ValueError, match="4321 is not a valid LiberaApid"):
            l1a_combine.force_l1a_combine(4321, ["2026-07-12"], boto_session=session)
        assert "entries" not in captured

    def test_no_dates_rejected(self, make_sdc_event_bus, make_event_capturing_session):
        session, captured = make_event_capturing_session()
        with pytest.raises(ValueError, match="At least one applicable date"):
            l1a_combine.force_l1a_combine(1057, [], boto_session=session)
        assert "entries" not in captured


class TestCliHandler:
    @pytest.fixture(autouse=True)
    def _session(self, make_sdc_event_bus, make_event_capturing_session):
        session, captured = make_event_capturing_session()
        self.captured = captured
        with patch.object(l1a_combine, "get_l2_team_role_session", return_value=session):
            yield

    def test_positional_dates(self):
        l1a_combine.force_l1a_combine_cli_handler(_cli_args(applicable_dates=["2026-07-12"]))
        _, detail = _detail_from_capture(self.captured)
        assert detail["applicable_dates"] == ["2026-07-12"]

    def test_start_end_expands_to_every_day(self):
        l1a_combine.force_l1a_combine_cli_handler(_cli_args(start="2026-07-01", end="2026-07-03"))
        _, detail = _detail_from_capture(self.captured)
        assert detail["applicable_dates"] == ["2026-07-01", "2026-07-02", "2026-07-03"]

    def test_both_date_forms_rejected(self):
        with pytest.raises(ValueError, match="not both"):
            l1a_combine.force_l1a_combine_cli_handler(
                _cli_args(applicable_dates=["2026-07-12"], start="2026-07-01", end="2026-07-03")
            )
        assert "entries" not in self.captured

    def test_half_a_range_rejected(self):
        with pytest.raises(ValueError, match="must be given together"):
            l1a_combine.force_l1a_combine_cli_handler(_cli_args(start="2026-07-01"))
        assert "entries" not in self.captured

    def test_no_dates_rejected(self):
        with pytest.raises(ValueError, match="No applicable dates"):
            l1a_combine.force_l1a_combine_cli_handler(_cli_args())
        assert "entries" not in self.captured

    def test_large_span_requires_confirmation(self):
        with patch("builtins.input", return_value="n") as prompt:
            l1a_combine.force_l1a_combine_cli_handler(_cli_args(start="2026-07-01", end="2026-07-31"))
        prompt.assert_called_once()
        assert "entries" not in self.captured

    def test_large_span_proceeds_once_confirmed(self):
        with patch("builtins.input", return_value="yes"):
            l1a_combine.force_l1a_combine_cli_handler(_cli_args(start="2026-07-01", end="2026-07-31"))
        _, detail = _detail_from_capture(self.captured)
        assert len(detail["applicable_dates"]) == 31

    def test_small_span_is_not_prompted(self):
        with patch("builtins.input", side_effect=AssertionError("should not prompt")):
            l1a_combine.force_l1a_combine_cli_handler(_cli_args(start="2026-07-01", end="2026-07-03"))
        _, detail = _detail_from_capture(self.captured)
        assert len(detail["applicable_dates"]) == 3
