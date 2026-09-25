"""Operator-triggered L1A day assembly.

The L1A Preprocessor normally combines a day on its own, once the L0 files covering that day
meet the APID's coverage policy (see :mod:`libera_utils.l1a.day_coverage`). Some days never
reach that bar: a campaign that only ran for part of a day, a downlink that was lost, a ground
capture set that was always going to be partial. This module emits the event that combines such
a day anyway.

It emits a single ``ManualL1APreprocessing`` event to the SDC central EventBridge bus (the
``LiberaSDCEventBus``); the L1A Preprocessor consumes it and runs one combine per applicable
date. The L0 files are assumed to already be ingested into the SDC (File Metadata rows exist)
before the event is sent, exactly as for the automatic path.
"""

import argparse
import json
import logging
from datetime import UTC, date, datetime, timedelta

import boto3

from libera_utils.aws.utils import (
    SDC_EVENT_BUS_PARTIAL_NAME,
    find_event_bus_in_account_by_partial_name,
    get_l2_team_role_session,
    to_date,
)
from libera_utils.constants import LiberaApid
from libera_utils.logutil import configure_task_logging

logger = logging.getLogger(__name__)

# These values are part of the ManualL1APreprocessing event contract and must match exactly what the L1A Preprocessor's
# EventBridge rule expects. If they don't match, the event is not routed and nothing happens.
MANUAL_L1A_EVENT_SOURCE = "manual-l1a-preprocessing"
MANUAL_L1A_EVENT_DETAIL_TYPE = "ManualL1APreprocessingEventDetail"

# Number of applicable dates above which the CLI asks the user to confirm before submitting.
MAX_UNCONFIRMED_APPLICABLE_DATES = 3


def dates_in_range(start: str | date | datetime, end: str | date | datetime) -> list[date]:
    """Return every UTC calendar date from ``start`` through ``end``, inclusive.

    Parameters
    ----------
    start, end : str, date, or datetime
        Range endpoints. ``end`` must not precede ``start``.

    Returns
    -------
    list[date]
        One entry per day in the closed interval.

    Raises
    ------
    ValueError
        If ``end`` precedes ``start``.
    """
    first, last = to_date(start), to_date(end)
    if last < first:
        raise ValueError(f"End date {last.isoformat()} precedes start date {first.isoformat()}.")
    return [first + timedelta(days=offset) for offset in range((last - first).days + 1)]


def force_l1a_combine(
    apid: int | LiberaApid,
    applicable_dates: list[str | date | datetime],
    *,
    boto_session: boto3.Session,
    force: bool = True,
    ground_data: bool = False,
    reason: str | None = None,
) -> list[date]:
    """Request L1A day assembly for one APID and one or more applicable dates.

    Parameters
    ----------
    apid : int or LiberaApid
        APID to assemble into day products.
    applicable_dates : list of str, date, or datetime
        UTC calendar days to process. One combine per date.
    boto_session : boto3.Session
        Boto3 session used for all AWS interactions. Created once by the CLI handler (with the
        LiberaUtils role assumed) and passed in so the same authenticated session is used
        throughout.
    force : bool, optional
        When True (default), the Preprocessor skips the coverage gates and combines whatever
        overlapping L0 is already indexed. When False, the gates are re-evaluated -- the way to
        retry a day without waiting for new L0 to arrive.
    ground_data : bool, optional
        When True, assemble in ground CCSDS mode. Default False (flight PDS).
    reason : str, optional
        Free text carried on the event for ops and audit logging.

    Returns
    -------
    list[date]
        The normalized applicable dates the event was emitted for.

    Raises
    ------
    ValueError
        If ``apid`` is not a ``LiberaApid`` member, or no applicable dates were given.
    RuntimeError
        If EventBridge rejects the event.
    """
    libera_apid = LiberaApid(apid)
    normalized_dates = [to_date(d) for d in applicable_dates]
    if not normalized_dates:
        raise ValueError("At least one applicable date is required.")

    detail = {
        "apid": int(libera_apid),
        "applicable_dates": [d.isoformat() for d in normalized_dates],
        "force": force,
        "ground_data": ground_data,
        "reason": reason,
    }
    logger.debug(f"ManualL1APreprocessing event detail: {detail}")

    event_bus_name = find_event_bus_in_account_by_partial_name(boto_session, SDC_EVENT_BUS_PARTIAL_NAME)
    events_client = boto_session.client("events")
    response = events_client.put_events(
        Entries=[
            {
                "Source": MANUAL_L1A_EVENT_SOURCE,
                "DetailType": MANUAL_L1A_EVENT_DETAIL_TYPE,
                "Detail": json.dumps(detail),
                "EventBusName": event_bus_name,
            }
        ]
    )
    if response.get("FailedEntryCount", 0) > 0:
        raise RuntimeError(
            f"Failed to put ManualL1APreprocessing event to event bus {event_bus_name}. Response entries: "
            f"{response['Entries']}"
        )

    logger.info(
        f"Put ManualL1APreprocessing event to event bus {event_bus_name} for APID {int(libera_apid)} "
        f"({libera_apid.name}), applicable date(s) {[d.isoformat() for d in normalized_dates]}, "
        f"force={force}, ground_data={ground_data}."
    )
    return normalized_dates


def force_l1a_combine_cli_handler(parsed_args: argparse.Namespace) -> None:
    """CLI handler function for the ``force-l1a-combine`` subcommand.

    Resolves the applicable dates from either the positional list or ``--start``/``--end``,
    confirms large submissions interactively, and emits a single event.

    Parameters
    ----------
    parsed_args : argparse.Namespace
        The parsed object of CLI arguments.

    Raises
    ------
    ValueError
        If neither or both date forms were supplied.
    """
    now = datetime.now(UTC)
    configure_task_logging(
        f"force_l1a_combine_{now}", limit_debug_loggers="libera_utils", console_log_level=logging.DEBUG
    )
    logger.debug(f"CLI args: {parsed_args}")

    ranged = parsed_args.start is not None or parsed_args.end is not None
    if ranged and parsed_args.applicable_dates:
        raise ValueError("Give applicable dates either positionally or as --start/--end, not both.")
    if ranged:
        if parsed_args.start is None or parsed_args.end is None:
            raise ValueError("--start and --end must be given together.")
        applicable_dates = dates_in_range(parsed_args.start, parsed_args.end)
    elif parsed_args.applicable_dates:
        applicable_dates = [to_date(d) for d in parsed_args.applicable_dates]
    else:
        raise ValueError("No applicable dates given. Pass them positionally or as --start/--end.")

    # Guard against accidentally combining a huge span (e.g. a year of data), which a date range makes easy to type.
    if len(applicable_dates) > MAX_UNCONFIRMED_APPLICABLE_DATES:
        confirmation = input(
            f"You are about to request L1A combining of APID {parsed_args.apid} for "
            f"{len(applicable_dates)} applicable dates ({applicable_dates[0].isoformat()} through "
            f"{applicable_dates[-1].isoformat()}), with force={parsed_args.force}. "
            "Type 'y' or 'yes' to continue: "
        )
        if confirmation.strip().lower() not in ("y", "yes"):
            logger.info("Aborted: user did not confirm L1A combining for %d applicable dates.", len(applicable_dates))
            return

    boto_session = get_l2_team_role_session(profile_name=parsed_args.profile)
    force_l1a_combine(
        parsed_args.apid,
        applicable_dates,
        boto_session=boto_session,
        force=parsed_args.force,
        ground_data=parsed_args.ground_data,
        reason=parsed_args.reason,
    )
