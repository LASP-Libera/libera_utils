"""Operator-triggered L1A day assembly.

The L1A Preprocessor normally assembles a day on its own, once the L0 files covering that day
meet the APID's coverage policy (see :mod:`libera_utils.l1a.day_coverage`). This module sends
the event that asks the L1A Preprocessor to re-evaluate the completeness check for a day, or,
with ``force=True``, to assemble a day whose L0 files do not meet that policy, such as a partial
campaign or a ground capture.

It emits a single ``L1AProcessingRequest`` event to the SDC central EventBridge bus (the
``LiberaSDCEventBus``); the L1A Preprocessor consumes it and runs day assembly for the applicable
date. The L0 files are assumed to already be ingested into the SDC (File Metadata rows exist)
before the event is sent, exactly as for the automatic path.
"""

import argparse
import json
import logging
from datetime import UTC, date, datetime

import boto3

from libera_utils.aws.utils import (
    SDC_EVENT_BUS_PARTIAL_NAME,
    find_event_bus_in_account_by_partial_name,
)
from libera_utils.constants import LiberaApid
from libera_utils.l1a.l1a_packet_configs import configured_apids
from libera_utils.logutil import configure_task_logging
from libera_utils.time import to_date

logger = logging.getLogger(__name__)

# These values are part of the L1AProcessingRequest event contract and must match exactly what the L1A Preprocessor's
# EventBridge rule expects. If they don't match, the event is not routed and nothing happens.
L1A_PROCESSING_REQUEST_EVENT_SOURCE = "l1a-processing-request"
L1A_PROCESSING_REQUEST_EVENT_DETAIL_TYPE = "L1AProcessingRequestEventDetail"


def request_l1a_processing(
    apid: int | LiberaApid,
    applicable_date: str | date | datetime,
    *,
    boto_session: boto3.Session,
    force: bool = False,
    ground_data: bool = False,
    reason: str | None = None,
) -> date:
    """Request L1A day assembly for one APID and one applicable date.

    Parameters
    ----------
    apid : int or LiberaApid
        APID to assemble into day products.
    applicable_date : str, datetime.date, or datetime
        UTC calendar day to process.
    boto_session : boto3.Session
        Boto3 session used for all AWS interactions. The CLI handler builds it from the
        ``--profile`` profile, or the default profile, and assumes no role.
    force : bool, optional
        When False (default), the Preprocessor re-evaluates the completeness check -- the way to
        retry a day without waiting for new L0 to arrive. When True, it skips the check and
        assembles whatever overlapping L0 is already indexed.
    ground_data : bool, optional
        When True, assemble in ground CCSDS mode. Default False (flight PDS).
    reason : str, optional
        Free text carried on the event for ops and audit logging.

    Returns
    -------
    datetime.date
        The normalized applicable date the event was emitted for.

    Raises
    ------
    ValueError
        If ``apid`` is not a ``LiberaApid`` member or has no packet configuration, the date string
        is not ISO 8601, or the account does not hold exactly one event bus matching
        ``LiberaSDCEventBus``.
    RuntimeError
        If EventBridge rejects the event.
    """
    libera_apid = LiberaApid(apid)
    if libera_apid not in configured_apids():
        choices = ", ".join(f"{int(a)} ({a.name})" for a in configured_apids())
        raise ValueError(
            f"APID {int(libera_apid)} ({libera_apid.name}) has no entry in l1a_processing_configs.yml, so the L1A "
            f"Preprocessor cannot parse it. Choose one of: {choices}."
        )
    normalized_date = to_date(applicable_date)

    detail = {
        "apid": int(libera_apid),
        "applicable_date": normalized_date.isoformat(),
        "force": force,
        "ground_data": ground_data,
        "reason": reason,
    }
    logger.debug(f"L1AProcessingRequest event detail: {detail}")

    event_bus_name = find_event_bus_in_account_by_partial_name(boto_session, SDC_EVENT_BUS_PARTIAL_NAME)
    events_client = boto_session.client("events")
    response = events_client.put_events(
        Entries=[
            {
                "Source": L1A_PROCESSING_REQUEST_EVENT_SOURCE,
                "DetailType": L1A_PROCESSING_REQUEST_EVENT_DETAIL_TYPE,
                "Detail": json.dumps(detail),
                "EventBusName": event_bus_name,
            }
        ]
    )
    if response.get("FailedEntryCount", 0) > 0:
        raise RuntimeError(
            f"Failed to put L1AProcessingRequest event to event bus {event_bus_name}. Response entries: "
            f"{response['Entries']}"
        )

    logger.info(
        f"Put L1AProcessingRequest event to event bus {event_bus_name} for APID {int(libera_apid)} "
        f"({libera_apid.name}), applicable date {normalized_date.isoformat()}, force={force}, "
        f"ground_data={ground_data}."
    )
    return normalized_date


def l1a_processing_cli_handler(parsed_args: argparse.Namespace) -> None:
    """CLI handler function for the ``l1a-processing`` subcommand.

    Parameters
    ----------
    parsed_args : argparse.Namespace
        The parsed object of CLI arguments.

    Raises
    ------
    ValueError
        If :func:`request_l1a_processing` raises it.
    RuntimeError
        If EventBridge rejects the event.
    """
    now = datetime.now(UTC)
    configure_task_logging(f"l1a_processing_{now}", limit_debug_loggers="libera_utils", console_log_level=logging.DEBUG)
    logger.debug(f"CLI args: {parsed_args}")

    boto_session = boto3.Session(profile_name=parsed_args.profile)
    request_l1a_processing(
        parsed_args.apid,
        parsed_args.applicable_date,
        boto_session=boto_session,
        force=parsed_args.force,
        ground_data=parsed_args.ground_data,
        reason=parsed_args.reason,
    )
