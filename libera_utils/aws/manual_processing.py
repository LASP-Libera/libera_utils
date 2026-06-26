"""Module for manual processing in the SDC

This module provides functionality to manually run processing jobs in the SDC by submitting custom processing graph
configurations. This supports testing of individual processing steps via single-node DAGs (the step-function-trigger
CLI), as well as fully custom DAGs defined by the user (the manual-processing CLI). The step-function-trigger CLI is
intended to be used for quick testing of individual processing steps, while the manual-processing CLI is intended to be
used for more complex processing runs that may involve multiple steps and custom input parameters. Manual processing
runs without explicit custom DAGs are also supported via the manual-processing CLI for purposes such as reprocessing.

Both CLIs work by emitting a single ``ManualProcessing`` event to the SDC central EventBridge bus (the
``LiberaSDCEventBus``). The SDC Job Creator picks up the event and builds the processing job(s) from it, then the rest
of the SDC's event-driven orchestration runs the node(s) end-to-end. Input products are assumed to already be ingested
into the SDC (File Metadata + Data Availability rows exist) before the event is sent.
"""

import argparse
import json
import logging
import time
from datetime import UTC, date, datetime

import boto3
from cloudpathlib import AnyPath
from ulid import ULID

from libera_utils.aws.utils import (
    SDC_EVENT_BUS_PARTIAL_NAME,
    find_dynamodb_table_in_account_by_partial_name,
    find_event_bus_in_account_by_partial_name,
    get_l2_team_role_session,
)
from libera_utils.constants import DataProductIdentifier, ProcessingStepIdentifier
from libera_utils.logutil import configure_task_logging

logger = logging.getLogger(__name__)

# These values are part of the ManualProcessing event contract and must match exactly what the SDC Job Creator's
# EventBridge rule expects. If they don't match, the event is not routed and nothing happens.
MANUAL_PROCESSING_EVENT_SOURCE = "manual-processing"
MANUAL_PROCESSING_EVENT_DETAIL_TYPE = "ManualProcessingEventDetail"

# Partial name used to uniquely identify the SDC Coordination Table by regex search (see find_*_by_partial_name).
COORDINATION_TABLE_PARTIAL_NAME = "CoordinationTable"
# Sort key of the per-job metadata record written by the Job Creator into the Coordination Table.
JOB_METADATA_SORT_KEY = "#JOBMETADATA"

# Region used for console URLs and AWS resources when a session does not specify one. The SDC runs in us-west-2.
DEFAULT_AWS_REGION = "us-west-2"
# How long to wait between Coordination Table polls when verifying job creation.
VERIFY_POLL_INTERVAL_SECONDS = 5

# Kebab-case keys allowed on a custom DAG node. The SDC model rejects snake_case, so we do too (early validation).
_REQUIRED_NODE_KEYS = frozenset({"description", "output-products", "input-products", "upstream-nodes"})
_OPTIONAL_NODE_KEYS = frozenset({"algorithm-version"})
_ALLOWED_NODE_KEYS = _REQUIRED_NODE_KEYS | _OPTIONAL_NODE_KEYS


def _to_date(value: str | date | datetime) -> date:
    """Normalize a date-like value (ISO string, datetime, or date) to a ``datetime.date``."""
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if isinstance(value, datetime):
        value = value.date()
    return value


def _validate_product_id(product_id: str, node_id: str, field: str) -> None:
    """Validate that a product id is a known DataProductIdentifier, raising a clear error if not."""
    try:
        DataProductIdentifier(product_id)
    except ValueError as err:
        raise ValueError(f"Invalid data product id '{product_id}' in node '{node_id}' {field}.") from err


def _validate_dag_config(dag: dict) -> None:
    """Do basic validation of a custom DAG configuration before sending it to the SDC.

    This is a best-effort early check to catch obvious errors (invalid identifiers, snake_case keys, dangling upstream
    references) before the event is emitted. The authoritative validation (including acyclicity) happens in the SDC Job
    Creator via the ``ProcessingSystemDagConfig`` Pydantic model, which lives in the libera_cdk repo.

    Parameters
    ----------
    dag : dict
        The custom DAG configuration, shaped like ``{"nodes": {<node-id>: <node>}}`` with kebab-case node keys.

    Raises
    ------
    ValueError
        If the structure, node ids, product ids, node keys, or upstream references are invalid.
    """
    if not isinstance(dag, dict) or "nodes" not in dag:
        raise ValueError("Custom DAG config must be a mapping with a top-level 'nodes' key.")
    nodes = dag["nodes"]
    if not isinstance(nodes, dict) or not nodes:
        raise ValueError("Custom DAG config 'nodes' must be a non-empty mapping of node-id to node definition.")

    node_ids = set(nodes)
    for node_id, node in nodes.items():
        try:
            ProcessingStepIdentifier(node_id)
        except ValueError as err:
            raise ValueError(f"Invalid processing step id '{node_id}' in custom DAG nodes.") from err
        if not isinstance(node, dict):
            raise ValueError(f"Node '{node_id}' definition must be a mapping.")

        unknown_keys = set(node) - _ALLOWED_NODE_KEYS
        if unknown_keys:
            raise ValueError(
                f"Node '{node_id}' has unexpected key(s) {sorted(unknown_keys)}. Custom DAG node keys must be "
                f"kebab-case; allowed keys are {sorted(_ALLOWED_NODE_KEYS)}."
            )
        missing_keys = _REQUIRED_NODE_KEYS - set(node)
        if missing_keys:
            raise ValueError(f"Node '{node_id}' is missing required key(s) {sorted(missing_keys)}.")

        for product in node["output-products"]:
            _validate_product_id(product, node_id, "output-products")
        for input_product in node["input-products"]:
            if not isinstance(input_product, dict) or "id" not in input_product:
                raise ValueError(f"Each entry in node '{node_id}' input-products must be a mapping with an 'id' key.")
            _validate_product_id(input_product["id"], node_id, "input-products")
        for upstream in node["upstream-nodes"]:
            if upstream not in node_ids:
                raise ValueError(
                    f"Node '{node_id}' lists upstream node '{upstream}' which is not a key in the DAG nodes."
                )


def get_state_machine_console_url(step: ProcessingStepIdentifier, account_id: str, region: str) -> str:
    """Build the AWS Console URL for a processing step's Step Function state machine.

    We can no longer link directly to a specific execution (the SDC orchestrator starts and names the execution, so the
    CLI never sees its ARN), but the state machine ARN is deterministic from the processing step. The returned URL
    points at the state machine's view page, where the triggered execution appears once the job starts.

    Parameters
    ----------
    step : ProcessingStepIdentifier
        The processing step whose state machine to link to.
    account_id : str
        The AWS account ID hosting the state machine.
    region : str
        The AWS region hosting the state machine.

    Returns
    -------
    str
        The AWS Console URL for the state machine.
    """
    state_machine_arn = f"arn:aws:states:{region}:{account_id}:stateMachine:{step.step_function_name}"
    return (
        f"https://{region}.console.aws.amazon.com/states/home?region={region}#/statemachines/view/{state_machine_arn}"
    )


def _log_monitoring_urls(
    start_processing_step_ids: list[ProcessingStepIdentifier] | None, *, boto_session: boto3.Session
) -> None:
    """Log AWS Console URL(s) the user can use to monitor the triggered processing.

    When the start nodes are known, one state machine URL is logged per start node. Otherwise (default-DAG root nodes
    or a many-node custom DAG) a generic Step Functions console link is logged.
    """
    account_id = boto_session.client("sts").get_caller_identity()["Account"]
    region = boto_session.region_name or DEFAULT_AWS_REGION

    if start_processing_step_ids:
        for step in start_processing_step_ids:
            logger.info(
                f"View the '{step}' step function (executions appear once the job starts): "
                f"{get_state_machine_console_url(step, account_id, region)}"
            )
    else:
        logger.info(
            "View the Step Functions console to monitor executions: "
            f"https://{region}.console.aws.amazon.com/states/home?region={region}#/statemachines"
        )


def _verify_jobs_created(job_ids: list[ULID], *, boto_session: boto3.Session, wait_time: float) -> None:
    """Poll the SDC Coordination Table to confirm a job metadata record exists for each job id.

    This is a basic check that the event was well-formed enough for the Job Creator to create the job(s). It does not
    guarantee the processing will succeed, since the job can still fail after creation. Jobs that do not appear within
    ``wait_time`` are logged as warnings rather than raising, since job creation is asynchronous.

    Parameters
    ----------
    job_ids : list[ULID]
        The job ids to look for (PK of the metadata record).
    boto_session : boto3.Session
        Boto3 session used to discover and read the Coordination Table.
    wait_time : float
        Maximum number of seconds to wait for all jobs to appear.
    """
    table_name = find_dynamodb_table_in_account_by_partial_name(boto_session, COORDINATION_TABLE_PARTIAL_NAME)
    table = boto_session.resource("dynamodb").Table(table_name)

    pending = {str(job_id) for job_id in job_ids}
    deadline = time.monotonic() + wait_time
    while pending and time.monotonic() < deadline:
        for job_id in list(pending):
            response = table.get_item(Key={"PK": job_id, "SK": JOB_METADATA_SORT_KEY})
            if "Item" in response:
                logger.info(f"Verified job {job_id} was created in the Coordination Table ({table_name}).")
                pending.discard(job_id)
        if pending:
            time.sleep(VERIFY_POLL_INTERVAL_SECONDS)

    for job_id in pending:
        logger.warning(
            f"Job {job_id} did not appear in the Coordination Table ({table_name}) within {wait_time} seconds. "
            f"The event may still be processing; check the AWS console for job status."
        )


def start_manual_processing(
    applicable_dates: list[str | date | datetime],
    *,
    boto_session: boto3.Session,
    custom_dag_config: dict | None = None,
    start_processing_step_ids: list[str | ProcessingStepIdentifier] | None = None,
    process_downstream: bool = True,
    job_ids: list[ULID] | None = None,
    verify: bool = False,
    wait_time: float = 60,
) -> list[ULID]:
    """Start a manual processing job by emitting a single ManualProcessing event to the SDC event bus.

    One job graph is created per applicable date. The SDC Job Creator builds the job(s) from the event and the rest of
    the SDC orchestration runs them. Input products are assumed to already be ingested into the SDC.

    Parameters
    ----------
    applicable_dates : list of str, date, or datetime
        The applicable dates to process data for. A separate job is created for each date.
    boto_session : boto3.Session
        Boto3 session used for all AWS interactions. Created once by the CLI handler (with the LiberaUtils role
        assumed) and passed in so the same authenticated session is used throughout.
    custom_dag_config : dict, optional
        A custom DAG configuration (``{"nodes": {...}}`` with kebab-case node keys). When None (default) the SDC's
        static default DAG is used.
    start_processing_step_ids : list of str or ProcessingStepIdentifier, optional
        Entry nodes for the job graph. When None (default) the Job Creator starts from the DAG's root nodes.
    process_downstream : bool, optional
        When True (default) the job graph includes all downstream descendants of the start nodes. When False, only the
        start nodes themselves are included -- the simple way to run a single node from the default DAG.
    job_ids : list[ULID], optional
        Caller-supplied job ids, paired positionally with ``applicable_dates`` (lengths must match). When None and
        ``verify`` is True, fresh ulids are minted so the created jobs can be polled for. When None and ``verify`` is
        False, the Job Creator mints its own ids and none are sent.
    verify : bool, optional
        When True, poll the Coordination Table to confirm each job was created. This requires job ids to be known
        ahead of time, so they are minted when not supplied. Defaults to False. Note that the step function console
        URL(s) are logged regardless of this flag.
    wait_time : float, optional
        Maximum seconds to wait when verifying job creation, by default 60.

    Returns
    -------
    list[ULID]
        The job ids associated with the submitted jobs (empty if none were supplied or minted).
    """
    normalized_dates = [_to_date(d) for d in applicable_dates]
    normalized_steps = (
        [ProcessingStepIdentifier(step) for step in start_processing_step_ids]
        if start_processing_step_ids is not None
        else None
    )

    if custom_dag_config is not None:
        _validate_dag_config(custom_dag_config)

    # Resolve job ids. To verify, we must know the ids ahead of time to poll the Coordination Table, so we mint one per
    # applicable date (paired positionally) when the caller has not supplied them.
    if job_ids is not None:
        job_ids = [job_id if isinstance(job_id, ULID) else ULID.from_str(str(job_id)) for job_id in job_ids]
        if len(job_ids) != len(normalized_dates):
            raise ValueError(
                f"job_ids length ({len(job_ids)}) must equal applicable_dates length ({len(normalized_dates)})."
            )
    elif verify:
        job_ids = [ULID.from_datetime(datetime.now(UTC)) for _ in normalized_dates]

    detail = {
        "applicable_dates": [d.isoformat() for d in normalized_dates],
        "start_processing_step_ids": [str(step) for step in normalized_steps] if normalized_steps is not None else None,
        "process_downstream": process_downstream,
        "dag": custom_dag_config,
        "job_ids": [str(job_id) for job_id in job_ids] if job_ids is not None else None,
    }
    logger.debug(f"ManualProcessing event detail: {detail}")

    event_bus_name = find_event_bus_in_account_by_partial_name(boto_session, SDC_EVENT_BUS_PARTIAL_NAME)
    events_client = boto_session.client("events")
    response = events_client.put_events(
        Entries=[
            {
                "Source": MANUAL_PROCESSING_EVENT_SOURCE,
                "DetailType": MANUAL_PROCESSING_EVENT_DETAIL_TYPE,
                "Detail": json.dumps(detail),
                "EventBusName": event_bus_name,
            }
        ]
    )
    if response.get("FailedEntryCount", 0) > 0:
        raise RuntimeError(
            f"Failed to put ManualProcessing event to event bus {event_bus_name}. Response entries: "
            f"{response['Entries']}"
        )

    logger.info(
        f"Put ManualProcessing event to event bus {event_bus_name} for applicable date(s) "
        f"{[d.isoformat() for d in normalized_dates]}."
    )

    _log_monitoring_urls(normalized_steps, boto_session=boto_session)

    if verify:
        _verify_jobs_created(job_ids, boto_session=boto_session, wait_time=wait_time)

    return job_ids or []


def step_function_trigger(
    algorithm_name: str | ProcessingStepIdentifier,
    applicable_day: str | date | datetime,
    *,
    boto_session: boto3.Session,
    verify: bool = False,
    wait_time: float = 60,
) -> list[ULID]:
    """Manually trigger a single processing step for a single applicable date.

    This is a thin special case of :func:`start_manual_processing`: it submits a ManualProcessing event using the SDC's
    default DAG with ``process_downstream=False`` and the single step as the start node, which the Job Creator reduces
    to a one-node job. Input/output products for the node come from the SDC's static DAG asset, so no custom DAG is
    needed. Input products are assumed to already be ingested into the SDC.

    Parameters
    ----------
    algorithm_name : str or ProcessingStepIdentifier
        The processing step to run.
    applicable_day : str, date, or datetime
        The day of data to process.
    boto_session : boto3.Session
        Boto3 session used for all AWS interactions.
    verify : bool, optional
        When True, poll the Coordination Table to confirm the job was created. Defaults to False. The step function
        console URL is logged regardless of this flag.
    wait_time : float, optional
        Maximum seconds to wait when verifying job creation, by default 60.

    Returns
    -------
    list[ULID]
        The job ids associated with the submitted job.
    """
    step = (
        algorithm_name
        if isinstance(algorithm_name, ProcessingStepIdentifier)
        else ProcessingStepIdentifier(algorithm_name)
    )
    return start_manual_processing(
        [_to_date(applicable_day)],
        boto_session=boto_session,
        start_processing_step_ids=[step],
        process_downstream=False,
        verify=verify,
        wait_time=wait_time,
    )


def step_function_trigger_cli_handler(parsed_args: argparse.Namespace) -> None:
    """CLI handler function for the ``step-function-trigger`` subcommand.

    Parameters
    ----------
    parsed_args : argparse.Namespace
        The parsed object of CLI arguments.
    """
    now = datetime.now(UTC)
    configure_task_logging(
        f"step_function_trigger_{now}", limit_debug_loggers="libera_utils", console_log_level=logging.DEBUG
    )
    logger.debug(f"CLI args: {parsed_args}")

    boto_session = get_l2_team_role_session(profile_name=parsed_args.profile)
    step_function_trigger(
        parsed_args.algorithm_name,
        parsed_args.applicable_day,
        boto_session=boto_session,
        verify=parsed_args.verify,
        wait_time=parsed_args.wait_time,
    )


def manual_processing_cli_handler(parsed_args: argparse.Namespace) -> None:
    """CLI handler function for the ``manual-processing`` subcommand.

    Reads an optional custom DAG configuration from a JSON file and submits a ManualProcessing event for the given
    applicable date(s).

    Parameters
    ----------
    parsed_args : argparse.Namespace
        The parsed object of CLI arguments.
    """
    now = datetime.now(UTC)
    configure_task_logging(
        f"manual_processing_{now}", limit_debug_loggers="libera_utils", console_log_level=logging.DEBUG
    )
    logger.debug(f"CLI args: {parsed_args}")

    custom_dag_config = None
    if parsed_args.dag_config:
        custom_dag_config = json.loads(AnyPath(parsed_args.dag_config).read_text())

    start_processing_step_ids = (
        [ProcessingStepIdentifier(step) for step in parsed_args.start_steps] if parsed_args.start_steps else None
    )

    boto_session = get_l2_team_role_session(profile_name=parsed_args.profile)
    start_manual_processing(
        parsed_args.applicable_dates,
        boto_session=boto_session,
        custom_dag_config=custom_dag_config,
        start_processing_step_ids=start_processing_step_ids,
        process_downstream=parsed_args.process_downstream,
        verify=parsed_args.verify,
        wait_time=parsed_args.wait_time,
    )
