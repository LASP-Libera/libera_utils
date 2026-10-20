"""The L1A processing step: combine one day of an APID's L0 packet files into its L1A product, from a manifest.

The SDC runs this as the algorithm of its ``l1a-*`` processing steps (one per APID; an AWS Batch job from the
``libera-utils-l1a`` image, ``libera-utils l1a-step <manifest>``). The step's Manifest Creator lists every L0 file of
the APID that reaches the applicable day or its buffers, and records in the manifest configuration:

- ``start_time``: the applicable day (``Manifest.add_desired_time_range``);
- ``l0_spans``: each file's index span (data time where recorded, otherwise packet time), filename -> ISO start and
  end, merged over the file's File Metadata rows;
- ``force``: True when the day's L0 was reported ready only because its deadline passed.

Unless ``force`` is set, the step first applies the APID's coverage gate (``evaluate_day_coverage`` with the APID's
``day_coverage`` policy) to ``l0_spans`` and fails when it does not pass: the orchestrator judged the same files
complete, so a failure means the two disagree. Then it parses the files (``ground_data=True`` when they are ground-test
captures, named ``LIBERA_SDC_<apid>_ccsds_...``, not by the L0 PDS convention; the Manifest Creator never mixes
ground-test captures and flight files), trims the dataset to the buffered day window, checks its data times, writes
the L1A product and writes the output manifest.
"""

import argparse
import logging
import os
from datetime import UTC, date, datetime

from cloudpathlib import AnyPath

from libera_utils.constants import LiberaApid, ProcessingStepIdentifier
from libera_utils.io.filenaming import LiberaGroundCcsdsFilename, PathType
from libera_utils.io.manifest import Manifest
from libera_utils.io.netcdf import write_libera_data_product
from libera_utils.l1a.day_coverage import evaluate_day_coverage
from libera_utils.l1a.day_window import check_data_times, trim_l1a_to_day_window
from libera_utils.l1a.l1a_packet_configs import get_l1a_product_definition_path, get_packet_config
from libera_utils.l1a.packets import parse_packets_to_l1a_dataset
from libera_utils.logutil import configure_task_logging

logger = logging.getLogger(__name__)

# Optional: files the step parses concurrently (``parse_packets_to_l1a_dataset``'s ``max_workers``; default 1).
MAX_WORKERS_ENV = "LIBERA_L1A_MAX_WORKERS"


def step_apid(step_name: str) -> LiberaApid:
    """Return the APID an ``l1a-*`` processing step decodes.

    Parameters
    ----------
    step_name : str
        The processing step name, e.g. ``"l1a-nom-hk"``.

    Returns
    -------
    LiberaApid
        The APID of the step's L1A product.

    Raises
    ------
    ValueError
        If the step is unknown or its product is not an L1A product of an APID.
    """
    apid = ProcessingStepIdentifier(step_name).products[0].associated_apid
    if apid is None:
        raise ValueError(f"Processing step {step_name} does not produce an L1A product of an APID")
    return apid


def _is_ground_capture(path: str) -> bool:
    """Return True for a ground-test capture name, False for a flight L0 file."""
    try:
        LiberaGroundCcsdsFilename(path)
    except ValueError:
        return False
    return True


def run_l1a_step(
    input_manifest: str | PathType, step_name: str, output_dir: str | PathType, *, max_workers: int = 1
) -> Manifest:
    """Combine the manifest's L0 files into the applicable day's L1A product and write the output manifest.

    Parameters
    ----------
    input_manifest : str | PathType
        The input manifest (see the module docstring for its configuration).
    step_name : str
        The processing step name, which names the APID (``step_apid``).
    output_dir : str | PathType
        Where the product and the output manifest are written.
    max_workers : int
        Files parsed concurrently.

    Returns
    -------
    Manifest
        The output manifest, listing the one L1A product.

    Raises
    ------
    ValueError
        If the manifest lists no files, fails the coverage gate without ``force``, or the written product's
        applicable date is not the day; also if a listed file's checksum does not match the manifest
        (``Manifest.validate_checksums``).
    """
    manifest = Manifest.from_file(input_manifest)
    manifest.validate_checksums()
    apid = step_apid(step_name)
    config = get_packet_config(apid)
    day = datetime.strptime(manifest.configuration["start_time"], "%Y-%m-%d:%H:%M:%S").date()
    files = [str(entry.filename) for entry in manifest.files]
    if not files:
        raise ValueError("Input manifest lists no L0 files")
    ground_data = all(_is_ground_capture(path) for path in files)
    force = bool(manifest.configuration.get("force", False))
    logger.info(
        "L1A step %s: APID %d, day %s, %d file(s), ground_data=%s, force=%s",
        step_name,
        apid,
        day,
        len(files),
        ground_data,
        force,
    )

    if not force:
        _check_coverage(manifest.configuration.get("l0_spans") or {}, apid, day)

    dataset = parse_packets_to_l1a_dataset(files, apid, ground_data=ground_data, max_workers=max_workers)
    dataset = trim_l1a_to_day_window(dataset, day=day)
    check_data_times(dataset)
    written = write_libera_data_product(
        get_l1a_product_definition_path(apid),
        dataset,
        output_dir,
        time_variable=config.packet_time_coordinate,
    )
    if written.applicable_date != day:
        raise ValueError(f"Wrote {written.path.name}, whose applicable date {written.applicable_date} is not {day}")
    logger.info("Wrote %s", written.path)

    output = Manifest.output_manifest_from_input_manifest(manifest)
    output.add_files(written.path)
    output.write(output_dir)
    return output


def _check_coverage(l0_spans: dict[str, list[str]], apid: LiberaApid, day: date) -> None:
    """Raise unless the files' spans pass the APID's coverage gate for the day."""
    intervals = [(datetime.fromisoformat(start), datetime.fromisoformat(end)) for start, end in l0_spans.values()]
    result = evaluate_day_coverage(intervals, day=day, policy=get_packet_config(apid).day_coverage)
    if not result.is_complete:
        raise ValueError(
            f"APID {int(apid)} L0 for {day} fails its coverage gate without force: left buffer "
            f"{result.left_frac:.3f}, day {result.day_frac:.3f}, right buffer {result.right_frac:.3f}"
        )


def l1a_step_cli_handler(parsed_args: argparse.Namespace) -> Manifest:
    """Run the L1A step named by ``PROCESSING_STEP_NAME`` on an input manifest, writing to ``PROCESSING_PATH``.

    Parameters
    ----------
    parsed_args : argparse.Namespace
        Parsed CLI arguments. Uses ``parsed_args.input_manifest`` and ``parsed_args.verbose``.

    Returns
    -------
    Manifest
        The output manifest.

    Raises
    ------
    KeyError
        If ``PROCESSING_STEP_NAME`` or ``PROCESSING_PATH`` is not set (the SDC's Batch job definition sets both).
    """
    configure_task_logging(
        f"l1a_step_{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}",
        limit_debug_loggers=["libera_utils"],
        console_log_level=logging.DEBUG if parsed_args.verbose else logging.INFO,
    )
    return run_l1a_step(
        AnyPath(parsed_args.input_manifest),
        os.environ["PROCESSING_STEP_NAME"],
        os.environ["PROCESSING_PATH"],
        max_workers=int(os.environ.get(MAX_WORKERS_ENV, "1")),
    )
