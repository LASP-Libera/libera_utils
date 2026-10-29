"""Golden input staging, the run report and the day-window invariants for the end-to-end tests.

The golden input lives under one root, an ``s3://`` URI or a local directory, laid out as the
keys of ``l1a_ditl2_golden_manifest.csv``. Keys and messages name objects relative to the root,
never the root itself, because the root names a bucket that must not appear in public CI logs.
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import logging
import subprocess
import threading
import warnings
from collections.abc import Mapping
from datetime import date, timedelta
from pathlib import Path, PurePosixPath
from typing import Any

import boto3
import numpy as np
import psutil
import xarray as xr
from cloudpathlib import AnyPath, S3Path

from libera_utils.config import config
from libera_utils.constants import LiberaApid
from libera_utils.io.smart_open import is_s3, smart_copy_file
from libera_utils.l1a.day_coverage import TimeAxisCoverage
from libera_utils.l1a.day_window import day_bounds
from libera_utils.l1a.ground_ccsds import apid_from_ground_ccsds_filename
from libera_utils.l1a.l1a_packet_configs import CoverageMode, DayCoveragePolicy
from libera_utils.l1a.packet_slicing import PACKET_DIM, data_time_axes, find_sample_dims, sample_to_packet_index
from libera_utils.l1a.wfov_image_metadata import CAMERA_TIME_COORD, PACKET_IMAGE_ID_VAR

GOLDEN_DAY = date(2026, 7, 12)
GOLDEN_MANIFEST_PATH = Path(__file__).parent / "l1a_ditl2_golden_manifest.csv"
GOLDEN_ROOT_ENV = "LIBERA_E2E_GOLDEN_ROOT"
OUTPUT_DIR_ENV = "LIBERA_E2E_OUTPUT_DIR"
# Subdirectories of the output directory; each uploaded object is tagged kind=<kind> so that
# lifecycle rules can expire each kind on its own schedule.
OUTPUT_KINDS = ("products", "logs", "reports")
# The first key component under the output root, so that runs from several repositories can share one root.
REPOSITORY = "libera_utils"
_MANIFEST_HEADER = ["key", "sha256"]


def defined_l1a_apids() -> list[LiberaApid]:
    """Return every APID with an L1A product definition, in ascending order."""
    return sorted(LiberaApid(int(apid)) for apid in config.get("LIBERA_L1A_PRODUCT_DEFINITIONS_BY_APID"))


def golden_apid(key: str) -> LiberaApid:
    """Return the APID of a golden object from the ground CCSDS name its key ends in.

    Raises
    ------
    ValueError
        If the key's last component is not a ground CCSDS name, or its APID is not a ``LiberaApid``.
    """
    return LiberaApid(apid_from_ground_ccsds_filename(PurePosixPath(key).name))


def golden_root_from_env(environ: Mapping[str, str]) -> str:
    """Return the golden root, an ``s3://`` URI or a local directory, from ``LIBERA_E2E_GOLDEN_ROOT``.

    Raises
    ------
    RuntimeError
        If ``AWS_ACCESS_KEY_ID`` holds the unit tests' mock credentials, which a session that also
        selected non-E2E tests installs for every test.
    ValueError
        If ``LIBERA_E2E_GOLDEN_ROOT`` is unset or empty.
    """
    if environ.get("AWS_ACCESS_KEY_ID") == "testing":
        raise RuntimeError(
            "AWS_ACCESS_KEY_ID holds the unit tests' mock credentials, so this session selected non-E2E tests too. "
            "Run the E2E tests on their own: pytest -m e2e tests/e2e"
        )
    root = environ.get(GOLDEN_ROOT_ENV)
    if not root:
        raise ValueError(
            f"{GOLDEN_ROOT_ENV} is not set. Set it to the golden input root, an s3:// URI or a local directory."
        )
    return root


def output_dir_from_env(environ: Mapping[str, str]) -> Path:
    """Return the local output directory from ``LIBERA_E2E_OUTPUT_DIR``.

    The directory must be absent or empty, so a run's outputs are only its own.

    Raises
    ------
    ValueError
        If ``LIBERA_E2E_OUTPUT_DIR`` is unset, empty or an ``s3://`` URI.
    FileExistsError
        If the directory already holds files.
    """
    output = environ.get(OUTPUT_DIR_ENV)
    if not output:
        raise ValueError(
            f"{OUTPUT_DIR_ENV} is not set. Set it to a local directory for the run's products, logs and report."
        )
    if is_s3(output):
        raise ValueError(
            f"{OUTPUT_DIR_ENV} is an S3 URI. Set it to a local directory; CI uploads the outputs after the run."
        )
    output_dir = Path(output)
    if output_dir.is_dir() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"{OUTPUT_DIR_ENV} names {output_dir}, which already holds a run's outputs. Set it to a new or empty "
            f"directory for each run."
        )
    return output_dir


def run_commit(environ: Mapping[str, str], repo_dir: Path) -> str | None:
    """Return the commit the run tests: ``GITHUB_SHA`` in CI, else the ``HEAD`` of ``repo_dir``.

    A local commit carries a ``-dirty`` suffix when the working tree has uncommitted changes,
    since the run then tests code no commit holds.

    Returns
    -------
    str or None
        The commit SHA, or None outside CI when ``repo_dir`` is not in a git working tree or git
        is not installed.
    """
    if environ.get("GITHUB_SHA"):
        return environ["GITHUB_SHA"]

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=repo_dir, check=True, capture_output=True, text=True).stdout

    try:
        sha = git("rev-parse", "HEAD").strip()
        dirty = bool(git("status", "--porcelain").strip())
    except (OSError, subprocess.CalledProcessError):
        return None
    return f"{sha}-dirty" if dirty else sha


def load_golden_manifest(path: Path) -> dict[str, str]:
    """Read a golden manifest into a mapping of object key to SHA-256 hex digest.

    Parameters
    ----------
    path : Path
        CSV file with the header ``key,sha256``.

    Returns
    -------
    dict[str, str]
        SHA-256 hex digest by key, in file order.

    Raises
    ------
    ValueError
        If the header is not ``key,sha256``, a key repeats, or a key names a file whose APID has
        no L1A product definition.
    """
    defined = set(defined_l1a_apids())
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header != _MANIFEST_HEADER:
            raise ValueError(f"{path.name} must start with the header {','.join(_MANIFEST_HEADER)}, not {header}.")
        manifest: dict[str, str] = {}
        for key, sha256 in reader:
            if key in manifest:
                raise ValueError(f"{path.name} lists {key} twice. Remove the repeated row.")
            if golden_apid(key) not in defined:
                raise ValueError(
                    f"{path.name} lists {key}, whose APID has no L1A product definition. Remove the row, or add the "
                    f"APID's definition to LIBERA_L1A_PRODUCT_DEFINITIONS_BY_APID."
                )
            manifest[key] = sha256
    return manifest


def run_prefix(commit: str, run_id: str) -> str:
    """Return the key prefix of one run under the output root: ``<REPOSITORY>/<commit>/<run_id>``."""
    return f"{REPOSITORY}/{commit}/{run_id}"


def upload_run_outputs(output_dir: Path, output_root: str, run_prefix: str) -> list[str]:
    """Upload a run's outputs to ``<output_root>/<run_prefix>/<kind>/...``, tagging each object ``kind=<kind>``.

    Parameters
    ----------
    output_dir : Path
        The run's local output directory, holding one subdirectory per kind in :data:`OUTPUT_KINDS`;
        a missing kind is skipped.
    output_root : str
        ``s3://`` URI of the output root.
    run_prefix : str
        Key prefix of this run under the root, from :func:`run_prefix`.

    Returns
    -------
    list[str]
        The uploaded keys, relative to ``output_root``.

    Raises
    ------
    ValueError
        If ``output_root`` is not an ``s3://`` URI.
    """
    if not is_s3(output_root):
        raise ValueError("The output root is not an s3:// URI. Set E2E_OUTPUT_ROOT to s3://<bucket>[/<prefix>].")
    root = S3Path(output_root)
    s3 = boto3.client("s3")
    uploaded = []
    for kind in OUTPUT_KINDS:
        kind_dir = output_dir / kind
        if not kind_dir.is_dir():
            continue
        for path in sorted(p for p in kind_dir.rglob("*") if p.is_file()):
            relative = f"{run_prefix.strip('/')}/{kind}/{path.relative_to(kind_dir).as_posix()}"
            key = f"{root.key.rstrip('/')}/{relative}".lstrip("/")
            s3.upload_file(str(path), root.bucket, key, ExtraArgs={"Tagging": f"kind={kind}"})
            uploaded.append(relative)
    return uploaded


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_golden_inputs(golden_root: str, manifest: Mapping[str, str], staging_dir: Path) -> dict[str, Path]:
    """Return a local path for every golden object, copying from S3 when the root is an S3 URI.

    A local root is read in place. Every object is checked to exist before any is copied, and
    every local file is checked against its manifest digest.

    Parameters
    ----------
    golden_root : str
        ``s3://`` URI or local directory holding the manifest's keys.
    manifest : Mapping[str, str]
        SHA-256 hex digest by object key, relative to ``golden_root``.
    staging_dir : Path
        Local directory to copy S3 objects into. Unused for a local root.

    Returns
    -------
    dict[str, Path]
        Local path by object key.

    Raises
    ------
    FileNotFoundError
        If any object is missing under ``golden_root``; the message names every missing key.
    ValueError
        If any object's SHA-256 differs from the manifest; the message names every such key.
    """
    root = AnyPath(golden_root)
    missing = [key for key in manifest if not (root / key).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} golden input object(s) are missing under the golden root: {', '.join(missing)}. "
            f"Upload them there, or point the golden root at a copy that holds them."
        )

    if not is_s3(golden_root):
        staged = {key: Path(golden_root) / key for key in manifest}
    else:
        staged = {}
        for key in manifest:
            local = staging_dir / key
            local.parent.mkdir(parents=True, exist_ok=True)
            with warnings.catch_warnings():
                # Ground CCSDS names carry no extension, which smart_copy_file warns about on every copy.
                warnings.filterwarnings("ignore", message="You have copied a file without a file extension")
                smart_copy_file(str(root / key), local)
            staged[key] = local

    changed = [key for key, local in staged.items() if _sha256(local) != manifest[key]]
    if changed:
        raise ValueError(
            f"{len(changed)} golden input object(s) differ from the manifest's SHA-256: {', '.join(changed)}. "
            f"The golden input has changed; restore it, or regenerate the manifest if the change is intended."
        )
    return staged


class PeakMemorySampler:
    """Context manager that records this process's peak resident set size while it is open.

    A thread samples the RSS every ``interval_s`` seconds, and once on entry and on exit. Unlike
    ``ru_maxrss``, the peak starts afresh for each sampler. Memory held only by child processes
    is not counted.

    Parameters
    ----------
    interval_s : float
        Seconds between samples.
    """

    def __init__(self, interval_s: float = 0.05):
        self.interval_s = interval_s
        self.peak_bytes = 0
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        self.peak_bytes = max(self.peak_bytes, self._process.memory_info().rss)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def __enter__(self) -> PeakMemorySampler:
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self._stop.set()
        self._thread.join()
        self._sample()


# Loggers whose INFO records are findings: the packet-order summary and how sample times step
# across packets. Every other logger contributes only WARNING and above.
FINDINGS_INFO_LOGGERS = ("libera_utils.l1a.packet_ordering", "libera_utils.l1a.packet_slicing")


class LogFindings(logging.Handler):
    """Context manager that collects the ``libera_utils`` log records a run report shows.

    While open, it collects every WARNING and above from ``libera_utils``, and INFO from
    :data:`FINDINGS_INFO_LOGGERS`, such as duplicate data times and packet-time inversions
    along ``SRC_SEQ_CTR``. Records still reach the other handlers.
    It sees only records the loggers' levels let through, so INFO findings need those loggers
    at INFO or below, as :func:`~libera_utils.logutil.configure_task_logging` sets them.
    """

    def __init__(self):
        super().__init__(level=logging.INFO)
        self.findings: list[dict[str, str]] = []
        self._logger = logging.getLogger("libera_utils")

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno >= logging.WARNING or record.name in FINDINGS_INFO_LOGGERS:
            self.findings.append({"level": record.levelname, "logger": record.name, "message": record.getMessage()})

    def __enter__(self) -> LogFindings:
        self._logger.addHandler(self)
        return self

    def __exit__(self, *exc_info) -> None:
        self._logger.removeHandler(self)


def _to_json(value: Any) -> Any:
    if isinstance(value, timedelta):
        return value.total_seconds()
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if isinstance(value, Path | date):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


class RunReport:
    """The run report: a header and one entry per APID, rewritten to ``path`` on every change.

    Rewriting after each APID keeps the entries already recorded if a later APID kills the run.

    Parameters
    ----------
    path : Path
        JSON file to write.
    header : dict[str, Any]
        Run-level fields, such as the golden day and the commit.
    """

    def __init__(self, path: Path, header: dict[str, Any]):
        self.path = path
        self.header = header
        self.apids: dict[str, dict[str, Any]] = {}
        self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        partial = self.path.with_suffix(".json.partial")
        partial.write_text(json.dumps({"header": self.header, "apids": self.apids}, indent=2, default=_to_json))
        partial.replace(self.path)

    def record(self, apid: LiberaApid, entry: dict[str, Any]) -> None:
        """Set the entry for ``apid`` and rewrite the report.

        Raises
        ------
        TypeError
            If ``entry`` holds a value that is not a JSON type, timedelta, dataclass, Path or date.
        """
        self.apids[apid.name] = entry
        self._write()

    def write_step_summary(self, summary_path: Path) -> None:
        """Append a Markdown summary of the report to ``summary_path``, such as ``$GITHUB_STEP_SUMMARY``."""
        lines = [
            "## L1A golden-day E2E run",
            "",
            f"Golden day: {self.header.get('golden_day')}. "
            f"Defined L1A APIDs absent from the golden input: {', '.join(self.header.get('absent_apids', [])) or 'none'}.",
            "",
            "| APID | Status | Packets in | Packets out | Peak memory (GiB) | Run time (s) | Day coverage | Warnings |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for name, entry in self.apids.items():
            peak = entry.get("peak_memory_bytes")
            day_frac = entry.get("data_time_coverage", {}).get("day_frac")
            warning_count = sum(finding["level"] != "INFO" for finding in entry.get("log_findings", []))
            lines.append(
                f"| {name} | {entry.get('status')} | {entry.get('packets_in', '')} | {entry.get('packets_out', '')} "
                f"| {'' if peak is None else f'{peak / 2**30:.2f}'} | {entry.get('total_s', '')} "
                f"| {'' if day_frac is None else f'{day_frac:.4f}'} | {warning_count} |"
            )
        for name, entry in self.apids.items():
            if entry.get("log_findings"):
                lines += ["", f"### {name} log findings", ""]
                # A message may span lines, such as a numpy array repr; a list item must not.
                lines += [
                    f"- {f['level']} `{f['logger']}`: {' '.join(f['message'].split())}" for f in entry["log_findings"]
                ]
        with summary_path.open("a") as f:
            f.write("\n".join(lines) + "\n")


def _window_mask(times: np.ndarray, day: date) -> np.ndarray:
    bounds = day_bounds(day)
    # Data-time axes are datetime64, which carries no timezone; the bounds are UTC by construction.
    start = np.datetime64(bounds.window_start.replace(tzinfo=None), "us")
    end = np.datetime64(bounds.window_end.replace(tzinfo=None), "us")
    return (times >= start) & (times <= end)


def in_window_data_times(dataset: xr.Dataset, day: date) -> dict[str, np.ndarray]:
    """Return each data-time axis's values inside the closed day window of ``day``, sorted.

    The window edges come from :func:`~libera_utils.l1a.day_window.day_bounds`, independently of
    the trim under test.
    """
    in_window = {}
    for axis in data_time_axes(dataset):
        times = dataset[axis].values
        in_window[axis] = np.sort(times[_window_mask(times, day)])
    return in_window


def assert_coverage_meets_policy(coverage: TimeAxisCoverage, policy: DayCoveragePolicy) -> None:
    """Assert that a daily product's measured time-axis coverage meets its APID's day-coverage policy.

    Under ``FRACTION`` the day core must reach ``day_coverage_frac`` and each midnight buffer
    ``buffer_coverage_frac``, the thresholds the L1A combine gate applies to L0 spans. Under
    ``PRESENCE`` the axis must hold at least one time.

    Parameters
    ----------
    coverage : TimeAxisCoverage
        :func:`~libera_utils.l1a.day_coverage.measure_time_axis_coverage` of the product's data-time axis.
    policy : DayCoveragePolicy
        ``get_packet_config(apid).day_coverage``.

    Raises
    ------
    AssertionError
        If the coverage falls short of the policy; the message names each short region.
    """
    if policy.mode is CoverageMode.PRESENCE:
        if coverage.n_times == 0:
            raise AssertionError("The product holds no data time, but its day_coverage policy requires presence.")
        return
    short = [
        f"{region} {value:.4f} < {minimum}"
        for region, value, minimum in (
            ("day core", coverage.day_frac, policy.day_coverage_frac),
            ("left buffer", coverage.left_frac, policy.buffer_coverage_frac),
            ("right buffer", coverage.right_frac, policy.buffer_coverage_frac),
        )
        if value < minimum
    ]
    if short:
        raise AssertionError(
            f"Coverage falls short of the APID's day_coverage policy: {'; '.join(short)}. Check that the golden "
            f"input holds every file whose data times reach the day window."
        )


def assert_day_window_invariants(product: xr.Dataset, expected_in_window: dict[str, np.ndarray], day: date) -> None:
    """Assert that a daily L1A product holds exactly the day window of its parsed input.

    Three things are checked on every data-time axis:

    - no data time repeats;
    - the product's data times inside the window are exactly ``expected_in_window``, the parsed
      input's, so no in-window data was lost or added;
    - every packet carries at least one data time inside the window. Whole packets survive the
      trim, so a packet may also carry data times outside it.

    Parameters
    ----------
    product : xr.Dataset
        Daily L1A product, as written.
    expected_in_window : dict[str, np.ndarray]
        :func:`in_window_data_times` of the parsed input before the trim.
    day : date
        Applicable UTC day.

    Raises
    ------
    AssertionError
        If the product has no data-time axis, its axes differ from ``expected_in_window``'s, or
        any check fails; the message names the axis.
    """
    axes = data_time_axes(product)
    if not axes:
        raise AssertionError("The product has no data-time axis, so its day window cannot be checked.")
    if sorted(axes) != sorted(expected_in_window):
        raise AssertionError(
            f"The product's data-time axes {axes} differ from the input's {sorted(expected_in_window)}."
        )

    sample_dims = find_sample_dims(product)
    packet_in_window = np.zeros(product.sizes[PACKET_DIM], dtype=bool)
    for axis in axes:
        times = product[axis].values
        values, counts = np.unique(times, return_counts=True)
        repeated = values[counts > 1]
        if repeated.size:
            raise AssertionError(
                f"Data times on {axis!r} are not unique: {repeated.size} repeat, first {repeated[:5]}."
            )

        in_window = _window_mask(times, day)
        kept = np.sort(times[in_window])
        expected = expected_in_window[axis]
        if not np.array_equal(kept, expected):
            lost = np.setdiff1d(expected, kept)
            added = np.setdiff1d(kept, expected)
            raise AssertionError(
                f"Data times on {axis!r} inside the day window differ from the parsed input's: {lost.size} lost "
                f"(first {lost[:5]}), {added.size} not in the input (first {added[:5]})."
            )

        if axis == CAMERA_TIME_COORD:
            image_id = product[PACKET_IMAGE_ID_VAR].values
            packet_in_window |= np.isin(image_id, image_id[sample_to_packet_index(product, axis)[in_window]])
        elif axis in sample_dims:
            packet_in_window[sample_to_packet_index(product, axis)[in_window]] = True
        else:
            packet_in_window |= in_window

    outside = np.flatnonzero(~packet_in_window)
    if outside.size:
        raise AssertionError(
            f"{outside.size} packet(s) carry no data time inside the day window, first at {PACKET_DIM} index "
            f"{outside[:5]}."
        )
