"""Plugin module for the end-to-end tests under ``tests/e2e``.

The golden root and the output directory come from the environment, so no bucket name is
written into the repository.
"""

import hashlib
import logging
import os
import platform
from pathlib import Path

import pytest

from libera_utils.logutil import configure_task_logging
from libera_utils.version import version as libera_utils_version
from tests.e2e.harness import (
    GOLDEN_DAY,
    GOLDEN_MANIFEST_PATH,
    OUTPUT_KINDS,
    RunReport,
    defined_l1a_apids,
    golden_apid,
    golden_root_from_env,
    load_golden_manifest,
    output_dir_from_env,
    run_commit,
    stage_golden_inputs,
)


@pytest.fixture(scope="session")
def e2e_golden_root() -> str:
    """The golden root, as :func:`tests.e2e.harness.golden_root_from_env` reads it."""
    return golden_root_from_env(os.environ)


@pytest.fixture(scope="session")
def e2e_output_dir() -> Path:
    """The local output directory, as :func:`tests.e2e.harness.output_dir_from_env` reads it, with one
    subdirectory per output kind."""
    root = output_dir_from_env(os.environ)
    for kind in OUTPUT_KINDS:
        (root / kind).mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture(scope="session")
def golden_manifest() -> dict[str, str]:
    """SHA-256 by key of the golden input."""
    return load_golden_manifest(GOLDEN_MANIFEST_PATH)


@pytest.fixture(scope="session")
def golden_inputs(e2e_golden_root, golden_manifest, tmp_path_factory) -> dict[str, Path]:
    """Local path by key of every golden object, staged from S3 when the root is an S3 URI."""
    return stage_golden_inputs(e2e_golden_root, golden_manifest, tmp_path_factory.mktemp("golden_input"))


@pytest.fixture(scope="session")
def run_report(e2e_output_dir, golden_manifest):
    """The run report, written to ``reports/run_report.json`` and summarized to ``$GITHUB_STEP_SUMMARY`` if set."""
    golden = {golden_apid(key) for key in golden_manifest}
    header = {
        "golden_day": GOLDEN_DAY,
        "manifest": GOLDEN_MANIFEST_PATH.name,
        "manifest_sha256": hashlib.sha256(GOLDEN_MANIFEST_PATH.read_bytes()).hexdigest(),
        "libera_utils_version": libera_utils_version(),
        "python_version": platform.python_version(),
        "commit": run_commit(os.environ, GOLDEN_MANIFEST_PATH.parent),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "golden_apids": sorted(apid.name for apid in golden),
        "absent_apids": [apid.name for apid in defined_l1a_apids() if apid not in golden],
    }
    report = RunReport(e2e_output_dir / "reports" / "run_report.json", header)
    yield report
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        report.write_step_summary(Path(summary))


@pytest.fixture
def e2e_task_log(request, e2e_output_dir):
    """Log this test to ``logs/<test id>.log`` at DEBUG."""
    configure_task_logging(
        request.node.callspec.id, log_dir=e2e_output_dir / "logs", limit_debug_loggers="libera_utils"
    )
    yield
    root = logging.getLogger()
    for handler in root.handlers:
        handler.close()
    root.handlers = []
