"""Upload an E2E run's outputs to ``<output root>/libera_utils/<commit>/<run id>/``.

Usage: ``python -m tests.e2e.upload_outputs <output dir> <output root> [--run-id <run id>]``. The commit is the
one the run report records, so the upload is filed under the code the run tested.
"""

import argparse
import json
from pathlib import Path

from tests.e2e.harness import run_prefix, upload_run_outputs


def report_commit(output_dir: Path) -> str:
    """Return the commit that ``<output_dir>/reports/run_report.json`` records.

    Raises
    ------
    FileNotFoundError
        If the run report does not exist.
    ValueError
        If the report records no commit.
    """
    report_path = output_dir / "reports" / "run_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(
            f"No run report at {report_path}. Pass the LIBERA_E2E_OUTPUT_DIR of a run that has started."
        )
    commit = json.loads(report_path.read_text())["header"].get("commit")
    if not commit:
        raise ValueError(
            f"The run report at {report_path} records no commit. Run the tests from a git checkout, or in CI."
        )
    return commit


def main(argv: list[str] | None = None) -> None:
    """Parse the arguments and upload the run's outputs."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", type=Path, help="The run's local output directory, LIBERA_E2E_OUTPUT_DIR")
    parser.add_argument("output_root", help="s3:// URI of the output root")
    parser.add_argument(
        "--run-id",
        help="Names the run under its commit: <run id>-<attempt> in CI. Defaults to local-<output dir name>.",
    )
    args = parser.parse_args(argv)
    prefix = run_prefix(report_commit(args.output_dir), args.run_id or f"local-{args.output_dir.name}")
    uploaded = upload_run_outputs(args.output_dir, args.output_root, prefix)
    print(f"Uploaded {len(uploaded)} object(s) under {prefix}/")


if __name__ == "__main__":
    main()
