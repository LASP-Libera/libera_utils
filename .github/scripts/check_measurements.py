#!/usr/bin/env python3
"""Check the claims in standards/ that a machine can still verify.

The corpus states numbers and then reasons from them: a test count, a rule cap, a record
cap. Those go stale silently, and a threshold derived from a stale number is worse than no
threshold, because it looks measured. This re-derives them and reports what no longer holds.

Deliberately not checked: wall clock. `standards/README.md` records it as machine-local and
load-sensitive, and the same lane has measured 102 s and 357 s on this hardware. A CI check
on it would fail for reasons that say nothing about the repository. Remeasuring it stays a
person's job, at the ratchet.

Exit 1 on a mismatch. --strict makes an unverifiable check a failure too.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "standards" / "README.md"
RULES = ROOT / "standards" / "rules.md"
LOG = ROOT / "standards" / "log"


class Result:
    def __init__(self, name: str, ok: bool | None, detail: str) -> None:
        self.name, self.ok, self.detail = name, ok, detail


def collected(args: list[str]) -> int | None:
    """Number of tests pytest collects, or None when collection could not run."""
    # S603: every element is a literal defined in this file or sys.executable, the
    # interpreter already running. No shell, and nothing here reaches user input.
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "pytest", *args, "--collect-only", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    # A non-zero status from --collect-only means collection itself failed, usually a module
    # that will not import. Do not sniff the text for "error": test names contain it.
    if proc.returncode != 0:
        return None
    # "1001 tests collected" or "1071/1073 tests collected (2 deselected)"
    match = re.search(r"(\d+)(?:/\d+)? tests? collected", proc.stdout)
    return int(match.group(1)) if match else None


def stated(pattern: str, text: str) -> int | None:
    match = re.search(pattern, text)
    return int(match.group(1).replace(",", "")) if match else None


def check_lane(label: str, pattern: str, pytest_args: list[str], readme: str) -> Result:
    want = stated(pattern, readme)
    if want is None:
        return Result(label, False, f"standards/README.md no longer states this; pattern {pattern!r} found nothing")
    got = collected(pytest_args)
    if got is None:
        return Result(label, None, "pytest could not collect; check skipped")
    if got != want:
        return Result(label, False, f"README says {want}, pytest collects {got}")
    return Result(label, True, f"{got}")


def check_rule_cap(readme: str) -> Result:
    cap = re.search(r"\*\*(\d+) rules, (\d+) lines\*\*", readme)
    if not cap:
        return Result("rule cap", False, "standards/README.md no longer states the rule cap")
    max_rules, max_lines = int(cap.group(1)), int(cap.group(2))
    body = RULES.read_text()
    live = len([m for m in re.findall(r"^### (R-\d+) · (.+)$", body, re.M) if "retired" not in m[1].lower()])
    lines = len(body.splitlines())
    problems = []
    if live > max_rules:
        problems.append(f"{live} live rules over a cap of {max_rules}")
    if lines > max_lines:
        problems.append(f"{lines} lines over a cap of {max_lines}")
    if problems:
        return Result("rule cap", False, "; ".join(problems))
    return Result("rule cap", True, f"{live}/{max_rules} rules, {lines}/{max_lines} lines")


def check_record_cap() -> Result:
    readme = (LOG / "README.md").read_text()
    cap = re.search(r"(\d+)[- ]line", readme)
    if not cap:
        return Result("record cap", False, "standards/log/README.md no longer states a cap")
    limit = int(cap.group(1))
    over = [
        f"{p.name} is {len(p.read_text().splitlines())} lines"
        for p in sorted(LOG.glob("pr-*.yaml"))
        if len(p.read_text().splitlines()) > limit
    ]
    if over:
        return Result("record cap", False, f"cap {limit}: " + ", ".join(over))
    n = len(list(LOG.glob("pr-*.yaml")))
    return Result("record cap", True, f"{n} record(s), all within {limit} lines")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="treat an unverifiable check as a failure")
    args = parser.parse_args()

    readme = README.read_text()
    results = [
        check_lane("unit lane count", r"\| (\d[\d,]*) tests, measured on", ["tests/unit/"], readme),
        check_lane("PR lane count", r"\| (\d[\d,]*) tests, same run", ["-m", "not e2e", "tests/"], readme),
        check_rule_cap(readme),
        check_record_cap(),
    ]

    failed = skipped = 0
    for r in results:
        if r.ok is True:
            print(f"  ok       {r.name}: {r.detail}")
        elif r.ok is None:
            skipped += 1
            print(f"  skipped  {r.name}: {r.detail}")
        else:
            failed += 1
            print(f"  STALE    {r.name}: {r.detail}")

    if failed:
        print(f"\n{failed} stated measurement(s) no longer hold. Remeasure and update standards/, or")
        print("say in the row why the number stands. A threshold reasoned from a stale number")
        print("looks measured and is not.")
        return 1
    if skipped and args.strict:
        print(f"\n{skipped} check(s) could not run, and --strict was given.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
