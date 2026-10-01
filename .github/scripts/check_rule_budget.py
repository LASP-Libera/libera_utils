#!/usr/bin/env python3
"""Check that the review rules fit their line budget, and on request re-count the test lanes.

CI runs this with no options, from .github/workflows/rule-budget.yml. It counts the lines of the
rule ledger (standards/review-rules.md) and the rule wording (the Review Rules section of
.github/instructions/libera-utils.instructions.md) as one text, leaves out check-tier rules,
which a tool enforces, and compares the total with the "Rule budget" row of standards/README.md.
Every agent session in the repository loads the wording and the reviewer applies every rule, so
the budget caps how far the rules can grow before one has to be retired or merged.

--lanes also re-counts the tests pytest collects in the unit and PR lanes and compares them with
the Tests column of standards/test-lanes.md. Those counts move with every test added, so a
person checks them at the monthly revision; CI never does. Wall clock is never checked: it is
machine-local and load-sensitive, and the same lane has measured 102 s and 357 s on one machine.

Prints one line per check: ok; STALE, when the stated number no longer holds or the file no
longer states it; or skipped, when pytest could not collect. Exits 1 on any STALE line, and with
--strict on a skipped one too.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "standards" / "README.md"
LANES = ROOT / "standards" / "test-lanes.md"
RULES = ROOT / "standards" / "review-rules.md"
INSTRUCTIONS = ROOT / ".github" / "instructions" / "libera-utils.instructions.md"
# The Tests cell of each row in test-lanes.md's Measured table.
UNIT_LANE = r"^\| Unit lane\s*\| (\d[\d,]*)\s*\|"
PR_LANE = r"^\| PR lane\s*\| (\d[\d,]*)\s*\|"


class Result:
    """One check's outcome: ok is True, False when stale, or None when the check could not run."""

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
    """The number a standards/ file states where pattern matches, or None when it no longer does."""
    match = re.search(pattern, text, re.M)
    return int(match.group(1).replace(",", "")) if match else None


def check_lane(label: str, pattern: str, pytest_args: list[str], lanes: str) -> Result:
    """Compare the test count test-lanes.md states for one lane with what pytest collects."""
    want = stated(pattern, lanes)
    if want is None:
        return Result(label, False, f"standards/test-lanes.md no longer states this; pattern {pattern!r} found nothing")
    got = collected(pytest_args)
    if got is None:
        return Result(label, None, "pytest could not collect; check skipped")
    if got != want:
        return Result(label, False, f"test-lanes.md says {want}, pytest collects {got}")
    return Result(label, True, f"{got}")


def check_rule_budget(readme: str) -> Result:
    """Compare the ledger and wording line count with the budget the README text states."""
    budget = re.search(r"^\| Rule budget\s*\| \*\*(\d+) lines\*\*", readme, re.M)
    if not budget:
        return Result("rule budget", False, "standards/README.md no longer states the rule budget")
    max_lines = int(budget.group(1))
    body = RULES.read_text()
    entries = re.findall(r"^### (R-\d+) · ([^\n]+)\n(.*?)(?=^### |^## |\Z)", body, re.M | re.S)
    # A check-tier rule is outside the budget: a tool enforces it and the reviewer holds a pointer.
    checked = [e for e in entries if re.search(r"^tier: check\b", e[2], re.M)]
    live = len([e for e in entries if "retired" not in e[1].lower() and e not in checked])
    wording = re.search(r"^## Review Rules$.*?(?=^## |\Z)", INSTRUCTIONS.read_text(), re.M | re.S)
    if not wording:
        return Result("rule budget", False, f"{INSTRUCTIONS.name} has no Review Rules section")
    # Counted as if the ledger and the wording were one file: each linked rule's second title and
    # its link line, each with a blank line, repeat structure rather than add rule text.
    linked = len(re.findall(r"^Wording: \[", body, re.M))
    lines = (
        len(body.splitlines())
        + len(wording.group(0).splitlines())
        - 4 * linked
        - sum(len(f"{e[1]}\n{e[2]}".splitlines()) for e in checked)
    )
    if lines > max_lines:
        return Result("rule budget", False, f"{lines} lines over a budget of {max_lines}; {live} live rules")
    return Result("rule budget", True, f"{lines}/{max_lines} lines; {live} live rules")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lanes", action="store_true", help="also check the stated lane test counts")
    parser.add_argument("--strict", action="store_true", help="treat an unverifiable check as a failure")
    args = parser.parse_args()

    results = []
    if args.lanes:
        lanes = LANES.read_text()
        results += [
            check_lane("unit lane count", UNIT_LANE, ["tests/unit/"], lanes),
            check_lane("PR lane count", PR_LANE, ["-m", "not e2e", "tests/"], lanes),
        ]
    results.append(check_rule_budget(README.read_text()))

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
        print(f"\n{failed} number(s) stated in standards/ no longer hold. Re-measure and update the")
        print("row, or say in it why the number stands: a setting reasoned from a stale number only")
        print("looks measured.")
        return 1
    if skipped and args.strict:
        print(f"\n{skipped} check(s) could not run, and --strict was given.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
