#!/usr/bin/env python3
"""Check that the review rules fit their line budget.

Pre-commit runs this, as the `rule-budget` hook, on a change under standards/ or
.github/instructions/. It counts the lines of the rule ledger (standards/review-rules.md) and the
rule wording (the Review Rules section of .github/instructions/libera-utils.instructions.md) as
one text, leaves out check-tier rules, which a tool enforces, and compares the total with the
budget the ledger's header states, `**Budget: N lines**`. Every agent session in the repository
loads the wording and the reviewer applies every rule, so the budget caps how far the rules can
grow before one has to be retired or merged.

Prints one line: ok, or STALE when the rules are over the budget, naming the count and the
budget. Exits 1 on STALE, and 2 with a CANNOT RUN line when a file or heading it reads is
missing.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RULES = ROOT / "standards" / "review-rules.md"
INSTRUCTIONS = ROOT / ".github" / "instructions" / "libera-utils.instructions.md"
BUDGET = re.compile(r"\*\*Budget: (\d+) lines\*\*")


class CannotRun(Exception):
    """A file or heading the count reads is missing."""


def check_rule_budget() -> tuple[bool, str]:
    """Whether the ledger and wording fit the ledger's stated budget, with the count.

    Returns
    -------
    tuple[bool, str]
        True when the rules fit, and the count, the budget and the number of live rules.

    Raises
    ------
    CannotRun
        When the ledger states no budget or the instruction file has no Review Rules section.
    OSError
        When either file cannot be read.
    """
    body = RULES.read_text()
    budget = BUDGET.search(body)
    if not budget:
        raise CannotRun(f"{RULES.name} states no '**Budget: N lines**'")
    max_lines = int(budget.group(1))
    entries = re.findall(r"^### (R-\d+) · ([^\n]+)\n(.*?)(?=^### |^## |\Z)", body, re.M | re.S)
    # A check-tier rule is outside the budget: a tool enforces it and the reviewer holds a pointer.
    checked = [e for e in entries if re.search(r"^tier: check\b", e[2], re.M)]
    live = len([e for e in entries if "retired" not in e[1].lower() and e not in checked])
    wording = re.search(r"^## Review Rules$.*?(?=^## |\Z)", INSTRUCTIONS.read_text(), re.M | re.S)
    if not wording:
        raise CannotRun(f"{INSTRUCTIONS.name} has no Review Rules section")
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
        return False, f"{lines} lines over a budget of {max_lines}; {live} live rules"
    return True, f"{lines}/{max_lines} lines; {live} live rules"


def main() -> int:
    try:
        ok, detail = check_rule_budget()
    except (OSError, CannotRun) as e:
        print(f"  CANNOT RUN {e}")
        return 2
    if ok:
        print(f"  ok       rule budget: {detail}")
        return 0
    print(f"  STALE    rule budget: {detail}")
    print("\nRetire or merge a rule, or raise the budget in the ledger's header with the reason.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
