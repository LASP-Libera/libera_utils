"""Tests for .github/scripts/check_rule_budget.py, the script the rule-budget pre-commit hook runs.

Covers the rule line count against the budget the ledger's header states, and exit 2 with a
CANNOT RUN line when the budget, the Review Rules section or a file is missing. The script is
imported by file path, since .github/scripts is not a package.
"""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / ".github" / "scripts" / "check_rule_budget.py"
_spec = importlib.util.spec_from_file_location("check_rule_budget", SCRIPT)
check_rule_budget = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_rule_budget)

# A budget line, one reviewer rule with a link line, one check-tier rule, and the wording for the
# first.
LEDGER = """# Review rules

**Budget: {budget} lines**

### R-001 · Validate early

tier: reviewer · status: provisional

Wording: [R-001](x.md#r-001)

Do not flag: a deferred validator.

### R-002 · Bump the version

tier: check · status: graduated

Check: a hook.
"""
WORDING = """# Instructions

## Review Rules

### R-001 · Validate early

Validate in the constructor.

## Restrictions for AI Agents
"""
# The ledger's 17 lines and the section's 6, less 4 for the one linked rule's second title and
# link line with their blank lines, less the check-tier entry's 5 lines from its title on.
COUNTED = 17 + 6 - 4 - 5


@pytest.fixture
def files(tmp_path, monkeypatch):
    """Point the script at a scratch ledger and instruction file, returned for the test to write."""
    rules, instructions = tmp_path / "review-rules.md", tmp_path / "instructions.md"
    monkeypatch.setattr(check_rule_budget, "RULES", rules)
    monkeypatch.setattr(check_rule_budget, "INSTRUCTIONS", instructions)
    return rules, instructions


@pytest.mark.parametrize(
    ("budget", "ok", "code"),
    [
        pytest.param(COUNTED, True, 0, id="at-budget"),
        pytest.param(COUNTED - 1, False, 1, id="one-line-over"),
    ],
)
def test_rule_budget_counts_the_ledger_and_the_wording(files, capsys, budget, ok, code):
    rules, instructions = files
    rules.write_text(LEDGER.format(budget=budget))
    instructions.write_text(WORDING)
    result, detail = check_rule_budget.check_rule_budget()
    assert result is ok
    assert f"{COUNTED}" in detail
    assert "1 live rules" in detail
    assert check_rule_budget.main() == code
    assert ("ok" if ok else "STALE") in capsys.readouterr().out


@pytest.mark.parametrize(
    ("ledger", "wording", "reason"),
    [
        pytest.param(LEDGER.replace("**Budget: {budget} lines**", ""), WORDING, "states no", id="no-budget"),
        pytest.param(
            LEDGER, WORDING.replace("## Review Rules", "## Other"), "no Review Rules section", id="no-section"
        ),
        pytest.param(None, WORDING, "No such file", id="no-ledger"),
    ],
)
def test_a_missing_budget_heading_or_file_exits_2(files, capsys, ledger, wording, reason):
    rules, instructions = files
    if ledger is not None:
        rules.write_text(ledger.format(budget=300))
    instructions.write_text(wording)
    assert check_rule_budget.main() == 2
    out = capsys.readouterr().out
    assert "CANNOT RUN" in out
    assert reason in out
