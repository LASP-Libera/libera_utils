"""Tests for .github/scripts/check_rule_budget.py, the script the Rule budget workflow runs.

Covers the rule line count against the budget, and the --lanes re-count against the numbers
standards/test-lanes.md states, including the stale and skipped outcomes. The script is imported
by file path, since .github/scripts is not a package.
"""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / ".github" / "scripts" / "check_rule_budget.py"
_spec = importlib.util.spec_from_file_location("check_rule_budget", SCRIPT)
check_rule_budget = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_rule_budget)

LANE = check_rule_budget.UNIT_LANE

# One reviewer rule with a link line, one check-tier rule, and the wording for the first.
LEDGER = """# Review rules

### R-001 · Validate early

tier: reviewer · status: provisional

Wording: [R-001](x.md#r-001)

Do not flag: a deferred validator.

### R-002 · Bump the version

tier: check · status: graduated

Check: a workflow.
"""
WORDING = """# Instructions

## Review Rules

### R-001 · Validate early

Validate in the constructor.

## Restrictions for AI Agents
"""
# The ledger's 15 lines and the section's 6, less 4 for the one linked rule's second title and
# link line with their blank lines, less the check-tier entry's 5 lines from its title on.
COUNTED = 15 + 6 - 4 - 5


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A scratch repository root for the script to measure."""
    monkeypatch.setattr(check_rule_budget, "ROOT", tmp_path)
    return tmp_path


def _write_tests(root: Path, body: str) -> None:
    (root / "tests").mkdir()
    (root / "tests" / "test_sample.py").write_text(body)


def test_lane_count_matches_what_pytest_collects(tree):
    _write_tests(tree, "def test_a():\n    pass\n\n\ndef test_b():\n    pass\n")
    lanes = "| Unit lane | 2 | **1 s** | on main | `pytest tests/` |"
    result = check_rule_budget.check_lane("unit lane count", LANE, ["tests/"], lanes)
    assert (result.ok, result.detail) == (True, "2")


def test_lane_count_that_differs_is_stale(tree):
    _write_tests(tree, "def test_a():\n    pass\n")
    lanes = "| Unit lane | 2 | **1 s** | on main | `pytest tests/` |"
    result = check_rule_budget.check_lane("unit lane count", LANE, ["tests/"], lanes)
    assert result.ok is False
    assert result.detail == "test-lanes.md says 2, pytest collects 1"


def test_lane_count_is_skipped_when_pytest_cannot_collect(tree):
    _write_tests(tree, "import a_module_that_does_not_exist\n\n\ndef test_a():\n    pass\n")
    lanes = "| Unit lane | 1 | **1 s** | on main | `pytest tests/` |"
    result = check_rule_budget.check_lane("unit lane count", LANE, ["tests/"], lanes)
    assert result.ok is None
    assert "could not collect" in result.detail


def test_lane_count_test_lanes_no_longer_states_is_stale():
    result = check_rule_budget.check_lane("unit lane count", LANE, ["tests/"], "no counts here")
    assert result.ok is False
    assert "no longer states this" in result.detail


@pytest.mark.parametrize(
    ("budget", "ok"),
    [
        pytest.param(COUNTED, True, id="at-budget"),
        pytest.param(COUNTED - 1, False, id="one-line-over"),
    ],
)
def test_rule_budget_counts_the_ledger_and_the_wording(tmp_path, monkeypatch, budget, ok):
    rules = tmp_path / "review-rules.md"
    rules.write_text(LEDGER)
    instructions = tmp_path / "instructions.md"
    instructions.write_text(WORDING)
    monkeypatch.setattr(check_rule_budget, "RULES", rules)
    monkeypatch.setattr(check_rule_budget, "INSTRUCTIONS", instructions)
    result = check_rule_budget.check_rule_budget(f"| Rule budget | **{budget} lines**, the ledger and the wording |")
    assert result.ok is ok
    assert f"{COUNTED}" in result.detail
    assert "1 live rules" in result.detail


def test_rule_budget_the_readme_no_longer_states_is_stale():
    result = check_rule_budget.check_rule_budget("no budget here")
    assert result.ok is False
    assert "no longer states the rule budget" in result.detail
