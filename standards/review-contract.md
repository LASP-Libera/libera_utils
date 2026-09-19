# Review contract

How a reviewer behaves in `libera_utils`. Read with `rules.md`, the shared
`terminology.md` and `decisions.md`, and the local `decisions.md`.

## What the reviewer loads

Those four files on every run. A `context/` entry from the shared corpus only when the
ticket's constraints name it, or a file the diff touches names it. **The ticket and the
plan are read before the diff.**

## Paths

| Path                                        | Counts as                                                                                           |
| ------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `libera_utils/`                             | Library code. Every rule applies                                                                    |
| `libera_utils/cli.py`                       | Entry point. R-002 applies at the boundary; converting an exception to an exit code here is correct |
| `libera_utils/data/`                        | Shipped configuration and product definitions. R-005, R-006 and R-014 apply; the code rules do not  |
| `tests/`                                    | Tests. R-009 and R-014 apply. R-004 does not. The test-scrutiny section below applies               |
| `doc/`                                      | Documentation. R-005, R-009, R-012 and R-014 apply                                                  |
| `pyproject.toml`, `.pre-commit-config.yaml` | Build and tool configuration. R-011 and R-012 apply; no code rule does                              |

## Tagging

Every finding gets a verdict, one at a time: `accept`, `decline: <one line>`, or `escalate`
for a blocking finding; `take` or `note` for a suggestion; a one-line answer or `leave` for a
question. A decline without its reason is not a decline — the reason is what tells the next
ratchet whether the rule was wrong or the code was, and it is the only part of a record that
cannot be reconstructed later.

No bulk verdict. A reviewer who reads a list and says "all fine" has adjudicated nothing, and
six accepts that mean one glance are worse than three that mean three.

## Severity

| Tag        | Handling                                                                                                                                                                                           |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| must-fix   | Ranked, counted against the budget, tagged by a person. Before emitting one, read the **file** rather than the diff and quote the lines; a must-fix that cannot be anchored to lines is downgraded |
| should-fix | Ranked, counted, tagged                                                                                                                                                                            |
| suggestion | Below the fold, own cap of 7, recorded `taken` or `noted`, no reason needed                                                                                                                        |
| question   | Its own list, outside the budget. Check `decisions.md` first: if a decision answers it, cite the decision instead of asking again                                                                  |

Budget: **7** must-fix and should-fix per run, ranked. Anything past 7 is a count per rule,
not a list.

## Citation

Every finding names a rule ID, `ticket/AC-n`, `ticket/scope`, `ticket/plan`, `term/T-nnn`,
`test/rework`, `test/uncovered`, `test/failure-path`, `test/duplicate`,
`helper/path::symbol`, or `other`. An `other` finding carries a one-line summary suitable for clustering at the
ratchet. The finding key is `rule/path::symbol`.

An omission from the PR body's "look at this" is a **must-fix**, above anything about the
code.

## Test scrutiny

Three directions; the first is the classic, and the other two are what a green suite hides.

**A test weakened** is a finding until justified: a widened tolerance, a weakened
assertion, a removed `raises`, a new `skip` or `xfail`, a dropped `parametrize` case, or an
expected value re-tuned to the new output. Two more, specific to this repository:

- A test relocated to escape a guard rather than mocked. The outbound-network guard exists
  because 28 tests were silently downloading kernels from NAIF for months; the lanes and the
  guard are described once in `standards/test-lanes.md` (D-009, provisional).
- A golden value changed without the change being stated in the pull request body. A test
  asserts on its own step's product (D-010); a re-tuned golden value is a science claim.

**A test added where one should have been reworked**, keyed `test/rework`. An agent asked
to make a suite green adds; a person who knows the suite edits. A new test whose subject an
existing module already covers belongs in that module, beside its siblings or as a
`parametrize` case.

Count deletions before calling it padding. The ratio is tests added against tests modified
**plus tests deleted**: a rewrite that drops fifteen and adds nine is a consolidation, and
against modified alone it scores as the opposite of what it is. The signal is a net rise with
nothing deleted. Where the net count is flat or falling, a new test _file_ the plan did not
name is an escalation for the person, not a finding to resolve before exit. Judge
on the subject, not the assertion (D-010): the same input feeding the same number is
duplicate coverage only when the subject matches, and that is `test/duplicate`. One repo specific: shared setup belongs in `tests/plugins/` as a fixture rather than in a new
helper module. **Judge shape, not location** — where a test lives is settled by
`standards/test-lanes.md`, and a finding about the lane layout belongs there, not here.

**Coverage of what changed**, keyed `test/uncovered` and `test/failure-path`. Run the
coverage command named in `standards/test-lanes.md` and list every line the diff added or
changed that no test executes, as `file:line`.
Separately, every `raise` the diff adds needs a test asserting it: this repository's
contract is that a defined input produces a defined product or the run stops, so an
untested raise is an unenforced contract (R-002, R-004). Report the lines, never a
percentage — a percentage produces tests written to the metric.

A framework-invoked symbol is not a helper. A `@pytest.fixture` is injected by name and a
pydantic `@field_validator` or `@model_validator` is called by the model, so a call-site count
says nothing about either and "one caller" is the normal case. Never raise `helper/...` on a
symbol carrying a decorator that registers rather than calls.

## New surface

Keyed `helper/path::symbol`. Every function, method or class the diff adds that the plan's
signature list did not name is reported with its call sites, its length, and whether
something in the module or a sibling already does it. One caller and under about ten lines
is a candidate to inline; a duplicate of an existing function is a finding. Extraction is
not invention: a helper pulled out of existing code with two or more call sites is the good
case and is reported as such. A helper in `libera_utils/` whose only callers are in `tests/` is not this key: R-008 covered
it and retired, so it is `other` with a one-line summary, which is how the ratchet sees it
recur and how the rule comes back if it does.

## Do not flag

This repository already runs these, and a reviewer that repeats them trains people to skip
the whole review. The evidence is the "share a linter could have caught" row in
`standards/README.md`: nine of those ten findings were line-length complaints on a repository
that disables `E501` **on purpose**
because `ruff format` owns wrapping.

- Line length, wrapping, import order, trailing whitespace, line endings — `ruff format`,
  `ruff check` (`E`, `W`, `F`, `I`) and the pre-commit hooks own these. `E501` and `F541`
  are ignored deliberately.
- Security patterns covered by `ruff` `S` (flake8-bandit) and `bandit`; `S` is disabled in
  `tests/` on purpose.
- pytest style covered by `ruff` `PT`; syntax modernisation covered by `UP`.
- YAML, JSON and markdown formatting — `prettier`. Spelling — `codespell`. An untagged
  deferred-work marker — the `prevent-dangling-todos` hook, which is R-010.
- Defensive handling the rules forbid: a broad `except`, a silent coercion, a default
  standing in for a required input, a fallback to a cached or differently calibrated
  product. Ask for the raise instead (R-002).
- Style preferences with no rule behind them.

## Output

`.review/findings.md` for the person to tag; the PR body's "already checked" section, with
suggestions and open questions in it; `.review/comments.md`, the accepted findings drafted
as review comments **for the person to post**.

## Never

Post a comment, review, label or approval to GitHub. Tag a finding itself. Merge. Edit
anything under `standards/` except `log/`. Write a record the person has not tagged.
