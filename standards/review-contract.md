# Review contract

How a reviewer behaves in `libera_utils`: the implementation reviewer at the end of a build,
which reads this file because `AGENTS.md` points at it, and the `github-pr-reviewer` agent
when `pr-findings` runs it. Read with `rules.md`, the shared `terminology.md` and
`decisions.md`, the local `decisions.md`, and `test-lanes.md`.

## What the reviewer loads

Those five files on every run. A `context/` entry from the shared corpus only when the
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

## Answering a finding

Findings arrive as one comment on the pull request, each with an identifier. A verdict is one
line in a reply, `<id> <verdict>`: `accept`, `decline: <reason>` or `escalate: <reason>` for a
blocking finding; `take` or `note` for a suggestion; a one-line answer or `leave` for a
question. Prose around the line is ignored, so argue in the same comment you answer in.

A decline or an escalation without its reason is not recorded. The reason is what tells the
next ratchet whether the rule was wrong or the code was, and it is the only part of a record
that cannot be reconstructed later.

A finding nobody answers stays unanswered and out of the record. Silence is not assent, and
the review says which findings it is.

A finding gets one verdict. Anyone may trigger the review, but only the first run against a
given head posts — the findings comment is keyed to the commit it reviewed, so a second run is
a no-op and nobody has to coordinate. Several reviewers then answer in that one thread, which
is the point of putting it there. Two of them disagreeing is an escalation the author settles
in the thread; the record stores the settled verdict, `contested: true`, and both original
lines.

**A merged pull request is tagged in a file, not a thread.** Calibration and backfill review
code that already shipped, and reopening a merged pull request to comment on it would be
noise. The findings go to `.review/calibration/pr-NNNN-findings.md` in the same shape, with
the same closed vocabulary and the same requirement that a decline carry its reason; a person
writes the verdicts into the file. The record marks it `tagged: in-file` and
`reviewed_after_merge: true`, because a finding raised against shipped code never had the
chance to change it — evidence that a rule fires, not that the team acted on it.

No bulk verdict. A reviewer who reads a list and says "all fine" has adjudicated nothing, and
six accepts that mean one glance are worse than three that mean three.

## Severity

| Tag        | Handling                                                                                                                                                                                           |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| must-fix   | Ranked, counted against the budget, tagged by a person. Before emitting one, read the **file** rather than the diff and quote the lines; a must-fix that cannot be anchored to lines is downgraded |
| should-fix | Ranked, counted, tagged                                                                                                                                                                            |
| suggestion | Below the fold, own cap of 7, recorded `taken` or `noted`, no reason needed                                                                                                                        |
| question   | Its own list, outside the budget. Check both `decisions.md` files first: if a decision answers it, cite the decision instead of asking again                                                       |

Budget: **7** must-fix and should-fix per run, ranked. Anything past 7 is a count per rule,
not a list.

The implementation reviewer grades findings `blocking` or `non-blocking`: a blocking finding
is must-fix or should-fix by the same test as above, and a non-blocking one is a suggestion.

## Citation

Every finding names a rule ID, a decision, `ticket/AC-n`, `ticket/scope`, `ticket/plan`,
`term/T-nnn`, `test/rework`, `test/uncovered`, `test/failure-path`, `test/duplicate`,
`helper/path::symbol`, or `other`. A shared decision is cited `D-nnn`; one of this
repository's own is cited `D-nnn@libera_utils`, such as `D-008@libera_utils`, because the
citation must contain no `/`. An `other` finding carries a one-line summary suitable for
clustering at the ratchet.

The finding key is that citation, then `/`, the file path, `::` and the enclosing symbol:
`R-012/pyproject.toml::version`, `D-008@libera_utils/libera_utils/cli.py::main`,
`test/uncovered/libera_utils/io/netcdf.py::write`. The citation is everything before the
first `/` that follows it; the two-part `test/` and `ticket/` citations are the only ones
with a `/` of their own. The symbol is the enclosing function or class, or in a non-code file the key or heading; `-`
when there is none. A `helper/path::symbol` citation is already a full key. The comment and
the record both carry the whole key, which is what deduplication matches on.

An omission from the PR body's "look at this" is a **must-fix**, above anything about the
code.

## Test scrutiny

Three directions; the first is the classic, and the other two are what a green suite hides.

**A test weakened** is a finding until justified: a widened tolerance, a weakened
assertion, a removed `raises`, a new `skip` or `xfail`, a dropped `parametrize` case, or an
expected value re-tuned to the new output. Two more, specific to this repository:

- A test relocated to escape a guard rather than mocked. The outbound-network guard exists
  because 28 tests were silently downloading kernels from NAIF for months; the lanes and the
  guard are described once in `standards/test-lanes.md` (shared D-009, provisional).
- A golden value changed without the change being stated in the pull request body. A test
  asserts on its own step's product (shared D-010); a re-tuned golden value is a science claim.

**A test added where one should have been reworked**, keyed `test/rework`. An agent asked
to make a suite green adds; a person who knows the suite edits. A new test whose subject an
existing module already covers belongs in that module, beside its siblings or as a
`parametrize` case.

Count deletions before calling it padding. Count test functions added, modified (the same
name on both sides of the diff) and deleted, and compare added against modified **plus
deleted**: a rewrite that drops fifteen and adds nine is a consolidation, and against
modified alone it scores as the opposite of what it is. When added is the larger, the change
grew the suite, and a new test _file_ the plan did not name is a must-fix keyed
`test/rework`. When added is not the larger, files are being split or merged on purpose, and
an unplanned file is an escalation for the person instead. Judge on the subject, not the
assertion (shared D-010): the same input feeding the same number is duplicate coverage only
when the subject matches, and that is `test/duplicate`. One repository specific: shared
setup belongs in `tests/plugins/` as a fixture rather than in a new helper module. **Judge shape, not location** — where a test lives is settled by
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
symbol carrying a decorator that registers rather than calls: `@pytest.fixture`,
`@field_validator`, `@model_validator`, `@contextmanager` used as a fixture,
`@functools.singledispatch` registrations, and framework hooks generally. At most 7 test
findings and 7 helper findings; anything past that is a count.

## New surface

Keyed `helper/path::symbol`. Every function, method or class the diff adds that the plan's
signature list did not name is reported with its call sites, its length, and whether
something in the module or a sibling already does it. One caller and under about ten lines
is a candidate to inline; a duplicate of an existing function is a finding. Extraction is
not invention: a helper pulled out of existing code with two or more call sites is the good
case and is reported as such. A helper in `libera_utils/` whose only callers are in
`tests/` is not this key: R-008 covered it and retired, so it is `other` with a one-line summary, which is how the ratchet sees it
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
- Style preferences with no rule behind them.

**Never suggest defensive handling** the rules forbid — a broad `except`, a silent coercion, a
default standing in for a required input, a fallback to a cached or differently calibrated
product. Where the code needs to handle a condition, ask for the raise (R-002).

## Output

What `pr-findings` produces; the "already checked" section of the PR body is written by the
build, not by a reviewer:

- one comment on the pull request, carrying every finding, headed so a reader knows a
  machine wrote it and a machine will read the replies
- `standards/log/pr-NNNN.yaml`, written by `pr-record` from the replies in that thread —
  never from anything the reviewer decided on its own — and committed by a person, never the
  agent

## What the agent may post

This section and Output govern `pr-findings`; the implementation reviewer posts nothing.

One comment per review, and nothing else. It is an **issue comment** on the pull request,
posted through the GitHub MCP server with the credentials of the person who runs
`pr-findings`. It never pushes to the branch, opens a review, approves, requests changes or
applies a label. The person's confirmation is the enforcement: nothing posts until they have
read the comment as it will appear.

A review, an approval and a label are a person's signature on someone else's work. A comment
is a proposal anyone can read and argue with, which is what an agent's findings are.

The person who runs `pr-findings` sees the comment before it posts and confirms it; their
login goes in its attribution line.

**Without write access the review still works.** The skill writes the same body to
`.review/comment.md` and stops; a person pastes it into the pull request. Same findings, same
heading, same reply convention — only who presses the button changes.

## Never

Open a review, approve, request changes, apply a label, merge, or push. Post more than the
one findings comment. Adjudicate a finding itself. Edit anything under `standards/` except
`log/`. Write a record from replies that are not there.
