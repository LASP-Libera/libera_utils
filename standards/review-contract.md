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

Findings arrive as one review on the pull request, each finding its own thread. A verdict is
the first line of a reply under that thread, starting with **Agree**, **Disagree** and a
reason, or **Discuss**; anything after that line is free, so argue in the same reply you answer
in. For a suggestion, Agree means taken; a question takes any answer. The verdict that counts
is the first one from a person, bots excluded, and that includes whoever ran `pr-findings`:
the findings are the agent's, not theirs.

A must-fix or should-fix disagreement without its reason is not recorded. The reason is what tells the
next ratchet whether the rule was wrong or the code was, and it is the only part of a record
that cannot be reconstructed later. A finding
nobody answers stays unanswered and out of the record; silence is not assent, a resolved
thread is not a verdict, and the review says which findings it is. `Discuss` holds the
finding open until a later Agree or Disagree in the same thread.

A finding gets one verdict. Anyone may trigger the review, but only the first run against a
given head posts — the review is keyed to the commit it reviewed, so a second run posts no
findings and nobody has to coordinate. Several reviewers then answer in the same threads,
which is the point of putting them there. A second person answering differently makes the
finding contested; the author settles it in the thread, and the record stores the settled
verdict, `contested: true`, and both original lines.

**A merged pull request is tagged in a file, not a thread.** Calibration and backfill review
code that already shipped, and reopening a merged pull request to comment on it would be
noise. The findings go to `.review/calibration/pr-NNNN-findings.md` in the same shape, with
the same three words and the same requirement that a disagreement carry its reason; a person
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
| suggestion | Below the fold, own cap of 7, answered Agree (taken) or Disagree, no reason needed                                                                                                                 |
| question   | Its own list, outside the budget. Check both `decisions.md` files first: if a decision answers it, cite the decision instead of asking again                                                       |

Budget: **7** must-fix and should-fix per run, ranked. Anything past 7 is a count per rule,
not a list.

The implementation reviewer grades findings `blocking` or `non-blocking`: a blocking finding
is must-fix or should-fix by the same test as above, and a non-blocking one is a suggestion.

## Citation

Every finding names a rule ID, a decision, `ticket/AC-n`, `ticket/scope`, `ticket/plan`,
`term/T-nnn`, a `test/` key from `test-suite-review` (`test/loosened`, `test/failure-path`,
`test/uncovered`, `test/mock`, `test/rework`, `test/duplicate`, `test/placement`,
`test/lane`), `helper/path::symbol`, or `other`. A shared decision is cited `D-nnn`; one of this
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

Tests are judged by the `test-suite-review` skill in the `libera-tools` plugin: loosening,
failure paths, changed lines no test runs, mocking away the change, factoring (extend before
adding, parametrize, existing fixtures) and lanes. The reviewer applies it whenever the diff
touches tests, and its keys are the `test/` citations above. What follows is what it cannot
know without this repository.

- **Lanes and commands** are in `standards/test-lanes.md`, including the coverage command
  `test/uncovered` uses. A fast unit test may call several levels deep rather than mock them.
- **Mocking** is `moto` for AWS and `responses` for HTTP. Fixtures live in `tests/plugins/`;
  non-fixture helpers in `tests/helpers.py`. New shared setup becomes a fixture there, not a
  new helper module.
- **A test relocated to escape the network guard** rather than mocked is `test/loosened`. The
  guard exists because 28 tests were silently downloading kernels from NAIF for months (shared
  D-009, provisional).
- **A golden value changed without the change being stated in the pull request body** is a
  must-fix `test/loosened`. A test asserts on its own step's product (shared D-010); a
  re-tuned golden value is a science claim.
- **Every `raise` the diff adds needs a test asserting it**, `test/failure-path`: a defined
  input produces a defined product or the run stops, so an untested raise is an unenforced
  contract (R-002, R-004).

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

- one `COMMENT` review on the pull request, a thread per finding, headed so a reader knows
  a machine wrote it and a machine will read the replies
- `standards/log/pr-NNNN.yaml`, written by `pr-findings` from the replies in those threads —
  never from anything the reviewer decided on its own — and committed by a person, never the
  agent

## What the agent may post

This section and Output govern `pr-findings`; the implementation reviewer posts nothing.

One review per head, of type **COMMENT**, with one inline comment per finding, and replies
under those threads that the person running `pr-findings` chose and confirmed. It is posted
through the GitHub MCP server with that person's credentials. It never pushes to the branch,
approves, requests changes, resolves a thread or applies a label. The person's confirmation is
the enforcement: nothing posts until they have read it as it will appear.

A `COMMENT` review is the one kind that neither approves nor blocks: a proposal anyone can
read and answer next to the code it is about, which is what an agent's findings are. An
approval, a request for changes and a label are a person's signature on someone else's work.

The person who runs `pr-findings` sees the review before it posts and confirms it; their login
goes in each attribution line.

**Without write access the review still works.** The skill writes the review to
`.review/review.md`, each comment with its path and line, and stops; a person posts it. Same
findings, same summary, same way of answering — only who presses the button changes.

## Never

Approve, request changes, resolve a thread, apply a label, merge, or push. Post more than one
findings review per head, or a reply the person did not choose. Adjudicate a finding itself.
Edit anything under `standards/` except `log/`. Write a record from replies that are not
there.
