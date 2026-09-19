# Standards for libera_utils

The review standard this repository holds itself to, and the measurements that set every
number in it. Written 2026-09-17. Re-measure at the tuning pass, not from memory.

**When this corpus and an authoritative source disagree, the source wins.** Open a pull
request to fix the corpus.

## What is here

| File                 | Holds                                                                          |
| -------------------- | ------------------------------------------------------------------------------ |
| `rules.md`           | The rules a reviewer checks, capped at 14                                      |
| `review-contract.md` | How the reviewer behaves: severity, budget, what not to flag                   |
| `decisions.md`       | Decisions local to this repository. Cross-repository ones are shared           |
| `SHARED.md`          | Where the shared vocabulary, decisions and context live, and how to reach them |
| `checks/`            | Rules that graduated into a tool, each naming the rule it replaced             |
| `archive.md`         | One line per rule that has left `rules.md`, with where its reasoning lives     |
| `test-lanes.md`      | Every lane marker, path and command the corpus depends on, in one place        |
| `log/`               | One record per reviewed pull request, and one report per ratchet               |

`.review/` is the agents' scratch directory and is gitignored.

## Measured, 2026-09-17

Window 2026-06-15 to 2026-09-15 unless stated. Source: the GitHub MCP server for pull
requests, the Atlassian MCP server for Jira, and a local run for test timings.

The pull-request counts were **re-pulled by merge date on 2026-09-18** and confirmed. The
first pass listed pull requests by last update, which makes any count a floor: a stale pull
request someone touched last week displaces a merged one. Sorting the closed list on
`merged_at` gives the same 21 non-dependabot merges and the same 4 dependabot merges in the
window, so the dials below stand. Over the repository's whole life — 40 non-dependabot
merges from 2026-03-06 to 2026-09-09 — the rate is 6.5 a month, close enough to the
windowed 7 that the cap does not move.

| Quantity                                             | Measured                         | How                                                                                                                                                           |
| ---------------------------------------------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Merged PRs per month, excluding dependabot           | **7**                            | 21 merged to `main` in the 3-month window, listed by `merged_at`                                                                                              |
| Merged PRs per month, including dependabot           | 8.3                              | 4 dependabot merges in the window                                                                                                                             |
| Review threads per PR, on the 7 reviewed PRs sampled | 1, 6, 14, 15, 22, 23, 24         | `pull_request_read` / `get_review_comments` on #50, #58, #41, #37, #48, #43, #27                                                                              |
| Authors in that sample                               | 2 of 5                           | `mmaclay` and `mwatwood-cu`; `medley56`, `c-poling` and `hcronk` also merge here. Too narrow — see Calibration                                                |
| Median threads that asked for a change               | **11**                           | Excluding acknowledgements, answered questions and praise                                                                                                     |
| Distinct concerns raised more than once              | **17**                           | 14 admitted as rules, 3 held as candidates                                                                                                                    |
| Share a linter could have caught                     | ~10%                             | 10 of ~105 sampled threads, 9 of them in one PR                                                                                                               |
| Unit lane wall clock                                 | **27.6 s**                       | 1019 tests; commands in `standards/test-lanes.md`                                                                                                             |
| PR lane wall clock                                   | **71.2 s**                       | 1091 tests; commands in `standards/test-lanes.md`                                                                                                             |
| Where work originates                                | LIBSDC Jira, written by the team | 112 issues closed or updated in 180 days; ops opens a ticket when a flight procedure changes an ObsID name                                                    |
| Repositories sharing this vocabulary                 | **6**                            | curryer, libera_utils, libera_rad, libera_cam, libera_analysis, CSDS                                                                                          |
| What already states a convention                     | 8 files                          | `.github/instructions/*.instructions.md` (2), `copilot-instructions.md`, `CLAUDE.md`, `GEMINI.md`, `doc/source/developer-docs/{testing,git,build_release}.md` |

One measurement is worth more than its row. The three largest reviews in the window took
**37, 45 and 83 days** from open to merge (#48, #41, #27). The cost here is not the number
of review comments, it is how long a pull request sits between them.

## Derived settings

| Dial                 | Value                                                                              | Derivation                                                                                                                                                                                                                                  |
| -------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Rule cap             | **14 rules, 300 lines**                                                            | 2 × 7 merged PRs a month, floor 12, ceiling 40                                                                                                                                                                                              |
| New-rule cluster     | **2 occurrences across 2 PRs**                                                     | A quarter of the monthly count, floor 2                                                                                                                                                                                                     |
| Provisional expiry   | **20 merged PRs** ≈ 3 months                                                       | Volume, never the calendar                                                                                                                                                                                                                  |
| Decline-rate trigger | Over 1/3 across 3 or more firings                                                  | Default; nothing measured yet                                                                                                                                                                                                               |
| Never-fired trigger  | 6 months                                                                           | Default; long enough that a release-only rule survives                                                                                                                                                                                      |
| Ratchet cadence      | **Monthly, or 10 merged PRs**                                                      | At 7 a month the calendar fires first; the count catches a busy month                                                                                                                                                                       |
| Finding budget       | **7 must-fix and should-fix, 7 suggestions**                                       | 2 × median 11 exceeds the ceiling, so the ceiling binds                                                                                                                                                                                     |
| Reviewer rounds      | **3**                                                                              | The PR lane runs in 71 s, comfortably under five minutes                                                                                                                                                                                    |
| Loop wall-clock exit | **25 minutes**                                                                     | _Not_ three suite runs. Three PR-lane runs is 3.5 minutes, so the suite is not what bounds this loop — the agent's own rounds are. 25 minutes is roughly what four full rounds cost end to end; revise it once the first exits are recorded |
| Gate 0 (contract)    | **On**                                                                             | Other repositories import this one                                                                                                                                                                                                          |
| Plan approval        | A second reader for M and L, and for any public-signature change whatever the size | Downstream consumers                                                                                                                                                                                                                        |

## The five questions

**1. What costs the most time?** Review latency. The comment counts are high but the days
are higher, and a pull request that sits for six weeks is re-reviewed from scratch every
time someone returns to it. Phases 0, 1 and 3 are in scope now: the corpus, the ticket, and
a PR body that says where to look. Phase 2 comes next, because a change that arrives already
through the gates is one that does not bounce. Phase 4 starts when the log has ten records.

**2. Who reads this code, and who depends on it?** It is a shared library. `libera_rad`,
`libera_cam`, `libera_analysis` and CSDS import it, and it is published on PyPI for L2
algorithm developers outside the team. That makes it the shared-library archetype: the
contract is the product, a silent contract break is the expensive failure, gate 0 is on, and
`terminology.md` is the highest-value file in the corpus. Ripple matters more than
duplication — a ticket here forces tickets in the consuming repositories, which is why
`loop-situate` looks outside this repository.

**3. How many pull requests a month merge with a review?** Seven. Every threshold above
scales off that number and none of them was inherited.

**4. What already exists?** Eight files already state a convention, plus a pre-commit
configuration and a ruff select list. Phase 0 here was restructuring, not archaeology, which
is why the rules could start at the cap rather than at ten.

**5. What may leave the repository?** **This repository is public**, and `standards/` does
not ship: a `poetry build` on this branch puts only `libera_utils/` in the wheel, and only
LICENSE, README, `pyproject.toml` and PKG-INFO beside it in the sdist. Confirmed 2026-09-18,
so nothing here reaches a PyPI consumer. It is still readable by anyone with the repository. No internal Confluence
or Jira URL and no internal document content goes into source, docstrings, tests, or
anything under `standards/`. That is R-014, and it is why the shared corpus — which may
carry internal links, because it is private — lives in `libera_llm_tooling` and is named
from here rather than copied in. Decided before Phase 0, deliberately: retrofitting it
would mean re-reading every file in the corpus.

## Calibration, before anyone tags a finding

Two live reviewer passes exist, on PR #73 (`mwatwood-cu`) and PR #49 (`mmaclay`), both merged
2026-09-08, written out at `.review/calibration/`. Tagging findings against two September
pull requests would calibrate the reviewer to a fortnight of one part of the repository.

The set grows first, to:

- **five or more merged pull requests**, with review threads that asked for changes;
- **at least two authors other than whoever runs the pass** — `medley56`, `c-poling` and
  `hcronk` all merge here and none of their pull requests has been sampled;
- **at least one merged before the repository adopted AI assistance.** The instruction files
  landed in PR #3, merged 2026-03-12, so anything merged before then is evidence that
  predates an assistant's own suggestions.

Every thread carries who wrote it and whether it was human or AI-drafted. AI-drafted threads
stay in the record and out of the evidence: a rule built from them encodes what a model
tends to say, not what this team asks for. `loop-harvest` draws the sample and marks the
threads; `loop-ratchet` applies the filter and holds a single-author rule at provisional.

## What is not settled

- **Owner and deputy for `standards/`.** A corpus with one owner stops being maintained when
  that person moves on. Name two.
- **The first ratchet slot.** A calendar trigger with no named person and no recurring slot
  is the failure mode this pattern is most prone to in practice.
- **Whether Copilot's automatic PR review is the suggestion tier or replaces the Phase 3
  reviewer.** Today both would run, holding two different standards, which is the drift this
  pattern exists to prevent. The evidence in the table above — 9 of its 10 mechanical
  findings in one PR were line-length complaints that `ruff format` already owns — argues for
  the suggestion tier.
