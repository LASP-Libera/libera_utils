# Standards for libera_utils

The review standard this repository holds itself to, and the measurements that set every
number in it. Written 2026-09-17. Re-measure at the tuning pass, not from memory.

**When this corpus and an authoritative source disagree, the source wins.** Open a pull
request to fix the corpus.

The phases this file refers to — 0 set up the standard, 1 define the work, 2 build and verify,
3 review and merge, 4 the monthly ratchet — are the standards workflow's, described in
`libera_llm_tooling`'s README and summarised in `AGENTS.md` under "Working a change".

## What is here

| File                 | Holds                                                                                           |
| -------------------- | ----------------------------------------------------------------------------------------------- |
| `review-rules.md`    | The ledger of the rules a reviewer checks, capped at 14; the wording is in the instruction file |
| `review-contract.md` | How the reviewer behaves: severity, budget, what not to flag                                    |
| `decisions.md`       | Decisions local to this repository. Cross-repository ones are shared                            |
| `SHARED.md`          | Where the shared vocabulary, decisions and context live, and how to reach them                  |
| `checks/`            | Rules that graduated into a tool, each naming the rule it replaced                              |
| `archive.md`         | One line per rule that has left the ledger, with where its reasoning lives                      |
| `test-lanes.md`      | Every lane marker, path and command the corpus depends on, in one place                         |

`.review/` is the agents' scratch directory and is gitignored.

The review records, one per reviewed pull request, and the monthly revision reports are not
here. They live in `libera_llm_tooling/standards/libera_utils/log/`, the shared clone beside this
repository, so a record never sits in a pull request's diff and committing one never touches a
code branch. `SHARED.md` says how to reach it.

## Measured, 2026-09-17

Window 2026-06-15 to 2026-09-15 unless stated. Source: the GitHub MCP server for pull
requests, the Atlassian MCP server for Jira, and a local run for test timings.

The pull-request counts were **re-pulled by merge date on 2026-09-18** and confirmed. The
first pass listed pull requests by last update, which makes any count a floor: a stale pull
request someone touched last week displaces a merged one. Sorting the closed list on
`merged_at` gives the same 21 non-dependabot merges and the same 4 dependabot merges in the
window, so the dials below stand. Over the repository's life on GitHub — 40 non-dependabot
merges from 2026-03-06 to 2026-09-09 — the rate is 6.5 a month, close enough to the
windowed 7 that the cap does not move.

| Quantity                                             | Measured                         | How                                                                                                                                                                                                         |
| ---------------------------------------------------- | -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Merged PRs per month, excluding dependabot           | **7**                            | 21 merged to `main` in the 3-month window, listed by `merged_at`                                                                                                                                            |
| Merged PRs per month, including dependabot           | 8.3                              | 4 dependabot merges in the window                                                                                                                                                                           |
| Review threads per PR, on the 7 reviewed PRs sampled | 1, 6, 14, 15, 22, 23, 24         | `pull_request_read` / `get_review_comments` on #50, #58, #41, #37, #48, #43, #27                                                                                                                            |
| Pull request authors in the first harvest's sample   | 2                                | `mmaclay` and `mwatwood-cu`. Five in the combined sample: those two plus `medley56`, `c-poling` and `hcronk`                                                                                                |
| People who wrote the review threads, combined        | **7**                            | Those five, plus `maxineofficial` and `jgristey`, who review but have not authored a sampled PR                                                                                                             |
| Pull requests harvested, both rounds                 | **17 of 40**                     | Non-dependabot merges, listed by `merged_at`                                                                                                                                                                |
| Review threads read, both rounds                     | ~212                             | ~105 in the first harvest, 107 in the second                                                                                                                                                                |
| Share of second-harvest threads written by a bot     | **35%**                          | 37 of 107, `copilot-pull-request-reviewer`. None used as rule evidence                                                                                                                                      |
| Median threads that asked for a change               | **11**                           | Excluding acknowledgements, answered questions and praise                                                                                                                                                   |
| Distinct concerns raised more than once              | **21**                           | 14 admitted as rules, 7 held as candidates below the cap                                                                                                                                                    |
| Share a linter could have caught                     | ~10%                             | 10 of the first harvest's ~105 threads. 9 of the 10 were line-length complaints in a single PR, from an automated reviewer, on a repo that disables `E501`                                                  |
| Unit lane wall clock                                 | **70.7 s**                       | 1004 tests, measured on `main` 2026-09-22 at 1001, plus the three `check_version` tests; commands in `standards/test-lanes.md`                                                                              |
| PR lane wall clock                                   | **446 s** (7 min 26 s)           | 1098 tests, same run, on 2026-09-25, at 1095 plus the three `check_version` tests. Machine-local and load-sensitive: the same lane measured 102 s on the LIBSDC-703 base, which freezes the kernel fixtures |
| Where work originates                                | LIBSDC Jira, written by the team | 112 issues closed or updated in 180 days; ops opens a ticket when a flight procedure changes an ObsID name                                                                                                  |
| Repositories sharing this vocabulary                 | **6**                            | curryer, libera_utils, libera_rad, libera_cam, libera_analysis, CSDS                                                                                                                                        |
| What already states a convention                     | 8 files                          | `.github/instructions/*.instructions.md` (2), `copilot-instructions.md`, `CLAUDE.md`, `GEMINI.md`, `doc/source/developer-docs/{testing,git,build_release}.md`                                               |

One measurement is worth more than its row. The three largest reviews in the window took
**37, 45 and 83 days** from open to merge (#48, #41, #27). The cost here is not the number
of review comments, it is how long a pull request sits between them.

## Derived settings

| Dial                  | Value                                                                                                 | Derivation                                                                                                                                                                                                                                                             |
| --------------------- | ----------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Rule cap              | **14 rules, 300 lines**                                                                               | 2 × 7 merged PRs a month, floor 12, ceiling 40                                                                                                                                                                                                                         |
| New-rule cluster      | **2 occurrences across 2 PRs**                                                                        | A quarter of the monthly count, floor 2                                                                                                                                                                                                                                |
| Provisional expiry    | **20 merged PRs with records** ≈ 3 months                                                             | Volume, never the calendar. Counted from the rule's admission; a merged PR with no record is no evidence either way and does not count                                                                                                                                 |
| Decline-rate trigger  | Over 1/3 across 3 or more firings                                                                     | Default; nothing measured yet                                                                                                                                                                                                                                          |
| Never-fired trigger   | 6 months                                                                                              | Default; long enough that a release-only rule survives                                                                                                                                                                                                                 |
| Ratchet cadence       | **Monthly, or 10 merged PRs**                                                                         | At 7 a month the calendar fires first; the count catches a busy month                                                                                                                                                                                                  |
| Finding budget        | **7 must-fix and should-fix, 7 suggestions**                                                          | 2 × median 11 exceeds the ceiling, so the ceiling binds                                                                                                                                                                                                                |
| Reviewer rounds       | **5**, upstream's                                                                                     | The implementation reviewer's own cap in `implement-change`, with a stop when one finding survives two fixes. Not a local dial: changing it is an upstream change. The PR lane takes 5 min 57 s on `main`, so five rounds can cost half an hour of suite time          |
| Wall clock            | Measured, not enforced                                                                                | The build hooks record it in the PR body's "already checked". The one recorded run, PR #66, took about 45 minutes; budgets of 25 and then 60 minutes were tried and dropped, because a wall-clock exit ends a converging run for a reason that is not about the change |
| Contract tests first  | **On**                                                                                                | Other repositories import this one; `AGENTS.md` asks for the accept, return and raise tests before the implementation when a public signature changes                                                                                                                  |
| Split before starting | **Estimate 8 or above**                                                                               | The team's Fibonacci story-point scale, estimated in the session by `ticket-draft` and not written to the ticket. A first cut, expected to move once a few tickets are behind us                                                                                       |
| Group ticket session  | **Estimate 5 or above**, and every epic                                                               | Below that a ticket is well enough defined that group design time costs more than it returns; its author runs `ticket-draft` and `implement-change` alone and posts the plan for async approval                                                                        |
| Plan approval         | A second reader at **estimate 5 or above**, and for any public-signature change whatever the estimate | Downstream consumers                                                                                                                                                                                                                                                   |

## The five questions

**1. What costs the most time?** Review latency. The comment counts are high but the days
are higher, and a pull request that sits for six weeks is re-reviewed from scratch every
time someone returns to it. Phases 0, 1 and 3 are in scope now: the corpus, the ticket, and
a PR body that says where to look. Phase 2 comes next, because a change that arrives already
verified is one that does not bounce. Phase 4 starts when the log has ten records.

**2. Who reads this code, and who depends on it?** It is a shared library. `libera_rad`,
`libera_cam`, `libera_analysis` and CSDS import it, and it is published on PyPI for L2
algorithm developers outside the team. That makes it the shared-library archetype: the
contract is the product, a silent contract break is the expensive failure, contract tests come first, and
`terminology.md` is the highest-value file in the corpus. Ripple matters more than
duplication — a ticket here forces tickets in the consuming repositories, which is why
`ticket-draft` looks outside this repository.

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

Numbers in the table above that a machine can re-derive are checked by
`.github/scripts/check_measurements.py`, which runs on a pull request touching `standards/`
or `tests/`. It re-collects each lane, checks the rules (the ledger and the wording in the instruction
file) against their cap, and reports what no longer holds. The record cap lives with the
records, in the shared clone. Wall clock is deliberately not
checked: it is machine-local, and the same lane has measured 102 s and 357 s on one machine.
Remeasuring it stays a person's job at the ratchet.

The rest of the table is a record of a past harvest rather than a live measurement, and does
not change unless someone harvests again.

The harvest history (how these rules were generated, which review threads a bot wrote, and
the calibration runs) is in the shared corpus at `libera_utils/harvest.md`.

## What is not settled

- **Owner and deputy for `standards/`.** A corpus with one owner stops being maintained when
  that person moves on. Name two.
- **The first ratchet slot.** A calendar trigger with no named person and no recurring slot
  is the failure mode this pattern is most prone to in practice.
- **Whether Copilot's automatic PR review is the suggestion tier or replaces the Phase 3
  reviewer.** Today both would run, holding two different standards, which is the drift this
  pattern exists to prevent. The evidence is the "share a linter could have
  caught" row above, and it argues for
  the suggestion tier.
