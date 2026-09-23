# Standards for libera_utils

The review standard this repository holds itself to, and the measurements that set every
number in it. Written 2026-09-17. Re-measure at the tuning pass, not from memory.

**When this corpus and an authoritative source disagree, the source wins.** Open a pull
request to fix the corpus.

The phases this file refers to — 0 set up the standard, 1 define the work, 2 build inside the
gates, 3 review and merge, 4 the monthly ratchet — are the review-loop pattern's, described in
`libera_llm_tooling` and summarised in `AGENTS.md` under "Working a change".

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
window, so the dials below stand. Over the repository's life on GitHub — 40 non-dependabot
merges from 2026-03-06 to 2026-09-09 — the rate is 6.5 a month, close enough to the
windowed 7 that the cap does not move.

| Quantity                                             | Measured                         | How                                                                                                                                                           |
| ---------------------------------------------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Merged PRs per month, excluding dependabot           | **7**                            | 21 merged to `main` in the 3-month window, listed by `merged_at`                                                                                              |
| Merged PRs per month, including dependabot           | 8.3                              | 4 dependabot merges in the window                                                                                                                             |
| Review threads per PR, on the 7 reviewed PRs sampled | 1, 6, 14, 15, 22, 23, 24         | `pull_request_read` / `get_review_comments` on #50, #58, #41, #37, #48, #43, #27                                                                              |
| Pull request authors in the first harvest's sample   | 2                                | `mmaclay` and `mwatwood-cu`. Five in the combined sample: those two plus `medley56`, `c-poling` and `hcronk`                                                  |
| People who wrote the review threads, combined        | **7**                            | Those five, plus `maxineofficial` and `jgristey`, who review but have not authored a sampled PR                                                               |
| Pull requests harvested, both rounds                 | **17 of 40**                     | Non-dependabot merges, listed by `merged_at`                                                                                                                  |
| Review threads read, both rounds                     | ~212                             | ~105 in the first harvest, 107 in the second                                                                                                                  |
| Share of second-harvest threads written by a bot     | **35%**                          | 37 of 107, `copilot-pull-request-reviewer`. None used as rule evidence                                                                                        |
| Median threads that asked for a change               | **11**                           | Excluding acknowledgements, answered questions and praise                                                                                                     |
| Distinct concerns raised more than once              | **21**                           | 14 admitted as rules, 7 held as candidates below the cap                                                                                                      |
| Share a linter could have caught                     | ~10%                             | 10 of the first harvest's ~105 threads. 9 of the 10 were line-length complaints in a single PR, from an automated reviewer, on a repo that disables `E501`    |
| Unit lane wall clock                                 | **70.7 s**                       | 1001 tests, measured on `main` 2026-09-22; commands in `standards/test-lanes.md`                                                                              |
| PR lane wall clock                                   | **357 s** (5 min 57 s)           | 1071 tests, same run. Machine-local and load-sensitive: the same lane measured 102 s on the LIBSDC-703 base, which freezes the kernel fixtures                |
| Where work originates                                | LIBSDC Jira, written by the team | 112 issues closed or updated in 180 days; ops opens a ticket when a flight procedure changes an ObsID name                                                    |
| Repositories sharing this vocabulary                 | **6**                            | curryer, libera_utils, libera_rad, libera_cam, libera_analysis, CSDS                                                                                          |
| What already states a convention                     | 8 files                          | `.github/instructions/*.instructions.md` (2), `copilot-instructions.md`, `CLAUDE.md`, `GEMINI.md`, `doc/source/developer-docs/{testing,git,build_release}.md` |

One measurement is worth more than its row. The three largest reviews in the window took
**37, 45 and 83 days** from open to merge (#48, #41, #27). The cost here is not the number
of review comments, it is how long a pull request sits between them.

## Derived settings

| Dial                  | Value                                                                                                   | Derivation                                                                                                                                                                                                                                                                                                                    |
| --------------------- | ------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Rule cap              | **14 rules, 300 lines**                                                                                 | 2 × 7 merged PRs a month, floor 12, ceiling 40                                                                                                                                                                                                                                                                                |
| New-rule cluster      | **2 occurrences across 2 PRs**                                                                          | A quarter of the monthly count, floor 2                                                                                                                                                                                                                                                                                       |
| Provisional expiry    | **20 merged PRs with records** ≈ 3 months                                                               | Volume, never the calendar. Counted from the rule's admission; a merged PR with no record is no evidence either way and does not count                                                                                                                                                                                        |
| Decline-rate trigger  | Over 1/3 across 3 or more firings                                                                       | Default; nothing measured yet                                                                                                                                                                                                                                                                                                 |
| Never-fired trigger   | 6 months                                                                                                | Default; long enough that a release-only rule survives                                                                                                                                                                                                                                                                        |
| Ratchet cadence       | **Monthly, or 10 merged PRs**                                                                           | At 7 a month the calendar fires first; the count catches a busy month                                                                                                                                                                                                                                                         |
| Finding budget        | **7 must-fix and should-fix, 7 suggestions**                                                            | 2 × median 11 exceeds the ceiling, so the ceiling binds                                                                                                                                                                                                                                                                       |
| Reviewer rounds       | **3**                                                                                                   | Held at 3, but the reason it was set there no longer holds: the PR lane takes 5 min 57 s on `main`, not 71 s, so three rounds can cost ~18 minutes of suite time alone. The first ratchet decides whether the rounds or the budget moves                                                                                      |
| Loop wall-clock exit  | **60 minutes**, provisional                                                                             | A first guess from one run: PR #66, three reviewer rounds, took about 45 minutes, and three PR-lane runs alone take about 18 at 357 s each. It replaced 25 minutes, which was set when the lane looked like 71 s and would have ended every real run `timed out`. The first ratchet with recorded exits sets it from evidence |
| Gate 0 (contract)     | **On**                                                                                                  | Other repositories import this one                                                                                                                                                                                                                                                                                            |
| Split before starting | **Complexity 8 or above**                                                                               | The Fibonacci estimate the ticket already carries. A first cut, mapped from the S/M/L it replaces, and expected to move once a few tickets are behind us                                                                                                                                                                      |
| Group ticket session  | **Complexity 5 or above**, and every epic                                                               | Below that a ticket is well enough defined that group design time costs more than it returns; its author runs `loop-situate` and `loop-plan` alone and posts the plan for async approval                                                                                                                                      |
| Plan approval         | A second reader at **complexity 5 or above**, and for any public-signature change whatever the estimate | Downstream consumers                                                                                                                                                                                                                                                                                                          |

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

Numbers in the table above that a machine can re-derive are checked by
`.github/scripts/check_measurements.py`, which runs on a pull request touching `standards/`
or `tests/`. It re-collects each lane, checks `rules.md` against its own cap and each record
against the record cap, and reports what no longer holds. Wall clock is deliberately not
checked: it is machine-local, and the same lane has measured 102 s and 357 s on one machine.
Remeasuring it stays a person's job at the ratchet.

The rest of the table is a record of a past harvest rather than a live measurement, and does
not change unless someone harvests again.

## How these rules were generated

Two harvests, both reading merged pull request review threads through the GitHub MCP server,
read-only.

|                              | First harvest                     | Second harvest                                      |
| ---------------------------- | --------------------------------- | --------------------------------------------------- |
| Date                         | 2026-09-17                        | 2026-09-19                                          |
| Pull requests                | 7                                 | 10                                                  |
| Numbers                      | #27, #37, #41, #43, #48, #50, #58 | #2, #4, #12, #15, #22, #28, #30, #32, #42, #60      |
| Threads read                 | ~105                              | 107                                                 |
| Pull request authors         | 2                                 | 5 across both rounds                                |
| People who wrote the threads | 2                                 | 7 across both rounds                                |
| Window                       | 2026-06 to 2026-09                | 2026-03 to 2026-09, the whole GitHub-hosted history |

Together that is **17 of the 40 non-dependabot pull requests merged on GitHub**, listed by
`merged_at` rather than by last update. The review threads in them were written by
`medley56`, `mwatwood-cu`, `mmaclay`, `c-poling`, `hcronk`, `maxineofficial` and `jgristey`,
and by `copilot-pull-request-reviewer`. The other 23 were not read, so those seven are the
reviewer population of the sample rather than of the repository.

Also read, for terminology and decisions rather than rules: the instruction files,
`pyproject.toml`, `.pre-commit-config.yaml`, `doc/source/developer-docs/`, the product
definition YAML in `libera_utils/data/`, and the Confluence pages and Jira epics named in
`standards/context/`.

### What the second harvest changed

The first harvest read two people's pull requests and produced fourteen rules. The second
read three more authors and two more reviewers, and found that three of those rules were
being asked for in pull requests the first sample never saw. The bar is two accepted
citations in pull requests **written by two different people**, and all three clear it on
authors rather than only on reviewers: R-006 cites four pull requests by three authors, R-009
three by two, R-012 five pull requests by
three authors. All three are marked **established**. It also found one rule nobody but its original
author had ever asked for, R-008, which retired to make room for **R-015**, the strongest
single finding of either harvest: seven requests in one pull request, and three pull requests
in total, to narrow a parameter's type rather than widen the function.

Seven candidates sit below the cap with their evidence, ranked — three from the first
harvest and four from the second. The first to promote is
"an error message says what went wrong and what to do next", which four people asked for in
three pull requests.

### One third of the review comments were written by a bot

`copilot-pull-request-reviewer` wrote **37 of the 107 threads** in the second harvest — 12 of
18 in PR #2, 8 of 17 in PR #4, 6 of 12 in PR #30. It has been reviewing since the first pull
request this repository ever merged: PR #1, merged 2026-03-06, has three review threads and
all three are the bot's.

That has two consequences and they pull in opposite directions.

It means **no pull request reachable through the GitHub MCP predates assistant review**, so
the requirement to include one cannot be met from this sample.

It is a limit of what can be read rather than an absence. This repository moved to GitHub in
March 2026; its git history carries 83 merge commits titled `Pull request #N`, from
2021-11-22 to 2026-02-26, on the previous host. Those genuinely predate assistant review and
none of them has been read, because their threads are not reachable through the GitHub MCP.
Reaching them is an open question for the team, not a closed door.

It also means the authorship marker is doing real work rather than guarding against a
hypothetical. None of the 37 bot threads was used as rule evidence. Read in bulk they are a
recognisable and narrow kind: docstrings disagreeing with signatures, stale references in
docs, an APID in a comment that does not match the enum, two typos. Useful, mostly accepted,
and almost entirely mechanical — which is an argument for a check, not for a rule. What the
bot almost never produced is the kind of finding the humans produced constantly: this
parameter should not accept that type, this error message is useless to the person who will
read it, this helper has one caller.

The team already marks its own AI use in-thread — "Reply drafted with AI assistance (Claude
Code) and reviewed by me" — which is what makes the marker answerable at all (D-013).

## Calibration, before anyone tags a finding

Two live reviewer passes exist, on PR #73 (`mwatwood-cu`) and PR #49 (`mmaclay`), both merged
2026-09-08, written out at `.review/calibration/`. Tagging findings against two September
pull requests would calibrate the reviewer to a fortnight of one part of the repository.

The rule _harvest_ is now wide — 17 pull requests, seven reviewers, the whole GitHub history,
recorded above. The reviewer _calibration_ is not: that still means running the reviewer and
having a person tag what it produced, and it has happened twice. The set grows first, to:

- **five or more merged pull requests**, with review threads that asked for changes;
- **at least two authors other than whoever runs the pass** — `medley56`, `c-poling` and
  `hcronk` all merge here and none of their pull requests has been sampled;
- **at least one merged before the repository adopted AI assistance, which nothing reachable
  satisfies.** `copilot-pull-request-reviewer` reviewed PR #1, the first pull request on
  GitHub. The 83 that merged on the previous host do predate it, and filling this slot means
  reaching them. Until then the authorship marker carries the weight, and every rule's
  evidence line is human threads only.

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
  pattern exists to prevent. The evidence is the "share a linter could have
  caught" row above, and it argues for
  the suggestion tier.
