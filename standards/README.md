# Standards for libera_utils

The review standard this repository holds itself to, and the measurements that set every
number in it. Written 2026-09-17. Re-measure at the tuning pass, not from memory.

**When this corpus and an authoritative source disagree, the source wins.** Open a pull
request to fix the corpus.

The phases this file refers to — 0 set up the standard, 1 define the work, 2 build and verify,
3 review and merge, 4 the monthly ratchet — are the standards workflow's, described in
`libera_llm_tooling`'s README and summarized in `AGENTS.md` under "Working a change".

## What is here

| File                 | Holds                                                                                                       |
| -------------------- | ----------------------------------------------------------------------------------------------------------- |
| `review-rules.md`    | The ledger of the rules a reviewer checks, within a 300-line budget; the wording is in the instruction file |
| `review-contract.md` | How the reviewer behaves: severity, finding criteria, what not to flag                                      |
| `decisions.md`       | Decisions local to this repository. Cross-repository ones are shared                                        |
| `test-lanes.md`      | Every lane marker, path and command the corpus depends on, in one place                                     |

`.review/` is the agents' scratch directory and is gitignored.

The review records, one per reviewed pull request, and the monthly revision reports are not
here. They live in `libera_llm_tooling/standards/libera_utils/log/`, the shared clone beside this
repository, so a record never sits in a pull request's diff and committing one never touches a
code branch. "Shared tier" below says how to reach it.

## Shared tier

Three of this corpus's files are not here. They are the same in every Libera repository and
live in `libera_llm_tooling/standards/`:

| File             | Why it is shared                                                                                                                                                                                  |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `terminology.md` | One vocabulary across curryer, utils, rad, cam, analysis and CSDS. A term corrected once should reach every repository at once                                                                    |
| `decisions.md`   | Decisions that constrain more than one repository: ObsID ownership, dependency pinning, the upstream-first rule                                                                                   |
| `context/`       | Confluence pages, Jira epics and PR threads. **It may carry internal links because that repository is private. This one is public (R-014), so it names context entries rather than copying them** |

**The review records** live there too, in `standards/libera_utils/log/`: one `pr-NNNN.yaml`
per reviewed pull request and one report per monthly revision. `pr-findings` writes a record
there and a person commits it in `libera_llm_tooling`, so no record sits in this repository's
pull requests.

### Reaching it

Two separate things, installed two different ways.

The **procedure** is the private `libera-tools` Claude Code plugin: the skills and agents
that plan, build and review, the standards skills (`draft-standard`, `ticket-draft`,
`pr-findings`, `revise-standard`) and the build hooks. Once per machine,
covering every repository you open, from a clone of `libera_llm_tooling` beside this
repository:

```bash
cd ../libera_llm_tooling
./bootstrap.sh            # installs libera-tools@libera
./bootstrap.sh --check    # checks the install and this repository's setup
```

Adding the tooling repository to `permissions.additionalDirectories` does not install anything: that
setting grants read access, and Claude Code discovers skills only from `~/.claude/skills`, a
repository's own `.claude/skills`, and installed plugins.

The **corpus** is a clone kept beside this repository, so that `libera_llm_tooling/standards/`
is a sibling of `libera_utils/`. That convention is the whole configuration. The monthly
revision writes to it on a branch, which is why it stays a clone rather than travelling inside
the plugin.

A skill that cannot find the shared corpus says so and continues without the vocabulary,
rather than inventing terms. It does not fall back to a copy, because a copy is how two
repositories end up disagreeing about what a footprint is.

### Why not a submodule

A submodule pins the vocabulary to a commit per repository, which is exactly the wrong
property for a vocabulary: a corrected term should reach every repository at once, the way
the procedure does. The cost is that a checkout without `libera_llm_tooling` beside it has
no shared tier, which the skills report rather than work around.

## How a rule lives

Tiers say where a rule lives. `prose` is the instruction file only: the author is expected to
know it, the reviewer does not check it, and it has no ledger entry. `reviewer` is the
instruction file plus an entry in `review-rules.md`, which the reviewer checks against.
`check` is the tool's configuration plus the shared archive: the configuration is the rule, the
archive entry in `archive/libera_utils/` holds its reasoning, and its ledger entry is a pointer
so the reviewer knows the ground is covered. Statuses: `provisional` · `established` ·
`graduated` · `retired`. Every entry carries its own status line; read that rather than
assuming.

A rule becomes `established` when the reviewer has cited it and a person has accepted the
finding in two different pull requests, **written by two different people**. During v0 a
harvest may establish a rule on review-thread evidence that clears the same bar, and its
evidence line points at those pull requests. A rule whose evidence is one author's pull
requests, or comes only from AI-drafted review comments, stays provisional however often it is
cited: the citation count measures how often something came up, and breadth measures whether
it is the team's standard or one person's. A `provisional` rule that has become neither
`established` nor `graduated` within the provisional expiry below is retired by default.

Where a rule restates a decision, shared or local, its evidence line names it. A rule and a
decision on the same concern must not disagree about status: the decision is what the team
settled, the rule is how a reviewer checks it, and the reviewer loads both.

v0 evidence points at the merged pull request whose review threads produced the rule, and a
rule resting on one author or one pull request says so in its evidence line. From the first
ratchet on, evidence points at the review records in
`libera_llm_tooling/standards/libera_utils/log/`.

## Graduated

Rules that graduated out of the reviewer and into a tool. Each entry names the rule it
replaced, so the rule's wording can be deleted from the instruction file and the history stays
legible.

A rule graduates when a check catches every accepted instance in the evidence window with no
false positive on `main`. The ratchet drafts the check; an ordinary pull request lands it;
the same ratchet marks the rule `graduated` and writes its reasoning to the archive.

**Every check carries a header** naming the rule ID it replaced, the archive entry that
holds the reasoning, and one sentence on what it catches. That is the pointer at the point
of use, and it is all that may go into this repository — the reasoning itself lives in the
private shared corpus, because R-014 keeps internal discussion out of a public repository.
The pointer goes in the check's configuration, never in a test or in source: R-009 keeps
history out of comments, and a regression test's name says what it guards rather than which
finding produced it.

The number of rules the reviewer holds should be flat or falling over a year while the
number of checks grows. If two consecutive ratchet reports propose no graduations, the rules
being written are not the mechanical kind, and the workflow is delivering a second opinion
rather than a smaller job.

| Rule                                        | Check                                                                       | Where                                                                   |
| ------------------------------------------- | --------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| R-010 · Deferred work carries a ticket tag  | `lasp/prevent-dangling-todos`, tags `LIBSDC,CURRYER`                        | `.pre-commit-config.yaml` · archive `libera_utils/R-010.md`             |
| R-012 · The version bump matches the change | `check_version.py`: heading equals version; a bump is above the highest tag | `.github/workflows/version-check.yml` · archive `libera_utils/R-012.md` |

### Already checked by tools, and therefore never rules

The tools' own configuration is the list: `pyproject.toml` (`[tool.ruff]`) and
`.pre-commit-config.yaml`. It is not copied here or into `review-contract.md`, whose
do-not-flag section points at the same files, so enabling or disabling a check never leaves a
stale copy for the reviewer to follow.

## Archived

One line per rule whose reasoning has been archived — retired, expired, rewritten, or
graduated into a check. This section exists so a check is never orphaned:
someone who hits a failing check follows the ID here, and here to the reasoning.

The full entries live in `libera_llm_tooling/standards/archive/libera_utils/`, and a copy is
published to Confluence for readers who do not read repositories. They are not in this
repository because this repository is public (R-014) and an archive entry says what went
wrong, on which mission, and what the team decided about it.

A rule's entry in `review-rules.md` becomes a stub, and its wording leaves the instruction
file, in the pull request that removes it, which
merges after the archive entry's pull request in `libera_llm_tooling`: a
retired rule keeps its heading and a sentence saying where the reasoning went, a graduated
rule keeps a `check`-tier pointer at the tool that now enforces it. The prose moves to the
entry, not to this section, which holds one line. IDs are never reused.

**No rule leaves without its reasoning being kept.** Whoever graduates, retires or rewrites a
rule writes its biography to the entry: the text as it read, why it was admitted, every
decline reason quoted, and what the replacement cannot catch. During v0 a harvest may propose
retirements as well as admissions, as it did for R-008, with a person making the call; after
v0 proposing them is the ratchet's job alone.

| ID    | The rule, in a clause                                | End state                        | Became                                                     | Entry                           |
| ----- | ---------------------------------------------------- | -------------------------------- | ---------------------------------------------------------- | ------------------------------- |
| R-008 | Test scaffolding does not ship in the package        | retired 2026-09-19               | `pyproject.toml` `packages`                                | `archive/libera_utils/R-008.md` |
| R-010 | Deferred work carries a LIBSDC or CURRYER ticket tag | graduated at admission           | `lasp/prevent-dangling-todos` in `.pre-commit-config.yaml` | `archive/libera_utils/R-010.md` |
| R-012 | The version bump matches the change                  | graduated 2026-09-29             | `.github/workflows/version-check.yml`                      | `archive/libera_utils/R-012.md` |
| R-015 | The annotation says what the code actually accepts   | rewritten 2026-09-22, 2026-09-23 | R-015, reworded in place                                   | `archive/libera_utils/R-015.md` |

### The archive is a lookup, not a graveyard

This is what makes the entries worth writing. Before admitting a new rule from a finding
cluster, the ratchet searches this index for the same concern. Three outcomes, all useful:

- **It graduated.** The concern is already enforced mechanically, so what the team is seeing
  is a gap in the check, not a missing rule. Fix the check.
- **It was retired for a low accept rate.** The team tried this and disagreed with it more
  often than not. Admitting it again unchanged repeats an experiment whose result is written
  down; if something has changed, the new rule's evidence line has to say what.
- **It expired unfired.** Cheap to admit again, but the entry says how long it sat idle last
  time — which is the argument for making it a check straight away rather than a reviewer
  rule.

Without this, a small team re-litigates the same three conventions every eighteen months,
usually once the person who remembered the reasoning has moved to another mission.

## Measured, 2026-09-17

Window 2026-06-15 to 2026-09-15 unless stated. Source: the GitHub MCP server for pull
requests, the Atlassian MCP server for Jira, and a local run for test timings.

The pull-request counts were **re-pulled by merge date on 2026-09-18** and confirmed. The
first pass listed pull requests by last update, which makes any count a floor: a stale pull
request someone touched last week displaces a merged one. Sorting the closed list on
`merged_at` gives the same 21 non-dependabot merges and the same 4 dependabot merges in the
window, so the dials below stand. Over the repository's life on GitHub — 40 non-dependabot
merges from 2026-03-06 to 2026-09-09 — the rate is 6.5 a month, close enough to the
windowed 7 that the dials derived from it do not move.

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
| Median threads that asked for a change               | **11**                           | Excluding acknowledgments, answered questions and praise                                                                                                                                                    |
| Distinct concerns raised more than once              | **21**                           | 14 admitted as rules, 7 held as candidates; re-judged against the criteria on 2026-09-30, 1 met them and was admitted as R-017                                                                              |
| Share a linter could have caught                     | ~10%                             | 10 of the first harvest's ~105 threads. 9 of the 10 were line-length complaints in a single PR, from an automated reviewer, on a repo that disables `E501`                                                  |
| Unit lane wall clock                                 | **70.7 s**                       | 1004 tests, measured on `main` 2026-09-22 at 1001, plus the three `check_version` tests; commands in `standards/test-lanes.md`                                                                              |
| PR lane wall clock                                   | **446 s** (7 min 26 s)           | 1098 tests, same run, on 2026-09-25, at 1095 plus the three `check_version` tests. Machine-local and load-sensitive: the same lane measured 102 s on the LIBSDC-703 base, which freezes the kernel fixtures |
| Where work originates                                | LIBSDC Jira, written by the team | 112 issues closed or updated in 180 days; ops opens a ticket when a flight procedure changes an ObsID name                                                                                                  |
| Repositories sharing this vocabulary                 | **7**                            | curryer, libera_utils, libera_rad, libera_cam, libera_analysis, CSDS, and libera_cdk (private)                                                                                                              |
| What already states a convention                     | 8 files                          | `.github/instructions/*.instructions.md` (2), `copilot-instructions.md`, `CLAUDE.md`, `GEMINI.md`, `doc/source/developer-docs/{testing,git,build_release}.md`                                               |

One measurement is worth more than its row. The three largest reviews in the window took
**37, 45 and 83 days** from open to merge (#48, #41, #27). The cost here is not the number
of review comments, it is how long a pull request sits between them.

## Derived settings

| Dial                  | Value                                                                                                                                                            | Derivation                                                                                                                                                                                                                                                             |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Rule budget           | **300 lines**, the ledger and the wording together                                                                                                               | What one reviewer can hold; the count is whatever meets the criteria                                                                                                                                                                                                   |
| New-rule cluster      | **2 occurrences across 2 PRs**                                                                                                                                   | A quarter of the monthly count, floor 2                                                                                                                                                                                                                                |
| Provisional expiry    | **20 merged PRs with records** ≈ 3 months                                                                                                                        | Volume, never the calendar. Counted from the rule's admission; a merged PR with no record is no evidence either way and does not count                                                                                                                                 |
| Decline-rate trigger  | Over 1/3 across 3 or more firings                                                                                                                                | Default; nothing measured yet                                                                                                                                                                                                                                          |
| Never-fired trigger   | 6 months                                                                                                                                                         | Default; long enough that a release-only rule survives                                                                                                                                                                                                                 |
| Ratchet cadence       | **Monthly, or 10 merged PRs**                                                                                                                                    | At 7 a month the calendar fires first; the count catches a busy month                                                                                                                                                                                                  |
| Finding criteria      | **Must-fix and should-fix always**, when keyed as the contract's Citation section says and carrying a Proposed line; **at most 7 suggestions**; questions always | A finding worth escalating is never dropped for a count; nits are the build's to catch. Suggestions keep a ceiling because 2 × median 11 exceeds it                                                                                                                    |
| Reviewer rounds       | **5**, upstream's                                                                                                                                                | The implementation reviewer's own cap in `implement-change`, with a stop when one finding survives two fixes. Not a local dial: changing it is an upstream change. The PR lane takes 5 min 57 s on `main`, so five rounds can cost half an hour of suite time          |
| Wall clock            | Measured, not enforced                                                                                                                                           | The build hooks record it in the PR body's "already checked". The one recorded run, PR #66, took about 45 minutes; budgets of 25 and then 60 minutes were tried and dropped, because a wall-clock exit ends a converging run for a reason that is not about the change |
| Contract tests first  | **On**                                                                                                                                                           | Other repositories import this one; `AGENTS.md` asks for the accept, return and raise tests before the implementation when a public signature changes                                                                                                                  |
| Split before starting | **Estimate 8 or above**                                                                                                                                          | The team's Fibonacci story-point scale, estimated in the session by `ticket-draft` and not written to the ticket. A first cut, expected to move once a few tickets are behind us                                                                                       |
| Group ticket session  | **Estimate 5 or above**, and every epic                                                                                                                          | Below that a ticket is well enough defined that group design time costs more than it returns; its author runs `ticket-draft` and `implement-change` alone and posts the plan for async approval                                                                        |
| Plan approval         | A second reader at **estimate 5 or above**, and for any public-signature change whatever the estimate                                                            | Downstream consumers                                                                                                                                                                                                                                                   |

## The five questions

**1. What costs the most time?** Review latency. The comment counts are high but the days
are higher, and a pull request that sits for six weeks is re-reviewed from scratch every
time someone returns to it. Phases 0, 1 and 3 are in scope now: the corpus, the ticket, and
a PR body that says where to look. Phase 2 comes next, because a change that arrives already
verified is one that does not bounce. Phase 4 starts when the log has ten records.

**2. Who reads this code, and who depends on it?** It is a shared library. `libera_rad`,
`libera_cam`, `libera_analysis`, `libera_cdk` (private) and CSDS import it, and it is published on PyPI for L2
algorithm developers outside the team. That makes it the shared-library archetype: the
contract is the product, a silent contract break is the expensive failure, contract tests come first, and
`terminology.md` is the highest-value file in the corpus. Ripple matters more than
duplication — a ticket here forces tickets in the consuming repositories, which is why
`ticket-draft` looks outside this repository.

**3. How many pull requests a month merge with a review?** Seven. Every threshold above
scales off that number and none of them was inherited.

**4. What already exists?** Eight files already state a convention, plus a pre-commit
configuration and a ruff select list. Phase 0 here was restructuring, not archaeology, which
is why fourteen rules met the criteria at the first harvest rather than a handful.

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
file) against their line budget with the live rule count beside it, and reports what no longer holds. The record cap lives with the
records, in the shared clone. Wall clock is deliberately not
checked: it is machine-local, and the same lane has measured 102 s and 357 s on one machine.
Remeasuring it stays a person's job at the ratchet.

The rest of the table is a record of a past harvest rather than a live measurement, and does
not change unless someone harvests again.

The harvest history (how these rules were generated, which review threads a bot wrote, and
the calibration runs) is in the shared corpus at `libera_utils/harvest.md`.

## What is not settled

- **A rotating driver for the monthly revision, and its slot in a meeting we already have.**
- **The first ratchet slot.** A calendar trigger with no named person and no recurring slot
  is the failure mode this pattern is most prone to in practice.
- **Whether Copilot's automatic PR review is the suggestion tier or replaces the Phase 3
  reviewer.** Today both would run, holding two different standards, which is the drift this
  pattern exists to prevent. The evidence is the "share a linter could have
  caught" row above, and it argues for
  the suggestion tier.
