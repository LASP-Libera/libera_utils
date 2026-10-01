# Standards for libera_utils

The review standard this repository holds itself to. Written 2026-09-17. Re-measure at the
monthly revision, not from memory.

**When this corpus and an authoritative source disagree, the source wins.** Open a pull
request to fix the corpus.

## What is here

| File                 | Holds                                                                                                       |
| -------------------- | ----------------------------------------------------------------------------------------------------------- |
| `review-rules.md`    | The ledger of the rules a reviewer checks, within a 300-line budget; the wording is in the instruction file |
| `review-contract.md` | How the reviewer behaves: severity, finding criteria, what not to flag                                      |
| `decisions.md`       | Decisions local to this repository. Cross-repository ones are shared                                        |
| `test-lanes.md`      | Every lane marker, path and command the corpus depends on, in one place                                     |
| `README.md`          | The settings the tools read, how a rule lives, what graduated and what was archived                         |

The review rules are the code-level, diff-checkable half of what the team's reviews said, and
their count is not the measure of the corpus. The design of the mission and its software, what
a function is for, what a name means, what was settled and why, lives in `decisions.md`, the
shared `terminology.md` and `context/`, and enters through the ticket session. A design
constraint becomes a rule only when a diff can be checked against it, at a revision, with
evidence.

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
| `terminology.md` | One vocabulary across curryer, utils, rad, cam, analysis, cdk and CSDS. A term corrected once should reach every repository at once                                                               |
| `decisions.md`   | Decisions that constrain more than one repository: ObsID ownership, dependency pinning, the upstream-first rule                                                                                   |
| `context/`       | Confluence pages, Jira epics and PR threads. **It may carry internal links because that repository is private. This one is public (R-014), so it names context entries rather than copying them** |

File names keep "ratchet" from the first design: each revision's report is
`ratchet-YYYY-MM.md`.

**The review records** live there too, in `standards/libera_utils/log/`: one `pr-NNNN.yaml`
per reviewed pull request and one report per monthly revision. `pr-findings` writes a record
there and the month's driver, a role that rotates, runs the monthly revision and commits that
month's records in `libera_llm_tooling`, so no record sits in this repository's pull requests.

### Reaching it

Two separate things, installed two different ways.

The **procedure** is the private `libera-tools` Claude Code plugin: the skills and agents
that plan, build and review, the standards skills (`draft-standard`, `ticket-draft`,
`pr-findings`, `revise-standard`) and the build hooks. Once per machine,
covering every repository you open, from a clone of `libera_llm_tooling` beside this
repository:

```bash
cd ../libera_llm_tooling
./bootstrap.sh --github standards/shared-corpus-v0   # installs libera-tools from GitHub
./bootstrap.sh --check    # checks the install and this repository's setup
```

The plugin README's Install section has the rest, including the one-time switch once
libera_llm_tooling#3 merges.

Adding the tooling repository to `permissions.additionalDirectories` does not install anything: that
setting grants read access, and Claude Code discovers skills only from `~/.claude/skills`, a
repository's own `.claude/skills`, and installed plugins.

The **corpus** is a clone kept beside this repository, so that `libera_llm_tooling/standards/`
is a sibling of `libera_utils/`. That convention is the whole configuration. The monthly
revision writes to it on a branch, which is why it stays a clone rather than traveling inside
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

A rule is admitted, as `provisional`, on one merged pull request with a review thread written by
a person, when it meets the other criteria in `review-rules.md`. It becomes `established` when
the reviewer has cited it and a person has accepted the finding in two different pull requests,
**written by two different people**. A harvest may establish a rule on review-thread evidence
that clears the same bar, and its
evidence line points at those pull requests. A rule whose evidence is one author's pull
requests, or comes only from AI-drafted review comments, stays provisional however often it is
cited: the citation count measures how often something came up, and breadth measures whether
it is the team's standard or one person's. A `provisional` rule that has become neither
`established` nor `graduated` within the provisional expiry below is retired by default. An
evidence line reading "needs a second" names what the rule still lacks for establishment.

Where a rule restates a decision, shared or local, its evidence line names it. A rule and a
decision on the same concern must not disagree about status: the decision is what the team
settled, the rule is how a reviewer checks it, and the reviewer loads both.

Harvest evidence points at the merged pull request whose review threads produced the rule, and a
rule resting on one author or one pull request says so in its evidence line. From the first
revision on, evidence points at the review records in
`libera_llm_tooling/standards/libera_utils/log/`.

## Graduated

Rules that graduated out of the reviewer and into a tool. Each entry names the rule it
replaced, so the rule's wording can be deleted from the instruction file and the history stays
legible.

A rule graduates when a check catches every accepted instance in the evidence window with no
false positive on `main`. The monthly revision drafts the check; an ordinary pull request lands it;
the same revision marks the rule `graduated` and writes its reasoning to the archive.

**Every check carries a header** naming the rule ID it replaced, the archive entry that
holds the reasoning, and one sentence on what it catches. That is the pointer at the point
of use, and it is all that may go into this repository — the reasoning itself lives in the
private shared corpus, because R-014 keeps internal discussion out of a public repository.
The pointer goes in the check's configuration, never in a test or in source: R-009 keeps
history out of comments, and a regression test's name says what it guards rather than which
finding produced it.

The number of rules the reviewer holds should be flat or falling over a year while the
number of checks grows. If two consecutive revision reports propose no graduations, the rules
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
decline reason quoted, and what the replacement cannot catch. A harvest may propose
retirements as well as admissions, as the second harvest did for R-008, with a person making
the call; once the monthly revision runs, proposing them is its job alone.

| ID    | The rule, in a clause                                | End state                        | Became                                                     | Entry                           |
| ----- | ---------------------------------------------------- | -------------------------------- | ---------------------------------------------------------- | ------------------------------- |
| R-008 | Test scaffolding does not ship in the package        | retired 2026-09-19               | `pyproject.toml` `packages`                                | `archive/libera_utils/R-008.md` |
| R-010 | Deferred work carries a LIBSDC or CURRYER ticket tag | graduated at admission           | `lasp/prevent-dangling-todos` in `.pre-commit-config.yaml` | `archive/libera_utils/R-010.md` |
| R-012 | The version bump matches the change                  | graduated 2026-09-29             | `.github/workflows/version-check.yml`                      | `archive/libera_utils/R-012.md` |
| R-015 | The annotation says what the code actually accepts   | rewritten 2026-09-22, 2026-09-23 | R-015, reworded in place                                   | `archive/libera_utils/R-015.md` |

### The archive is a lookup, not a graveyard

This is what makes the entries worth writing. Before admitting a new rule from a finding
cluster, the revision searches this index for the same concern. Three outcomes, all useful:

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

## Settings

The dials the skills and the checks read; where each came from is in the private corpus at
`libera_utils/harvest.md`.

| Dial                  | Value                                                                                                                                                            | Derivation                                                                                                                                                                                                                                                                       |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Rule budget           | **300 lines**, the ledger and the wording together                                                                                                               | What one reviewer can hold; the count is whatever meets the criteria                                                                                                                                                                                                             |
| New-rule admission    | **1 merged PR with a review thread written by a person**                                                                                                         | The admission bar in `review-rules.md`; establishment needs 2 merged PRs by 2 authors                                                                                                                                                                                            |
| Provisional expiry    | **20 merged PRs with records** ≈ 3 months                                                                                                                        | Volume, never the calendar. Counted from the rule's admission; a merged PR with no record is no evidence either way and does not count                                                                                                                                           |
| Decline-rate trigger  | Over 1/3 across 3 or more firings                                                                                                                                | Default; nothing measured yet                                                                                                                                                                                                                                                    |
| Never-fired trigger   | 6 months                                                                                                                                                         | Default; long enough that a release-only rule survives                                                                                                                                                                                                                           |
| Revision cadence      | **Monthly, or when ten records are waiting, whichever comes first**                                                                                              | At 7 a month the calendar fires first; the count catches a busy month                                                                                                                                                                                                            |
| Finding criteria      | **Must-fix and should-fix always**, when keyed as the contract's Citation section says and carrying a Proposed line; **at most 7 suggestions**; questions always | A finding worth escalating is never dropped for a count; nits are the build's to catch. Suggestions keep a ceiling because 2 × median 11 exceeds it                                                                                                                              |
| Reviewer rounds       | **5**, upstream's                                                                                                                                                | The implementation reviewer's own cap in `implement-change`, with a stop when one finding survives two fixes. Not a local dial: changing it is an upstream change. At the PR lane's 446 s (2026-09-25, in `test-lanes.md`), five rounds can cost over half an hour of suite time |
| Wall clock            | Measured, not enforced                                                                                                                                           | The build hooks record it in the PR body's "already checked". The one recorded run, PR #66, took about 45 minutes; budgets of 25 and then 60 minutes were tried and dropped, because a wall-clock exit ends a converging run for a reason that is not about the change           |
| Contract tests first  | **On**                                                                                                                                                           | Other repositories import this one; `AGENTS.md` asks for the accept, return and raise tests before the implementation when a public signature changes                                                                                                                            |
| Split before starting | **Estimate 8 or above**                                                                                                                                          | The team's Fibonacci story-point scale, estimated in the session by `ticket-draft` and not written to the ticket. A first cut, expected to move once a few tickets are behind us                                                                                                 |
| Group ticket session  | **Estimate 5 or above**, and every epic                                                                                                                          | Below that a ticket is well enough defined that group design time costs more than it returns; its author runs `ticket-draft` and `implement-change` alone and posts the plan for async approval                                                                                  |
| Plan approval         | A second reader at **estimate 5 or above**, and for any public-signature change whatever the estimate                                                            | Downstream consumers                                                                                                                                                                                                                                                             |

Numbers that a machine can re-derive are checked by `.github/scripts/check_rule_budget.py`.
On a pull request touching `standards/` it checks the rules (the ledger and the wording in the
instruction file) against the rule budget in the table above, with the live rule count beside
it, and reports what no longer holds. The lane counts are in `standards/test-lanes.md` and move
with every test added, so the revision runs it by hand with `--lanes`, which re-collects each
lane as well. The record cap lives with the records, in the shared clone. Wall clock is
deliberately not checked: it is machine-local, and the same lane has measured 102 s and 357 s
on one machine. Remeasuring it stays a person's job at the revision.

The harvest history (how these rules were generated, which review threads a bot wrote, and
the calibration runs) is in the shared corpus at `libera_utils/harvest.md`.
