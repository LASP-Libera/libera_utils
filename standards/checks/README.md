# Checks

Rules that graduated out of the reviewer and into a tool. Each entry names the rule it
replaced, so the rule's wording can be deleted from the instruction file and the history stays
legible.

A rule graduates when a check catches every accepted instance in the evidence window with no
false positive on `main`. The ratchet drafts the check; an ordinary pull request lands it;
the same ratchet marks the rule `graduated` and writes its reasoning to the archive.

**Every file here carries a header** naming the rule ID it replaced, the archive entry that
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

**A `check`-tier rule lives in the tool's configuration plus the shared archive**, not in the
instruction file: the configuration is the rule, the archive entry holds its reasoning, and
its entry in `review-rules.md` is a pointer so the reviewer knows the ground is covered. The
other tiers: `prose` is the instruction file only, and `reviewer` is the instruction file plus
a ledger entry.

## Graduated so far

| Rule                                       | Check                                                | Where                                                       |
| ------------------------------------------ | ---------------------------------------------------- | ----------------------------------------------------------- |
| R-010 · Deferred work carries a ticket tag | `lasp/prevent-dangling-todos`, tags `LIBSDC,CURRYER` | `.pre-commit-config.yaml` · archive `libera_utils/R-010.md` |

## Already checked by tools, and therefore never rules

The tools' own configuration is the list: `pyproject.toml` (`[tool.ruff]`) and
`.pre-commit-config.yaml`. It is not copied here or into `review-contract.md`, whose
do-not-flag section points at the same files, so enabling or disabling a check never leaves a
stale copy for the reviewer to follow.
