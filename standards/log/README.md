# Log

One record per reviewed pull request, `pr-NNNN.yaml`, at most 140 lines, written by the
`pr-record` skill from the verdicts people gave the `pr-findings` comment — not typed by hand.
One report per ratchet, `ratchet-YYYY-MM.md`.

This directory is the ratchet's entire input, and the only mandatory artifact of a review. If
records stop appearing, the next ratchet report says so on its first line, and a missing
record count above a third of merged pull requests is itself a trigger. Only pull requests
opened after `standards/` reached `main` count: one opened before had no standard to be
reviewed against, and counting it would make every early ratchet open on a false alarm.

A record holds: the PR and ticket, the branch, the date, the `standards/` git sha the reviewer
ran against, who adjudicated, the plan status, the build's gate counts, its tokens
and `Build-exit`, every finding with its key, severity and verdict (with a reason for every
decline), and the questions with their answers. A finding's `key` is the full finding key
`review-contract.md` defines, such as `R-012/pyproject.toml::version`, and `at:` holds the
line. `pr-0066.yaml` predates the full key and carries the citation alone.

`pr-record` writes a record and never commits it; a person does. A commit that changes
nothing outside this directory does not count as a new head, so committing a record does not
trigger a delta review. Which branch a record is committed to is not yet settled
(`standards/README.md`).

A record built from a file rather than a pull request thread carries `tagged: in-file` and
`reviewed_after_merge: true` — calibration and backfill, where the code had already shipped
and no finding could have changed it. The ratchet counts those separately from findings a
person acted on before merge.

A finding two reviewers answered differently carries `contested: true` alongside the verdict
the author settled on, and both original lines. The ratchet counts those per rule: a rule
people keep disagreeing about is usually one whose do-not-flag sentence is wrong.

Four more fields, parsed from the PR body's "already checked" section. The first two come
from the `Tests:` and `Helpers:` lines and let the ratchet see whether changes are being
reworked or padded;
the last two say who wrote the change and how often review sent it back:

```yaml
tests:
  added: 2
  reworked: 5 # modified: the same test name on both sides of the diff
  deleted: 1
  failure_path_assertions: 3
  uncovered_changed_lines: [] # file:line, empty is the healthy case
helpers:
  added: 1 # each with its call-site count
  extracted: 2
authored: agent-assisted # agent-assisted | by hand
author: mmaclay # the handle that opened the pull request
rework_rounds:
  - round: 1
    commits: a71d5f3..952f547 # the range the round covers
    sent_back_by: R-004 # a rule id or a finding key
    applied: 2
    declined: 1 # each with the reason the writer gave
```

`rework_rounds` is empty on a pull request that merged on its first review, which is the
healthy case. A record is updated, never duplicated, when the branch moves: the next
`pr-record` run after a delta review moves `head` to the new commit, re-parses the gate and
test fields from the PR body, and appends the round. A record whose `head` is behind the
branch at merge is missing whatever happened after it. A finding key that appears there
repeatedly is the strongest graduation evidence the ratchet gets: the build's reviewer passed,
a person still had to ask.

**The cap is 140 because prettier decides the shape, not the schema.** The first record was
written at 53 lines of block YAML against an original cap of 50, rewritten as flow mappings to
fit at 30, and the `prettier` pre-commit hook expanded it straight back to block at 66. Fighting the
formatter would mean excluding this directory from a hook the whole repository runs, which is
a worse trade than a bigger number.

A finding costs seven lines once prettier has formatted it. The header, a rework round and the
questions take about 35, and a pull request that goes through a delta review carries a second
budget of seven findings on top of the first, so 35 + 14 × 7 is 133. The cap was 80 until
`pr-0066.yaml` reached 78 after its first delta, with the delta's findings not yet answered.
A record nearing 140 has been through more than one delta, which is worth the ratchet's
attention in itself.

`pr-0066.yaml` is the first record. The two calibration runs described in `standards/README.md`
— PR #73 and PR #49 — are still not records, because nobody has tagged them.
