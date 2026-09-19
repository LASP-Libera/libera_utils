# Log

One record per reviewed pull request, `pr-NNNN.yaml`, at most 50 lines, written by the
`loop-review` skill from the tags a person gave its findings — not typed by hand. One report
per ratchet, `ratchet-YYYY-MM.md`.

This directory is the ratchet's entire input, and the only mandatory artifact of a review.
If records stop appearing, the next ratchet report says so on its first line, and a missing
record count above a third of merged pull requests is itself a trigger.

A record holds: the PR and ticket, the branch, the date, the `standards/` git sha the
reviewer ran against, who adjudicated, the plan status, the refinement counts, the inner
loop's gates and exit, every finding with its key, severity and verdict (with a reason for
every decline), and the questions with their answers.

Four more fields, parsed from the PR body's "already checked" section. The first two come
from the gate 3b lines and let the ratchet see whether changes are being reworked or padded;
the last two say who wrote the change and how often review sent it back:

```yaml
tests:
  added: 2
  reworked: 5
  failure_path_assertions: 3
  uncovered_changed_lines: [] # file:line, empty is the healthy case
helpers:
  added: 1 # each with its call-site count
  extracted: 2
authored: agent-assisted # agent-assisted | by hand; plus the handle that opened it
rework_rounds:
  - round: 1
    sent_back_by: R-004 # a rule id or a finding key
    applied: 2
    declined: 1 # each with the reason the writer gave
```

`rework_rounds` is empty on a pull request that merged on its first review, which is the
healthy case. A finding key that appears there repeatedly is the strongest graduation
evidence the ratchet gets: gate 4 passed, a person still had to ask.

Write each finding as a **YAML flow mapping**, one entry over one or two lines. Block style
costs five lines a finding, and seven findings at the budget plus the header does not fit in
fifty — `pr-0066.yaml` is block style rewritten as flow, 53 lines down to 30.

`pr-0066.yaml` is the first record. The two calibration runs described in `standards/README.md`
— PR #73 and PR #49 — are still not records, because nobody has tagged them.
