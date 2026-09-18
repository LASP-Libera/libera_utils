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

Two more blocks, parsed from the PR body's gate 3b lines, so the ratchet can see whether
changes are being reworked or padded:

```yaml
tests:
  added: 2
  reworked: 5
  failure_path_assertions: 3
  uncovered_changed_lines: [] # file:line, empty is the healthy case
helpers:
  added: 1 # each with its call-site count
  extracted: 2
```

Empty today. The three calibration runs described in `standards/README.md` are the first
records this directory is expecting.
