Closes #NNN · LIBSDC-NNNN · Outcome: <the ticket's outcome line>

Loop-exit: clean

<!--
LIBSDC is the tracker of record. Keep both refs when the work has a GitHub issue too.

Loop-exit is one line and must be present. Exactly one of:
    Loop-exit: clean | capped | flapping | timed out | halted
written by loop-self-review, saying how the build loop ended. capped, flapping and timed out
are not failures to hide: they say a bound was hit and a person should look. halted means an
existing test was weakened rather than the code fixed, and it also goes on the line above
everything else.

If the loop did not run, replace that line with a reason and add the skip-loop label:
    Skip-loop: <why>

A pull request carrying neither line is one nobody can tell about, which is the whole point
of the line. A CI check that enforces its presence is parked, not abandoned -- the format is
fixed now so the check is a one-line grep later.
-->

## In context

Function: <which function of this repository the change touches, in the repository's own words>
Problem: <what that function cannot do or does wrong, with the failure, the number and its unit,
or the ticket>
Options: <chosen> (chosen) · <alternative> (why it lost) · <alternative> (why it lost)
Decision: <what was chosen and why, in terms of the function>
Consequences: <what the repository can do now that it could not; what is harder or now committed;
any accepted limitation, stated plainly>

## Look at this

- <every escalation, every design choice the plan left open with what was chosen and why,
  every departure from the approved plan with its reason>

<!-- When there is nothing: "Nothing outside the plan." and nothing else. -->

## How to verify

| #   | Criterion | Test | Status |
| --- | --------- | ---- | ------ |
| 1   |           |      |        |

Run: `<the one command>`

## What changed

- <files grouped by purpose, one line each>
- Public signatures: <added or altered, or "none">

<details><summary>Already checked</summary>

Authored: <agent-assisted | by hand>, opened by <handle>
Gates: contract N · lint N · types N · tests N · reviewer rounds N · exit <clean|capped|flapping|halted|timed out> · wall clock N min
Plan: <approved by handle | skimmed by handle | none (S ticket)>
Refinement: N questions answered · N constraints added by hand · N terms flagged · plan <amended|not amended>
Tests: N added · M reworked · K failure-path assertions · uncovered changed lines: <none | file:line, ...>
Helpers: N added (call sites each) · M extracted from existing code
Rules checked: <R-ids>
Declined: <R-id at path::symbol — one-sentence reason>
Suggestions: <R-id at path::symbol — what was suggested>
Questions: <open questions for the reviewer, or "none">
Escalated: <what, or "none">

</details>
