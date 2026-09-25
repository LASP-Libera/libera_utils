Closes #NNN · LIBSDC-NNNN · Driving requirement: <the ticket's, in one line>

Build-exit: <clean | capped | flapping | blocked | halted>

<!--
LIBSDC is the tracker of record. Keep both refs when the work has a GitHub issue too.

Build-exit is one line and must be present. Exactly one of:
    Build-exit: clean | capped | flapping | blocked | halted
saying how the build's review loop ended, from the implementation reviewer's last verdict in
implement-change. clean: SATISFIED. capped: five review rounds without SATISFIED. flapping: the
same finding survived two fixes. blocked: a check could not run for a reason outside the
change, such as a pre-existing failure, a module that will not import or an absent system
dependency; the line names it. halted: the build stopped for a person, because an existing
test was weakened rather than the code fixed or a finding would change the agreed plan; say
which on the line above everything else. capped, flapping and blocked are not failures to
hide: they say a bound was hit and a person should look.

If the change was not built through implement-change, replace that line with a reason and add
the no-build-gates label:
    No-build-gates: <why>

A pull request carrying neither line is one nobody can tell about, which is the whole point
of the line. Where the repository runs the build-exit check, it annotates a pull request that
carries neither, and fails it once the check is set to blocking.
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

Gates: lint N · types N · tests N · reviewer rounds N · wall clock N min
Tokens since the build started: N in · N out · N cached
Plan: <approved by handle | skimmed by handle | none (below the plan-approval threshold)>
Rules checked: <R-ids>
Declined: <R-id at path::symbol — one-sentence reason>
Suggestions: <R-id at path::symbol — what was suggested>
Questions: <open questions for the reviewer, or "none">
Escalated: <what, or "none">

</details>
