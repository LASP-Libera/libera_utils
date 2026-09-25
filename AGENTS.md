# libera_utils

**When this file and an authoritative source disagree, the source wins** — `pyproject.toml`,
`.pre-commit-config.yaml`, the workflows, `standards/review-rules.md` and the code itself. Open a
pull request to fix this file rather than working from it.

Tool-neutral entry point. Claude Code reads `CLAUDE.md`, Copilot reads
`.github/copilot-instructions.md`, Gemini reads `GEMINI.md`; all three resolve to
`.github/instructions/libera-utils.instructions.md`, which points back here for how a change
moves from ticket to merge. A convention is written once, in `standards/`, and everything
else points at it.

## Coding rules, testing, and agent restrictions

`.github/instructions/libera-utils.instructions.md`. Read it before changing anything.

## The review standard

`standards/` holds what a reviewer checks and why:

- `standards/README.md` — the measurements that set every threshold, and the five tuning
  answers. Read this first; it says what this repository is optimising for.
- `standards/review-rules.md` — the ledger of the rules, capped at 14: tier, status, evidence
  and do-not-flag for each. The wording of each rule is in the Review Rules section of
  `.github/instructions/libera-utils.instructions.md`
- `standards/review-contract.md` — how a reviewer behaves, and what it must not flag
- `standards/decisions.md` — convention decisions already made here
- `standards/SHARED.md` — where the shared vocabulary, decisions and context live

**This repository is public.** No internal Confluence or Jira URL and no internal document
content goes into source, docstrings, tests or `standards/`. Cite an internal document by
name (R-014).

## Working a change

A ticket first, in the team's format, as `.github/ISSUE_TEMPLATE/change.yml` or a LIBSDC
description: context, driving requirements, technical requirements, acceptance criteria, out of
scope, considerations.
Then a plan, approved before code. Then the change, verified by a review loop. Then a pull
request body in the shape of `.github/PULL_REQUEST_TEMPLATE.md`, written for the person who
will read it.

The skills come from the `libera-tools` Claude Code plugin in `libera_llm_tooling`;
`standards/SHARED.md` says how to install it. `ticket-draft` writes and places the ticket,
`implement-change` plans, builds and verifies it, `pr-create` opens the pull request,
`pr-findings` reviews it against this standard and, once people reply, writes the record,
and `revise-standard` runs the monthly revision. None of them merges or edits a standard, and
nothing is posted without a person's yes. Everything an agent produces is a proposal.

## Reviewers and builders in this repository

Any agent that plans, builds or reviews a change here, including `implementation-planner`,
`implementation-plan-reviewer`, `implementation-reviewer` and `github-pr-reviewer`:

- Loads `standards/review-rules.md`, the Review Rules section of
  `.github/instructions/libera-utils.instructions.md`, `standards/review-contract.md`,
  `standards/decisions.md`, the
  shared decisions and terminology named in `standards/SHARED.md`, and `standards/test-lanes.md`
  before judging anything. The test and coverage commands come from `test-lanes.md`, never a
  guess.
- When a change adds or alters a public signature, plans and writes what it accepts, returns
  and raises, and tests that fail for those reasons, before the implementation. Other
  repositories import this one.
- Holds the change to the rules and to the contract's Test scrutiny and New surface sections,
  and does not raise anything on the contract's do-not-flag list.
- Keys every finding the way the contract's Citation section says: the citation, `/`, the path,
  `::`, the enclosing symbol. At most seven must-fix and should-fix.
- Stops the build and brings the person the latest report, rather than fixing on, when a fix
  would weaken an existing test (a loosened tolerance or assertion, a removed `pytest.raises`,
  a new skip or xfail, a dropped parametrize case), a fix would change the agreed plan, a
  pre-existing failure blocks a check, or a check cannot run at all.
- Leaves the `Build-exit:` line and the "already checked" section to the plugin:
  `implement-change` records how the build ended in `.review/build/outcome.md`, mapped as
  `.github/PULL_REQUEST_TEMPLATE.md` defines, and `pr-create` copies it and the build hooks'
  counts into the body. A body written by hand follows the template the same way, and opens in
  the order of the shared `templates/brief.md`.

If you are an AI agent opening a pull request here, say so in the body's `Authored:` line.
