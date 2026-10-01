# libera_utils

**When this file and an authoritative source disagree, the source wins** — `pyproject.toml`,
`.pre-commit-config.yaml`, the workflows, `standards/review-rules.md` and the code itself. Open a
pull request to fix this file rather than working from it.

Tool-neutral entry point. Claude Code reads `CLAUDE.md` and Gemini reads `GEMINI.md`, and both
import this file and `.github/instructions/libera-utils.instructions.md`; Copilot reads
`.github/copilot-instructions.md` and applies the instruction file. A convention is written
once and everything else points at it.

## Coding rules, testing, and agent restrictions

`.github/instructions/libera-utils.instructions.md`. Read it before changing anything.

## The review standard

`standards/` holds what a reviewer checks and why:

- `standards/README.md` — the settings the tools read. Read it when a number is questioned, not
  first.
- `standards/review-rules.md` — the ledger of the rules, within a 300-line budget: tier,
  status, evidence and do-not-flag for each. The wording of each rule is in the Review Rules section of
  `.github/instructions/libera-utils.instructions.md`
- `standards/review-contract.md` — how a reviewer behaves, and what it must not flag
- `standards/decisions.md` — convention decisions already made here
- `standards/README.md`, "Shared tier" — where the shared vocabulary, decisions and context live

**This repository is public.** No internal Confluence or Jira URL and no internal document
content goes into source, docstrings, tests or `standards/`. Cite an internal document by
name (R-014).

## Working a change

A ticket first, in LIBSDC, in the team's format: context with its footprint, driving
requirements with technical requirements and acceptance criteria, not this ticket,
considerations when needed, and open questions until the estimation meeting settles them.
Then a plan, approved before code. Then the change, verified by a review loop. Then a pull
request body in the shape of `.github/PULL_REQUEST_TEMPLATE.md`, written for the person who
will read it.

Commit subjects on `main` read `LIBSDC-NNN: <subject>` where there is a ticket, as its history
shows; `doc/source/developer-docs/git.md` sets no stricter standard. A pull request is
squash-merged, which adds the `(#NN)` suffix, so a branch does not need squashing by hand.

The skills come from the `libera-tools` Claude Code plugin in `libera_llm_tooling`;
`standards/README.md` says how to install it, under "Shared tier". `ticket-draft` writes and places the ticket,
`implement-change` plans, builds and verifies it, `pr-create` opens the pull request,
`pr-findings` reviews it against this standard and, once people reply, writes the record
(into `libera_llm_tooling/standards/libera_utils/log/`; a session with no clone beside this
repository falls back to `.review/log/`, and nothing under `.review/` is committed),
and `revise-standard` runs the monthly revision. None of them merges. No agent changes a
standard on its own: a person chooses every change to `standards/`, and it lands through a
reviewed pull request. Nothing is posted without a person's yes. Everything an agent produces is a proposal.

No agent pushes in this repository: the instruction file forbids it. `pr-create`, once a person
says yes to opening a pull request, and `pr-fix`, once the person approves its fixes, stop at
the push, print the command, and continue after the person runs the push. Nothing else pushes.

## Reviewers and builders in this repository

Any agent that plans, builds or reviews a change here, including `implementation-planner`,
`implementation-plan-reviewer`, `implementation-reviewer` and `github-pr-reviewer`:

- Loads `standards/review-rules.md`, the Review Rules section of
  `.github/instructions/libera-utils.instructions.md`, `standards/review-contract.md`,
  `standards/decisions.md`, the
  shared decisions and terminology named in `standards/README.md` ("Shared tier"), and `standards/test-lanes.md`
  before judging anything. The test and coverage commands come from `test-lanes.md`, never a
  guess.
- When a change adds or alters a public signature, plans and writes what it accepts, returns
  and raises, and tests that fail for those reasons, before the implementation. Other
  repositories import this one.
- Holds the change to the rules and to the contract's Test scrutiny and New surface sections,
  and does not raise anything on the contract's do-not-flag list.
- Keys every finding the way the contract's Citation section says: the citation, `/`, the path,
  `::`, the enclosing symbol. Raises every must-fix and should-fix that meets the contract's
  finding criteria, and at most seven suggestions.
- Stops the build and brings the person the latest report, rather than fixing on, when a fix
  would weaken an existing test (a loosened tolerance or assertion, a removed `pytest.raises`,
  a new skip or xfail, a dropped parametrize case), a fix would change the agreed plan, a
  pre-existing failure blocks a check, or a check cannot run at all.
- Leaves the `Build-exit:` line and the "already checked" section to the plugin:
  `implement-change` records how the build ended in `.review/build/outcome.md`, mapped as
  `implement-change` step 8 defines, and `pr-create` copies it and the build hooks'
  counts into the body. A body written by hand follows the template the same way, carries the
  `No-build-gates: <reason>` line where `Build-exit:` would be, and follows the template's
  headings.

If you are an AI agent working here, declare it where shared D-013 says: the `Co-Authored-By`
trailer on commits, the generated-with line on a pull request body, and the attribution line
on comments and replies. Nowhere else.
