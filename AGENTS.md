# libera_utils

Tool-neutral entry point. Claude Code reads `CLAUDE.md`, Copilot reads
`.github/copilot-instructions.md`, Gemini reads `GEMINI.md`, and all three point here or at
the same instruction file, so a convention is written once.

## Coding rules, testing, and agent restrictions

`.github/instructions/libera-utils.instructions.md`. Read it before changing anything.

## The review standard

`standards/` holds what a reviewer checks and why:

- `standards/README.md` — the measurements that set every threshold, and the five tuning
  answers. Read this first; it says what this repository is optimising for.
- `standards/rules.md` — the rules, capped at 14, each with its evidence
- `standards/review-contract.md` — how a reviewer behaves, and what it must not flag
- `standards/decisions.md` — convention decisions already made here
- `standards/SHARED.md` — where the shared vocabulary, decisions and context live

**This repository is public.** No internal Confluence or Jira URL and no internal document
content goes into source, docstrings, tests or `standards/`. Cite an internal document by
name (R-014).

## Working a change

A ticket first, in the shape of `.github/ISSUE_TEMPLATE/change.yml` or the Jira description
block: outcome, why, acceptance criteria, non-goals, constraints, open questions, size. Then
a plan, approved before code. Then the change, through the gates. Then a pull request body
in the shape of `.github/PULL_REQUEST_TEMPLATE.md`, written for the person who will read it.

The skills that run each step live in `libera_llm_tooling/.github/skills/` (`loop-plan`,
`loop-self-review`, `loop-review`, `loop-ratchet` and three more); `standards/SHARED.md` says
how to reach them. None of them posts, merges, or edits a standard. Everything an agent
produces is a proposal.

If you are an AI agent opening a pull request here, say so in the body's `Authored:` line.
