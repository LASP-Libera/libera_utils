# Archive index

One line per rule whose reasoning has been archived — retired, expired, rewritten, or
graduated into a check. This file exists so a check is never orphaned:
someone who hits a failing check follows the ID here, and here to the reasoning.

The full entries live in `libera_llm_tooling/standards/archive/libera_utils/`, and a copy is
published to Confluence for readers who do not read repositories. They are not in this
repository because this repository is public (R-014) and an archive entry says what went
wrong, on which mission, and what the team decided about it.

A rule's entry in `rules.md` becomes a stub in the same pull request that archives it: a
retired rule keeps its heading and a sentence saying where the reasoning went, a graduated
rule keeps a `check`-tier pointer at the tool that now enforces it. The prose moves to the
entry, not to this file, which holds one line. IDs are never reused.

| ID    | The rule, in a clause                                | End state              | Became                                                     | Entry                           |
| ----- | ---------------------------------------------------- | ---------------------- | ---------------------------------------------------------- | ------------------------------- |
| R-008 | Test scaffolding does not ship in the package        | retired 2026-09-19     | `pyproject.toml` `packages`                                | `archive/libera_utils/R-008.md` |
| R-010 | Deferred work carries a LIBSDC or CURRYER ticket tag | graduated at admission | `lasp/prevent-dangling-todos` in `.pre-commit-config.yaml` | `archive/libera_utils/R-010.md` |
| R-015 | The annotation says what the code actually accepts   | rewritten 2026-09-22   | R-015, reworded in place                                   | `archive/libera_utils/R-015.md` |

## The archive is a lookup, not a graveyard

This is what makes the entries worth writing. Before admitting a new rule from a finding
cluster, the ratchet searches this index for the same concern. Three outcomes, all useful:

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
