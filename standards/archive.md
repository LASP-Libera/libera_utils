# Archive index

One line per rule that has left `rules.md`. This file exists so a check is never orphaned:
someone who hits a failing check follows the ID here, and here to the reasoning.

The full entries live in `libera_llm_tooling/standards/archive/libera_utils/`, and a copy is
published to Confluence for readers who do not read repositories. They are not in this
repository because this repository is public (R-014) and an archive entry says what went
wrong, on which mission, and what the team decided about it.

A rule keeps its entry in `rules.md` for one ratchet cycle after it leaves, then drops to a
line here. IDs are never reused.

| ID    | The rule, in a clause                                | End state              | Became                                                     | Entry                           |
| ----- | ---------------------------------------------------- | ---------------------- | ---------------------------------------------------------- | ------------------------------- |
| R-010 | Deferred work carries a LIBSDC or CURRYER ticket tag | graduated at admission | `lasp/prevent-dangling-todos` in `.pre-commit-config.yaml` | `archive/libera_utils/R-010.md` |

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
