# The shared tier

Three of this corpus's files are not here. They are the same in every Libera repository and
live in `libera_llm_tooling/standards/`:

| File             | Why it is shared                                                                                                                                                                                  |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `terminology.md` | One vocabulary across curryer, utils, rad, cam, analysis and CSDS. A term corrected once should reach every repository at once                                                                    |
| `decisions.md`   | Decisions that constrain more than one repository: ObsID ownership, dependency pinning, the upstream-first rule                                                                                   |
| `context/`       | Confluence pages, Jira epics and PR threads. **It may carry internal links because that repository is private. This one is public (R-014), so it names context entries rather than copying them** |

## Reaching it

Two separate things, installed two different ways.

The **procedure** — the seven `loop-*` skills and the `loop-reviewer` agent — is a private
Claude Code plugin. Once per machine, covering every repository you open:

```bash
claude plugin marketplace add LASP-Libera/libera_llm_tooling
claude plugin install libera-loops@libera
```

`claude plugin details libera-loops` should report seven skills and one agent. Adding the
tooling repository to `permissions.additionalDirectories` does not install anything: that
setting grants read access, and Claude Code discovers skills only from `~/.claude/skills`, a
repository's own `.claude/skills`, and installed plugins.

The **corpus** is a clone kept beside this repository, so that `libera_llm_tooling/standards/`
is a sibling of `libera_utils/`. That convention is the whole configuration. The monthly
ratchet writes to it on a branch, which is why it stays a clone rather than travelling inside
the plugin.

A skill that cannot find the shared corpus says so and continues without the vocabulary,
rather than inventing terms. It does not fall back to a copy, because a copy is how two
repositories end up disagreeing about what a footprint is.

## Why not a submodule

A submodule pins the vocabulary to a commit per repository, which is exactly the wrong
property for a vocabulary: a corrected term should reach every repository at once, the way
the procedure does. The cost is that a checkout without `libera_llm_tooling` beside it has
no shared tier, which the skills report rather than work around.
