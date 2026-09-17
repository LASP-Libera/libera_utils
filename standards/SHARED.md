# The shared tier

Three of this corpus's files are not here. They are the same in every Libera repository and
live in `libera_llm_tooling/standards/`:

| File             | Why it is shared                                                                                                                                                                                  |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `terminology.md` | One vocabulary across curryer, utils, rad, cam, analysis and CSDS. A term corrected once should reach every repository at once                                                                    |
| `decisions.md`   | Decisions that constrain more than one repository: ObsID ownership, dependency pinning, the upstream-first rule                                                                                   |
| `context/`       | Confluence pages, Jira epics and PR threads. **It may carry internal links because that repository is private. This one is public (R-014), so it names context entries rather than copying them** |

## Reaching it

Clone `libera_llm_tooling` somewhere stable, then add to `.claude/settings.local.json`
(gitignored, personal):

```json
{
  "permissions": {
    "additionalDirectories": [
      "/absolute/path/to/libera_llm_tooling/standards",
      "/absolute/path/to/libera_llm_tooling/.github/skills",
      "/absolute/path/to/libera_llm_tooling/.github/agents"
    ]
  }
}
```

Copilot reads the same skills and agents from `.github/skills` and `.github/agents` in that
repository.

A skill that cannot find the shared corpus says so and continues without the vocabulary,
rather than inventing terms. It does not fall back to a copy, because a copy is how two
repositories end up disagreeing about what a footprint is.

## Why not a submodule

A submodule pins the vocabulary to a commit per repository, which is exactly the wrong
property for a vocabulary: a corrected term should reach every repository at once, the way
the procedure does. The cost is that a checkout without `libera_llm_tooling` beside it has
no shared tier, which the skills report rather than work around.
