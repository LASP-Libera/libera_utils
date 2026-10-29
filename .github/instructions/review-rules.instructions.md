---
applyTo: "**"
---

## Review Standards

The rules below are the team's review conventions, numbered `R-NNN` so a review comment can cite
one; they are not Claude Code's `.claude/rules/` files. The Libera standard,
`LASP-Libera/libera_standards`, holds each rule with its status and evidence, and its ledger's
header carries the budget and the admission criteria; `standards/profile.yaml` here names it. This
repository is public, so R-014 applies to everything in it. A change to a rule lands through a
reviewed pull request.

## Review Rules

What a reviewer checks here, beyond what the tools check, and what anyone writing code here
is expected to follow. This file is the copy of each rule's wording that Copilot reads; Claude
Code's libera-tools plugin loads the standard's own, so `CLAUDE.md` does not import this file. The
standard's ledger holds each rule's tier, status, evidence and do-not-flag sentence, and a change
to a rule lands there and here together.

### R-001 · Validate a name or identifier where it is constructed, not where it is first used

An invalid name or identifier is rejected by the constructor or setter that receives it, so the
failure reaches the caller who can fix it rather than a later stage. A pattern that accepts what
the parser cannot parse is part of the same defect.

### R-002 · A condition that invalidates the output raises; it does not warn or no-op

A condition that makes the output wrong or empty raises; a warning, a log line or a quiet return
is not a failure. A defined input produces a defined product or the run stops, because a crash
gets noticed and a silently wrong number gets published.

### R-003 · One exception type per condition, and a predicate returns rather than raises

Each condition a caller may handle differently gets its own exception type, named for that
condition. A predicate, a function that asks "is this X", answers with `False` or another
no-answer value rather than raising.

### R-004 · Every public symbol has a numpydoc docstring, including what it raises

Every public function, class and method has a numpydoc docstring with its parameters and a
`Raises` section for every exception it raises: with fail-loud design the failure modes are part
of the interface. Units, frames and epochs are R-005.

### R-005 · A published quantity states its unit; a time states its epoch and frame

Every quantity that reaches a data product carries its unit, every time its epoch, and every
pointing angle its frame. A number with no unit is not a measurement, and a consumer will guess.

### R-006 · One source of truth for a value; tabular data lives in a data file

A value is defined once and everything else derives from it, so two copies cannot drift. A table
lives in a data file that is read and validated, not as a literal in code or a list in a comment.

### R-007 · Delete dead code rather than leaving it unreferenced

Code nothing calls, a branch nothing reaches and a variable nothing reads are deleted in the
change that makes them dead, not kept with a comment explaining why.

### R-009 · Comments and names describe the code as it is, not how it got there

A comment or a name describes the code as it is, not how it got there or where it is going, and
reads correctly with only the code around it. It carries no ticket number, historical title,
"new" or "old" qualifier, or context from a conversation, a prompt or a review; forward-looking
work takes R-010's tagged marker.

### R-011 · A dependency pins to an immutable ref

A dependency pins to a commit or a tagged release, never a branch, with a comment saying why it is
pinned and what unpins it. A moving ref makes the build non-reproducible and lets an upstream
merge break CI, and a direct-URL dependency also blocks publishing.

### R-012 · The version bump matches the change, and the changelog heading matches it

What the version check cannot judge is the size of a bump: a new public module, filename class,
enum member or keyword argument makes a minor release, not a patch, since downstream pins of the
form `~=5.10.3` take a patch silently. The "2. Development Lifecycle" and "SDC Data Product
Versioning Scheme" pages make a change that may break an API in minor ways, that adds a feature
with new usage patterns, or that users can see in a product a minor release.

### R-013 · Hold one copy of a large input, and pass over it once

A large input is held once, and read, parsed or sorted once with the result handed to each
consumer. A change does not keep a second full copy alive, or redo a full pass per consumer or per
loop iteration: a granule-sized copy doubles peak memory, and a repeated pass multiplies run time.

### R-014 · No internal URL or internal document content in this repository

This repository is public and ships to PyPI. No internal URL, meaning any Confluence or Jira URL on
`lasp.colorado.edu` and anything on the DMZ, and no internal document content goes into source,
docstrings, tests or `standards/`. Cite a page by its title and a ticket by its key, and say what
it decides.

### R-015 · The annotation says what the code actually accepts

A parameter's annotation names exactly what the code handles, so a caller can tell what is
accepted. Where a function is annotated with a type it cannot handle, narrow the annotation rather
than widen the function.

### R-016 · An error message says what went wrong and what to do next

An error message states the condition and the next action, in terms a reader outside the SDC can
act on, rather than naming internals alone.

### R-017 · A valid range or an enumeration cites its source

A valid range or a set of categories names the document, algorithm or upstream definition its
values come from, by name rather than by internal link (R-014). A claim with no source cannot be
checked in review.

### R-018 · Use the code that already does a job, and do not run it twice

Where this repository already does a job, such as opening a product, building a filename,
checking conformance or serializing a model, a change calls that code rather than writing its own,
and does not run a step again that the code it calls already runs. A second version drifts from
the first, and a repeated step hides which run the product depends on.
