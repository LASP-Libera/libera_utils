# Rules

What a reviewer checks in this repository, beyond what the tools already check. Each rule
carries a tier, a status, the month it was admitted, and the evidence that earned it.

**Cap: 14 rules or 300 lines.** The cap is twice the measured monthly reviewed-PR count
(`standards/README.md`). At the cap, a rule is admitted only by retiring one.

Tiers: `prose` — the author is expected to know it and the reviewer does not check it.
`reviewer` — the reviewer checks it. `check` — a tool checks it and the entry is a pointer.
Statuses: `provisional` · `established` · `graduated` · `retired`.

A rule becomes `established` when the reviewer has cited it and a person has accepted the
finding in two different pull requests. A `provisional` rule that has not done that by the
second ratchet after admission is retired by default. Everything below is `provisional`
except R-010, which a pre-commit hook already enforces.

v0 evidence points at the merged pull request whose review threads produced the rule.
From the first ratchet on, evidence points at `log/pr-NNNN.yaml` records.

---

### R-001 · Validate a name or identifier where it is constructed, not where it is first used

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0048

A class that accepts an invalid value and raises later moves the failure away from the
caller who could fix it. `LiberaGroundCcsdsFilename` accepted day-of-year 999 because the
setter only ran the regex, and the `strptime` round trip that would have caught it did not
run until `archive_prefix` was computed at staging — after ingest had accepted the file.
Validate in the constructor or the setter, and make the regex reject what the parser cannot
parse.
Do not flag: a validator deliberately deferred because it needs data the constructor does
not have, when the docstring says so.

### R-002 · A condition that invalidates the output raises; it does not warn or no-op

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0037, pr-0027, pr-0041

A warning is not a failure. When an Az/El CK had no encoder columns in its L1A input, the
code returned quietly and produced a kernel with nothing in it; it now raises. The rule is
the repository's fail-loud posture in review form: a defined input produces a defined
product, or the run stops, because a crash gets noticed and a silently wrong number gets
published.
Do not flag: a genuinely optional input whose absence has a defined meaning, when the
docstring names it; a `logger.warning` beside a raise, for context.

### R-003 · One exception type per condition, and a predicate returns rather than raises

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0048

`GroundCcsdsApidAbsentError` was raised for four unrelated conditions, only one of which was
an absent APID, so callers could not tell an unparsable APID from a missing one and the name
misled on three of the four. Separately, `is_data_time_indexed_apid()` raised `ValueError`
on an unknown APID, which a question of the form "is this X" should answer with `False`.
Either give each condition its own type, or return the no-answer value the caller can act on.
Do not flag: one exception type covering conditions a caller genuinely handles identically,
when the message distinguishes them.

### R-004 · Every public symbol has a numpydoc docstring, including what it raises

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0041, pr-0027

Numpydoc on public symbols is a project standard and the `Raises` section is the half that
gets left out. With fail-loud design the failure modes are part of the interface, so a
function that raises and does not say so has an undocumented contract. Parameters, units,
frames and epochs belong here too.
Do not flag: private helpers; a one-line docstring on a symbol whose signature is
self-describing and that raises nothing.

### R-005 · A published quantity states its unit; a time states its epoch and frame

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0027, pr-0043

`WFOV_FSW_COMMANDED_EXP_TIME_1/2` and the FPGA actual exposure times shipped with no `units`
attribute, and the conversion to milliseconds is still unconfirmed with FSW. A number in a
data product with no unit is not a measurement, and a consumer will guess. The same applies
to a time with no epoch and a pointing angle with no frame.
Do not flag: dimensionless counters and flags; a field whose unit is stated once for a group
in the product definition.

### R-006 · One source of truth for a value; tabular data lives in a data file

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0027, pr-0041

`PACKET_DATA_WIDTH` restated a width that the `|S972` dtype already carried, so the two
could diverge silently. The ObsID registry started as a large literal inside a module and
became `data/obsid_registry.csv`, read and validated at import, because a table in code
cannot be validated as data and a table in a comment cannot be used at all.
Do not flag: a named constant that gives a meaning to a literal used in one place; a value
duplicated in a test on purpose, so the test fails when the source changes.

### R-007 · Delete dead code rather than leaving it unreferenced

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0027, pr-0048

Three instances in one pull request: a function whose only mention was a comment explaining
why it was not used, a `try`/`except` whose result was discarded and whose branch was no
longer reachable, and three counters that were incremented and never read. This is the kind
of residue LLM-assisted drafting leaves behind, so it is worth a deliberate pass.
Do not flag: a public symbol kept for backwards compatibility with a deprecation note; code
behind a feature flag that the changelog names.

### R-008 · Test scaffolding does not ship in the package

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0027

`wfov_image_metadata.py` carried a function whose docstring said it packed footer fields
"for unit tests", whose only callers were in `tests/`, and which the pipeline never used —
the L1A path only ever decodes a footer, never encodes one. An encoder in the shipped module
is a public surface the team has to keep working. It belongs in a fixture.
Do not flag: a documented round-trip helper the package genuinely exposes to its users.

### R-009 · Comments describe the code as it is, not how it got there

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0037, pr-0058

The most repeated request in the window, six times in one review: remove the ticket number,
remove the historical title, remove the comment that says what this used to be. A test's
subject is the behaviour, not the ticket that asked for it. Ticket references are for
forward-looking work, which is what R-010 covers.
Do not flag: a tagged deferred-work marker such as `TODO[LIBSDC-1234]`, which is
forward-looking; a comment citing an external
document that the code implements.

### R-010 · Deferred work carries a LIBSDC or CURRYER ticket tag

tier: check · status: graduated · since: 2026-09 · evidence: pre-commit, standing convention

A deferred decision with no ticket is a deferred decision nobody will make.
Do not flag: this rule at all. The hook owns it, and a marker in a file the hook excludes
(`.pre-commit-config.yaml`) is deliberately out of scope.
Check: `.pre-commit-config.yaml`, the `lasp/prevent-dangling-todos` hook, with its tags set
to `LIBSDC,CURRYER` and its comment markers to the two it scans for. The reviewer does not
check this; the hook does.

### R-011 · A dependency pins to an immutable ref

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0058

A `@main` ref makes the build non-reproducible and lets an upstream merge break CI with no
commit on this side. This is not hypothetical: a moving ref in `libera_rad` took main and
three pull requests red overnight. Pin to the commit or the tagged release, with a comment
saying why it is pinned and what unpins it. A direct-URL dependency also blocks publishing.
Do not flag: a pin in a local development extra that is never published.

### R-012 · The version bump matches the change, and the changelog heading matches it

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0048, pr-0037, pr-0027

New public modules, a new filename class, a new enum member or a new keyword argument make
a minor release, not a patch — downstream pins of the form `~=5.10.3` will take a patch
silently. And a changelog headed `5.8.5` above a `pyproject.toml` that says `5.8.5rc1`
leaves a reader unable to tell which artifact they have.
Do not flag: a pre-release suffix used deliberately for downstream testing, when the
changelog heading carries it too.

### R-013 · Parse or sort an input once, not once per consumer

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0048, pr-0041, pr-0027

Three shapes of the same mistake: a scan that re-read and re-parsed a whole packet file once
per APID, twelve passes over a 2 MB fixture in the ingest path where real captures are far
larger; a trim loop that re-sorted a full-day dataset and re-read a YAML definition on every
one of ~35 runs; and a stitching path holding three copies of the image data live at once.
Hoist the parse, the sort and the definition load out of the loop.
Do not flag: a repeated read of something small and mutable, where the reread is the point;
a copy that exists to avoid mutating a caller's array.

### R-014 · No internal URL or internal document content in this repository

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0041, pr-0027

`libera_utils` is public and ships to PyPI. Cite an internal document by name — "the FSW
user's guide", "the ICIE ObsID page" — say what it decides, and stop. No Confluence or Jira
URL, no pasted internal content, in source, docstrings, tests or anything under
`standards/`. Links rot as well as leak, so naming the document is also the more durable
pointer. The background that needs a link lives in the private shared corpus.
Do not flag: a LIBSDC ticket key on its own, which is an identifier rather than a link; a
public URL, such as NAIF or the CERES documentation.

---

## Candidates not admitted, because the cap binds

Kept here with their evidence so the ratchet can promote one when a rule retires.

- **Private symbols do not cross module boundaries.** A second consumer outside the defining
  module makes a symbol public in fact; rename and document it, or wrap it. Evidence:
  pr-0048 (`_expand_sample_times`, `_extract_wfov_header_metadata_from_blob`).
- **A registry whose values reach a filename has a uniqueness invariant test.** Evidence:
  pr-0041 (one ObsID on two instruments produced two writes of the same filename).
- **Optional flags are keyword-only.** Evidence: pr-0048 (`ground_data`, `verbose`).
