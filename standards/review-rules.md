# Review rules

The ledger of what a reviewer checks in this repository beyond what the tools check: each
rule's tier, status, evidence and do-not-flag sentence. The wording of each rule lives once,
under its ID, in `.github/instructions/libera-utils.instructions.md`, and each entry links there.
**Budget: 300 lines**, counted across this file and the wording together, each rule's title
once, its link line not at all, and a `check`-tier rule not at all. A rule is admitted, as
provisional, on one merged pull request with a review thread written by a person, when it is not
something a tool can check, not a one-off design question, and not a bare restatement of a
decision: a rule may implement one when it says what a diff must show and names the decision on
its evidence line, as R-011, R-014, R-015 and R-016 do. It becomes established on
two merged pull requests by two authors; a provisional rule retires at the expiry unless
established. No rule becomes established on authorship until the first revision records it,
since authors were not recorded at the harvest. A rule's wording is two to four lines.
How a rule is admitted, graduates and retires is the tooling's documentation, not this
repository's.

---

### R-001 · Validate a name or identifier where it is constructed, not where it is first used

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0048 (a filename setter accepted day-of-year 999, and the parse that would have caught it ran only at staging, after ingest) · **one pull request, needs a second**

Wording: [R-001 in the instruction file](../.github/instructions/libera-utils.instructions.md#r-001--validate-a-name-or-identifier-where-it-is-constructed-not-where-it-is-first-used)

Do not flag: a validator deliberately deferred because it needs data the constructor does
not have, when the docstring says so.

### R-002 · A condition that invalidates the output raises; it does not warn or no-op

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0037, pr-0027, pr-0041, pr-0060, pr-0012 (in one of these, an Az/El CK whose L1A input had no encoder columns returned quietly with an empty kernel; it now raises)

Wording: [R-002 in the instruction file](../.github/instructions/libera-utils.instructions.md#r-002--a-condition-that-invalidates-the-output-raises-it-does-not-warn-or-no-op)

Do not flag: a genuinely optional input whose absence has a defined meaning, when the
docstring names it; a `logger.warning` beside a raise, for context.

### R-003 · One exception type per condition, and a predicate returns rather than raises

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0048 (one exception type was raised for four unrelated conditions, and a predicate raised `ValueError` on an unknown APID) · **one pull request, needs a second**

Wording: [R-003 in the instruction file](../.github/instructions/libera-utils.instructions.md#r-003--one-exception-type-per-condition-and-a-predicate-returns-rather-than-raises)

Do not flag: one exception type covering conditions a caller genuinely handles identically,
when the message distinguishes them.

### R-004 · Every public symbol has a numpydoc docstring, including what it raises

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0041, pr-0027, pr-0015

Wording: [R-004 in the instruction file](../.github/instructions/libera-utils.instructions.md#r-004--every-public-symbol-has-a-numpydoc-docstring-including-what-it-raises)

Do not flag: private helpers; a one-line docstring on a symbol whose signature is
self-describing and that raises nothing.

### R-005 · A published quantity states its unit; a time states its epoch and frame

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0027 (commanded exposure times and integration-time registers went up for review with no `units` attribute, and merged as milliseconds and raw counts) · **one author, needs a second**

Wording: [R-005 in the instruction file](../.github/instructions/libera-utils.instructions.md#r-005--a-published-quantity-states-its-unit-a-time-states-its-epoch-and-frame)

Do not flag: dimensionless counters and flags; a field whose unit is stated once for a group
in the product definition.

### R-006 · One source of truth for a value; tabular data lives in a data file

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0027, pr-0041 (the ObsID registry moved from a module literal to a data file validated at import), pr-0015, pr-0002 (in one of these, a packet width constant restated what the dtype already carried)

Wording: [R-006 in the instruction file](../.github/instructions/libera-utils.instructions.md#r-006--one-source-of-truth-for-a-value-tabular-data-lives-in-a-data-file)

Do not flag: a named constant that gives a meaning to a literal used in one place; a value
duplicated in a test on purpose, so the test fails when the source changes.

### R-007 · Delete dead code rather than leaving it unreferenced

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0027, pr-0048 (between them, a function mentioned only by a comment explaining why it was unused, a `try`/`except` whose result was discarded, and three counters never read)

Wording: [R-007 in the instruction file](../.github/instructions/libera-utils.instructions.md#r-007--delete-dead-code-rather-than-leaving-it-unreferenced)

Do not flag: a public symbol kept for backwards compatibility with a deprecation note; code
behind a feature flag that the changelog names.

### R-008 · Test scaffolding does not ship in the package — **retired 2026-09-19**

tier: — · status: retired · reasoning: shared `archive/libera_utils/R-008.md`

Retired at the second harvest: the wider sample found it once, against three times for R-015,
and a configuration line could carry it instead. The concern is real and has not gone away; it
is now carried by `packages = [{include = "libera_utils"}]` in `pyproject.toml`, which was verified
by building the wheel.

### R-009 · Comments describe the code as it is, not how it got there

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0037, pr-0058, pr-0030 (the most repeated request in the harvested reviews, six times in one of them: remove the ticket number, the historical title, the comment saying what this used to be)

Wording: [R-009 in the instruction file](../.github/instructions/libera-utils.instructions.md#r-009--comments-describe-the-code-as-it-is-not-how-it-got-there)

Do not flag: a tagged deferred-work marker such as `TODO[LIBSDC-1234]`, which is
forward-looking; a comment citing an external
document that the code implements.

### R-010 · Deferred work carries a LIBSDC or CURRYER ticket tag

tier: check · status: graduated · since: 2026-09 · evidence: pre-commit, standing convention

Reasoning: shared `archive/libera_utils/R-010.md`

Do not flag: this rule at all. The hook owns it, and a marker in a file the hook excludes
(`.pre-commit-config.yaml`) is deliberately out of scope.
Check: `.pre-commit-config.yaml`, the `lasp/prevent-dangling-todos` hook, with its tags set
to `LIBSDC,CURRYER` and its comment markers to the two it scans for. The reviewer does not
check this; the hook does.

### R-011 · A dependency pins to an immutable ref

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0058 (a moving ref in libera_rad had taken main and three pull requests red overnight) · implements shared D-005 · **one pull request, needs a second**

Wording: [R-011 in the instruction file](../.github/instructions/libera-utils.instructions.md#r-011--a-dependency-pins-to-an-immutable-ref)

Do not flag: a pin in a local development extra that is never published.

### R-012 · The version bump matches the change, and the changelog heading matches it

tier: check · status: graduated · since: 2026-09 · evidence: pr-0048, pr-0037, pr-0027, pr-0022, pr-0032

Reasoning: shared `archive/libera_utils/R-012.md`

Do not flag: this rule at all; the hook owns the two checks, and whether a change is minor or
patch is the author's call, stated in the instruction file.
Check: the `version-check` pre-commit hook, running `.github/scripts/check_version.py`, which the
pre-commit workflow runs on every pull request: the first changelog heading equals the
`pyproject.toml` version, and that version is at or above the highest release tag. The reviewer
does not check this; the hook does.

### R-013 · Parse or sort an input once, not once per consumer

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0048 (a scan re-parsed a whole packet file once per APID, twelve passes over a 2 MB fixture in the ingest path), pr-0041 (a trim loop re-sorted a full-day dataset and re-read a YAML definition on each of about 35 runs)

Wording: [R-013 in the instruction file](../.github/instructions/libera-utils.instructions.md#r-013--parse-or-sort-an-input-once-not-once-per-consumer)

Do not flag: a repeated read of something small and mutable, where the reread is the point;
a loop that runs a fixed handful of times over a small input.

### R-014 · No internal URL or internal document content in this repository

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0041, pr-0027 · implements `libera_utils/D-006`

Wording: [R-014 in the instruction file](../.github/instructions/libera-utils.instructions.md#r-014--no-internal-url-or-internal-document-content-in-this-repository)

Do not flag: a LIBSDC ticket key on its own, which is an identifier rather than a link; a
public URL, such as NAIF or the CERES documentation.

### R-015 · The annotation says what the code actually accepts

tier: reviewer · status: **established** · since: 2026-09 · evidence: pr-0012 (seven requests to take `LiberaDataProductFilename` rather than `str`, and `PathType` where an `S3Path` can reach), pr-0028 (narrowing the annotation to local paths was chosen over rejecting cloud paths at runtime; the merged signature takes `list[Path]`, and the command line converts each `str` before the call), pr-0060 · rewritten 2026-09-22, 2026-09-23, corrected 2026-10-03 · archive/libera_utils/R-015.md · implements `libera_utils/D-007`

Wording: [R-015 in the instruction file](../.github/instructions/libera-utils.instructions.md#r-015--the-annotation-says-what-the-code-actually-accepts)

Do not flag: a union every branch of which the body handles, as pr-0041's
`source_product_filename: str | PathType` is by `Path(str(...))`; a genuine union the product
definition names; a constructor that documents a single coercion at the boundary and says so in its
docstring; a runtime check beside a narrowed annotation, or its absence, since the annotation is
the baseline and a check is neither required nor a finding.

---

### R-016 · An error message says what went wrong and what to do next

tier: reviewer · status: **established** · since: 2026-09 · evidence: pr-0028 (the reviewer dictated the replacement text naming the role to log in as and whom to contact), pr-0060 (an error telling the caller to provide a tag rather than defaulting to `latest`), pr-0015 (the log to name which data variables did not match) · established 2026-10-04, to match the decision it implements · implements `libera_utils/D-008`

Wording: [R-016 in the instruction file](../.github/instructions/libera-utils.instructions.md#r-016--an-error-message-says-what-went-wrong-and-what-to-do-next)

Do not flag: a message whose condition and next action are already in the exception type's
name; re-raising the original exception when its message already says what went wrong. Never
propose wrapping a standard exception in a custom type only to add a message: custom exception
types are public API and make common exceptions hard to catch.

### R-017 · A valid range or an enumeration cites its source

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0004 (a valid range asked for its reasoning had none and was removed), pr-0042 (a land-surface enumeration declared six categories where the ADM algorithm has five)

Wording: [R-017 in the instruction file](../.github/instructions/libera-utils.instructions.md#r-017--a-valid-range-or-an-enumeration-cites-its-source)

Do not flag: a range that is the dtype's own limits; an enumeration defined once elsewhere in
this repository and referenced by name.

## Candidates below the criteria

Kept here with their evidence and the criterion each fails, or why it is held, so a revision
can admit one when its evidence meets them.

- **Private symbols do not cross module boundaries.** A second consumer outside the defining
  module makes a symbol public in fact; rename and document it, or wrap it. Evidence: pr-0048
  (`_expand_sample_times`, `_extract_wfov_header_metadata_from_blob`). Held: judged against the
  two-pull-request criterion admission used until 2026-09-30, and not re-judged since; the first
  revision re-judges it.
- **A registry whose values reach a filename has a uniqueness invariant test.** Evidence:
  pr-0041 (one ObsID on two instruments produced two writes of the same filename). Held: judged
  against the two-pull-request criterion admission used until 2026-09-30, and not re-judged
  since; the first revision re-judges it.
- **Optional flags are keyword-only.** Evidence: pr-0048 (`ground_data`, `verbose`). Fails: a
  tool can check it (ruff's `FBT` rules).
- **A helper with one call site is inlined.** Evidence: pr-0030, where the same reviewer removed
  three of them in one pass — "yet another unnecessary helper function". Fails: the review
  contract's New surface section already reports call-site counts, so this is a check waiting
  for a firing rate rather than a rule waiting for a reviewer.
- **Do not hold a large array twice.** Evidence: pr-0027, a stitching path holding three copies
  of the image data live at once on a full downlink. It was part of R-013 until that rule was
  trimmed to what two pull requests support. Held: judged against the two-pull-request criterion
  admission used until 2026-09-30, and not re-judged since; the first revision re-judges it.
- **A name is renamed when its contract widens.** Evidence: pr-0028 (`get_libera_utils_session`
  → `get_l2_team_role_session` once it took a `role_name`). Held: judged against the
  two-pull-request criterion admission used until 2026-09-30, and not re-judged since; the first
  revision re-judges it.
