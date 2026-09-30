# Review rules

The ledger of what a reviewer checks in this repository beyond what the tools check: each
rule's tier, status, evidence and do-not-flag sentence. The wording of each rule lives once,
under its ID, in `.github/instructions/libera-utils.instructions.md`, and each entry links there.
**Budget: 300 lines**, counted across this file and the wording together, each rule's title
once, its link line not at all, and a `check`-tier rule not at all. A rule is admitted when it
meets the criteria: two merged pull requests from two authors, not something a tool can check,
not a one-off design question, and not a restatement of a decision. Tiers, statuses,
establishment, retirement and graduation are in `standards/README.md`.

---

### R-001 · Validate a name or identifier where it is constructed, not where it is first used

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0048 · **one pull request, needs a second**

Wording: [R-001 in the instruction file](../.github/instructions/libera-utils.instructions.md#r-001--validate-a-name-or-identifier-where-it-is-constructed-not-where-it-is-first-used)

Do not flag: a validator deliberately deferred because it needs data the constructor does
not have, when the docstring says so.

### R-002 · A condition that invalidates the output raises; it does not warn or no-op

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0037, pr-0027, pr-0041, pr-0060, pr-0012

Wording: [R-002 in the instruction file](../.github/instructions/libera-utils.instructions.md#r-002--a-condition-that-invalidates-the-output-raises-it-does-not-warn-or-no-op)

Do not flag: a genuinely optional input whose absence has a defined meaning, when the
docstring names it; a `logger.warning` beside a raise, for context.

### R-003 · One exception type per condition, and a predicate returns rather than raises

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0048 · **one pull request, needs a second**

Wording: [R-003 in the instruction file](../.github/instructions/libera-utils.instructions.md#r-003--one-exception-type-per-condition-and-a-predicate-returns-rather-than-raises)

Do not flag: one exception type covering conditions a caller genuinely handles identically,
when the message distinguishes them.

### R-004 · Every public symbol has a numpydoc docstring, including what it raises

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0041, pr-0027, pr-0015

Wording: [R-004 in the instruction file](../.github/instructions/libera-utils.instructions.md#r-004--every-public-symbol-has-a-numpydoc-docstring-including-what-it-raises)

Do not flag: private helpers; a one-line docstring on a symbol whose signature is
self-describing and that raises nothing.

### R-005 · A published quantity states its unit; a time states its epoch and frame

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0027 · **one author, needs a second**

Wording: [R-005 in the instruction file](../.github/instructions/libera-utils.instructions.md#r-005--a-published-quantity-states-its-unit-a-time-states-its-epoch-and-frame)

Do not flag: dimensionless counters and flags; a field whose unit is stated once for a group
in the product definition.

### R-006 · One source of truth for a value; tabular data lives in a data file

tier: reviewer · status: **established** · since: 2026-09 · evidence: pr-0027, pr-0041, pr-0015, pr-0002

Wording: [R-006 in the instruction file](../.github/instructions/libera-utils.instructions.md#r-006--one-source-of-truth-for-a-value-tabular-data-lives-in-a-data-file)

Do not flag: a named constant that gives a meaning to a literal used in one place; a value
duplicated in a test on purpose, so the test fails when the source changes.

### R-007 · Delete dead code rather than leaving it unreferenced

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0027, pr-0048

Wording: [R-007 in the instruction file](../.github/instructions/libera-utils.instructions.md#r-007--delete-dead-code-rather-than-leaving-it-unreferenced)

Do not flag: a public symbol kept for backwards compatibility with a deprecation note; code
behind a feature flag that the changelog names.

### R-008 · Test scaffolding does not ship in the package — **retired 2026-09-19**

tier: — · status: retired · reasoning: `standards/README.md` "Archived" → `archive/libera_utils/R-008.md`

Retired at the second harvest to make room for R-015, which the wider sample evidences three
times over against this rule's one. The concern is real and has not gone away; it is now
carried by `packages = [{include = "libera_utils"}]` in `pyproject.toml`, which was verified
by building the wheel.

### R-009 · Comments describe the code as it is, not how it got there

tier: reviewer · status: **established** · since: 2026-09 · evidence: pr-0037, pr-0058, pr-0030

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

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0058 · implements shared D-005 · **one pull request, needs a second**

Wording: [R-011 in the instruction file](../.github/instructions/libera-utils.instructions.md#r-011--a-dependency-pins-to-an-immutable-ref)

Do not flag: a pin in a local development extra that is never published.

### R-012 · The version bump matches the change, and the changelog heading matches it

tier: check · status: graduated · since: 2026-09 · evidence: pr-0048, pr-0037, pr-0027, pr-0022, pr-0032

Reasoning: shared `archive/libera_utils/R-012.md`

Do not flag: this rule at all; the workflow owns the two equalities, and whether a change is
minor or patch is the author's call, stated in the instruction file.
Check: `.github/workflows/version-check.yml`, running `.github/scripts/check_version.py`: the
first changelog heading equals the `pyproject.toml` version, and on a pull request that
changes `pyproject.toml` the version is above the highest tag. The reviewer does not check
this; the workflow does.

### R-013 · Parse or sort an input once, not once per consumer

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0048, pr-0041

Wording: [R-013 in the instruction file](../.github/instructions/libera-utils.instructions.md#r-013--parse-or-sort-an-input-once-not-once-per-consumer)

Do not flag: a repeated read of something small and mutable, where the reread is the point;
a loop that runs a fixed handful of times over a small input.

### R-014 · No internal URL or internal document content in this repository

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0041, pr-0027 · implements `libera_utils/D-006`

Wording: [R-014 in the instruction file](../.github/instructions/libera-utils.instructions.md#r-014--no-internal-url-or-internal-document-content-in-this-repository)

Do not flag: a LIBSDC ticket key on its own, which is an identifier rather than a link; a
public URL, such as NAIF or the CERES documentation.

### R-015 · The annotation says what the code actually accepts

tier: reviewer · status: **established** · since: 2026-09 · evidence: pr-0012, pr-0028, pr-0060 · implements `libera_utils/D-007`

Wording: [R-015 in the instruction file](../.github/instructions/libera-utils.instructions.md#r-015--the-annotation-says-what-the-code-actually-accepts)

Do not flag: a union the code genuinely handles — `Path | str` is the fix pr-0028 agreed on,
not a violation of this rule; a genuine union the product definition names; a constructor that
documents a single coercion at the boundary and says so in its docstring.

---

### R-016 · An error message says what went wrong and what to do next

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0028, pr-0060, pr-0015 · implements `libera_utils/D-008`

Wording: [R-016 in the instruction file](../.github/instructions/libera-utils.instructions.md#r-016--an-error-message-says-what-went-wrong-and-what-to-do-next)

Do not flag: a message whose condition and next action are already in the exception type's
name.

### R-017 · A valid range or an enumeration cites its source

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0004, pr-0042

Wording: [R-017 in the instruction file](../.github/instructions/libera-utils.instructions.md#r-017--a-valid-range-or-an-enumeration-cites-its-source)

Do not flag: a range that is the dtype's own limits; an enumeration defined once elsewhere in
this repository and referenced by name.

## Candidates below the criteria

Kept here with their evidence and the criterion each fails, so a revision can admit one when
its evidence meets them.

- **Private symbols do not cross module boundaries.** A second consumer outside the defining
  module makes a symbol public in fact; rename and document it, or wrap it. Evidence:
  pr-0048 (`_expand_sample_times`, `_extract_wfov_header_metadata_from_blob`). Fails: one pull
  request.
- **A registry whose values reach a filename has a uniqueness invariant test.** Evidence:
  pr-0041 (one ObsID on two instruments produced two writes of the same filename). Fails: one
  pull request.
- **Optional flags are keyword-only.** Evidence: pr-0048 (`ground_data`, `verbose`). Fails: one
  pull request, and a tool can check it (ruff's `FBT` rules).
- **A helper with one call site is inlined.** Evidence: pr-0030, where the same reviewer
  removed three of them in one pass — "yet another unnecessary helper function". Fails: one
  pull request, and the review contract's New surface section already reports call-site
  counts, so this is a check waiting for a firing rate rather than a rule waiting for a
  reviewer.
- **Do not hold a large array twice.** Evidence: pr-0027, a stitching path holding three
  copies of the image data live at once on a full downlink. Fails: one pull request; it was
  part of R-013 until that rule was trimmed to what two pull requests support.
- **A name is renamed when its contract widens.** Evidence: pr-0028
  (`get_libera_utils_session` → `get_l2_team_role_session` once it took a `role_name`). Fails:
  one pull request.
