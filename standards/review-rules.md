# Review rules

The ledger of what a reviewer checks in this repository beyond what the tools check: each
rule's tier, status, evidence and do-not-flag sentence. The wording of each rule lives once,
under its ID, in `.github/instructions/libera-utils.instructions.md`, and each entry links there.
**Budget: 300 lines**, counted across this file and the wording together, each rule's title
once, its link line not at all, and a `check`-tier rule not at all. A rule that implements a
decision carries that decision's status, as R-015 and R-016 do. How a rule is admitted,
established, graduated and retired, and every threshold that decides it, is the tooling's:
`standards/dials.md` in libera_llm_tooling.

---

### R-001 · Validate a name or identifier where it is constructed, not where it is first used

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0012 (two reviewers asked that a product filename be validated where it is built, not passed on as a `str` that bypasses the filename pattern) · **one pull request, needs a second**

Wording: [R-001 in the instruction file](../.github/instructions/libera-utils.instructions.md#r-001--validate-a-name-or-identifier-where-it-is-constructed-not-where-it-is-first-used)

Do not flag: a validator deliberately deferred because it needs data the constructor does
not have, when the docstring says so.

### R-002 · A condition that invalidates the output raises; it does not warn or no-op

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0037 (an Az/El CK whose L1A input had no encoder columns returned quietly with an empty kernel; it now raises), pr-0060, pr-0022 (an integer array cast silently to a string dtype; it now raises), pr-0041

Wording: [R-002 in the instruction file](../.github/instructions/libera-utils.instructions.md#r-002--a-condition-that-invalidates-the-output-raises-it-does-not-warn-or-no-op)

Do not flag: a genuinely optional input whose absence has a defined meaning, when the
docstring names it; a `logger.warning` beside a raise, for context.

### R-003 · One exception type per condition, and a predicate returns rather than raises

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0048 (one exception type was raised for four unrelated conditions, and a predicate raised `ValueError` on an unknown APID; the first ask is a reviewed draft and the second a bot's, so it needs a person's ask) · **one pull request, needs a second**

Wording: [R-003 in the instruction file](../.github/instructions/libera-utils.instructions.md#r-003--one-exception-type-per-condition-and-a-predicate-returns-rather-than-raises)

Do not flag: one exception type covering conditions a caller genuinely handles identically,
when the message distinguishes them.

### R-004 · Every public symbol has a numpydoc docstring, including what it raises

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0027, pr-0015, pr-0012

Wording: [R-004 in the instruction file](../.github/instructions/libera-utils.instructions.md#r-004--every-public-symbol-has-a-numpydoc-docstring-including-what-it-raises)

Do not flag: private helpers; a one-line docstring on a symbol whose signature is
self-describing and that raises nothing.

### R-005 · A published quantity states its unit; a time states its epoch and frame

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0027 (commanded exposure times and integration-time registers went up for review with no `units` attribute, and merged as milliseconds and raw counts) · **one author, needs a second**

Wording: [R-005 in the instruction file](../.github/instructions/libera-utils.instructions.md#r-005--a-published-quantity-states-its-unit-a-time-states-its-epoch-and-frame)

Do not flag: dimensionless counters and flags; a field whose unit is stated once for a group
in the product definition.

### R-006 · One source of truth for a value; tabular data lives in a data file

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0027, pr-0041 (the ObsID registry moved from a module literal to a data file validated at import), pr-0015, pr-0002, pr-0042 (in one of these, a packet width constant restated what the dtype already carried; in pr-0042, a bin list written into descriptions beside the dimension that defines it)

Wording: [R-006 in the instruction file](../.github/instructions/libera-utils.instructions.md#r-006--one-source-of-truth-for-a-value-tabular-data-lives-in-a-data-file)

Do not flag: a named constant that gives a meaning to a literal used in one place; a value
duplicated in a test on purpose, so the test fails when the source changes.

### R-007 · Delete dead code rather than leaving it unreferenced

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0027 (a constant, a branch and three counters nothing read), pr-0018 (a decorator nothing used once the kernel manager furnished kernels), pr-0015 (an unused parameter)

Wording: [R-007 in the instruction file](../.github/instructions/libera-utils.instructions.md#r-007--delete-dead-code-rather-than-leaving-it-unreferenced)

Do not flag: a public symbol kept for backwards compatibility with a deprecation note; code
behind a feature flag that the changelog names.

### R-008 · Test scaffolding does not ship in the package — **retired 2026-09-19**

tier: — · status: retired · reasoning: shared `archive/libera_utils/R-008.md`

Retired at the second harvest: the wider sample found it once, against three times for R-015,
and a configuration line could carry it instead. The concern is real and has not gone away; it
is now carried by `packages = [{include = "libera_utils"}]` in `pyproject.toml`, which was verified
by building the wheel.

### R-009 · Comments and names describe the code as it is, not how it got there

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0037, pr-0058, pr-0030, pr-0036 (the most repeated request in the harvested reviews, six times in one of them: remove the ticket number, the historical title, the comment saying what this used to be), pr-0021 (a `NEW_` constant and tests named for being new) · widened to names 2026-10-06

Wording: [R-009 in the instruction file](../.github/instructions/libera-utils.instructions.md#r-009--comments-and-names-describe-the-code-as-it-is-not-how-it-got-there)

Do not flag: a tagged deferred-work marker such as `TODO[LIBSDC-1234]`, which is
forward-looking; a comment citing an external
document that the code implements; a name that tells apart two versions the code keeps side by
side on purpose.
scope: libera_utils/**/*.py, tests/**/*.py
example: fires on libera_utils#21 PRRT_kwDORdqWxM6B0Gb3; not on libera_utils#21@7ef5d28 tests/unit/test_io/test_product_definition.py::TestLiberaStandardDimensions.test_standard_dimensions_match_yaml_file "Every dimension in libera_dimensions.yml loads as a LiberaDimensionDefinition."

### R-010 · Deferred work carries a LIBSDC or CURRYER ticket tag

tier: check · status: graduated · since: 2026-09 · evidence: pre-commit, standing convention

Reasoning: shared `archive/libera_utils/R-010.md`

Do not flag: this rule at all. The hook owns it, and a marker in a file the hook excludes
(`.pre-commit-config.yaml`) is deliberately out of scope.
Check: `.pre-commit-config.yaml`, the `lasp/prevent-dangling-todos` hook, with its tags set
to `LIBSDC,CURRYER` and its comment markers to the two it scans for. The reviewer does not
check this; the hook does.

### R-011 · A dependency pins to an immutable ref

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0058 (a moving ref in libera_rad had taken main and three pull requests red overnight; the thread asking for the pin is a bot's) · implements shared D-005 · **one pull request, needs a second**

Wording: [R-011 in the instruction file](../.github/instructions/libera-utils.instructions.md#r-011--a-dependency-pins-to-an-immutable-ref)

Do not flag: a pin in a local development extra that is never published.

### R-012 · The version bump matches the change, and the changelog heading matches it

tier: check · status: graduated · since: 2026-09 · evidence: pr-0048, pr-0037, pr-0027, pr-0022, pr-0032

Reasoning: shared `archive/libera_utils/R-012.md`

Do not flag: this rule at all; the hook owns the two checks, and whether a change is minor or
patch is the author's call, stated in the instruction file.
Check: the `version-check` pre-commit hook, running `.github/scripts/check_version.py`, which the
pre-commit workflow runs on every pull request: the latest release heading in `CHANGELOG.md`
equals the `pyproject.toml` version, and that version is at or above the highest release tag. The reviewer
does not check this; the hook does.

### R-013 · Hold one copy of a large input, and pass over it once

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0027 (the stitching path held the payload list and a padded copy of the same bytes at once, and copied the image data again to keep its attributes) · reworded 2026-10-06, since its earlier evidence, pr-0048 and pr-0041, is AI-drafted · **one pull request, needs a second**

Wording: [R-013 in the instruction file](../.github/instructions/libera-utils.instructions.md#r-013--hold-one-copy-of-a-large-input-and-pass-over-it-once)

Do not flag: a repeated read of something small and mutable, where the reread is the point;
a loop that runs a fixed handful of times over a small input; a copy a library makes that the
change cannot avoid, when a comment says so.
scope: libera_utils/**/*.py
example: fires on libera_utils#27 PRRT_kwDORdqWxM6UjzXp; not on libera_utils#27@4a8f12e libera_utils/l1a/wfov_image_metadata.py::_build_camera_dataset "blob_array[row, : len(image.payload)] = np.frombuffer(image.payload, dtype=np.uint8)"

### R-014 · No internal URL or internal document content in this repository

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0041, pr-0027 (both asks AI-drafted; the rule rests on the decision) · implements `libera_utils/D-006`

Wording: [R-014 in the instruction file](../.github/instructions/libera-utils.instructions.md#r-014--no-internal-url-or-internal-document-content-in-this-repository)

Do not flag: a LIBSDC ticket key on its own, which is an identifier rather than a link; a
public URL, such as NAIF or the CERES documentation.

### R-015 · The annotation says what the code actually accepts

tier: reviewer · status: **established** · since: 2026-09 · evidence: pr-0012 (seven requests to take `LiberaDataProductFilename` rather than `str`, and `PathType` where an `S3Path` can reach), pr-0028 (narrowing the annotation to local paths was chosen over rejecting cloud paths at runtime; the merged signature takes `list[Path]`, and the command line converts each `str` before the call), pr-0060 (a bot's ask, as is pr-0028's thread; pr-0012's are people's) · rewritten 2026-09-22, 2026-09-23, corrected 2026-10-03 · archive/libera_utils/R-015.md · implements `libera_utils/D-007`

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

tier: reviewer · status: provisional · since: 2026-09 · evidence: pr-0004 (a valid range asked for its reasoning had none and was removed), pr-0002 (a reviewer asked whether an enumeration's string length and a temperature's float dtype came from flight software)

Wording: [R-017 in the instruction file](../.github/instructions/libera-utils.instructions.md#r-017--a-valid-range-or-an-enumeration-cites-its-source)

Do not flag: a range that is the dtype's own limits; an enumeration defined once elsewhere in
this repository and referenced by name.

### R-018 · Use the code that already does a job, and do not run it twice

tier: reviewer · status: provisional · since: 2026-10 · evidence: pr-0012 (open a product through the repository's netCDF engine setting, and serialize a model with its own method rather than a custom null filter), pr-0023 (a test ran the conformance check that the product writer already runs)

Wording: [R-018 in the instruction file](../.github/instructions/libera-utils.instructions.md#r-018--use-the-code-that-already-does-a-job-and-do-not-run-it-twice)

Do not flag: a second implementation whose contract differs, when the change says why; a test
of a helper that calls the helper's parts directly.
scope: libera_utils/**/*.py, tests/**/*.py
example: fires on libera_utils#23 PRRT_kwDORdqWxM6FyIrR; not on libera_utils#23@d4aeb68 tests/integration/test_l1a_processing.py::test_process_packets_to_l1a_product "output_filename = write_libera_data_product("

## Candidates below the criteria

Kept here with their evidence and the criterion each fails, or why it is held, so a revision
can admit one when its evidence meets them.

- **Private symbols do not cross module boundaries.** A second consumer outside the defining
  module makes a symbol public in fact; rename and document it, or wrap it. Evidence: pr-0048
  (`_expand_sample_times`, `_extract_wfov_header_metadata_from_blob`). Held: its one ask is
  AI-drafted.
- **A registry whose values reach a filename has a uniqueness invariant test.** Evidence:
  pr-0041 (one ObsID on two instruments produced two writes of the same filename). Held: its one
  ask is AI-drafted.
- **Optional flags are keyword-only.** Evidence: pr-0048 (`ground_data`, `verbose`). Fails: a
  tool can check it (ruff's `FBT` rules, which this repository does not enable).
- **A helper with one call site is inlined.** Evidence: pr-0030 ("yet another unnecessary helper
  function"). Held: those threads are the pull request author's own, which are never an ask; and
  the review contract's New surface section already reports call-site counts.
- **A name is renamed when its contract widens.** Evidence: pr-0028 (`get_libera_utils_session`
  → `get_l2_team_role_session` once it took a `role_name`). Held: the thread is the pull
  request author's own, which is never an ask.
- **A test covers every member of a family the same way.** Evidence: pr-0004 (one test per
  packet configuration, run over every configuration), pr-0021 (every dimension tested alike,
  not only the added ones). Fails: the review contract's test check already asks for a test to
  be extended or parametrized before one is added.
