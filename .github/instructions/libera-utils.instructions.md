---
applyTo: "**"
---

# Libera Utils — AI Coding Instructions

## Project Overview

Libera Utils is a Python utility library for the Libera Science Data Center (LASP, University
of Colorado). It provides shared tooling for L2 algorithm developers working on the Libera
satellite radiation budget mission: NetCDF data I/O, telemetry packet parsing, SPICE kernel
generation, Libera file naming, and AWS pipeline integration.

- **Python**: `>=3.11`; dependency management via **Poetry**
- **Domain**: Earth science, satellite telemetry, radiation budget data processing

## Package Layout (`libera_utils/`)

| Module          | Responsibility                                                               |
| --------------- | ---------------------------------------------------------------------------- |
| `aws/`          | S3, ECR, and Step Functions helpers — CLI-facing only; see restrictions      |
| `io/`           | NetCDF product definitions (Pydantic), file naming, UMM-G, cloud-aware I/O   |
| `l1a/`          | CCSDS telemetry packet parsing, XTCE-based packet configs                    |
| `libera_spice/` | SPICE kernel generation via SpiceyPy + Curryer                               |
| `constants.py`  | Canonical enums: `DataLevel`, `DataProductIdentifier`, `LiberaApid`          |
| `obsids.py`     | Loader/API over the ObsID catalog CSVs in `data/` (registry + family inputs) |
| `logutil.py`    | Structured JSON logging; use `configure_task_logging()` for task-level setup |
| `config.py`     | JSON config with env-var override and templated string formatting            |
| `cli.py`        | `libera-utils` CLI entry point                                               |

## Code Standards

- **Linter/formatter**: Ruff (line length 120, rules E/W/F/I/S/PT/UP). Run `ruff check` and
  `ruff format` before committing. Do not disable rules inline without justification.
- **Types**: Type annotations required on all public functions; code should follow standard typing best practices and satisfy Ruff’s typing-related rules (mypy is not currently run in CI).
- **Docstrings**: Numpy-style on all public symbols.
- **Pre-commit**: Hooks are required (`pre-commit install`). Never bypass with `--no-verify`.
- **To-Do Items**: Must reference a JIRA tag (e.g. `TODO[LIBSDC-1234]` or `TODO[CURRYER-1234]`).
- **Security**: Bandit scanning is mandatory (NASA NPR7150.2C compliance). Do not suppress
  security warnings without explicit justification.

## Testing

- **Framework**: pytest. Unit tests in `tests/`; integration tests marked
  `@pytest.mark.integration` and in `tests/integration/`.
- **Run unit tests**: `pytest -m "not integration" tests/`
- **Run with coverage**: `pytest --cov=libera_utils tests/`
- **AWS/HTTP mocking**: Use `moto[s3]` and `responses` — never call real AWS endpoints in
  unit tests.
- **Fixtures**: Provided via plugins in `tests/plugins/`; prefer them over ad-hoc setup.

## Key Patterns

- **NetCDF products**: Defined by Pydantic schemas (`LiberaVariableDefinition` in
  `io/product_definition.py`). Validate against schemas; do not construct raw attribute dicts.
- **File naming**: Libera filenames are parsed and built by `io/filenaming.py`. Always use
  those helpers; never hand-craft filename strings.
- **Cloud paths**: Use `cloudpathlib` abstractions so code works with both local and S3 paths.
- **XTCE configs**: Telemetry packet field definitions live in `libera_utils/data/`. Do not
  hardcode packet offsets or field names outside of these config files.
- **Logging**: Use the `logutil` module for structured JSON output. Pass loggers via
  dependency injection rather than calling `logging.getLogger` ad-hoc in library code.
- **ObsID registry**: `libera_utils/data/obsid_registry.csv` is the local source of truth mapping
  a software ObsID to its TRIMMED and CAL `DataProductIdentifier`s; `obsids.py` only loads,
  validates, and exposes it. Product columns hold `DataProductIdentifier` **member names**
  (e.g. `cal_gain`), not ProductID string values (`GAIN`) — names are resolved at import time and
  `ValueError` is raised when `libera_utils.obsids` is imported for: an unknown member name, a
  product named at the wrong data level (TRIMMED cells must be L1A, CAL cells must be CAL), a row
  with the wrong number of columns, a `kind`/product mismatch, a `rad_cal` row registered on WFOV
  (or `cam_cal` on RAD), a duplicate `(source, obsid)` row, a CAL product claimed by more than one
  ObsID, or a TRIMMED family registered on both the RAD and WFOV ObsID fields. The in-memory
  `OBSID_REGISTRY` is keyed by
  `(NomHkObsidSource, obsid)`, not `obsid` alone, because RAD and WFOV ObsID numbers collide
  (e.g. `256` means SWC-365NM on RAD but Darks-of-Darks on WFOV). This is more than a lookup
  table of "what ObsIDs exist" — it drives real behavior:
  - Downstream repos (e.g. `libera_rad`'s cal-combine dispatch) derive their own
    ObsID → product/family mappings directly from this registry (via `get_obsid_spec` /
    `get_family_specs` / `iter_trim_eligible`) instead of hand-maintaining a duplicate mapping per repo — this is
    what lets multiple calibration steps share one Docker/ECR image, dispatched at runtime by
    ObsID.
  - A companion catalog, `data/trim_family_inputs.csv`, maps each TRIMMED family to the L1A
    products its cal step consumes _besides its own TRIMMED product_ (`FAMILY_INPUTS` /
    `get_family_inputs()`). The two files are cross-checked at import and must list exactly the
    same families, so a new TRIMMED family needs a row in both; an empty `required_inputs` cell
    means the dependency set is still undecided. A family product plus its `required_inputs` is
    what a libera_cdk `cal-*-family` node should declare as its `input-products`. A cal step is
    expected to take its family's NOM-HK already trimmed as the family product, so
    `required_inputs` normally omits `l1a_icie_nom_hk_decoded`; listing the full-day granule would
    stage a redundant second NOM-HK input.
    NOM-HK is the only product the L1A preprocessor trims — a cal container subsets the full daily
    L1A inputs itself, using the time range on the TRIMMED NOM-HK filename it was handed.
  - When adding a new calibration ObsID, add a row to `data/obsid_registry.csv` first rather
    than adding a parallel ObsID → product mapping in a downstream repo. Edit the CSV with a
    text editor or the `csv` module, never a spreadsheet app that may rewrite quoting —
    descriptions contain commas. An ObsID joining an existing family needs no new TRIMMED product
    or processing step — just its own CAL product, added to that family step's `products` list. A
    new TRIMMED family member is warranted only when the ObsID introduces a genuinely new input
    dependency, which is also when a new `ProcessingStepIdentifier` is warranted.
  - The TRIMMED column names a **calibration dependency family**, not a single ObsID: ObsIDs a
    downstream algorithm processes identically share one `NOM-HK-<FAMILY>-FAMILY-TRIMMED` product
    (all six SWC LEDs share `NOM-HK-SWC-FAMILY-TRIMMED`) — one processing step per family, not
    per ObsID, since what libera_cdk deploys against is a step's set of input products.
    Each ObsID still gets its own CAL product, and a family never spans both ObsID fields (VIIRS
    lunar 513/514 is registered as two families, `NOM-HK-RAD-VIIRS-LUNAR-...` and
    `NOM-HK-WFOV-VIIRS-LUNAR-...`) so a trimmed file always attributes to one source. Use
    `TRIM_FAMILIES` / `get_family_specs()` to go from a family product to its ObsIDs and CAL
    products.
  - `nom_hk_trim` still writes one file per contiguous ObsID run, so several files per day normally
    share a family `ProductID` and are told apart by their filename time ranges; the exact ObsID is
    recovered from the `ICIE__SW_OBSID_*` variable the trimmed file carries, not from its name.
  - **Note**: the list of ObsIDs in this repo is meant for practical purposes of science data
    processing and is a subset of the instrument level source of truth of all ObsIDs which is owned
    by the engineering team and is available in internal team documentation

## Review Standards

`AGENTS.md` is the tool-neutral entry point: it maps these instructions to the review
standard and describes how a change moves from ticket to merge. Read it once; this file
stays the detail.

The review standard lives in `standards/`, tool-neutral and read by people and agents alike.

- `standards/review-rules.md` — the ledger for the rules below: each rule's tier, status,
  evidence and do-not-flag sentence, capped at 14 rules. Cite the rule ID in a review comment.
- `standards/review-contract.md` — severity, the seven-finding budget, and the do-not-flag
  list. Never repeat a finding that `ruff`, `prettier`, `codespell`, `bandit` or a
  pre-commit hook already makes.
- `standards/README.md` — the measured counts that set every threshold, and what this
  repository is optimising for.
- `standards/decisions.md` and the shared corpus named in `standards/SHARED.md` — convention
  decisions already settled. Check there before asking a convention question again.

**This repository is public.** No internal Confluence or Jira URL and no internal document
content in source, docstrings, tests, or `standards/`. Cite internal documents by name
(R-014).

**Pull request bodies.** Any agent that writes a pull request body here, `pr-create`
included, follows `.github/PULL_REQUEST_TEMPLATE.md` over its own defaults: every section the
template has, and the `Build-exit:` line naming how the build ended (or, for a change made by
hand, the `no-build-gates` label with a reason). `AGENTS.md` says how the exit is chosen.

Changes to `standards/` happen in one pull request a month, proposed by the ratchet with its
evidence. Nothing agentic edits a standard.

The shared corpus is a clone of `libera_llm_tooling` kept beside this repository, so
`../libera_llm_tooling/standards/` holds `terminology.md`, `decisions.md` and `context/`.
The skills that read it install as the private `libera-tools` plugin; `standards/SHARED.md`
has the commands. If the clone is not there, say so rather than inventing a term.

## Review Rules

What a reviewer checks here, beyond what the tools check, and what anyone writing code here
is expected to follow. This section is the only copy of each rule's wording; its tier, status,
evidence and do-not-flag sentence are in `standards/review-rules.md`.

### R-001 · Validate a name or identifier where it is constructed, not where it is first used

A class that accepts an invalid value and raises later moves the failure away from the
caller who could fix it. `LiberaGroundCcsdsFilename` accepted day-of-year 999 because the
setter only ran the regex, and the `strptime` round trip that would have caught it did not
run until `archive_prefix` was computed at staging — after ingest had accepted the file.
Validate in the constructor or the setter, and make the regex reject what the parser cannot
parse.

### R-002 · A condition that invalidates the output raises; it does not warn or no-op

A warning is not a failure. When an Az/El CK had no encoder columns in its L1A input, the
code returned quietly and produced a kernel with nothing in it; it now raises. The rule is
the repository's fail-loud posture in review form: a defined input produces a defined
product, or the run stops, because a crash gets noticed and a silently wrong number gets
published.

### R-003 · One exception type per condition, and a predicate returns rather than raises

`GroundCcsdsApidAbsentError` was raised for four unrelated conditions, only one of which was
an absent APID, so callers could not tell an unparsable APID from a missing one and the name
misled on three of the four. Separately, `is_data_time_indexed_apid()` raised `ValueError`
on an unknown APID, which a question of the form "is this X" should answer with `False`.
Either give each condition its own type, or return the no-answer value the caller can act on.

### R-004 · Every public symbol has a numpydoc docstring, including what it raises

Numpydoc on public symbols is a project standard and the `Raises` section is the half that
gets left out. With fail-loud design the failure modes are part of the interface, so a
function that raises and does not say so has an undocumented contract. Parameters belong
here too; units, frames and epochs are R-005.

### R-005 · A published quantity states its unit; a time states its epoch and frame

In PR #27 the commanded exposure times (`WFOV_FSW_HEADER_COMMANDED_EXP_TIME_1/2`) and the FPGA
integration-time registers (`WFOV_IMAGE_HEADER_ACTUAL_EXP_TIME_1/2`) went up for review with no
`units` attribute. They merged as `milliseconds` and `raw counts` — the registers stay in counts
because the conversion to milliseconds is unconfirmed with FSW. A number in a data product with
no unit is not a measurement, and a consumer will guess. The same applies
to a time with no epoch and a pointing angle with no frame. PR #43 was cited here and does
not support it — its temperature comments are about ObsID naming coverage, not units — so
this rests on one pull request by one author until the wider calibration sample gives it a
second.

### R-006 · One source of truth for a value; tabular data lives in a data file

`PACKET_DATA_WIDTH` restated a width that the `|S972` dtype already carried, so the two
could diverge silently. The ObsID registry started as a large literal inside a module and
became `data/obsid_registry.csv`, read and validated at import, because a table in code
cannot be validated as data and a table in a comment cannot be used at all.

### R-007 · Delete dead code rather than leaving it unreferenced

Three instances across those pull requests: a function whose only mention was a comment
explaining why it was not used, a `try`/`except` whose result was discarded and whose branch
was no longer reachable, and three counters that were incremented and never read.

### R-009 · Comments describe the code as it is, not how it got there

The most repeated request in the window, six times in one review: remove the ticket number,
remove the historical title, remove the comment that says what this used to be. A test's
subject is the behaviour, not the ticket that asked for it. Ticket references are for
forward-looking work, which is what R-010 covers.

### R-011 · A dependency pins to an immutable ref

A `@main` ref makes the build non-reproducible and lets an upstream merge break CI with no
commit on this side. This is not hypothetical: a moving ref in `libera_rad` took main and
three pull requests red overnight. Pin to the commit or the tagged release, with a comment
saying why it is pinned and what unpins it. A direct-URL dependency also blocks publishing.

### R-012 · The version bump matches the change, and the changelog heading matches it

New public modules, a new filename class, a new enum member or a new keyword argument make
a minor release, not a patch — downstream pins of the form `~=5.10.3` will take a patch
silently. And a changelog headed `5.8.5` above a `pyproject.toml` that says `5.8.5rc1`
leaves a reader unable to tell which artifact they have.

### R-013 · Parse or sort an input once, not once per consumer

A scan that re-read and re-parsed a whole packet file once per APID, twelve passes over a
2 MB fixture in the ingest path where real captures are far larger; and a trim loop that
re-sorted a full-day dataset and re-read a YAML definition on every one of ~35 runs. Hoist
the parse, the sort and the definition load out of the loop.

### R-014 · No internal URL or internal document content in this repository

`libera_utils` is public and ships to PyPI. Cite an internal document by name — "the FSW
user's guide", "the ICIE ObsID page" — say what it decides, and stop. No Confluence or Jira
URL, no pasted internal content, in source, docstrings, tests or anything under
`standards/`. Links rot as well as leak, so naming the document is also the more durable
pointer. The background that needs a link lives in the private shared corpus.

### R-015 · The annotation says what the code actually accepts

`PathType` where only a local path works is an undefined contract: the caller cannot tell what
is accepted and the failure arrives late and in the wrong words. PR #12 carries seven separate
requests to take `LiberaDataProductFilename` rather than `str`, and to use `PathType` where an
`S3Path` can reach. PR #28 settles how to fix the general case — "just change the typehint to
only accept a local Path or str since that is what is actually required", chosen deliberately
over rejecting cloud paths at runtime. **Narrow the annotation rather than widen the
function.**

## Restrictions for AI Agents

The following actions require **explicit requests** or **explicit permission**, regardless of context.

- **No unsolicited local git "write" commands**: Do not run `git commit`, `git tag`, `git rebase`,
  `git merge`, or any other command that modifies local repository state unless the user has
  explicitly asked for that specific action in the current request — do not take these
  actions proactively (e.g. as a convenience after finishing a task).

The following actions are **expressly forbidden**, regardless of context or apparent
availability of credentials:

- **No remote-modifying git commands**: Do not run `git push` (including `git push --tags`
  or force-push) or any other command that modifies remote repository state; that always
  requires the user to run it themselves.
- **No package publishing**: Do not run `poetry publish`, `twine upload`, or any command
  that pushes to PyPI or a package registry.
- **No AWS interactions**: Do not execute `ecr-upload`, `step-function-trigger`, `s3-utils put/cp/ls`,
  or run any commands or code that would contact real AWS endpoints (for example, unmocked
  `boto3`/`botocore` calls). Writing or modifying AWS-related code is allowed, but execution
  must be isolated from real AWS (e.g., via `moto`/`responses`). These affect shared cloud
  infrastructure.
- **No credential use**: Do not read, use, or reference AWS credentials or profiles even if
  they appear to be configured in the environment.
