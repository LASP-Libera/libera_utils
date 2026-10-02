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
- **Run unit tests**: `pytest -m "not integration and not e2e" tests/`
- **Run with coverage**: `pytest -m "not e2e" --cov=libera_utils --cov-report=term-missing`
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

The rules below are the team's review conventions, numbered `R-NNN` so a review comment can cite
one; they are not Claude Code's `.claude/rules/` files. Each rule's status and evidence are in
`standards/review-rules.md`, whose header carries the budget and the admission criteria. This
repository is public, so R-014 applies to everything in it. A change to a rule lands through a
reviewed pull request.

## Review Rules

What a reviewer checks here, beyond what the tools check, and what anyone writing code here
is expected to follow. This section is the only copy of each rule's wording; its tier, status,
evidence and do-not-flag sentence are in `standards/review-rules.md`.

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

### R-009 · Comments describe the code as it is, not how it got there

A comment describes the code as it is, not how it got there or where it is going, and reads
correctly with only the code around it. It carries no ticket number, historical title or context
from a conversation, a prompt or a review; forward-looking work takes R-010's tagged marker.

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

### R-013 · Parse or sort an input once, not once per consumer

An input is read, parsed or sorted once and the result handed to each consumer; a loop does not
re-read a file, re-sort a dataset or reload a definition on every pass.

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
