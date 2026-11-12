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

| Module          | Responsibility                                                                 |
| --------------- | ------------------------------------------------------------------------------ |
| `aws/`          | S3, ECR, and Step Functions helpers — CLI-facing only; see restrictions        |
| `io/`           | NetCDF product definitions (Pydantic), file naming, UMM-G, cloud-aware I/O     |
| `l1a/`          | Packet parsing, XTCE configs, packet slicing, day-window trim, day coverage    |
| `libera_spice/` | SPICE kernel generation via SpiceyPy + Curryer                                 |
| `constants.py`  | Canonical enums: `DataLevel`, `DataProductIdentifier`, `LiberaApid`            |
| `obsids.py`     | Loader/API over the ObsID catalog CSVs in `data/` (registry + combiner inputs) |
| `logutil.py`    | Structured JSON logging; use `configure_task_logging()` for task-level setup   |
| `config.py`     | JSON config with env-var override and templated string formatting              |
| `cli.py`        | `libera-utils` CLI entry point                                                 |

## Code Standards

- **Linter/formatter**: Ruff (line length 120, rules E/W/F/I/S/PT/UP). Run `ruff check` and
  `ruff format` before committing. Do not disable rules inline without justification.
- **Types**: Type annotations required on all public functions; code should follow standard typing best practices and satisfy Ruff’s typing-related rules (mypy is not currently run in CI).
- **Docstrings**: Numpy-style on all public symbols.
- **Pre-commit**: Hooks are required (`pre-commit install`). Never bypass with `--no-verify`.
- **To-Do Items**: Must reference a JIRA tag (e.g. `TODO[LIBSDC-1234]` or `TODO[CURRYER-1234]`).
- **Changelog**: `CHANGELOG.md` at the repo root follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
  Record every user-visible change under `## [Unreleased]` in the matching `### Added` / `### Changed` /
  `### Deprecated` / `### Removed` / `### Fixed` / `### Security` subsection, prefixing backwards-incompatible entries
  with `**BREAKING:**`. Do not bump `pyproject.toml` or convert `[Unreleased]` into a versioned heading unless the PR is
  the one releasing that version; see `doc/source/developer-docs/build_release.md`.
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
- **Docs build**: Build the docs as part of testing any change to code, docstrings or `doc/`:
  `poetry install --with docgen` (Python 3.12 or newer), then `make html` in `doc/`. The Makefile passes
  `-W --keep-going`, so the build fails on any warning and lists them all. Read the Docs builds with
  `fail_on_warning: true`, and GitHub Actions does not build the docs, so a warning here is a failed
  docs check on the pull request. The build is nitpicky and documents private members, so a type in a
  private method's numpydoc `Parameters` section must resolve too: write `collections.abc.Iterable[str]`,
  not `iterable of str`.

## Key Patterns

- **NetCDF products**: Defined by Pydantic schemas (`LiberaVariableDefinition` in
  `io/product_definition.py`). Validate against schemas; do not construct raw attribute dicts.
- **File naming**: Libera filenames are parsed and built by `io/filenaming.py`. Always use
  those helpers; never hand-craft filename strings.
- **Cloud paths**: Use `cloudpathlib` abstractions so code works with both local and S3 paths.
- **XTCE configs**: Telemetry packet field definitions live in `libera_utils/data/`. Do not
  hardcode packet offsets or field names outside of these config files.
- **Day coverage policies**: Every entry in `data/l1a_processing_configs.yml` needs a `day_coverage`
  block (`PacketConfiguration.day_coverage`); loading the packet configurations fails without one.
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
  ObsID, a TRIMMED product claimed by more than one ObsID, or a CAL product that no
  `ProcessingStepIdentifier` produces. The in-memory `OBSID_REGISTRY` is keyed by
  `(NomHkObsidSource, obsid)`, not `obsid` alone, because RAD and WFOV ObsID numbers collide
  (e.g. `256` means SWC-365NM on RAD but Darks-of-Darks on WFOV). This is more than a lookup
  table of "what ObsIDs exist" — it drives real behavior:
  - Each calibration ObsID has its own TRIMMED NOM-HK product (`NOM-HK-<CAL ProductID>-TRIMMED`),
    its own CAL product, and one combiner `ProcessingStepIdentifier` (`cal_combine_<x>`,
    `cal-combine-<x>`) that produces only that CAL product: the three are 1:1:1. All combiner steps
    share the `cal-rad-docker-repo` ECR repository. `get_trimmed_obsid_spec` and
    `get_combiner_step` go from a TRIMMED product to its ObsID and its step.
  - Downstream repos (e.g. `libera_rad`'s combiner) derive their ObsID → product mappings directly
    from this registry (via `get_trimmed_obsid_spec` / `get_obsid_spec` / `iter_trim_eligible`)
    instead of hand-maintaining a duplicate mapping per repo.
  - A companion catalog, `data/cal_combiner_inputs.csv`, maps each TRIMMED product to the L1A
    products its combiner consumes _besides that TRIMMED product_ (`COMBINER_INPUTS` /
    `get_combiner_inputs()`). The two files are cross-checked at import and must list exactly the
    same TRIMMED products, so a new calibration ObsID needs a row in both; an empty
    `required_inputs` cell means the dependency set is still undecided. A TRIMMED product plus its
    `required_inputs` is what a libera_cdk `cal-combine-*` node should declare as its
    `input-products`. `required_inputs` normally omits `l1a_icie_nom_hk_decoded`, since the
    combiner gets its NOM-HK already trimmed; listing the full-day granule would stage a redundant
    second NOM-HK input. NOM-HK is the only product the L1A preprocessor trims — a combiner subsets
    the full daily L1A inputs itself, using the time range on the TRIMMED NOM-HK filename it was
    handed.
  - A "calibration family" (e.g. the six SWC LED ObsIDs) is a way to talk about ObsIDs whose
    combiners take the same inputs. It is not a product or a step, and no catalog file records it.
  - A TRIMMED or CAL file holds one calibration event; an ObsID can recur within a day, and each
    occurrence is its own file, told apart by its filename time range. `applicable_date` (and so
    `archive_prefix`) dates these files by the UTC day of `utc_start`, not the midpoint
    (`is_calibration_event_product`). The exact ObsID is also in the `ICIE__SW_OBSID_*` variable
    the trimmed file carries.
  - When adding a new calibration ObsID, add its CAL, TRIMMED and `cal_combine_*` members in
    `constants.py`, then its rows in `data/obsid_registry.csv` and `data/cal_combiner_inputs.csv`,
    rather than adding a parallel ObsID → product mapping in a downstream repo. Edit the CSVs with a
    text editor or the `csv` module, never a spreadsheet app that may rewrite quoting —
    descriptions contain commas.
  - **Note**: the list of ObsIDs in this repo is meant for practical purposes of science data
    processing and is a subset of the instrument level source of truth of all ObsIDs which is owned
    by the engineering team and is available in internal team documentation

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
