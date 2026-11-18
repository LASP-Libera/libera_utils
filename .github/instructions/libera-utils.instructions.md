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
- **ObsID registry**: `libera_utils/data/obsid_registry.csv` and `data/cal_combiner_inputs.csv` are
  the source of truth for calibration ObsIDs, their TRIMMED and CAL products and their combiner
  inputs; `obsids.py` loads and validates them at import, keyed by `(NomHkObsidSource, obsid)`
  because RAD and WFOV ObsID numbers collide. Read `doc/source/user-docs/obsid_registry.md`,
  including "Import-time checks" and "Extending the registry", before changing either file or
  adding an ObsID → product mapping anywhere else.

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
