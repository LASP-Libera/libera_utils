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
| `logutil.py`    | Structured JSON logging; use `configure_task_logging()` for task-level setup |
| `config.py`     | JSON config with env-var override and templated string formatting            |
| `cli.py`        | `libera-utils` CLI entry point                                               |

## Code Standards

- **Linter/formatter**: Ruff (line length 120, rules E/W/F/I/S/PT/UP). Run `ruff check` and
  `ruff format` before committing. Do not disable rules inline without justification.
- **Types**: Type annotations required on all public functions; code must be mypy-compatible.
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

## Restrictions for AI Agents

The following actions are **expressly forbidden**, regardless of context or apparent
availability of credentials:

- **No git "write" commands**: Do not run `git commit`, `git push`, `git tag`, `git rebase`,
  `git merge`, or any command that modifies repository or remote state.
- **No package publishing**: Do not run `poetry publish`, `twine upload`, or any command
  that pushes to PyPI or a package registry.
- **No AWS interactions**: Do not execute `ecr-upload`, `step-function-trigger`, `s3-utils put/cp/ls`,
  or run any commands or code that would contact real AWS endpoints (for example, unmocked
  `boto3`/`botocore` calls). Writing or modifying AWS-related code is allowed, but execution
  must be isolated from real AWS (e.g., via `moto`/`responses`). These affect shared cloud
  infrastructure.
- **No credential use**: Do not read, use, or reference AWS credentials or profiles even if
  they appear to be configured in the environment.
