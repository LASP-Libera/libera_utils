# Testing

Testing is run with `pytest`. To run all tests, make sure the dev dependencies are installed and run:

```bash
# Run all unit and integration tests
pytest tests
```

Pytest configuration is stored in `pyproject.toml` under `[tools.pytest.ini_options]`.

Tests are stored in the `tests` directory and are divided into
`integration` and `unit` tests. Unit tests check for expected behavior of small pieces of the
package (e.g. a single function). Integration tests check for "larger" behavior across wider swaths of the package,
testing that components function together cohesively. Integration test modules contain the pytest mark
`pytestmark = pytest.mark.integration`. This allows you to selectively exclude running integration tests with

```
pytest -m "not integration" tests
```

Unit tests heavily utilize pytest parameterization in order to
run a single test against many sets of inputs and outputs without duplicating code (see
[parameterize]
(https://docs.pytest.org/en/stable/example/parametrize.html#set-marks-or-test-id-for-individual-parametrized-test)).

We also heavily utilize custom pytest [fixtures]
(https://docs.pytest.org/en/stable/how-to/fixtures.html#requesting-fixtures),
defined in `tests/plugins`. These plugins are made available to
pytest in `conftest.py`, the main test configuration file for pytest.

In order to better ensure test independence, we use `pytest-randomly` to randomize the order of tests.
The random seed used to set the order is printed at the beginning of a test run. You
can re-run the tests with a specific random seed by running `pytest --randomly-seed=<seed>`.

## Generating Coverage and Test Reports

With coverage for generating reports on code coverage:

```bash
# Create coverage data (stored in .coverage)
pytest --cov=libera_utils --junit-xml=junit.xml
# Generate interactive HTML coverage report
pytest --cov-report=html:coverage_report --cov=libera_utils
# Generate Corbertura-compatible XML report
pytest --cov-report=xml:coverage.xml --cov=libera_utils
```

## Testing in Docker

To run the unit tests in docker, run

```shell
docker-compose up [--build] --exit-code-from=tests tests --attach=tests
```

This runs the `tests` container service defined in the `docker-compose.yml` file. The `--build` option forces
docker to rebuild the testing container image before running (e.g. if things have changed).

### Copying Test Report Artifacts from Docker

When we run tests in Docker on Jenkins, we often want to copy and save Corbertura and JUnit test reports. Jenkins
has facility for doing this easily with

```Groovy
always {
    junit '**/*junit.xml'
    cobertura coberturaReportFile: '**/*coverage.xml'
}
```

The challenge when running in Docker is to make these test artifacts available to Jenkins. By default these files
exist only inside the Docker container so we must copy them out. Do this with

```shell
docker-compose --exit-code-from tests up tests
docker-compose cp tests:/path/to/report.xml .
```

## Static Analysis

NASA requirements document NPR7150.2C requires that we perform static analysis of our codebase to check for
common vulnerabilities and statically detectable code weaknesses and vulnerabilities (CWEs and CVEs).

We use the [Ruff](https://docs.astral.sh/ruff/#ruff) tool to perform a comprehensive static analysis of our code. Ruff includes configurations for
pycodestyle, flake8, Bandit, and more. It is configured to run automatically as a pre-commit hook via the
pre-commit tool.

To manually run all ruff checks, run

```shell
ruff check
```

Configuration for ruff is declared in `pyproject.toml`.

## Pre-commit Hooks

To ensure code quality with minimal effort on the part of developers, we use pre-commit to run automatic linting
before commits are allowed. Configuration for pre-commit is in `.pre-commit-config.yaml`.

To install pre-commit, run:

```shell
pre-commit install
```

To run all hooks on all files manually, run:

```shell
pre-commit run --all-files
```
