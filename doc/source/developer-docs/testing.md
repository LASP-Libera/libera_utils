# Testing

Testing is run with `pytest`. To run all tests, make sure the dev dependencies are installed and run:

```bash
# Run every lane, including the ones that contact NAIF
pytest tests
```

Pytest configuration is stored in `pyproject.toml` under `[tools.pytest.ini_options]`.

## Test lanes

Tests are stored in the `tests` directory and divided into three lanes. Which lane a test belongs
in is decided by what it _depends on_, not by how important it is.

| Lane        | Directory           | Marker                    | Runs     | Admission rule                                                                                                 |
| ----------- | ------------------- | ------------------------- | -------- | -------------------------------------------------------------------------------------------------------------- |
| unit        | `tests/unit`        | none                      | every PR | Exercises one function or class. No network, no external process.                                              |
| integration | `tests/integration` | `pytest.mark.integration` | every PR | Several components together, or a real subprocess such as `mkspk`/`msopck`. Everything it reads is checked in. |
| e2e         | `tests/e2e`         | `pytest.mark.e2e`         | daily    | Needs a live external service, or is too slow to sit in front of a PR.                                         |

```bash
pytest -m "not integration and not e2e" tests   # unit only
pytest -m "not e2e" tests                       # what a PR runs
pytest -m e2e tests                             # what the daily build adds
```

Markers are set per module with `pytestmark = pytest.mark.integration` (or `.e2e`) rather than per
test, so a module belongs to exactly one lane.

The PR and daily GitHub workflows both call `_run-tests.yml`, passing `lane: pr` or `lane: daily`.
Nothing else distinguishes them: the daily build exists so that an outage at NAIF, or a dependency
that drifted over a quiet week, does not block a pull request.

### The network guard

Every test outside the `e2e` lane runs under `block_outbound_network` (an autouse fixture in
`tests/conftest.py`), which raises if the test opens a socket to a remote host. Mocking layers such
as `responses` and `moto` sit above the socket and are unaffected.

This is enforced structurally because vigilance was not enough. A unit test that patched the method
it was written about still reached the NAIF server through its _setup_ path -- `load_static_kernels`
calls `load_naif_kernels` first -- and spent up to 34 seconds a run doing it, unnoticed for months.
If the guard fires, either mock the transport or move the test into `tests/e2e` and mark it.

## Writing a test here

These are the conventions the kernel and geolocation tests were reorganised around
(LIBSDC-703). They generalise to the rest of the package.

### Test the step, not the aggregate

Assert on the thing the code under test actually produces. A test whose subject is a _kernel_
asserts on the kernel -- its coverage span, the rotation angle read back out, the quaternion
convention. A test whose subject is a _frame definition_ asserts on pointing. Reaching for a
downstream number because it is easy to obtain lets an error anywhere upstream of it pass.

The corollary is that the same input feeding the same number is not necessarily duplicate
coverage: what differs is the subject.

### Precision at the step, tolerance at the end

A step-level test should assert to the precision the step is actually capable of -- a round trip
through an encoder correction closes to machine precision, so assert that, not `atol=1e-3`.
Loose bounds belong only at the end-to-end level, where many small errors legitimately accumulate.

This is not a style preference. The frozen kernel fixtures went stale for eight months while the
end-to-end geolocation assertion kept passing, because a 1.28 km median footprint shift fits
comfortably inside a tolerance wide enough to survive limb geometry. The three step-level
assertions added alongside it fail immediately against the same stale kernels.

### Golden values are scoped to the step they validate

Where an external source has given us a known answer -- engineering line-of-sight vectors, a
reference ground track, a CERES product -- use it against the single step that produces it, and
say in the module docstring where it came from and what it establishes. A golden number checked
several steps downstream of its source no longer tells you which step broke.

Test modules descended from a formal internal validation carry a `Validation provenance` section
naming it. Those assertions must not be weakened without re-running the validation.

### Fixtures, and what they cost

Prefer the custom [fixtures](https://docs.pytest.org/en/stable/how-to/fixtures.html#requesting-fixtures)
in `tests/plugins` over ad-hoc setup; they are made available to pytest by `tests/conftest.py`.
Geometry helpers that take arguments and return values, rather than participating in injection, live
in `tests/helpers.py`.

Generating SPICE kernels is expensive and generating them is usually not the subject of the test.
`tests/test_data/dynamic_kernels` holds a frozen set standing in for the output of the SPICE
processing step, which is also what production hands the L1B container. Rebuild it with
`tests/fixture_generation/generate_dynamic_kernels.py`, whose module docstring documents the source
data and the coverage window. Regenerating it changes the geometry downstream tests see, and
`libera_rad` carries a copy of the same fixture set.

### Parametrize repeats -- but check the parameter does something

[Parametrization](https://docs.pytest.org/en/stable/example/parametrize.html) is the right way to run
one test body over several kernel types or input shapes, rather than duplicating the body. Confirm
the parameter actually changes the input: four AWS tests were parametrized over three path wrapper
types that each stringified identically before use, costing 34 seconds a run for one test's worth of
coverage.

### Test order is randomized

`pytest-randomly` randomizes test order so that hidden inter-test dependencies surface. The seed is
printed at the start of a run; reproduce an ordering with `pytest --randomly-seed=<seed>`. A test
that only passes in a particular order is a bug in the test, not in the plugin.

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

## Test Profiling

### Time Profiling

We use the `pytest-profiling` plugin for time profiling of tests.

To create profiling output for a test:

```
pytest --profile tests/test_module.py::test_specific_test[PARAM_ID]
```

This generates a `prof` directory. To visualize the results, we have a tool called `snakeviz` included in dev dependencies.

To visualize profiling results with `snakeviz`:

```
snakeviz prof/combined.prof
```

It will start a web server with a navigable GUI of timing profiles for each part of the stack.

### Memory Profiling

We use the `memory-profiler` package for profiling memory usage in tests.
Documentation here: https://github.com/pythonprofilers/memory_profiler

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

# Testing Another Package Against a Working Version of Libera Utils

When developing Libera Utils, it is often useful to test that package against a specific commit hash of Libera Utils to make sure it works as expected.

Update the `pyproject.toml` file of the dependent project with a reference to the specific git hash of Libera Utils.

```toml
[tool.poetry.dependencies]
libera_utils = { git = "ssh_or_http_url_to_libera_utils_repo", rev = "abc123def456..." }
```

If your pyproject.toml is using the PEP 621 format:

```toml
[project]
dependencies = [
    "libera_utils @ git+ssh_or_http_url_to_libera_utils_repo@abc123def456..."
]
```

Then uninstall the old version (if you don't, Poetry sometimes assumes it's already installed if the version isn't different in the specific commit you want to reference) and reinstall dependencies:

```
pip uninstall libera_utils
poetry lock && poetry sync
```
