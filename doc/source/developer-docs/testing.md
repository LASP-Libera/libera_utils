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
pytest -m "not integration and not e2e" tests
```

End-to-end tests (`e2e`) are deselected by default through `addopts = "-m 'not e2e'"` in `pyproject.toml`.
A `-m` given on the command line replaces that one rather than combining with it, so any command-line `-m`
must exclude `e2e` itself, as above.

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

## End-to-End Tests

End-to-end tests live in `tests/e2e` and carry the `e2e` mark, which the default `pytest` run deselects. They run the
library on golden input too large for the repository. The L1A test parses, trims and writes the daily L1A product of
every APID in the DITL2 golden day 2026-07-12, and checks each product's day window and data-time coverage. A full run
takes about 20 minutes and peaks near 8 GiB of memory, for the WFOV APID.

### When to run them

Run them on your branch before asking for review of a change to any of:

- packet parsing, the XTCE definitions or `l1a_processing_configs.yml`;
- packet ordering, slicing, the day-window trim or day coverage;
- the L1A product definitions or NetCDF product writing.

Unit tests cover the logic on small synthetic data; the E2E tests are what catch a change that only fails at the scale
and on the quirks of real telemetry, such as duplicate times or a packet clock that steps backward. Other changes do
not need them. GitHub Actions also runs them daily on `main`, so a failure there points at a change merged since the
last passing run.

### Run them locally

Run them on their own, never in the same `pytest` session as other tests: the unit tests' AWS mocks replace real
credentials for the whole session, and the E2E fixtures fail fast when they find them.

| Variable                 | Meaning                                                                    |
| ------------------------ | -------------------------------------------------------------------------- |
| `LIBERA_E2E_GOLDEN_ROOT` | The golden root: an `s3://` URI, or a local directory with the same layout |
| `LIBERA_E2E_OUTPUT_DIR`  | A new or empty local directory for the run's products, logs and report     |

With a local copy of the golden input (see Golden input, below):

```bash
export LIBERA_E2E_GOLDEN_ROOT=~/libera-e2e-golden
LIBERA_E2E_OUTPUT_DIR=~/libera-e2e-output/$(date +%Y%m%dT%H%M%S) pytest -m e2e -p no:randomly tests/e2e
```

With the golden input in S3, also set a profile that can read it, and its region. Each object is downloaded to a
temporary directory first, so the run needs free disk for the whole golden input:

```bash
export AWS_PROFILE=<profile> AWS_REGION=us-west-2
export LIBERA_E2E_GOLDEN_ROOT=s3://<golden bucket>/golden
LIBERA_E2E_OUTPUT_DIR=~/libera-e2e-output/$(date +%Y%m%dT%H%M%S) pytest -m e2e -p no:randomly tests/e2e
```

Run one APID with `-k`, e.g. `-k icie_nom_hk`. Missing golden objects fail the run, naming each missing key; they are
not skipped. A run's results are in `$LIBERA_E2E_OUTPUT_DIR/reports/run_report.json`, which Outputs and the run report,
below, describes.

To share a local run, upload its outputs to the output root, which needs `s3:PutObject` and `s3:PutObjectTagging` there:

```bash
python -m tests.e2e.upload_outputs "$LIBERA_E2E_OUTPUT_DIR" s3://<golden bucket>/test_run_output
```

It files the run under `libera_utils/<commit>/local-<output directory name>/`, taking the commit from the run report.

### Run them in CI

`.github/workflows/e2e-tests.yml` runs the E2E tests against the golden input in S3 and uploads the outputs. It runs
daily on `main` and on demand on any branch, never on pull requests, so a fork cannot request its AWS role. To run it on
your branch, open the repository's **Actions** tab, select **E2E Tests**, choose **Run workflow** and pick the branch,
or:

```bash
gh workflow run e2e-tests.yml --ref <branch>
gh run watch
```

Anyone with write access to the repository can start it. The run's summary page shows a table of each APID's status,
packet counts, peak memory, run time, day coverage and warning count, followed by each APID's log findings. The full
outputs are in S3 under `<output root>/libera_utils/<commit>/<run id>-<attempt>/`.

The job runs in the `libera-utils-e2e` GitHub Environment, the only place its AWS role trusts. The environment holds
three secrets, so that no bucket name appears in the public logs:

| Secret            | Value                                                                    |
| ----------------- | ------------------------------------------------------------------------ |
| `E2E_ROLE_ARN`    | The role the job assumes through GitHub's OIDC provider                  |
| `E2E_GOLDEN_ROOT` | `s3://` URI of the golden root                                           |
| `E2E_OUTPUT_ROOT` | `s3://` URI of the output root, shared with other repositories' E2E runs |

The role reads the golden root and writes, with tags, under the output root.

### What the L1A test checks

The L1A test checks each daily product's day window, and checks the coverage of its data-time axis against the APID's
`day_coverage` policy in `l1a_processing_configs.yml`. The window checks compare the product with the parsed input, so
only the coverage check fails when the golden input lacks files whose data reach the day.

### Golden input

The golden root holds the keys listed in `tests/e2e/l1a_ditl2_golden_manifest.csv`, each with its SHA-256. The L1A
test's keys are `l1a/ditl2/<ground CCSDS name>`: the demuxed DITL2 2-hour files of each L1A APID with a product
definition whose data times reach the golden day 2026-07-12 or its 10-minute day buffer. The files are named by
capture time, which for most APIDs matches their data time, so their set is the day's files plus one either side. WFOV
SCI images reach the capture hours after their camera time, so its set runs from `2026_193_00` to `2026_194_10`. A local
copy only needs that layout, for example a symlink:

```bash
mkdir -p ~/libera-e2e-golden/l1a
ln -s /path/to/2hr_demux ~/libera-e2e-golden/l1a/ditl2
export LIBERA_E2E_GOLDEN_ROOT=~/libera-e2e-golden
```

A local root is read in place. Each file's SHA-256 is checked against the manifest, so a changed golden object fails
the run.

### Outputs and the run report

`LIBERA_E2E_OUTPUT_DIR` must be a new or empty directory; a run refuses one that already holds files, so its outputs
are only its own. Set it on the command line, as above, so each run gets its own.

Each run writes, under `LIBERA_E2E_OUTPUT_DIR`:

- `products/`: the daily L1A product of each APID;
- `logs/`: one DEBUG log per APID;
- `reports/run_report.json`: the golden day, the manifest's SHA-256, the `libera_utils` and Python versions, the
  commit (`GITHUB_SHA` in CI, else the local `HEAD`, suffixed `-dirty` when the tree has uncommitted changes), the
  workflow run ID in CI, the defined L1A APIDs the golden input lacks, and per APID the input file and packet
  counts, the parse, trim and write times in seconds, the peak resident memory in bytes, the time-axis coverage
  of the daily product, and `log_findings`: every `libera_utils` WARNING the run logged, such as duplicate data times
  and packet-time inversions along `SRC_SEQ_CTR`, plus the INFO packet-order summary from `packet_ordering` and
  `packet_slicing`, each as its level, logger and message.

Peak memory is this process's resident set size, sampled every 50 ms. The parse runs in-process, so this is the
whole parse's memory.

`python -m tests.e2e.upload_outputs` uploads a run to `<output root>/libera_utils/<commit>/<run id>/`, keeping the
`products/`, `logs/` and `reports/` subdirectories; the repository name leads so that other repositories can share the
output root. Each object is tagged `kind=products`, `kind=logs` or `kind=reports`. Lifecycle rules on the bucket, set
outside this repository, filter on that tag to expire products after 14 days and logs after 6 months; reports are kept.

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
