# Libera Science Data Processing
Science data processing algorithms for the Libera mission.


## Installation from LASP PyPI
```bash
pip install libera-sdp --index https://artifacts.pdmz.lasp.colorado.edu/repository/lasp-pypi/simple
```


## Basic Usage


### CLI Entrypoints
Depending on how you have install `libera_sdp`, your CLI runner may vary. The commands below assume that your 
virtual environment's `bin` directory is in your `PATH`. If you are developing the package, you will
likely want to use `poetry run` to run CLI commands.

#### `make-jpss-spk`
```shell
make-jpss-spk [-h] [--outdir OUTDIR] [--overwrite] packet_data_filepaths [packet_data_filepaths ...]
```


#### `make-jpss-ck`
```shell
make-jpss-ck [-h] [--outdir OUTDIR] [--overwrite] packet_data_filepaths [packet_data_filepaths ...]
```


#### `make-libera-az-el-ck`
```shell
Not yet implemented
```


## Developer Documentation
The LASP Python style guide can be found here on Confluence, here: [https://confluence.lasp.colorado.edu/x/XiqyAw]()


### Installing Optional Dependencies
To install without development dependencies, as specified in `pyproject.toml` under `tool.poetry.dev-dependencies`, run 
`poetry install --no-dev`. By default, dev dependencies are installed.

To install optional dependencies, specified in groups in `pyproject.toml` under `tool.poetry.extras`, run
`poetry install -E <name-of-group>`. For example, `poetry install -E plotting` to include plotting-related packages. 


### Testing
Testing is run with `pytest`. To run all tests, make sure the `test` optional dependencies are installed and run:
```bash
pytest tests
```

With coverage for generating reports on code coverage:
```bash
# Create coverage data (stored in .coverage)
pytest --cov=libera_sdp --junit-xml=junit.xml tests
# Generate interactive HTML coverage report
pytest --cov-report=html:coverage_report --cov=libera_sdp tests
# Generate Corbertura-compatible XML report
pytest --cov-report=xml:coverage.xml --cov=libera_sdp tests
```


### Release Process
Reference: [https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow]()

1. Create a release candidate branch named according to the version to be released. This branch is used to polish
   the release while work continues on dev (towards the next release). The naming convention is `release/X.Y.Z`

2. Bump the version of the package to the version you are about to release, either manually by editing `pyproject.toml`
   or by running `poetry version X.Y.Z` or bumping according to a valid bump rule like `poetry version patch`
   (see poetry docs).
   
3. Open a PR to merge the release branch into master. This informs the rest of the team how the release 
   process is progressing as you polish the release branch.

4. When you are satisfied that the release branch is ready, merge the PR into `master`. 

5. Check out the `master` branch, pull the merged changes, and tag the newly created merge commit with the 
   desired version `X.Y.Z` and push the tag upstream. 
   
   ```bash
   git tag -a X.Y.Z -m "version release X.Y.Z"
   git push origin X.Y.Z
   ```
   
6. Checkout the tag you just created (ensures proper behavior of setuptools_scm) and build the package (see below).
   Check that the version of the built artifacts is as you expect (should match the version git tag).
   
7. Optionally distribute the artifacts to PyPI/Nexus if desired (see below).
   
8. Open a PR to merge `master` back into `dev` so that any changes made during the release process are also captured
   in `dev`. 


### Building and Distribution

1. Ensure that `poetry` is installed by running `poetry --version`.
   
2. To build the distribution archives, run `poetry build`.
   
3. To upload the wheel to Nexus, run `poetry publish --repository lasp-pypi`. Note that the repository must first
   be configured according to the Poetry docs 
   (here)[https://python-poetry.org/docs/repositories/#using-a-private-repository]
