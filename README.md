# Libera Science Data Processing
Science data processing algorithms for the Libera mission.


## Installation from LASP PyPI
```bash
pip install libera-sdp --index https://artifacts.pdmz.lasp.colorado.edu/repository/lasp-pypi/simple
```


## Basic Usage

### CLI Entrypoints

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


### Git-LFS
We use Git-LFS to manage large file storage while maintaining git clone performance. To add a large file to repository,
first consider whether it's really necessary. Upon that consideration, run `git-lfs track <pattern>` where `<pattern>` 
is a pattern that matches the file you are adding. The patterns are the same as those you might use 
in .gitignore (though it's not an intuitive syntax unfortunately).

To see a list of files managed by Git-LFS, run `git lfs ls-files`.

If you have accidentally stored a large file normally, you must "migrate" and "import" the file into the git-lfs
system. To see a dry run of what you will be importing, run 
`git lfs migrate info --everything --include="path/to/large_file.big"`

[See here](https://github.com/git-lfs/git-lfs/blob/main/docs/man/git-lfs-migrate.1.ronn) for more documentation on 
importing objects into git-lfs.

### Installing Optional Dependencies
The package comes with three sets of options: `dev`, `test`, and `build`. The `build` option contains all the 
dependencies for both `dev` and `test`. To install an option (e.g. `dev`):
```bash
pip install "libera_sdp[dev]" --index https://artifacts.pdmz.lasp.colorado.edu/repository/lasp-pypi/simple
```
Or from a local copy, from the repo root:
```bash
pip install ".[dev]"
```


### Testing
Testing is run with `pytest`. To run all tests, make sure the `test` optional dependencies are installed and run:
```bash
pytest tests
```

With coverage for generating reports on code coverage:
```bash
# Create coverage data (stored in .coverage)
coverage run --source=libera_sdp --module pytest -ra --junit-xml=libera_sdp_unit_test_report.xml tests
# Generate interactive HTML coverage report
coverage html -d coverage_report
# Generate Corbertura-compatible XML report
coverage xml -o libera_sdp_corbertura_report.xml
```

For convenience during development, we have a script called `test.sh`. This script will run the tests with coverage, 
generate an interactive HTML report (if tests all pass), and open the report in your default browser. The script only
supports Darwin and Linux platforms.

### Release Process
Reference: [https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow]()

1. Create a release candidate branch named according to the version to be released. This branch is used to polish
   the release while work continues on dev (towards the next release). The naming convention is `release/X.Y.Z`
   
2. Open a PR to merge the release branch into master. This informs the rest of the team how the release 
   process is progressing as you polish the release branch.

3. When you are satisfied that the release branch is ready, merge the PR into `master`. 

4. Check out the `master` branch, pull the merged changes, and tag the newly created merge commit with the 
   desired version `X.Y.Z` and push the tag upstream. 
   
   ```bash
   git tag -a X.Y.Z -m "version release X.Y.Z"
   git push origin X.Y.Z
   ```
   
5. Checkout the tag you just created (ensures proper behavior of setuptools_scm) and build the package (see below).
   Check that the version of the built artifacts is as you expect (should match the version git tag).
   
6. Optionally distribute the artifacts to PyPI/Nexus if desired (see below).
   
7. Open a PR to merge `master` back into `dev` so that any changes made during the release process are also captured
   in `dev`. 


### Building and Distribution

1. Ensure build dependencies are installed by running `pip install ".[build]"` from the repo root.
   
2. To build the distribution archives, run `python -m build`. 
   This should produce a `build` directory, a `dist` directory, and an `egg-info` directory.
   
3. Upload artifacts to Nexus:

   ```bash
   twine upload --repository-url https://artifacts.pdmz.lasp.colorado.edu/repository/lasp-pypi/ dist/*
   ```

   Note that the trailing slash on the URL is required.


### Optional Configurations for LASP PyPI
You can set up your system with named repositories so you don't have to remember the Nexus URL by
creating a `.pypirc` file in your home directory:

```ini
# .pypirc

[distutils]
index-servers=lasppypi

[lasppypi]
repository: https://artifacts.pdmz.lasp.colorado.edu/repository/lasp-pypi/
username: <someusername>
```

Note again that the trailing slash on the URL is absolutely necessary.

With .pypirc in place, you can then push distributions like so:

```bash
twine upload -r lasppypi dist/*
```
