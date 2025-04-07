# Build and Release

## Local package building for CLI testing

To build the package locally for testing especially for the cli interface, use the following steps:

1. Ensure that you have activated a virtual environment where you would like libera-utils to be installed.
2. run `python -m pip install .` from the root of the repository.
3. You should now be able to run the `libera-utils --version` command from the command line as see that the version
number matches the one in `pyproject.toml`.

## Release Process
[Atlassian Git Workflow Reference:](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow)

1. Create a release candidate branch named according to the version to be released. This branch is used to polish
   the release while work continues on dev (towards the next release). The naming convention is `release/X.Y.Z`

2. Bump the version of the package to the version you are about to release, either manually by editing `pyproject.toml`
   or by running `poetry version X.Y.Z` or bumping according to a valid bump rule like `poetry version patch`
   (see poetry docs).

3. Open a PR to merge the release branch into main. This informs the rest of the team how the release
   process is progressing as you polish the release branch.

4. When you are satisfied that the release branch is ready, merge the PR into `main`. This should be a purely
   "fast-forward" merge. Do not delete the release branch when merging as you will need it later.

5. Check out the `main` branch, pull the merged changes, and tag the newly created merge commit with the
   desired version `X.Y.Z` and push the tag upstream.

   ```bash
   git tag -a X.Y.Z -m "version release X.Y.Z"
   git push origin X.Y.Z
   ```

6. Checkout the tag you just created (ensures the correct version is recorded in the build artifacts) and build the
   package (see below).
   Check that the version of the built artifacts is as you expect (should match the version git tag).

7. Optionally distribute the artifacts to PyPI/Nexus if desired (see below).

8. Open a PR to merge `release/X.Y.Z` back into `dev` so that any changes made during the release process are also captured
   in `dev`. This should be a purely "fast-forward" merge.


## Building and Distribution to Public PyPI

1. Ensure that `poetry` is installed by running `poetry --version`.

2. Checkout the tag of the version you are releasing.

3. To build the distribution archives, run `poetry build`.

4. To upload the wheel to PyPI, first set your environment variables with the API token for the correct PyPI account:
   ```shell
   export PYPI_USERNAME=__token__
   export PYPI_TOKEN=<Your API Token>
   ```
   Then run `poetry publish --username $PYPI_USERNAME --password $PYPI_TOKEN`.
   You will need the account information for the `liberasdc` PyPI account.
