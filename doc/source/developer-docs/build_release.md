# Build and Release

Our release pipeline is an automated process that allows us to build and release the `libera-utils` package in a
consistent manner whenever tags are pushed to the repo. When a tag is pushed, the CI/CD pipeline will automatically
build the package and upload it to PyPI with the version number specified in the `pyproject.toml` file.

Following our [git workflow](git.md) as trunk based development, any release branch (and some feature branches) will
have updates to the version number in `pyproject.toml` to indicate the next version of the package. This is not required
for all branches, but it is recommended especially for changes that affect other packages that depend on `libera-utils`.

**Note:** The version number in `pyproject.toml` is the source of truth for the version of the package, not the tag
text. The tag text is used to trigger the CI/CD pipeline, but the version number in `pyproject.toml` is what is used
when building and publishing the package to PyPI. The convention is to use the tag text as the version number for
consistency.

## Local package building for CLI testing

To build the package locally for testing especially for the cli interface, use the following steps:

1. Ensure that you have activated a virtual environment where you would like libera-utils to be installed.
2. run `python -m pip install .` from the root of the repository.
3. You should now be able to run the `libera-utils --version` command from the command line as see that the version
   number matches the one in `pyproject.toml`.

## Release Process for Major and (Usually) Minor Releases

1. Create a release candidate branch named according to the version to be released. This branch is used to polish
   the release while work continues elsewhere (towards the next release). The naming convention is `release/X.Y.Z`

2. Bump the version of the package to the version you are about to release, either manually by editing `pyproject.toml`
   or by running `poetry version X.Y.Z` or bumping according to a valid bump rule like `poetry version minor`
   (see poetry docs).

3. Update the `CHANGELOG.md` file to include a new section for the release, and add any relevant changes
   that have been made since the last release.
4. Update any other documentation that may need to be changed as it relates to the release.

5. Open a PR to merge the release branch into main. This informs the rest of the team how the release
   process is progressing as you polish the release branch.

6. When you are satisfied that the release branch is ready and the team has approved, merge the PR into `main`. This
   should be a purely "fast-forward" merge. You can delete the release branch after merging.

7. Check out the `main` branch, pull the merged changes, and tag the newly created merge commit with the
   desired version `X.Y.Z` and push the tag upstream. **This triggers the automatic build and publish by Jenkins.
   Poetry uses the version number in `pyproject.toml` not the tag text for publishing in PyPI.**

   ```bash
   git tag -a X.Y.Z -m "version release X.Y.Z"
   git push origin X.Y.Z
   ```

## Release Process for Patch Releases

1. In your working branch, bump the version of the package to the next patch version, either manually by editing
   `pyproject.toml` or by running `poetry version`.

2. Update the `CHANGELOG.md` file to include a new section for the patch release, and add any relevant changes
   that have been made since the last release.

3. Update any other documentation that may need to be changed as it relates to the patch release.

4. Follow the standard PR process to merge the changes into `main`. This informs the rest of the team how the patch
   release process is progressing.

5. When you are satisfied that the patch release branch is ready and the team has approved, merge the PR into `main`.
   This should be a purely "fast-forward" merge. You can delete the patch release branch after merging.

6. Check out the `main` branch, pull the merged changes, and tag the newly created merge commit with the
   desired version `X.Y.Z` and push the tag upstream. **This triggers the automatic build and publish by Jenkins.**

## Release Process for Pre-Releases and Testing in Dependent Packages

1. In your working branch, bump the version of the package to the next pre-release version, either manually by editing
   `pyproject.toml` or by running `poetry version prepatch`, `poetry version preminor`, or `poetry version premajor`
   depending on the type of pre-release you are making.
2. Tag your working branch with the pre-release version, e.g. `X.Y.Z-alpha.1`, and push the tag upstream.
   This will trigger the automatic build and publish by Jenkins.

   ```bash
   git tag -a X.Y.Z-alpha.1 -m "pre-release X.Y.Z-alpha.1"
   git push origin X.Y.Z-alpha.1
   ```

## Manually Building and Distribution to Public PyPI

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
