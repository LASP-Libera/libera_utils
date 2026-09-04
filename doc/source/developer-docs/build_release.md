# Build and Release

Our release pipeline is an automated process that allows us to build and release the `libera-utils` package in a
consistent manner whenever tags are pushed to the repo. When a tag is pushed, the CI/CD pipeline will automatically
build the package and upload it to PyPI with the version number specified in the `pyproject.toml` file.

**Note:** The version number in `pyproject.toml` is the source of truth for the version of the package, not the tag
text. The tag text is used to trigger the CI/CD pipeline, but the version number in `pyproject.toml` is what is used
when building and publishing the package to PyPI. The convention is to use the tag text as the version number for
consistency.

## Release Process for Major and (Usually) Minor Releases

Every PR (with a few exceptions) should record its changes in `CHANGELOG.md`, which follows the
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format:

- Add entries under the `## [Unreleased]` heading at the top, in the appropriate `### Added`, `### Changed`,
  `### Deprecated`, `### Removed`, `### Fixed`, or `### Security` subsection. Prefix backwards-incompatible entries with
  `**BREAKING:**`.
- The PR that releases a version (usually the last PR before a tag, or the only PR when a single change is released on
  its own) bumps the version in `pyproject.toml`, renames `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD`, inserts a
  fresh empty `## [Unreleased]` heading above it, and adds the `[X.Y.Z]: .../compare/<previous>...X.Y.Z` link at the
  bottom of the file (updating the `[Unreleased]` compare link to start from the new version).

When a release PR is merged, the version should already be updated. One of the lead developers is then responsible
for tagging the commit in `main` and pushing the tag to trigger the CI pipeline that pushes the release to PyPI.

To tag a commit and push the tag, use an "annotated" tag as follows:

```bash
git tag -a X.Y.Z -m "version release X.Y.Z"  # -a indicates annotated, -m is the annotation message
git push origin X.Y.Z
```

## Release Process for Pre-Releases and Testing in Dependent Packages

_Note:_ This should rarely be necessary. See [`testing.md`](testing.md) for how to test other packages against a specific
commit hash of Libera Utils without using a pre-release on PyPI.

1. In your working branch, bump the version of the package to the next pre-release version (e.g. `X.Y.Zrc1`) by editing
   `pyproject.toml`. Make sure that the version you are specifying doesn't already exist in PyPI.
2. Commit the version change.
3. Tag your working branch with the pre-release version, e.g. `X.Y.Zrc1`, and push the tag upstream.
   This will trigger the automatic build and publish by Jenkins.

   ```bash
   git tag -a X.Y.Zrc1 -m "pre-release X.Y.Zrc1"
   git push origin X.Y.Zrc1
   ```
