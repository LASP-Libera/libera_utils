# Release Process
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


# Building and Distribution

1. Ensure that `poetry` is installed by running `poetry --version`.
   
2. To build the distribution archives, run `poetry build`.
   
3. To upload the wheel to Nexus, run `poetry publish --repository lasp-pypi`. Note that the repository, which is 
   named `lasp-pypi` in this example must first be configured according to the Poetry docs 
   (here)[https://python-poetry.org/docs/repositories/#using-a-private-repository]. To configure the repo for 
   publishing, run 
   ```
   poetry config repositories.lasp-pypi https://artifacts.pdmz.lasp.colorado.edu/repository/lasp-pypi/
   ```
   Note that the trailing slash is required at the end of the URL.