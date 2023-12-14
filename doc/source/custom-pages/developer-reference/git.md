# Git Usage

## Basic Workflow

This workflow is knows as [GitFlow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow)

This is the ideal order of events. If you know what you're doing, it is possible to deviate from this process in 
minor ways.
1. Create a branch from `dev` for a feature. Name it based on your feature or ticket. 
   e.g. `feature/LIBSDC-XXX-add-something-cool`
2. Add commits.
3. (Optional) Put up a PR prefixed with `WIP:` to signify a work in progress while still allowing team members visibility.
4. Ensure your branch is rebased onto `dev`
5. Ensure your changes are squashed into a single commit (e.g. by running `git rebase -i dev` and resolving conflicts).
6. Put up a PR to merge into `dev`
7. Wait for review.
8. Merge using the fast-forward only strategy in Bitbucket. If properly squashed, should only add 1 commit.

_If any of this doesn't make sense, don't just run commands! Ask someone! 
You can really hose your local repo state and even lose your work if you don't know what you're doing._

## Reasons for Rebasing

Rebasing allows us to keep our git history completely linear and avoids creating unnecessary merge commits.

## Reasons for Squashing

Squashing reduces the total number of commits in the repository to ~1 per pull request. Since each commit contains 
a full snapshot of the codebase, this drastically reduces the amount of storage necessary to host our git repo 
and makes life slightly easier for our beloved Web Team.

# Git LFS Usage

We use Git LFS to store large files in a way that doesn't blow up the size of our repo on the git server. Usually 
each commit contains a snapshot of the repository at that point in time. If you store a 100MB file, your entire repo 
size will be 100MB * number of commits, which can easily balloon to a big number. Incidentally, this is also 
why we squash PRs to reduce the number of commits in the main branch over time.

[Git LFS Documentation](https://git-lfs.com/)

[Tutorial on Using Git LFS](https://sabicalija.github.io/git-lfs-intro/)

[Atlassian Docs on Git LFS](https://www.atlassian.com/git/tutorials/git-lfs)

## Set Up

Install Git LFS according to the Git LFS official documentation (linked above).

Run the following to initialize Git LFS for your user:
```shell
git lfs install
```

## A Note on Git LFS Authentication and Git GUIs

Git LFS authenticates separately from Git. Historically, Git LFS supported only HTTP auth, 
which was a huge pain because best practice is to use SSH to authenticate with Git servers
(note that GitHub has actually deprecated HTTP authentication). This situation meant that 
even if you were using SSH for git auth, you still had to provide and store HTTP credentials 
for Git LFS.

However, as of Git LFS v3.0, [SSH authentication is supported!](https://github.com/git-lfs/git-lfs/pull/4446)

Git GUIs will require you to configure authentication for both Git and Git LFS. It is recommended 
to use SSH for both but the configuration processes for GUI programs (and IDEs) are different 
so we're not addressing it here. GLHF!

## Tracking New Files in LFS

LFS keeps track of which files are stored in LFS via the `.gitattributes` file.

To track specific files:
```shell
git lfs track "<pattern>"  # The double quotes matter to prevent shell expansion
# e.g. track all files in every directory named test_data
git lfs track "**/test_data/*"
# This ^ grabs all the filepaths that match and writes them directly into .gitattributes
```

To track files based on a pattern in `.gitattributes`:
```text
# .gitattributes
# Track all netCDF files that live anywhere inside a test_data directory
**/test_data/**/*.nc
```

[Documentation on Pattern Syntax](https://git-scm.com/docs/gitignore#_pattern_format)

## Tracking Existing Files in LFS

If you have already committed a file and you wish to move that file to Git LFS, you can:

Add it specifically using `git lfs track` e.g.

```shell
git lfs track my_large_file.big
```
This method appears to have some magic sugar behind it that automatically removes and re-adds 
the file to git tracking.

Alternatively, you can add the appropriate pattern to `.gitattributes` and then remove and re-add the file
to git history so Git LFS picks it up.

```shell
echo "**/*.big" >> .gitattributes  # Add pattern to .gitattributes
git rm --cached my_tracked_file.big  # Remove file from git tracking
git add my_tracked_file.big  # Git LFS should pick it up at this point
```

This method is preferable if you want your files generally tracked by pattern rather than individually.

_NOTE: As a bit of a trick, you can combine the above strategies by running `git lfs track` on the pattern
you wish to store in Git LFS. Then replace the individual records added to `.gitattributes` with the appropriate generic 
pattern that matches the specific files to be tracked._


## Useful Git LFS Commands

List all files in the current git ref (branch, commit, tag, etc.) currently managed by Git LFS:
```shell
git lfs ls-files
```

Update the files in your `.git/lfs` directory with the version for your current ref:
```shell
git lfs fetch
```

Convert local pointer files to full files (from `.git/lfs` directory):
```shell
git lfs checkout
```

Combine fetch and pull into one step:
```shell
git lfs pull
```

