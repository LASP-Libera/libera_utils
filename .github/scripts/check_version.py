#!/usr/bin/env python3
"""Check that the changelog heading, the package version and the release tags agree.

CI runs this from .github/workflows/version-check.yml on every pull request. Two checks:

1. The first `## <version>` heading in doc/source/changelog.md equals the `version` in
   pyproject.toml. Always run.
2. With --bumped, that version is also above the highest bare-version tag (`5.11.1`, not
   `v5.11.1` or `5.11.1rc1`). The workflow passes --bumped only when the pull request's diff
   changes the `version =` line: a dependency update edits pyproject.toml without bumping, and
   on `main` the version equals the newest tag, so an unconditional tag check would fail both.

A release is cut by pushing a tag, and the package published carries pyproject.toml's version.
A heading that disagrees mislabels the release notes, and a version at or below the newest tag
repeats or predates a published release (rule R-012).

Not checked: whether a change is minor or patch. That is the author's call, under R-012 in the
instruction file.

Prints one ok or MISMATCH line per check, naming both values, and exits 1 on any MISMATCH.
Raises ValueError when pyproject.toml has no version line, the changelog has no
`## <version>` heading, or --bumped finds no bare-version tag.
"""

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BARE = re.compile(r"^\d+(?:\.\d+)*$")


def release(version: str) -> tuple[int, ...]:
    """The numeric release segment of a version: `5.11.2rc1` gives (5, 11, 2)."""
    match = re.match(r"^(\d+(?:\.\d+)*)", version)
    if not match:
        raise ValueError(f"{version!r} does not start with a numeric release segment")
    return tuple(int(part) for part in match.group(1).split("."))


def pyproject_version(path: Path) -> str:
    """The `version = "..."` value in pyproject.toml; raises ValueError when there is none."""
    match = re.search(r'^version\s*=\s*"([^"]+)"', path.read_text(), re.M)
    if not match:
        raise ValueError(f"{path} has no version line")
    return match.group(1)


def changelog_heading(path: Path) -> str:
    """The version in the changelog's first `## ` heading; raises ValueError when there is none."""
    match = re.search(r"^## (\S+)", path.read_text(), re.M)
    if not match:
        raise ValueError(f"{path} has no '## <version>' heading")
    return match.group(1)


def tags_from_git() -> list[str]:
    """Every tag in the repository, as `git tag` lists them."""
    # S603, S607: a literal argument list, no shell, nothing from user input.
    proc = subprocess.run(["git", "tag"], cwd=ROOT, capture_output=True, text=True, check=True)  # noqa: S603, S607
    return proc.stdout.splitlines()


def highest_tag(tags: list[str]) -> str:
    """The highest bare-version tag, compared by release segment; raises ValueError when none is bare."""
    bare = [t.strip() for t in tags if BARE.match(t.strip())]
    if not bare:
        raise ValueError("no bare-version tag found")
    return max(bare, key=release)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pyproject", type=Path, default=ROOT / "pyproject.toml")
    parser.add_argument("--changelog", type=Path, default=ROOT / "doc" / "source" / "changelog.md")
    parser.add_argument("--tags", type=Path, help="a file with one tag per line; default: git tag")
    parser.add_argument("--bumped", action="store_true", help="also require the version to exceed the highest tag")
    args = parser.parse_args(argv)

    version = pyproject_version(args.pyproject)
    heading = changelog_heading(args.changelog)
    failed = 0

    if heading == version:
        print(f"  ok       changelog heading: {heading}")
    else:
        failed += 1
        print(f"  MISMATCH changelog heading {heading} does not equal pyproject.toml version {version}")

    if args.bumped:
        tags = args.tags.read_text().splitlines() if args.tags else tags_from_git()
        top = highest_tag(tags)
        if release(version) > release(top):
            print(f"  ok       version {version} is above the highest tag {top}")
        else:
            failed += 1
            print(f"  MISMATCH pyproject.toml version {version} is not above the highest tag {top}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
