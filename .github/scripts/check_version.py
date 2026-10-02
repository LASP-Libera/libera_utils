#!/usr/bin/env python3
"""Check that the changelog heading, the package version and the release tags agree.

Pre-commit runs this, as the `version-check` hook, on a change to pyproject.toml or
doc/source/changelog.md. Two checks:

1. The first `## <version>` heading in doc/source/changelog.md equals the `version` in
   pyproject.toml.
2. That version is at or above the highest bare-version tag (`5.11.1`, not `v5.11.1` or
   `5.11.1rc1`). That passes `main`, where the version equals the newest tag, and a bump, and
   fails a downgrade or a stacked branch left below a newer release.

A release is cut by pushing a tag, and the package published carries pyproject.toml's version.
A heading that disagrees mislabels the release notes, and a version below the newest tag
predates a published release (rule R-012, with its reasoning in libera_llm_tooling at
standards/archive/libera_utils/R-012.md).

Not checked: whether a change is minor or patch. That is the author's call, under R-012 in the
instruction file.

Prints one ok or MISMATCH line per check, naming both values, and exits 1 on any MISMATCH.
Exits 2 with one CANNOT RUN line saying why when the check cannot run: pyproject.toml is
unreadable or has no version line, the changelog is unreadable or has no `## <version>`
heading, or there is no bare-version tag. The readers below raise ValueError for a
missing line; main turns that, an unreadable file or a failed `git tag` into exit 2.
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
    args = parser.parse_args(argv)

    try:
        version = pyproject_version(args.pyproject)
        heading = changelog_heading(args.changelog)
        top = highest_tag(args.tags.read_text().splitlines() if args.tags else tags_from_git())
        release(version)
    except (OSError, ValueError, subprocess.CalledProcessError) as e:
        print(f"  CANNOT RUN {e}")
        return 2
    failed = 0

    if heading == version:
        print(f"  ok       changelog heading: {heading}")
    else:
        failed += 1
        print(f"  MISMATCH changelog heading {heading} does not equal pyproject.toml version {version}")

    if release(version) >= release(top):
        print(f"  ok       version {version} is at or above the highest tag {top}")
    else:
        failed += 1
        print(f"  MISMATCH pyproject.toml version {version} is below the highest tag {top}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
