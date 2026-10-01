#!/usr/bin/env python3
"""Check that the package version, the changelog and the tags agree.

Two checks. The first `## <version>` heading in the changelog equals the version
`pyproject.toml` carries, always. With --bumped, that version is also greater than the
highest bare-version tag (`5.11.1`, not `v5.11.1` or `5.11.1rc1`).

The tag check is behind a flag because not every pull request bumps: a dependency update edits
`pyproject.toml` and leaves the version alone, and on `main` the version equals the newest tag,
so an unconditional tag check would fail every such pull request. The workflow passes --bumped
when the pull request's diff changes the `version =` line.

Deliberately not checked: whether a change is minor or patch. That is the author's call, and
the instruction file states the rule for it.

Exit 1 naming both values on a mismatch.
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
    match = re.search(r'^version\s*=\s*"([^"]+)"', path.read_text(), re.M)
    if not match:
        raise ValueError(f"{path} has no version line")
    return match.group(1)


def changelog_heading(path: Path) -> str:
    match = re.search(r"^## (\S+)", path.read_text(), re.M)
    if not match:
        raise ValueError(f"{path} has no '## <version>' heading")
    return match.group(1)


def tags_from_git() -> list[str]:
    # S603, S607: a literal argument list, no shell, nothing from user input.
    proc = subprocess.run(["git", "tag"], cwd=ROOT, capture_output=True, text=True, check=True)  # noqa: S603, S607
    return proc.stdout.splitlines()


def highest_tag(tags: list[str]) -> str:
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
