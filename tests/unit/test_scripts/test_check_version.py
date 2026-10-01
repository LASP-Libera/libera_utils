"""Tests for .github/scripts/check_version.py, the script the Version check workflow runs.

Covers the changelog heading against the pyproject.toml version, the bumped version against the
highest bare tag, and the ValueError raised when a file lacks the line the check reads. The
script is imported by file path, since .github/scripts is not a package.
"""

import importlib.util
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / ".github" / "scripts" / "check_version.py"
_spec = importlib.util.spec_from_file_location("check_version", SCRIPT)
check_version = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check_version)


def _args(tmp_path: Path, version: str, heading: str, top_tag: str, bumped: bool) -> list[str]:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(f'[project]\nname = "libera_utils"\nversion = "{version}"\n')
    changelog = tmp_path / "changelog.md"
    changelog.write_text(f"# Version Changes\n\n## {heading}\n\n- FIX: something\n\n## 5.0.0\n")
    tags = tmp_path / "tags.txt"
    tags.write_text(f"5.0.0\nv9.9.9\n{top_tag}\n5.10.0rc1\n")
    args = ["--pyproject", str(pyproject), "--changelog", str(changelog), "--tags", str(tags)]
    return [*args, "--bumped"] if bumped else args


@pytest.mark.parametrize(
    ("version", "heading", "top_tag", "bumped", "expected", "mismatches"),
    [
        pytest.param("5.11.2", "5.11.2", "5.11.1", True, 0, 0, id="bumped-above-tag"),
        pytest.param("5.10.9", "5.10.8", "5.10.9", True, 1, 2, id="heading-behind-and-version-at-tag"),
        pytest.param("5.11.1", "5.11.1", "5.11.1", False, 0, 0, id="no-bump-at-tag"),
    ],
)
def test_check_version(tmp_path, capsys, version, heading, top_tag, bumped, expected, mismatches):
    assert check_version.main(_args(tmp_path, version, heading, top_tag, bumped)) == expected
    out = capsys.readouterr().out
    assert out.count("MISMATCH") == mismatches
    if mismatches:
        for value in (version, heading, top_tag):
            assert value in out


def test_release_reads_the_numeric_segment_before_an_rc_suffix():
    assert check_version.release("5.11.2rc1") == (5, 11, 2)
    assert check_version.release("5.11.2") > check_version.release("5.11.1")


def test_release_raises_on_a_version_with_no_numeric_segment():
    with pytest.raises(ValueError, match="numeric release segment"):
        check_version.release("dev")


def test_pyproject_version_raises_without_a_version_line(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "libera_utils"\n')
    with pytest.raises(ValueError, match="has no version line"):
        check_version.pyproject_version(pyproject)


def test_changelog_heading_raises_without_a_version_heading(tmp_path):
    changelog = tmp_path / "changelog.md"
    changelog.write_text("# Version Changes\n\nNothing yet.\n")
    with pytest.raises(ValueError, match="has no '## <version>' heading"):
        check_version.changelog_heading(changelog)


def test_highest_tag_ignores_prefixed_and_rc_tags_and_raises_with_none_bare():
    assert check_version.highest_tag(["v9.9.9", "5.10.0rc1", "5.9.0", "5.10.0", " 5.2.1 "]) == "5.10.0"
    with pytest.raises(ValueError, match="no bare-version tag"):
        check_version.highest_tag(["v1.0.0", "1.0.0rc1"])


def test_tags_from_git_reads_the_repository_tags(tmp_path, monkeypatch):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)  # noqa: S603, S607

    git("init", "-q")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "init")
    git("tag", "5.1.0")
    git("tag", "v5.2.0")
    monkeypatch.setattr(check_version, "ROOT", tmp_path)
    assert sorted(check_version.tags_from_git()) == ["5.1.0", "v5.2.0"]
