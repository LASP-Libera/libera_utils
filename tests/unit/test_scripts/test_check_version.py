"""Tests for .github/scripts/check_version.py, imported by file path since it is not a package."""

import importlib.util
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
