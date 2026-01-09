"""Tests for the caching module"""

import os
from pathlib import Path
from unittest import mock
from unittest.mock import patch

import pytest

from libera_utils.io import caching


def test_get_local_cache_dir(monkeypatch):
    """Test the function for finding the proper cache path based on the system"""

    with mock.patch("libera_utils.io.caching.version", return_value="0.0.0"):
        with mock.patch("sys.platform", "darwin"):
            assert caching.get_local_cache_dir() == Path("~/Library/Caches").expanduser() / "libera_utils/0.0.0"
        with mock.patch("sys.platform", "linux of some type"):
            assert caching.get_local_cache_dir() == Path("~/.cache").expanduser() / "libera_utils/0.0.0"

            monkeypatch.setenv("XDG_CACHE_HOME", "/home/myuser/.cache")
            assert caching.get_local_cache_dir() == Path("/home/myuser/.cache/libera_utils/0.0.0")


def test_empty_local_cache_dir(tmp_path):
    """Test function that clears out the local cache of all files"""

    with mock.patch("libera_utils.io.caching.get_local_cache_dir", return_value=tmp_path):
        (tmp_path / "foofile").touch()  # Create file in cache directory
        assert list(tmp_path.glob("*")) == [tmp_path / "foofile"]  # Check file is really there
        assert caching.empty_local_cache_dir() == [tmp_path / "foofile"]  # Check list of removed files
        assert list(tmp_path.glob("*")) == []  # Check directory is empty but still exists


# ============================================================================
# Helper Function Tests
# ============================================================================


class TestGetLocalShortTempDir:
    """Test get_local_short_temp_dir helper function."""

    def test_uses_environment_variable_when_set(self, tmp_path):
        """Should respect LIBERA_TEMP_DIR environment variable."""
        custom_path = tmp_path / "custom_temp"
        custom_path.mkdir()

        with patch.dict(os.environ, {"LIBERA_TEMP_DIR": str(custom_path)}):
            result = caching.get_local_short_temp_dir()

        assert result == custom_path
        assert result.exists()

    def test_creates_directory_if_not_exists(self, tmp_path):
        """Should create directory if it doesn't exist."""
        nonexistent_path = tmp_path / "will_be_created"
        assert not nonexistent_path.exists()

        with patch.dict(os.environ, {"LIBERA_TEMP_DIR": str(nonexistent_path)}):
            result = caching.get_local_short_temp_dir()

        assert result == nonexistent_path
        assert nonexistent_path.exists()

    @patch("platform.system")
    def test_windows_default_path(self, mock_system):
        """Should use C:/Temp on Windows."""
        mock_system.return_value = "Windows"

        with patch.dict(os.environ, {}, clear=True):
            with patch("pathlib.Path.mkdir"):  # Don't actually create C:\Temp
                result = caching.get_local_short_temp_dir()

        assert result == Path("C:/Temp")

    @patch("platform.system")
    def test_unix_default_path(self, mock_system):
        """Should use /tmp on Unix-like systems."""
        mock_system.return_value = "Linux"

        with patch.dict(os.environ, {}, clear=True):
            result = caching.get_local_short_temp_dir()

        assert result == Path("/tmp")


class TestValidatePathLength:
    """Test _validate_path_length helper function."""

    def test_short_path_passes_validation(self):
        """Short paths should pass validation."""
        temp_path = caching.get_local_short_temp_dir()
        short_path = temp_path / "short"
        caching.validate_path_length(short_path, max_length=80)  # Should not raise

    def test_long_path_fails_validation(self):
        """Long paths should raise RuntimeError."""
        # Create a path that's definitely too long
        long_path = Path("/a" * 50)  # 100 characters

        with pytest.raises(RuntimeError) as exc_info:
            caching.validate_path_length(long_path, max_length=80)

        error_msg = str(exc_info.value)
        assert "exceeds limit" in error_msg
        assert "80" in error_msg
        assert "LIBERA_TEMP_DIR" in error_msg

    def test_custom_max_length(self, tmp_path):
        """Should respect custom max_length parameter."""
        path = tmp_path / "test"

        # Should pass with generous limit
        caching.validate_path_length(path, max_length=1000)

        # Should fail with strict limit
        with pytest.raises(RuntimeError):
            caching.validate_path_length(path, max_length=5)
