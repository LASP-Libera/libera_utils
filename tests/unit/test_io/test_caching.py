"""Tests for the caching module"""

from pathlib import Path
from unittest import mock

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
