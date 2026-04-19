"""Tests for kernels module"""

import logging
from pathlib import Path
from unittest import mock

import pytest
import requests
import responses

from libera_utils.config import config
from libera_utils.libera_spice import spice_utils


@responses.activate
def test_find_most_recent_naif_kernel(test_data_path):
    """Test finding recent kernel in NAIF webpage"""
    test_kernel_filename = "earth_000101_211220_210926.bpc"
    test_index_url = "https://fake-naif-page/"

    # Mock the response for the index page with the saved html
    with open(test_data_path / "naif_pck_index.html") as fh:
        responses.add(responses.GET, test_index_url, body=fh.read(), status=200, content_type="text/html")
    recent_kernel = spice_utils.find_most_recent_naif_kernel(
        "https://fake-naif-page", "earth_[0-9]{6}_[0-9]{6}_[0-9]{6}.bpc"
    )
    assert recent_kernel == test_index_url + test_kernel_filename


@responses.activate
def test_kernel_file_cache(spice_test_data_path, test_data_path, tmp_path):
    """Test caching a kernel file from NAIF, mocking out the actual HTTP requests."""
    # Name of a file mentioned in the test naif page
    test_kernel_filename = "earth_000101_211220_210926.bpc"
    full_file_url = f"https://fake-naif-page/{test_kernel_filename}"

    cache = spice_utils.KernelFileCache(full_file_url, fallback_kernel=spice_test_data_path / test_kernel_filename)

    with open(test_data_path / "naif_pck_index.html") as fh:
        responses.add(responses.GET, "https://fake-naif-page/", body=fh.read(), status=200, content_type="text/html")

    # Mock out the download URL for the kernel file with the local test file
    with open(spice_test_data_path / test_kernel_filename, "rb") as fh:
        responses.add(
            responses.GET,
            full_file_url,
            body=fh.read(),
            status=200,
            content_type="application/octet-stream",
            adding_headers={"Transfer-Encoding": "chunked"},
        )

    with mock.patch(
        "libera_utils.libera_spice.spice_utils.KernelFileCache.cache_dir",
        new_callable=mock.PropertyMock,
        return_value=tmp_path,
    ):
        # Prove that the download logic works for putting a file in the cache
        cache.download_kernel(full_file_url)
        assert cache.is_cached() is True
        assert cache.kernel_path == tmp_path / test_kernel_filename

        # Prove that the automatic download logic works when we ask for the kernel_path
        cache.clear()
        assert cache.is_cached() is False
        # Tests the automatic searching and downloading of a new (mocked) kernel
        assert cache.kernel_path == tmp_path / test_kernel_filename

        # Test fallback kernel functionality
        cache.clear()
        assert cache.is_cached() is False
        responses.replace(responses.GET, full_file_url, status=500)
        assert cache.kernel_path == spice_test_data_path / test_kernel_filename


def test_kernel_file_cache_s3(write_file_to_s3, test_jpss_spk, tmp_path):
    """Test caching and furnishing a kernel stored as an S3 object"""
    s3_url = f"s3://test-bucket/{test_jpss_spk.name}"
    write_file_to_s3(test_jpss_spk, s3_url)
    cache = spice_utils.KernelFileCache(s3_url)

    with mock.patch(
        "libera_utils.libera_spice.spice_utils.KernelFileCache.cache_dir",
        new_callable=mock.PropertyMock,
        return_value=tmp_path,
    ):
        assert cache.is_cached() is False
        assert cache.is_cached() is False  # still
        assert cache.kernel_path == tmp_path / test_jpss_spk.name
        assert cache.is_cached() is True
        cache.furnsh()
        assert spice_utils.ls_kernels() == [spice_utils.KernelFileRecord("SPK", str(cache.kernel_path))]


def test_kernel_file_cache_local_absolute_path(spice_test_data_path, tmp_path):
    """Local kernel Path is copied into the cache directory."""
    test_kernel_filename = "earth_000101_211220_210926.bpc"
    src = spice_test_data_path / test_kernel_filename
    cache = spice_utils.KernelFileCache(src)
    with mock.patch(
        "libera_utils.libera_spice.spice_utils.KernelFileCache.cache_dir",
        new_callable=mock.PropertyMock,
        return_value=tmp_path,
    ):
        out = cache.download_kernel(src)
        assert out == tmp_path / test_kernel_filename
        assert out.read_bytes() == src.read_bytes()
        assert cache.is_cached() is True
        assert cache.kernel_path == tmp_path / test_kernel_filename
        assert str(cache) == str(tmp_path / test_kernel_filename)


@pytest.mark.parametrize("source", ["earth_000101_211220_210926.bpc", Path("earth_000101_211220_210926.bpc")])
def test_kernel_file_cache_local_relative_path(spice_test_data_path, tmp_path, monkeypatch, source):
    """Relative local str/Path resolves against CWD when materializing into the cache."""
    test_kernel_filename = "earth_000101_211220_210926.bpc"
    cache_subdir = tmp_path / "cache"
    cache_subdir.mkdir()
    monkeypatch.chdir(tmp_path)
    src = spice_test_data_path / test_kernel_filename
    (tmp_path / test_kernel_filename).write_bytes(src.read_bytes())
    cache = spice_utils.KernelFileCache(source)
    with mock.patch(
        "libera_utils.libera_spice.spice_utils.KernelFileCache.cache_dir",
        new_callable=mock.PropertyMock,
        return_value=cache_subdir,
    ):
        assert cache.kernel_path == cache_subdir / test_kernel_filename
        assert (cache_subdir / test_kernel_filename).read_bytes() == src.read_bytes()


def test_kernel_file_cache_local_missing_raises(tmp_path):
    """Missing local kernel path raises FileNotFoundError."""
    missing = tmp_path / "nonexistent.bsp"
    cache = spice_utils.KernelFileCache(missing)
    with mock.patch(
        "libera_utils.libera_spice.spice_utils.KernelFileCache.cache_dir",
        new_callable=mock.PropertyMock,
        return_value=tmp_path,
    ):
        with pytest.raises(FileNotFoundError, match="Local kernel file not found"):
            cache.download_kernel(missing)


def test_ls_kernels(furnish_sclk, caplog):
    """Test listing all furnished kernels"""
    caplog.set_level(logging.DEBUG)
    result = spice_utils.ls_kernels(verbose=True, log=True)
    assert result == [spice_utils.KernelFileRecord("TEXT", config.get("JPSS_SCLK"))]
    assert "jpss_sclk_v01.tsc" in caplog.records[0].message


def test_ls_spice_constants(furnish_test_lsk, furnish_fk):
    """Test listing all kernel pool variables"""
    spice_pool = spice_utils.ls_spice_constants(True)
    print(spice_pool)
    assert spice_pool["TKFRAME_EARTH_FIXED_RELATIVE"] == ["ITRF93"]
    assert spice_pool["DELTET/DELTA_T_A"] == [32.184]


def test_ls_kernel_coverage(furnish_test_jpss_ck, furnish_test_jpss_spk, furnish_sclk):
    """Test listing all kernel time coverage"""
    spice_utils.ls_kernel_coverage("CK", True)
    spice_utils.ls_kernel_coverage("SPK", True)

    with pytest.raises(ValueError, match="Invalid kernel_type argument to ls_kernel_coverage"):
        spice_utils.ls_kernel_coverage("FOO", True)


@pytest.mark.parametrize(
    ("mock_responses", "expectation"),
    [
        (
            [
                responses.Response(
                    method="GET",
                    url="https://fake-naif-page/",
                    body=requests.exceptions.ConnectionError("Connection error"),
                ),
                responses.Response(
                    method="GET", url="https://fake-naif-page/", body=requests.exceptions.Timeout("Timeout error")
                ),
                responses.Response(
                    method="GET", url="https://fake-naif-page/", body=requests.exceptions.HTTPError("HTTP error")
                ),
            ],
            requests.exceptions.HTTPError(),
        ),
        (
            [
                responses.Response(
                    method="GET", url="https://fake-naif-page/", body=requests.exceptions.Timeout("Timeout error")
                ),
                responses.Response(
                    method="GET",
                    url="https://fake-naif-page/",
                    status=200,
                    body='href="earth_000101_211220_210926.bpc"',
                ),
            ],
            "https://fake-naif-page/earth_000101_211220_210926.bpc",
        ),
        (
            [
                responses.Response(
                    method="GET", url="https://fake-naif-page/", body=requests.exceptions.Timeout("Timeout error")
                ),
                responses.Response(
                    method="GET", url="https://fake-naif-page/", body=requests.exceptions.Timeout("Timeout error")
                ),
                responses.Response(
                    method="GET",
                    url="https://fake-naif-page/",
                    status=200,
                    body='href="earth_000101_211220_210926.bpc"',
                ),
            ],
            "https://fake-naif-page/earth_000101_211220_210926.bpc",
        ),
    ],
)
@responses.activate(registry=responses.registries.OrderedRegistry)
def test_find_most_recent_naif_kernel_timeout_loop(mock_responses, expectation, test_data_path, spice_test_data_path):
    """Testing error handling for connectionHTTP, and timeout errors"""
    for mock_response in mock_responses:
        responses.add(mock_response)

    if isinstance(expectation, Exception):
        with pytest.raises(requests.RequestException):
            _ = spice_utils.find_most_recent_naif_kernel(
                "https://fake-naif-page", "earth_[0-9]{6}_[0-9]{6}_[0-9]{6}.bpc"
            )
    else:
        success = spice_utils.find_most_recent_naif_kernel(
            "https://fake-naif-page", "earth_[0-9]{6}_[0-9]{6}_[0-9]{6}.bpc"
        )
        assert success == expectation


@pytest.mark.parametrize(
    ("mock_responses", "expectation"),
    [
        (
            [
                responses.Response(
                    method="GET",
                    url="https://fake-naif-page/earth_000101_211220_210926.bpc",
                    body=requests.exceptions.ConnectionError("Connection error"),
                ),
                responses.Response(
                    method="GET",
                    url="https://fake-naif-page/earth_000101_211220_210926.bpc",
                    body=requests.exceptions.Timeout("Timeout error"),
                ),
                responses.Response(
                    method="GET",
                    url="https://fake-naif-page/earth_000101_211220_210926.bpc",
                    body=requests.exceptions.HTTPError("HTTP error"),
                ),
            ],
            requests.exceptions.HTTPError(),
        ),
        (
            [
                responses.Response(
                    method="GET",
                    url="https://fake-naif-page/earth_000101_211220_210926.bpc",
                    body=requests.exceptions.ConnectionError("Connection error"),
                ),
                responses.Response(
                    method="GET",
                    url="https://fake-naif-page/earth_000101_211220_210926.bpc",
                    body=requests.exceptions.Timeout("Timeout error"),
                ),
                responses.Response(
                    method="GET", url="https://fake-naif-page/earth_000101_211220_210926.bpc", status=200
                ),
            ],
            None,
        ),
        (
            [
                responses.Response(
                    method="GET",
                    url="https://fake-naif-page/earth_000101_211220_210926.bpc",
                    body=requests.exceptions.Timeout("Timeout error"),
                ),
                responses.Response(
                    method="GET", url="https://fake-naif-page/earth_000101_211220_210926.bpc", status=200
                ),
            ],
            None,
        ),
    ],
)
@responses.activate(registry=responses.registries.OrderedRegistry)
def test_download_failure(mock_responses, expectation, spice_test_data_path, test_data_path, tmp_path):
    """Testing retry loop for downloading naif kernel"""
    for mock_response in mock_responses:
        responses.add(mock_response)

    test_kernel_filename = "earth_000101_211220_210926.bpc"
    full_file_url = f"https://fake-naif-page/{test_kernel_filename}"

    cache = spice_utils.KernelFileCache(full_file_url, fallback_kernel=spice_test_data_path / test_kernel_filename)

    if isinstance(expectation, Exception):
        with pytest.raises(expectation.__class__):
            cache.download_kernel(full_file_url, allowed_attempts=3)
    else:
        _ = cache.download_kernel(full_file_url, allowed_attempts=3)

    for mock_response in mock_responses:
        assert mock_response.call_count == 1
