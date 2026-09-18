"""Tests for kernels module"""

import logging
import shutil
from pathlib import Path
from unittest import mock

import pytest
import requests
import responses
import spiceypy as spice
from spiceypy.utils.exceptions import SpiceyError

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
def test_find_most_recent_naif_kernel_earth_extended_pck(test_data_path):
    """Extended Earth PCK uses the same NAIF index scraper as other kernel types."""
    test_index_url = "https://fake-naif-page/"
    with open(test_data_path / "naif_pck_index.html") as fh:
        responses.add(responses.GET, test_index_url, body=fh.read(), status=200, content_type="text/html")
    recent_kernel = spice_utils.find_most_recent_naif_kernel(test_index_url, spice_utils.NAIF_EARTH_EXTENDED_PCK_REGEX)
    assert recent_kernel.endswith("earth_2025_250826_2125_predict.bpc")


@responses.activate
def test_kernel_file_cache(spice_test_data_path, test_data_path, tmp_path, recorded_retry_backoff):
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
        # The fallback is only reached after the download retries are exhausted, not on first failure.
        assert recorded_retry_backoff == [1, 1]


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
    assert "jpss_sclk.tsc" in caplog.records[0].message


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
def test_find_most_recent_naif_kernel_timeout_loop(
    mock_responses, expectation, test_data_path, spice_test_data_path, recorded_retry_backoff
):
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

    # Every attempt but the last is followed by a one-second backoff.
    assert recorded_retry_backoff == [1] * (len(mock_responses) - 1)


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
def test_download_failure(
    mock_responses, expectation, spice_test_data_path, test_data_path, tmp_path, recorded_retry_backoff
):
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

    # Every attempt but the last is followed by a one-second backoff.
    assert recorded_retry_backoff == [1] * (len(mock_responses) - 1)


def _write_metakernel(directory: Path, kernel_names: list[str]) -> Path:
    """Write a SPICE metakernel naming ``kernel_names`` relative to the current working directory."""
    listed = ",\n".join(f"                      '{name}'" for name in kernel_names)
    metakernel = directory / "test_metakernel.tm"
    metakernel.write_text(f"KPL/MK\n\n\\begindata\n\nKERNELS_TO_LOAD = (\n{listed}\n)\n\n\\begintext\n")
    return metakernel


# ``ensure_spice`` recovers from an unfurnished kernel pool in three ways depending on configuration.
# Until LIBSDC-703 these branches were only reached incidentally, by the time-conversion tests in
# tests/unit/test_time.py, which meant a unit test silently downloaded an LSK from NAIF on every run.
# They are exercised here directly instead, with the transport mocked.
def test_ensure_spice_furnishes_the_configured_metakernel(test_lsk, tmp_path, monkeypatch):
    """With SPICE_METAKERNEL set, a failed first call is retried after furnishing that metakernel."""
    shutil.copy(test_lsk, tmp_path / test_lsk.name)
    metakernel = _write_metakernel(tmp_path, [test_lsk.name])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SPICE_METAKERNEL", str(metakernel))

    @spice_utils.ensure_spice
    def convert_utc():
        return spice.utc2et("2021-04-09T13:58:49.745289445")

    assert convert_utc() == pytest.approx(671248798.9309382, abs=1e-6)
    furnished = spice_utils.ls_kernels()
    assert spice_utils.KernelFileRecord("META", str(metakernel)) in furnished
    assert any(record.file_name.endswith(test_lsk.name) for record in furnished)


@responses.activate
def test_ensure_spice_downloads_an_lsk_when_only_time_kernels_are_needed(spice_test_data_path, tmp_path, monkeypatch):
    """Without SPICE_METAKERNEL, a time-only function falls back to the newest LSK published by NAIF."""
    monkeypatch.delenv("SPICE_METAKERNEL", raising=False)
    lsk_name = "naif0012.tls"
    responses.add(
        responses.GET,
        spice_utils.NAIF_LSK_INDEX_URL,
        body=f'<a href="naif0011.tls">naif0011.tls</a><a href="{lsk_name}">{lsk_name}</a>',
        status=200,
        content_type="text/html",
    )
    responses.add(
        responses.GET,
        f"{spice_utils.NAIF_LSK_INDEX_URL}{lsk_name}",
        body=(spice_test_data_path / lsk_name).read_bytes(),
        status=200,
        content_type="application/octet-stream",
    )

    @spice_utils.ensure_spice(time_kernels_only=True)
    def convert_utc():
        return spice.utc2et("2021-04-09T13:58:49.745289445")

    with mock.patch(
        "libera_utils.libera_spice.spice_utils.KernelFileCache.cache_dir",
        new_callable=mock.PropertyMock,
        return_value=tmp_path,
    ):
        assert convert_utc() == pytest.approx(671248798.9309382, abs=1e-6)

    assert (tmp_path / lsk_name).exists()
    # The SCLK is furnished alongside the LSK so clock-string conversions work on the same retry.
    assert config.get("JPSS_SCLK") in [record.file_name for record in spice_utils.ls_kernels()]


def test_ensure_spice_raises_without_a_metakernel(monkeypatch):
    """Without SPICE_METAKERNEL and without time_kernels_only, there is nothing to fall back to."""
    monkeypatch.delenv("SPICE_METAKERNEL", raising=False)

    @spice_utils.ensure_spice
    def convert_utc():
        return spice.utc2et("2021-04-09T13:58:49.745289445")

    with pytest.raises(SpiceyError, match="SPICE_METAKERNEL is not set"):
        convert_utc()


def test_ensure_spice_rejects_a_non_callable():
    """The decorator reports a misuse rather than failing later at call time."""
    with pytest.raises(ValueError, match="must be a callable object"):
        spice_utils.ensure_spice("not a function")
