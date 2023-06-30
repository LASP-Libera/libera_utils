"""Tests for kernels module"""
# Standard
import logging
import unittest
from unittest import mock, TestCase
from unittest.mock import patch
import pytest
# Installed
import numpy as np
import responses
import requests
# Local
import libera_utils.kernel_maker
from libera_utils import spice_utils
from libera_utils.config import config


@responses.activate
def test_find_most_recent_naif_kernel(test_data_path):
    """Test finding recent kernel in NAIF webpage"""
    test_kernel_filename = 'earth_000101_211220_210926.bpc'
    test_index_url = 'https://fake-naif-page/'

    # Mock the response for the index page with the saved html
    with open(test_data_path / 'naif_pck_index.html', 'r') as fh:
        responses.add(
            responses.GET, test_index_url,
            body=fh.read(), status=200,
            content_type='text/html'
        )
    recent_kernel = spice_utils.find_most_recent_naif_kernel("https://fake-naif-page",
                                                             "earth_[0-9]{6}_[0-9]{6}_[0-9]{6}.bpc")
    assert recent_kernel == test_index_url + test_kernel_filename


@responses.activate
def test_kernel_file_cache(spice_test_data_path, test_data_path, tmp_path):
    """Test caching a kernel file from NAIF, mocking out the actual HTTP requests."""
    # Name of a file mentioned in the test naif page
    test_kernel_filename = 'earth_000101_211220_210926.bpc'
    full_file_url = f"https://fake-naif-page/{test_kernel_filename}"

    cache = spice_utils.KernelFileCache(full_file_url,
                                        fallback_kernel=spice_test_data_path / test_kernel_filename)

    with open(test_data_path / 'naif_pck_index.html', 'r') as fh:
        responses.add(
            responses.GET, 'https://fake-naif-page/',
            body=fh.read(), status=200,
            content_type='text/html'
        )

    # Mock out the download URL for the kernel file with the local test file
    with open(spice_test_data_path / test_kernel_filename, 'rb') as fh:
        responses.add(
            responses.GET, full_file_url,
            body=fh.read(), status=200,
            content_type='application/octet-stream',
            adding_headers={'Transfer-Encoding': 'chunked'}
        )

    with mock.patch('libera_utils.spice_utils.KernelFileCache.cache_dir',
                    new_callable=mock.PropertyMock, return_value=tmp_path):
        # Prove that the download logic works for putting a file in the cache
        cache.download_kernel(full_file_url)
        assert cache.is_cached() is True
        assert cache.kernel_path == tmp_path / test_kernel_filename

        # Prove that the automatic download logic works when we ask for the kernel_path
        cache.clear()
        assert cache.is_cached() is False
        # Tests the automomatic searching and downloading of a new (mocked) kernel
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

    with mock.patch('libera_utils.spice_utils.KernelFileCache.cache_dir',
                    new_callable=mock.PropertyMock, return_value=tmp_path):
        assert cache.is_cached() is False
        assert cache.is_cached() is False  # still
        assert cache.kernel_path == tmp_path / test_jpss_spk.name
        assert cache.is_cached() is True
        cache.furnsh()
        assert spice_utils.ls_kernels() == [spice_utils.KernelFileRecord('SPK', str(cache.kernel_path))]


def test_ls_kernels(furnish_sclk, caplog):
    """Test listing all furnished kernels"""
    caplog.set_level(logging.DEBUG)
    result = spice_utils.ls_kernels(verbose=True, log=True)
    assert result == [spice_utils.KernelFileRecord('TEXT', config.get('JPSS_SCLK'))]
    assert 'jpss_sclk_v01.tsc' in caplog.records[0].message


def test_ls_spice_constants(furnish_test_lsk, furnish_fk):
    """Test listing all kernel pool variables"""
    spice_pool = spice_utils.ls_spice_constants(True)
    print(spice_pool)
    assert spice_pool['TKFRAME_EARTH_FIXED_RELATIVE'] == ['ITRF93']
    assert spice_pool['DELTET/DELTA_T_A'] == [32.184]


def test_ls_kernel_coverage(furnish_test_jpss_ck, furnish_test_jpss_spk, furnish_sclk):
    """Test listing all kernel time coverage"""
    spice_utils.ls_kernel_coverage('CK', True)
    spice_utils.ls_kernel_coverage('SPK', True)

    with pytest.raises(ValueError):
        spice_utils.ls_kernel_coverage('FOO', True)


def test_write_kernel_input_file(tmp_path):
    """Test writing ephemeris or attitude data to an input file for MSOPCK or MKSPK"""
    filepath = tmp_path / 'ephemeris.txt'
    data = np.array([
        (23109, 7, 137, 159, 23109, 30, 941, 6389695.5, 2786021.5, 1825377.375, 2383.52880859, -785.88641357,
         -7105.89892578, 23108, 86399930, 941, -0.21635266, 0.76247245, 0.25699475, 0.5529747),
        (23109, 1005, 176, 159, 23109, 1030, 945, 6392075.5, 2785233.75, 1818270.5, 2376.63305664, -789.18908691,
         -7107.84667969, 23109, 930, 945, -0.21621905, 0.76218551, 0.25710732, 0.55337006),
        (23109, 2007, 518, 159, 23109, 2030, 940, 6394449., 2784443., 1811161.75, 2369.73461914, -792.4899292,
         -7109.78662109, 23109, 1930, 940, -0.2160861, 0.76189834, 0.25721949, 0.55376518),
        (23109, 3005, 706, 159, 23109, 3030, 940, 6396815., 2783648.75, 1804051., 2362.83325195, -795.78894043,
         -7111.71875, 23109, 2930, 940, -0.21595182, 0.76161075, 0.25733218, 0.55416065),
        (23109, 4007, 267, 159, 23109, 4030, 940, 6399174.5, 2782851.25, 1796938.25, 2355.92871094, -799.08605957,
         -7113.64355469, 23109, 3930, 940, -0.21582, 0.7613225, 0.25744519, 0.55455542)
    ], dtype=[
        ('DOY', '<i8'), ('MSEC', '<i8'), ('USEC', '<i8'),
        ('ADAESCID', '<i8'),
        ('ADAET1DAY', '<i8'), ('ADAET1MS', '<i8'), ('ADAET1US', '<i8'),
        ('ADGPSPOSX', '<f8'), ('ADGPSPOSY', '<f8'), ('ADGPSPOSZ', '<f8'),
        ('ADGPSVELX', '<f8'), ('ADGPSVELY', '<f8'), ('ADGPSVELZ', '<f8'),
        ('ADAET2DAY', '<i8'), ('ADAET2MS', '<i8'), ('ADAET2US', '<i8'),
        ('ADCFAQ1', '<f8'), ('ADCFAQ2', '<f8'), ('ADCFAQ3', '<f8'), ('ADCFAQ4', '<f8')
    ])
    fields = [
        'ADAET1DAY', 'ADAET1MS', 'ADAET1US',
        'ADGPSPOSX', 'ADGPSPOSY', 'ADGPSPOSZ',
        'ADGPSVELX', 'ADGPSVELY', 'ADGPSVELZ'
    ]
    libera_utils.kernel_maker.write_kernel_input_file(data, filepath, fields)


def test_write_kernel_setup_file(tmp_path):
    """Test writing a setup file for MSOPCK or MKSPK"""
    filepath = tmp_path / 'setup.txt'
    defaults = {
        "INPUT_DATA_TYPE": "MATRICES",
        "INPUT_TIME_TYPE": "ET",
        "ANGULAR_RATE_PRESENT": 'MAKE UP/NO AVERAGING',
        "CK_TYPE": 3,
        "LIST_OF_VALUES": ["FIELD1", "FIELD2", "FIELD3"],
        "DICT_OF_VALUES": {'DISTANCES': 'METERS', 'ANGLES': 'DEGREES'},
        "SOME_FILEPATH": "myfile"
    }
    libera_utils.kernel_maker.write_kernel_setup_file(defaults, filepath)


@pytest.mark.parametrize(
    ["mock_responses", "expectation"],
    [
        ([
             responses.Response(method="GET", url="https://fake-naif-page/",
                                body=requests.exceptions.ConnectionError("Connection error")),
             responses.Response(method="GET", url="https://fake-naif-page/",
                                body=requests.exceptions.Timeout("Timeout error")),
             responses.Response(method="GET", url="https://fake-naif-page/",
                                body=requests.exceptions.HTTPError("HTTP error"))
         ],
         requests.exceptions.HTTPError()),
        ([
             responses.Response(method="GET", url="https://fake-naif-page/",
                                body=requests.exceptions.Timeout("Timeout error")),
             responses.Response(method="GET", url="https://fake-naif-page/", status=200,
                                body='href="earth_000101_211220_210926.bpc"')
         ],
         "https://fake-naif-page/earth_000101_211220_210926.bpc"),
        ([
             responses.Response(method="GET", url="https://fake-naif-page/",
                                body=requests.exceptions.Timeout("Timeout error")),
             responses.Response(method="GET", url="https://fake-naif-page/",
                                body=requests.exceptions.Timeout("Timeout error")),
             responses.Response(method="GET", url="https://fake-naif-page/", status=200,
                                body='href="earth_000101_211220_210926.bpc"')
         ],
         "https://fake-naif-page/earth_000101_211220_210926.bpc")
    ]
)
@responses.activate(registry=responses.registries.OrderedRegistry)
def test_find_most_recent_naif_kernel_timeout_loop(mock_responses, expectation, test_data_path, spice_test_data_path):
    """Testing error handling for connectionHTTP, and timeout errors"""
    for mock_response in mock_responses:
        responses.add(mock_response)

    if isinstance(expectation, Exception):
        with pytest.raises(requests.RequestException):
            recent_kernel = spice_utils.find_most_recent_naif_kernel("https://fake-naif-page",
                                                                     "earth_[0-9]{6}_[0-9]{6}_[0-9]{6}.bpc")
    else:
        success = spice_utils.find_most_recent_naif_kernel("https://fake-naif-page",
                                                           "earth_[0-9]{6}_[0-9]{6}_[0-9]{6}.bpc")
        assert success == expectation


@pytest.mark.parametrize(
    ["mock_responses", "expectation"],
    [
        ([
             responses.Response(method="GET", url="https://fake-naif-page/earth_000101_211220_210926.bpc",
                                body=requests.exceptions.ConnectionError("Connection error")),
             responses.Response(method="GET", url="https://fake-naif-page/earth_000101_211220_210926.bpc",
                                body=requests.exceptions.Timeout("Timeout error")),
             responses.Response(method="GET", url="https://fake-naif-page/earth_000101_211220_210926.bpc",
                                body=requests.exceptions.HTTPError("HTTP error"))
         ],
         requests.exceptions.HTTPError()),
        ([
             responses.Response(method="GET", url="https://fake-naif-page/earth_000101_211220_210926.bpc",
                                body=requests.exceptions.ConnectionError("Connection error")),
             responses.Response(method="GET", url="https://fake-naif-page/earth_000101_211220_210926.bpc",
                                body=requests.exceptions.Timeout("Timeout error")),
             responses.Response(method="GET", url="https://fake-naif-page/earth_000101_211220_210926.bpc", status=200)
         ],
         None),
        ([
             responses.Response(method="GET", url="https://fake-naif-page/earth_000101_211220_210926.bpc",
                                body=requests.exceptions.Timeout("Timeout error")),
             responses.Response(method="GET", url="https://fake-naif-page/earth_000101_211220_210926.bpc", status=200)
         ],
         None)
    ]
)
@responses.activate(registry=responses.registries.OrderedRegistry)
def test_download_failure(mock_responses, expectation, spice_test_data_path, test_data_path, tmp_path):
    """Testing retry loop for downloading naif kernel"""
    for mock_response in mock_responses:
        responses.add(mock_response)

    test_kernel_filename = 'earth_000101_211220_210926.bpc'
    full_file_url = f"https://fake-naif-page/{test_kernel_filename}"

    cache = spice_utils.KernelFileCache(full_file_url,
                                        fallback_kernel=spice_test_data_path / test_kernel_filename)

    if isinstance(expectation, Exception):
        with pytest.raises(expectation.__class__):
            cache.download_kernel(full_file_url, allowed_attempts=3)
    else:
        success = cache.download_kernel(full_file_url, allowed_attempts=3)

    for mock_response in mock_responses:
        assert mock_response.call_count == 1
