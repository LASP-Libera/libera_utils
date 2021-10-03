"""Tests for kernels module"""
# Standard
import logging
import pytest
from pathlib import Path
from unittest import mock
# Installed
import numpy as np
import responses
# Local
import libera_sdp.cli.kernel_maker
from libera_sdp import spiceutil
from libera_sdp.config import config


def test_kernel_file_cache_cache_dir(monkeypatch):
    """Test the property getter for finding the proper cache path based on the system"""
    cache = spiceutil.KernelFileCache("irrelevent-url", 'irrelevant-regex')

    with mock.patch("libera_sdp.spiceutil.version", return_value="0.0.0"):
        with mock.patch("sys.platform", "darwin"):
            assert cache.cache_dir == Path('~/Library/Caches').expanduser() / "libera_sdp/0.0.0"
        with mock.patch("sys.platform", "linux of some type"):
            assert cache.cache_dir == Path("~/.cache").expanduser() / 'libera_sdp/0.0.0'

            monkeypatch.setenv('XDG_CACHE_HOME', '/home/myuser/.cache')
            assert cache.cache_dir == Path("/home/myuser/.cache/libera_sdp/0.0.0")


@responses.activate
def test_kernel_file_cache(libera_sdp_test_data_dir, tmp_path, monkeypatch):
    """Test caching a kernel file from NAIF, mocking out the actual HTTP requests."""
    # Name of a file mentioned in the test naif page
    test_kernel_filename = 'earth_000101_211220_210926.bpc'

    cache = spiceutil.KernelFileCache("https://fake-naif-page", "earth_[0-9]{6}_[0-9]{6}_[0-9]{6}.bpc",
                                      fallback_kernel=libera_sdp_test_data_dir / test_kernel_filename)

    # Mock the response for the index page with the saved html
    with open(libera_sdp_test_data_dir / 'naif_pck_index.html', 'r') as fh:
        responses.add(
            responses.GET, 'https://fake-naif-page/',
            body=fh.read(), status=200,
            content_type='text/html'
        )

    # Prove that the regex search of the html index page works
    assert cache.find_most_recent_kernel() == test_kernel_filename

    # Mock out the download URL for the kernel file with the local test file
    with open(libera_sdp_test_data_dir / test_kernel_filename, 'rb') as fh:
        responses.add(
            responses.GET, 'https://fake-naif-page/{}'.format(test_kernel_filename),
            body=fh.read(), status=200,
            content_type='application/octet-stream',
            adding_headers={'Transfer-Encoding': 'chunked'},
            stream=True
        )

    with mock.patch('libera_sdp.spiceutil.KernelFileCache.cache_dir',
                    new_callable=mock.PropertyMock, return_value=tmp_path):

        # Prove that the download logic works for putting a file in the cache
        cache.download_kernel(test_kernel_filename)
        assert cache.get_cached_kernels() == [cache.cache_dir / test_kernel_filename]
        assert cache.kernel_path == tmp_path / test_kernel_filename

        # Prove that the automatic download logic works when we ask for the kernel_path
        cache.clear()
        assert cache.get_cached_kernels() == []
        # Tests the automomatic searching and downloading of a new (mocked) kernel
        assert cache.kernel_path == tmp_path / test_kernel_filename

        # Test fallback kernel functionality
        cache.clear()
        assert cache.get_cached_kernels() == []
        responses.replace(responses.GET, 'https://fake-naif-page/', status=500)
        assert cache.kernel_path == libera_sdp_test_data_dir / test_kernel_filename


def test_ls_kernels(furnish_sclk, caplog):
    """Test listing all furnished kernels"""
    caplog.set_level(logging.DEBUG)
    result = spiceutil.ls_kernels(verbose=True, log=True)
    assert result == [spiceutil.KernelFileRecord('TEXT', config.get('JPSS_SCLK'))]
    assert 'jpss1_contrived.tsc' in caplog.records[0].message


def test_ls_spice_constants(furnish_lsk, furnish_fk):
    """Test listing all kernel pool variables"""
    spice_pool = spiceutil.ls_spice_constants(True)
    print(spice_pool)
    assert spice_pool['TKFRAME_EARTH_FIXED_RELATIVE'] == ['ITRF93']


def test_ls_kernel_coverage(furnish_jpss_ck, furnish_jpss_spk, furnish_sclk):
    """Test listing all kernel time coverage"""
    spiceutil.ls_kernel_coverage('CK', True)
    spiceutil.ls_kernel_coverage('SPK', True)

    with pytest.raises(ValueError):
        spiceutil.ls_kernel_coverage('FOO', True)


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
    libera_sdp.cli.kernel_maker.write_kernel_input_file(data, filepath, fields)


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
    libera_sdp.cli.kernel_maker.write_kernel_setup_file(defaults, filepath)
    print(filepath)
