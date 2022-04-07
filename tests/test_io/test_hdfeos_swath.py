"""Tests module for Swath HDF-EOS5 filehandling"""
# Standard
import h5py as h5
import numpy as np
import pytest
# Local
from libera_sdp.io.hdfeos_swath import SwathHdfEos5
from libera_sdp.io.hdf import h5dump


@pytest.fixture()
def dict(tmp_path):
    """Test dictionary"""

    dir = tmp_path / "sub"
    dir.mkdir()
    path = dir / "swath_test.he5"

    testdict = {
        'path': str(path),
        'attribute_path': '../../libera_sdp/data/hdf5/attributes.json',
        'swath_names': ['Swath1'],
        'dataset_path': 'HDFEOS/SWATHS/Swath1/DataField',
        'dataset_names': ['Temperature', 'SunglintAngle'],
        'datasets': [np.array([1, 1]), np.array([2, 2])],
        'dataset_units': ['Kelvin', 'radians']
    }

    return testdict


def test_create_swath_groups(dict):
    """Test ability to add groups"""

    hdfeos = SwathHdfEos5(dict['path'], dict['attribute_path'], mode="w")

    hdfeos.create_swath_groups(['Swath1', 'Swath2'])
    f = h5dump(h5.File(dict['path']))

    assert 'Swath1' in f


def test_add_swath_datasets(dict):
    """Test ability to add datasets and metadata"""

    hdfeos = SwathHdfEos5(dict['path'], dict['attribute_path'], mode="w")

    hdfeos.create_swath_groups(['Swath1', 'Swath2'])

    dataset_path_1 = 'HDFEOS/SWATHS/Swath1/DataField'
    dataset_path_2 = 'HDFEOS/SWATHS/Swath2/DataField'
    dataset_names = ['Temperature', 'SunglintAngle']
    datasets = [np.array([1, 1]), np.array([2, 2])]
    dataset_units = ['Kelvin', 'radians']

    hdfeos.add_swath_dataset(dataset_path_1, dataset_names, datasets, dataset_units)
    hdfeos.add_swath_dataset(dataset_path_2, dataset_names, datasets, dataset_units)

    dataset_path_2 = 'HDFEOS/SWATHS/Swath2/GeoField'
    dataset_names = ['scantime', 'tai93time']
    datasets = [np.array([1, 1]), np.array([2, 2])]
    dataset_units = ['ugps', 'sec']

    hdfeos.add_swath_dataset(dataset_path_2, dataset_names, datasets, dataset_units)
    hdfeos.add_swath_metadata()

    f = h5dump(h5.File(dict['path']))

    assert 'Temperature' in f
    assert 'tai93time' in f


def test_add_swath_file_attr(dict):
    """Test file attributes"""

    hdfeos = SwathHdfEos5(dict['path'], dict['attribute_path'], mode="w")

    hdfeos.add_swath_file_attr()

    f = h5dump(h5.File(dict['path']))

    assert 'Level 2 Libera Data' in f


def test_validate(dict):
    """Test file attributes"""

    try:
        SwathHdfEos5.validate(dict)
    except AssertionError as err:
        print(err)
