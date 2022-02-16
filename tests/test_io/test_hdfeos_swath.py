"""Tests module for Swath HDF-EOS5 filehandling"""
# Standard
import nexusformat.nexus as nx
import numpy as np
import os
import pytest
from os.path import exists
# Local
from libera_sdp.io.hdfeos_swath import SwathHdfEos5


def test_create_swath_file():
    """Test ability to create file"""

    path = '../../libera_sdp/data/hdf5'
    filename = 'swath_test.he5'

    if exists('/'.join([path, filename])):
        os.remove('/'.join([path, filename]))

    SwathHdfEos5(path, filename, '0.0.0', 'doi')

    assert exists('/'.join([path, filename]))


def test_create_swath_groups():
    """Test ability to add groups"""

    path = '../../libera_sdp/data/hdf5'
    filename = 'swath_test.he5'

    if exists('/'.join([path, filename])):
        os.remove('/'.join([path, filename]))

    hdfeos = SwathHdfEos5(path, filename, '0.0.0', 'doi')

    hdfeos.create_swath_groups(['Swath1', 'Swath2'])

    f = nx.nxload('/'.join([path, filename]))

    assert 'Swath1' in f.tree


def test_add_swath_datasets():
    """Test ability to add datasets and metadata"""

    path = '../../libera_sdp/data/hdf5'
    filename = 'swath_test.he5'

    if exists('/'.join([path, filename])):
        os.remove('/'.join([path, filename]))

    hdfeos = SwathHdfEos5(path, filename, '0.0.0', 'doi')

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

    f = nx.nxload('/'.join([path, filename]))

    print('\n\nEntire Directory')
    print(f.tree)

    struct_metadata = f['HDFEOS INFORMATION/StructMetadata.0']
    print('\nMetadata')
    print(struct_metadata)

    assert 'Temperature' in f.tree
    assert 'tai93time' in f.tree


def test_add_swath_file_attr():
    """Test file attributes"""

    path = '../../libera_sdp/data/hdf5'
    filename = 'swath_test.he5'

    if exists('/'.join([path, filename])):
        os.remove('/'.join([path, filename]))

    hdfeos = SwathHdfEos5(path, filename, '0.0.0', 'doi')

    hdfeos.add_swath_file_attr()

    f = nx.nxload('/'.join([path, filename]))

    assert 'Level 2 Libera Data' in f.tree

def test_validate():
    path = '../../libera_sdp/data/hdf5'
    filename = 'swath_test.he5'

    if exists('/'.join([path, filename])):
        os.remove('/'.join([path, filename]))

    hdfeos = SwathHdfEos5(path, filename, '0.0.0', 'doi')

    swath_names = ['Swath1', 'Swath2']
    dataset_path = 'HDFEOS/SWATHS/Swath1/DataField'
    dataset_names = ['Temperature', 'SunglintAngle']
    datasets = [np.array([1, 1]), np.array([2, 2])]
    dataset_units = ['Kelvin', 'radians']

    hdfeos.validate(swath_names, dataset_path, dataset_names, datasets, dataset_units)

    #c = SwathHdfEos5(path, filename, '0.0.0', 'doi').validate(swath_names, dataset_path, dataset_names, datasets, dataset_units)
    #c.add_swath_file_attr()
    print('hi')

