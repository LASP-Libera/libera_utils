"""Tests for hdf module"""
# Installed
import h5py as h5
import numpy as np
import pytest
# Local
from libera_sdp.config import config
from libera_sdp.io import hdf


@pytest.fixture
def minimal_h5(tmp_path):
    """Create a minimal HDF5 file to test against"""
    filepath = tmp_path / 'minimal.h5'
    with h5.File(filepath, 'w') as f:
        f['/'].attrs['rootattr'] = 'attr at root'
        f.create_group('/Test Group')
        f.create_dataset('/Test Group/test_ds', shape=(3, 3), dtype=float)
        f['Test Group'].attrs['strattr'] = 'a string'
        f['Test Group'].attrs['floatattr'] = 3.14
        f['Test Group/test_ds'].attrs['dsattr'] = 42
        f.create_group('/deeply/nested/h5/group')
    return filepath


@pytest.mark.parametrize(
    ('group', 'kwargs', 'expected_string'),
    [
        ('/', {'stdout': True}, """\
Group:/ (2 members, 1 attributes)
    @ rootattr = attr at root
Group:/Test Group (1 members, 2 attributes)
    @ floatattr = 3.14
    @ strattr = a string
Dataset:/Test Group/test_ds (shape=(3, 3), type=float64, 1 attributes)
    @ dsattr = 42
Group:/deeply (1 members, 0 attributes)
Group:/deeply/nested (1 members, 0 attributes)
Group:/deeply/nested/h5 (1 members, 0 attributes)
Group:/deeply/nested/h5/group (0 members, 0 attributes)
"""),
        ('Test Group', {'stdout': True}, """\
Group:/Test Group (1 members, 2 attributes)
    @ floatattr = 3.14
    @ strattr = a string
Dataset:/Test Group/test_ds (shape=(3, 3), type=float64, 1 attributes)
    @ dsattr = 42
"""),
        ('Test Group', {'stdout': False, 'include_attrs': False}, """\
Group:/Test Group (1 members, 2 attributes)
Dataset:/Test Group/test_ds (shape=(3, 3), type=float64, 1 attributes)
""")
    ]
)
def test_h5dump(minimal_h5, capsys, group, kwargs, expected_string):
    """Test printing of HDF5 contents"""

    with h5.File(minimal_h5) as f:
        s = hdf.h5dump(f[group], **kwargs)
        captured = capsys.readouterr()
        print(s)
        print(expected_string)
        print(captured.out)
        assert s == expected_string
        if 'stdout' in kwargs and kwargs['stdout'] is True:
            assert captured.out == expected_string
        else:
            assert captured.out == ""


@pytest.fixture()
def test_attributes(tmp_path):
    """Test dictionary"""

    dir = tmp_path / "sub"
    dir.mkdir()  # Ensure directory exists
    path = dir / "swath_test.he5"

    testdict = {
        'path': str(path),
        'attribute_path': config.get("LIBSDP_ATTR"),
        'swath_names': ['Swath1'],
        'dataset_path': 'HDFEOS/SWATHS/Swath1/DataField',
        'dataset_names': ['Temperature', 'SunglintAngle'],
        'datasets': [np.array([1, 1]), np.array([2, 2])],
        'dataset_units': ['Kelvin', 'radians']
    }

    return testdict


def test_create_swath_groups(test_attributes):
    """Test ability to add groups"""

    hdfeos = hdf.SwathHdfEos5(test_attributes['path'], test_attributes['attribute_path'], mode="w")

    hdfeos.create_swath_groups(['Swath1', 'Swath2'])
    f = hdf.h5dump(h5.File(test_attributes['path']))

    assert 'Swath1' in f


def test_add_swath_datasets(test_attributes):
    """Test ability to add datasets and metadata"""

    hdfeos = hdf.SwathHdfEos5(test_attributes['path'], test_attributes['attribute_path'], mode="w")

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

    f = hdf.h5dump(h5.File(test_attributes['path']))

    assert 'Temperature' in f
    assert 'tai93time' in f


def test_add_swath_file_attr(test_attributes):
    """Test file attributes"""

    hdfeos = hdf.SwathHdfEos5(test_attributes['path'], test_attributes['attribute_path'], mode="w")

    hdfeos.add_swath_file_attr()

    f = hdf.h5dump(h5.File(test_attributes['path']))

    assert 'Level 2 Libera Data' in f


def test_validate(test_attributes):
    """Test file attributes"""

    try:
        hdf.SwathHdfEos5.validate(test_attributes)
    except AssertionError as err:
        print(err)
