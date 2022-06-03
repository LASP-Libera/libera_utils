"""Tests for hdf module"""
# Installed
import h5py as h5
# Local
import pytest

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
