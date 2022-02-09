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
        f.create_group('/Test Group')
        f.create_dataset('/Test Group/test_ds', shape=(3, 3), dtype=float)
        f['Test Group'].attrs['strattr'] = 'a string'
        f['Test Group'].attrs['floatattr'] = 3.14
        f['Test Group/test_ds'].attrs['dsattr'] = 42
    return filepath


def test_h5dump(minimal_h5, capsys):
    """Test printing of HDF5 contents"""

    with h5.File(minimal_h5) as f:
        hdf.h5dump(f)
        captured = capsys.readouterr()
        assert captured.out == """\
Group:Test Group (1 members, 2 attributes)
    @floatattr = 3.14
    @strattr = a string
Dataset:Test Group/test_ds (shape=(3, 3), type=float64, 1 attributes)
    @dsattr = 42
"""
        hdf.h5dump(f['Test Group'])
        captured = capsys.readouterr()
        assert captured.out == """\
Dataset:test_ds (shape=(3, 3), type=float64, 1 attributes)
    @dsattr = 42
"""
        hdf.h5dump(f['Test Group'], include_attrs=False)
        captured = capsys.readouterr()
        assert captured.out == """\
Dataset:test_ds (shape=(3, 3), type=float64, 1 attributes)
"""
