"""Worker-compressed product writes on a process-based dask.distributed cluster"""

import dask.array as da
import h5py
import numpy as np
import pytest
import xarray as xr
from distributed import Client

from libera_utils.io.netcdf import _write_worker_compressed

pytestmark = pytest.mark.integration


def test_worker_compressed_write_on_worker_processes_matches_to_netcdf(tmp_path):
    """Worker processes receive the compression task by pickle, and the file matches a to_netcdf write"""
    values = np.random.default_rng(0).normal(size=(4, 64, 64)).astype(np.float32)
    values[:, :16, :16] = -999
    dataset = xr.Dataset({"values": (("t", "y", "x"), da.from_array(values, chunks=(2, 32, 64)))})
    dataset["values"].attrs["_FillValue"] = np.float32(-999)
    dataset["values"].encoding = {"zlib": True, "complevel": 4, "chunksizes": (1, 16, 16), "shuffle": True}
    dataset.to_netcdf(tmp_path / "reference.nc", engine="h5netcdf")

    with Client(processes=True, n_workers=2, threads_per_worker=1, dashboard_address=None):
        _write_worker_compressed(dataset, tmp_path / "out.nc", ("values",))

    with h5py.File(tmp_path / "reference.nc") as expected, h5py.File(tmp_path / "out.nc") as actual:
        np.testing.assert_array_equal(actual["values"][()].view(np.uint32), expected["values"][()].view(np.uint32))
        assert actual["values"].id.get_num_chunks() == expected["values"].id.get_num_chunks()
