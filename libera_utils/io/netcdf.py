"""Module containing utilities for writing Libera-conforming NetCDF4 data products"""

import itertools
import logging
import os
import zlib
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import xarray as xr
from cloudpathlib import AnyPath, CloudPath
from cloudpathlib.enums import FileCacheMode
from dask import delayed
from distributed import as_completed, get_client
from numpy.typing import NDArray
from xarray.backends import H5NetCDFStore
from xarray.backends.common import ArrayWriter

from libera_utils.config import config
from libera_utils.io.filenaming import LiberaDataProductFilename, PathType
from libera_utils.io.product_definition import LiberaDataProductDefinition

T_XarrayNetcdfEngine = Literal["netcdf4", "h5netcdf"]

logger = logging.getLogger(__name__)


class NetcdfEngine(StrEnum):
    """String enum class for our allowed NetCDF engines for xarray

    Neither engine streams to object storage. Products destined for S3 are staged on local disk
    and uploaded, so both engines support cloud output paths equally. See `_write_dataset`.
    """

    netcdf4 = "netcdf4"
    h5netcdf = "h5netcdf"

    @classmethod
    def get_from_config(cls) -> T_XarrayNetcdfEngine:
        """Retrieve the current netcdf engine config from the package configuration"""
        return cls(config.get("XARRAY_NETCDF_ENGINE"))  # type: ignore[return-value]


def write_libera_data_product(
    data_product_definition: str | PathType | LiberaDataProductDefinition,
    data: dict[str, NDArray] | xr.Dataset,
    output_path: str | PathType,
    time_variable: str,
    dynamic_product_attributes: dict[str, Any] | None = None,
    strict: bool = True,
    add_archive_path_prefix: bool = False,
    worker_compressed_variables: tuple[str, ...] = (),
) -> LiberaDataProductFilename:
    """Write a Libera data product NetCDF4 file that conforms to data product definition requirements

    Steps:

    1. (Numpy array input only) Create a product Dataset from the input arrays and optional dynamic attributes, using the product definition to determine the expected structure and metadata.
    2. Bring the Dataset into conformance with the product definition, fixing any issues that can be automatically fixed and issuing warnings for any issues that cannot be automatically fixed.
    3. Check the final conformance of the Dataset against the product definition, raising an exception if any issues are found in strict mode.
    4. Generate the data product filename using the product definition and the specified time variable.
    5. Write the Dataset to a NetCDF4 file at the specified output path with the generated filename, using the configured NetCDF engine.

    A product already present at the generated path is replaced, on local disk and in S3
    alike. Reprocessing rewrites a granule at the same key by design: a run that halted because
    the object in the bucket was newer than the file it just produced would fail on its own
    success.

    Parameters
    ----------
    data_product_definition : str | PathType | LiberaDataProductDefinition
        Path to the data product definition against which to verify conformance
    data : dict[str, np.ndarray] | xr.Dataset
        Data mapping variable names to numpy data arrays or a fully formed L1A xarray Dataset.
    output_path : str | PathType
        Base path (directory or S3 prefix) at which to write the product file
    time_variable : str
        Name of variable that indicates time. This is used to generate the start and end time for the filename.
    dynamic_product_attributes : dict[str, Any] | None
        Optional dictionary of additional global attributes to add to the data product file. Must conform to the data
        product definition.
    strict : bool
        Default True. Raises an exception if the final Dataset doesn't conform to the data product definition.
    add_archive_path_prefix : bool
        Note: do not use this to write to a processing dropbox! L2 devs do not need this kwarg. Default False. If True,
        adds the archive path prefix to the output path when generating the full output path.
    worker_compressed_variables : tuple[str, ...]
        Default empty, which writes the Dataset with ``to_netcdf``. Names Dask-backed data variables whose chunks
        are shuffled and deflated on the workers of the active ``dask.distributed`` Client, so compression runs in
        parallel instead of one chunk at a time under the HDF5 write lock. This process stores the compressed
        chunks with h5py ``write_direct_chunk``, and the file reads back identical to a ``to_netcdf`` write. Requires
        the ``h5netcdf`` engine and an active Client. Other Dask-backed variables are computed in the same graph, so
        tasks shared between variables run once, and are written by this process uncompressed by the workers. If
        this write fails after creating its local file, the file is deleted before the error propagates: a local
        product already at that path is gone as well, while for an S3 destination the existing object is untouched
        and nothing is uploaded.

    Returns
    -------
    : LiberaDataProductFilename
        Filename object containing the full path to the written NetCDF4 data product file.

    Raises
    ------
    ValueError
        If `dynamic_product_attributes` is passed alongside a Dataset, if `time_variable` does
        not have a datetime64 dtype, or, when `strict` is True, if the Dataset does not conform
        to the product definition. With `worker_compressed_variables`, before any data is
        computed: if the engine is not ``h5netcdf``; if a name is not a Dask-backed data variable
        or is repeated; if no ``dask.distributed`` Client is active; if a named variable is not
        chunked on disk, has filters other than deflate with or without shuffle, has a dtype
        that is not integer or floating point, or has a Dask chunk length that is not a whole
        multiple of its on-disk chunk length; or if another Dask-backed variable has a dtype
        that is not numeric or boolean. Then, if a computed Dask block's dtype or shape differs
        from what its array declared.
    pydantic_core.ValidationError
        If `data_product_definition` is a path and a variable's ``encoding['chunksizes']`` is not
        a list of positive integers with as many entries as its ``dimensions``.
    """
    logger.info("Writing Libera data product")

    engine = NetcdfEngine.get_from_config()
    if worker_compressed_variables and engine != NetcdfEngine.h5netcdf:
        raise ValueError(
            f"worker_compressed_variables requires the h5netcdf engine, but the configured engine is {engine}; "
            "set XARRAY_NETCDF_ENGINE=h5netcdf or omit worker_compressed_variables"
        )

    if isinstance(data_product_definition, LiberaDataProductDefinition):
        logger.info("Using provided LiberaDataProductDefinition object")
        definition = data_product_definition
    else:
        logger.info(f"Loading data product definition from {data_product_definition}")
        definition = LiberaDataProductDefinition.from_yaml(data_product_definition)

    logger.info(f"Product definition defines product level attributes: {definition.attributes}")

    if isinstance(data, xr.Dataset):
        if dynamic_product_attributes is not None:
            raise ValueError(
                "dynamic_product_attributes is invalid when passing in a Dataset. To set dynamic attributes for a dataset, modify the Dataset attrs before passing it in."
            )
        # This is how L1A products are typically created (starting as a Dataset)
        dataset = data
    else:
        # This is the expectation for L2 product creation (from numpy arrays)
        logger.info(f"Creating Dataset from data arrays with variables: {list(data.keys())}")
        dataset = definition.create_product_dataset(data, dynamic_product_attributes=dynamic_product_attributes)

    logger.info(f"Bringing Dataset into conformance with product definition")
    dataset = definition.enforce_dataset_conformance(dataset)

    logger.info("Checking final Dataset conformance against the product definition")
    definition.check_dataset_conformance(dataset, strict=strict)

    if "datetime64" not in str(dataset[time_variable].dtype):
        raise ValueError(f"Specified time variable {time_variable} does not have dtype datetime64.")

    data_product_filename = definition.generate_data_product_filename(dataset, time_variable)

    if add_archive_path_prefix:
        prefixed_path = AnyPath(output_path) / data_product_filename.archive_prefix
        prefixed_path.mkdir(parents=True, exist_ok=True)
        data_product_filename.path = prefixed_path / data_product_filename.path.name
    else:
        data_product_filename.path = AnyPath(output_path) / data_product_filename.path.name

    logger.info(f"Writing data product with the {engine} engine")
    _write_dataset(dataset, data_product_filename.path, engine, worker_compressed_variables)
    return data_product_filename


def _load_netcdf_into_memory(path: str | PathType) -> xr.Dataset:
    """Read a NetCDF file fully into memory and close it

    Parameters
    ----------
    path : str | PathType
        Local or S3 path to a NetCDF file.

    Returns
    -------
    xr.Dataset
        The Dataset, loaded into memory with the file closed.

    Raises
    ------
    ValueError
        If the file cannot be opened or read as NetCDF.
    """
    path = AnyPath(path)
    logger.info("Opening NetCDF file: %s", path)
    try:
        with xr.open_dataset(path, engine="h5netcdf") as dataset:
            return dataset.load()
    except Exception as e:
        raise ValueError(f"Failed to open or read NetCDF file {path}: {e}") from e


def _write_dataset(
    dataset: xr.Dataset,
    path: PathType,
    engine: T_XarrayNetcdfEngine,
    worker_compressed_variables: tuple[str, ...],
) -> None:
    """Write a Dataset to `path`, always handing the NetCDF engine a real filesystem path

    Each Dask worker under the distributed scheduler reopens the file from the pickled engine
    argument. An open file object cannot be pickled and fails with `TypeError: cannot pickle
    '_io.BufferedRandom' object`; a path can, so the workers must share this process's
    filesystem, as those of a `LocalCluster` do.

    A cloud destination is written to the path's cloudpathlib cache file and uploaded, as
    `CloudPath.open` does on close, so a later read of `path` in this process, such as the
    manifest checksum, is served from local disk. The cache file's modification time is then set
    to the object's, because the cache treats an older file as stale, and under
    `FileCacheMode.close_file` the cache file is removed. The upload replaces any existing object
    at the key (`libera_utils/D-005`); `force_overwrite_to_cloud=True` is needed because the
    default compares the local clock against S3's and could refuse the replacement.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset to write.
    path : PathType
        Destination path. A `CloudPath` is written to its cache file and uploaded; a `Path` is
        written directly. An existing file or object at this path is replaced.
    engine : T_XarrayNetcdfEngine
        NetCDF engine to write with.
    worker_compressed_variables : tuple[str, ...]
        Variables written by `_write_worker_compressed`; empty writes with ``to_netcdf``.
    """
    if isinstance(path, CloudPath):
        # cloudpathlib has no public accessor for the cache location that does not first
        # download an existing object, so this reads `_local` directly.
        cache_file = path._local
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        if worker_compressed_variables:
            _write_worker_compressed(dataset, cache_file, worker_compressed_variables)
        else:
            dataset.to_netcdf(cache_file, engine=engine)
        logger.info(f"Uploading {cache_file} to {path}")
        path.upload_from(cache_file, force_overwrite_to_cloud=True)
        cloud_mtime = path.stat().st_mtime
        os.utime(cache_file, times=(cloud_mtime, cloud_mtime))
        if path.client.file_cache_mode == FileCacheMode.close_file:
            path.clear_cache()
    elif worker_compressed_variables:
        _write_worker_compressed(dataset, path, worker_compressed_variables)
    else:
        dataset.to_netcdf(path, engine=engine)


def _write_worker_compressed(dataset: xr.Dataset, path: Path, variable_names: tuple[str, ...]) -> None:
    """Write `dataset` with the named variables' chunks compressed on the Dask workers

    xarray's own `Dataset.dump_to_store` creates the file, so every variable, attribute, dimension
    and HDF5 filter setting is what ``to_netcdf`` would write, and it writes the non-Dask
    variables. It leaves the CF-encoded Dask arrays unwritten. Every block of every one of them is
    then computed in a single ``client.compute``, so a task feeding several variables runs once,
    and the blocks are written as their futures complete. Each future is released once its
    block is written, so the workers hold only the results this process has not yet drained.

    Name and attribute validation that ``to_netcdf`` runs first is not repeated here; the product
    definition's conformance check has already run on `dataset`. The storage checks run once the
    file exists, so a call that raises after creating it deletes it before re-raising.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset to write.
    path : Path
        Local filesystem path to write; an existing file is replaced.
    variable_names : tuple[str, ...]
        Dask-backed data variables whose chunks are shuffled and deflated on the workers.

    Raises
    ------
    ValueError
        As `write_libera_data_product` documents for `worker_compressed_variables`.
    """
    if len(set(variable_names)) != len(variable_names):
        raise ValueError(f"worker_compressed_variables repeats a name: {variable_names}; name each variable once")
    not_dask = [name for name in variable_names if name not in dataset.data_vars or dataset[name].chunks is None]
    if not_dask:
        raise ValueError(
            f"worker_compressed_variables names variables that are not Dask-backed data variables: {not_dask}; "
            "name only data variables backed by Dask arrays"
        )
    try:
        client = get_client()
    except ValueError as err:
        raise ValueError(
            "worker_compressed_variables requires an active dask.distributed Client; "
            "create one before writing or omit worker_compressed_variables"
        ) from err

    writer = ArrayWriter()
    unlimited_dims = dataset.encoding.get("unlimited_dims")
    if isinstance(unlimited_dims, str):
        unlimited_dims = [unlimited_dims]
    try:
        store = H5NetCDFStore.open(os.fspath(path), mode="w")
        try:
            dataset.dump_to_store(store, writer=writer, unlimited_dims=unlimited_dims)
        finally:
            store.close()

        with h5py.File(path, "r+") as h5file:
            units = []
            placements = []
            for source, target in zip(writer.sources, writer.targets, strict=True):
                name = target.variable_name
                dset = h5file[name]
                compressed = name in variable_names
                if compressed:
                    shuffle, level = _worker_compression_settings(dset, source.chunks)
                elif dset.dtype.kind not in "iufb":
                    # Written through h5py directly, which skips the enum, string and compound
                    # handling of h5netcdf's Variable.__setitem__; numeric and boolean data need none.
                    raise ValueError(
                        f"Dask-backed variable {name} has dtype {dset.dtype}, which is not numeric or boolean; "
                        "load it into memory before writing, or omit worker_compressed_variables"
                    )
                starts = [np.cumsum((0, *lengths[:-1])) for lengths in source.chunks]
                blocks = source.to_delayed(optimize_graph=False)
                for index in np.ndindex(*blocks.shape):
                    origin = tuple(int(axis_starts[i]) for axis_starts, i in zip(starts, index, strict=True))
                    block_shape = tuple(lengths[i] for lengths, i in zip(source.chunks, index, strict=True))
                    if compressed:
                        units.append(
                            delayed(_compress_block)(
                                blocks[index], name, dset.chunks, dset.dtype, block_shape, shuffle, level
                            )
                        )
                    else:
                        units.append(blocks[index])
                    placements.append((dset, name, compressed, origin, block_shape))

            futures = client.compute(units)
            # Identically built blocks share a task key, so one result can fill several placements.
            placements_of: dict[str, list[tuple]] = {}
            futures_of: dict[str, list] = {}
            for future, placement in zip(futures, placements, strict=True):
                placements_of.setdefault(future.key, []).append(placement)
                futures_of.setdefault(future.key, []).append(future)
            stored = 0
            try:
                for future in as_completed([same_key[0] for same_key in futures_of.values()]):
                    result = future.result()
                    for dset, name, compressed, origin, block_shape in placements_of[future.key]:
                        if compressed:
                            for chunk_start, data in result:
                                dset.id.write_direct_chunk(
                                    tuple(o + c for o, c in zip(origin, chunk_start, strict=True)), data
                                )
                            stored += len(result)
                        else:
                            _check_block(result, name, dset.dtype, block_shape)
                            dset[tuple(slice(o, o + n) for o, n in zip(origin, block_shape, strict=True))] = result
                    for same_key in futures_of[future.key]:
                        same_key.release()
            except BaseException:
                client.cancel(futures)
                raise
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    logger.info(f"Stored {stored} worker-compressed chunks for {len(variable_names)} variables")


def _worker_compression_settings(dset: h5py.Dataset, source_chunks: tuple[tuple[int, ...], ...]) -> tuple[bool, int]:
    """Check that the workers can compress `dset`'s chunks, and return how HDF5 compresses them

    The workers must produce the same bytes HDF5 would store, so this reads the compression
    settings xarray gave the new dataset: deflate (zlib compression) at some level, optionally
    preceded by shuffle, which regroups each value's bytes so deflate compresses them better.

    Parameters
    ----------
    dset : h5py.Dataset
        The HDF5 dataset xarray created for the variable, with its chunk shape and filters set and
        no data written yet.
    source_chunks : tuple[tuple[int, ...], ...]
        Dask chunk lengths of the encoded source array, one tuple per axis.

    Returns
    -------
    : tuple[bool, int]
        Whether to shuffle before deflating, and the deflate level.

    Raises
    ------
    ValueError
        If the variable is not chunked on disk; if its filters are anything other than deflate,
        with or without shuffle before it; if its dtype is not integer or floating point; or if a
        Dask chunk length is not a whole multiple of the on-disk chunk length on its axis. That
        last rule also means every chunk handed to `write_direct_chunk` is complete, with none cut
        short at the array's edge.
    """
    plist = dset.id.get_create_plist()
    if dset.chunks is None:
        raise ValueError(
            f"Variable {dset.name} is not chunked on disk, so its chunks cannot be written directly; "
            "give it encoding chunksizes or leave it out of worker_compressed_variables"
        )
    filter_ids = [plist.get_filter(i)[0] for i in range(plist.get_nfilters())]
    if filter_ids == [h5py.h5z.FILTER_DEFLATE]:
        shuffle = False
    elif filter_ids == [h5py.h5z.FILTER_SHUFFLE, h5py.h5z.FILTER_DEFLATE]:
        shuffle = True
    else:
        raise ValueError(
            f"Variable {dset.name} has HDF5 filters {filter_ids}; the workers apply only deflate, with or without "
            "shuffle, so drop the other filters or leave it out of worker_compressed_variables"
        )
    level = int(plist.get_filter(len(filter_ids) - 1)[2][0])
    if dset.dtype.kind not in "iuf":
        raise ValueError(
            f"Variable {dset.name} has dtype {dset.dtype}; the workers compress only integer and floating-point data, "
            "so leave it out of worker_compressed_variables"
        )
    for lengths, chunk_length in zip(source_chunks, dset.chunks, strict=True):
        if any(length % chunk_length for length in lengths):
            raise ValueError(
                f"Variable {dset.name} has Dask chunk lengths {lengths} on an axis whose on-disk chunk length is "
                f"{chunk_length}; each must be a whole multiple, so rechunk the Dask array or change chunksizes"
            )
    return shuffle, level


def _check_block(block: np.ndarray, name: str, dtype: np.dtype, block_shape: tuple[int, ...]) -> None:
    """Raise ValueError unless a computed Dask block has the dtype and shape its array declared

    Dask does not check a computed block against its declaration, and a block stored with
    `write_direct_chunk` bypasses the casting and shape checks of an h5py write.
    """
    if block.dtype != dtype or block.shape != block_shape:
        raise ValueError(
            f"A computed block of {name} has dtype {block.dtype} and shape {block.shape}, but its Dask array "
            f"declared {dtype} and {block_shape}; make the task that produces it return the declared dtype and "
            "shape, or declare the ones it returns (the dtype and shape given to from_delayed or map_blocks)"
        )


def _compress_block(
    block: np.ndarray,
    name: str,
    chunk_shape: tuple[int, ...],
    dtype: np.dtype,
    block_shape: tuple[int, ...],
    shuffle: bool,
    level: int,
) -> list[tuple[tuple[int, ...], bytes]]:
    """Split a Dask block into on-disk chunks and compress each as HDF5's shuffle and deflate filters do

    Runs on a Dask worker.

    Parameters
    ----------
    block : np.ndarray
        Computed block.
    name : str
        Variable the block belongs to, for the error message.
    chunk_shape : tuple[int, ...]
        On-disk chunk shape; each block length is a whole multiple of it.
    dtype : np.dtype
        Stored dtype, byte order included.
    block_shape : tuple[int, ...]
        Shape the block's Dask array declared.
    shuffle : bool
        Whether to byte-shuffle before deflating.
    level : int
        Deflate level.

    Returns
    -------
    : list[tuple[tuple[int, ...], bytes]]
        Each chunk's start within the block and its compressed bytes.

    Raises
    ------
    ValueError
        If the block's dtype or shape differs from `dtype` or `block_shape`.
    """
    _check_block(block, name, dtype, block_shape)
    compressed = []
    for start in itertools.product(*(range(0, n, c) for n, c in zip(block_shape, chunk_shape, strict=True))):
        chunk = np.ascontiguousarray(block[tuple(slice(s, s + c) for s, c in zip(start, chunk_shape, strict=True))])
        raw = chunk.view(np.uint8).reshape(-1, dtype.itemsize).T.tobytes() if shuffle else chunk.tobytes()
        compressed.append((start, zlib.compress(raw, level)))
    return compressed
