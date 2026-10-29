"""Module for reading packet data using Space Packet Parser"""

import logging
import multiprocessing as mp
import traceback
import warnings
from collections.abc import Iterator
from contextlib import closing
from os import PathLike
from typing import cast

import numpy as np
import xarray as xr
from cloudpathlib import AnyPath
from space_packet_parser import load_xtce
from space_packet_parser.xarr import create_dataset
from space_packet_parser.xtce.definitions import XtcePacketDefinition

from libera_utils.config import config
from libera_utils.constants import LiberaApid
from libera_utils.io import filenaming
from libera_utils.io.product_definition import ALGORITHM_VERSION_ATTRIBUTE
from libera_utils.l1a.l1a_packet_configs import (
    AggregationGroup,
    ArrayGroup,
    SampleGroup,
    TimeFieldMapping,
    get_packet_config,
)
from libera_utils.l1a.packet_ordering import check_packet_order, sort_by_sequence_counter
from libera_utils.l1a.wfov_image_metadata import enhance_wfov_l1a_dataset
from libera_utils.time import multipart_to_dt64
from libera_utils.version import version

logger = logging.getLogger(__name__)

# SPP always creates Datasets with a non-coordinate "packet" dimension
# This is a constant and is not expected to change in SPP
SDC_PACKET_DIMENSION = "PACKET"
SPP_PACKET_DIMENSION = "packet"  # The original dimension name from SPP before we swap it to uppercase

DATETIME_USEC_DTYPE = np.dtype("datetime64[us]")

# Pinned rather than taken from the platform, whose default is spawn on macOS and changed from fork to
# forkserver on Linux in Python 3.14. fork needs no ``if __name__ == "__main__":`` guard in the caller.
_WORKER_START_METHOD = "fork"


def drop_implausible_telemetry_times(times_us: np.ndarray, *, context: str) -> np.ndarray:
    """Filter out packet/sample times outside the plausible telemetry window.

    The window is ``MIN_VALID_TELEMETRY_TIME`` to ``MAX_VALID_TELEMETRY_TIME`` from config. Both
    are fixed dates, not offsets from now, so a span written to File Metadata does not depend on
    when extraction ran; the ceiling admits simulated-clock captures running at a mission-era
    epoch. A clock that has not yet received its first time-sync command reads out a near-zero
    day/second counter, decoding to just after ``CCSDS_EPOCH`` (1958-01-01), and one such packet
    stretches a ``min()``/``max()`` span across ~68 years.

    A time exactly at the floor is dropped. ``NaT`` is dropped explicitly, since it compares false
    against both bounds.

    Parameters
    ----------
    times_us : np.ndarray
        Array of ``datetime64[us]`` packet or sample times.
    context : str
        Description of what is being processed (APID/file), used in the warning log and in the
        error raised if nothing remains.

    Returns
    -------
    np.ndarray
        ``times_us`` with any implausible entries removed.

    Raises
    ------
    ValueError
        If no times remain after filtering.
    """
    floor = np.datetime64(config.get("MIN_VALID_TELEMETRY_TIME"), "us")
    ceiling = np.datetime64(config.get("MAX_VALID_TELEMETRY_TIME"), "us")

    invalid = np.isnat(times_us)
    below = times_us <= floor
    above = times_us > ceiling
    n_invalid = int(invalid.sum())
    n_below = int(below.sum())
    n_above = int(above.sum())
    if n_invalid:
        logger.warning("Excluded %d NaT time(s) for %s", n_invalid, context)
    if n_below:
        logger.warning(
            "Excluded %d time(s) at or before sanity floor %s (likely clock not yet time-synced) for %s",
            n_below,
            floor,
            context,
        )
    if n_above:
        logger.warning(
            "Excluded %d time(s) after sanity ceiling %s (likely corrupted time counter) for %s",
            n_above,
            ceiling,
            context,
        )

    filtered = times_us[~(invalid | below | above)]
    if filtered.size == 0:
        raise ValueError(
            f"No times for {context} fall between the sanity floor {floor} and ceiling {ceiling}; none remain."
        )
    return filtered


class _PacketAxisAccumulator:
    """Concatenate packet datasets along the packet axis by filling preallocated arrays.

    ``xr.concat`` allocates its result while every input is still referenced, so peak memory is
    roughly 2.9 times the concatenated size (measured on a day of APID 1040). Copying each
    dataset into a buffer and releasing it immediately holds the buffer plus one input instead.

    Buffers start at ``capacity`` packets and grow geometrically, so a file carrying more packets
    than the estimate costs one reallocation rather than a failure. A variable whose dtype is
    wider in a later file (``space_packet_parser`` sizes string columns to the widest label it
    actually saw, so a file containing ``SINGLE`` yields ``<U6`` where others yield ``<U3``)
    widens the buffer, matching what ``xr.concat`` would have promoted it to.
    """

    def __init__(self, template: xr.Dataset, apid: int, capacity: int):
        self._apid = apid
        self._dims = {name: var.dims for name, var in template.data_vars.items()}
        self._buffers = {
            name: np.empty((capacity, *var.shape[1:]), dtype=var.dtype) for name, var in template.data_vars.items()
        }
        self._n = 0

    @property
    def capacity(self) -> int:
        """Packets the buffers can hold before they must grow."""
        return len(next(iter(self._buffers.values())))

    def _reallocate(self, capacity: int, dtypes: dict[str, np.dtype]) -> None:
        # One variable at a time: the transient is a single column, not a second full copy.
        for name, buffer in self._buffers.items():
            grown = np.empty((capacity, *buffer.shape[1:]), dtype=dtypes[name])
            grown[: self._n] = buffer[: self._n]
            self._buffers[name] = grown

    def append(self, packet_file: PathLike | str, file_ds: xr.Dataset) -> None:
        """Copy one parsed file onto the end of the packet axis.

        Raises
        ------
        ValueError
            If ``file_ds`` does not carry exactly the variables the accumulator was built with.
            Concatenating datasets whose variable sets differ fills the missing side with NaN,
            which would silently manufacture data; the XTCE is flat per APID, so a mismatch means
            a file is not what it claims to be.
        """
        if set(file_ds.data_vars) != set(self._buffers):
            raise ValueError(
                f"Variables parsed from {packet_file} for APID {self._apid} do not match the other files: "
                f"missing {sorted(set(self._buffers) - set(file_ds.data_vars))}, "
                f"unexpected {sorted(set(file_ds.data_vars) - set(self._buffers))}."
            )

        n = file_ds.sizes[SDC_PACKET_DIMENSION]
        promoted = {name: np.promote_types(buffer.dtype, file_ds[name].dtype) for name, buffer in self._buffers.items()}
        needs_widening = any(promoted[name] != buffer.dtype for name, buffer in self._buffers.items())
        if self._n + n > self.capacity:
            self._reallocate(max(self._n + n, self.capacity * 2), promoted)
        elif needs_widening:
            self._reallocate(self.capacity, promoted)

        for name, buffer in self._buffers.items():
            buffer[self._n : self._n + n] = file_ds[name].values
        self._n += n

    def to_dataset(self) -> xr.Dataset:
        """Build the concatenated Dataset.

        Trims the buffers to the appended packets, one variable at a time, and returns arrays that
        are views onto them.
        """
        if self._n < self.capacity:
            self._reallocate(self._n, {name: buffer.dtype for name, buffer in self._buffers.items()})
        return xr.Dataset({name: (self._dims[name], buffer[: self._n]) for name, buffer in self._buffers.items()})


def _parse_one_file(
    packet_file: PathLike | str,
    packet_definition: XtcePacketDefinition,
    apid: int,
    generator_kwargs: dict,
) -> xr.Dataset | None:
    """Parse a single file to a packet Dataset, or None when it holds no packet of ``apid``.

    Raises
    ------
    ValueError
        If the file yields any APID other than ``apid``.
    """
    dataset_dict = create_dataset(
        packet_files=[AnyPath(packet_file)],
        xtce_packet_definition=packet_definition,
        generator_kwargs=generator_kwargs,
        packet_filter=lambda packet_bytes: packet_bytes.apid == apid,
    )
    if not dataset_dict:
        return None
    if set(dataset_dict.keys()) != {apid}:
        raise ValueError(
            f"Expected only APID {apid} in parsed dataset, but found APIDs: {list(dataset_dict.keys())}. "
            f"This probably means the packet filter function is not working."
        )
    file_ds = dataset_dict[apid]
    # Swap standard SPP "packet" dimension to uppercase "PACKET" to align with config naming
    if SPP_PACKET_DIMENSION in file_ds.dims:
        file_ds = file_ds.swap_dims({SPP_PACKET_DIMENSION: SDC_PACKET_DIMENSION})
    return file_ds


def _parse_file_in_subprocess(
    connection,
    packet_file: str,
    packet_definition_path: str,
    apid: int,
    generator_kwargs: dict,
) -> None:
    """Worker entry point: parse one file and send the Dataset, or the exception, to the parent.

    Takes the XTCE path rather than a loaded definition so that the entry point does not depend on
    the start method: ``XtcePacketDefinition`` is not picklable, which ``spawn`` and ``forkserver``
    would require. Loading it here costs ~0.2 s against a parse of tens of seconds.

    Pickling drops an exception's traceback, so the worker's traceback is attached as a note.
    """
    try:
        definition = load_xtce(packet_definition_path)
        connection.send((None, _parse_one_file(packet_file, definition, apid, generator_kwargs)))
    except Exception as exc:  # noqa: BLE001 - re-raised in the parent with this traceback as a note
        exc.add_note(traceback.format_exc())
        connection.send((exc, None))
    finally:
        connection.close()


def _worker_context() -> mp.context.BaseContext:
    """Return the multiprocessing context workers start from, using ``_WORKER_START_METHOD``."""
    return mp.get_context(_WORKER_START_METHOD)


def _iter_parsed_files_concurrently(
    packet_files: list[PathLike | str],
    packet_definition_path: str,
    apid: int,
    max_workers: int,
    generator_kwargs: dict,
) -> Iterator[tuple[PathLike | str, xr.Dataset | None]]:
    """Parse files in worker processes, yielding results in the order given.

    Parsing is CPU-bound single-threaded Python, so a container sized for several vCPUs leaves
    all but one idle. Files are independent, so they parse concurrently; results are yielded in the
    order given, which ``parse_packets_to_dataset`` promises and which decides the packet
    ``parse_packets_to_l1a_dataset`` keeps among repeated counters when ``ground_data=True``.

    Workers run ``max_workers`` at a time. Each holds a parse in flight, ``create_dataset``'s
    per-field Python objects and then the parsed file until the parent reads it, so peak memory is
    the accumulated axis plus ``max_workers`` in-flight parses, which is what bounds a useful worker
    count rather than the available cores.

    Closing the generator terminates and joins any worker still running in the current batch.
    """
    context = _worker_context()
    for start in range(0, len(packet_files), max_workers):
        workers = []
        # Closing the generator early, on a worker error or a consumer error, must not leave the
        # rest of the batch running or holding pipe handles.
        try:
            for packet_file in packet_files[start : start + max_workers]:
                receive, send = context.Pipe(duplex=False)
                process = context.Process(
                    target=_parse_file_in_subprocess,
                    args=(send, str(packet_file), packet_definition_path, apid, generator_kwargs),
                )
                process.start()
                send.close()  # the parent must drop its copy or the pipe never reports EOF
                workers.append((packet_file, process, receive))

            for packet_file, process, receive in workers:
                try:
                    error, file_ds = receive.recv()
                except EOFError:
                    process.join()
                    raise RuntimeError(
                        f"Worker parsing {packet_file} for APID {apid} exited without a result "
                        f"(exit code {process.exitcode}). A negative code is a signal, of which -9 is the "
                        f"container running out of memory; a positive code means the worker died before it "
                        f"could report, and its traceback is on stderr."
                    ) from None
                process.join()
                if error is not None:
                    raise error
                yield packet_file, file_ds
        finally:
            for _, process, receive in workers:
                receive.close()
                if process.is_alive():
                    process.terminate()
                process.join()


def parse_packets_to_dataset(
    packet_files: list[PathLike | str],
    packet_definition: str | PathLike | XtcePacketDefinition,
    apid: int,
    *,
    max_workers: int = 1,
    **generator_kwargs,
) -> xr.Dataset:
    """Parse packets from files into an xarray Dataset using specified packet definition.

    This function does not make any changes to the packet data other than filtering by a single APID.

    Files are parsed one at a time and concatenated, rather than handed to
    ``create_dataset`` together. ``create_dataset`` retains a Python object per parsed field
    until it converts the whole batch to numpy, which costs roughly 800 MiB per two-hour
    NOM-HK file; a full day of files in one batch exhausts an 8 GiB container partway through.
    Parsing per file bounds that at one file's worth.

    Parsed files are copied onto a preallocated packet axis and released one at a time, rather
    than held and handed to ``xr.concat``, which allocates its result while every input is still
    referenced.

    A file holding no packets of ``apid`` is skipped with a warning.

    Parameters
    ----------
    packet_files : list[PathLike | str]
        List of filepaths to packet files.
    packet_definition : str | PathLike | space_packet_parser.xtce.definitions.XtcePacketDefinition
        Path to the XTCE packet definition file, or an already loaded definition. Pass a loaded
        definition to avoid re-parsing the XTCE once per file. ``max_workers`` above 1 with more
        than one file requires a path, which each worker loads, so that workers do not depend on
        the start method.
    apid : int
        Application Process Identifier to filter for.
    max_workers : int, optional
        Files to parse concurrently, in worker processes. Default 1 (in-process). Parsing is
        CPU-bound single-threaded Python, so concurrency is the only way to use a container
        sized for more than one vCPU. Peak memory rises by one in-flight parse per worker (about
        800 MiB for a two-hour NOM-HK file), which is the practical limit rather than the core
        count. Must be at least 1. Workers start with ``fork``, which Windows does not provide, so
        the caller needs no ``if __name__ == "__main__":`` guard; a caller that already runs
        threads risks a worker inheriting a held lock (Python 3.12+ warns with
        ``DeprecationWarning``).
    **generator_kwargs
        Additional keyword arguments passed to the packet generator.

    Returns
    -------
    xr.Dataset
        xarray Dataset containing parsed packet data, with files concatenated in the order
        given.

    Raises
    ------
    ValueError
        If no file holds any packet of ``apid``, if a file yields an APID other than ``apid``,
        if the files do not all carry the same set of variables, if ``max_workers`` is below 1,
        or if ``max_workers`` above 1 with more than one file is combined with an already-loaded
        packet definition or with a platform that has no ``fork`` start method.
    RuntimeError
        If a worker process exits without reporting a result, such as when it is killed for
        running out of memory.
    Exception
        Any exception raised while parsing a file in a worker process is re-raised in the caller,
        with the worker's traceback attached as a note.
    """
    if max_workers < 1:
        raise ValueError(f"max_workers must be at least 1, got {max_workers}; pass 1 to parse in process.")
    n_workers = max(1, min(max_workers, len(packet_files)))
    logger.info("Parsing packets (APID %d) from %d file(s) with %d worker(s)", apid, len(packet_files), n_workers)

    if n_workers > 1:
        if _WORKER_START_METHOD not in mp.get_all_start_methods():
            raise ValueError(
                f"max_workers above 1 needs the {_WORKER_START_METHOD!r} start method, which this platform "
                f"does not provide; pass max_workers=1."
            )
        if isinstance(packet_definition, XtcePacketDefinition):
            raise ValueError(
                "max_workers above 1 requires the XTCE path rather than a loaded definition, so that workers "
                "do not depend on the start method; pass the path to the XTCE file."
            )
        parsed_files = _iter_parsed_files_concurrently(
            packet_files, str(packet_definition), apid, n_workers, generator_kwargs
        )
    else:
        if not isinstance(packet_definition, XtcePacketDefinition):
            packet_definition = load_xtce(packet_definition)
        parsed_files = (
            (packet_file, _parse_one_file(packet_file, packet_definition, apid, generator_kwargs))
            for packet_file in packet_files
        )

    # The first contributing file is held whole until a second one arrives: a single-file parse
    # then returns it untouched, with no buffer allocated and nothing copied.
    first: tuple[PathLike | str, xr.Dataset] | None = None
    accumulator: _PacketAxisAccumulator | None = None

    # Closed explicitly so that an error raised here stops the workers now, rather than when the
    # traceback that references the generator is released.
    with closing(parsed_files):
        for packet_file, file_ds in parsed_files:
            if file_ds is None:
                logger.warning("No APID %d packets in %s; excluded from the parsed dataset.", apid, packet_file)
                continue

            if first is None and accumulator is None:
                first = (packet_file, file_ds)
                continue

            if accumulator is None:
                first_file, first_ds = first
                accumulator = _PacketAxisAccumulator(
                    first_ds, apid, capacity=first_ds.sizes[SDC_PACKET_DIMENSION] * len(packet_files)
                )
                accumulator.append(first_file, first_ds)
                # Release the first file before the next one is copied in, so the peak is the buffer
                # plus a single file rather than the buffer plus every file parsed so far.
                first = None
                del first_ds

            accumulator.append(packet_file, file_ds)
            del file_ds

    if accumulator is not None:
        return accumulator.to_dataset()
    if first is not None:
        return first[1]
    raise ValueError(f"No APID {apid} packets found in any of the {len(packet_files)} file(s) given.")


def parse_packets_to_l1a_dataset(
    packet_files: list[PathLike | str],
    apid: int,
    *,
    ground_data: bool = False,
    verbose: bool = False,
    skip_header_bytes: int | None = None,
    max_workers: int = 1,
) -> xr.Dataset:
    """Parse packets to L1A dataset with configurable sample expansion.

    This function parses binary packet files and expands multi-sample fields
    according to the a configuration identified by APID. It creates proper xarray Datasets
    with time coordinates as dimensions.

    Parameters
    ----------
    packet_files : list[PathLike | str]
        List of filepaths to packet files, in any order. They are parsed in file name order, which is time
        order for ground CCSDS names and creation order for flight PDS names.
    apid : int
        The APID (Application Process Identifier) value for the packet type. Used to select the appropriate
        configuration for generating the L1A Dataset structure.
    ground_data : bool, optional
        If True, packets that repeat a ``SRC_SEQ_CTR`` with differing contents, or samples that repeat a
        sample time with differing data, produce a warning instead of a ValueError. This is useful for
        ground test data where such repeats may be expected. Default is False.
    verbose : bool, optional
        If True and ground_data is True, a warning will be issued for each such repeat. Default is False.
    skip_header_bytes : int | None, optional
        Bytes to skip before each CCSDS primary header. When ``None``, uses ``SKIP_PACKET_HEADER_BYTES`` from
        config (default ``0``, correct for flight PDS and demuxed ground CCSDS; raw ground captures that still
        carry a per-packet record header need ``8``).
    max_workers : int, optional
        Files to parse concurrently, in worker processes. Default 1 (in-process); must be at
        least 1. See ``parse_packets_to_dataset``; peak memory rises by one in-flight parse per
        worker, and workers start with ``fork``, which Windows does not provide.

    Returns
    -------
    xr.Dataset
        xarray Dataset with:
        - Main packet data array with packet timestamp dimension
        - Separate arrays for each sample group with optional multi-field expansion
        - All time coordinates properly set as dimensions

    Raises
    ------
    ValueError
        If the decoded packets have no ``SRC_SEQ_CTR``; if packets adjacent in an input file with
        consecutive ``SRC_SEQ_CTR`` would be placed a counter wrap apart; if ``ground_data`` is False and
        packets with the same ``SRC_SEQ_CTR`` differ; as ``_drop_duplicates`` raises for samples; or as
        ``parse_packets_to_dataset`` raises, including for ``max_workers`` below 1 or a platform without
        ``fork``.
    OSError
        If a packet file cannot be opened or read.
    RuntimeError
        If a worker process exits without reporting a result (``max_workers`` above 1).
    Exception
        Any exception raised while parsing a file in a worker process is re-raised in the caller, with the
        worker's traceback attached as a note.

    Notes
    -----
    The ``PACKET`` axis is in ``SRC_SEQ_CTR`` order, with the counter's wraps resolved by packet time
    (see :mod:`libera_utils.l1a.packet_ordering`), whatever the order of the input
    files. A packet received twice is kept once. Packet time is not monotonic along the axis;
    :func:`~libera_utils.l1a.packet_ordering.check_packet_order` warns where it steps backward.
    Input files are parsed in file name order, which ``input_files`` lists.

    For APID 1040 (ICIE WFOV SCI), the packet dataset is post-processed by
    ``enhance_wfov_l1a_dataset``: complete SOP→EOP images are stitched onto a ``CAMERA_TIME``
    dimension, compressed payloads and decoded header metadata are attached, ``ICIE__WFOV_DATA``
    is dropped, and ``PACKET_IMAGE_ID`` traces packets back to stitched images.
    """
    # The wrong-wrap check in sort_by_sequence_counter reads packets adjacent in the input as built one after the
    # other, which holds across a file join only when the files are in time order.
    _packet_files = sorted((cast(filenaming.PathType, AnyPath(f)) for f in packet_files), key=lambda path: path.name)
    packet_config = get_packet_config(LiberaApid(apid))
    packet_definition_path = str(config.get(packet_config.packet_definition_config_key))
    if skip_header_bytes is None:
        skip_header_bytes = config.get("SKIP_PACKET_HEADER_BYTES")
    packet_ds = parse_packets_to_dataset(
        _packet_files, packet_definition_path, apid, max_workers=max_workers, skip_header_bytes=skip_header_bytes
    )
    packet_times_dt64 = multipart_to_dt64(packet_ds, **packet_config.packet_time_fields.multipart_kwargs)
    packet_times_us = packet_times_dt64.values.astype(DATETIME_USEC_DTYPE)

    # Set packet time as a non-dimension coordinate with "PACKET" dimension
    # The packet dimension remains "PACKET" to enable sample-to-packet tracing
    packet_time_coordinate = packet_config.packet_time_coordinate
    packet_ds = packet_ds.assign_coords({packet_time_coordinate: (SDC_PACKET_DIMENSION, packet_times_us)})

    # Order the PACKET axis by SRC_SEQ_CTR and drop packets received twice, before samples are expanded:
    # the "{sample_group}_packet_index" variables enumerate this axis by position, and WFOV image
    # stitching needs each image's packets in one contiguous run. Packet time can step backward while
    # the counter advances, so it never orders packets.
    packet_ds = sort_by_sequence_counter(packet_ds, packet_time_coordinate, ground_data=ground_data, verbose=verbose)
    check_packet_order(packet_ds, packet_time_coordinate)
    packet_times_us = packet_ds[packet_time_coordinate].values

    # Start building the dataset containing expanded sample fields
    sample_ds = xr.Dataset()

    # Process each sample group
    expanded_fields = set()  # Track fields that are expanded to remove from main array

    for sample_group in packet_config.sample_groups:
        # Process sample group (unified handling for single and multi-sample cases)
        field_arrays, sample_times = _expand_sample_group(packet_ds, sample_group)

        # Create dimension name
        sample_time_dimension = sample_group.sample_time_dimension

        # Create separate DataArray for each field
        for field_name, field_data in field_arrays.items():
            sample_ds[field_name] = xr.DataArray(
                data=field_data,
                dims=[sample_time_dimension],
                coords={sample_time_dimension: (sample_time_dimension, sample_times)},
            )

        # Create packet_index variable to map samples back to their originating packets
        n_packets = packet_ds.sizes[SDC_PACKET_DIMENSION]
        n_samples = sample_group.sample_count
        # Create an array that repeats each packet index n_samples times
        # e.g., for 3 packets with 2 samples each: [0, 0, 1, 1, 2, 2]
        packet_indices = np.repeat(np.arange(n_packets), n_samples)
        packet_index_var_name = f"{sample_group.name}_packet_index"
        sample_ds[packet_index_var_name] = xr.DataArray(
            data=packet_indices,
            dims=[sample_time_dimension],
            coords={sample_time_dimension: (sample_time_dimension, sample_times)},
        )

        # Track expanded sample fields (including time fields) to remove from main array
        expanded_fields.update(_get_expanded_field_names(packet_ds, sample_group))

        # Drop and warn about duplicate samples
        # NOTE: This should never find duplicates in flight but in ground testing, FSW was generating
        # packets that had repeated sample timestamps due to an issue with Hydra simulating SC time pulses
        # incorrectly, causing a microsecond counter to roll over at 1E6 without incrementing the second counter.
        sample_ds, _ = _drop_duplicates(sample_ds, sample_time_dimension, ground_data=ground_data, verbose=verbose)

        # Sort the data by the newly added dimension for the sample group
        sample_ds = sample_ds.sortby(sample_time_dimension)

    # Drop expanded sample fields from packet_ds to reduce data duplication
    packet_ds = packet_ds.drop_vars(expanded_fields)

    # Process aggregation groups
    aggregated_fields = set()  # Track fields that are aggregated to remove from main array

    for agg_group in packet_config.aggregation_groups:
        # Aggregate the fields
        aggregated_data = _aggregate_fields(packet_ds, agg_group)

        # Add aggregated variable to packet dataset with packet dimension
        packet_ds[agg_group.name] = xr.DataArray(
            data=aggregated_data,
            dims=[SDC_PACKET_DIMENSION],
            coords={packet_time_coordinate: (SDC_PACKET_DIMENSION, packet_times_us)},
        )

        # Track aggregated fields to remove from main array
        aggregated_fields.update(_get_aggregated_field_names(packet_ds, agg_group))

    # Drop aggregated fields from packet_ds to reduce data duplication
    packet_ds = packet_ds.drop_vars(aggregated_fields)

    # Process array groups
    array_group_fields = set()  # Track fields that are stacked to remove from main array

    for array_group in packet_config.array_groups:
        # Stack the fields
        stacked_data = _stack_fields(packet_ds, array_group)
        index_coord = np.arange(array_group.field_count, dtype=np.int64)

        # Add stacked variable to packet dataset with packet and array dimensions
        packet_ds[array_group.name] = xr.DataArray(
            data=stacked_data,
            dims=[SDC_PACKET_DIMENSION, array_group.dimension],
            coords={
                packet_time_coordinate: (SDC_PACKET_DIMENSION, packet_times_us),
                array_group.dimension: (array_group.dimension, index_coord),
            },
        )

        # Track stacked fields to remove from main array
        array_group_fields.update(_get_array_group_field_names(packet_ds, array_group))

    # Drop stacked fields from packet_ds to reduce data duplication
    packet_ds = packet_ds.drop_vars(array_group_fields)

    # Merge sample variables into packet_ds
    # This works because the coordinates and dimensions in sample_ds are different than the
    # coordinates and dimensions in packet_ds
    packet_ds = packet_ds.merge(sample_ds)

    # The "PACKET" dimension is retained to enable sample-to-packet traceability
    # packet_time_dimension remains as a non-dimension coordinate

    # Force any Unicode (U-type) string variables to byte (S-type) strings. We know all the FSW strings
    # are ASCII and sticking to bytes preserves better compatibility in NetCDF (i.e. doesn't rely on the NC_STRING
    # type).
    for var_name, var in packet_ds.data_vars.items():
        if var.dtype.kind == "U":
            packet_ds[var_name] = var.astype("S")

    # Add global dynamic attributes that are required per the data product configs but do not have static values
    global_attrs: dict[str, str | list | set] = {
        ALGORITHM_VERSION_ATTRIBUTE: version(),
    }
    global_attrs["input_files"] = [f.name for f in _packet_files]
    packet_ds.attrs.update(global_attrs)

    if packet_config.packet_apid == LiberaApid.icie_wfov_sci:
        packet_ds = enhance_wfov_l1a_dataset(packet_ds)

    return packet_ds


def _validate_duplicate_values(
    dataset: xr.Dataset,
    coordinate_name: str,
    duplicates: np.ndarray,
    ground_data: bool = False,
    verbose: bool = False,
) -> None:
    """Validate that duplicate coordinate entries are identical across all data variables.

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset containing the duplicates to validate.
    coordinate_name : str
        The name of the coordinate being deduplicated (used in error messages).
    duplicates : np.ndarray
        The coordinate values that appear more than once.
    ground_data : bool, optional
        If True, non-identical duplicate rows are tolerated instead of raising ``ValueError`` (so
        deduplication can proceed). Whether a ``UserWarning`` is emitted for each case is controlled
        by ``verbose``. Default is False, which raises for any non-identical duplicates.
    verbose : bool, optional
        When ``ground_data`` is True, controls whether a warning is issued for each non-identical
        duplicate. Default True so direct calls surface diagnostics; ``_drop_duplicates`` passes its
        own ``verbose`` (default False) to limit noise during batch parsing.

    Raises
    ------
    ValueError
        If values have differing data values across any variable (when
        ``ground_data`` is False).
    """
    dim_name = dataset[coordinate_name].dims[0]
    coord_values = dataset[coordinate_name].values

    # Build a boolean mask marking every row whose coord value appears in `duplicates`.
    dup_mask = np.isin(coord_values, duplicates)
    dup_indices_all = np.where(dup_mask)[0]
    dup_coord_vals = coord_values[dup_indices_all]

    sort_order = np.argsort(dup_coord_vals, kind="stable")
    dup_indices_sorted = dup_indices_all[sort_order]
    dup_coord_vals_sorted = dup_coord_vals[sort_order]

    dup_slice = dataset.isel({dim_name: dup_indices_sorted})

    # Computed once, outside the loop, using the sorted array:
    unique_dup_vals, group_starts = np.unique(dup_coord_vals_sorted, return_index=True)
    group_ends = np.append(group_starts[1:], len(dup_coord_vals_sorted))

    for var_name, var in dup_slice.data_vars.items():
        data = var.values

        for dup_val, start, end in zip(unique_dup_vals, group_starts, group_ends):
            group_data = data[start:end]
            # Vectorized identity check: broadcast first row against all rows in the group.
            if not np.all(group_data == group_data[0]):
                pairs = list(zip(group_data, dup_coord_vals[start:end]))
                if ground_data:
                    if verbose:
                        warnings.warn(
                            f"Duplicate coordinate value '{dup_val}' in '{coordinate_name}' "
                            f"has differing values in variable '{var_name}' at {pairs}. "
                            f"Proceeding because ground_data=True."
                        )
                    continue
                raise ValueError(
                    f"Duplicate coordinate value '{dup_val}' in '{coordinate_name}' "
                    f"has differing values in variable '{var_name}' at {pairs}. "
                    f"Dropping this duplicate would result in data loss."
                )


def _drop_duplicates(dataset: xr.Dataset, coordinate_name: str, ground_data: bool = False, verbose: bool = False):
    """Detect and drop duplicate values based on a coordinate

    Runs a validation function which raises an error if duplicate
    coordinate values in the dataset have differing data values, as only identical duplicates are safe to drop.

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset to deduplicate
    coordinate_name : str
        The name of the dimension coordinate over which to search for duplicates.
    ground_data : bool, optional
        If True, non-identical duplicates are allowed instead of raising ``ValueError``. Per-duplicate
        ``UserWarning`` messages are emitted only when ``verbose`` is also True. Default is False.
    verbose : bool, optional
        If True and ``ground_data`` is True, emit a ``UserWarning`` for each non-identical duplicate
        coordinate value. Default False so batch parsing stays quiet unless opted in.

    Returns
    -------
    dataset : xr.Dataset
        Deduplicated dataset
    n_duplicates : int
        Number of duplicates detected and dropped

    Raises
    ------
    KeyError
        If the coordinate is not found in the dataset.
    ValueError
        If the coordinate is not 1-dimensional, or if duplicate coordinate.
    """
    # Validate coordinate exists
    if coordinate_name not in dataset.coords:
        raise KeyError(f"Coordinate '{coordinate_name}' not found in dataset")

    coord = dataset[coordinate_name]

    # Ensure coordinate is 1-dimensional
    if len(coord.dims) != 1:
        raise ValueError(
            f"Coordinate '{coordinate_name}' must be 1-dimensional to deduplicate, but has dimensions: {coord.dims}"
        )

    dim_name = coord.dims[0]

    # Optimize for the common case (no duplicates) by checking size first
    # Use np.unique to find first occurrence of each unique value
    coord_values = coord.values
    # Obtain unique values, their first-occurrence indices, and per-value counts in one pass.
    unique_values, unique_indices, counts = np.unique(coord_values, return_index=True, return_counts=True)

    # Sort indices to maintain original order in the dataset
    # np.unique sorts the VALUES (not indices), so indices may be out of order
    # We want to preserve the original row order when selecting
    unique_indices_sorted = np.sort(unique_indices)

    original_size = len(coord_values)
    n_duplicates = original_size - len(unique_indices_sorted)

    if n_duplicates > 0:
        duplicates = unique_values[counts > 1]

        _validate_duplicate_values(dataset, coordinate_name, duplicates, ground_data, verbose)
        # Select only the first occurrence of each unique coordinate value
        dataset_deduped = dataset.isel({dim_name: unique_indices_sorted})

        warnings.warn(
            f"Detected {n_duplicates} duplicate {coordinate_name} in dataset. Use verbose=True to see warnings for the duplicates."
        )
        logger.warning(
            f"Duplicate coordinates detected ({n_duplicates}) in {coordinate_name}: {duplicates}. Use verbose=True to see warnings for the duplicates."
        )
    else:
        # No duplicates, return original dataset
        dataset_deduped = dataset

    return dataset_deduped, n_duplicates


def _expand_sample_group(dataset: xr.Dataset, group: SampleGroup) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Expand a sample group (timestamps and measured values) into separate field arrays.

    For samples within a packet (1 or many), expand those samples into separate arrays,
    with coordinates of sample time rather than packet time.

    Notes
    -----
    For periodic samples based on an epoch, we use the epoch and the period to calculate sample times assuming
    that the epoch is the first sample time.
    For samples that each have their own timestamp, we convert each sample time to microseconds.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset containing the packet data.
    group : SampleGroup
        Configuration for the sample group.

    Returns
    -------
    tuple[dict[str, np.ndarray], np.ndarray]
        Dictionary of field name to field array, and time array.
    """
    n_samples = group.sample_count

    # Calculate sample times
    if group.time_field_patterns:
        # Explicit per-sample timestamps
        sample_times = expand_sample_times(dataset, group.time_field_patterns, n_samples)
    elif group.epoch_time_fields and group.sample_period:
        # Use epoch + period to calculate sample timestamps
        epoch_times_dt64 = multipart_to_dt64(dataset, **group.epoch_time_fields.multipart_kwargs)
        epoch_times_us = epoch_times_dt64.values.astype(DATETIME_USEC_DTYPE)
        period_us = np.timedelta64(int(group.sample_period.total_seconds() * 1e6), "us")
        # Epoch times are 1 per packet so create an array that is (n_samples, n_packets), transpose, and flatten it
        sample_times = np.array([epoch_times_us + i * period_us for i in range(n_samples)]).T.flatten()
    else:
        raise ValueError(f"Sample group {group.name} must have either time_fields or epoch_time_fields+sample_period")

    # Expand data fields into individual arrays
    field_arrays = {}
    for field_pattern, clean_field_name in zip(group.data_field_patterns, group.sample_data_fields):
        if group.sample_count > 1:
            # Multi-sample field - collect all samples for this field pattern
            field_data = []
            for i in range(n_samples):
                if (field_name_i := field_pattern % i) in dataset:
                    field_data.append(dataset[field_name_i].values)
            # field_data is a list of length n_samples containing arrays with length n_packets
            # Stack samples (n_packets, n_samples) and flatten: (n_packets, n_samples) -> (n_packets * n_samples,)
            stacked_data = np.stack(field_data, axis=1)
            field_arrays[clean_field_name] = stacked_data.flatten()
        else:
            # Single sample per packet
            field_arrays[field_pattern] = dataset[field_pattern].values

    return field_arrays, sample_times


def expand_sample_times(dataset: xr.Dataset, time_fields: TimeFieldMapping, n_samples: int) -> np.ndarray:
    """Expand sample time fields into a flat array.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset containing the time fields.
    time_fields : TimeFieldMapping
        Time field mapping with patterns that may include %i placeholders.
    n_samples : int
        Number of samples per packet.

    Returns
    -------
    np.ndarray
        Flattened array of sample times as datetime64[us].
    """
    if n_samples > 1:
        # Multiple samples per packet - need to expand %i patterns
        sample_times_list = []
        for i in range(n_samples):
            # Create TimeFieldMapping for this specific sample index
            sample_kwargs = {}
            for field_type, field_pattern in time_fields.multipart_kwargs.items():
                if field_pattern is not None:
                    sample_field_name = field_pattern % i
                    if sample_field_name in dataset:
                        sample_kwargs[field_type] = sample_field_name

            if sample_kwargs:
                sample_time_dt64 = multipart_to_dt64(dataset, **sample_kwargs)
                sample_times_list.append(sample_time_dt64.values.astype(DATETIME_USEC_DTYPE))

        # Stack samples (n_packets, n_samples) and flatten
        if sample_times_list:
            stacked_times = np.stack(sample_times_list, axis=1)
            return stacked_times.flatten()
        else:
            # No valid time fields found
            return np.array([], dtype=DATETIME_USEC_DTYPE)
    else:
        # Single sample per packet - use time_fields directly
        sample_time_dt64 = multipart_to_dt64(dataset, **time_fields.multipart_kwargs)
        return sample_time_dt64.values.astype(DATETIME_USEC_DTYPE)


def _get_expanded_field_names(dataset: xr.Dataset, group: SampleGroup) -> set[str]:
    """Get all field names that are expanded for a sample group.

    This extracts all the field names for a sample group that we use to expand the samples
    (time fields and data fields) so that we can remove these fields from the primary array to save space.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset containing the fields.
    group : SampleGroup
        Sample group configuration.

    Returns
    -------
    set[str]
        Set of field names that are expanded.
    """
    expanded = set()

    # Add data fields
    for field_pattern in group.data_field_patterns:
        if (n_samples := group.sample_count) > 1:
            # Multi-sample pattern
            for i in range(n_samples):
                if (field_name := field_pattern % i) in dataset:
                    expanded.add(field_name)
        else:
            # Single field
            if field_pattern in dataset:
                expanded.add(field_pattern)

    # Add time fields
    if group.time_field_patterns:
        # Times provided per sample
        for field_pattern in group.time_field_patterns.multipart_kwargs.values():
            if field_pattern is not None:
                if (n_samples := group.sample_count) > 1:
                    for i in range(n_samples):
                        if (field_name := field_pattern % i) in dataset:
                            expanded.add(field_name)
                else:
                    if field_pattern in dataset:
                        expanded.add(field_pattern)
    elif group.epoch_time_fields:
        # Times calculated from epoch and periodic sampling
        for field_name in group.epoch_time_fields.multipart_kwargs.values():
            if field_name is not None and field_name in dataset:
                expanded.add(field_name)

    return expanded


def _aggregate_fields(dataset: xr.Dataset, group: AggregationGroup) -> np.ndarray:
    """Aggregate multiple sequential fields into a single binary blob per packet.

    Optimized using vectorized numpy operations with zero-copy view conversion.
    Assumes all fields exist (validated by Space Packet Parser during parsing).
    Assumes all fields are interpretable as bytes objects regardless of original dtype.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset containing the individual fields to aggregate.
    group : AggregationGroup
        Configuration for the aggregation group.

    Returns
    -------
    np.ndarray
        Array of aggregated binary data with dtype matching group.dtype.
    """
    n_packets = dataset.sizes[SDC_PACKET_DIMENSION]

    if group.dtype.kind != "S" or group.dtype.itemsize <= 0:
        raise ValueError(f"Aggregation group {group.name} requires fixed-width byte-string dtype; got {group.dtype}.")
    if group.dtype.itemsize % group.field_count != 0:
        raise ValueError(
            f"Aggregation group {group.name} has invalid dtype size {group.dtype.itemsize} for "
            f"field_count {group.field_count}; size must be divisible by field_count."
        )

    # Extract all field arrays at once (fail fast if any missing)
    field_arrays = []
    aggregate_size = 0  # Track total size of aggregated fields (per packet)
    bytes_per_field = group.dtype.itemsize // group.field_count

    for i in range(group.field_count):
        field_name = group.field_pattern % i
        if field_name not in dataset:
            raise KeyError(f"Required field {field_name} not found for aggregation group {group.name}")

        # The field_array is the data for one field across all packets
        field_array = dataset[field_name].values
        field_size = field_array.dtype.itemsize

        if field_array.dtype.kind in {"U", "S"}:
            # Packet parser commonly emits fixed-width ASCII-like telemetry as Unicode.
            # Cast explicitly to bytes so itemsize checks and final view() are deterministic.
            field_array = field_array.astype(f"S{bytes_per_field}")
            field_size = field_array.dtype.itemsize

        # Adds up total size of the aggregated fields (per packet)
        # This allows for fields that are not all the same size/dtype to be aggregated together as bytes
        aggregate_size += field_size

        field_arrays.append(field_array)

    if aggregate_size != group.dtype.itemsize:
        raise ValueError(
            f"Aggregation group {group.name} size mismatch: "
            f"expected total size {group.dtype.itemsize} bytes, got {aggregate_size} bytes."
        )

    # Stack all fields: shape (n_fields, n_packets)
    stacked_and_transposed = np.stack(field_arrays, axis=-1)

    # Use view() to reinterpret each row as a single bytes string - zero copy!
    # This is a key optimization - no iteration, just memory reinterpretation
    return stacked_and_transposed.view(dtype=group.dtype).reshape(n_packets)


def _get_aggregated_field_names(dataset: xr.Dataset, group: AggregationGroup) -> set[str]:
    """Get all field names that are aggregated for an aggregation group.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset containing the fields.
    group : AggregationGroup
        Aggregation group configuration.

    Returns
    -------
    set[str]
        Set of field names that are aggregated.
    """
    aggregated = set()
    for i in range(group.field_count):
        field_name = group.field_pattern % i
        if field_name in dataset:
            aggregated.add(field_name)
    return aggregated


def _normalize_field_dtype(
    field_array: np.ndarray,
    target_dtype: np.dtype,
    *,
    field_name: str,
    group_name: str,
) -> np.ndarray:
    """Normalize a field array to the expected group dtype with explicit coercion rules.

    Only Unicode-to-bytes normalization is allowed for string fields. All other dtype
    mismatches raise ValueError to avoid silent cross-kind coercion (e.g. int -> S).
    """
    if field_array.dtype == target_dtype:
        return field_array

    source_kind = field_array.dtype.kind
    target_kind = target_dtype.kind

    if source_kind == "U" and target_kind == "S":
        return field_array.astype(target_dtype)

    if source_kind == "S" and target_kind == "S":
        if field_array.dtype.itemsize != target_dtype.itemsize:
            raise ValueError(
                f"Array group {group_name}: field {field_name} has width {field_array.dtype}, expected {target_dtype}."
            )
        return field_array.astype(target_dtype)

    raise ValueError(
        f"Array group {group_name}: field {field_name} has dtype {field_array.dtype}, expected {target_dtype}."
    )


def _stack_fields(dataset: xr.Dataset, group: ArrayGroup) -> np.ndarray:
    """Stack multiple sequential fields into a fixed-size array per packet.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset containing the individual fields to stack.
    group : ArrayGroup
        Configuration for the array group.

    Returns
    -------
    np.ndarray
        Array with shape (n_packets, field_count) and dtype matching group.dtype.
    """
    field_arrays = []

    for i in range(group.field_count):
        field_name = group.field_pattern % i
        if field_name not in dataset:
            raise KeyError(f"Required field {field_name} not found for array group {group.name}")

        field_array = _normalize_field_dtype(
            dataset[field_name].values,
            group.dtype,
            field_name=field_name,
            group_name=group.name,
        )
        field_arrays.append(field_array)

    return np.stack(field_arrays, axis=1)


def _get_array_group_field_names(dataset: xr.Dataset, group: ArrayGroup) -> set[str]:
    """Get all field names that are stacked for an array group.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset containing the fields.
    group : ArrayGroup
        Array group configuration.

    Returns
    -------
    set[str]
        Set of field names that are stacked.
    """
    array_fields = set()
    for i in range(group.field_count):
        field_name = group.field_pattern % i
        if field_name in dataset:
            array_fields.add(field_name)
    return array_fields
