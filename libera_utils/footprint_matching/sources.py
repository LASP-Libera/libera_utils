"""The active ancillary sources of one FMATCH run.

:class:`AncillarySources` holds one instantiated reader per source active in the
operational mode and answers the orchestrator's per-footprint queries: the native
samples of a source inside a footprint's bounding box, matched to the footprint
time by that reader's :class:`~libera_utils.footprint_matching.types.TimeMatch`
policy. It owns no cache of its own: each reader holds its source once (a static
reader its whole indexed body of data, a stepped reader the one hourly step
nearest the current footprint, a pass reader the hourly granules within
``MAX_TIME_OFFSET`` of the current footprint), and every query is answered from that.

A reader that cannot read its source raises from the first query that loads the
granule, and the run stops: a product with a source silently missing would be wrong,
and a crash gets noticed.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from libera_utils.footprint_matching.readers.registry import ReaderRegistry

if TYPE_CHECKING:
    from libera_utils.footprint_matching.readers.base import AncillaryReader
    from libera_utils.footprint_matching.types import BoundingBox, OperationalMode, SampleSet


class AncillarySources:
    """The readers active for one operational mode, queried per footprint.

    Production code constructs this through :func:`build_sources`; direct
    construction is for tests and demonstrations.

    Parameters
    ----------
    readers : dict[str, AncillaryReader]
        Mapping of reader registry key to an *instantiated* reader, already filtered
        to the active mode (see :func:`build_sources`).
    active_mode : OperationalMode
        The operational mode these sources serve; it labels error messages.
    """

    def __init__(self, readers: dict[str, AncillaryReader], active_mode: OperationalMode) -> None:
        self._readers = dict(readers)
        self._active_mode = active_mode

    @property
    def active_mode(self) -> OperationalMode:
        """The operational mode these sources were built for."""
        return self._active_mode

    @property
    def sources(self) -> list[str]:
        """Sorted registry keys of the active readers."""
        return sorted(self._readers)

    def reader(self, source: str) -> AncillaryReader:
        """Return the instantiated reader for ``source``.

        Raises
        ------
        KeyError
            If ``source`` is not active for this mode. This is a mode-configuration
            error, distinct from a read failure (which the reader raises itself).
        """
        if source not in self._readers:
            raise KeyError(
                f"Source {source!r} is not active for mode {self._active_mode.value!r}. Active sources: {self.sources}"
            )
        return self._readers[source]

    def query(self, source: str, bbox: BoundingBox, time: np.datetime64) -> SampleSet:
        """Return ``source``'s native samples inside ``bbox`` matched to ``time``.

        Parameters
        ----------
        source : str
            Registry key of a reader active in this mode.
        bbox : BoundingBox
            The footprint's PSF bounding box.
        time : np.datetime64
            The footprint's observation time.

        Returns
        -------
        SampleSet
            See :meth:`~libera_utils.footprint_matching.readers.base.AncillaryReader.samples_in`.

        Raises
        ------
        KeyError
            If ``source`` is not active for this mode.
        ValueError
            Propagated from the reader; see
            :meth:`~libera_utils.footprint_matching.readers.base.AncillaryReader.samples_in`.
        TypeError
            Propagated from the reader; see
            :meth:`~libera_utils.footprint_matching.readers.base.AncillaryReader.samples_in`.
        OSError
            If a staged granule cannot be opened, raised by the first query that loads it.
            A reader whose I/O library has its own error raises that instead, such as
            ``pyhdf.error.HDF4Error`` from the HDF4 readers (IGBP, NISE).
        """
        return self.reader(source).samples_in(bbox, time)


def build_sources(mode: OperationalMode, source_file_paths: dict[str, Sequence[Path]]) -> AncillarySources:
    """Instantiate the readers active for ``mode`` over their staged granules.

    Selects the active reader classes via
    :meth:`~libera_utils.footprint_matching.readers.registry.ReaderRegistry.get_readers_for_mode`,
    instantiates each with the granule paths supplied in ``source_file_paths``, and wraps them in an
    :class:`AncillarySources`. This is the single construction helper the product code
    calls so the reader wiring lives in one place. Each reader checks at construction
    that its staged granules exist but opens none of them, so an unreadable staged file
    raises from the first :meth:`AncillarySources.query` that loads it.

    Parameters
    ----------
    mode : OperationalMode
        Operational mode whose active readers should be built.
    source_file_paths : dict[str, Sequence[Path]]
        Mapping of reader registry key to the staged local granules for that source.
        Its keys must be exactly the readers active in ``mode``.

    Returns
    -------
    AncillarySources

    Raises
    ------
    KeyError
        If ``source_file_paths`` holds a key for a source ``mode`` does not use, or is
        missing an entry for any active reader.
    TypeError
        If an entry is a single ``str`` or ``Path`` rather than a sequence of them.
    ValueError
        If an entry holds no granule.
    FileNotFoundError
        If any staged granule path is not an existing file.
    """
    reader_classes = ReaderRegistry.get_readers_for_mode(mode)
    unexpected = sorted(set(source_file_paths) - set(reader_classes))
    if unexpected:
        raise KeyError(
            f"source_file_paths has keys {unexpected} that mode {mode.value!r} does not use. "
            f"Remove them, or correct a misspelled key; required sources: {sorted(reader_classes)}"
        )
    readers: dict[str, AncillaryReader] = {}
    for key, reader_cls in reader_classes.items():
        if key not in source_file_paths:
            raise KeyError(
                f"No file path supplied for active source {key!r} (mode {mode.value!r}). "
                f"Required sources: {sorted(reader_classes)}"
            )
        readers[key] = reader_cls(source_file_paths[key])
    return AncillarySources(readers, mode)
