"""The ancillary readers active in one FMATCH run.

:func:`build_sources` instantiates one reader per source the operational mode's
product uses, over the granules staged for it, and returns them keyed by reader
registry key. A footprint's native samples of a source come from that reader's
:meth:`~libera_utils.footprint_matching.readers.base.AncillaryReader.samples_in`,
matched to the footprint time by its
:class:`~libera_utils.footprint_matching.types.TimeMatch` policy. No cache is kept
here: each reader holds its source once (a static reader its whole indexed body of
data, a stepped reader the one hourly step nearest the current footprint, a pass
reader the hourly granules within ``MAX_TIME_OFFSET`` of the current footprint).

A staged granule that cannot be read raises from the first ``samples_in`` call that
loads it, and the run stops: a product with a source silently missing would be wrong,
and a crash gets noticed.
"""

from collections.abc import Iterable, Sequence
from pathlib import Path

from libera_utils.footprint_matching.readers.base import AncillaryReader
from libera_utils.footprint_matching.readers.registry import ReaderRegistry
from libera_utils.footprint_matching.types import OperationalMode


def require_mode_reader_keys(mode: OperationalMode, keys: Iterable[str], argument: str) -> None:
    """Check that ``keys`` are exactly the readers active in ``mode``.

    A source left out would leave its product columns at fill and score coverage over
    fewer sources than the product declares; a source the mode does not use would
    aggregate variables the product does not carry.

    Parameters
    ----------
    mode : OperationalMode
        Operational mode whose active readers the keys must match.
    keys : collections.abc.Iterable[str]
        Reader registry keys a caller supplied.
    argument : str
        Name of the caller's argument that holds the keys, for the error message.

    Raises
    ------
    KeyError
        If ``keys`` holds a key for a source ``mode`` does not use, or is missing a
        reader active in ``mode``.
    """
    required = set(ReaderRegistry.get_readers_for_mode(mode))
    keys = set(keys)
    unexpected = sorted(keys - required)
    if unexpected:
        raise KeyError(
            f"{argument} has keys {unexpected} that mode {mode.value!r} does not use. "
            f"Remove them, or correct a misspelled key; required sources: {sorted(required)}"
        )
    missing = sorted(required - keys)
    if missing:
        raise KeyError(
            f"{argument} has no entry for active sources {missing} (mode {mode.value!r}). "
            f"Required sources: {sorted(required)}"
        )


def build_sources(mode: OperationalMode, source_file_paths: dict[str, Sequence[Path]]) -> dict[str, AncillaryReader]:
    """Instantiate the readers active for ``mode`` over their staged granules.

    Selects the active reader classes via
    :meth:`~libera_utils.footprint_matching.readers.registry.ReaderRegistry.get_readers_for_mode`
    and instantiates each with the granule paths supplied in ``source_file_paths``. This
    is the single construction path the product code calls, so the reader wiring lives
    in one place. Each reader checks at construction that its staged granules exist but
    opens none of them, so an unreadable staged file raises from the first
    :meth:`~libera_utils.footprint_matching.readers.base.AncillaryReader.samples_in`
    call that loads it.

    Parameters
    ----------
    mode : OperationalMode
        Operational mode whose active readers should be built.
    source_file_paths : dict[str, Sequence[Path]]
        Mapping of reader registry key to the staged local granules for that source.
        Its keys must be exactly the readers active in ``mode``.

    Returns
    -------
    dict[str, AncillaryReader]
        Reader registry key to instantiated reader, one entry per source active in ``mode``.

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
    require_mode_reader_keys(mode, source_file_paths, "source_file_paths")
    reader_classes = ReaderRegistry.get_readers_for_mode(mode)
    return {key: reader_cls(source_file_paths[key]) for key, reader_cls in reader_classes.items()}
