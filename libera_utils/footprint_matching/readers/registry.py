"""Reader plugin registry for the footprint matching reader plugin system.

Concrete GriddedDataReader subclasses auto-register in this registry via the
``__init_subclass__`` hook defined in the base class. Callers should not
populate ``_registry`` directly.

Usage
-----
Import the readers subpackage to trigger registration of all built-in readers::

    import libera_utils.footprint_matching.readers  # registers all built-in readers
    from libera_utils.footprint_matching.readers.registry import ReaderRegistry

    cls = ReaderRegistry.get("igbp")
    reader = cls(Path("MCD12Q1.A2023001.h09v05.061.hdf"))

See Also
--------
libera_utils.footprint_matching.readers.base.GriddedDataReader :
    Abstract base class whose ``__init_subclass__`` performs registration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Avoid a circular import at module load time; only used for type hints
    from libera_utils.footprint_matching.readers.base import GriddedDataReader
    from libera_utils.footprint_matching.types import OperationalMode


class ReaderRegistry:
    """Plugin registry mapping string keys to GriddedDataReader subclasses.

    This is a stateless class (all methods are static) that wraps a module-level
    ``_registry`` dict. Readers register themselves automatically when their module
    is imported via the ``__init_subclass__`` hook in GriddedDataReader. Manual
    registration is neither necessary nor recommended.

    Examples
    --------
    >>> import libera_utils.footprint_matching.readers  # triggers registration
    >>> ReaderRegistry.list_readers()
    ['cldpix', 'era5', 'igbp', 'nise', 'ssf', 'viirs_aod', 'viirs_brdf', 'viirs_cloud']
    >>> cls = ReaderRegistry.get("igbp")
    >>> cls.RESOLUTION_KM
    1.0
    """

    # Shared registry dict. All concrete GriddedDataReader subclasses are added
    # here by GriddedDataReader.__init_subclass__ at class-definition time
    # (i.e., when their module is first imported).
    _registry: dict[str, type[GriddedDataReader]] = {}

    @staticmethod
    def get(name: str) -> type[GriddedDataReader]:
        """Return the reader class registered under ``name``.

        Parameters
        ----------
        name : str
            Registry key (e.g., ``"igbp"``).

        Returns
        -------
        type[GriddedDataReader]
            The registered reader class. Callers can then instantiate it with a
            file path: ``cls(Path("some_file.hdf"))``.

        Raises
        ------
        KeyError
            If no reader is registered under ``name``.
        """
        if name not in ReaderRegistry._registry:
            raise KeyError(
                f"No reader registered with name {name!r}. Available readers: {ReaderRegistry.list_readers()}"
            )
        return ReaderRegistry._registry[name]

    @staticmethod
    def list_readers() -> list[str]:
        """Return a sorted list of all registered reader keys.

        Returns
        -------
        list[str]
            Alphabetically sorted registry keys.
        """
        return sorted(ReaderRegistry._registry.keys())

    @staticmethod
    def get_readers_for_mode(mode: OperationalMode) -> dict[str, type[GriddedDataReader]]:
        """Return readers whose ``REQUIRED_MODE`` rank is <= the given mode's rank.

        The TileManager calls this during orchestrator initialization to build
        the set of readers that are active for the current operational mode.
        Readers with a higher-latency REQUIRED_MODE are excluded.

        Parameters
        ----------
        mode : OperationalMode
            The active operational mode to filter by.

        Returns
        -------
        dict[str, type[GriddedDataReader]]
            Mapping of registry key to reader class for all readers active in
            ``mode``.

        Examples
        --------
        >>> from libera_utils.footprint_matching.types import OperationalMode
        >>> import libera_utils.footprint_matching.readers
        >>> readers = ReaderRegistry.get_readers_for_mode(OperationalMode.CAM)
        >>> sorted(readers.keys())
        ['era5', 'igbp', 'nise', 'viirs_brdf', 'viirs_cloud']
        """
        return {key: cls for key, cls in ReaderRegistry._registry.items() if cls.REQUIRED_MODE.rank <= mode.rank}
