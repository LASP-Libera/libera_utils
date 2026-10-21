"""Reader plugin registry for the footprint matching reader plugin system.

Concrete AncillaryReader subclasses auto-register in this registry via the
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
libera_utils.footprint_matching.readers.base.AncillaryReader :
    Abstract base class whose ``__init_subclass__`` performs registration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Avoid a circular import at module load time; only used for type hints
    from libera_utils.footprint_matching.readers.base import AncillaryReader


class ReaderRegistry:
    """Plugin registry mapping string keys to AncillaryReader subclasses.

    This is a stateless class (all methods are static) that wraps a module-level
    ``_registry`` dict. Readers register themselves automatically when their module
    is imported via the ``__init_subclass__`` hook in AncillaryReader. Manual
    registration is neither necessary nor recommended.

    Examples
    --------
    >>> import libera_utils.footprint_matching.readers  # triggers registration
    >>> ReaderRegistry.list_readers()
    ['era5', 'era5_pressure', 'igbp', 'nise', 'viirs_aod', 'viirs_brdf']
    >>> ReaderRegistry.get("igbp").__name__
    'IGBPReader'
    """

    # Shared registry dict. All concrete AncillaryReader subclasses are added
    # here by AncillaryReader.__init_subclass__ at class-definition time
    # (i.e., when their module is first imported).
    _registry: dict[str, type[AncillaryReader]] = {}

    @staticmethod
    def get(name: str) -> type[AncillaryReader]:
        """Return the reader class registered under ``name``.

        Parameters
        ----------
        name : str
            Registry key (e.g., ``"igbp"``).

        Returns
        -------
        type[AncillaryReader]
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
