"""Reader plugin registry for the footprint matching reader plugin system.

Concrete AncillaryReader subclasses auto-register in this registry via the
``__init_subclass__`` hook defined in the base class. Callers should not
populate ``_registry`` directly.

Usage
-----
Import the readers subpackage to trigger registration of all built-in readers::

    import libera_utils.footprint_matching.readers  # registers all built-in readers
    from libera_utils.footprint_matching.readers.registry import ReaderRegistry

    cls = ReaderRegistry.get("IGBP")
    reader = cls([Path("MCD12Q1.A2023001.h09v05.061.hdf")])

See Also
--------
libera_utils.footprint_matching.readers.base.AncillaryReader :
    Abstract base class whose ``__init_subclass__`` performs registration.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

# OperationalMode is imported at runtime (not just for type checking): it keys the
# per-product reader-set map below. types.py is dependency-free, so this cannot
# create an import cycle.
from libera_utils.footprint_matching.types import OperationalMode

if TYPE_CHECKING:
    # Avoid a circular import at module load time; only used for type hints
    from libera_utils.footprint_matching.readers.base import AncillaryReader


# ---------------------------------------------------------------------------
# Per-product reader membership
# ---------------------------------------------------------------------------
#: The readers that contribute to each shipped FMATCH product, keyed by OperationalMode. A
#: reader contributes to a product when any of its product variables names that product in
#: its ``VariableSpec.modes``. ``readers/__init__.py`` fills this once, from the built-in
#: readers, after they have all registered; it is empty while that package is importing.
FMATCH_MODE_READERS: dict[OperationalMode, frozenset[str]] = {}


def _derive_mode_readers(readers: Mapping[str, type[AncillaryReader]]) -> dict[OperationalMode, frozenset[str]]:
    """Return, for every operational mode, the keys of the readers that contribute to it.

    A reader contributes to a mode when any spec in its ``product_variable_specs()``
    names that mode in ``VariableSpec.modes``.

    Parameters
    ----------
    readers : Mapping[str, type[AncillaryReader]]
        Registry key to reader class, for the readers whose membership to derive.

    Returns
    -------
    dict[OperationalMode, frozenset[str]]
        One entry per :class:`OperationalMode`, holding the keys of its readers.

    Raises
    ------
    ValueError
        If a mode has no reader, so its product would carry no ancillary variables.
    """
    mode_readers = {
        mode: frozenset(
            key
            for key, reader in readers.items()
            if any(mode in spec.modes for spec in reader.product_variable_specs())
        )
        for mode in OperationalMode
    }
    empty = [mode.value for mode, keys in mode_readers.items() if not keys]
    if empty:
        raise ValueError(
            f"No reader supplies a variable for {', '.join(empty)}: give at least one reader's "
            "VariableSpec.modes each of these products, or remove the product."
        )
    return mode_readers


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
    ['CLDPIX', 'ERA5_Pressure', 'ERA5_Surface', 'FLASHFlux', 'IGBP', 'NISE', 'VIIRS_AOD', 'VIIRS_BRDF', 'VIIRS_CLDPROP_D3']
    >>> ReaderRegistry.get("IGBP").__name__
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
            Registry key (e.g., ``"IGBP"``).

        Returns
        -------
        type[AncillaryReader]
            The registered reader class. Callers can then instantiate it with a
            list of staged granule paths: ``cls([Path("some_file.hdf")])``.

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
    def get_readers_for_mode(mode: OperationalMode) -> dict[str, type[AncillaryReader]]:
        """Return the readers that contribute to a product's definition.

        Membership is read from :data:`FMATCH_MODE_READERS`, which is derived from the
        built-in readers' ``VariableSpec.modes``.

        Parameters
        ----------
        mode : OperationalMode
            The operational mode (product) whose readers to return.

        Returns
        -------
        dict[str, type[AncillaryReader]]
            Mapping of registry key to reader class for every reader in the product's
            set. Only the built-in readers are returned; readers other code registers
            after the package import are not included.

        Examples
        --------
        >>> from libera_utils.footprint_matching.types import OperationalMode
        >>> import libera_utils.footprint_matching.readers  # triggers registration
        >>> sorted(ReaderRegistry.get_readers_for_mode(OperationalMode.CAM))
        ['IGBP', 'NISE', 'VIIRS_CLDPROP_D3']
        >>> imager = ReaderRegistry.get_readers_for_mode(OperationalMode.IMAGER)
        >>> 'CLDPIX' in imager and 'ERA5_Pressure' in imager  # RBSP alongside ERA5
        True
        >>> camtime = ReaderRegistry.get_readers_for_mode(OperationalMode.IMAGER_CAMTIME)
        >>> 'ERA5_Pressure' in camtime  # pressure-level ERA5 is radiometer-timescale only
        False
        """
        return {key: ReaderRegistry._registry[key] for key in FMATCH_MODE_READERS[mode]}
