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

# Both imported at runtime (not just for type checking): OperationalMode keys the per-product
# reader-set map and DataProductIdentifier keys the auxiliary-product map below. types.py is
# dependency-free and constants.py only pulls in aws.constants, so neither creates an import cycle.
from libera_utils.constants import DataProductIdentifier
from libera_utils.footprint_matching.types import OperationalMode

if TYPE_CHECKING:
    # Avoid a circular import at module load time; only used for type hints
    from libera_utils.footprint_matching.readers.base import GriddedDataReader


# ---------------------------------------------------------------------------
# Per-product reader membership
# ---------------------------------------------------------------------------
# One fully-enumerated reader set per shipped FMATCH product -- the single source of truth
# for which readers contribute to which product (readers do not self-gate by mode). Keyed by
# OperationalMode like ``product.FMATCH_DEFINITION_FILENAMES``; ``get_readers_for_mode``
# resolves ``mode`` with the same logic as ``load_fmatch_definition``, so the active reader
# set and the loaded definition can never drift. Each set is spelled out in full (names
# repeat by design) rather than composed. Note FMATCH-IMAGER-CAMTIME omits ``era5_pressure``:
# the ERA5 pressure levels are a radiometer-timescale quantity (see fmatch_imager_camtime.yml).
FMATCH_MODE_READERS: dict[OperationalMode, frozenset[str]] = {
    OperationalMode.CAM: frozenset({"era5", "igbp", "nise", "viirs_brdf", "viirs_cloud"}),
    OperationalMode.CAM_CAMTIME: frozenset({"era5", "igbp", "nise", "viirs_brdf", "viirs_cloud"}),
    OperationalMode.IMAGER_FLASH: frozenset({"era5", "igbp", "nise", "viirs_brdf", "viirs_cloud", "ssf"}),
    OperationalMode.IMAGER: frozenset(
        {"era5", "igbp", "nise", "viirs_brdf", "viirs_cloud", "viirs_aod", "era5_pressure", "cldpix", "ssf"}
    ),
    OperationalMode.IMAGER_CAMTIME: frozenset(
        {"era5", "igbp", "nise", "viirs_brdf", "viirs_cloud", "viirs_aod", "cldpix", "ssf"}
    ),
}


# ---------------------------------------------------------------------------
# Auxiliary product -> reader mapping
# ---------------------------------------------------------------------------
# Maps each external auxiliary DataProductIdentifier (the ``auxiliary_*`` members) to the
# ReaderRegistry key of the reader that consumes it. This lives here -- in the footprint matching
# module -- rather than on the enum in constants.py because reader membership is a
# footprint-matching concern; constants.py must stay free of that coupling. A single reader may
# consume more than one product (VIIRS BRDF params and albedo are both read by ``viirs_brdf``),
# so the mapping is product -> reader (many-to-one). Every reader named in FMATCH_MODE_READERS
# must appear here and vice versa; the consistency guard lives in the registry tests.
AUXILIARY_PRODUCT_READERS: dict[DataProductIdentifier, str] = {
    DataProductIdentifier.auxiliary_igbp_mcd12q1: "igbp",
    DataProductIdentifier.auxiliary_nise: "nise",
    DataProductIdentifier.auxiliary_viirs_brdf: "viirs_brdf",
    DataProductIdentifier.auxiliary_viirs_brdf_albedo: "viirs_brdf",
    DataProductIdentifier.auxiliary_viirs_cloud: "viirs_cloud",
    DataProductIdentifier.auxiliary_viirs_aod: "viirs_aod",
    DataProductIdentifier.auxiliary_ceres_ssf: "ssf",
    DataProductIdentifier.auxiliary_ceres_cldpix: "cldpix",
    DataProductIdentifier.auxiliary_era5_single_level: "era5",
    DataProductIdentifier.auxiliary_era5_pressure_level: "era5_pressure",
}


def reader_key_for_auxiliary_product(product_id: DataProductIdentifier) -> str:
    """Return the reader key that consumes a given auxiliary product.

    Parameters
    ----------
    product_id : DataProductIdentifier
        One of the ``auxiliary_*`` members.

    Returns
    -------
    str
        The ReaderRegistry key of the reader that consumes ``product_id``.

    Raises
    ------
    KeyError
        If ``product_id`` is not a mapped auxiliary product (e.g. a Libera-produced product).
    """
    if product_id not in AUXILIARY_PRODUCT_READERS:
        raise KeyError(
            f"{product_id!r} is not an auxiliary input product. "
            f"Auxiliary products: {sorted(p.name for p in AUXILIARY_PRODUCT_READERS)}"
        )
    return AUXILIARY_PRODUCT_READERS[product_id]


def auxiliary_products_for_reader(reader_key: str) -> tuple[DataProductIdentifier, ...]:
    """Return the auxiliary products consumed by a given reader key.

    Parameters
    ----------
    reader_key : str
        A ReaderRegistry key (e.g. ``"igbp"``, ``"viirs_brdf"``).

    Returns
    -------
    tuple[DataProductIdentifier, ...]
        Every auxiliary product mapped to ``reader_key``, in mapping order. A reader may own more
        than one (``viirs_brdf`` reads both VJ143C1 params and VJ143C3 albedo). Empty if no product
        names this reader key.
    """
    return tuple(product_id for product_id, key in AUXILIARY_PRODUCT_READERS.items() if key == reader_key)


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
    ['cldpix', 'era5', 'era5_pressure', 'igbp', 'nise', 'ssf', 'viirs_aod', 'viirs_brdf', 'viirs_cloud']
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
        """Return the readers that contribute to a product's definition.

        Membership is read directly from the per-product reader set
        (:data:`FMATCH_MODE_READERS`), keyed by the *same* ``mode`` as
        :func:`~libera_utils.footprint_matching.product.load_fmatch_definition`. Because
        both resolve identically, this returns exactly the readers whose variable blocks
        appear in the definition that ``load_fmatch_definition(mode)`` loads -- the active
        reader set and the product schema cannot drift.

        Parameters
        ----------
        mode : OperationalMode
            The operational mode (product) whose readers to return.

        Returns
        -------
        dict[str, type[GriddedDataReader]]
            Mapping of registry key to reader class for every reader in the product's
            set. Only production readers named in the sets are returned; throwaway
            readers other code may have registered are not included.

        Examples
        --------
        >>> from libera_utils.footprint_matching.types import OperationalMode
        >>> import libera_utils.footprint_matching.readers  # triggers registration
        >>> sorted(ReaderRegistry.get_readers_for_mode(OperationalMode.CAM))
        ['era5', 'igbp', 'nise', 'viirs_brdf', 'viirs_cloud']
        >>> imager = ReaderRegistry.get_readers_for_mode(OperationalMode.IMAGER)
        >>> 'cldpix' in imager and 'era5_pressure' in imager  # RBSP alongside ERA5
        True
        >>> camtime = ReaderRegistry.get_readers_for_mode(OperationalMode.IMAGER_CAMTIME)
        >>> 'era5_pressure' in camtime  # pressure-level ERA5 is radiometer-timescale only
        False
        """
        return {key: ReaderRegistry._registry[key] for key in FMATCH_MODE_READERS[mode]}
