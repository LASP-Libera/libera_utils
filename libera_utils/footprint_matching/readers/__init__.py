"""Reader plugin subpackage for the footprint matching pipeline.

Importing this package triggers the registration of all built-in reader
subclasses via the ``__init_subclass__`` hook defined in
:class:`~libera_utils.footprint_matching.readers.base.GriddedDataReader`.
After this import, :class:`~libera_utils.footprint_matching.readers.registry.ReaderRegistry`
will list all five readers: ``era5``, ``igbp``, ``nise``, ``viirs_brdf``, ``viirs_cloud``.

Public API
----------
ReaderRegistry : class
    Plugin registry — use ``ReaderRegistry.get(name)`` to retrieve a reader class.
GriddedDataReader : abstract class
    Base class for all reader plugins. Subclass this to add new readers.
TILE_SIZE_DEG : float
    Global 2° tile size constant shared by all readers.

Examples
--------
>>> import libera_utils.footprint_matching.readers as readers_pkg
>>> from libera_utils.footprint_matching.readers.registry import ReaderRegistry
>>> ReaderRegistry.list_readers()
['era5', 'igbp', 'nise', 'viirs_brdf', 'viirs_cloud']
"""
# Importing each reader module causes its class to be defined, which triggers
# GriddedDataReader.__init_subclass__, which calls ReaderRegistry._registry[key] = cls.
# Order does not matter for correctness, but alphabetical is easiest to maintain.
from libera_utils.footprint_matching.readers import brdf, era5, igbp, nsidc, viirs  # noqa: F401
from libera_utils.footprint_matching.readers.base import TILE_SIZE_DEG, GriddedDataReader
from libera_utils.footprint_matching.readers.registry import ReaderRegistry

__all__ = [
    "ERA5Reader",
    "GriddedDataReader",
    "IGBPReader",
    "NISEReader",
    "ReaderRegistry",
    "TILE_SIZE_DEG",
    "VIIRSBRDFReader",
    "VIIRSCloudReader",
]
