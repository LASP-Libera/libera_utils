"""Reader plugin subpackage for the footprint matching pipeline.

Importing this package triggers the registration of all built-in reader
subclasses via the ``__init_subclass__`` hook defined in
:class:`~libera_utils.footprint_matching.readers.base.GriddedDataReader`.
After this import, :class:`~libera_utils.footprint_matching.readers.registry.ReaderRegistry`
will list all readers: ``era5``, ``era5_pressure``, ``igbp``, ``nise``,
``viirs_aod``, ``viirs_brdf``.

Note: the swath-product readers (``ssf``, ``cldpix``, ``viirs_cloud``) are added by the
next PR in this stack; this module then registers all nine readers.

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
['era5', 'era5_pressure', 'igbp', 'nise', 'viirs_aod', 'viirs_brdf']
"""

# Importing each reader class defines it, which triggers
# GriddedDataReader.__init_subclass__, which calls ReaderRegistry._registry[key] = cls.
# Binding the classes here (not just the modules) keeps ``__all__`` importable so
# ``from ...readers import IGBPReader`` and ``import *`` work as advertised.
from libera_utils.footprint_matching.readers.aod import VIIRSAODReader
from libera_utils.footprint_matching.readers.base import TILE_SIZE_DEG, GriddedDataReader
from libera_utils.footprint_matching.readers.brdf import VIIRSBRDFReader
from libera_utils.footprint_matching.readers.era5 import ERA5Reader
from libera_utils.footprint_matching.readers.era5_pressure import ERA5PressureLevelReader
from libera_utils.footprint_matching.readers.igbp import IGBPReader
from libera_utils.footprint_matching.readers.nsidc import NISEReader
from libera_utils.footprint_matching.readers.registry import ReaderRegistry

__all__ = [
    "ERA5PressureLevelReader",
    "ERA5Reader",
    "GriddedDataReader",
    "IGBPReader",
    "NISEReader",
    "ReaderRegistry",
    "TILE_SIZE_DEG",
    "VIIRSAODReader",
    "VIIRSBRDFReader",
]
