"""Reader plugin subpackage for the footprint matching pipeline.

Importing this package triggers the registration of all built-in reader
subclasses via the ``__init_subclass__`` hook defined in
:class:`~libera_utils.footprint_matching.readers.base.AncillaryReader`.
After this import, :class:`~libera_utils.footprint_matching.readers.registry.ReaderRegistry`
lists all readers: ``cldpix``, ``era5_pressure``, ``era5_single_level``, ``igbp``,
``nise``, ``ssf``, ``viirs_aod``, ``viirs_brdf``, ``viirs_cldprop_d3``.

Public API
----------
ReaderRegistry : class
    Plugin registry -- use ``ReaderRegistry.get(name)`` to retrieve a reader class.
AncillaryReader : abstract class
    Base class for all reader plugins; subclass ``StaticReader`` (one body of data
    per period, served from memory), ``SteppedGridReader`` (a time series of global
    grids, served one step at a time) or ``PassReader`` (hourly swath granules, held
    by time window) to add a reader.
"""

# Importing each reader class defines it, which triggers
# AncillaryReader.__init_subclass__, which calls ReaderRegistry._registry[key] = cls.
# Binding the classes here (not just the modules) keeps ``__all__`` importable so
# ``from ...readers import IGBPReader`` and ``import *`` work as advertised.
from libera_utils.footprint_matching.readers.aod import VIIRSAODReader
from libera_utils.footprint_matching.readers.base import AncillaryReader, PassReader, StaticReader, SteppedGridReader
from libera_utils.footprint_matching.readers.brdf import VIIRSBRDFReader
from libera_utils.footprint_matching.readers.cldpix import CLDPIXReader
from libera_utils.footprint_matching.readers.era5 import ERA5Reader
from libera_utils.footprint_matching.readers.era5_pressure import ERA5PressureLevelReader
from libera_utils.footprint_matching.readers.igbp import IGBPReader
from libera_utils.footprint_matching.readers.nsidc import NISEReader
from libera_utils.footprint_matching.readers.registry import ReaderRegistry
from libera_utils.footprint_matching.readers.ssf import SSFReader
from libera_utils.footprint_matching.readers.viirs import VIIRSCldpropD3Reader

__all__ = [
    "AncillaryReader",
    "CLDPIXReader",
    "ERA5PressureLevelReader",
    "ERA5Reader",
    "IGBPReader",
    "NISEReader",
    "PassReader",
    "ReaderRegistry",
    "SSFReader",
    "StaticReader",
    "SteppedGridReader",
    "VIIRSAODReader",
    "VIIRSBRDFReader",
    "VIIRSCldpropD3Reader",
]
