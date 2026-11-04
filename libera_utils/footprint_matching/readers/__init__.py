"""Reader plugin subpackage for the footprint matching pipeline.

Importing this package triggers the registration of all built-in reader
subclasses via the ``__init_subclass__`` hook defined in
:class:`~libera_utils.footprint_matching.readers.base.AncillaryReader`.
After this import, :class:`~libera_utils.footprint_matching.readers.registry.ReaderRegistry`
lists all readers: ``CLDPIX``, ``ERA5_Pressure``, ``ERA5_Surface``, ``FLASHFlux``,
``IGBP``, ``NISE``, ``VIIRS_AOD``, ``VIIRS_BRDF``, ``VIIRS_CLDPROP_D3``, and
:data:`~libera_utils.footprint_matching.readers.registry.FMATCH_MODE_READERS` holds
the readers each product draws on, derived from their ``VariableSpec.modes``.

Public API
----------
ReaderRegistry : class
    Plugin registry -- use ``ReaderRegistry.get(name)`` to retrieve a reader class.
AncillaryReader : abstract class
    Base class for all reader plugins; subclass ``StaticReader`` (one body of data
    per period, served from memory), ``SteppedGridReader`` (a time series of global
    grids, served one step at a time) or ``PassReader`` (hourly swath granules, held
    by time window) to add a reader.

Notes
-----
Reader configuration is code, not a configuration file: each reader declares its key,
its source and its variables as class attributes and field tables, and the FMATCH
product definitions (``libera_utils/data/product_definitions/fmatch_*.yml``) are written
to match. Two tests in ``tests/unit/test_footprint_matching/test_fmatch_definitions.py``
hold the two sides together: ``test_external_variables_match_readers`` fails when a
product definition lacks a reader variable or has a different dtype, and
``test_reader_instrument_is_recorded_in_long_name`` fails when a ``long_name`` does not
end with the reader's ``INSTRUMENT``. Run them after any of the changes below.

**Swap the satellite or instrument a source comes from**, for example moving the VIIRS
readers from NOAA-20 to NOAA-21:

1. In each affected reader, set ``INSTRUMENT`` (``INSTRUMENT = "NOAA21"`` in ``aod.py``,
   ``brdf.py``, ``viirs.py``, ``cldpix.py`` and ``flashflux.py``), and update the module
   docstring's product and file-naming lines.
2. In every FMATCH product definition, change the instrument in parentheses at the end
   of each affected ``long_name``, for example ``(NOAA20)`` to ``(NOAA21)``.
3. If the new product's file layout differs (group paths, variable names, dimension
   order, fill or geolocation conventions), update the reader's field table and its
   ``_load_granule`` or ``_load_native``, and add a cut of a real granule under
   ``tests/test_data/footprint_matching/`` with a test that reads it.
4. Stage the new granules under the same directory as before: the directory is named
   by the reader key, not the instrument.

The reader key and the product variable names do not change, so downstream consumers
are unaffected unless step 3 changes a variable.

**Add a variable to an existing reader:**

1. Add the field to the reader's table: ``_CLDPIX_FIELDS`` in ``cldpix.py``,
   ``_FLASHFLUX_BASE_FIELDS`` or ``_IMAGER_FLASHFLUX_SOURCES`` in ``flashflux.py``,
   ``_D3_FIELDS`` in ``viirs.py``, or the reader's ``VARIABLES`` elsewhere. Give it its
   aggregation, the products that carry it in ``modes``, its valid range with the
   document it comes from, and for a mode-aggregated code ``n_categories`` and an
   ``int16`` dtype.
2. Add the product variable to each product definition whose mode carries it, named
   :func:`~libera_utils.io.product_variable_names.product_variable_name` of
   ``<reader key>_<variable name>`` (e.g. ``ERA5_Surface_Wind_U10``), with a
   ``long_name`` ending in ``(<INSTRUMENT>)`` and its units. A token the naming table
   does not spell yet, such as a new acronym, needs a ``token`` row in
   ``libera_utils/data/product_variable_names.csv``. A continuous variable also gets a
   ``..._Standard_Deviation`` companion, and an integer category declares
   ``_FillValue: -1``. A variable the reader stacks in ``PRODUCT_STACKS`` (ERA5's
   pressure levels) is published as its stack instead, named from
   ``<reader key>_<stack name>`` on the stack's dimension.
3. Extend the reader's synthetic fixture in ``tests/plugins/footprint_matching_fixtures.py``
   and its tests, and record the change under ``## [Unreleased]`` in ``CHANGELOG.md``.

**Add a reader or change which products a reader feeds:**

1. Write the reader as a subclass of ``StaticReader``, ``SteppedGridReader`` or
   ``PassReader`` with a unique ``READER_KEY``, and import it in this package's
   ``__init__.py`` to register it.
2. Name the products it feeds in each of its variables' ``modes``;
   ``FMATCH_MODE_READERS`` is derived from them.
3. Add its variables to those products' definitions, as above.
4. Stage its granules under ``$FMATCH_ANCILLARY_PATH/<reader key>/``; the runners derive
   the directories they read from the registry.

Renaming a reader key renames its product variables and its staging directory, so it is
a change consumers and the staging pipeline must follow.
"""

# Importing each reader class defines it, which triggers
# AncillaryReader.__init_subclass__, which calls ReaderRegistry._registry[key] = cls.
# Binding the classes here (not just the modules) keeps ``__all__`` importable so
# ``from ...readers import IGBPReader`` and ``import *`` work as advertised.
from libera_utils.footprint_matching.readers.aod import VIIRSAODReader
from libera_utils.footprint_matching.readers.base import AncillaryReader, PassReader, StaticReader, SteppedGridReader
from libera_utils.footprint_matching.readers.brdf import VIIRSBRDFReader
from libera_utils.footprint_matching.readers.cldpix import CLDPIXReader
from libera_utils.footprint_matching.readers.era5_pressure import ERA5PressureLevelReader
from libera_utils.footprint_matching.readers.era5_surface import ERA5SurfaceReader
from libera_utils.footprint_matching.readers.flashflux import FLASHFluxReader
from libera_utils.footprint_matching.readers.igbp import IGBPReader
from libera_utils.footprint_matching.readers.nsidc import NISEReader
from libera_utils.footprint_matching.readers.registry import FMATCH_MODE_READERS, ReaderRegistry, _derive_mode_readers
from libera_utils.footprint_matching.readers.viirs import VIIRSCldpropD3Reader

# Every built-in reader has registered by this line, and no reader defined elsewhere can
# have yet (defining one imports this package first), so the table holds exactly these.
FMATCH_MODE_READERS.update(_derive_mode_readers(ReaderRegistry._registry))

__all__ = [
    "AncillaryReader",
    "CLDPIXReader",
    "ERA5PressureLevelReader",
    "ERA5SurfaceReader",
    "FLASHFluxReader",
    "IGBPReader",
    "NISEReader",
    "PassReader",
    "ReaderRegistry",
    "StaticReader",
    "SteppedGridReader",
    "VIIRSAODReader",
    "VIIRSBRDFReader",
    "VIIRSCldpropD3Reader",
]
