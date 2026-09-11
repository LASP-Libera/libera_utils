"""Abstract base class for all gridded data reader plugins.

Readers follow the Template Method pattern: subclasses implement
``_load_spatial_region(bbox)`` and the base class provides the concrete
``load_tile(key)`` method that converts a TileKey to a BoundingBox and
calls the subclass hook.

Auto-registration via ``__init_subclass__``
-------------------------------------------
Every class that subclasses ``GriddedDataReader`` is automatically added to
``ReaderRegistry._registry`` using the subclass's ``READER_KEY`` class attribute
as the key. This means a reader module just needs to be *imported* — no manual
registration call is needed. The readers package ``__init__.py`` imports all
built-in reader modules to make this happen at startup.

Tile geometry (per-reader tile size)
------------------------------------
Each reader declares its own square tile edge length via the class attribute
``TILE_SIZE_DEG`` (default :data:`DEFAULT_TILE_SIZE_DEG` = 2.0°, the CERES-heritage
~200 km-at-nadir tile). Making the size per-reader lets a fine-resolution source
(e.g. 750 m VIIRS) choose a *smaller* tile so a single cached tile stays memory
bounded, while a coarse source (e.g. 25 km NSIDC) can keep the 2° default —
resolving the "per-reader tile dimensions" decision flagged in the design doc.

For a reader with edge length ``s`` degrees, a ``TileKey(source, lat_idx, lon_idx)``
maps to::

    lat_min = -90.0 + lat_idx * s
    lat_max = lat_min + s
    lon_min = -180.0 + lon_idx * s
    lon_max = lon_min + s

The index origin stays anchored at (−90°, −180°) *regardless of tile size*, so a
tile key is globally interpretable **within a given source**. Keys from different
sources are never mixed because :class:`~libera_utils.footprint_matching.types.TileKey`
carries the ``source`` string, so two sources may safely use different tile sizes.
"""

from __future__ import annotations

import abc
from pathlib import Path

from libera_utils.footprint_matching.types import (
    BoundingBox,
    GridTile,
    TileKey,
    VariableSpec,
    with_standard_deviation_companions,
)

# Default tile edge length in degrees. 2° matches the CERES heritage design
# (~200 km tiles at the equator, four times the largest nadir footprint). A reader
# uses this size unless it overrides the ``TILE_SIZE_DEG`` *class attribute*.
DEFAULT_TILE_SIZE_DEG: float = 2.0

# Backward-compatible module-level alias. Historically the tile size was a single
# global constant; callers and tests still import ``TILE_SIZE_DEG`` from this
# module. It equals the default and is the size used by every reader that does not
# override the class attribute below.
TILE_SIZE_DEG: float = DEFAULT_TILE_SIZE_DEG


class GriddedDataReader(abc.ABC):
    """Abstract base class for all gridded ancillary data readers.

    Each concrete subclass reads one specific data source (e.g., IGBP land
    cover, NSIDC sea ice, ERA5 wind) and serves rectangular 2° × 2° spatial
    tiles to the TileManager on demand.

    Class-Level Attributes (required on every subclass)
    ----------------------------------------------------
    READER_KEY : str
        Unique string key used to register the reader in ``ReaderRegistry``.
        Examples: ``"igbp"``, ``"nsidc"``, ``"era5"``, ``"viirs_l2l3"``.
    INSTRUMENT : str
        Instrument / platform token recorded in each output variable's ``long_name``
        for provenance, e.g. ``"NOAA20"``, ``"MODIS"``, ``"SSMIS"``, ``"ECMWF"``. The
        variable *name* is ``f"{READER_KEY}_{spec.name}"`` (e.g. ``igbp_surface_type``);
        the instrument is appended to the product-definition ``long_name`` as
        ``"... ({INSTRUMENT})"`` rather than embedded in the name. For model/reanalysis
        sources that have no instrument (ERA5) the producing center is used so the
        provenance tag stays uniform.
    RESOLUTION_KM : float
        Native spatial resolution of the data source in km.
    VARIABLES : tuple[VariableSpec, ...]
        Ordered tuple of variable specifications this reader *reads from its
        source file*.  For multi-variable readers the first axis of the returned
        data array corresponds to this tuple's ordering. This is the runtime
        read/aggregation contract; it deliberately does NOT include derived
        outputs (see ``product_variable_specs``).
    ADDITIONAL_PRODUCT_VARIABLES : tuple[VariableSpec, ...]
        Extra variables this reader contributes to the *product definition* that
        are not read directly from the source file but are derived during PSF
        aggregation (e.g. IGBP's second/third most-common scene). Empty for most
        readers. Standard-deviation companions are added automatically and need
        not be listed here — see ``product_variable_specs``.

    Parameters
    ----------
    file_path : Path
        Absolute path to the ancillary data file on disk. The caller
        (TileManager or test harness) is responsible for providing a valid
        path; the reader does not resolve S3 or cloud paths.

    Notes
    -----
    *Subclasses must not override ``__init__``* without calling ``super().__init__``
    — the base ``__init__`` stores the file path.
    """

    # --- Required class-level attributes (declared here for static analysis) ---
    READER_KEY: str
    INSTRUMENT: str
    RESOLUTION_KM: float
    VARIABLES: tuple[VariableSpec, ...]
    # Square tile edge length in degrees for *this reader's* tile grid. Defaults to
    # the CERES-heritage 2° tile; a reader overrides it to trade cache-tile memory
    # against merge overhead for its native resolution (see the module docstring's
    # "Tile geometry" section). The TileManager reads this attribute to resolve a
    # bounding box into this reader's tile keys.
    TILE_SIZE_DEG: float = DEFAULT_TILE_SIZE_DEG
    # Derived product outputs not read straight from the source file. Defaults to
    # empty; readers like IGBP override it to add ranked-scene outputs.
    ADDITIONAL_PRODUCT_VARIABLES: tuple[VariableSpec, ...] = ()

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Auto-register every concrete subclass in ReaderRegistry.

        Called automatically by Python when a class inherits from
        GriddedDataReader. Abstract intermediate classes (those that still have
        ``abc.abstractmethod`` members) are skipped because we only want to
        register classes that can actually be instantiated.

        The local import of ReaderRegistry avoids the circular import that would
        arise if registry.py imported base.py at module load time.
        """
        super().__init_subclass__(**kwargs)

        # Skip abstract intermediates — they still have unimplemented abstract
        # methods and cannot be instantiated, so registering them would cause
        # confusing errors. ``__abstractmethods__`` is the reliable signal here;
        # a concrete reader that has implemented every hook registers normally
        # even if it happens to list ``abc.ABC`` among its bases.
        if getattr(cls, "__abstractmethods__", None):
            return
        # Check that the subclass has defined all required class attributes.
        # We check for READER_KEY presence as the canary; if it is missing, the
        # class is likely a partial/abstract intermediate class and we skip it.
        if not hasattr(cls, "READER_KEY") or cls.READER_KEY is None:
            return

        # INSTRUMENT is the provenance tag recorded in every output variable's
        # long_name, so a concrete reader that forgot to declare it would silently
        # drop provenance. Fail fast at import/registration time rather than later.
        if not hasattr(cls, "INSTRUMENT") or cls.INSTRUMENT is None:
            raise TypeError(
                f"Reader {cls.__name__!r} (READER_KEY={cls.READER_KEY!r}) must define an "
                f"INSTRUMENT class attribute; it is recorded in each output variable's "
                f"long_name as '... ({{INSTRUMENT}})'."
            )

        # Local import to break the circular dependency: base → registry → base.
        from libera_utils.footprint_matching.readers.registry import ReaderRegistry  # noqa: PLC0415

        ReaderRegistry._registry[cls.READER_KEY] = cls

    @classmethod
    def product_variable_specs(cls) -> tuple[VariableSpec, ...]:
        """Return every variable this reader contributes to the product definition.

        This is the full set of *output* variables — what appears in the FMATCH
        product definition YAMLs and what the product/test cross-check is built
        against. It is distinct from ``VARIABLES`` (the smaller set actually read
        from the source file) and is composed of, in order:

        1. each read variable in ``VARIABLES``;
        2. a ``<name>_standard_deviation`` companion for every *continuous*
           (mean-aggregated) read variable, inserted right after its parent by
           :func:`~libera_utils.footprint_matching.types.with_standard_deviation_companions`;
        3. any reader-specific derived outputs declared in
           ``ADDITIONAL_PRODUCT_VARIABLES`` (e.g. IGBP ranked scenes).

        Keeping the standard-deviation expansion in one place guarantees the
        readers, the product definition YAMLs, and the cross-check test all agree
        on the exact derived-variable names without hand-duplicating ~30 specs.
        """
        return with_standard_deviation_companions(cls.VARIABLES) + cls.ADDITIONAL_PRODUCT_VARIABLES

    def __init__(self, file_path: Path) -> None:
        """Store the path to the ancillary data file.

        Parameters
        ----------
        file_path : Path
            Path to the ancillary data file. Must exist on the local filesystem
            when ``load_tile()`` is called.
        """
        self._file_path = Path(file_path)

    @property
    def file_path(self) -> Path:
        """Path to the ancillary data file passed at construction time."""
        return self._file_path

    def load_tile(self, key: TileKey) -> GridTile:
        """Load and return the data tile identified by ``key``.

        This is the *template method*: it converts the TileKey to a BoundingBox
        and delegates to the abstract ``_load_spatial_region(bbox)`` hook that
        each subclass implements.

        Parameters
        ----------
        key : TileKey
            Tile cache key. ``key.lat_idx`` and ``key.lon_idx`` locate the tile
            in the 2° global grid.

        Returns
        -------
        GridTile
            Rectangular region of data with coordinate arrays and metadata.
        """
        # Use this reader's own tile size so per-reader grids resolve correctly.
        bbox = self._tile_key_to_bbox(key, self.TILE_SIZE_DEG)
        data, lats, lons = self._load_spatial_region(bbox)
        return GridTile(
            data=data,
            lats=lats,
            lons=lons,
            bounds=bbox,
            source=self.READER_KEY,
            timestamp_source=None,
        )

    @staticmethod
    def _tile_key_to_bbox(key: TileKey, tile_size_deg: float = DEFAULT_TILE_SIZE_DEG) -> BoundingBox:
        """Convert a TileKey's integer indices to a geographic BoundingBox.

        Kept a ``staticmethod`` (rather than an instance method) so it can be called
        without a reader instance — e.g. by the TileManager and by tests — while the
        tile size is supplied explicitly. ``load_tile`` passes ``self.TILE_SIZE_DEG``
        so a reader's own grid is used; callers that omit ``tile_size_deg`` get the
        2° default grid.

        Parameters
        ----------
        key : TileKey
            Tile key with ``lat_idx`` and ``lon_idx`` integers.
        tile_size_deg : float, optional
            Square tile edge length in degrees. Defaults to
            :data:`DEFAULT_TILE_SIZE_DEG` (2.0°).

        Returns
        -------
        BoundingBox
            Bounding box in degrees covering the ``tile_size_deg`` × ``tile_size_deg``
            tile identified by ``key``.
        """
        lat_min = -90.0 + key.lat_idx * tile_size_deg
        lat_max = lat_min + tile_size_deg
        lon_min = -180.0 + key.lon_idx * tile_size_deg
        lon_max = lon_min + tile_size_deg

        # Detect dateline wrapping: only possible at the extreme eastern tile
        # (lon_idx = 179 → lon_max = 180°, the exact boundary, so no wrapping).
        # Actual wrapping would only occur if a BoundingBox is constructed from
        # a footprint boresight; tile keys never wrap the dateline.
        wraps_dateline = lon_max > 180.0
        is_polar = abs(lat_max) >= 85.0 or abs(lat_min) >= 85.0

        return BoundingBox(lat_min, lat_max, lon_min, lon_max, wraps_dateline, is_polar)

    @abc.abstractmethod
    def _load_spatial_region(self, bbox: BoundingBox) -> tuple:
        """Load data for the requested geographic region.

        Subclasses must implement this method to open their data file, slice out
        the pixels within ``bbox``, and return them as a (data, lats, lons) tuple.

        Parameters
        ----------
        bbox : BoundingBox
            Geographic bounds of the region to load.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(data, lats, lons)`` where:

            - ``data`` is shape ``(n_lat, n_lon)`` for single-variable readers or
              ``(n_var, n_lat, n_lon)`` for multi-variable readers.
            - ``lats`` is a 1-D array of latitudes in degrees for the row axis.
            - ``lons`` is a 1-D array of longitudes in degrees for the column axis.

        Notes
        -----
        Implementers are responsible for handling the fill / missing-value sentinel
        appropriate to their data source. The caller (``load_tile``) does not
        perform any fill-value processing after calling this hook.
        """
