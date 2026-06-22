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

Tile geometry
-------------
The global tile grid uses ``TILE_SIZE_DEG = 2.0`` degrees per tile edge. A
``TileKey(source, lat_idx, lon_idx)`` maps to::

    lat_min = -90.0 + lat_idx * TILE_SIZE_DEG
    lat_max = lat_min + TILE_SIZE_DEG
    lon_min = -180.0 + lon_idx * TILE_SIZE_DEG
    lon_max = lon_min + TILE_SIZE_DEG

This gives a 90 × 180 grid of 2° × 2° tiles covering the entire globe.
"""

from __future__ import annotations

import abc
from pathlib import Path

from libera_utils.footprint_matching.types import BoundingBox, GridTile, OperationalMode, TileKey, VariableSpec

# Global tile size constant.  2° matches CERES heritage design (~200 km tiles
# at the equator). All readers assume this tile size when slicing their data.
TILE_SIZE_DEG: float = 2.0


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
        Instrument / platform token embedded in every output variable name, e.g.
        ``"NOAA20"``, ``"MODIS"``, ``"SSMIS"``, ``"ECMWF"``. Combined with
        ``READER_KEY`` and each ``VariableSpec`` name to form the product variable
        name ``f"{READER_KEY}_{INSTRUMENT}_{spec.name}"`` (e.g.
        ``igbp_MODIS_surface_type``). For model/reanalysis sources that have no
        instrument (ERA5) the producing center is used so the naming stays uniform.
    RESOLUTION_KM : float
        Native spatial resolution of the data source in km.
    REQUIRED_MODE : OperationalMode
        Minimum operational mode for which this reader is active. The registry
        uses ``mode.rank`` ordering to exclude higher-latency readers when the
        pipeline is running in a lower-latency mode.
    VARIABLES : tuple[VariableSpec, ...]
        Ordered tuple of variable specifications this reader produces.  For
        multi-variable readers the first axis of the returned data array
        corresponds to this tuple's ordering.

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
    REQUIRED_MODE: OperationalMode
    VARIABLES: tuple[VariableSpec, ...]

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

        # Skip abstract classes — they have abstract methods still declared and
        # cannot be instantiated, so registering them would cause confusing errors.
        if abc.ABC in cls.__bases__:
            return
        if getattr(cls, "__abstractmethods__", None):
            return
        # Check that the subclass has defined all required class attributes.
        # We check for READER_KEY presence as the canary; if it is missing, the
        # class is likely a partial/abstract intermediate class and we skip it.
        if not hasattr(cls, "READER_KEY") or cls.READER_KEY is None:
            return

        # INSTRUMENT is part of every output variable name, so a concrete reader
        # that forgot to declare it would silently produce malformed names. Fail
        # fast at import/registration time rather than at product-write time.
        if not hasattr(cls, "INSTRUMENT") or cls.INSTRUMENT is None:
            raise TypeError(
                f"Reader {cls.__name__!r} (READER_KEY={cls.READER_KEY!r}) must define an "
                f"INSTRUMENT class attribute; it is embedded in output variable names "
                f"as f'{{READER_KEY}}_{{INSTRUMENT}}_{{spec.name}}'."
            )

        # Local import to break the circular dependency: base → registry → base.
        from libera_utils.footprint_matching.readers.registry import ReaderRegistry  # noqa: PLC0415

        ReaderRegistry._registry[cls.READER_KEY] = cls

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
        bbox = self._tile_key_to_bbox(key)
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
    def _tile_key_to_bbox(key: TileKey) -> BoundingBox:
        """Convert a TileKey's integer indices to a geographic BoundingBox.

        Parameters
        ----------
        key : TileKey
            Tile key with ``lat_idx`` and ``lon_idx`` integers.

        Returns
        -------
        BoundingBox
            Bounding box in degrees covering the 2° × 2° tile.
        """
        lat_min = -90.0 + key.lat_idx * TILE_SIZE_DEG
        lat_max = lat_min + TILE_SIZE_DEG
        lon_min = -180.0 + key.lon_idx * TILE_SIZE_DEG
        lon_max = lon_min + TILE_SIZE_DEG

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
        Implementors are responsible for handling the fill / missing-value sentinel
        appropriate to their data source. The caller (``load_tile``) does not
        perform any fill-value processing after calling this hook.
        """
