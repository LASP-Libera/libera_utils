"""VIIRS / merged aerosol optical depth (AOD) reader plugin.

Data source: Deep Blue GEO-LEO merged daily Level-3 aerosol optical depth
- Product: AERDB_D3_GEOLEO_Merged (Collection 001)
- Format: NetCDF4 with one group per contributing sensor
- Spatial resolution: 1° × 1° (~111 km at equator)
- Grid: 360 × 180 regular lat/lon (global), latitude ASCENDING (−89.5 → 89.5),
  longitude −179.5 → 179.5. Variables are stored in (Latitude, Longitude)
  dimension order, so — unlike the VIIRS CLDPROP_D3 cloud product — **no
  transpose is required**.
- Temporal resolution: Daily composites

Why the ``Merged`` group?
-------------------------
The file carries an AOD field for each individual sensor (NOAA20_VIIRS,
SNPP_VIIRS, Aqua_MODIS, Terra_MODIS, G16_ABI, G17_ABI, H08_AHI) plus a
``Merged`` group that fuses them into a single best-estimate field. Footprint
matching wants one consistent gridded AOD value per cell, so this reader reads
the ``Merged`` group. The per-sensor groups remain available for QA but are not
exposed as reader variables.

Roadmap note
------------
``data_products.md`` lists AOD as an external dependency of FMATCH-IMAGER
("AOD (NOAA-21 initially; NOAA-22 VIIRS when available)") and COMP-FLUX
("VIIRS (AOD, Aerosol Type)"). AOD does not appear in the CAM/NRT products, so
this reader is gated at ``OperationalMode.IMAGER``. Aerosol *type* is not
present in this merged AOD product (it is carried by the CERES SSF product), so
this reader exposes AOD only.

NetCDF4 layout (AERDB_D3_GEOLEO_Merged)
---------------------------------------
Root:
  Latitude  (180,)  — 1° bin centers, −89.5 → 89.5 (ascending)
  Longitude (360,)  — 1° bin centers, −179.5 → 179.5
  Merged/
    Aerosol_Optical_Thickness_550_Land_Ocean (180, 360)  — daily merged AOD at 550 nm

Fill value: −999.0 (replaced with NaN). valid_range: [0, 5].

References
----------
Deep Blue aerosol products:
    https://earthdata.nasa.gov/sensors/viirs  (Deep Blue / SOAR algorithms)
NASA Deep Blue:
    https://deepblue.gsfc.nasa.gov/
File naming:
    AERDB_D3_GEOLEO_Merged.A{YYYYDDD}.{collection}.{YYYYDDDHHMMSS}.nc
"""
from __future__ import annotations

import numpy as np

from libera_utils.footprint_matching.readers.base import GriddedDataReader
from libera_utils.footprint_matching.types import BoundingBox, OperationalMode, VariableSpec

# Group + variable path of the merged AOD field inside the NetCDF4 file.
_AOD_GROUP: str = "Merged"
_AOD_VARIABLE: str = "Aerosol_Optical_Thickness_550_Land_Ocean"

# Fill / missing value and physically valid range for the AOD field.
_AOD_FILL_VALUE: float = -999.0
_AOD_VALID_MAX: float = 5.0


class VIIRSAODReader(GriddedDataReader):
    """Read merged daily aerosol optical depth from an AERDB_D3_GEOLEO file.

    Loads a single variable (``aod_550``) from the ``Merged`` group of a Deep
    Blue GEO-LEO merged Level-3 daily file and returns a 2-D data array of shape
    ``(n_lat, n_lon)``.

    Class Attributes
    ----------------
    READER_KEY : str
        Registry key ``"viirs_aod"``.
    RESOLUTION_KM : float
        111 km (1° × 1° daily L3 grid resolution at equator).
    REQUIRED_MODE : OperationalMode
        ``IMAGER`` — AOD is a climate-quality (post-Year-1) dependency.
    VARIABLES : tuple[VariableSpec, ...]
        One variable: ``aod_550`` (continuous, ``weighted_log_mean``).

    Parameters
    ----------
    file_path : Path
        Path to an AERDB_D3_GEOLEO_Merged NetCDF4 file.
    """

    READER_KEY: str = "viirs_aod"
    RESOLUTION_KM: float = 111.0
    REQUIRED_MODE: OperationalMode = OperationalMode.IMAGER
    VARIABLES: tuple[VariableSpec, ...] = (
        VariableSpec(
            name="aod_550",
            dtype="float32",
            # AOD is approximately log-normally distributed, so a geometric
            # (log) mean is the appropriate spatial aggregation — matching how
            # cloud optical thickness is treated in VIIRSCloudReader.
            aggregation="weighted_log_mean",
            required_mode=OperationalMode.IMAGER,
            n_categories=None,
        ),
    )

    def _load_spatial_region(self, bbox: BoundingBox) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load merged AOD within ``bbox`` from an AERDB_D3_GEOLEO file.

        Opens the NetCDF4 file, reads the root coordinate arrays (ascending
        latitude, no transpose needed), subsets the ``Merged`` AOD field to the
        requested bounding box, and replaces fill / out-of-range values with NaN.

        Parameters
        ----------
        bbox : BoundingBox
            Geographic region to extract.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            ``(data, lats, lons)`` where ``data`` is float32 shape
            ``(n_lat, n_lon)``. Fill pixels (originally −999.0 or outside
            [0, 5]) are returned as NaN.
        """
        import netCDF4  # noqa: PLC0415

        with netCDF4.Dataset(str(self._file_path), "r") as ds:
            # Root-level coordinate arrays: (180,) and (360,) respectively.
            lats_full = np.array(ds.variables["Latitude"][:], dtype=np.float64)
            lons_full = np.array(ds.variables["Longitude"][:], dtype=np.float64)

            # Compute bbox index masks on the full coordinate arrays.
            lat_mask = (lats_full >= bbox.lat_min) & (lats_full <= bbox.lat_max)
            lon_mask = (lons_full >= bbox.lon_min) & (lons_full <= bbox.lon_max)

            lat_indices = np.where(lat_mask)[0]
            lon_indices = np.where(lon_mask)[0]

            if lat_indices.size == 0 or lon_indices.size == 0:
                return (
                    np.empty((0, 0), dtype=np.float32),
                    np.empty(0, dtype=np.float64),
                    np.empty(0, dtype=np.float64),
                )

            lats_out = lats_full[lat_indices]
            lons_out = lons_full[lon_indices]

            # Variable is stored (Latitude, Longitude) — no transpose needed.
            group = ds.groups[_AOD_GROUP]
            raw = np.array(group.variables[_AOD_VARIABLE][:], dtype=np.float32)
            sub = raw[np.ix_(lat_indices, lon_indices)]

            # Replace fill (−999.0) and out-of-range (>5 or <0) values with NaN.
            sub[sub <= _AOD_FILL_VALUE] = np.nan
            sub[(sub < 0.0) | (sub > _AOD_VALID_MAX)] = np.nan

        return sub, lats_out, lons_out
