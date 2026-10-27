"""VIIRS cloud property reader plugin for the footprint matching pipeline.

Data source: NOAA-20 (JPSS-1) VIIRS Daily Level-3 Cloud Properties
- Product: CLDPROP_D3_VIIRS_NOAA20 (Collection 011)
- Format: NetCDF4 with nested groups
- Spatial resolution: 1° × 1° (~111 km at equator)
- Grid: 360 × 180 regular lat/lon (global)
- Data layout: variables stored with dimension order (longitude, latitude),
  requiring a transpose to the expected (latitude, longitude) output order
- Temporal resolution: Daily composites
- Variables: cloud_optical_thickness (Mean), cloud_top_pressure (Mean)

NetCDF4 group layout (CLDPROP_D3)
----------------------------------
Root:
  latitude  (180,)   — 1° bin centers, −89.5 to 89.5
  longitude (360,)   — 1° bin centers, −179.5 to 179.5
  Cloud_Optical_Thickness_Combined/
    Mean      (360, 180) — daily mean cloud optical thickness
  Cloud_Top_Pressure/
    Mean      (360, 180) — daily mean cloud top pressure [hPa]

Fill value: each field's own ``_FillValue`` (−9999.0 in the distributed files), applied by
netCDF4's automatic masking and replaced with NaN before output.

References
----------
Product page:
    https://www.ncei.noaa.gov/products/climate-data-records/cloud-properties-viirs
ATBD:
    https://www.ncei.noaa.gov/pub/data/sds/cdr/CDRs/Cloud_Properties_VIIRS/
    AlgorithmDescriptionVIIRS_01B-20a.pdf
Data access:
    https://www.ncei.noaa.gov/data/cloud-properties-viirs/access/
File naming:
    CLDPROP_D3_VIIRS_NOAA20.A{YYYYDDD}.{collection}.{YYYYDDDHHMMSS}.nc
"""

import netCDF4
import numpy as np

from libera_utils.footprint_matching.readers._points import field_fill_value
from libera_utils.footprint_matching.readers.base import StaticReader
from libera_utils.footprint_matching.types import GridSamples, OperationalMode, TimeMatch, VariableSpec

# Each output variable, the (group, variable) it reads in the CLDPROP_D3 NetCDF4 file,
# and its aggregation. This order is VARIABLES order and axis 0 of the reader's data array.
_D3_FIELDS: tuple[tuple[str, str, str, str], ...] = (
    ("cloud_optical_thickness", "Cloud_Optical_Thickness_Combined", "Mean", "weighted_log_mean"),
    ("cloud_top_pressure", "Cloud_Top_Pressure", "Mean", "weighted_mean"),
)


class VIIRSCldpropD3Reader(StaticReader):
    """Read VIIRS daily cloud properties from a CLDPROP_D3 NetCDF4 file.

    Loads two cloud variables (cloud_optical_thickness, cloud_top_pressure)
    from a CLDPROP_D3 VIIRS Level-3 daily file as a ``(2, n_lat, n_lon)`` grid
    stacked in ``VARIABLES`` order. The file is a daily composite with no time
    axis, so the staged day is the time match (``TIME_MATCH = DAILY``).

    The CLDPROP_D3 product stores data with a non-standard ``(longitude,
    latitude)`` dimension order. This reader transposes each variable to
    ``(latitude, longitude)`` before returning.

    Class Attributes
    ----------------
    READER_KEY : str
        Registry key ``"viirs_cldprop_d3"``, after the product's short name.
    RESOLUTION_KM : float
        111 km (1° × 1° daily L3 grid resolution at equator).
    VARIABLES : tuple[VariableSpec, ...]
        Two variables: cloud_optical_thickness, cloud_top_pressure.

    Parameters
    ----------
    file_paths : Path or Sequence[Path]
        The one CLDPROP_D3 NetCDF4 file for the processing day.
    """

    READER_KEY: str = "viirs_cldprop_d3"
    # CLDPROP_D3_VIIRS_NOAA20 cloud properties are from VIIRS aboard NOAA-20 (JPSS-1).
    # TODO[LIBSDC-899]: decide whether NOAA-22 VIIRS replaces NOAA-20 here (a cutover of this reader) or gets a
    # second reader.
    INSTRUMENT: str = "NOAA20"
    RESOLUTION_KM: float = 111.0
    TIME_MATCH: TimeMatch = TimeMatch.DAILY
    VARIABLES: tuple[VariableSpec, ...] = tuple(
        VariableSpec(
            name=name,
            dtype="float32",
            aggregation=aggregation,
            required_mode=OperationalMode.CAM,
            n_categories=None,
        )
        for name, _, _, aggregation in _D3_FIELDS
    )

    def _load_native(self) -> GridSamples:
        """Read the full VIIRS cloud-property grid once.

        Opens the NetCDF4 file, navigates to each variable's group, transposes
        from (longitude, latitude) to (latitude, longitude) storage order, and
        replaces each field's own fill values with NaN.

        Returns
        -------
        GridSamples
            ``data`` is float32 shape ``(2, n_lat, n_lon)`` with axis 0 =
            [cloud_optical_thickness, cloud_top_pressure] (fill as NaN).

        Raises
        ------
        ValueError
            If a field has no ``_FillValue`` attribute.
        """
        with netCDF4.Dataset(str(self.single_file_path), "r") as ds:
            # Root-level coordinate arrays: (180,) and (360,) respectively
            lats_full = np.array(ds.variables["latitude"][:], dtype=np.float64)
            lons_full = np.array(ds.variables["longitude"][:], dtype=np.float64)

            variable_arrays: list[np.ndarray] = []
            for _, group_name, var_name, _ in _D3_FIELDS:
                variable = ds.groups[group_name].variables[var_name]
                # Require the field's own _FillValue, which netCDF4's automatic masking applies.
                field_fill_value(variable, source=f"{self.single_file_path} {group_name}")
                # Data shape is (n_lon, n_lat) in the file -- transpose to (n_lat, n_lon).
                variable_arrays.append(variable[:].astype(np.float32).filled(np.nan).T)

        data = np.stack(variable_arrays, axis=0)  # (2, n_lat_full, n_lon_full)
        return GridSamples(lat_axis=lats_full, lon_axis=lons_full, data=data)
