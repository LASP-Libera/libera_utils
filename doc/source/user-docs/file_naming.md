# File Naming

The Libera Utils `Filename` classes allow reliable file naming, checking, and path management to support conformity
with the Libera filenaming conventions. Each type of filename contains regex that validates every definition or update
of the internally tracked filename string. These classes transparently support both S3 paths and local filepaths,
including dynamic switching between the two, to simplify the transition between local development environments and AWS.

Full specifics including all available file naming classes are available [in the filenaming API documentation here](../api-doc/generated/libera_utils.io.filenaming.rst)

## Data Product Filename Convention

Libera data products (L1A, L1B, L2, CAL, and SPICE kernels) are named:

```text
LIBERA_{data_level}_{product_name}_{version}_{utc_start}_{utc_end}_{revision}.{extension}
LIBERA_L1B_RAD-4CH_V1-2-3_20270102T112233_20270102T122233_R20270103T080910123456.nc
```

| Part           | Example                  | Meaning                                                                          |
| -------------- | ------------------------ | -------------------------------------------------------------------------------- |
| `data_level`   | `L1B`                    | String value of `DataLevel`                                                      |
| `product_name` | `RAD-4CH`                | String value of `DataProductIdentifier`                                          |
| `version`      | `V1-2-3` or `V1-2-3RC1`  | Algorithm semantic version, `VM-m-p` with an optional release candidate suffix   |
| `utc_start`    | `20270102T112233`        | First timestamp of the data in the file                                          |
| `utc_end`      | `20270102T122233`        | Last timestamp of the data in the file                                           |
| `revision`     | `R20270103T080910123456` | UTC creation time of the file, `R` + `YYYYMMDDTHHMMSS` + microseconds (`ffffff`) |
| `extension`    | `nc`                     | `nc` for NetCDF-4 products, `bsp` for SPKs, `bc` for CKs                         |

Build filenames with `LiberaDataProductFilename.from_filename_parts` rather than hand-crafting strings. Only the
product, version, and time range are required; the revision defaults to the current UTC time at the call:

```python
from datetime import datetime, timezone
from libera_utils.io.filenaming import LiberaDataProductFilename

fn = LiberaDataProductFilename.from_filename_parts(
    product_name="RAD-4CH",
    version="1.2.3",  # semantic versions are converted to V1-2-3
    utc_start=datetime(2027, 1, 2, 11, 22, 33, tzinfo=timezone.utc),
    utc_end=datetime(2027, 1, 2, 12, 22, 33, tzinfo=timezone.utc),
)
parts = fn.filename_parts
assert parts.revision.tzinfo is not None  # creation time as a UTC datetime
assert fn.applicable_date.isoformat() == "2027-01-02"  # midpoint of the time range
assert fn.ummg_metadata_filename.name.endswith(".cmr.json")  # companion UMM-G metadata filename
```

## Ancillary Product Filenames

External ancillary products (`DataLevel.ANC`) are ingested under the names their upstream providers give them, so
their classes parse those names and do not build them. ERA5 is the exception: the SDC names its CDS downloads.
`AbstractValidFilename.from_file_path` recognizes every form below.

| Class                        | Products                                         | Form                                                                                           | `version`                                       | Applicable date                           |
| ---------------------------- | ------------------------------------------------ | ---------------------------------------------------------------------------------------------- | ----------------------------------------------- | ----------------------------------------- |
| `CeresFilename`              | `CERES-NOAA20-SSF`, `CERES-NOAA20-CLDPIX`        | `CER_SSF_NOAA20-FM6-VIIRS_Edition1C_101103.2023010100.nc`                                      | Production strategy (`Edition1C`, `alpha4`)     | Date of the granule's start hour          |
| `ViirsL3Filename`            | `VIIRS-NOAA20-AOD`, `-CLOUD-PROPERTIES`, `-BRDF` | `AERDB_D3_VIIRS_NOAA20.A2026270.021.2026274001051.nc`                                          | Collection (`021`)                              | The `A` date (BRDF: the composite's date) |
| `ViirsSdrFilename`           | `VIIRS-NOAA20-M4-SDR`                            | `IVCDB_j01_d20241002_t1244587_e1246229_b67009_c20241004192414197049_oebc_ops.h5` (or `ICDBG_`) | Creation time (`c` field)                       | The `d` (start) date                      |
| `NsidcNiseFilename`          | `NSIDC-ICE-SNOW`                                 | `NISE_AMSR2_20260610.HDFEOS`                                                                   | `1` (NISE_A2 collection version)                | The named date                            |
| `NsidcLandOceanMaskFilename` | `NSIDC-LAND-OCEAN-MASK`                          | `EASE2_M36km.LOCImask_land50_coast0km.964x406.bin`                                             | `1` (NSIDC-0609 dataset version)                | None: the mask is static                  |
| `Era5Filename`               | `ERA5-PRESSURE-LEVEL`, `ERA5-SINGLE-LEVEL`       | `ECMWF_ERA5-SINGLE-LEVEL_ERA5T_20250101T000000_20250101T230000_R20251028T120000000000.nc`      | Stream: `ERA5` (final) or `ERA5T` (preliminary) | Date of `utc_start`                       |

The CERES and VIIRS forms fix the NOAA-20 platform their products name: other platforms' files are rejected. A dated product is
archived under `<ProductID>/YYYY/MM/DD` for its applicable date; the static land/ocean mask under
`<ProductID>/<version>`. Only the global (`M`) EASE-Grid 2.0 mask grids are accepted, as `.bin`.

The ERA5 name follows the Libera data product form: `utc_start` and `utc_end` are the first and last hourly steps in
the file (`YYYYMMDDTHHMMSS`), and `revision` is the time the SDC named it, `R` followed by `YYYYMMDDTHHMMSSffffff`.
`Era5Filename.from_filename_parts` builds one.

## Hashing and Equality

All `Filename` classes are hashable and compare equal when their paths are equal, so they can be used as dictionary
keys and set members. Two filenames with the same basename in different directories or buckets are distinct. A
filename's equality check only handles other filenames; for anything else it returns `NotImplemented` and Python
falls back to the other operand, so comparing a filename with a string, `None` or a path is `False`. Reassigning
`path` changes the object's hash, so do not modify a filename that is already a set member or a dictionary key.

## Working With Paths

Below is an example test using a `LiberaDataProductFilename` instance to manage a filename string, including switching
between S3 and local paths to show the flexibility of the classes.

```python
from pathlib import Path
from cloudpathlib import S3Path
from libera_utils.io import filenaming

p = filenaming.LiberaDataProductFilename(
    'LIBERA_L2_CF-CAM_V1-2-3_20270102T112233_20270102T122233_R20270103T080910123456.nc')
# Add an S3 prefix
p.path = S3Path('s3://bucket') / p.path
assert isinstance(p.path, S3Path)
# Change prefix to local
p.path = Path('/tmp/path') / p.path.name
assert isinstance(p.path, Path)
# Remove basepath altogether
p.path = p.path.name
assert isinstance(p.path, Path)
# Check that providing a bad value for a basepath doesn't pollute the instance's valid path
try:
    p.path = '/bad/prefix' + p.path.name  # The missing / will make this fail regex validation
    raise Exception('The previous line should have raised a ValueError')
except ValueError as e:
    assert "failed validation against regex pattern" in str(e)
assert p.path.name == 'LIBERA_L2_CF-CAM_V1-2-3_20270102T112233_20270102T122233_R20270103T080910123456.nc'
```
