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

`applicable_date` is the UTC date a file is archived under (`archive_prefix`). It is the date of the
midpoint of the time range, except for a calibration event file (a calibration ObsID's TRIMMED or
CAL product, which holds one event), which is dated by the day its `utc_start` falls on. See
[the ObsID Registry page](obsid_registry.md#calibration-event-files).

## Ancillary Product Filenames

External ancillary products (`DataLevel.ANC`) are ingested under the names their upstream providers give them, so
their classes parse those names and do not build them. ERA5 is the exception: the SDC names its CDS downloads.
`AbstractValidFilename.from_file_path` recognizes every form below.

| Class                             | Products                                         | Form                                                                                           | `version`                                       | Applicable date                           | `utc_start` / `utc_end`         |
| --------------------------------- | ------------------------------------------------ | ---------------------------------------------------------------------------------------------- | ----------------------------------------------- | ----------------------------------------- | ------------------------------- |
| `CeresFilename`                   | `CERES-NOAA20-SSF`, `-CLDPIX`                    | `CER_SSF_NOAA20-FM6-VIIRS_Edition1C_101103.2023010100.nc`                                      | Production strategy (`Edition1C`, `1P9test`)    | Date of the granule's start hour          | The hour                        |
| `ModisViirsDailyFilename`         | `VIIRS-NOAA20-AOD`, `-CLOUD-PROPERTIES`, `-BRDF` | `AERDB_D3_VIIRS_NOAA20.A2026270.021.2026274001051.nc`                                          | Collection (`021`)                              | The `A` date (BRDF: the composite's date) | The applicable day              |
| `ViirsSdrFilename`                | `VIIRS-NOAA20-M4-SDR`                            | `IVCDB_j01_d20241002_t1244587_e1246229_b67009_c20241004192414197049_oebc_ops.h5` (or `ICDBG_`) | Creation time (`c` field)                       | The `d` (start) date                      | The `t` and `e` times           |
| `NsidcNiseFilename`               | `NSIDC-ICE-SNOW`                                 | `NISE_AMSR2_20260610.HDFEOS`                                                                   | `1` (NISE_A2 collection version)                | The named date                            | The applicable day              |
| `AncillaryVersionArchiveFilename` | `MODIS-IGBP`, `NSIDC-LAND-OCEAN-MASK`            | `MODIS-IGBP_061_20250724_R20261020T120000000000.zip`                                           | IGBP: collection (`061`); mask: its version     | None: the product is version-partitioned  | None                            |
| `Era5Filename`                    | `ERA5-PRESSURE-LEVEL`, `ERA5-SINGLE-LEVEL`       | `ECMWF_ERA5-SINGLE-LEVEL_ERA5T_20250101T000000_20250101T230000_R20251028T120000000000.nc`      | Stream: `ERA5` (final) or `ERA5T` (preliminary) | Date of `utc_start`                       | The first and last hourly steps |

Every class gives `filename_parts.utc_start` and `utc_end`, the UTC period the file is used for. CERES gives its
hour, and VIIRS L3 and NISE their day, each ending at the start of the next hour or day (end exclusive); the VJ143C1
BRDF composite gives its nominal day, although it spans 16 days; a VIIRS SDR granule gives its `t` and `e` times; ERA5
gives its first and last hourly steps (end inclusive). They are `None` for a version archive and the files inside it,
whose names give no period. The CERES and VIIRS forms fix the NOAA-20 platform their products name: other platforms' files are
rejected.

A daily product is archived under `<ProductID>/YYYY/MM/DD` for its applicable date.

### Version archives

IGBP and the land/ocean mask are version-partitioned (`PartitionScheme.VERSION`): neither has a file per day, and
one version is many files. Each version is ingested as one zip archive holding every file of that version, named by
the SDC `<ProductID>_<version>_<production YYYYMMDD>_<revision>.zip`, and stored under
`<ProductID>/<version>/<production YYYYMMDD>`:

- `version` is, for IGBP, the three-digit MCD12Q1 collection (`061`); for the mask, letters and digits.
- `production YYYYMMDD` is the UTC date the provider produced the version's files. An IGBP archive holds one map
  year and is dated by its oldest tile's production date; a mask archive is dated by the date NSIDC produced that
  version.
- `revision` is the time the SDC built the archive, `R` followed by `YYYYMMDDTHHMMSSffffff`.

For example, `MODIS-IGBP_061_20250724_R20261020T120000000000.zip` is stored under `MODIS-IGBP/061/20250724`.

The files inside an archive keep their providers' names, in any directory of the zip. Their classes parse those
names so a reader can pick out the files it reads, but a loose tile or grid is not ingested on its own:
`from_file_path` rejects one with a message naming its version archive. `AncillaryVersionArchiveFilename.member_filename`
parses a file's name with its product's class, gives `None` for a file not in that form, such as a README or a browse
image, and raises `ValueError` for a name in that form whose fields do not parse.

| Class                        | Product                 | Form                                               | `version`                        |
| ---------------------------- | ----------------------- | -------------------------------------------------- | -------------------------------- |
| `ModisIgbpFilename`          | `MODIS-IGBP`            | `MCD12Q1.A2024001.h21v07.061.2025205222320.hdf`    | Collection (`061`)               |
| `NsidcLandOceanMaskFilename` | `NSIDC-LAND-OCEAN-MASK` | `EASE2_M36km.LOCImask_land50_coast0km.964x406.bin` | `1` (NSIDC-0609 dataset version) |

Only the global (`M`) EASE-Grid 2.0 mask grids are accepted, as `.bin`.

Each ancillary file also gives `partition_key`, `version` and `revision` (see Partition Schemes). Its `revision` is
the production time for the MODIS and VIIRS L3 products and an IGBP tile, the `R` field for ERA5 and a version
archive, and `None` for CERES, the VIIRS SDR, NISE and a mask grid.

The ERA5 name follows the Libera data product form: `utc_start` and `utc_end` are the first and last hourly steps in
the file (`YYYYMMDDTHHMMSS`), and `revision` is the time the SDC named it, `R` followed by `YYYYMMDDTHHMMSSffffff`.
`Era5Filename.from_filename_parts` builds one.

## Hashing and Equality

All `Filename` classes are hashable and compare equal when their paths are equal, so they can be used as dictionary
keys and set members. Two filenames with the same basename in different directories or buckets are distinct. A
filename's equality check only handles other filenames; for anything else it returns `NotImplemented` and Python
falls back to the other operand, so comparing a filename with a string, `None` or a path is `False`. Reassigning
`path` changes the object's hash, so do not modify a filename that is already a set member or a dictionary key.

## Partition Schemes

Every `DataProductIdentifier` has one `PartitionScheme`, its `partition_scheme`, which decides how its files are
named and archived and what partition each file belongs to. A partition holds one logical file; a newer revision of
that file, such as an L0 gap fill, is a new materialization of the same partition. Every data product filename class
gives three properties:

- `partition_key`: the file's partition, formatted by the scheme's `PartitionScheme.key_format`. It is the file's
  Dagster partition key, so libera_cdk's partitions definitions take their formats from `key_format`. Times are UTC.
- `version`: in PEP 440 form for a Libera product (`V1-2-3RC1` becomes `1.2.3rc1`), as the name carries it for an
  external one. `filename_parts.version` keeps the form in the name.
- `revision`: the UTC time this revision of the file was made, or `None` where the name carries none.

| Scheme     | Products                                                       | Filename class                                                                   | `partition_key` example |
| ---------- | -------------------------------------------------------------- | -------------------------------------------------------------------------------- | ----------------------- |
| `TWO_HOUR` | L0                                                             | `L0Filename`, `LiberaGroundCcsdsFilename`                                        | `2025-11-14-12:00`      |
| `HOURLY`   | `CERES-NOAA20-SSF`, `CERES-NOAA20-CLDPIX`                      | `CeresFilename`                                                                  | `2023-01-01-00:00`      |
| `DAILY`    | Every product with no other scheme                             | `LiberaDataProductFilename` and the daily ancillaries                            | `2027-01-02`            |
| `EVENT`    | The TRIMMED and CAL product of each calibration ObsID          | `LiberaDataProductFilename`                                                      | `20270102T193000`       |
| `VERSION`  | `ADM-CAM`, `ADM-IMAGER`, `MODIS-IGBP`, `NSIDC-LAND-OCEAN-MASK` | `LiberaVersionPartitionedDataProductFilename`, `AncillaryVersionArchiveFilename` | `1.2.3`, `061`          |

- **TWO_HOUR.** An EDOS PDS name carries no chunk start, so an `L0Filename`'s `partition_key` is `None`: the chunk
  start comes from its Construction Record. Its `revision` is the EDOS creation time in the name. A ground-test
  capture is named by the start of the two-hour bin it was cut from, an even UTC hour, and that bin start is its key;
  `LiberaGroundCcsdsFilename` rejects a name whose time is not a two-hour bin start. Every L0 product but the
  Construction Record also has a `longest_file_span`, the bound the L0 coverage index's window query looks back by.
- **HOURLY.** A CERES file holds one hour; its key is the start of that hour. It is still archived under the UTC day
  of that hour.
- **DAILY.** A daily data product, such as `RAD-4CH`, is named by `LiberaDataProductFilename` with its data level and
  the time range of its data, `LIBERA_<Level>_<ProductID>_V<Version>_<Start>_<End>_R<Revision>.<ext>`. It is
  archived under the date its time range falls on, `<ProductID>/<YYYY>/<MM>/<DD>`, which its `applicable_date`
  property gives and which is its key.
- **EVENT.** A TRIMMED or CAL file holds one calibration event and is named like a daily product. Its key is its
  `utc_start`, so two events of one ObsID on one day have different keys; it is archived under the UTC day of
  `utc_start`.
- **VERSION.** A version-partitioned product, such as an ADM, is valid over more than one day (indefinitely until the
  next version update). A Libera one is named by `LiberaVersionPartitionedDataProductFilename` with its data level
  and no time range, `LIBERA_<Level>_<ProductID>_V<Version>_R<Revision>.nc`, and archived once under
  `<ProductID>/<Version>`, with the version in the form the name carries. It has no `applicable_date`, and the period
  it covers is recorded in the file's variables and metadata. Its key is its PEP 440 version. IGBP and the land/ocean
  mask are ingested as version archives (`AncillaryVersionArchiveFilename`, see Version archives), each keyed by the
  version its name carries; a file inside an archive gives the same key.

`LiberaDataProductFilename` accepts only DAILY and EVENT products, and `LiberaVersionPartitionedDataProductFilename`
only VERSION products that are not ancillary; each rejects any other product where the name is made.
`filenaming.PARTITION_SCHEME_FILENAME_CLASSES` gives the filename class `write_libera_data_product` uses for each
scheme the SDC writes: DAILY, EVENT and VERSION.

```python
from libera_utils.io import filenaming

adm = filenaming.LiberaVersionPartitionedDataProductFilename.from_filename_parts(
    product_name="ADM-CAM", version="1.2.3"
)
print(adm.path.name)  # LIBERA_AUX_ADM-CAM_V1-2-3_R<revision>.nc
print(adm.archive_prefix)  # ADM-CAM/V1-2-3
print(adm.partition_key)  # 1.2.3

event = filenaming.LiberaDataProductFilename(
    "LIBERA_CAL_SWC-365NM_V1-0-0_20270102T193000_20270102T200000_R20270103T080910123456.nc"
)
print(event.partition_key)  # 20270102T193000
print(event.applicable_date)  # 2027-01-02
```

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
