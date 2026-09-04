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
LIBERA_L1B_RAD-4CH_V1-2-3_20270102T112233_20270102T122233_01J8ZQ3K9X7M2N4P6Q8R0S1T2V.nc
```

| Part           | Example                      | Meaning                                                                                                                               |
| -------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `data_level`   | `L1B`                        | String value of `DataLevel`                                                                                                           |
| `product_name` | `RAD-4CH`                    | String value of `DataProductIdentifier`                                                                                               |
| `version`      | `V1-2-3` or `V1-2-3RC1`      | Algorithm semantic version, `VM-m-p` with an optional release candidate suffix                                                        |
| `utc_start`    | `20270102T112233`            | First timestamp of the data in the file                                                                                               |
| `utc_end`      | `20270102T122233`            | Last timestamp of the data in the file                                                                                                |
| `revision`     | `01J8ZQ3K9X7M2N4P6Q8R0S1T2V` | A [ULID](https://github.com/ulid/spec) uniquely identifying this production of the file. Its embedded timestamp is the creation time. |
| `extension`    | `nc`                         | `nc` or `h5` for NetCDF/HDF5, `bsp` for SPKs, `bc` for CKs                                                                            |

Build filenames with `LiberaDataProductFilename.from_filename_parts` rather than hand-crafting strings. Only the
product, version, and time range are required; the revision is generated automatically:

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
assert parts.revision.datetime  # a ULID; carries the creation time
assert fn.applicable_date.isoformat() == "2027-01-02"  # midpoint of the time range
assert fn.ummg_metadata_filename.name.endswith(".cmr.json")  # companion UMM-G metadata filename
```

## Hashing and Equality

All `Filename` classes are hashable and compare equal when their paths are equal, so they can be used as dictionary
keys and set members. Two filenames with the same basename in different directories or buckets are distinct, and
comparing a filename with anything that is not a filename is simply `False`. Reassigning `path` changes the object's
hash, so do not modify a filename that is already a set member or a dictionary key.

## Working With Paths

Below is an example test using a `LiberaDataProductFilename` instance to manage a filename string, including switching
between S3 and local paths to show the flexibility of the classes.

```python
from pathlib import Path
from cloudpathlib import S3Path
from libera_utils.io import filenaming

p = filenaming.LiberaDataProductFilename(
    'LIBERA_L2_CF-CAM_V1-2-3_20270102T112233_20270102T122233_01J8ZQ3K9X7M2N4P6Q8R0S1T2V.nc')
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
assert p.path.name == 'LIBERA_L2_CF-CAM_V1-2-3_20270102T112233_20270102T122233_01J8ZQ3K9X7M2N4P6Q8R0S1T2V.nc'
```
