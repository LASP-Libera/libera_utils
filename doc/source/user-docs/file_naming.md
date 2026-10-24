# File Naming

The Libera Utils `Filename` classes allow reliable file naming, checking, and path management to support conformity
with the Libera filenaming conventions. Each type of filename contains regex that validates every definition or update
of the internally tracked filename string. These classes transparently support both S3 paths and local filepaths,
including dynamic switching between the two, to simplify the transition between local development environments and AWS.

Full specifics including all available file naming classes are available [in the filenaming API documentation here](../api-doc/generated/libera_utils.io.filenaming.rst)

## Hashing and Equality

All `Filename` classes are hashable and compare equal when their paths are equal, so they can be used as dictionary
keys and set members. Two filenames with the same basename in different directories or buckets are distinct. A
filename's equality check only handles other filenames; for anything else it returns `NotImplemented` and Python
falls back to the other operand, so comparing a filename with a string, `None` or a path is `False`. Reassigning
`path` changes the object's hash, so do not modify a filename that is already a set member or a dictionary key.

## Daily and Non-Daily Data Products

A daily data product, such as `RAD-4CH`, is named by `LiberaDataProductFilename` with its data level and the time
range of its data, `LIBERA_<Level>_<ProductID>_V<Version>_<Start>_<End>_R<Revision>.<ext>`. It is archived under the
date its time range falls on, `<ProductID>/<YYYY>/<MM>/<DD>`, which its `applicable_date` property gives.

A non-daily data product, such as an ADM, is valid over a period other than one day. It is named by
`LiberaDynamicDataProductFilename` with no data level and no time range, `LIBERA_<ProductID>_V<Version>_R<Revision>.nc`,
and archived once under `<ProductID>/<Version>`. It has no `applicable_date`; the period it covers is recorded in the
file's variables and attributes. A product is non-daily when its `DataProductIdentifier` has `is_dynamic` set, and
`LiberaDynamicDataProductFilename` rejects any other product.

```python
from libera_utils.io import filenaming

adm = filenaming.LiberaDynamicDataProductFilename.from_filename_parts(product_name="ADM-CAM", version="1.2.3")
print(adm.path.name)  # LIBERA_ADM-CAM_V1-2-3_R<revision>.nc
print(adm.archive_prefix)  # ADM-CAM/V1-2-3
```

## Working With Paths

Below is an example test using a `LiberaDataProductFilename` instance to manage a filename string, including switching
between S3 and local paths to show the flexibility of the classes.

```python
from pathlib import Path
from cloudpathlib import S3Path
from libera_utils.io import filenaming

p = filenaming.LiberaDataProductFilename(
    'LIBERA_L2_CF-CAM_V1-2-3_20270102T112233_20270102T122233_R27002112233.nc')
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
assert p.path.name == 'LIBERA_L2_CF-CAM_V1-2-3_20270102T112233_20270102T122233_R27002112233.nc'
```
