# Making and Using Manifest Files

All science algorithms that run on the Libera Science Data Center system need capabilities for dealing with Manifest
Files. Specifics on the usage of manifest files can be found in the [Manifest API documentation here](../api-doc/generated/libera_utils.io.manifest.rst)

The `Manifest` class is designed to handle reading, writing, and interacting with manifest files during
processing. It performs such tasks as validating manifest file structure and naming conventions as well as storing
the manifest contents as easily accessible python objects and providing helper methods for common tasks related
to manifest file handling.

## The typical algorithm workflow

```python
from libera_utils.io.manifest import Manifest

# Manifest filenames are passed into your Docker image CLI as its only argument
input_manifest = Manifest.from_file("s3://some-dropbox/LIBERA_INPUT_MANIFEST_01H2GK8J6XM93VKQP4CQFM1TAN.json")
# Read from manifest file to do processing
for record in input_manifest.files:
    print(record.filename, record.checksum)

# Create an output manifest named according to the input manifest. The ULID in the filename is preserved so the
# output can be traced back to its input; the input's file records are kept under
# output_manifest.configuration["input_manifest_files"].
output_manifest = Manifest.for_output_from_input(input_manifest)

# Add files. This will raise a credentials error because it tries to checksum the file but can't access S3
# without credentials provided (your Docker images will have proper credentials attached).
output_manifest.add_files(
    "s3://some-dropbox/LIBERA_L2_CF-CAM_V1-2-3_20270102T112233_20270102T122233_R27002112233.nc"
)

# Writes LIBERA_OUTPUT_MANIFEST_01H2GK8J6XM93VKQP4CQFM1TAN.json into the directory (or S3 prefix) given.
# Usually this path is retrieved from the environment, like `os.environ["PROCESSING_PATH"]`.
written_path = output_manifest.write("s3://some-dropbox/")
```

## Where the ULID lives

A manifest's ULID is carried by its filename, `LIBERA_<INPUT|OUTPUT>_MANIFEST_<ULID>.json`. `Manifest.ulid_code`
is derived from `Manifest.filename` and is `None` when no filename is set. You can still construct a manifest with
an explicit `ulid_code=`; without a `filename` this produces a bare filename carrying that ULID, and with one it
must agree with the filename's ULID.

## Reading is lenient, writing is strict

- `Manifest.from_file()` reads any JSON manifest. If the file's name is not a valid manifest filename a warning is
  logged and the manifest is loaded with `filename=None` and no ULID. Such a manifest cannot be saved or used with
  `for_output_from_input` until the file is renamed. The path on disk is the source of truth: a `filename` or
  `ulid_code` recorded inside the JSON is ignored.
- `Manifest.write()` and `Manifest.save()` refuse to produce a file whose name is not a valid manifest filename, or
  whose name says `INPUT` when the manifest is an `OUTPUT` (or vice versa), raising `ManifestError`.

## Writing

`write(out_path, filename=None)` has no side effects on the object, so one manifest can be written to several
locations. `out_path` may be a directory or S3 prefix, or the full path of the file to write:

```python
m = Manifest.for_input(files=["/data/LIBERA_L1A_..."], configuration={"start_time": "..."})

m.write("/processing/")                                                 # /processing/LIBERA_INPUT_MANIFEST_<ULID>.json
m.write("s3://bucket/prefix", filename=m.filename.path.name)             # explicit bare filename
m.write("/elsewhere/LIBERA_INPUT_MANIFEST_01H2GK8J6XM93VKQP4CQFM1TAN.json")  # full path
```

When no filename can be taken from the arguments or the object, a filename with a fresh ULID is generated.

## File-backed manifests: `save()` and `copy()`

A manifest returned by `from_file()` remembers where it came from (`source_path`, `is_file_backed`). Call `save()`
to write it back to that location (overwriting the file). A manifest built in code is not file-backed and `save()`
raises `ManifestError`; use `write()` instead. `copy()` returns a detached deep copy that can be modified and written
elsewhere without touching the original file.

```python
m = Manifest.from_file("/processing/LIBERA_INPUT_MANIFEST_01H2GK8J6XM93VKQP4CQFM1TAN.json")
m.configuration["comment"] = "reviewed"
m.save()

detached = m.copy()  # detached.is_file_backed is False
```
