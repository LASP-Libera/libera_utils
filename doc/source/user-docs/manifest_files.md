# Making and Using Manifest Files

All science algorithms that run on the Libera Science Data Center system need capabilities for dealing with Manifest
Files. Specifics on the usage of manifest files can be found in the [Manifest API documentation here](../api-doc/generated/libera_utils.io.manifest.rst)

The `Manifest` class is designed to handle reading, writing, and interacting with manifest files during
processing. It performs such tasks as validating manifest file structure and naming conventions as well as storing
the manifest contents as easily accessible python objects and providing helper methods for common tasks related
to manifest file handling.

Every path below may be a local path or an S3 path, and algorithms must handle both (an input manifest may sit in a
local dropbox directory bind-mounted into the running container).

## Reading an input manifest and writing an output manifest

```python
from libera_utils.io.manifest import Manifest

# Manifest filenames are passed into your Docker image CLI as its only argument
input_manifest = Manifest.from_file("s3://some-dropbox/LIBERA_INPUT_MANIFEST_01H2GK8J6XM93VKQP4CQFM1TAN.json")
# Read from manifest file to do processing
input_manifest.validate_checksums()

# Create an output manifest named according to the input manifest (the ULID matches for traceability)
output_manifest = Manifest.output_manifest_from_input_manifest(input_manifest)

# Add files. This will raise a credentials error because it tries to checksum the file but can't access S3
# without credentials provided (your Docker images will have proper credentials attached).
output_manifest.add_files(
    "s3://some-dropbox/LIBERA_L2_CF-CAM_V1-2-3_20270102T112233_20270102T122233_R27002112233.nc"
)

# Writes the output manifest, under its filename, into the directory specified.
# Usually this path is retrieved from the environment, like `os.environ["PROCESSING_PATH"]`.
output_manifest.write("s3://some-dropbox/")
```

## Writing an input manifest

```python
from libera_utils.constants import ManifestType
from libera_utils.io.manifest import Manifest

input_manifest = Manifest(manifest_type=ManifestType.INPUT)
input_manifest.add_files("/data/LIBERA_L1B_RAD-4CH_V1-2-3_20270102T112233_20270102T122233_R27002112233.nc")

# Generates a manifest filename with a new ULID and returns the path written
path = input_manifest.write("/dropbox/")
```

`write()` refuses to overwrite an existing file unless you pass `overwrite=True`.
