# Making and Using Manifest Files

All science algorithms that run on the Libera Science Data Center system need capabilities for dealing with Manifest 
Files. Specifics on the usage of manifest files can be found in the [Manifest API documentation here](../api-doc/generated/libera_utils.io.manifest.rst)

The `Manifest` class is designed to handle reading, writing, and interacting with manifest files during 
processing. It performs such tasks as validating manifest file structure and naming conventions as well as storing 
the manifest contents as easily accessible python objects and providing helper methods for common tasks related 
to manifest file handling.

```python
from libera_utils.io.manifest import Manifest

my_manifest = Manifest.from_file("s3://some-dropbox/LIBERA_INPUT_MANIFEST_20270102T122233.json")
# Work with manifest file
```
