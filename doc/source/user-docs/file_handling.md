# File Handling

[See the smart_open API documentation here](../api-doc/generated/libera_utils.io.smart_open.rst)

The libera-utils smart_open function has the capability to read and write files to/from a local directory or S3 bucket
transparently. It supports a context manager pattern and the usual modes for reading/writing/binary provided by most
Python filelike objects:

```python
from libera_utils.io.smart_open import smart_open
from libera_utils.io.filenaming import LiberaDataProductFilename
f = LiberaDataProductFilename("s3://some-bucket/LIBERA_L1B_RAD_V1-2-3_20250102T120000_20250103T120000_R25005112233.nc")
with smart_open(f.path, "r") as filehandler:
    # Work with file contents
    pass
```

