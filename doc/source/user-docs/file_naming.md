# File Naming

[See the filenaming API documentation here](../api-doc/generated/libera_utils.io.filenaming.rst)

The Libera Utils `Filename` classes allow reliable file naming, checking, and path management. Each type of
filename (ManifestFilename, ProductFilename, etc.) contains regex that validates every definition or update of the
internally tracked filename string. These classes transparently support both S3 paths and local filepaths, including
dynamic switching between the two.

```python
from pathlib import Path
from cloudpathlib import S3Path
from libera_utils.io import filenaming

p = filenaming.LiberaDataProductFilename('libera_cam_l2_20270102t112233_20270102t122233_vM1m2p3_r27002112233.h5')
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
p.path = '/bad/prefix' + p.path.name  # The missing / will make this fail regex validation
assert p.path.name == 'libera_cam_l2_20270102t112233_20270102t122233_vM1m2p3_r27002112233.h5'
```

