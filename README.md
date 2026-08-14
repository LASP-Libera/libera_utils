# Libera Science Data Processing Utilities

Libera Utils is a package containing modules that are commonly used throughout the Libera Science Data Center codebase
and processing algorithms. This package is published on PyPI to support our L2 algorithm developers with standardized
code for interacting with our AWS resources and a consistent API for common tasks required of all developers.

## Documentation

Documentation site, including full API listing:
[https://lasp-libera-sdc-libera-utils.readthedocs-hosted.com](https://lasp-libera-sdc-libera-utils.readthedocs-hosted.com)

Additional documentation helpful for Level 2 Algorithm Developers is also available in the Libera SDC Developer Guide.
Please contact the Libera SDC Team at LASP for access to the Developer Guide.

## Installation

```bash
pip install libera-utils
```

To use the footprint-matching gridded ancillary-data readers
(`libera_utils.footprint_matching.readers`), install the optional `fmatch` extra, which pulls
in the geospatial / HDF stack (`pyproj`, `pyhdf`, `h5py`):

```bash
pip install 'libera_utils[fmatch]'
```

Note: `pyhdf` needs the HDF4 C library at build time when no prebuilt wheel is available for your
interpreter. On Debian/Ubuntu: `sudo apt-get install libhdf4-dev`; with conda:
`conda install -c conda-forge pyhdf`.

Other suffixed versions such as release candidate versions (version strings suffixed with `rc` followed by the candidate
number, e.g. `1.2.3rc2`) may also be available but are likely to contain bugs.
