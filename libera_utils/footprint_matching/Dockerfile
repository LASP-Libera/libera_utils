## Combined Dockerfile for all five Libera FMATCH algorithm images.
##
## One shared base stage (system deps + venv + poetry install), then one runtime target
## and one test target per algorithm. Build a specific image by selecting its target:
##   docker build -f libera_utils/footprint_matching/Dockerfile --target fmatch-cam .
##
## The build context MUST be the repository root - the COPY instructions below use paths
## relative to it (COPY libera_utils ..., COPY pyproject.toml ...).
##
## Runtime targets:  fmatch-imager  fmatch-imager-camtime  fmatch-imager-flash
##                   fmatch-cam     fmatch-cam-camtime
## Test target:      fmatch-test (one shared suite covering the whole package).

ARG BUILDPLATFORM=linux/amd64

# fmatch-base
# -----------
# Shared build stage for every FMATCH image. This is NOT a shippable image - it has no
# ENTRYPOINT. Because the expensive layers (apt, venv, poetry install) live here, they are
# built once and reused by every per-algorithm target in a single build.
FROM --platform=${BUILDPLATFORM} python:3.11-slim AS fmatch-base
USER root

# Core package installation location. Later images that inherit from this one can reuse this.
ENV LIBERA_UTILS_DIRECTORY=/opt/libera
WORKDIR $LIBERA_UTILS_DIRECTORY

# Turn off interactive shell prompts to suppress configuration errors during apt installs.
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies. Note this list is LARGER than the SCENE-ID images': footprint matching
# links against native libraries that scene identification does not, because its ancillary readers
# open third-party science formats directly.
# - libpq-dev so we can install psycopg2 (transitive dependency)
# - curl so we can install poetry
# - gcc/pkg-config because they are often required for python package builds
# - libhdf5-dev + libnetcdf-dev for netCDF4/h5netcdf/h5py (L1B inputs, ERA5 and VIIRS readers, product writing)
# - libhdf4-dev for pyhdf, used by the IGBP and NISE readers (HDF4 / HDF-EOS4 granules). pyhdf has no
#   manylinux wheel, so this header package must be present BEFORE the poetry install below builds it.
#
# Deliberately NOT installed: libgdal-dev. Nothing in footprint_matching uses GDAL - the only
# geospatial dependency is pyproj, which ships a manylinux wheel with PROJ bundled. libgdal-dev also
# depends on libhdf4-alt-dev, which Conflicts with libhdf4-dev, so adding it breaks this install.
#
# A missing native library surfaces only at import time, so verify a rebuilt image with:
#   docker run --rm --entrypoint python <image> -c "import libera_utils.footprint_matching.readers"
RUN apt-get update && apt-get install -y \
    libpq-dev \
    curl \
    gcc \
    pkg-config \
    libhdf5-dev \
    libnetcdf-dev \
    libhdf4-dev

# Create a virtual environment and permanently activate it for this image.
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
# Adding the venv bin directory to PATH activates the venv and exposes installed entrypoints.
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# Upgrade pip because poetry uses pip under the hood to install packages.
RUN pip install --upgrade pip

# Install poetry and add it to PATH.
RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH="$PATH:/root/.local/bin"

# Copy the files needed to install libera_utils (respecting .dockerignore). The per-algorithm FMATCH
# runners live inside the libera_utils package (libera_utils/footprint_matching/<alg>/fmatch_<alg>.py)
# and are included by this copy.
COPY libera_utils $LIBERA_UTILS_DIRECTORY/libera_utils
COPY pyproject.toml $LIBERA_UTILS_DIRECTORY
COPY README.md $LIBERA_UTILS_DIRECTORY
COPY LICENSE.txt $LIBERA_UTILS_DIRECTORY

# Workaround for a known docker build bug where consecutive COPY layers can be cached incorrectly.
# https://github.com/moby/moby/issues/37965
RUN true

# Install libera_utils and its (non-dev) dependencies according to pyproject.toml.
#
# The 'fmatch' extra is REQUIRED here, unlike in the SCENE-ID images. It carries pyproj, pyhdf and
# h5py, and pyproj in particular is imported at module load time by footprint_matching.geometry and
# footprint_matching.camera_segmentation - which the runners import. Without the extra the container
# would fail on startup with ModuleNotFoundError rather than at first ancillary read.
RUN poetry install --only main --extras fmatch


# fmatch-test
# -----------
# Test image. Adds dev dependencies and the test suite on top of fmatch-base, and runs the
# (non-integration) unit tests with coverage. The suite exercises the whole libera_utils package and
# is identical regardless of algorithm, so a single test target covers all of FMATCH.
FROM fmatch-base AS fmatch-test

# Install dev dependencies (not installed in the runtime images).
RUN poetry install --extras fmatch

# Copy tests over.
COPY tests $LIBERA_UTILS_DIRECTORY/tests

# Run the (non-integration) unit tests with coverage.
ENTRYPOINT ["pytest", "-m", "not integration", "--cov=libera_utils", "--cov-report=xml:coverage.xml", "--junit-xml=junit.xml"]


# Per-algorithm runtime targets
# -----------------------------
# Each target differs from fmatch-base only in its ENTRYPOINT: the manifest-driven runner for that
# algorithm. Arguments passed to the container (i.e. the path to the input manifest) are forwarded to
# the script. Each runner reads two environment variables:
#   PROCESSING_PATH       - dropbox directory (or S3 prefix) products and the output manifest are written to.
#   FMATCH_ANCILLARY_PATH - root of the staged ancillary granule tree, one subdirectory per reader
#                           registry key (era5/, igbp/, nise/, ...). See
#                           libera_utils/footprint_matching/ancillary.py for the full layout.

FROM fmatch-base AS fmatch-imager
ENTRYPOINT ["python", "/opt/libera/libera_utils/footprint_matching/imager/fmatch_imager.py"]

FROM fmatch-base AS fmatch-imager-camtime
ENTRYPOINT ["python", "/opt/libera/libera_utils/footprint_matching/imager_camtime/fmatch_imager_camtime.py"]

FROM fmatch-base AS fmatch-imager-flash
ENTRYPOINT ["python", "/opt/libera/libera_utils/footprint_matching/imager_flash/fmatch_imager_flash.py"]

FROM fmatch-base AS fmatch-cam
ENTRYPOINT ["python", "/opt/libera/libera_utils/footprint_matching/cam/fmatch_cam.py"]

FROM fmatch-base AS fmatch-cam-camtime
ENTRYPOINT ["python", "/opt/libera/libera_utils/footprint_matching/cam_camtime/fmatch_cam_camtime.py"]
