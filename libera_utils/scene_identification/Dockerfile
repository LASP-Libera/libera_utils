## Combined Dockerfile for all four Libera SCENE-ID algorithm images.
##
## One shared base stage (system deps + venv + poetry install), then one runtime target
## and one test target per algorithm. Build a specific image by selecting its target:
##   docker build -f libera_utils/scene_identification/Dockerfile --target scene-id-cam .
##
## The build context MUST be the repository root - the COPY instructions below use paths
## relative to it (COPY libera_utils ..., COPY pyproject.toml ...).
##
## Runtime targets:  scene-id-imager  scene-id-imager-flash  scene-id-cam  scene-id-cam-camtime
## Test target:      scene-id-test (one shared suite covering the whole package).

ARG BUILDPLATFORM=linux/amd64

# scene-id-base
# -------------
# Shared build stage for every SCENE-ID image. This is NOT a shippable image - it has no
# ENTRYPOINT. Because the expensive layers (apt, venv, poetry install) live here, they are
# built once and reused by every per-algorithm target in a single build.
FROM --platform=${BUILDPLATFORM} python:3.11-slim AS scene-id-base
USER root

# Core package installation location. Later images that inherit from this one can reuse this.
ENV LIBERA_UTILS_DIRECTORY=/opt/libera
WORKDIR $LIBERA_UTILS_DIRECTORY

# Turn off interactive shell prompts to suppress configuration errors during apt installs.
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies:
# - libpq-dev so we can install psycopg2 (transitive dependency)
# - curl so we can install poetry
# - gcc/pkg-config because they are often required for python package builds
# - libhdf5-dev because netCDF4/h5netcdf (used to read FMATCH inputs and write NetCDF products) links against HDF5
RUN apt-get update && apt-get install -y libpq-dev curl gcc pkg-config libhdf5-dev

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

# Copy the files needed to install libera_utils (respecting .dockerignore). The per-algorithm SCENE-ID
# runners live inside the libera_utils package (scene_identification/<alg>/scene_id_<alg>.py) and are
# included by this copy.
COPY libera_utils $LIBERA_UTILS_DIRECTORY/libera_utils
COPY pyproject.toml $LIBERA_UTILS_DIRECTORY
COPY README.md $LIBERA_UTILS_DIRECTORY
COPY LICENSE.txt $LIBERA_UTILS_DIRECTORY

# Workaround for a known docker build bug where consecutive COPY layers can be cached incorrectly.
# https://github.com/moby/moby/issues/37965
RUN true

# Install libera_utils and all its (non-dev) dependencies according to pyproject.toml.
RUN poetry install --only main


# scene-id-test
# -------------
# Test image. Adds dev dependencies and the test suite on top of scene-id-base, and runs the
# (non-integration) unit tests with coverage. The suite exercises the whole libera_utils package and
# is identical regardless of algorithm, so a single test target covers all of SCENE-ID.
FROM scene-id-base AS scene-id-test

# Install dev dependencies (not installed in the runtime images).
RUN poetry install

# Copy tests over.
COPY tests $LIBERA_UTILS_DIRECTORY/tests

# Run the (non-integration) unit tests with coverage.
ENTRYPOINT ["pytest", "-m", "not integration", "--cov=libera_utils", "--cov-report=xml:coverage.xml", "--junit-xml=junit.xml"]


# Per-algorithm runtime targets
# -----------------------------
# Each target differs from scene-id-base only in its ENTRYPOINT: the manifest-driven runner for that
# algorithm. Arguments passed to the container (i.e. the path to the input manifest) are forwarded to
# the script. Each runner also reads the PROCESSING_PATH environment variable to determine where to
# write outputs.

FROM scene-id-base AS scene-id-imager
ENTRYPOINT ["python", "/opt/libera/libera_utils/scene_identification/scene_id_imager.py"]

FROM scene-id-base AS scene-id-imager-flash
ENTRYPOINT ["python", "/opt/libera/libera_utils/scene_identification/scene_id_imager_flash.py"]

FROM scene-id-base AS scene-id-cam
ENTRYPOINT ["python", "/opt/libera/libera_utils/scene_identification/scene_id_cam.py"]

FROM scene-id-base AS scene-id-cam-camtime
ENTRYPOINT ["python", "/opt/libera/libera_utils/scene_identification/scene_id_cam_camtime.py"]
