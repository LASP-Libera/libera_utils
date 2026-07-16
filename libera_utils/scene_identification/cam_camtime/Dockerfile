## Dockerfile that installs libera_utils and runs the Libera SCENE-ID-CAM-CAMTIME algorithm.

ARG BUILDPLATFORM=linux/amd64

# scene-id-cam-camtime
# --------------------
FROM --platform=${BUILDPLATFORM} python:3.11-slim AS scene-id-cam-camtime
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
# - libhdf5-dev because netCDF4/h5netcdf (used to read CERES SSF files and write NetCDF products) links against HDF5
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

# Copy the files needed to install libera_utils (respecting .dockerignore). The SCENE-ID-CAM-CAMTIME runner lives inside
# the libera_utils package (scene_identification/cam_camtime/scene_id_cam_camtime.py) and is included by this copy.
COPY libera_utils $LIBERA_UTILS_DIRECTORY/libera_utils
COPY pyproject.toml $LIBERA_UTILS_DIRECTORY
COPY README.md $LIBERA_UTILS_DIRECTORY
COPY LICENSE.txt $LIBERA_UTILS_DIRECTORY

# Workaround for a known docker build bug where consecutive COPY layers can be cached incorrectly.
# https://github.com/moby/moby/issues/37965
RUN true

# Install libera_utils and all its (non-dev) dependencies according to pyproject.toml.
RUN poetry install --only main

# The container entrypoint is the manifest-driven SCENE-ID-CAM-CAMTIME runner. Arguments passed to the container (i.e.
# the path to the input manifest) are forwarded to the script. The runner also reads the PROCESSING_PATH environment
# variable to determine where to write outputs.
ENTRYPOINT ["python", "/opt/libera/libera_utils/scene_identification/cam_camtime/scene_id_cam_camtime.py"]


# scene-id-cam-camtime-test
# -------------------------
FROM scene-id-cam-camtime AS scene-id-cam-camtime-test

# Install dev dependencies (not installed in the scene-id-cam-camtime image).
RUN poetry install

# Copy tests over.
COPY tests $LIBERA_UTILS_DIRECTORY/tests

# Run the (non-integration) unit tests with coverage.
ENTRYPOINT ["pytest", "-m", "not integration", "--cov=libera_utils", "--cov-report=xml:coverage.xml", "--junit-xml=junit.xml"]
