# Dockerfile that installs libera_utils and its dependencies

# libera-utils
# ----------
ARG BASE_IMAGE_PYTHON_VERSION=3.12

FROM public.ecr.aws/docker/library/python:${BASE_IMAGE_PYTHON_VERSION}-slim AS libera-utils
USER root

# Location for Core package installation location. This can be used later by images that inherit from this one
ENV LIBERA_UTILS_DIRECTORY=/opt/libera
WORKDIR $LIBERA_UTILS_DIRECTORY

# Turn off interactive shell to suppress configuration errors
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y curl gcc ca-certificates && \
    update-ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install spice utilities directly from NAIF (precompiled for Linux)
ENV CSPICE_DIR=/opt/naif
RUN curl -L -o /tmp/cspice.tar.Z https://naif.jpl.nasa.gov/pub/naif/toolkit//C/PC_Linux_GCC_64bit/packages/cspice.tar.Z && \
    mkdir -p $CSPICE_DIR && tar -C $CSPICE_DIR -xvzf /tmp/cspice.tar.Z cspice/exe && rm -r /tmp/cspice.tar.Z
ENV PATH="$PATH:$CSPICE_DIR/cspice/exe"

# Create virtual environment and permanently activate it for this image
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
# This adds not only the venv python executable but also all installed entrypoints to the PATH
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# Upgrade pip to the latest version because poetry uses pip in the background to install packages
RUN pip install --upgrade pip

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python -
# Add poetry to path
ENV PATH="$PATH:/root/.local/bin"

# Copy necessary files over (except for dockerignore-d files)
COPY libera_utils $LIBERA_UTILS_DIRECTORY/libera_utils
COPY README.md $LIBERA_UTILS_DIRECTORY
COPY pyproject.toml $LIBERA_UTILS_DIRECTORY
COPY LICENSE.txt $LIBERA_UTILS_DIRECTORY

# This is so stupid but it fixes known a bug in docker build
# https://github.com/moby/moby/issues/37965
RUN true

# Install libera_utils and all its (non-dev) dependencies according to pyproject.toml
RUN poetry lock && poetry sync --only main

# TODO[LIBSDC-600]: Temporary until Curryer is updated to auto-download.
# Let the curryer library know where the leapsecond file is stored.
ENV LEAPSECOND_FILE_ENV=$LIBERA_UTILS_DIRECTORY/libera_utils/data/spice

# Define the entrypoint of the container. Passing arguments when running the
# container will be passed as arguments to the function
ENTRYPOINT ["libera-utils"]


# libera-utils-test
# ---------------
FROM libera-utils AS libera-utils-test

# Install dev dependencies (not installed in libera-utils image)
RUN poetry sync --without docgen

# Copy tests over
COPY tests $LIBERA_UTILS_DIRECTORY/tests

# Set entrypoint
ENTRYPOINT ["pytest", "--cov=libera_utils", "--cov-report=xml:coverage.xml", "--junit-xml=junit.xml"]


# CLI for creating SPICE JPSS kernels from a manifest file.
# ---------------------------------------------------------
FROM libera-utils AS libera-utils-make-kernel-jpss

ENTRYPOINT ["libera-utils", "make-kernel", "jpss"]


# CLI for creating SPICE AzEl kernels from a manifest file.
# ---------------------------------------------------------
FROM libera-utils AS libera-utils-make-kernel-azel

ENTRYPOINT ["libera-utils", "make-kernel", "azel"]
