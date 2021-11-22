# Dockerfile that installs libera_sdp and its dependencies

# libera-sdp
# ----------
FROM python:3.9.0-slim AS libera-sdp
USER root

# Location for Core package installation location. This can be used later by images that inherit from this one
ENV LIBSDP_INSTALL_LOCATION=/opt/libera
WORKDIR $LIBSDP_INSTALL_LOCATION

# Turn of interactive shell to suppress configuration errors
ARG DEBIAN_FRONTEND=noninteractive

# Install spice utilities directly from NAIF (precompiled for Linux)
ADD https://naif.jpl.nasa.gov/pub/naif/toolkit//C/PC_Linux_GCC_64bit/packages/cspice.tar.Z /tmp/cspice.tar.Z
ENV CSPICE_DIR=/opt/naif
RUN mkdir -p $CSPICE_DIR && tar -C $CSPICE_DIR -xvzf /tmp/cspice.tar.Z cspice/exe && rm -r /tmp/cspice.tar.Z
ENV PATH="$PATH:$CSPICE_DIR/cspice/exe"

# Create virtual environment and permanently activate it for this image
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
# This adds not only the venv python executable but also all installed entrypoints to the PATH
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
# Upgrade pip to the latest version because poetry uses pip in the background to install packages
RUN pip install --upgrade pip

# Install curl so we can install python poetry
RUN apt-get update && apt-get install -y curl
# Install poetry
RUN curl -sSL https://install.python-poetry.org | python -
# Add poetry to path
ENV PATH="$PATH:/root/.local/bin"

# Copy necessary files over (except for dockerignore-d files)
COPY libera_sdp $LIBSDP_INSTALL_LOCATION/libera_sdp
COPY README.md $LIBSDP_INSTALL_LOCATION
COPY doc $LIBSDP_INSTALL_LOCATION/doc
COPY pyproject.toml $LIBSDP_INSTALL_LOCATION
COPY LICENSE $LIBSDP_INSTALL_LOCATION

# This is so stupid but it fixes known a bug in docker build
# https://github.com/moby/moby/issues/37965
RUN true

# Install libera_sdp and all its (non-dev) dependencies according to pyproject.toml
RUN poetry install --no-dev

# Define the entrypoint of the container. Passing arguments when running the
# container will be passed as arguments to the function
ENTRYPOINT ["sdp"]


# libera-sdp-test
# ---------------
FROM libera-sdp AS libera-sdp-test

# Install dev dependencies (not installed in libera-sdp image)
RUN poetry install

# Copy tests over
COPY tests $LIBSDP_INSTALL_LOCATION/tests
COPY pylintrc $LIBSDP_INSTALL_LOCATION

# Set entrypoint
ENTRYPOINT pytest $LIBSDP_INSTALL_LOCATION