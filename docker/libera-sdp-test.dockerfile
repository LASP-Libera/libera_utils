# Dockerfile that adds tests to the the liber-sdp docker image

FROM libera-sdp:latest

# Install dev dependencies (not installed in libera-sdp image)
RUN poetry install

# Copy tests over
COPY tests $LIBSDP_INSTALL_LOCATION/tests

# Set entrypoint
ENTRYPOINT pytest $LIBSDP_INSTALL_LOCATION
