#!/usr/bin/env bash
# This script builds dockerfiles for the Libera SDP project
set -e

# PARSE INPUT
# -----------
IMAGE_NAMES=()
if [[ "$1" = "all" ]]; then
  IMAGE_NAMES=("libera-sdp" "libera-sdp-test")
  echo "Building all images: ${IMAGE_NAMES[@]}"
else
  while [[ $# -gt 0 ]]; do
    IMAGE_NAMES+=("$1")
    shift  # Next argument
  done
fi

# FUNCTIONS
# ---------
function build_image {
  local img_name=$1

  # Set the .dockerignore contents dynamically to reduce build context
  cp ${DOCKERFILE_DIR}/${img_name}.dockerignore ${CONTEXT_DIR}/.dockerignore

  # Build the image
  docker build -t ${img_name}:${LIBERA_SDP_VERSION} \
      -f ${DOCKERFILE_DIR}/${img_name}.dockerfile ${CONTEXT_DIR}

  # Tag newly built image with latest
  docker tag ${img_name}:${LIBERA_SDP_VERSION} ${img_name}:latest
  #TODO: Add some error handling so we always clean up the .dockerignore regardless of whether the build succeeds
  rm ${CONTEXT_DIR}/.dockerignore
}

# SCRIPT
# ------
echo "Building docker containers for Libera SDP"
# This is just a clever way to dynamically get the directory that the script is in, regardless of where it was run from
CONTEXT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo "Build context directory assumed to be $CONTEXT_DIR (based on the location of this script)."

DOCKERFILE_DIR="$CONTEXT_DIR/docker"
echo "Using Dockerfiles in directory $DOCKERFILE_DIR"

LIBERA_SDP_VERSION=$(poetry version -s)
echo "Libera SDP (libera_sdp) package version $LIBERA_SDP_VERSION"

for image_name in ${IMAGE_NAMES[@]}; do
  echo "Building named image ${image_name}"
  build_image $image_name
done

echo "Finished!"