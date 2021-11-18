#!/usr/bin/env bash
# This script builds dockerfiles for the Libera SDP project
set -e

if [[ $# == 0 ]]; then
  echo "ERROR: Must provide at least one target image name as an argument to docker_build.sh."
  exit 1
fi

echo "Building Libera SDP docker image..."

# This is just a clever way to dynamically get the directory that the script is in, regardless of where it was run from
CONTEXT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo "Build context directory: $CONTEXT_DIR (based on the location of this script)."

LIBERA_SDP_VERSION=$(poetry version -s)
echo "Libera SDP (libera_sdp) package version: $LIBERA_SDP_VERSION"

TARGETS=()
while [[ $# -gt 0 ]]; do
  TARGETS+=("$1")
  shift  # Next argument
done

echo "Building target images: ${TARGETS[@]}"

for target in ${TARGETS[@]}; do
  echo; echo "Building target image: ${target}..."
  docker build -t ${target}:$LIBERA_SDP_VERSION --target ${target} .
done

echo "Finished! Yay!"
