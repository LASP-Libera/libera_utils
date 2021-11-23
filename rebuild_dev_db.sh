#!/usr/bin/env bash

# Optional argument for target version
FLYWAY_TARGET=${1:-latest}

COMPOSE_FILE=db-compose.yml
# Try removing any existing services associated with our compose file, removing any volumes
docker compose -f $COMPOSE_FILE down -v
# Bring services back up
docker compose -f $COMPOSE_FILE up -d
