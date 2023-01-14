#!/bin/bash
# Sets up the development database with roles, users, database, and default permissions
# This script gets copied into the Dockerized postgres instance so it brings this up on start

export PGUSER=libera_master
export PGPASSWORD=masterpass

DBNAME=libera
SDPSCHEMA=sdp

cat << EOF | psql -d postgres
CREATE ROLE reader_role;
CREATE ROLE processor_role;  -- privileges for processing data
CREATE ROLE tester_role;  -- privileges for running unit tests

CREATE USER libera_unit_tester PASSWORD 'testerpass' IN ROLE tester_role;
CREATE USER libera_processor PASSWORD 'processorpass' IN ROLE processor_role;
CREATE USER libera_reader PASSWORD 'readerpass' IN ROLE reader_role;

CREATE DATABASE libera OWNER libera_master;
EOF

echo "Granting default permissions to libera_processor, libera_unit_tester, and libera_reader"
cat << EOF | psql -d libera
CREATE SCHEMA sdp AUTHORIZATION libera_master;

GRANT USAGE ON SCHEMA sdp TO processor_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA sdp GRANT SELECT, INSERT, UPDATE ON TABLES TO processor_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA sdp GRANT USAGE, SELECT ON SEQUENCES TO processor_role;

GRANT ALL ON SCHEMA sdp TO tester_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA sdp GRANT ALL PRIVILEGES ON TABLES TO tester_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA sdp GRANT USAGE, SELECT ON SEQUENCES TO tester_role;

GRANT USAGE ON SCHEMA sdp TO reader_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA sdp GRANT SELECT ON TABLES TO reader_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA sdp GRANT USAGE, SELECT ON SEQUENCES TO reader_role;
EOF
