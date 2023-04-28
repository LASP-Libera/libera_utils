#!/bin/bash
# Sets up the development database with roles, users, database, and default permissions
# This script gets copied into the Dockerized postgres instance so it brings this up on start

export PGUSER=libera_master
export PGPASSWORD=masterpass

echo "Creating processor, tester, and reader roles and users"
cat << EOF | psql -d libera
CREATE ROLE reader_role;
CREATE ROLE processor_role;  -- privileges for processing data
CREATE ROLE tester_role;  -- privileges for running unit tests

CREATE USER libera_unit_tester PASSWORD 'testerpass' IN ROLE tester_role;
CREATE USER libera_processor PASSWORD 'processorpass' IN ROLE processor_role;
CREATE USER libera_reader PASSWORD 'readerpass' IN ROLE reader_role;
EOF

echo "Creating sdp schema in libera database"
cat << EOF | psql -d libera
CREATE SCHEMA sdp AUTHORIZATION libera_master;
EOF
