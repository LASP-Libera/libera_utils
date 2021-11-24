#!/bin/bash
# Sets up the development database with roles, users, databases, and default permissions
# This script gets copied into the Dockerized postgres instance so it brings this up on start

export PGUSER=libera_master
export PGPASSWORD=masterpass

cat << EOF | psql -d postgres
CREATE ROLE reader_role;
CREATE ROLE processor_role;  -- privileges for processing data
CREATE ROLE tester_role;  -- privileges for running unit tests

CREATE USER libera_unit_tester PASSWORD 'testerpass' IN ROLE tester_role;
CREATE USER libera_processor PASSWORD 'processorpass' IN ROLE processor_role;
CREATE USER libera_reader PASSWORD 'readerpass' IN ROLE reader_role;

CREATE DATABASE sdp_dev OWNER libera_master;
CREATE DATABASE sdp_test OWNER libera_master;
CREATE DATABASE sdp_prod OWNER libera_master;
EOF

echo "Granting default permissions to libera_processor, libera_unit_tester, and libera_reader"
# ON sdp_dev DATABASE
cat << EOF | psql -d sdp_dev
GRANT USAGE ON SCHEMA public TO processor_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE ON TABLES TO processor_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO processor_role;

GRANT USAGE ON SCHEMA public TO reader_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO reader_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO reader_role;
EOF

# ON sdp_test DATABASE
cat << EOF | psql -d sdp_test
GRANT USAGE ON SCHEMA public TO tester_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL PRIVILEGES ON TABLES TO tester_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO tester_role;

GRANT USAGE ON SCHEMA public TO reader_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO reader_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO reader_role;
EOF

# ON sdp_prod DATABASE
cat << EOF | psql -d sdp_prod
GRANT USAGE ON SCHEMA public TO processor_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE ON TABLES TO processor_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO processor_role;

GRANT USAGE ON SCHEMA public TO reader_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO reader_role;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO reader_role;
EOF
