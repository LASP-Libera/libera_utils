# Databases, Roles, and Permissions


## Databases:
- `sdp_prod` - Production processing database.
- `sdp_dev` - Development database.
- `sdp_test` - Testing database.


## Schemas
We use the default `public` schema for all our databases.


## Roles:
- `libera_master:masterpass` - Master user on all databases. Full access. Do not 
    use except for schema management and other administrative tasks.
- `libera_unit_tester:testerpass` - User for unit testing.
- `libera_processor:processorpass` - Data processing user.
- `libera_reader:readerpass` - Read only on all databases.


# Dockerized Dev Database


The Dockerized development database is a carbon copy of the production schema, 
dynamically generated based on the SQL migration scripts in the `database` 
directory. It can be redeployed locally by running the `rebuild_dev_db.sh` 
script, optionally with a target version argument. The script uses docker
compose to bring up flyway services along with a postgres container. The 
flyway services mount the migrations and flyway config file and migrate 
each database (prod, dev, test) to the specified schema version.
