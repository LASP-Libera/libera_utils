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


# Accessing Databases from Python


To facilitate multiprocessing safety and reduce strain on database resources, we use a manager pattern to
manage database connections. A unique manager object is identified by its connection URL and the process ID
in which it was created. SQLAlchemy engines are not safe to pass between forked processes (they seem 
to be OK if processes are spawned) so we use a caching mechanism. When a process is forked, it does 
copy the existing cache but the cache lookup no longer matches that manager because the PID has changed.

In short, use the following pattern to create database connections:

```python
from libera_sdp.db import getdb
from libera_sdp.db.models import Level0


db = getdb()
with db.session() as s:
    records = s.query(Level0).filter(Level0.filename == 'foofile.txt').all()
```


# Dockerized Dev Database


The Dockerized development database is a carbon copy of the production schema, 
dynamically generated based on the SQL migration scripts in the `database` 
directory. It can be deployed (and redeployed) locally by running 

```shell
docker compose up flyway-sdp-dev flyway-sdp-test flyway-sdp-prod
```
