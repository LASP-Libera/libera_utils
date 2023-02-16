# Databases, Roles, and Permissions


## Databases:
- `libera` - Name given to the main database in all contexts.

## Schemas:
- `sdp` - Name given to the science data processing schema. 

Using a named schema is postgres best practice and gives us the flexibility to add another schema later if we need to.


## Roles:
- `libera_master:masterpass` - Master user on all databases. Full access. Do not 
    use except for schema management and other administrative tasks.
- `libera_unit_tester:testerpass` - User for unit testing. Should only be created in the local dev DB.
- `libera_processor:processorpass` - Data processing user.
- `libera_reader:readerpass` - Read only on all databases.


## Modifying Database Schema
_Note: Until we have a production database up and running, we can blow away and recreate our entire 
database schema without risk. Once in production, we need to incrementally change the schema using 
migrations._ 

We use a tool called Flyway to modify database schema using pure SQL migrations. This allows us full control
over the schema (as opposed to using an ORM framework like Alembic to manage the DB). And separates the
concerns of the database from the concerns of the pipeline code. 

To create a schema change, create a new migration .sql script in `database/migrations`, named as 
`V<major>[.<minor>[.<patch>]]__<imperative_description>.sql` and add your SQL script that will change
the schema.

Important considerations:
1. Remember that the production schema contains data. Many schema changes work correctly on an empty 
   schema but fail in the presence of data. e.g. Adding a NOT NULL constraint to a column containing
   NULL values will need to fill those NULLs first.
2. Repeatable migrations MUST be idempotent and are not guaranteed to run in a particular order relative to versioned
   migrations. They run based on a difference in checksum between the current migration script and the last.
3. There is no such thing as truly reversing a database change. Reversion of a DB schema is much like reverting a
   erroneous git commit. Rather than truly backing out the changes, you simply add a new change that alters things
   back to the way you want them. The upshot: NEVER delete a sql migration 


## Accessing Databases from Python
To facilitate multiprocessing safety and reduce strain on database resources, we use a manager pattern to
manage database connections. A unique manager object is identified by its connection URL and the process ID
in which it was created. SQLAlchemy engines are not safe to pass between forked processes (they seem 
to be OK if processes are spawned) so we use a caching mechanism. When a process is forked, it does 
copy the existing cache but the cache lookup no longer matches that manager because the PID has changed.

In short, use the following pattern to create database connections:

```python
from libera_utils.db import getdb
from libera_utils.db.models import PdsFile

db = getdb()
with db.session() as s:
    records = s.query(PdsFile).filter(PdsFile.file_name == 'foofile.txt').all()
```


## Dockerized Dev Database
The Dockerized development database is a carbon copy of the production schema, 
dynamically generated based on the SQL migration scripts in the `database` 
directory. It can be deployed (and redeployed) locally by running 

```shell
docker compose up flyway
```
