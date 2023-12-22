# Logging

[See the logging API documentation here](../api-doc/generated/libera_utils.logutil.rst)

## Concept: Module Level Logging

Module level logging is the practice of defining a logger at the top of each module (.py file) and using that 
logger object for the entire module. Modules in libera_utils should all have module level loggers when appropriate.
e.g.
```
logger = logging.getLogger(__name__)
```

One advantage of module level logging is that it provides a logger, named for the module from which it is logging 
but doesn't configure the logger at the module level. Since much of this code is intended to be reused by others,
we avoid configuring loggers in reusable code. If a logger is needed in a context, it should be configured at the 
"application level". That is, the top level application (e.g. CLI tool) that is running a process should take care of
logging configuration and assume that each module has generic module level loggers configured for the internal code
to use.

Another advantage of module level logging is that our loggers come out with automatically structured names like 
`libera_utils.db.database` and they all start with `libera_utils`. This allows us to treat those loggers differently than
those named, for example, `some_spammy_library.emit_spam`. In our logging setup, we turn off debug messages for all 
loggers that aren't children of `libera_utils`. This allows us to pass up debug messages from our code but ignore debug 
messages from dependency code (AWS boto APIs in particular spam a LOT of debug messages).


## Setting Up Logging in Applications
The `logutil` module provides utilities for configuring logging easily for filesystem, stream, and 
AWS Cloudwatch logging.

```python
"""Logging setup example"""
import logging
import os
from libera_utils.logutil import configure_task_logging

os.environ['LIBERA_LOG_DIR'] = "/tmp"  # Turn on file logging
os.environ['LIBERA_CONSOLE_LOG_LEVEL'] = "DEBUG"  # Set console log level to DEBUG
task_id = 'processing-task-1'
configure_task_logging(task_id, "my_package")
# my_log is a logger from inside your library (name prefixed with your library name).
my_log = logging.getLogger('my_package.my_application')
# The following is an example. External libraries will create their own loggers internally.
external_library_log = logging.getLogger('some_spammy_library')

my_log.debug('subtle but important message gets passed through')
external_library_log.debug('this external library debug spam gets filtered out')
external_library_log.info('and external library info messages still get through')
```


### Configuring Stream Logging
Stream logging needs only a level. It gets it from (in descending precedence)
* `console_log_level` argument to `configure_task_logging()`
* `LIBERA_CONSOLE_LOG_LEVEL` environment variable (this is the preferred method)
* `config.json` default `INFO`

Stream logging cannot be disabled but is easily ignored.


### Configuring Filesystem Logging
Filesystem logging needs only a log directory (it always logs DEBUG level). It gets it from (in descending precedence)
* `LIBERA_LOG_DIR` environment variable
* `config.json` default (`null`)

The default log dir in `config.json` is `null` and will result in no file logging handlers being created.

_Note: Logging to a directory is really only useful for local testing. Any logs written to a directory in a 
docker container will evaporate upon completion of the docker container process._


### Configuring Cloudwatch Logging
Cloudwatch logging needs a log group (like a log directory). It gets it from (in descending precedence)
* `cloudwatch_group` argument to `setup_task_logger`
* `LIBERA_CLOUDWATCH_GROUP` environment variable
* `config.json` default (`null`)

The default log group in `config.json` is `null` and will result in no cloudwatch logging.

_Note: In Batch, any logs written to stdout or sterr are captured by cloudwatch as a string. If you want 
logs to be more searchable, you may configure cloudwatch logging. Write messages in JSON format and they will be
sent to Cloudwatch as nested JSON objects as follows._

```
"json": {
    "format": '{"time": "%(asctime)s", level": "%(levelname)s", "module": "%(filename)s", '
              '"function": "%(funcName)s", "line": %(lineno)d, "message": "%(message)s"}',
}
```
The above is an excerpt from the logging configuration dict in `logutil`.
