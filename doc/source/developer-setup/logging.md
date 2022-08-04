# Logging


## Concept: Module Level Logging

Module level logging is the practice of defining a logger at the top of each module (.py file) and using that 
logger object for the entire module. Modules in libera_sdp should all have module level loggers, like
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
`libera_sdp.db.database` and they all start with `libera_sdp`. This allows us to treat those loggers differently than
those named, for example, `some_spammy_library.emit_spam`. In our logging setup, we turn off debug messages for all 
loggers that aren't children of `libera_sdp`. This allows us to pass up debug messages from our code but ignore debug 
messages from dependency code (AWS boto APIs in particular spam a LOT of debug messages).


## Setting Up Logging in Applications
The `libera_sdp.logutil` module provides utilities for configuring logging easily for filesystem, stream, and 
AWS Cloudwatch logging.

```python
"""Logging setup example"""
import logging
from libera_utils.logutil import setup_task_logger

log_dir = '/tmp'
task_id = 'processing-task-1'
stream_log_level = 'DEBUG'
setup_task_logger(task_id, stream_log_level, log_dir)
libera_log = logging.getLogger('libera_utils.my_application')
external_library_log = logging.getLogger('some_library')

libera_log.debug('subtle but important message gets passed through')
external_library_log.debug('this external library debug spam gets filtered out')
external_library_log.info('and external library info messages still get through')
```


### Configuring Stream Logging
Stream logging needs only a level. It gets it from (in descending precedence)
* `stream_log_level` argument to `setup_task_logger`
* `LIBSDP_STREAM_LOG_LEVEL` environment variable
* `config.json` default (`INFO`)

Stream logging cannot be disabled but is easily ignored.


### Configuring Filesystem Logging
Filesystem logging needs only a log directory (it always logs DEBUG level). It gets it from (in descending precedence)
* `logdir` argument to `setup_task_logger`
* `LIBSDP_LOG_DIR` environment variable
* `config.json` default (`null`)

The default log dir in `config.json` is `null` and will result in no file logging handlers being created. 


### Configuring Cloudwatch Logging
Cloudwatch logging needs a log group (like a log directory). It gets it from (in descending precedence)
* `cloudwatch_group` argument to `setup_task_logger`
* `LIBSDP_CLOUDWATCH_GROUP` environment variable
* `config.json` default (`null`)

The default log group in `config.json` is `null` and will result in no cloudwatch logging.
