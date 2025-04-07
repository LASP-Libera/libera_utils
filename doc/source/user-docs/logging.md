# Logging
High quality logging is an important part of operational processing and the Libera SDC Team has made logging setup as painless as possible, while also offering a high degree of
configuration for processing algorithms. See below for a general discussion of logging principles followed by some
example use cases.

[See the `libera_utils.logutil` API documentation here](../api-doc/generated/libera_utils.logutil.rst)

## Logging vs. `print`

Printing is a valid way to log from your code. However, it is limited in a few major ways:

1. You get no logging information from library code you have pulled in as dependencies.
2. There is no easy way to automate adding context to print statements such as current function, line, module, etc.
3. Formatting is only a convention with print calls, which makes log analysis and monitoring difficult.
4. There is no easy way to control the verbosity of your print statements.
5. You can only send messages to the console (stdout/stderr).

When using the Libera Utils logging module, you get:

1. Fine-grained information from all the libraries you are using.
2. Configurable standard context added to logs such as time, severity level, module, line number, function name, etc.
3. Consistent formatting to make logs easily searchable.
4. Ability to easily turn logging on/off from one place in the code.
5. Send log messages to multiple configurable destinations (console, file, etc).

See examples of these use cases in the code examples throughout this page.

## Logging Levels in Python

* DEBUG - Detailed information, typically of interest only when diagnosing problems.
* INFO - Confirmation that things are working as expected.
* WARNING - An indication that something unexpected happened, or indicative of some problem in the near future
  (e.g. ‘disk space low’). The software is still working as expected.
* ERROR - Due to a more serious problem, the software has not been able to perform some function.
* CRITICAL - A serious error, indicating that the program itself may be unable to continue running.

## Setting Up Logging in Applications

The `libera_utils.logutil` module provides utilities for configuring logging easily for file-based, stream
(stdout/stderr), and custom AWS CloudWatch logging.

### Simplest Logging Setup

This example allows all messages through to the console at the specified level. This does not filter out DEBUG logs from
verbose libraries.

```python
"""Simplest logging setup"""
import logging
from datetime import datetime, timezone
import boto3
from botocore.exceptions import NoCredentialsError
from libera_utils.logutil import configure_task_logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    task_id = f'processing-task-{datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")}'
    configure_task_logging(task_id, console_log_level="DEBUG")

    logger.debug("test debug message")

    try:
        # The following will demonstrate why we might want to filter out debug messages
        buckets = boto3.client('s3').list_buckets()
        logger.info(buckets)
    except NoCredentialsError:
        logger.error("No credentials found")
```

produces

```text
2024-04-23 07:54:37,523 INFO      [libera_utils.logutil:logutil.py:257 in configure_task_logging()]: Console logging configured at level DEBUG.
2024-04-23 07:54:37,523 DEBUG     [__main__:scratch_11.py:15 in <module>()]: test debug message
2024-04-23 07:54:37,523 DEBUG     [botocore.hooks:hooks.py:482 in _alias_event_name()]: Changing event name from creating-client-class.iot-data to creating-client-class.iot-data-plane
2024-04-23 07:54:37,524 DEBUG     [botocore.hooks:hooks.py:482 in _alias_event_name()]: Changing event name from before-call.apigateway to before-call.api-gateway
2024-04-23 07:54:37,524 DEBUG     [botocore.hooks:hooks.py:482 in _alias_event_name()]: Changing event name from request-created.machinelearning.Predict to request-created.machine-learning.Predict
2024-04-23 07:54:37,524 DEBUG     [botocore.hooks:hooks.py:482 in _alias_event_name()]: Changing event name from before-parameter-build.autoscaling.CreateLaunchConfiguration to before-parameter-build.auto-scaling.CreateLaunchConfiguration
2024-04-23 07:54:37,525 DEBUG     [botocore.hooks:hooks.py:482 in _alias_event_name()]: Changing event name from before-parameter-build.route53 to before-parameter-build.route-53
2024-04-23 07:54:37,525 DEBUG     [botocore.hooks:hooks.py:482 in _alias_event_name()]: Changing event name from request-created.cloudsearchdomain.Search to request-created.cloudsearch-domain.Search
2024-04-23 07:54:37,525 DEBUG     [botocore.hooks:hooks.py:482 in _alias_event_name()]: Changing event name from docs.*.autoscaling.CreateLaunchConfiguration.complete-section to docs.*.auto-scaling.CreateLaunchConfiguration.complete-section
2024-04-23 07:54:37,526 DEBUG     [botocore.hooks:hooks.py:482 in _alias_event_name()]: Changing event name from before-parameter-build.logs.CreateExportTask to before-parameter-build.cloudwatch-logs.CreateExportTask
2024-04-23 07:54:37,526 DEBUG     [botocore.hooks:hooks.py:482 in _alias_event_name()]: Changing event name from docs.*.logs.CreateExportTask.complete-section to docs.*.cloudwatch-logs.CreateExportTask.complete-section
2024-04-23 07:54:37,526 DEBUG     [botocore.hooks:hooks.py:482 in _alias_event_name()]: Changing event name from before-parameter-build.cloudsearchdomain.Search to before-parameter-build.cloudsearch-domain.Search
2024-04-23 07:54:37,526 DEBUG     [botocore.hooks:hooks.py:482 in _alias_event_name()]: Changing event name from docs.*.cloudsearchdomain.Search.complete-section to docs.*.cloudsearch-domain.Search.complete-section
...and much much more
```

*Notice the huge volume of DEBUG messages originating from loggers in the `botocore` package.*

### Filtered Logging Setup

Now we want to alter the code above to take advantage of the ability to reduce the amount of DEBUG spam from libraries
that we are not interested in. Note the use of `__main__` in the `limit_debug_loggers` tuple. This allows debug messages
that originate at the level of a runnable python script that is wrapped in the standard `if __name__=="__main__"` guard.

```python
"""Simplest logging setup"""
import logging
from datetime import datetime, timezone
import boto3
from botocore.exceptions import NoCredentialsError
from libera_utils.logutil import configure_task_logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    task_id = f'processing-task-{datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")}'
    configure_task_logging(task_id, limit_debug_loggers=("__main__", "libera_utils"), console_log_level="DEBUG")

    logger.debug("test debug message")

    try:
        # The following will demonstrate why we might want to filter out debug messages
        buckets = boto3.client('s3').list_buckets()
        logger.info(buckets)
    except NoCredentialsError:
        logger.error("No credentials found")
```

produces

```text
2024-04-23 07:52:56,009 INFO      [libera_utils.logutil:logutil.py:257 in configure_task_logging()]: Console logging configured at level DEBUG.
2024-04-23 07:52:56,009 DEBUG     [__main__:scratch_11.py:15 in <module>()]: test debug message
2024-04-23 07:52:56,186 ERROR     [__main__:scratch_11.py:22 in <module>()]: No credentials found
```

### Contrived Runnable Example

This illustrates how the `limit_debug_loggers` kwarg works for preventing other libraries from logging at DEBUG level
while still allowing DEBUG messages from libraries you care about.

```python
"""Logging setup contrived example"""
from datetime import datetime, timezone
import logging
from libera_utils.logutil import configure_task_logging

task_id = f'processing-task-{datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")}'
configure_task_logging(task_id, limit_debug_loggers=("my_package",), console_log_level=logging.DEBUG)
# my_log is a logger from inside your library (name prefixed with your library name).
my_log = logging.getLogger('my_package.my_application')
# The following is an example. External libraries will create their own loggers internally.
external_library_log = logging.getLogger('some_spammy_library')

my_log.debug('subtle but important message gets passed through')
external_library_log.debug('this external library debug spam gets filtered out')
external_library_log.info('and external library info messages still get through')
```

### Configuring Stream Logging

Stream logging needs only a level and it defaults to `INFO`. Stream logging cannot be disabled but is easily ignored.

Note: You can change the default formatter for stream logging from plaintext to json by passing `console_log_json=True`
to `configure_task_logging`. This is convenient for logging in AWS services that push their stdout logs to CloudWatch.

```python
"""Example of setting up console logging (JSON formatted)"""
import logging
from libera_utils.logutil import configure_task_logging

configure_task_logging("test-task-id-1", console_log_json=True, console_log_level=logging.DEBUG)
```

### Configuring Filesystem Logging

Filesystem logging needs only a log directory (it always logs at DEBUG level). If you don't pass `log_dir`, no
file-based logging will occur. The log directory must exist and will not be dynamically created.

_Note: Logging to a directory is really only useful for local testing. Any logs written to a directory in a docker
container will evaporate upon completion of the docker container process._

```python
"""Example of setting up file-based logging"""
from pathlib import Path
from libera_utils.logutil import configure_task_logging

configure_task_logging("test-task-id-1", log_dir=Path("/tmp"))
```

## Logging Exceptions

The Python logging module provides logging calls associated with each level. In addition, it provides a logging call for
logging exceptions that includes the current stack trace for debugging.

```python
"""Example logging calls"""
import logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger.debug("test debug")
    logger.info("test info")
    logger.warning("test warning")
    logger.error("test error")
    logger.critical("test critical")

    try:
        raise ValueError("example exception to log")
    except ValueError:
        logger.exception("encountered exception")  # This logs the message followed by the exception traceback
        # raise  # Optional re-raise of exception after logging it
```

## Concept: Module Level Logging

(AKA library logging)

Module level logging is the practice of defining a logger at the top of each module (.py file) and using that logger
object for the entire module. Modules in libera_utils should all have module level loggers when appropriate. e.g.

```python
"""Example of module level logger instantiation"""
import logging

logger = logging.getLogger(__name__)

# Module code
```

One advantage of module level logging is that it provides a logger, named for the module from which it is logging but
doesn't configure the logger at the module level. Since much of this code is intended to be reused by others, we avoid
configuring loggers in reusable code. If a logger is needed in a context, it should be configured at the
"application level". That is, the top level application (e.g. CLI tool) that is running a process should take care of
logging configuration and assume that each module has generic module level loggers configured for the internal code to
use.

Another advantage of module level logging is that our loggers come out with automatically structured names like
`libera_utils.db.database` and they all start with `libera_utils`. This allows us to treat those loggers differently
than those named, for example, `some_spammy_library.emit_spam`. In our logging setup, we allow users to turn off debug
messages for all loggers that aren't named with specific prefixes. This allows us to pass up debug messages from our
code but ignore debug messages from dependency code (AWS boto APIs in particular spam a LOT of debug messages).


## Fully Customized Logging

If you want complete control over your logging configuration, you can use our provided `configure_static_logging`
function, which reads a YAML configuration file that represents a Python logging configuration. This is a completely static
configuration and should be supplied as part of your processing algorithm application code.

```python
"""Example of configuring logging with static config file"""
import logging
from pathlib import Path
from libera_utils.logutil import configure_static_logging

config = Path("/path/to/config_file.yml")
configure_static_logging(config)

logger = logging.getLogger()
logger.info("handling depends on your supplied configuration")
```

An example of a logging config file:

```yaml
# Example parameterized logging configuration
version: 1
disable_existing_loggers: False
formatters:
    json:
        format: '{"time": "%(asctime)s",
                  "level": "%(levelname)s",
                  "module": "%(filename)s",
                  "function": "%(funcName)s",
                  "line": %(lineno)d,
                  "message": "%(message)s"}'
    plaintext:
        format: "%(asctime)s %(levelname)-9.9s [%(filename)s:%(lineno)s in %(funcName)s()]: %(message)s"
handlers:
    console:
        class: logging.StreamHandler
        formatter: plaintext
        level: INFO
        stream: ext://sys.stdout
    logfile:
        class: logging.handlers.RotatingFileHandler
        formatter: plaintext
        level: DEBUG
        filename: /tmp/libera_utils_test_log.log
        maxBytes: 1000000
        backupCount: 3
root:
    level: INFO
    propagate: True
    handlers: [console, logfile]
loggers:
    libera_utils:
        qualname: libera_utils
        level: DEBUG
        handlers: []
```
