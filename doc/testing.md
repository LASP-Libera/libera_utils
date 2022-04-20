# Testing


## Testing Locally
Testing is run with `pytest`. To run all tests, make sure the dev dependencies are installed and run:
```bash
pytest
```

With coverage for generating reports on code coverage:
```bash
# Create coverage data (stored in .coverage)
pytest --cov=libera_sdp --junit-xml=junit.xml
# Generate interactive HTML coverage report
pytest --cov-report=html:coverage_report --cov=libera_sdp
# Generate Corbertura-compatible XML report
pytest --cov-report=xml:coverage.xml --cov=libera_sdp
```


## Testing in Docker


To run the unit tests in docker, run
```shell
docker-compose up [--build] --exit-code-from=tests tests
```
This ensures the dev database server is up, runs the latest flyway migrations against it, 
and runs the tests container service defined in the `docker-compose.yml` file. The `--build` option forces
docker to rebuild the testing container image before running (e.g. if things have changed).


# Static Analysis
NPR7150.2C requires that we perform static analysis of our codebase to check for common vulnerabilities and 
statically detectable code weaknesses.


## Pylint for Code Style
[Pylint](https://pylint.pycqa.org/en/latest/) is a powerful and highly configurable static analysis tool that 
analyzes Python code for coding standards (e.g. PEP8), code smells, type errors, and violations of common 
best practices. Together, these violations are common code weaknesses and we strive to eliminate them.

To run pylint locally, run 

```pylint libera_sdp```

from the repo root. This will lint the `libera_sdp` directory and automatically pick up our `pylintrc` 
configuration file.

To run pylint inside Docker, run 

```docker-compose up [--build] linting```

This starts up the linting service described in `docker-compose.yml`, which runs pylint inside a 
testing docker container.

NOTE: Pylint almost always produces non-zero exit codes. It's extremely strict to require pylint to exit 0.


## Bandit for Automated Security Testing (AST)
[Bandit](https://github.com/PyCQA/bandit) is a security vulnerability tester that analyzes Python code for common 
weaknesses, including the "official" CWE set provided by the [MITRE Corp.](https://cwe.mitre.org/). 

To run bandit locally, run 

```bandit -r libera_sdp``` 

from the repo root. This will recursively (`-r`) analyze the `libera_sdp` directory and report results.

To run bandit inside Docker, run 

```docker-compose up [--build] ast```

This starts up the ast service described in `docker-compose.yml`, which runs bandit inside a testing docker container.

NOTE: Bandit almost always produced non-zero exit codes. It's extremely strict to require bandit to exit 0.
