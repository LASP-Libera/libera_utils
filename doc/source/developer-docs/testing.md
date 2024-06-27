# Testing


## Testing Locally
Testing is run with `pytest`. To run all tests, make sure the dev dependencies are installed and navigate to the _tests_
folder in the repository and run:
```bash
pytest
```

With coverage for generating reports on code coverage:
```bash
# Create coverage data (stored in .coverage)
pytest --cov=libera_utils --junit-xml=junit.xml
# Generate interactive HTML coverage report
pytest --cov-report=html:coverage_report --cov=libera_utils
# Generate Corbertura-compatible XML report
pytest --cov-report=xml:coverage.xml --cov=libera_utils
```


## Testing in Docker


To run the unit tests in docker, run
```shell
docker-compose up [--build] --exit-code-from=tests tests --attach=tests
```
This ensures the dev database server is up, runs the latest flyway migrations against it, 
and runs the tests container service defined in the `docker-compose.yml` file. The `--build` option forces
docker to rebuild the testing container image before running (e.g. if things have changed).


### Copying Test Report Artifacts from Docker

When we run tests in Docker on Jenkins, we often want to copy and save Corbertura and JUnit test reports. Jenkins 
has facility for doing this easily with
```Groovy
always {
    junit '**/*junit.xml'
    cobertura coberturaReportFile: '**/*coverage.xml'
}
```
The challenge when running in Docker is to make these test artifacts available to Jenkins. By default these files 
exist only inside the Docker container so we must copy them out. Do this with
```shell
docker-compose --exit-code-from tests up tests
docker-compose cp tests:/path/to/report.xml .
```


## Static Analysis
NPR7150.2C requires that we perform static analysis of our codebase to check for common vulnerabilities and 
statically detectable code weaknesses.


### Pylint for Code Style
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


### Bandit for Automated Security Testing (AST)
[Bandit](https://github.com/PyCQA/bandit) is a security vulnerability tester that analyzes Python code for common 
weaknesses, including the "official" CWE set provided by the [MITRE Corp.](https://cwe.mitre.org/). 

To run bandit locally, run 

```bandit -r libera_sdp``` 

from the repo root. This will recursively (`-r`) analyze the `libera_sdp` directory and report results.

To run bandit inside Docker, run 

```docker-compose up [--build] ast```

This starts up the `ast` service described in `docker-compose.yml`, which runs bandit inside a testing docker container.


### Static Analysis pre-commit Git Hook

The static analysis checking can be annoying when you only catch it after you have committed, rebased, and put up a PR.
To alleviate that suffering, here is a pre-commit git hook that will execute before every commit and refuse to 
continue until you have fixed your static analysis problems.

```shell
#!/bin/sh

# .git/hooks/pre-commit

# Redirect output to stderr.
exec 1>&2

# Run pylint.
echo "Running pylint..."
files_to_check=$(git diff --name-only --cached --diff-filter=AM libera_utils | grep '.py$')
if [ -n "$files_to_check" ]; then
  pylint $files_to_check
else
  echo "No .py file changes to lint"
fi

# Capture the output of pylint.
pylint_exit_code=$?

# If pylint returns a non-zero exit code, cancel the commit.
if [ $pylint_exit_code -ne 0 ]; then
  echo "Pylint check failed, aborting commit..."
  exit 1
fi

# Otherwise, proceed with the commit.
echo "Pylint check passed, proceeding with commit..."

echo "Running bandit..."
bandit -r --quiet libera_utils
bandit_exit_code=$?

# If Bandit returned a non-zero exit code, cancel the commit.
if [ $bandit_exit_code -ne 0 ]; then
    echo "Bandit check failed, aborting commit..."
    exit 1
fi

# Otherwise, proceed
echo "Bandit AST passed, proceeding with commit..."

exit 0
```