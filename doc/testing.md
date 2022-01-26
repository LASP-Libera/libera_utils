# Testing


## Testing Locally
Testing is run with `pytest`. To run all tests, make sure the `test` optional dependencies are installed and run:
```bash
pytest tests
```

With coverage for generating reports on code coverage:
```bash
# Create coverage data (stored in .coverage)
pytest --cov=libera_sdp --junit-xml=junit.xml tests
# Generate interactive HTML coverage report
pytest --cov-report=html:coverage_report --cov=libera_sdp tests
# Generate Corbertura-compatible XML report
pytest --cov-report=xml:coverage.xml --cov=libera_sdp tests
```


## Testing in Docker


To run the unit tests in docker, run
```shell
docker compose up -d flyway-sdp-dev flyway-sdp-test flyway-sdp-prod && docker compose run tests
```
This ensures the dev database server is up, runs the latest flyway migrations against it, 
and runs the tests container service defined in the `docker-compose.yml` file.