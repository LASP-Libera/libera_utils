#!/usr/bin/env bash
# Run unit tests with coverage and open the coverage html file in a browser
set -e  # Exit if anything fails

if [ "$(uname)" == "Darwin" ]; then
    HOST_OS="Darwin"
    HTML_OPEN_CMD="open"
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    HOST_OS="Linux"
    HTML_OPEN_CMD="xdg-open"
else
    echo "ERROR: Unrecognized OS. Must be darwin or linux-gnu"
    exit 1
fi

# Run tests with coverage
coverage run --source=libera_sdp --module pytest -ra tests

# Create HTML report
coverage html -d coverage_report

# Open html report file
$HTML_OPEN_CMD coverage_report/index.html
