#!/usr/bin/env bash
# Run unit tests, build package, and push to LASP PyPI
set -e  # Exit if anything fails

# Run tests
pytest tests

# Build distribution artifacts
python -m build

# Push to Nexus PyPI - requires credential input
twine upload --repository-url https://artifacts.pdmz.lasp.colorado.edu/repository/lasp-pypi/ dist/*