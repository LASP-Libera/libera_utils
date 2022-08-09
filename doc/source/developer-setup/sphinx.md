# Installing Sphinx Autodocumentation

## Step by Step Guide
This should all be done while in the virtual environment that is configured for this project.
See documentation dev-environment-setup.md in the libera-sdp repository for poetry instructions.

Ensure that the poetry install is up-to-date
1. Run `poetry update`

Install the packages defined in pyproject.toml extras as "docs" to install Sphinx and dependencies in the environment
2. Run `poetry install --extras docs`

Build the documentation using sphinx autoapi package
3. Navigate to the Sphinx document source folder `cd ./doc/source`
4. Build the html files in the build folder `sphinx-build -b html . ../build`