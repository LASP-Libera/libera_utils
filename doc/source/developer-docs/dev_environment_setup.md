# Setting Up a Development Environment

## Managing Multiple Base Python Versions

In order to develop with multiple different versions of Python and create virtual environments associated with 
different versions of Python, you will need multiple base Python interpreters.
There are several ways to manage this including Conda, PyEnv, and building Python from source. 
We recommended using Conda and outline the steps below for using Conda to manage multiple base Python installations.

1. Install miniconda according to the [official documentation](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html).
   If you already have miniconda or anaconda installed, you can skip this step.
2. Optionally run `conda config --set auto_activate_base false` to add a configuration to your `.condarc` file to 
   disable auto-activation of the `base` conda environment on shell startup.
3. Create a conda environment with your preferred version of python: `conda create -n conda-python3.11 python=3.11`
   - Note: Name this environment with a convention that makes sense to you for a base interpreter. 
     *Do not delete this conda environment!* Deleting it will break all subsequent virtual environments based on it.
4. The Python interpreter provided by your new conda environment is a full base interpreter and you can use it to
   create virtual environments. You can find the full path to the base interpreter by running something similar 
   to the following (run `conda env list` to see why this works):
   ```shell
   PATH_TO_PYTHON=$(conda env list | grep "conda-python3.11" | awk '{print $2}')/bin/python
   $PATH_TO_PYTHON -m venv path/to/new/venv
   ```

## Installing Poetry
Poetry is a command line tool that helps manage a python development environment, 
including package management, virtual environment management, and package building.

Poetry official installation instructions can be found here: https://python-poetry.org/docs/#installation

To ensure that Poetry is always available and in the `PATH` 
it is recommended to install Poetry with your pre-installed system python interpreter rather than as a package in a 
conda environment or in a virtual environment. The specific version of python with which you install Poetry 
is inconsequential (as long as it is currently supported by Poetry). If your system python is not supported by Poetry, 
you can install Poetry in your conda base environment. Just remember that Poetry will only be available when that 
environment is activated. Things can get a bit confusing when you have a conda environment active and a derived
virtual environment activated on top of it.

Once poetry is installed, check that it works by running `poetry --version`. You should get something like 
```
Poetry version 1.8.3
```

### Installing Poetry with System Python

Ensure that all your virtual environments and conda environments are deactivated and that `which python3` refers to your 
system python interpreter (usually `/usr/bin/python3`).

```
curl -sSL https://install.python-poetry.org | python3 -
```

## Configuring Poetry

We recommend creating your own virtual environments in locations of your choosing (common to create them in the 
project directory, as `venv` or `.venv`).

## Setting Up Development Virtual Environment(s)

Poetry will
dynamically detect the presence of an activated virtual environment and use that if present. If none is present,
Poetry will automatically create one for you in your user cache location 
(e.g. on mac in `~/Library/Caches/pypoetry/virtualenvs`). This can be confusing so we recommend creating your venv 
and letting poetry use it when activated.

1. Deactivate all Conda environments and virtual environments
2. Activate the conda environment you wish to use for your base python interpreter (e.g. `conda activate conda-python3.11`)
3. Create a virtual environment anywhere you wish. `python -m venv path/to/venv`.
4. Activate newly created venv: `source path/to/venv/bin/activate`
5. [Recommended]: Configure your IDE to recognize the correct poetry-managed virtual environment for the version you wish to develop with.
6. Run `poetry env info` and verify that Poetry is recognizing your virtual environment properly:
    ```
    Virtualenv
    Python:         3.9.9
    Implementation: CPython
    Path:           /Users/myuser/path/to/libera_utils/venv
    Valid:          True
    ```

### Changing Python Versions
It is common to recreate your virtual environment on a regular basis in order to use different python versions.
You can do this by making sure that `python` points to the base interpreter you wish to use (e.g. 3.12) and 
going through the steps above to create a new venv (you can name it differently) and activating it for poetry to use.

### Installing Dependencies
1. Run `poetry lock && poetry install` in the same directory as the `pyproject.toml` file. You should see poetry solving the 
   dependency tree and then installing dependencies. This also installs dev group dependencies, as specified in 
   `pyproject.toml`. Lastly you should see it installing the local package.
2. To install optional "extra" dependencies, run `poetry install -E extra_name1 -E extra_name2`.
   These extra dependencies are specified in `pyproject.toml` under `[tool.poetry.extras]`. Note that any subsequent
   `poetry install` command without `--extras` will implicitly uninstall any previously installed extras.
3. To install dependency "groups" (think labels), which may or may not be optional, use the `--with` and `--without` 
   flags for Poetry. e.g. `poetry install --with docgen` will install the dependencies for the optional group "docgen".
4. Verify that the `libera_utils` package was installed correctly by running `libera-utils --version`. This runs the
   `libera-utils` command line utility that is included in the package. 
   This can also be run with `poetry run libera-utils --version`.
5. Next, [go run the tests](testing.md).

