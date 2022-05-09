# Setting Up a Development Environment


## Managing Multiple System Python Versions with `pyenv`
Pyenv is one solution that allows you to maintain many versions 
of Python that are kept entirely siloed from the core 
system version of Python, and even from Python versions installed by other means such as Homebrew.

The Github for `pyenv` can be found here: https://github.com/pyenv/pyenv

1. Install pyenv with `brew update && brew install pyenv`.
2. Follow the instructions on the Github page to set up your shell's rc files or profile files 
   (every user tends to manage these differently).
3. Check that pyenv works by running `pyenv versions` to see a list of currently installed versions.
4. Install the latest patch of Python 3.9 by running `pyenv install 3.9.9`.
5. You can check the current version by running `pyenv global` and set the current version by running `pyenv global 3.9.9`.
6. To see a full list of all available python distributions, run `pyenv install --list`.
7. To set the python version only for the current directory, run `pyenv local 3.9.9`. This will create a file in that
   directory called `.python-version` that is used by pyenv. Inside that directory, running `python` will always resolve
   to the specified version. 
8. To set the global python back to the default system python, run `pyenv global system`. This will remove any shims
   currently in place.


## Creating a Virtual Environment
Now that you have access to whatever version of python you need, set your pyenv-provided
Python version to something recent (3.9.9 is a good option at time of writing).

1. Create a virtual environment in the `libera_sdp` directory with `python -m venv venv`. This will create
   a virtual environment named `venv`, which is included in our `.gitignore`. 
2. Activate the virtual environment by running `source venv/bin/activate`. Note: if you are running csh, you must
   run `source venv/bin/activate.csh`
3. Configure your IDE to use the virtual environment's interpreter for the project. In PyCharm, this setting is in 
   `Pycharm -> Preferences -> Project: libera_sdp -> Python Interpreter`.
4. Check that when you open a Terminal in your IDE, `which python` points to your local virtual environment, indicating
   that your virtual environment is being automatically activated. 
5. Update pip to the latest version with `pip install --upgrade pip`.


## Installing Poetry
Poetry is a command line tool that helps manage a python development environment, 
including package management, virtual environment management, and package building.

Poetry installation instructions can be found here: https://python-poetry.org/docs/#installation

Once poetry is installed, check that it works by running `poetry --version`. You should get something like 
`Poetry version 1.1.11`.


## Installing Package Dependencies
1. Ensure your virtual environment is activated by running `which python` and checking that it points to 
   the virtual environment in your repo directory (probably `libera_sdp`).
2. Run `poetry env info` and verify that Poetry is recognizing your virtual environment properly:
    ```
    Virtualenv
    Python:         3.9.9
    Implementation: CPython
    Path:           /Users/myuser/path/to/libera_sdp/venv
    Valid:          True
    ```
3. Run `poetry install` in the same directory as the `pyproject.toml` file. You should see poetry solving the 
   dependency tree and then installing dependencies. This also installs dev dependencies, as specified in 
   `pyproject.toml`. Lastly you should see it installing the local package.
4. To install "extra" dependencies, which are strictly optional, run `poetry install -E <extra_name>`. e.g. to install 
   libraries to support plotting, run `poetry install -E plotting`. 
   These extra dependencies are specified in `pyproject.toml` under `[tool.poetry.extras]`.
5. Verify that the `libera_sdp` package was installed correctly by running `sdp --version`. This runs the
   `sdp` command line utility that is included in the package. You can also directly check that the `sdp` entrypoint
   exists in `venv/bin`. This can also be run with `poetry run sdp --version`.
6. Next, go run the tests.
