# Sphinx Autodocumentation

## Step-by-Step Installation Guide
This should all be done while in the virtual environment that is configured for this project.
See documentation dev-environment-setup.md in the libera-sdp repository for poetry instructions.

1. Run `poetry update`
   - This ensures that the poetry install is up-to-date
2. Run `poetry install`  
   - This installs the packages defined in pyproject.toml extras as "docs" to install Sphinx and dependencies in the environment
3. Navigate to the Sphinx document source folder `cd ./doc/source`
4. Build the html files in the build folder `sphinx-build -b html . ../build`

## Building Directly to Confluence
The confluence configuration is set in conf.py.

Change the `confluence_server_user` to your username.
If you have built this documentation and looked it over already then comment out 
` confluence_publish_dryrun = True `

Next, following steps 1-3 in the Step-by-Step guide then run `make confluence` to publish to LASP Galaxy

# Writing Sphinx Compatible Comments and Docstrings
This project is configured to read and interpret both markdown *.md and reStructured Text *.rst documents.

The following are two basic guides to writing proper comments that will create well formatted outputs.

* [reStructured Text](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
* [Markdown](https://www.markdownguide.org/cheat-sheet/)
