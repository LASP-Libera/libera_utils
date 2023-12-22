# Documentation Generation for Libera Utils

## Creating Documentation with Sphinx 
Sphinx documentation is a python library for generating documentation from docstring comments throughout a
codebase in combination with other rst or md pages separate from the generated documentation. We use it to 
generate automatic API documentation for the codebase as well as building developer documentation and user 
documentation from markdown and restructured text documents (such as this one).

You will need `make` installed on your machine in order to build sphinx docs.

### Compatible Comments and Docstrings
Docstrings in this project are expected in the [Numpy dosctring format](https://numpydoc.readthedocs.io/en/latest/format.html).

### Writing Custom Documentation Pages
If you wish to add pages that are not part of the generation from the code such as reference pages, use 
the following steps.
- Create your files and folders and add them to the project folder `doc/source/` 
- edit the `doc/source/index.rst` file to add the file you have just created to the toctree (table of contents
tree)
- If you added developer documentation, instead of editing the `index.rst` edit the `doc/source/developer-docs.rst`
document or create a new folder and associated rst document containing a toctree that includes your new file.

The folder structure is an organizational convention but the actual page structure comes from the toctree structure 
in the rst files (note that these documents could also be markdown but rst has better support for TOC listings).

**Note**: This project is configured to read and interpret both markdown (.md) and reStructured Text (.rst) documents.
The following are two basic guides to writing proper comments that will create well formatted outputs.

- [reStructured Text](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- [Markdown](https://www.markdownguide.org/cheat-sheet/)

If you have need to translate between rst and markdown, use [`pandoc`](https://pandoc.org/).

## Building HTML Documentation Locally

This should all be done while in the virtual environment that is configured for this project.
See documentation dev-environment-setup.md in the libera-sdp repository for poetry instructions.

1. Run `poetry update --with docgen`
   - This ensures that the poetry install is up-to-date, including the "docgen" dependency group.
2. Run `poetry install --with docgen`  
   - This installs all project dependencies, including the `pyproject.toml` docgen group (e.g. Sphinx, etc.)
   - This also ensures that the project itself is installed with its most recent version 
     (as defined in `pyproject.toml`)
3. Navigate to the Sphinx document source folder `cd ./doc`
4. Build the html files in the build folder using the make command `make html`
5. Open the generated `build/index.html` file to go to the documentation homepage.

## Automatic Documentation Publishing with ReadTheDocs

_ReadTheDocs.org_ is a free service (ad-supported) for developers to publish documentation for open source 
code for free. The only requirement is that the repository is publicly available to clone. You're probably reading 
this on readthedocs.io.

Our configuration file for readthedocs is located in `.readthedocs.yaml`. The Libera SDC readthedocs account 
is shared between the Libera SDC development team and is not tied to a specific developer. Configuration for 
how readthedocs decides which versions to build is configured in the readthedocs account page. 
