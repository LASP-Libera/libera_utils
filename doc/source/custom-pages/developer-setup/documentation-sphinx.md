# Sphinx Autodocumentation for LASP Libera

## Creating Documentation with Sphinx 
Sphinx documentation is a method of generating documentation from docstring comments throughout a
codebase in combination with other custom pages separate from the generated documentation.

### Compatible Comments and Docstrings
Docstrings in this project are expected in the [Numpy dosctring format](https://numpydoc.readthedocs.io/en/latest/format.html).

### Adding custom pages
If you wish to add pages that are not part of the generation from the code such as reference pages use 
the following steps.
- Create your files and folders and add them to the project folder _doc/source/custom-pages_ 
- edit the _doc/source/index.rst_ file to add the file you have just created to the toctree (table of contents
tree)
- If you added developer documentation, instead of editing the _index.rst_ edit the _doc/source/dev-env-docs.rst_ document
or _doc/source/dev-ref-docs.rst_ to include your new file.

**Note**: This project is configured to read and interpret both markdown (.md) and reStructured Text (.rst) documents.
The following are two basic guides to writing proper comments that will create well formatted outputs.

- [reStructured Text](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)
- [Markdown](https://www.markdownguide.org/cheat-sheet/)


## Local Testing and Development

### Step-by-Step Installation Guide to build HTML Documentation
This should all be done while in the virtual environment that is configured for this project.
See documentation dev-environment-setup.md in the libera-sdp repository for poetry instructions.

1. Run `poetry update --with docgen`
   - This ensures that the poetry install is up-to-date
2. Run `poetry install --with docgen`  
   - This installs the packages defined in pyproject.toml group as "docgen" to install Sphinx and dependencies in the environment
3. Navigate to the Sphinx document source folder `cd ./doc`
4. Build the html files in the build folder using the make command `make html`

### Building Directly to Confluence
This project is configured to use an Atlassian Personal Access Tokens (PAT) for authentication. To set up
one of these, follow instructions [from Atlassian.](https://confluence.atlassian.com/enterprise/using-personal-access-tokens-1026032365.html)
Once you have your PAT then you need to set an environment variable to be read during the make process.
- `export CONFLUENCE_PUBLISH_TOKEN='your-pat-here'`

If you have built this documentation as html and looked it over already, then you can uncomment the
following line in _conf.py_ to check your confluence credentials.

` confluence_publish_dryrun = True `

To build the documentation (whether publishing to Confluence or testing the credentials) navigate to the
_doc/source_ folder then run:

`make confluence`

**Note:** The general confluence configuration details are set in conf.py and a range of possibilities are
available in the [Sphinx Confluence Builder documentation](https://sphinxcontrib-confluencebuilder.readthedocs.io/en/stable/contents/).

## Running with Docker

### Building html completely
To build from scratch and ensure all css and additional files are overwritten, use the following commands.
```
docker-compose build  --no-cache docs-html
docker-compose up docs-html
```
To investigate what files were generated use the docker copy function.

`docker-compose cp docs-html:/opt/libera/doc/build/ ./local/path`

### Updating html
For small changes to the documentation that don't need significant formatting changes you can run:
```
docker-compose up docs-html
```

## Building Confluence documentation in Docker
```
docker-compose build --no-cache docs-confluence
docker-compose run -e CONFLUENCE_PUBLISH_TOKEN="your-pat-here" docs-confluence
```