# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import pkg_resources

sys.path.insert(0, os.path.abspath('../../libera_utils'))

# -- Project information -----------------------------------------------------

project = 'libera_utils'
copyright = '2022, Libera SDP'
author = 'Libera SDP'

libera_utils_ver = pkg_resources.get_distribution('libera_utils').version
# The full version, including alpha/beta/rc tags
release = libera_utils_ver


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.coverage",
              "numpydoc", "autoapi.extension",
              "sphinxcontrib.confluencebuilder", "m2r2"]

# Standard Confluence Settings
confluence_publish = True
confluence_space_key = 'LIBERASDPDOC'

# (or, for Confluence Server)
confluence_server_url = 'https://lasp.colorado.edu/galaxy/'

# Optional Confluence Settings
confluence_page_hierarchy = True
# Optional Parent Page
confluence_parent_page = 'Libera Science Data Processing Documentation Home'
confluence_version_comment = f'Automatically generated from libera_utils version {libera_utils_ver}.'
confluence_sourcelink = {
    'url': 'https://lasp.colorado.edu/nucleus/projects/LIBSDC/repos/libera_utils/browse',
}
# Use when testing
#confluence_publish_dryrun = True

autoapi_type = "python"
autoapi_dirs = ['../../libera_utils']

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown"
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
#html_static_path = []
