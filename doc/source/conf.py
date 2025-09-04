"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import importlib.metadata
import os
import sys

# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath("../../libera_utils"))

# List of modules to be excluded from documentation
# This list is used by autodoc when processing autodoc-skip-member events.
# This list is also passed to the jinja template used by autosummary in order to
# skip generation of autosummary stub pages.
excluded_modules = ["libera_utils.backports"]


def skip_module(app, what, name, obj, skip, options):
    """Determine whether to document or skip a member (object, class, attribute, module, etc)"""
    if hasattr(obj, "__module__"):
        module_name = obj.__module__

        # Skip this member if it's not part of libera_utils (prevents documenting imported modules like numpy)
        if module_name and not module_name.startswith("libera_utils"):
            return True

        # Skip this member if it's part of an excluded module
        if module_name and any(module_name.startswith(excluded) for excluded in excluded_modules):
            return True

    # Skip the top level excluded modules themselves
    # if what == 'module' and obj.__name__ in excluded_modules:
    #     print(f"excluding {obj}")
    #     return True

    # Skip all dunders because users shouldn't need to know about those
    if name.startswith("__") and name.endswith("__"):
        return True


def setup(app):
    """Set up the Sphinx documentation and activate functions for specific events"""
    app.connect("autodoc-skip-member", skip_module)


# -- Project information -----------------------------------------------------
project = "libera_utils"
copyright = "2022, University of Colorado"
author = "Libera SDC Team"

libera_utils_ver = importlib.metadata.version("libera_utils")
# The full version, including alpha/beta/rc tags
release = libera_utils_ver


# -- General configuration ---------------------------------------------------
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",  # Generates API docs
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",  # Link to other projects' documentation
    "sphinx.ext.napoleon",  # Handles numpy style docstrings
    "sphinx.ext.autosectionlabel",
    "myst_parser",  # Markdown
    "numpydoc",  # Numpy style docstrings
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_logo = "_static/libera_logo.png"

# -- Autodoc -----------------------------------------------------------------
autodoc_default_options = {
    "members": True,
    # "undoc-members": True,
    "private-members": True,
}

# -- Autosummary -------------------------------------------------------------
# The autosummary template is based on the following SO answer:
# https://stackoverflow.com/questions/2701998/automatically-document-all-modules-recursively-with-sphinx-autodoc/62613202#62613202
autosummary_generate = True
autosummary_imported_members = False
autosummary_context = {"excluded_modules": excluded_modules}

# -- Warning generation ------------------------------------------------------
nitpicky = True

# Ignore certain warnings.
# Some inherited method targets aren't found through intersphinx
# NOTE: When developing, periodically turn these off to see if we are accidentally excluding warnings we care about.
nitpick_ignore_regex = [
    # TODO[LIBSDC-617][https://github.com/lasp/space_packet_parser/issues/17]:
    #  Remove this warning ignore filter when space-packet-parser implements intersphinx
    (r"py:.*", r".*space_packet_parser.*"),
    (r"py:.*", r".*libera_utils\.backports.*"),  # Since we're not documenting this module, others can't link to it
    (r"py:.*", r".*bitstring.*"),  # Bitstring library doesn't appear to support intersphinx
    (r"py:.*", r".*h5py\._hl\.files\.File.*"),  # h5py.File doesn't resolve for some reason
    (r"py:.*", r".*sqlalchemy\.orm\.decl_api\.Base.*"),  # Can't find the intersphinx for some sqlalchemy classes
    (r"py:.*", r".*numpy.float64.*"),
    # Autodoc doesn't seem to handle enums well. The following filter out some known issues.
    (r"py:.*", r".*ManifestType\.[input|output|INPUT|OUTPUT].*"),
    (r"py:.*", r".*libera_utils\.spice_utils\.Spice[Body|Frame].*"),
    (r"py:.*", r".*libera_utils\.io\.filenaming\.DataLevel\.[L0|L1B|L2].*"),
    (r"py:.*", r".*APID\..*"),
    (r"py:.*", r".*IntEnum.*"),
]

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "pytest": ("https://pytest.org/en/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "cloudpathlib": ("https://cloudpathlib.drivendata.org/stable/", None),
    # TODO[LIBSDC-617][https://github.com/lasp/space_packet_parser/issues/17]:
    #  This is left commented out until space-packet-parser implements intersphinx support
    # "space_packet_parser": ("https://spacepacket-parser.readthedocs.io/en/latest/", None),
    "sqlalchemy": ("https://docs.sqlalchemy.org/en/14/", None),
    "h5py": (" https://docs.h5py.org/en/stable", None),
}
