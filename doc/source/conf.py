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
    # NOTE: numpydoc is deliberately NOT enabled. napoleon (above) already parses numpy style
    # docstrings, and running both is a long standing conflict (sphinx-doc/sphinx#1384). Worse,
    # numpydoc's autodoc-process-signature hook crashes Sphinx 9's rewritten autodoc on members
    # inherited from builtins, e.g. the str methods a StrEnum inherits (sphinx-doc/sphinx#14576),
    # which autosummary then reports as "failed to import object". numpydoc also emitted a second,
    # noisier copy of every class Methods table. Between them these accounted for roughly 7700 of
    # the ~9800 warnings this build used to produce. If we want numpydoc's docstring linting back,
    # numpydoc_validation_checks is available as a standalone pre-commit hook.
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# Prefix autosectionlabel targets with the document name so that two pages may both have a section
# with the same title (e.g. "Basic Usage") without colliding.
autosectionlabel_prefix_document = True

# Render a docstring "Attributes" section as an :ivar: field list rather than as standalone
# .. attribute:: directives. Several classes (the scene_id enums in particular) document their
# members in an Attributes section, which autodoc also documents from the source; emitting both as
# object descriptions makes them "duplicate object description".
napoleon_use_ivar = True

# Generate anchors for markdown headings down to <h3> so that in-page links like
# [Consumer 2](#consumer-2-downstream-cal-combine-dispatch) resolve.
myst_heading_anchors = 3

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
    # Enum members and pydantic model fields carry no docstrings of their own, so without this
    # autodoc skips them while the autosummary class template still lists them, producing a
    # "reference target not found" for every one. Documenting them is both more useful and quieter.
    "undoc-members": True,
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
# NOTE: nitpick_ignore_regex matches with re.fullmatch against BOTH the role and the target, so
# every pattern here has to describe the whole target, not a fragment. Beware of writing [a|b]
# where (a|b) was meant: the former is a character class and silently over-matches.
nitpick_ignore_regex = [
    (r"py:.*", r".*libera_utils\.backports.*"),  # Since we're not documenting this module, others can't link to it
    (r"py:.*", r".*bitstring.*"),  # Bitstring library doesn't appear to support intersphinx
    # pydantic Field constraint metadata leaking out of Annotated[...] (sphinx-doc/sphinx#12601).
    # Sphinx stringifies each Annotated metadata item and cross-references the callee of the
    # resulting ast.Call, so these render as dangling references by design.
    (r"py:.*", r"annotated_types\..*"),
    (r"py:class", r"(FieldInfo|MinLen|MaxLen|Ge|Gt|Le|Lt|MultipleOf|NoneType)"),
    # numpy's inventory registers its typing aliases as py:data, but autodoc emits signature
    # references as py:class, so nitpicky can never resolve them (sphinx-doc/sphinx#14005).
    (r"py:class", r"numpy\._?typing\..*"),
    (r"py:class", r"pandas\.core\..*"),  # pandas documents these under their public re-export path
    # Abbreviations and bare names used in numpy style docstring type strings. napoleon turns each
    # comma separated element of a "name : type" line into its own cross-reference, so these are
    # prose, not real targets.
    (r"py:class", r"(np|npt|pd|xr)\..*"),
    (r"py:class", r"(optional|Path|PathLike|PathType|S3Path|CloudPath|Dataset|DataArray|ndarray|datetime|ULID|str)"),
    (r"py:class", r"(AngleLike|ulid\.ULID|boto3\.Session|filenaming\.PathType|LiberaDataProductFilename)"),
    # pydantic internals that surface in model signatures and validator descriptors.
    (r"py:class", r"(PydanticUndefined|FieldValidatorDecoratorInfo)"),
    (r"py:class", r"pydantic\._internal\..*"),
    # netCDF4 publishes no intersphinx inventory.
    (r"py:class", r"netCDF4(\._netCDF4)?\.Dataset"),
    # Module private helpers that appear inside Annotated[...] validators or as TypeVars, neither of
    # which autodoc gives a documentable target.
    (r"py:class", r"libera_utils\.io\.umm_g\.validate_iso_datetime"),
    (r"py:class", r"libera_utils\.libera_spice\.spice_utils\._F"),
    # Third party exceptions whose projects publish no inventory (or none we subscribe to).
    (r"py:exc", r"(requests\.exceptions|docker\.errors|boto3\.exceptions)\..*"),
    # Artifact, not a real reference: Sphinx splits a pydantic model's generated __signature__ on
    # the comma inside a subscripted annotation, so "attributes: dict[str, Any]" is rendered as two
    # parameters and the fragment "dict[str" is looked up as a class.
    (r"py:class", r"dict\[str"),
]

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "pytest": ("https://pytest.org/en/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "cloudpathlib": ("https://cloudpathlib.drivendata.org/stable/", None),
    "space_packet_parser": ("https://space-packet-parser.readthedocs.io/en/stable/", None),
    "sqlalchemy": ("https://docs.sqlalchemy.org/en/14/", None),
    "h5py": ("https://docs.h5py.org/en/stable", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
    "boto3": ("https://boto3.amazonaws.com/v1/documentation/api/latest/", None),
}
