.. libera-utils documentation master file, created by
   sphinx-quickstart on Mon Jun 13 16:01:51 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Libera Utils Documentation
==========================

Version: |release|

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   user-docs
   developer-docs
   api-doc/libera-utils
   changelog

.. include:: ../../README.md
   :parser: myst_parser.sphinx_

Developing Libera Utils
-----------------------

Libera Utils is versioned formally using semantic versioning and released as new features are made
available and bugs are fixed. You can
find the complete release history on PyPI. Release candidate (rc) versions are also released in order for the SDC Team
to test new functionality without breaking downstream code using generous dependency specifications.

We recommend pinning major and minor release versions (e.g. 2.2) as minor releases may contain minor breaking changes.
Patch releases will be restricted to bug fixes that do not cause breaking changes to existing APIs.
