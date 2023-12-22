.. libera-utils documentation master file, created by
   sphinx-quickstart on Mon Jun 13 16:01:51 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Libera Utils Documentation
==========================

Version: |release|

This is the documentation space for the Libera Utils Python package. Libera Utils is a package containing modules that
are commonly used throughout the Libera Science Data Center codebase and processing algorithms. This package is
published on PyPI to support our L2 algorithm developers with standardized code for interacting with our AWS resources
and a consistent API for common tasks required of all developers.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   user-docs
   developer-docs
   api-doc/libera-utils
   changelog

Developing Libera Utils
-----------------------

Libera Utils is versioned formally and released as new features are made available and bugs are fixed. You can
find the complete release history on PyPI. Release candidate (rc) versions are also released in order for the SDC Team
to test new functionality without breaking downstream code using generous dependency specifications.

We recommend pinning major and minor release versions (e.g. 2.2) as minor releases may contain minor breaking changes.
Patch releases will be restricted to bug fixes that do not cause breaking changes to existing APIs.
