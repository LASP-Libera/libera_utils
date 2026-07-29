"""FMATCH-IMAGER algorithm runner.

This subpackage contains the manifest-driven processing runner (``fmatch_imager.py``) and container
``Dockerfile`` for the FMATCH-IMAGER data product: the radiometer-timescale product at RBSP Climate
Quality latency.

FMATCH-IMAGER is the only mode with two product definitions. During the first year of operation the
RBSP CLDPIX/SSF products it would otherwise use do not exist, so production substitutes ERA5
reanalysis fields; the RBSP-based definition is selected manually with ``--post-year-one`` once RBSP
data flows.
"""
