"""SCENE-ID-IMAGER algorithm runner.

This subpackage contains the manifest-driven processing runner (``scene_id_imager.py``) and container
``Dockerfile`` for the SCENE-ID-IMAGER data product: the radiometer-timescale imager scene-identification product
that classifies ERBE, unfiltering, AND TRMM scenes.

Its input is the *post-year-one* (RBSP-based) FMATCH-IMAGER product, read by
:meth:`libera_utils.scene_identification.FootprintData.from_fmatch_imager_post_year_one`. The year-one ERA5-based
FMATCH-IMAGER variant does not feed this product (it has no cloud-fraction or cloud-phase source).
"""
