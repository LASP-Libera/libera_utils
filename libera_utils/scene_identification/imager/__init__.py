"""SCENE-ID-IMAGER algorithm runner.

This subpackage contains the manifest-driven processing runner (``scene_id_imager.py``) and container
``Dockerfile`` for the SCENE-ID-IMAGER data product: the radiometer-timescale imager scene-identification product
that classifies ERBE, unfiltering, AND TRMM scenes.

Its input is the FMATCH-IMAGER product, read by
:meth:`libera_utils.scene_identification.FootprintData.from_fmatch_imager`.
"""
