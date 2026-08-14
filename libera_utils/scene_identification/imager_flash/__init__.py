"""SCENE-ID-IMAGER-FLASH algorithm runner.

This subpackage contains the manifest-driven processing runner (``scene_id_imager_flash.py``) and container
``Dockerfile`` for the SCENE-ID-IMAGER-FLASH data product: the radiometer-timescale imager (flash-latency)
scene-identification product.

Its input is the FMATCH-IMAGER-FLASH product, read by
:meth:`libera_utils.scene_identification.FootprintData.from_fmatch_imager_flash`. It runs ERBE, unfiltering, and
TRMM, but because FMATCH-IMAGER-FLASH carries no cloud-phase source its TRMM classification is limited to the
clear/surface scenes that do not bound cloud phase (every phase-gated cloudy TRMM scene is left unmatched).
"""
