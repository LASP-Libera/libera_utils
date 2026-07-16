"""SCENE-ID-CAM-CAMTIME algorithm runner.

This subpackage contains the manifest-driven processing runner (``scene_id_cam_camtime.py``) and container
``Dockerfile`` for the SCENE-ID-CAM-CAMTIME data product: the *camera*-timescale, camera / near-real-time latency
scene-identification product. It is the sibling of the radiometer-timescale runner in ``../cam`` and reuses the same
shared runner logic (:mod:`libera_utils.scene_identification._runner`); the two differ only by their FMATCH input
product, the :class:`~libera_utils.scene_identification.FootprintData` factory used to read it, and the output product
definition / time axis.
"""
