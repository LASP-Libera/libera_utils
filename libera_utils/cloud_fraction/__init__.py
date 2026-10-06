"""Libera camera cloud-fraction (CF-CAM) algorithm family.

This package produces the two Camera Cloud Fraction products that feed the FMATCH-CAM algorithms:

* ``CF-CAM`` -- radiometer timescale, from an L1B RAD-4CH file, one record per ``RADIOMETER_TIME``
  footprint (mirrors FMATCH-CAM);
* ``CF-CAM-CAMTIME`` -- camera timescale, from an L1B CAM file segmented into pseudo-footprints on
  the ``(CAMERA_TIME, PSEUDOFOOTPRINT)`` grid (mirrors FMATCH-CAM-CAMTIME).

Both reuse the camera segmentation / camtime-grid assembly from
:mod:`libera_utils.footprint_matching` so their record axes and pixel provenance match the FMATCH
products record-for-record.

The cloud-fraction values are currently a placeholder (see
:mod:`libera_utils.cloud_fraction.cloud_fraction`), pending the real Libera WFOV cloud-fraction
retrieval.
"""

from libera_utils.cloud_fraction.cloud_fraction import (
    generate_placeholder_cloud_fraction_camtime,
    generate_placeholder_cloud_fraction_radiometer,
)

__all__ = [
    "generate_placeholder_cloud_fraction_camtime",
    "generate_placeholder_cloud_fraction_radiometer",
]
