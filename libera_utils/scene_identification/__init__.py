"""Scene identification package.

This package groups all scene-ID and scene-definition code for the Libera Science Data Center. It classifies
radiometer footprints into discrete "scene IDs" that characterize the atmospheric and surface conditions each
footprint observed.

Modules
-------
scene_id
    The :class:`FootprintData` container plus the derived-variable calculations and extraction logic.
scene_definitions
    :class:`Scene` and :class:`SceneDefinition` — the CSV-backed classification rules (TRMM, ERBE, custom).

The most commonly used symbols are re-exported here so that callers (for example the SCENE-ID product runner in
the ``cam/`` runner) can simply do ``from libera_utils.scene_identification import FootprintData``. Code that
needs the internal helpers (e.g. tests) should import them from the specific submodule.
"""

from libera_utils.scene_identification.scene_definitions import Scene, SceneDefinition
from libera_utils.scene_identification.scene_id import (
    RADIOMETER_TIME_DIMENSION,
    RADIOMETER_TIME_VARIABLE,
    FootprintData,
    FootprintVariables,
)

__all__ = [
    "RADIOMETER_TIME_DIMENSION",
    "RADIOMETER_TIME_VARIABLE",
    "FootprintData",
    "FootprintVariables",
    "Scene",
    "SceneDefinition",
]
