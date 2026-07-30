"""ICIE software ObsID catalog for NOM-HK trimming and calibration pipelines.

Radiometer and camera ObsID numeric values are not globally unique: the same
integer can mean different events depending on whether it appears in
``ICIE__SW_OBSID_RAD`` or ``ICIE__SW_OBSID_WFOV``. Registry keys are therefore
``(NomHkObsidSource, obsid)``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from libera_utils.constants import DataProductIdentifier

DPI = DataProductIdentifier


class NomHkObsidSource(StrEnum):
    """Which NOM-HK ObsID variable to use for run detection / trimming."""

    RAD = "ICIE__SW_OBSID_RAD"
    WFOV = "ICIE__SW_OBSID_WFOV"


class ObsIdKind(StrEnum):
    """Category of a known ObsID entry."""

    RAD_CAL = "rad_cal"
    CAM_CAL = "cam_cal"
    SCIENCE = "science"


@dataclass(frozen=True)
class ObsIdSpec:
    """One known ICIE software ObsID and its product / telemetry binding."""

    obsid: int
    source: NomHkObsidSource
    kind: ObsIdKind
    description: str
    trimmed_product: DataProductIdentifier | None
    cal_product: DataProductIdentifier | None


def _entry(
    obsid: int,
    source: NomHkObsidSource,
    kind: ObsIdKind,
    description: str,
    trimmed: DataProductIdentifier | None,
    cal: DataProductIdentifier | None,
) -> tuple[tuple[NomHkObsidSource, int], ObsIdSpec]:
    """Build a registry key/value pair."""
    return (source, obsid), ObsIdSpec(
        obsid=obsid,
        source=source,
        kind=kind,
        description=description,
        trimmed_product=trimmed,
        cal_product=cal,
    )


def _rad_cal(
    obsid: int,
    description: str,
    trimmed: DataProductIdentifier,
    cal: DataProductIdentifier,
) -> tuple[tuple[NomHkObsidSource, int], ObsIdSpec]:
    return _entry(obsid, NomHkObsidSource.RAD, ObsIdKind.RAD_CAL, description, trimmed, cal)


def _cam_cal(
    obsid: int,
    description: str,
    trimmed: DataProductIdentifier,
    cal: DataProductIdentifier,
) -> tuple[tuple[NomHkObsidSource, int], ObsIdSpec]:
    return _entry(obsid, NomHkObsidSource.WFOV, ObsIdKind.CAM_CAL, description, trimmed, cal)


def _science(
    obsid: int,
    source: NomHkObsidSource,
    description: str,
) -> tuple[tuple[NomHkObsidSource, int], ObsIdSpec]:
    return _entry(obsid, source, ObsIdKind.SCIENCE, description, None, None)


#: Sole source of truth for ObsID → CAL / TRIMMED ProductIDs and catalog metadata.
#: Keyed by (source, obsid) because RAD and WFOV namespaces overlap.
OBSID_REGISTRY: dict[tuple[NomHkObsidSource, int], ObsIdSpec] = dict(
    (
        # Radiometer calibration (ICIE__SW_OBSID_RAD)
        _rad_cal(512, "Gain and noise calibration", DPI.l1a_icie_nom_hk_gain_trimmed, DPI.cal_gain),
        _rad_cal(
            256,
            "Shortwave LED calibration at 365 nm",
            DPI.l1a_icie_nom_hk_swc_365nm_trimmed,
            DPI.cal_swc_365nm,
        ),
        _rad_cal(
            257,
            "Shortwave LED calibration at 405 nm",
            DPI.l1a_icie_nom_hk_swc_405nm_trimmed,
            DPI.cal_swc_405nm,
        ),
        _rad_cal(
            258,
            "Shortwave LED calibration at 520 nm",
            DPI.l1a_icie_nom_hk_swc_520nm_trimmed,
            DPI.cal_swc_520nm,
        ),
        _rad_cal(
            259,
            "Shortwave LED calibration at 635 nm",
            DPI.l1a_icie_nom_hk_swc_635nm_trimmed,
            DPI.cal_swc_635nm,
        ),
        _rad_cal(
            260,
            "Shortwave LED calibration at 840 nm",
            DPI.l1a_icie_nom_hk_swc_840nm_trimmed,
            DPI.cal_swc_840nm,
        ),
        _rad_cal(
            261,
            "Shortwave LED calibration at 1550 nm",
            DPI.l1a_icie_nom_hk_swc_1550nm_trimmed,
            DPI.cal_swc_1550nm,
        ),
        _rad_cal(
            320,
            "Longwave blackbody calibration temperature 1 - 310 K",
            DPI.l1a_icie_nom_hk_lwc_310k_trimmed,
            DPI.cal_lwc_310k,
        ),
        _rad_cal(
            321,
            "Longwave blackbody calibration temperature 2 - 320 K",
            DPI.l1a_icie_nom_hk_lwc_320k_trimmed,
            DPI.cal_lwc_320k,
        ),
        _rad_cal(
            322,
            "Longwave blackbody calibration temperature 3 - 330 K",
            DPI.l1a_icie_nom_hk_lwc_330k_trimmed,
            DPI.cal_lwc_330k,
        ),
        _rad_cal(
            592,
            "Longwave blackbody calibration temperature 4 - 300 K",
            DPI.l1a_icie_nom_hk_lwc_300k_trimmed,
            DPI.cal_lwc_300k,
        ),
        _rad_cal(
            593,
            "Longwave blackbody calibration temperature 5 - 305 K",
            DPI.l1a_icie_nom_hk_lwc_305k_trimmed,
            DPI.cal_lwc_305k,
        ),
        _rad_cal(
            384,
            "Solar diffuser for the Split Shortwave channel using the primary face of the solar diffuser",
            DPI.l1a_icie_nom_hk_solar_ssw_pri_trimmed,
            DPI.cal_solar_ssw_pri,
        ),
        _rad_cal(
            385,
            "Solar diffuser for the Total channel using the primary face of the solar diffuser",
            DPI.l1a_icie_nom_hk_solar_tot_pri_trimmed,
            DPI.cal_solar_tot_pri,
        ),
        _rad_cal(
            386,
            "Solar diffuser for the Longwave channel using the primary face of the solar diffuser",
            DPI.l1a_icie_nom_hk_solar_lw_pri_trimmed,
            DPI.cal_solar_lw_pri,
        ),
        _rad_cal(
            387,
            "Solar diffuser for the Shortwave channel using the primary face of the solar diffuser",
            DPI.l1a_icie_nom_hk_solar_sw_pri_trimmed,
            DPI.cal_solar_sw_pri,
        ),
        _rad_cal(
            388,
            "Solar diffuser for the Split Shortwave channel using the secondary face of the solar diffuser",
            DPI.l1a_icie_nom_hk_solar_ssw_sec_trimmed,
            DPI.cal_solar_ssw_sec,
        ),
        _rad_cal(
            389,
            "Solar diffuser for the Total channel using the secondary face of the solar diffuser",
            DPI.l1a_icie_nom_hk_solar_tot_sec_trimmed,
            DPI.cal_solar_tot_sec,
        ),
        _rad_cal(
            390,
            "Solar diffuser for the Longwave channel using the secondary face of the solar diffuser",
            DPI.l1a_icie_nom_hk_solar_lw_sec_trimmed,
            DPI.cal_solar_lw_sec,
        ),
        _rad_cal(
            391,
            "Solar diffuser for the Shortwave channel using the secondary face of the solar diffuser",
            DPI.l1a_icie_nom_hk_solar_sw_sec_trimmed,
            DPI.cal_solar_sw_sec,
        ),
        _rad_cal(
            392,
            "Solar diffuser for the Split Shortwave channel using the tertiary face of the solar diffuser",
            DPI.l1a_icie_nom_hk_solar_ssw_ter_trimmed,
            DPI.cal_solar_ssw_ter,
        ),
        _rad_cal(
            393,
            "Solar diffuser for the Total channel using the tertiary face of the solar diffuser",
            DPI.l1a_icie_nom_hk_solar_tot_ter_trimmed,
            DPI.cal_solar_tot_ter,
        ),
        _rad_cal(
            394,
            "Solar diffuser for the Longwave channel using the tertiary face of the solar diffuser",
            DPI.l1a_icie_nom_hk_solar_lw_ter_trimmed,
            DPI.cal_solar_lw_ter,
        ),
        _rad_cal(
            395,
            "Solar diffuser for the Shortwave channel using the tertiary face of the solar diffuser",
            DPI.l1a_icie_nom_hk_solar_sw_ter_trimmed,
            DPI.cal_solar_sw_ter,
        ),
        # Radiometer lunar calibration (ICIE__SW_OBSID_RAD)
        _rad_cal(
            448,
            "Lunar Calibration #1 South Pole - Monthly, Azimuth scans from 57 to 69 degrees and Elevation scans from 62.5 to 73 degrees",
            DPI.l1a_icie_nom_hk_lunar_south_pole_trimmed,
            DPI.cal_lunar_south_pole,
        ),
        _rad_cal(
            449,
            "Lunar Calibration #2 North Pole - Quarterly, Azimuth scans from -67 to -57 degrees and Elevation scans from 62.5 to 73 degrees",
            DPI.l1a_icie_nom_hk_lunar_north_pole_trimmed,
            DPI.cal_lunar_north_pole,
        ),
        _rad_cal(
            513,
            "VIIRS lunar calibration several times a year with a positive azimuth position start. Azimuth scans from 110 to -5 and back to 110 degrees",
            DPI.l1a_icie_nom_hk_viirs_lunar_pos_start_trimmed,
            DPI.cal_rad_viirs_lunar_pos_start,
        ),
        _rad_cal(
            514,
            "VIIRS lunar calibration several times a year with a negative azimuth position start. Azimuth scans from -110 to 5 and back to -110 degrees",
            DPI.l1a_icie_nom_hk_viirs_lunar_neg_start_trimmed,
            DPI.cal_rad_viirs_lunar_neg_start,
        ),
        # Camera calibration (ICIE__SW_OBSID_WFOV)
        _cam_cal(
            129,
            "Cross track mode with video - 6 minute sequence",
            DPI.l1a_icie_nom_hk_ct_video_6min_trimmed,
            DPI.cal_ct_video_6min,
        ),
        _cam_cal(
            130,
            "Cross track mode with video - 12 minute sequence",
            DPI.l1a_icie_nom_hk_ct_video_12min_trimmed,
            DPI.cal_ct_video_12min,
        ),
        _cam_cal(
            131,
            "Cross track mode with video - 18 minute sequence",
            DPI.l1a_icie_nom_hk_ct_video_18min_trimmed,
            DPI.cal_ct_video_18min,
        ),
        _cam_cal(
            133,
            "RAPS mode with video - 6 minute sequence",
            DPI.l1a_icie_nom_hk_raps_video_6min_trimmed,
            DPI.cal_raps_video_6min,
        ),
        _cam_cal(
            134,
            "RAPS mode with video - 12 minute sequence",
            DPI.l1a_icie_nom_hk_raps_video_12min_trimmed,
            DPI.cal_raps_video_12min,
        ),
        _cam_cal(
            135,
            "RAPS mode with video - 18 minute sequence",
            DPI.l1a_icie_nom_hk_raps_video_18min_trimmed,
            DPI.cal_raps_video_18min,
        ),
        _cam_cal(
            256,
            "Monthly WFOVC calibration-LED darks for dark current sampling and detector linearity/stability tracking",
            DPI.l1a_icie_nom_hk_darks_of_darks_trimmed,
            DPI.cal_darks_of_darks,
        ),
        _cam_cal(
            257,
            "Monthly WFOVC calibration-LED measurements for dark current sampling and detector linearity/stability tracking",
            DPI.l1a_icie_nom_hk_led_of_dark_trimmed,
            DPI.cal_led_of_dark,
        ),
        _cam_cal(
            258,
            "Monthly dark images at 1 ms and 12 ms integration times",
            DPI.l1a_icie_nom_hk_nominal_darks_trimmed,
            DPI.cal_nominal_darks,
        ),
        _cam_cal(
            513,
            "VIIRS lunar calibration several times a year with a positive azimuth position start. Azimuth scans from 110 to -5 and back to 110 degrees",
            DPI.l1a_icie_nom_hk_viirs_lunar_pos_start_trimmed,
            DPI.cal_wfov_viirs_lunar_pos_start,
        ),
        _cam_cal(
            514,
            "VIIRS lunar calibration several times a year with a negative azimuth position start. Azimuth scans from -110 to 5 and back to -110 degrees",
            DPI.l1a_icie_nom_hk_viirs_lunar_neg_start_trimmed,
            DPI.cal_wfov_viirs_lunar_neg_start,
        ),
        # Shared science / scan modes (both RAD and WFOV; catalog only)
        _science(128, NomHkObsidSource.RAD, "Cross Track Scan Mode"),
        _science(128, NomHkObsidSource.WFOV, "Cross Track Scan Mode"),
        _science(132, NomHkObsidSource.RAD, "RAP Scan Mode"),
        _science(132, NomHkObsidSource.WFOV, "RAP Scan Mode"),
        _science(136, NomHkObsidSource.RAD, "Along Track Scan Mode"),
        _science(136, NomHkObsidSource.WFOV, "Along Track Scan Mode"),
        _science(137, NomHkObsidSource.RAD, "Earth Target Scan Mode"),
        _science(137, NomHkObsidSource.WFOV, "Earth Target Scan Mode"),
        _science(138, NomHkObsidSource.RAD, "Geo Scan of the Libyan Desert (Arid/Meteosat)"),
        _science(138, NomHkObsidSource.WFOV, "Geo Scan of the Libyan Desert (Arid/Meteosat)"),
        _science(139, NomHkObsidSource.RAD, "Geo Scan of Papua New Guinea (Shoreline/Himawari)"),
        _science(139, NomHkObsidSource.WFOV, "Geo Scan of Papua New Guinea (Shoreline/Himawari)"),
        _science(140, NomHkObsidSource.RAD, "Geo Scan of the Pacific Ocean (Ocean/GOES West)"),
        _science(140, NomHkObsidSource.WFOV, "Geo Scan of the Pacific Ocean (Ocean/GOES West)"),
    )
)


def get_obsid_spec(source: NomHkObsidSource, obsid: int) -> ObsIdSpec:
    """Return the registry entry for ``(source, obsid)``.

    Parameters
    ----------
    source : NomHkObsidSource
        NOM-HK ObsID field that owns this ObsID namespace.
    obsid : int
        Software ObsID value.

    Returns
    -------
    ObsIdSpec
        Matching catalog entry.

    Raises
    ------
    KeyError
        If the pair is not in :data:`OBSID_REGISTRY`.
    """
    try:
        return OBSID_REGISTRY[(source, obsid)]
    except KeyError as exc:
        raise KeyError(f"Unknown ObsID {obsid} for source {source.name} ({source.value})") from exc


def iter_trim_eligible(source: NomHkObsidSource | None = None) -> Iterable[ObsIdSpec]:
    """Yield registry entries that produce TRIMMED NOM-HK products.

    Parameters
    ----------
    source : NomHkObsidSource or None
        If set, only yield entries for that NOM-HK ObsID field.

    Yields
    ------
    ObsIdSpec
        Entries with a non-null ``trimmed_product``.
    """
    for spec in OBSID_REGISTRY.values():
        if spec.trimmed_product is None:
            continue
        if source is not None and spec.source is not source:
            continue
        yield spec
