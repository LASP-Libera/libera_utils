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
    name: str
    description: str
    trimmed_product: DataProductIdentifier | None
    cal_product: DataProductIdentifier | None


def _entry(
    obsid: int,
    source: NomHkObsidSource,
    kind: ObsIdKind,
    name: str,
    description: str,
    trimmed: DataProductIdentifier | None,
    cal: DataProductIdentifier | None,
) -> tuple[tuple[NomHkObsidSource, int], ObsIdSpec]:
    """Build a registry key/value pair."""
    return (source, obsid), ObsIdSpec(
        obsid=obsid,
        source=source,
        kind=kind,
        name=name,
        description=description,
        trimmed_product=trimmed,
        cal_product=cal,
    )


def _rad_cal(
    obsid: int,
    name: str,
    description: str,
    trimmed: DataProductIdentifier,
    cal: DataProductIdentifier,
) -> tuple[tuple[NomHkObsidSource, int], ObsIdSpec]:
    return _entry(obsid, NomHkObsidSource.RAD, ObsIdKind.RAD_CAL, name, description, trimmed, cal)


def _cam_cal(
    obsid: int,
    name: str,
    description: str,
    trimmed: DataProductIdentifier,
    cal: DataProductIdentifier,
) -> tuple[tuple[NomHkObsidSource, int], ObsIdSpec]:
    return _entry(obsid, NomHkObsidSource.WFOV, ObsIdKind.CAM_CAL, name, description, trimmed, cal)


def _science(
    obsid: int,
    source: NomHkObsidSource,
    name: str,
    description: str,
) -> tuple[tuple[NomHkObsidSource, int], ObsIdSpec]:
    return _entry(obsid, source, ObsIdKind.SCIENCE, name, description, None, None)


#: Sole source of truth for ObsID → CAL / TRIMMED ProductIDs and catalog metadata.
#: Keyed by (source, obsid) because RAD and WFOV namespaces overlap.
OBSID_REGISTRY: dict[tuple[NomHkObsidSource, int], ObsIdSpec] = dict(
    (
        # Radiometer calibration (ICIE__SW_OBSID_RAD)
        _rad_cal(512, "Gain", "Gain and noise calibration", DPI.l1a_icie_nom_hk_gain_trimmed, DPI.cal_gain),
        _rad_cal(
            256,
            "SWC 365nm",
            "Shortwave LED calibration at 365 nm",
            DPI.l1a_icie_nom_hk_swc_365nm_trimmed,
            DPI.cal_swc_365nm,
        ),
        _rad_cal(
            257,
            "SWC 405nm",
            "Shortwave LED calibration at 405 nm",
            DPI.l1a_icie_nom_hk_swc_405nm_trimmed,
            DPI.cal_swc_405nm,
        ),
        _rad_cal(
            258,
            "SWC 520nm",
            "Shortwave LED calibration at 520 nm",
            DPI.l1a_icie_nom_hk_swc_520nm_trimmed,
            DPI.cal_swc_520nm,
        ),
        _rad_cal(
            259,
            "SWC 635nm",
            "Shortwave LED calibration at 635 nm",
            DPI.l1a_icie_nom_hk_swc_635nm_trimmed,
            DPI.cal_swc_635nm,
        ),
        _rad_cal(
            260,
            "SWC 840nm",
            "Shortwave LED calibration at 840 nm",
            DPI.l1a_icie_nom_hk_swc_840nm_trimmed,
            DPI.cal_swc_840nm,
        ),
        _rad_cal(
            261,
            "SWC 1550nm",
            "Shortwave LED calibration at 1550 nm",
            DPI.l1a_icie_nom_hk_swc_1550nm_trimmed,
            DPI.cal_swc_1550nm,
        ),
        _rad_cal(
            320,
            "LWC Temp1",
            "Longwave blackbody calibration temperature 1",
            DPI.l1a_icie_nom_hk_lwc_temp1_trimmed,
            DPI.cal_lwc_temp1,
        ),
        _rad_cal(
            321,
            "LWC Temp2",
            "Longwave blackbody calibration temperature 2",
            DPI.l1a_icie_nom_hk_lwc_temp2_trimmed,
            DPI.cal_lwc_temp2,
        ),
        _rad_cal(
            322,
            "LWC Temp3",
            "Longwave blackbody calibration temperature 3",
            DPI.l1a_icie_nom_hk_lwc_temp3_trimmed,
            DPI.cal_lwc_temp3,
        ),
        _rad_cal(
            384,
            "Solar SSW PRI",
            "Solar diffuser SSW primary face",
            DPI.l1a_icie_nom_hk_solar_ssw_pri_trimmed,
            DPI.cal_solar_ssw_pri,
        ),
        _rad_cal(
            385,
            "Solar TOT PRI",
            "Solar diffuser TOT primary face",
            DPI.l1a_icie_nom_hk_solar_tot_pri_trimmed,
            DPI.cal_solar_tot_pri,
        ),
        _rad_cal(
            386,
            "Solar LW PRI",
            "Solar diffuser LW primary face",
            DPI.l1a_icie_nom_hk_solar_lw_pri_trimmed,
            DPI.cal_solar_lw_pri,
        ),
        _rad_cal(
            387,
            "Solar SW PRI",
            "Solar diffuser SW primary face",
            DPI.l1a_icie_nom_hk_solar_sw_pri_trimmed,
            DPI.cal_solar_sw_pri,
        ),
        _rad_cal(
            388,
            "Solar SSW SEC",
            "Solar diffuser SSW secondary face",
            DPI.l1a_icie_nom_hk_solar_ssw_sec_trimmed,
            DPI.cal_solar_ssw_sec,
        ),
        _rad_cal(
            389,
            "Solar TOT SEC",
            "Solar diffuser TOT secondary face",
            DPI.l1a_icie_nom_hk_solar_tot_sec_trimmed,
            DPI.cal_solar_tot_sec,
        ),
        _rad_cal(
            390,
            "Solar LW SEC",
            "Solar diffuser LW secondary face",
            DPI.l1a_icie_nom_hk_solar_lw_sec_trimmed,
            DPI.cal_solar_lw_sec,
        ),
        _rad_cal(
            391,
            "Solar SW SEC",
            "Solar diffuser SW secondary face",
            DPI.l1a_icie_nom_hk_solar_sw_sec_trimmed,
            DPI.cal_solar_sw_sec,
        ),
        _rad_cal(
            392,
            "Solar SSW TER",
            "Solar diffuser SSW tertiary face",
            DPI.l1a_icie_nom_hk_solar_ssw_ter_trimmed,
            DPI.cal_solar_ssw_ter,
        ),
        _rad_cal(
            393,
            "Solar TOT TER",
            "Solar diffuser TOT tertiary face",
            DPI.l1a_icie_nom_hk_solar_tot_ter_trimmed,
            DPI.cal_solar_tot_ter,
        ),
        _rad_cal(
            394,
            "Solar LW TER",
            "Solar diffuser LW tertiary face",
            DPI.l1a_icie_nom_hk_solar_lw_ter_trimmed,
            DPI.cal_solar_lw_ter,
        ),
        _rad_cal(
            395,
            "Solar SW TER",
            "Solar diffuser SW tertiary face",
            DPI.l1a_icie_nom_hk_solar_sw_ter_trimmed,
            DPI.cal_solar_sw_ter,
        ),
        # Radiometer lunar calibration (ICIE__SW_OBSID_RAD)
        _rad_cal(
            448,
            "LUNAR CAL1",
            "Lunar Calibration #1 - Monthly, Azimuth scans from 57 to 69 degrees and Elevation scans from 62.5 to 73 degrees",
            DPI.l1a_icie_nom_hk_lunar_cal1_trimmed,
            DPI.cal_lunar_cal1,
        ),
        _rad_cal(
            449,
            "LUNAR CAL2",
            "Lunar Calibration #2 - Quarterly, Azimuth scans from -67 to -57 degrees and Elevation scans from 62.5 to 73 degrees",
            DPI.l1a_icie_nom_hk_lunar_cal2_trimmed,
            DPI.cal_lunar_cal2,
        ),
        _rad_cal(
            513,
            "VIIRS LUNAR CAL",
            "VIIRS lunar calibration several times a year",
            DPI.l1a_icie_nom_hk_viirs_lunar_cal_trimmed,
            DPI.cal_viirs_lunar_cal,
        ),
        # Camera calibration (ICIE__SW_OBSID_WFOV)
        _cam_cal(
            129,
            "CT Video 6 Min",
            "CT video 6 minute calibration",
            DPI.l1a_icie_nom_hk_ct_video_6min_trimmed,
            DPI.cal_ct_video_6min,
        ),
        _cam_cal(
            130,
            "CT Video 12 Min",
            "CT video 12 minute calibration",
            DPI.l1a_icie_nom_hk_ct_video_12min_trimmed,
            DPI.cal_ct_video_12min,
        ),
        _cam_cal(
            131,
            "CT Video 18 Min",
            "CT video 18 minute calibration",
            DPI.l1a_icie_nom_hk_ct_video_18min_trimmed,
            DPI.cal_ct_video_18min,
        ),
        _cam_cal(
            133,
            "RAPS Video 6 Min",
            "RAPS video 6 minute calibration",
            DPI.l1a_icie_nom_hk_raps_video_6min_trimmed,
            DPI.cal_raps_video_6min,
        ),
        _cam_cal(
            134,
            "RAPS Video 12 Min",
            "RAPS video 12 minute calibration",
            DPI.l1a_icie_nom_hk_raps_video_12min_trimmed,
            DPI.cal_raps_video_12min,
        ),
        _cam_cal(
            135,
            "RAPS Video 18 Min",
            "RAPS video 18 minute calibration",
            DPI.l1a_icie_nom_hk_raps_video_18min_trimmed,
            DPI.cal_raps_video_18min,
        ),
        _cam_cal(
            256,
            "Darks of Dark/LED",
            "Monthly WFOVC calibration-LED darks for dark current sampling and detector linearity/stability tracking",
            DPI.l1a_icie_nom_hk_darks_of_darks_trimmed,
            DPI.cal_darks_of_darks,
        ),
        _cam_cal(
            257,
            "LED of Dark/LED",
            "Monthly WFOVC calibration-LED measurements for dark current sampling and detector linearity/stability tracking",
            DPI.l1a_icie_nom_hk_led_of_dark_trimmed,
            DPI.cal_led_of_dark,
        ),
        _cam_cal(
            258,
            "Nominal Darks",
            "Monthly dark images at 1 ms and 12 ms integration times",
            DPI.l1a_icie_nom_hk_nominal_darks_trimmed,
            DPI.cal_nominal_darks,
        ),
        _cam_cal(
            513,
            "VIIRS Lunar Cal",
            "VIIRS lunar calibration",
            DPI.l1a_icie_nom_hk_viirs_lunar_cal_trimmed,
            DPI.cal_viirs_lunar_cal,
        ),
        # Shared science / scan modes (both RAD and WFOV; catalog only)
        _science(128, NomHkObsidSource.RAD, "Cross Track", "Cross Track Scan Mode"),
        _science(128, NomHkObsidSource.WFOV, "Cross Track", "Cross Track Scan Mode"),
        _science(132, NomHkObsidSource.RAD, "RAP Scan", "RAP Scan Mode"),
        _science(132, NomHkObsidSource.WFOV, "RAP Scan", "RAP Scan Mode"),
        _science(136, NomHkObsidSource.RAD, "Along Track", "Along Track Scan Mode"),
        _science(136, NomHkObsidSource.WFOV, "Along Track", "Along Track Scan Mode"),
        _science(137, NomHkObsidSource.RAD, "Earth Target", "Earth Target Scan Mode"),
        _science(137, NomHkObsidSource.WFOV, "Earth Target", "Earth Target Scan Mode"),
        _science(138, NomHkObsidSource.RAD, "Geo Scan - Arid/Meteosat", "Geo Scan of the Libyan Desert"),
        _science(138, NomHkObsidSource.WFOV, "Geo Scan - Arid/Meteosat", "Geo Scan of the Libyan Desert"),
        _science(139, NomHkObsidSource.RAD, "Geo Scan - Shoreline/Himawari", "Geo Scan of Papua New Guinea"),
        _science(139, NomHkObsidSource.WFOV, "Geo Scan - Shoreline/Himawari", "Geo Scan of Papua New Guinea"),
        _science(140, NomHkObsidSource.RAD, "Geo Scan - Ocean/GOES West", "Geo Scan of the Pacific Ocean"),
        _science(140, NomHkObsidSource.WFOV, "Geo Scan - Ocean/GOES West", "Geo Scan of the Pacific Ocean"),
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
