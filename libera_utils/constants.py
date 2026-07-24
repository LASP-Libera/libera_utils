"""Constants module used throughout the libera_utils package"""

import warnings
from enum import IntEnum, StrEnum
from typing import Union

from libera_utils.aws.constants import LiberaAccountSuffix, LiberaDataBucketName

# Shared ECR repository name for radiometer ObsID-specific calibration combine steps.
# CDK should create one ECR repo with this name and attach it to each cal-* Batch job.
CAL_RAD_SHARED_ECR_NAME = "cal-rad-docker-repo"


class ManifestType(StrEnum):
    """Enumerated legal manifest type values"""

    INPUT = "INPUT"
    input = INPUT
    OUTPUT = "OUTPUT"
    output = OUTPUT


class DataLevel(StrEnum):
    """Data product level"""

    L0 = "L0"
    SPICE = "SPICE"
    CAL = "CAL"
    L1A = "L1A"
    L1B = "L1B"
    L2 = "L2"
    AUX = "AUX"

    @property
    def archive_bucket_name(self) -> str:
        """Gets the archive bucket name for the data level.

        Notes
        -----
        This does not include any account suffix, which must be added separately.
        """
        match self.value:
            case "L0":
                return f"{LiberaDataBucketName.L0_ARCHIVE_BUCKET}"
            case "L1A":
                return f"{LiberaDataBucketName.L1A_ARCHIVE_BUCKET}"
            case "L1B":
                return f"{LiberaDataBucketName.L1B_ARCHIVE_BUCKET}"
            case "L2":
                return f"{LiberaDataBucketName.L2_ARCHIVE_BUCKET}"
            case "SPICE":
                return f"{LiberaDataBucketName.SPICE_ARCHIVE_BUCKET}"
            case "AUX":
                return f"{LiberaDataBucketName.AUXILIARY_ARCHIVE_BUCKET}"
            case "CAL":
                return f"{LiberaDataBucketName.CALIBRATION_ARCHIVE_BUCKET}"
            case _:
                raise ValueError(f"Unknown data level {self.value}")


class DataProductIdentifier(StrEnum):
    """Enumeration of data product canonical IDs used in AWS resource naming.

    These IDs refer to the data products (files) themselves, NOT the processing steps (since processing steps
    may produce multiple products). The string values are the product names used in filenames.

    This enum replaces the old ProductName enum from filenaming.py to provide a single source of truth.

    Each member is defined as a tuple: (product_name, data_level)
    - product_name: The string value used in filenames and AWS resources
    - data_level: The DataLevel enum value indicating the processing level

    Example:
        >>> product = DataProductIdentifier.l1b_rad
        >>> str(product)  # Returns "RAD-4CH"
        >>> product.level  # Returns DataLevel.L1B
        >>> product.level.archive_bucket_name  # Returns "libera-l1b-data"

    When adding new products:
        1. Add the enum member with its product name and DataLevel
        2. No need to update any lookup dictionaries - metadata is embedded!
    """

    _level: DataLevel

    def __new__(cls, value: str, level: DataLevel = None):  # type: ignore
        """Create a new DataProductIdentifier with embedded metadata.

        Parameters
        ----------
        value : str
            The string value for this data product (used in filenames)
        level : DataLevel
            The processing level for this data product
        """
        if value != value.upper():
            raise ValueError(
                f"Invalid Data Product ID. Data products are identified by uppercase hyphenated strings. Got {value}."
            )
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj._level = level
        return obj

    # L0 2hr PDS Products (Binary CCSDS)
    # ==================================
    # PDS Construction Record (metadata file)
    l0_pds_cr = ("PDS-CR", DataLevel.L0)
    # PDS data files (contain CCSDS packets for a single APID)
    # NOTE: These names are derived directly from the packet names used by Libera FSW (see LiberaApid)
    # JPSS spacecraft position (attitude quaternions and ephemeris coordinates) in 1Hz packets with 1 sample per packet for 1Hz samples
    l0_jpss_sc_pos_pds = ("SC-POS-PDS", DataLevel.L0)
    # Radiometer (4 bands) sample data in 4Hz packets with 50 samples per packet for 200Hz samples
    l0_icie_rad_sample_pds = ("RAD-SAMPLE-PDS", DataLevel.L0)
    # Camera science
    l0_icie_wfov_sci_pds = ("WFOV-SCI-PDS", DataLevel.L0)
    # Azimuth and Elevation encoder sample data in 4Hz packets with 50 samples per packet for 200Hz samples
    l0_icie_axis_sample_pds = ("AXIS-SAMPLE-PDS", DataLevel.L0)
    l0_pev_sw_stat_pds = ("PEV-SW-STAT-PDS", DataLevel.L0)
    l0_pec_sw_stat_pds = ("PEC-SW-STAT-PDS", DataLevel.L0)
    l0_icie_sw_stat_pds = ("SW-STAT-PDS", DataLevel.L0)
    l0_icie_seq_hk_pds = ("SEQ-HK-PDS", DataLevel.L0)
    l0_icie_fp_hk_pds = ("FP-HK-PDS", DataLevel.L0)
    l0_icie_log_msg_pds = ("LOG-MSG-PDS", DataLevel.L0)
    # Radiometer (4 bands) stream data in 10Hz packets with 100 samples per packet for 1kHz samples
    l0_icie_rad_full_pds = ("RAD-FULL-PDS", DataLevel.L0)
    l0_icie_axis_hk_pds = ("AXIS-HK-PDS", DataLevel.L0)
    l0_icie_wfov_hk_pds = ("WFOV-HK-PDS", DataLevel.L0)
    # Internal cal radiometer used in SW cal in 10Hz packets with 100 samples per packet for 1kHz samples
    l0_icie_cal_full_pds = ("CAL-FULL-PDS", DataLevel.L0)
    # Internal cal radiometer used in SW cal in 4Hz packets with 50 samples per packet for 200Hz samples
    l0_icie_cal_sample_pds = ("CAL-SAMPLE-PDS", DataLevel.L0)
    l0_icie_wfov_resp_pds = ("WFOV-RESP-PDS", DataLevel.L0)
    l0_icie_crit_hk_pds = ("CRIT-HK-PDS", DataLevel.L0)
    l0_icie_nom_hk_pds = ("NOM-HK-PDS", DataLevel.L0)
    l0_icie_ana_hk_pds = ("ANA-HK-PDS", DataLevel.L0)
    l0_icie_temp_hk_pds = ("TEMP-HK-PDS", DataLevel.L0)

    # L1A 24hr Decoded Packet Products
    # ================================
    l1a_jpss_sc_pos_decoded = ("SC-POS-DECODED", DataLevel.L1A)
    l1a_icie_rad_sample_decoded = ("RAD-SAMPLE-DECODED", DataLevel.L1A)
    l1a_icie_wfov_sci_decoded = ("WFOV-SCI-DECODED", DataLevel.L1A)
    l1a_icie_axis_sample_decoded = ("AXIS-SAMPLE-DECODED", DataLevel.L1A)
    l1a_pev_sw_stat_decoded = ("PEV-SW-STAT-DECODED", DataLevel.L1A)
    l1a_pec_sw_stat_decoded = ("PEC-SW-STAT-DECODED", DataLevel.L1A)
    l1a_icie_sw_stat_decoded = ("SW-STAT-DECODED", DataLevel.L1A)
    l1a_icie_seq_hk_decoded = ("SEQ-HK-DECODED", DataLevel.L1A)
    l1a_icie_fp_hk_decoded = ("FP-HK-DECODED", DataLevel.L1A)
    l1a_icie_log_msg_decoded = ("LOG-MSG-DECODED", DataLevel.L1A)
    l1a_icie_rad_full_decoded = ("RAD-FULL-DECODED", DataLevel.L1A)
    l1a_icie_axis_hk_decoded = ("AXIS-HK-DECODED", DataLevel.L1A)
    l1a_icie_wfov_hk_decoded = ("WFOV-HK-DECODED", DataLevel.L1A)
    l1a_icie_cal_full_decoded = ("CAL-FULL-DECODED", DataLevel.L1A)
    l1a_icie_cal_sample_decoded = ("CAL-SAMPLE-DECODED", DataLevel.L1A)
    l1a_icie_wfov_resp_decoded = ("WFOV-RESP-DECODED", DataLevel.L1A)
    l1a_icie_crit_hk_decoded = ("CRIT-HK-DECODED", DataLevel.L1A)
    l1a_icie_nom_hk_decoded = ("NOM-HK-DECODED", DataLevel.L1A)
    l1a_icie_ana_hk_decoded = ("ANA-HK-DECODED", DataLevel.L1A)
    l1a_icie_temp_hk_decoded = ("TEMP-HK-DECODED", DataLevel.L1A)

    # L1A ObsID-trimmed NOM-HK products (daily NOM-HK subset to one calibration ObsID)
    # =============================================================================
    # Produced by L1A preprocessing (Step 1); consumed by ObsID-specific cal-combine steps.
    l1a_icie_nom_hk_gain_trimmed = ("NOM-HK-GAIN-TRIMMED", DataLevel.L1A)
    l1a_icie_nom_hk_swc_365nm_trimmed = ("NOM-HK-SWC-365NM-TRIMMED", DataLevel.L1A)
    l1a_icie_nom_hk_swc_405nm_trimmed = ("NOM-HK-SWC-405NM-TRIMMED", DataLevel.L1A)
    l1a_icie_nom_hk_swc_520nm_trimmed = ("NOM-HK-SWC-520NM-TRIMMED", DataLevel.L1A)
    l1a_icie_nom_hk_swc_635nm_trimmed = ("NOM-HK-SWC-635NM-TRIMMED", DataLevel.L1A)
    l1a_icie_nom_hk_swc_840nm_trimmed = ("NOM-HK-SWC-840NM-TRIMMED", DataLevel.L1A)
    l1a_icie_nom_hk_swc_1550nm_trimmed = ("NOM-HK-SWC-1550NM-TRIMMED", DataLevel.L1A)
    l1a_icie_nom_hk_lwc_temp1_trimmed = ("NOM-HK-LWC-TEMP1-TRIMMED", DataLevel.L1A)
    l1a_icie_nom_hk_lwc_temp2_trimmed = ("NOM-HK-LWC-TEMP2-TRIMMED", DataLevel.L1A)
    l1a_icie_nom_hk_lwc_temp3_trimmed = ("NOM-HK-LWC-TEMP3-TRIMMED", DataLevel.L1A)
    l1a_icie_nom_hk_solar_ssw_pri_trimmed = ("NOM-HK-SOLAR-SSW-PRI-TRIMMED", DataLevel.L1A)
    l1a_icie_nom_hk_solar_tot_pri_trimmed = ("NOM-HK-SOLAR-TOT-PRI-TRIMMED", DataLevel.L1A)
    l1a_icie_nom_hk_solar_lw_pri_trimmed = ("NOM-HK-SOLAR-LW-PRI-TRIMMED", DataLevel.L1A)
    l1a_icie_nom_hk_solar_sw_pri_trimmed = ("NOM-HK-SOLAR-SW-PRI-TRIMMED", DataLevel.L1A)
    l1a_icie_nom_hk_solar_ssw_sec_trimmed = ("NOM-HK-SOLAR-SSW-SEC-TRIMMED", DataLevel.L1A)
    l1a_icie_nom_hk_solar_tot_sec_trimmed = ("NOM-HK-SOLAR-TOT-SEC-TRIMMED", DataLevel.L1A)
    l1a_icie_nom_hk_solar_lw_sec_trimmed = ("NOM-HK-SOLAR-LW-SEC-TRIMMED", DataLevel.L1A)
    l1a_icie_nom_hk_solar_sw_sec_trimmed = ("NOM-HK-SOLAR-SW-SEC-TRIMMED", DataLevel.L1A)
    l1a_icie_nom_hk_solar_ssw_ter_trimmed = ("NOM-HK-SOLAR-SSW-TER-TRIMMED", DataLevel.L1A)
    l1a_icie_nom_hk_solar_tot_ter_trimmed = ("NOM-HK-SOLAR-TOT-TER-TRIMMED", DataLevel.L1A)
    l1a_icie_nom_hk_solar_lw_ter_trimmed = ("NOM-HK-SOLAR-LW-TER-TRIMMED", DataLevel.L1A)
    l1a_icie_nom_hk_solar_sw_ter_trimmed = ("NOM-HK-SOLAR-SW-TER-TRIMMED", DataLevel.L1A)
    # Camera ObsID-trimmed NOM-HK (WFOV source field)
    l1a_icie_nom_hk_ct_video_6min_trimmed = ("NOM-HK-CT-VIDEO-6MIN-TRIMMED", DataLevel.L1A)
    l1a_icie_nom_hk_ct_video_12min_trimmed = ("NOM-HK-CT-VIDEO-12MIN-TRIMMED", DataLevel.L1A)
    l1a_icie_nom_hk_ct_video_18min_trimmed = ("NOM-HK-CT-VIDEO-18MIN-TRIMMED", DataLevel.L1A)
    l1a_icie_nom_hk_raps_video_6min_trimmed = ("NOM-HK-RAPS-VIDEO-6MIN-TRIMMED", DataLevel.L1A)
    l1a_icie_nom_hk_raps_video_12min_trimmed = ("NOM-HK-RAPS-VIDEO-12MIN-TRIMMED", DataLevel.L1A)
    l1a_icie_nom_hk_raps_video_18min_trimmed = ("NOM-HK-RAPS-VIDEO-18MIN-TRIMMED", DataLevel.L1A)
    l1a_icie_nom_hk_darks_of_darks_trimmed = ("NOM-HK-DARKS-OF-DARKS-TRIMMED", DataLevel.L1A)
    l1a_icie_nom_hk_led_of_dark_trimmed = ("NOM-HK-LED-OF-DARK-TRIMMED", DataLevel.L1A)
    l1a_icie_nom_hk_nominal_darks_trimmed = ("NOM-HK-NOMINAL-DARKS-TRIMMED", DataLevel.L1A)
    # VIIRS lunar (ObsID 513 on RAD and WFOV)
    l1a_icie_nom_hk_viirs_lunar_cal_trimmed = ("NOM-HK-VIIRS-LUNAR-CAL-TRIMMED", DataLevel.L1A)
    # Radiometer lunar calibration ObsID-trimmed NOM-HK (RAD source field)
    l1a_icie_nom_hk_lunar_cal1_trimmed = ("NOM-HK-LUNAR-CAL1-TRIMMED", DataLevel.L1A)
    l1a_icie_nom_hk_lunar_cal2_trimmed = ("NOM-HK-LUNAR-CAL2-TRIMMED", DataLevel.L1A)

    # Calibration Event Products (one product per radiometer / camera calibration ObsID)
    # ==================================================================================
    # Gain and Noise Calibration (ObsID 512)
    cal_gain = ("GAIN", DataLevel.CAL)
    # Shortwave LED Calibration (ObsIDs 256-261)
    cal_swc_365nm = ("SWC-365NM", DataLevel.CAL)
    cal_swc_405nm = ("SWC-405NM", DataLevel.CAL)
    cal_swc_520nm = ("SWC-520NM", DataLevel.CAL)
    cal_swc_635nm = ("SWC-635NM", DataLevel.CAL)
    cal_swc_840nm = ("SWC-840NM", DataLevel.CAL)
    cal_swc_1550nm = ("SWC-1550NM", DataLevel.CAL)
    # Longwave Blackbody Calibration (ObsIDs 320-322)
    cal_lwc_temp1 = ("LWC-TEMP1", DataLevel.CAL)
    cal_lwc_temp2 = ("LWC-TEMP2", DataLevel.CAL)
    cal_lwc_temp3 = ("LWC-TEMP3", DataLevel.CAL)
    # Solar Diffuser Calibration — Face 1 / primary (ObsIDs 384-387)
    cal_solar_ssw_pri = ("SOLAR-SSW-PRI", DataLevel.CAL)
    cal_solar_tot_pri = ("SOLAR-TOT-PRI", DataLevel.CAL)
    cal_solar_lw_pri = ("SOLAR-LW-PRI", DataLevel.CAL)
    cal_solar_sw_pri = ("SOLAR-SW-PRI", DataLevel.CAL)
    # Solar Diffuser Calibration — Face 2 / secondary (ObsIDs 388-391)
    cal_solar_ssw_sec = ("SOLAR-SSW-SEC", DataLevel.CAL)
    cal_solar_tot_sec = ("SOLAR-TOT-SEC", DataLevel.CAL)
    cal_solar_lw_sec = ("SOLAR-LW-SEC", DataLevel.CAL)
    cal_solar_sw_sec = ("SOLAR-SW-SEC", DataLevel.CAL)
    # Solar Diffuser Calibration — Face 3 / tertiary (ObsIDs 392-395)
    cal_solar_ssw_ter = ("SOLAR-SSW-TER", DataLevel.CAL)
    cal_solar_tot_ter = ("SOLAR-TOT-TER", DataLevel.CAL)
    cal_solar_lw_ter = ("SOLAR-LW-TER", DataLevel.CAL)
    cal_solar_sw_ter = ("SOLAR-SW-TER", DataLevel.CAL)
    # Lunar Calibration (ObsIDs 448-449); cal-combine / ProcessingStepIdentifiers deferred
    cal_lunar_cal1 = ("LUNAR-CAL1", DataLevel.CAL)
    cal_lunar_cal2 = ("LUNAR-CAL2", DataLevel.CAL)
    # Camera calibration events (WFOV ObsIDs; ProcessingStepIdentifiers deferred)
    cal_ct_video_6min = ("CT-VIDEO-6MIN", DataLevel.CAL)
    cal_ct_video_12min = ("CT-VIDEO-12MIN", DataLevel.CAL)
    cal_ct_video_18min = ("CT-VIDEO-18MIN", DataLevel.CAL)
    cal_raps_video_6min = ("RAPS-VIDEO-6MIN", DataLevel.CAL)
    cal_raps_video_12min = ("RAPS-VIDEO-12MIN", DataLevel.CAL)
    cal_raps_video_18min = ("RAPS-VIDEO-18MIN", DataLevel.CAL)
    cal_darks_of_darks = ("DARKS-OF-DARKS", DataLevel.CAL)
    cal_led_of_dark = ("LED-OF-DARK", DataLevel.CAL)
    cal_nominal_darks = ("NOMINAL-DARKS", DataLevel.CAL)
    # VIIRS lunar cal (ObsID 513 on both RAD and WFOV); ProcessingStepIdentifiers deferred
    cal_viirs_lunar_cal = ("VIIRS-LUNAR-CAL", DataLevel.CAL)

    # SPICE kernels
    # =============
    spice_az_ck = ("AZROT-CK", DataLevel.SPICE)
    spice_el_ck = ("ELSCAN-CK", DataLevel.SPICE)
    spice_jpss_ck = ("JPSS-CK", DataLevel.SPICE)
    spice_jpss_spk = ("JPSS-SPK", DataLevel.SPICE)

    # L1B Products
    # ============
    l1b_rad = ("RAD-4CH", DataLevel.L1B)
    l1b_cam = ("CAM", DataLevel.L1B)

    # L2 Products using Libera camera data for cloud fraction
    # ========================================================
    l2_unf_rad_cam = ("UNF-RAD-CAM", DataLevel.L2)  # unfiltered radiances using Libera camera data for cloud fraction
    l2_cf_cam = ("CF-CAM", DataLevel.L2)  # cloud fraction on the radiometer timescale
    l2_cf_cam_camtime = ("CF-CAM-CAMTIME", DataLevel.L2)  # cloud fraction on the camera timescale
    l2_nb_bb_cam_camtime = (
        "NB-BB-CAM-CAMTIME",
        DataLevel.L2,
    )  # narrowband to broadband radiances for camera pseudo-footprints
    l2_toa_flux_cam = (
        "TOA-FLUX-CAM",
        DataLevel.L2,
    )  # ERBE-like TOA SSW irradiance using Libera camera data for cloud fraction

    # Auxiliary Products using Libera camera data for cloud fraction
    # ============================================================
    aux_fmatch_cam = ("FMATCH-CAM", DataLevel.AUX)  # Footprint matching using Libera camera data for cloud fraction
    aux_fmatch_cam_camtime = (
        "FMATCH-CAM-CAMTIME",
        DataLevel.AUX,
    )  # Footprint matching using Libera camera data for cloud fraction on the camera timescale
    aux_scene_id_cam = ("SCENE-ID-CAM", DataLevel.AUX)  # Scene IDs using Libera camera data for cloud fraction
    aux_scene_id_cam_camtime = (
        "SCENE-ID-CAM-CAMTIME",
        DataLevel.AUX,
    )  # Scene IDs using Libera camera data for cloud fraction on the camera timescale
    aux_adm_stats_cam = (
        "ADM-STATS-CAM",
        DataLevel.AUX,
    )  # ERBE-like ADM daily binned statistics using Libera camera data for cloud fraction
    aux_adm_cam = ("ADM-CAM", DataLevel.AUX)  # ERBE-like ADMs using Libera camera data for cloud fraction

    # L2 Products using RBSP + VIIRS data for cloud properties
    # ========================================================
    l2_unf_rad_imager = (
        "UNF-RAD-IMAGER",
        DataLevel.L2,
    )  # unfiltered radiances using RBSP + VIIRS imager data for cloud properties
    l2_comp_flux = (
        "COMP-FLUX",
        DataLevel.L2,
    )  # Computed surface fluxes using RBSP + VIIRS imager data for cloud properties
    l2_nb_bb_imager_camtime = (
        "NB-BB-IMAGER-CAMTIME",
        DataLevel.L2,
    )  # narrowband to broadband radiances for imager pseudo-footprints
    l2_toa_flux_imager = (
        "TOA-FLUX-IMAGER",
        DataLevel.L2,
    )  # TRMM-like + ERBE-like TOA SSW irradiance using RBSP + VIIRS imager data for cloud properties

    # Auxiliary Products using RBSP + VIIRS data for cloud properties
    # ============================================================
    aux_fmatch_imager = (
        "FMATCH-IMAGER",
        DataLevel.AUX,
    )  # Footprint matching using RBSP + VIIRS imager data for cloud properties
    aux_fmatch_imager_camtime = (
        "FMATCH-IMAGER-CAMTIME",
        DataLevel.AUX,
    )  # Footprint matching using RBSP + VIIRS imager data for cloud properties on the camera timescale
    aux_scene_id_imager = (
        "SCENE-ID-IMAGER",
        DataLevel.AUX,
    )  # Scene IDs using RBSP + VIIRS imager data for cloud properties
    aux_scene_id_imager_camtime = (
        "SCENE-ID-IMAGER-CAMTIME",
        DataLevel.AUX,
    )  # Scene IDs using RBSP + VIIRS imager data for cloud properties on the camera timescale
    aux_adm_stats_imager = (
        "ADM-STATS-IMAGER",
        DataLevel.AUX,
    )  # TRMM-like + ERBE-like ADM daily binned statistics using RBSP + VIIRS imager data for cloud properties
    aux_adm_imager = (
        "ADM-IMAGER",
        DataLevel.AUX,
    )  # TRMM-like + ERBE-like ADMs using RBSP + VIIRS imager data for cloud properties

    @property
    def product_name(self) -> str:
        """Get the name formatted for AWS resources for this data product

        The name is used to create AWS resources that are specific to the data product.
        This is an alias to the string value for compatibility.
        """
        return str(self)

    @property
    def data_level(self) -> DataLevel:
        """Get the processing level for this data product.

        Returns
        -------
        DataLevel
            The processing level of this data product
        """
        return self._level

    @property
    def associated_apid(self) -> Union["LiberaApid", None]:
        """Get the associated LiberaApid for L0 and L1A data products.

        This relies on the strict naming convention that the packet name is part of the L0 and L1A data product ID name.

        Returns
        -------
        LiberaApid | None
            The associated LiberaApid for L0 and L1A data products, or None if not applicable
        """
        if self.data_level not in (DataLevel.L0, DataLevel.L1A):
            return None

        for apid in LiberaApid:
            # Use the naming convention for L1a products and APID packet names to determine the APID associated with
            # an L1A file
            if apid.name in self.name.lower():
                return apid

        return None

    def get_partial_archive_bucket_name(self) -> str:
        """Gets the archive bucket name from the data product identifier .

        Buckets are named according to the level of data they are storing and which account they are in. This is
        expected to be used by the L2 developers who will most commonly be working with the stage account.

        Returns
        -------
        str
            The name of the archive bucket for this data product without an account suffix
        """
        warnings.warn("Use DataProductIdentifier.level.archive_bucket_name instead", DeprecationWarning)
        return self.data_level.archive_bucket_name


class ProcessingStepIdentifier(StrEnum):
    """Enumeration of processing step IDs used in AWS resource naming and processing orchestration.

    In orchestration code, these are used as "NodeID" values to identify processing steps:
        The processing_step_node_id values used in libera_cdk deployment of processing steps
        and the node names in processing_system_dag.json must match these.
    They must also be passed to the ecr_upload module called by some libera_cdk integration tests.

    The string values are the processing step names used in orchestration.

    Each member is defined as a tuple: (step_name, products_list[, shared_ecr_name])
    - step_name: The string value used in orchestration and AWS resources
    - products_list: List of DataProductIdentifier members that this step produces
    - shared_ecr_name: Optional ECR repository name shared by multiple steps (e.g. cal-rad steps)

    Example:
        >>> step = ProcessingStepIdentifier.l1b_rad
        >>> str(step)  # Returns "l1b-rad"
        >>> step.products  # Returns [DataProductIdentifier.l1b_rad]
        >>> step.level  # Returns DataLevel.L1B (derived from products)

    When adding new processing steps:
        1. Add the enum member with its step name and list of produced products
        2. Optionally pass a shared ECR name when multiple steps use one image repository
        3. No need to update any lookup dictionaries - relationships are embedded!
    """

    _products: list["DataProductIdentifier"]
    _shared_ecr_name: str | None

    def __new__(
        cls,
        value: str,
        products: list[DataProductIdentifier] | None = None,
        shared_ecr_name: str | None = None,
    ):  # type: ignore
        """Create a new ProcessingStepIdentifier with embedded metadata.

        Parameters
        ----------
        value : str
            The string value for this processing step (used in orchestration)
        products : list, optional
            List of DataProductIdentifier members that this step produces
        shared_ecr_name : str, optional
            Shared ECR repository name. When set, ``ecr_name`` returns this value
            instead of ``{step}-docker-repo``. Used so ObsID-specific cal steps
            can share one radiometer calibration image repository.
        """
        if value != value.lower():
            raise ValueError(
                f"Invalid Processing Step ID. Processing Steps are identified by lowercase hyphenated strings. Got {value}."
            )
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj._products = products or []
        obj._shared_ecr_name = shared_ecr_name
        return obj

    # SPICE processing steps
    spice_azel = ("spice-azel", [DataProductIdentifier.spice_az_ck, DataProductIdentifier.spice_el_ck])
    spice_jpss = ("spice-jpss", [DataProductIdentifier.spice_jpss_ck, DataProductIdentifier.spice_jpss_spk])

    # L1B processing steps
    l1b_rad = ("l1b-rad", [DataProductIdentifier.l1b_rad])
    l1b_cam = ("l1b-cam", [DataProductIdentifier.l1b_cam])

    # Radiometer calibration event combination steps (shared cal-rad ECR; one step per ObsID product)
    cal_gain = ("cal-gain", [DataProductIdentifier.cal_gain], CAL_RAD_SHARED_ECR_NAME)
    cal_swc_365nm = ("cal-swc-365nm", [DataProductIdentifier.cal_swc_365nm], CAL_RAD_SHARED_ECR_NAME)
    cal_swc_405nm = ("cal-swc-405nm", [DataProductIdentifier.cal_swc_405nm], CAL_RAD_SHARED_ECR_NAME)
    cal_swc_520nm = ("cal-swc-520nm", [DataProductIdentifier.cal_swc_520nm], CAL_RAD_SHARED_ECR_NAME)
    cal_swc_635nm = ("cal-swc-635nm", [DataProductIdentifier.cal_swc_635nm], CAL_RAD_SHARED_ECR_NAME)
    cal_swc_840nm = ("cal-swc-840nm", [DataProductIdentifier.cal_swc_840nm], CAL_RAD_SHARED_ECR_NAME)
    cal_swc_1550nm = ("cal-swc-1550nm", [DataProductIdentifier.cal_swc_1550nm], CAL_RAD_SHARED_ECR_NAME)
    cal_lwc_temp1 = ("cal-lwc-temp1", [DataProductIdentifier.cal_lwc_temp1], CAL_RAD_SHARED_ECR_NAME)
    cal_lwc_temp2 = ("cal-lwc-temp2", [DataProductIdentifier.cal_lwc_temp2], CAL_RAD_SHARED_ECR_NAME)
    cal_lwc_temp3 = ("cal-lwc-temp3", [DataProductIdentifier.cal_lwc_temp3], CAL_RAD_SHARED_ECR_NAME)
    cal_solar_ssw_pri = ("cal-solar-ssw-pri", [DataProductIdentifier.cal_solar_ssw_pri], CAL_RAD_SHARED_ECR_NAME)
    cal_solar_tot_pri = ("cal-solar-tot-pri", [DataProductIdentifier.cal_solar_tot_pri], CAL_RAD_SHARED_ECR_NAME)
    cal_solar_lw_pri = ("cal-solar-lw-pri", [DataProductIdentifier.cal_solar_lw_pri], CAL_RAD_SHARED_ECR_NAME)
    cal_solar_sw_pri = ("cal-solar-sw-pri", [DataProductIdentifier.cal_solar_sw_pri], CAL_RAD_SHARED_ECR_NAME)
    cal_solar_ssw_sec = ("cal-solar-ssw-sec", [DataProductIdentifier.cal_solar_ssw_sec], CAL_RAD_SHARED_ECR_NAME)
    cal_solar_tot_sec = ("cal-solar-tot-sec", [DataProductIdentifier.cal_solar_tot_sec], CAL_RAD_SHARED_ECR_NAME)
    cal_solar_lw_sec = ("cal-solar-lw-sec", [DataProductIdentifier.cal_solar_lw_sec], CAL_RAD_SHARED_ECR_NAME)
    cal_solar_sw_sec = ("cal-solar-sw-sec", [DataProductIdentifier.cal_solar_sw_sec], CAL_RAD_SHARED_ECR_NAME)
    cal_solar_ssw_ter = ("cal-solar-ssw-ter", [DataProductIdentifier.cal_solar_ssw_ter], CAL_RAD_SHARED_ECR_NAME)
    cal_solar_tot_ter = ("cal-solar-tot-ter", [DataProductIdentifier.cal_solar_tot_ter], CAL_RAD_SHARED_ECR_NAME)
    cal_solar_lw_ter = ("cal-solar-lw-ter", [DataProductIdentifier.cal_solar_lw_ter], CAL_RAD_SHARED_ECR_NAME)
    cal_solar_sw_ter = ("cal-solar-sw-ter", [DataProductIdentifier.cal_solar_sw_ter], CAL_RAD_SHARED_ECR_NAME)

    # L2 processing steps — camera cloud fraction track
    l2_unf_rad_cam = ("l2-unf-rad-cam", [DataProductIdentifier.l2_unf_rad_cam])
    l2_cf_cam = ("l2-cf-cam", [DataProductIdentifier.l2_cf_cam])
    l2_cf_cam_camtime = ("l2-cf-cam-camtime", [DataProductIdentifier.l2_cf_cam_camtime])
    l2_nb_bb_cam_camtime = ("l2-nb-bb-cam-camtime", [DataProductIdentifier.l2_nb_bb_cam_camtime])
    l2_toa_flux_cam = ("l2-toa-flux-cam", [DataProductIdentifier.l2_toa_flux_cam])

    # L2 processing steps — RBSP + VIIRS imager track
    l2_unf_rad_imager = ("l2-unf-rad-imager", [DataProductIdentifier.l2_unf_rad_imager])
    l2_comp_flux = ("l2-comp-flux", [DataProductIdentifier.l2_comp_flux])
    l2_nb_bb_imager_camtime = ("l2-nb-bb-imager-camtime", [DataProductIdentifier.l2_nb_bb_imager_camtime])
    l2_toa_flux_imager = ("l2-toa-flux-imager", [DataProductIdentifier.l2_toa_flux_imager])

    # AUX processing steps — camera cloud fraction track
    aux_fmatch_cam = ("aux-fmatch-cam", [DataProductIdentifier.aux_fmatch_cam])
    aux_fmatch_cam_camtime = ("aux-fmatch-cam-camtime", [DataProductIdentifier.aux_fmatch_cam_camtime])
    aux_scene_id_cam = ("aux-scene-id-cam", [DataProductIdentifier.aux_scene_id_cam])
    aux_scene_id_cam_camtime = ("aux-scene-id-cam-camtime", [DataProductIdentifier.aux_scene_id_cam_camtime])
    aux_adm_stats_cam = ("aux-adm-stats-cam", [DataProductIdentifier.aux_adm_stats_cam])
    aux_adm_cam = ("aux-adm-cam", [DataProductIdentifier.aux_adm_cam])

    # AUX processing steps — RBSP + VIIRS imager track
    aux_fmatch_imager = ("aux-fmatch-imager", [DataProductIdentifier.aux_fmatch_imager])
    aux_fmatch_imager_camtime = ("aux-fmatch-imager-camtime", [DataProductIdentifier.aux_fmatch_imager_camtime])
    aux_scene_id_imager = ("aux-scene-id-imager", [DataProductIdentifier.aux_scene_id_imager])
    aux_scene_id_imager_camtime = ("aux-scene-id-imager-camtime", [DataProductIdentifier.aux_scene_id_imager_camtime])
    aux_adm_stats_imager = ("aux-adm-stats-imager", [DataProductIdentifier.aux_adm_stats_imager])
    aux_adm_imager = ("aux-adm-imager", [DataProductIdentifier.aux_adm_imager])

    @property
    def processing_step_name(self) -> str:
        """Get the name formatted for AWS resources for this processing step

        The name is used to create AWS resources that are specific to the processing step.
        This is an alias to the string value for compatibility.
        """
        return str(self)

    @property
    def products(self) -> list["DataProductIdentifier"]:
        """Get the list of data products produced by this processing step.

        Returns
        -------
        list[DataProductIdentifier]
            List of data products produced by this processing step
        """
        return self._products

    @property
    def level(self) -> DataLevel:
        """Get the processing level of the products produced by this step

        Raises
        ------
        ValueError
            If the step produces no products or produces products of multiple levels
        """
        products = self.products
        if not products:
            raise ValueError(f"Processing step {self} produces no products - this is a configuration error")

        levels = {product.data_level for product in products}
        if len(levels) > 1:
            raise ValueError(f"Processing step {self} produces products of multiple levels: {levels}")
        return levels.pop()

    @property
    def step_function_name(self):
        """Get the name formatted for the step function for this processing step

        The step function name is used to create a step function that orchestrates the processing step.
        """
        return f"{str(self).replace('_', '-')}-processing-step-function"

    @property
    def policy_name(self) -> str:
        """Get the name formatted IAM policy for this processing step

        The policy name is used to create an IAM policy that grants permissions to the aspects of the processing step.
        """
        spaced = str(self).replace("-", " ").replace("-", " ").lower()
        separate = spaced.split(" ")
        capitalized = [s.capitalize() for s in separate]
        return "LiberaSDC".join(capitalized) + "DevPolicy"

    @property
    def ecr_name(self) -> str | None:
        """Get the manually-configured ECR name for this processing step

        We name our ECRs in CDK because they are one of the few resources that humans will need to interact
        with on a regular basis.

        When a step was created with ``shared_ecr_name`` (e.g. radiometer cal-combine steps),
        that shared repository name is returned so CDK can attach one ECR to many Batch jobs.
        Otherwise returns ``{step}-docker-repo``.
        """
        if self._shared_ecr_name is not None:
            return self._shared_ecr_name
        return f"{str(self)}-docker-repo"

    @property
    def l2_team_iam_role(self) -> str | None:
        """Get the L2 Team IAM role name permitted to push this algorithm's image to its ECR repo.

        L2 (and ADM) algorithms are owned by an L2 team whose IAM role grants ECR push permissions for that
        algorithm. The role name is unqualified by IAM path (e.g. ``"L2-CloudFraction"``); callers prepend the
        shared L2 developer path prefix.

        Returns
        -------
        str | None
            The L2 Team IAM role name, or None for steps not owned by an L2 team (e.g. SPICE, L1B, and SDC
            intermediate steps).
        """
        return _L2_TEAM_IAM_ROLE_BY_STEP.get(self)

    def get_archive_bucket_name(
        self, account_suffix: str | LiberaAccountSuffix = LiberaAccountSuffix.STAGE
    ) -> str | None:
        """Gets the archive bucket name for this processing step.

        Buckets are named according to the level of data they are storing and which account they are in. This is
        expected to be used by the L2 developers who will most commonly be working with the stage account.

        Parameters
        ----------
        account_suffix : str | LiberaAccountSuffix, optional
            Account suffix for the bucket name, by default LiberaAccountSuffix.STAGE (stage account).
            Can be a string like "-test" for custom testing scenarios.

        Returns
        -------
        str
            The name of the archive bucket for this processing step
        """
        level = self.level
        if level is None:
            return None
        return level.archive_bucket_name + str(account_suffix)

    @classmethod
    def from_data_product(cls, data_product: DataProductIdentifier) -> Union["ProcessingStepIdentifier", None]:  # noqa: UP007
        """Get the ProcessingStepIdentifier that produces the given DataProductIdentifier

        Parameters
        ----------
        data_product : DataProductIdentifier
            The data product to find the processing step for

        Returns
        -------
        ProcessingStepIdentifier
            The processing step that produces this data product

        Raises
        ------
        ValueError
            If no processing step is found for the data product
        """
        for step in cls:
            if data_product in step.products:
                return step
        return None


# Maps L2 (and ADM) processing steps to the name of the L2 Team IAM role permitted to push that algorithm's image
# to its ECR repo. Sibling steps share a team's role (e.g. both cloud-fraction steps -> L2-CloudFraction). Steps not
# listed here are not owned by an L2 team. Exposed via ProcessingStepIdentifier.l2_team_iam_role.
_L2_TEAM_IAM_ROLE_BY_STEP: dict[ProcessingStepIdentifier, str] = {
    ProcessingStepIdentifier.l2_cf_cam: "L2-CloudFraction",
    ProcessingStepIdentifier.l2_cf_cam_camtime: "L2-CloudFraction",
    ProcessingStepIdentifier.l2_unf_rad_cam: "L2-Unfiltering",
    ProcessingStepIdentifier.l2_unf_rad_imager: "L2-Unfiltering",
    ProcessingStepIdentifier.l2_toa_flux_cam: "L2-SSW-TOA-Flux",
    ProcessingStepIdentifier.l2_toa_flux_imager: "L2-SSW-TOA-Flux",
    ProcessingStepIdentifier.l2_comp_flux: "L2-SFC-Flux",
    ProcessingStepIdentifier.aux_adm_stats_cam: "L2-ADM",
    ProcessingStepIdentifier.aux_adm_cam: "L2-ADM",
    ProcessingStepIdentifier.l2_nb_bb_cam_camtime: "L2-ADM",
    ProcessingStepIdentifier.aux_adm_stats_imager: "L2-ADM",
    ProcessingStepIdentifier.aux_adm_imager: "L2-ADM",
    ProcessingStepIdentifier.l2_nb_bb_imager_camtime: "L2-ADM",
}


class LiberaApid(IntEnum):
    """APIDs for packets

    The enum names here should be of the form <packet-source>_<system>_<contents>.
    e.g. for icie_rad_sample: packet_source is Libera "ICIE", the system is the "Radiometers", and the contents is radiometer "Samples".

    Notes
    -----
    This is useful for identifying the data product type from the APID in an L0 filename.

    The enum names (e.g. icie_seq_hk) for Libera packets here should be precisely the
    packet names used in FSW documents and packet definitions.
    """

    # JPSS spacecraft packet, not generated by Libera
    jpss_sc_pos = 11

    # Libera packets. Enum names are uppercased versions of the FSW
    pev_sw_stat = 1000
    pec_sw_stat = 1002
    icie_sw_stat = 1013
    icie_seq_hk = 1017
    icie_fp_hk = 1019
    icie_log_msg = 1026
    icie_rad_full = 1035
    icie_rad_sample = 1036
    icie_axis_hk = 1037
    icie_wfov_hk = 1038
    icie_wfov_sci = 1040
    icie_cal_full = 1043
    icie_cal_sample = 1044
    icie_axis_sample = 1048
    icie_wfov_resp = 1049
    icie_crit_hk = 1051
    icie_nom_hk = 1057
    icie_ana_hk = 1059
    icie_temp_hk = 1060

    @property
    def data_product_id(self) -> DataProductIdentifier:
        """Get the DataProductIdentifier for L0 PDS files with this APID

        This relies on the strict naming convention that the packet name is part of the L0 data product ID name
        """
        l0_dpis = (dpi for dpi in DataProductIdentifier if dpi.data_level == DataLevel.L0)
        for dpi in l0_dpis:
            if self.name in dpi.name:
                return dpi
        raise ValueError(
            f"Unable to find PDS DataProductIdentifier associated with {self}. This may mean the DPI enum name does not match our convention."
        )
