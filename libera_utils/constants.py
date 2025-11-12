"""Constants module used throughout the libera_utils package"""

import warnings
from enum import IntEnum, StrEnum
from typing import Union

from libera_utils.aws.constants import LiberaAccountSuffix, LiberaDataBucketName


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
    ANC = "ANC"

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
            case "CAL" | "ANC":
                return f"{LiberaDataBucketName.ANCILLARY_ARCHIVE_BUCKET}"
            case _:
                raise ValueError(f"Unknown processing step {self.value}")


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

    # SPICE kernels
    # =============
    spice_az_ck = ("AZROT-CK", DataLevel.SPICE)
    spice_el_ck = ("ELSCAN-CK", DataLevel.SPICE)
    spice_jpss_ck = ("JPSS-CK", DataLevel.SPICE)
    spice_jpss_spk = ("JPSS-SPK", DataLevel.SPICE)

    # Calibration Products
    # ====================
    cal_rad = ("CAL-RAD", DataLevel.CAL)
    cal_cam = ("CAL-CAM", DataLevel.CAL)

    # L1B Products
    # ============
    l1b_rad = ("RAD-4CH", DataLevel.L1B)
    l1b_cam = ("CAM", DataLevel.L1B)

    # L2 Products
    # ===========
    # TODO: reconcile this with the Libera-ASDC ICD [LIBSDC-544]
    l2_unf = ("UNF-RAD", DataLevel.L2)  # unfiltered radiances
    l2_cf_rad = ("CF-RAD", DataLevel.L2)  # cloud fraction on the radiometer timescale
    l2_cf_cam = ("CF-CAM", DataLevel.L2)  # cloud fraction on the camera timescale
    l2_ssw_toa_osse = (
        "SSW-TOA-FLUXES-OSSE",
        DataLevel.L2,
    )  # ERBE-like and TRMM-like TOA SSW irradiance from OSSEs only
    l2_ssw_toa_erbe = ("SSW-TOA-FLUXES-ERBE", DataLevel.L2)  # ERBE-like TOA SSW irradiance
    l2_ssw_toa_trmm = ("SSW-TOA-FLUXES-TRMM", DataLevel.L2)  # TRMM-like TOA SSW irradiance
    l2_ssw_toa_rt = ("SSW-TOA-FLUXES-RT", DataLevel.L2)  # ETOA SSW irradiance from a radiative transfer model lookup
    l2_ssw_surf = ("SFC-FLUXES", DataLevel.L2)  # SSW surface flux

    # Ancillary Products
    # ==================
    # TODO[LIBSDC-544]: Add in additional expected products
    anc_adm = ("ADM", DataLevel.ANC)
    anc_scene_id = ("SCENE-ID", DataLevel.ANC)

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

    Each member is defined as a tuple: (step_name, products_list)
    - step_name: The string value used in orchestration and AWS resources
    - products_list: List of DataProductIdentifier members that this step produces

    Example:
        >>> step = ProcessingStepIdentifier.l1b_rad
        >>> str(step)  # Returns "l1b-rad"
        >>> step.products  # Returns [DataProductIdentifier.l1b_rad]
        >>> step.level  # Returns DataLevel.L1B (derived from products)

    When adding new processing steps:
        1. Add the enum member with its step name and list of produced products
        2. No need to update any lookup dictionaries - relationships are embedded!
    """

    _products: list["DataProductIdentifier"]

    def __new__(cls, value: str, products: list[DataProductIdentifier] = None):  # type: ignore
        """Create a new ProcessingStepIdentifier with embedded metadata.

        Parameters
        ----------
        value : str
            The string value for this processing step (used in orchestration)
        products : list
            List of DataProductIdentifier members that this step produces
        """
        if value != value.lower():
            raise ValueError(
                f"Invalid Processing Step ID. Processing Steps are identified by lowercase hyphenated strings. Got {value}."
            )
        obj = str.__new__(cls, value)
        obj._value_ = value
        obj._products = products or []
        return obj

    # Calibration processing steps
    cal_rad = ("cal-rad", [DataProductIdentifier.cal_rad])
    cal_cam = ("cal-cam", [DataProductIdentifier.cal_cam])

    # SPICE processing steps
    spice_azel = ("spice-azel", [DataProductIdentifier.spice_az_ck, DataProductIdentifier.spice_el_ck])
    spice_jpss = ("spice-jpss", [DataProductIdentifier.spice_jpss_ck, DataProductIdentifier.spice_jpss_spk])

    # L1B processing steps
    l1b_rad = ("l1b-rad", [DataProductIdentifier.l1b_rad])
    l1b_cam = ("l1b-cam", [DataProductIdentifier.l1b_cam])

    # SDC Intermediate Processing Steps
    # TODO: LIBSDC-544 What is this?
    int_footprint_scene_id = ("int-footprint-scene-id", [DataProductIdentifier.anc_scene_id])

    # L2 processing steps
    # Camera Cloud Fraction (CF) products
    l2_cf_rad = ("l2-cf-rad", [DataProductIdentifier.l2_cf_rad])
    l2_cf_cam = ("l2-cf-cam", [DataProductIdentifier.l2_cf_cam])

    # Unfiltered radiances
    l2_unfiltered = ("l2-unfiltered", [DataProductIdentifier.l2_unf])

    # SSW TOA fluxes
    l2_ssw_toa_osse = ("l2-ssw-toa-osse", [DataProductIdentifier.l2_ssw_toa_osse])
    l2_ssw_toa_erbe = ("l2-ssw-toa-erbe", [DataProductIdentifier.l2_ssw_toa_erbe])
    l2_ssw_toa_trmm = ("l2-ssw-toa-trmm", [DataProductIdentifier.l2_ssw_toa_trmm])
    l2_ssw_toa_rt = ("l2-ssw-toa-rt", [DataProductIdentifier.l2_ssw_toa_rt])

    # SSW surface fluxes
    l2_surface_flux = ("l2-ssw-surface-flux", [DataProductIdentifier.l2_ssw_surf])

    # ADM processing steps
    adm_binning = ("adm-binning", [DataProductIdentifier.anc_adm])

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
        """
        return f"{str(self)}-docker-repo"

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
