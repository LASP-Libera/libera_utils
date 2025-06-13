"""AWS ECR Repository/Algorithm names"""

from enum import Enum, StrEnum


class ManifestType(StrEnum):
    """Enumerated legal manifest type values"""

    INPUT = "INPUT"
    input = INPUT
    OUTPUT = "OUTPUT"
    output = OUTPUT


class LiberaAccountSuffix(StrEnum):
    """Suffixes for the various account types"""

    STAGE = "-stage"
    PROD = "-prod"
    DEV = "-dev"
    TEST = "-test"


class DataProductIdentifier(StrEnum):
    """Enumeration of data product canonical IDs used in AWS resource naming
    These IDs refer to the data products (files) themselves, NOT the processing steps (since processing steps
    may produce multiple products).

    In general these names are of the form <level>-<source>-<type>
    """

    # L0 construction record
    # TODO: Re-evaluate these names to be caps to match other products [LIBSDC-445]: L0 Decoding step
    l0_cr = "l0-cr"

    # L0 PDS files
    l0_rad_pds = "l0-rad-pds"
    l0_cam_pds = "l0-cam-pds"
    l0_azel_pds = "l0-azel-pds"
    l0_jpss_pds = "l0-jpss-pds"

    # TODO: add other 24h L0 product ID [LIBSDC-445]: L0 Decoding step
    l0_rad_24h = "L0_RAD"

    # SPICE kernels
    spice_az_ck = "AZROT-CK"
    spice_el_ck = "ELSCAN-CK"
    spice_jpss_ck = "JPSS-CK"
    spice_jpss_spk = "JPSS-SPK"

    # Calibration products
    cal_rad = "cal-rad"
    cal_cam = "cal-cam"

    # L1B products
    l1b_rad = "L1B_RAD-4CH"
    l1b_cam = "L1B_CAM"

    # L2 products
    # TODO: reconcile this with the Libera-ASDC ICD [LIBSDC-544]
    l2_unf = "L2_UNF-RAD"  # unfiltered radiances
    l2_cf_rad = "L2_CF-RAD"  # cloud fraction on the radiometer timescale
    l2_cf_cam = "L2_CF-CAM"  # cloud fraction on the camera timescale
    l2_ssw_toa_osse = "L2_SSW-TOA-FLUXES-OSSE"  # ERBE-like and TRMM-like TOA SSW irradiance from OSSEs only
    l2_ssw_toa_erbe = "L2_SSW-TOA-FLUXES-ERBE"  # ERBE-like TOA SSW irradiance
    l2_ssw_toa_trmm = "L2_SSW-TOA-FLUXES-TRMM"  # TRMM-like TOA SSW irradiance
    l2_ssw_toa_rt = "L2_SSW-TOA-FLUXES-RT"  # ETOA SSW irradiance from a radiative transfer model lookup
    l2_ssw_surf = "L2_SFC-FLUXES"  # SSW surface flux

    # Ancillary products
    # TODO: decide whether to categorize these Auxiliary or Ancillary. Add in current expected products
    anc_adm = "anc-adm"

    @classmethod
    def validate(cls, product_name: str) -> tuple["DataProductIdentifier", int | None]:
        """Validate a product name string used by the DAG or the processing orchestration system.

        If successful, returns a tuple containing the DataProductIdentifier and the chunk_number,
        which can be None if the input string does not contain a valid chunk number.
        """
        if (idx := product_name.rfind("-")) > 0:
            # the dash could be internal to the enum name, check that
            try:
                # check against the snake-case value
                product_id = DataProductIdentifier(product_name[:idx])
                # the int conversion could also fail with ValueError
                return product_id, int(product_name[idx + 1 :])
            except ValueError:
                # assume that there is no chunk number, fall through to check that
                pass
        return DataProductIdentifier(product_name), None

    def to_str_with_chunk_number(self, chunk_number: int | None = None) -> str:
        """Convert the DataProductIdentifier to a string suitable for matching
        with a DAG key or in the processing orchestration system.

        The chunk_number can be specified when the data product represents
        a PDS file that is typically provided in 12 2-hour chunks per day.
        In that case, the chunk_number appears as a suffix to the orchestration
        product name.
        """
        return f"{self}-{chunk_number}" if chunk_number is not None else self


class ProcessingStepIdentifier(StrEnum):
    """Enumeration of processing step IDs used in AWS resource naming and processing orchestration

    In orchestration code, these are used as "NodeID" values to identify processing steps:
        The processing_step_node_id values used in libera_cdk deployment stackbuilder module
        and the node names in processing_system_dag.json must match these.
    They must also be passed to the ecr_upload module called by some libera_cdk integration tests.
    """

    l2_cam_cf = "l2-cloud-fraction"
    l2_rad_ssw_toa = "l2-ssw-toa"
    adms = "adms-libera"
    l2_surface_flux = "l2-ssw-surface-flux"
    l2_unfiltered = "l2-unfiltered"
    spice_azel = "spice-azel"
    spice_jpss = "spice-jpss"
    l1b_rad = "l1b-rad"
    l1b_cam = "l1b-cam"
    l0_jpss_pds = "l0-jpss"
    l0_azel_pds = "l0-azel"
    l0_rad_pds = "l0-rad"
    l0_cam_pds = "l0-cam"
    l0_cr = "l0-cr"
    cal_rad = "cal-rad"
    cal_cam = "cal-cam"

    @property
    def ecr_name(self) -> str | None:
        """Get the manually-configured ECR name for this processing step

        We name our ECRs in CDK because they are one of the few resources that humans will need to interact
        with on a regular basis.
        """
        if self.startswith("l0-"):
            # There is no ECR for the L0 processing steps. These are "dummy" processing steps used only for
            # purposes of orchestration management.
            return None
        return f"{self}-docker-repo"

    @classmethod
    def validate(cls, processing_step: str) -> tuple["ProcessingStepIdentifier", int | None]:
        """Validate a processing step string used by the DAG or the orchestration system.

        If successful, returns a tuple containing the ProcessingStepIdentifier and the chunk_number,
        which can be None if the input string does not contain a valid chunk number.

        Parameters
        ----------
        processing_step : str
            The processing step string to validate

        Returns
        -------
        tuple[ProcessingStepIdentifier, Optional[int]]
            The ProcessingStepIdentifier and the chunk number, if present

        """
        if (idx := processing_step.rfind("-")) > 0:
            # the dash could be internal to the enum name, check that
            try:
                # check against the snake-case value
                product_id = ProcessingStepIdentifier(processing_step[:idx])
                # the int conversion could also fail with ValueError
                return product_id, int(processing_step[idx + 1 :])
            except ValueError:
                # assume that there is no chunk number, fall through to check that
                pass
        return ProcessingStepIdentifier(processing_step), None

    def get_archive_bucket_name(self, account_suffix: LiberaAccountSuffix = LiberaAccountSuffix.STAGE) -> str | None:
        """Gets the archive bucket name for this processing step.

        Buckets are named according to the level of data they are storing and which account they are in. This is
        expected to be used by the L2 developers who will most commonly be working with the stage account.

        Parameters
        ----------
        account_suffix : LiberaAccountSuffix, optional
            Account suffix for the bucket name, by default LiberaAccountSuffix.STAGE (stage account)

        Returns
        -------
        str
            The name of the archive bucket for this processing step
        """
        match self.value.split("-", maxsplit=1)[0]:
            case "l0":
                return f"{LiberaDataBucketName.L0_ARCHIVE_BUCKET}{account_suffix}"
            case "l1b":
                return f"{LiberaDataBucketName.L1B_ARCHIVE_BUCKET}{account_suffix}"
            case "l2":
                return f"{LiberaDataBucketName.L2_ARCHIVE_BUCKET}{account_suffix}"
            case "spice":
                return f"{LiberaDataBucketName.SPICE_ARCHIVE_BUCKET}{account_suffix}"
            case "cal" | "adms":
                return f"{LiberaDataBucketName.ANCILLARY_ARCHIVE_BUCKET}{account_suffix}"
            case _:
                raise ValueError(f"Unknown processing step {self.value}")

    def to_str_with_chunk_number(self, chunk_number: int | None = None) -> str:
        """Convert the ProcessingStepIdentifier to a string suitable for matching
        with a DAG key or in the processing orchestration system.

        The chunk_number can be specified when the data product represents
        a PDS file that is typically provided in 12 2-hour chunks per day.
        In that case, the chunk_number appears as a suffix to the orchestration
        step identifier

        Parameters
        ----------
        chunk_number : int, optional
            The chunk number, if applicable for L0 files
        """
        return f"{self}-{chunk_number}" if chunk_number is not None else self.value


class CkObject(StrEnum):
    """Enum of valid CK objects"""

    JPSS = "JPSS-CK"
    AZROT = "AZROT-CK"
    ELSCAN = "ELSCAN-CK"

    @property
    def data_product_id(self) -> DataProductIdentifier:
        """DataProductIdentifier for CKs associated with this CK object"""
        _product_id_map = {
            CkObject.JPSS: DataProductIdentifier.spice_jpss_ck,
            CkObject.AZROT: DataProductIdentifier.spice_az_ck,
            CkObject.ELSCAN: DataProductIdentifier.spice_el_ck,
        }
        return _product_id_map[self]

    @property
    def processing_step_id(self) -> ProcessingStepIdentifier:
        """ProcessingStepIdentifier for the processing step that produces CKs for this CK object"""
        _processing_step_id_map = {
            CkObject.JPSS: ProcessingStepIdentifier.spice_jpss,
            CkObject.AZROT: ProcessingStepIdentifier.spice_azel,
            CkObject.ELSCAN: ProcessingStepIdentifier.spice_azel,
        }
        return _processing_step_id_map[self]


class SpkObject(StrEnum):
    """Enum of valid SPK objects"""

    JPSS = "JPSS-SPK"

    @property
    def data_product_id(self) -> DataProductIdentifier:
        """DataProductIdentifier for SPKs associated with this SPK object"""
        # Only one data product for SPKs
        return DataProductIdentifier.spice_jpss_spk

    @property
    def processing_step_id(self) -> ProcessingStepIdentifier:
        """ProcessingStepIdentifier for the processing step that produces SPKs for this SPK object"""
        # Only one processing step that produces an SPK
        return ProcessingStepIdentifier.spice_jpss


class DataLevel(StrEnum):
    """Data product level"""

    L0 = "L0"
    SPICE = "SPICE"
    CAL = "CAL"
    L1B = "L1B"
    L2 = "L2"


class LiberaApid(Enum):
    """APIDs for L0 packets"""

    JPSS_ATTITUDE_EPHEMERIS = 11
    FILTERED_RADIOMETER = 1036
    FILTERED_AZEL = 1048
    CAMERA = 9999


class LiberaDataBucketName(StrEnum):
    """Names of the data archive buckets"""

    L0_ARCHIVE_BUCKET = "libera-l0-data"
    SPICE_ARCHIVE_BUCKET = "libera-spice-kernels"
    ANCILLARY_ARCHIVE_BUCKET = "libera-ancillary-data"
    L1B_ARCHIVE_BUCKET = "libera-l1b-data"
    L2_ARCHIVE_BUCKET = "libera-l2-data"
