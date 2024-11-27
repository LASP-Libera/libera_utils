"""AWS ECR Repository/Algorithm names"""
# Standard
from enum import Enum
from typing import Optional, Union


class ManifestType(Enum):
    """Enumerated legal manifest type values"""
    INPUT = 'INPUT'
    input = INPUT
    OUTPUT = 'OUTPUT'
    output = OUTPUT


class DataProductIdentifier(Enum):
    """Enumeration of data product canonical IDs used in AWS resource naming
    These IDs refer to the data products (files) themselves, NOT the processing steps (since processing steps
    may produce multiple products).

    In general these names are of the form <level>-<source>-<type>
    """
    # L0 construction record
    l0_cr = "l0-cr"

    # L0 PDS files
    l0_rad_pds = "l0-rad-pds"
    l0_cam_pds = "l0-cam-pds"
    l0_azel_pds = "l0-azel-pds"
    l0_jpss_pds = "l0-jpss-pds"

    # SPICE kernels
    spice_az_ck = "spice-az-ck"
    spice_el_ck = "spice-el-ck"
    spice_jpss_ck = "spice-jpss-ck"
    spice_jpss_spk = "spice-jpss-spk"

    # Calibration products
    cal_rad = "cal-rad"
    cal_cam = "cal-cam"

    # L1B products
    l1b_rad = "l1b-rad"
    l1b_cam = "l1b-cam"

    # L2 products
    # TODO: L2 product IDs TBD
    # l2_unf = "l2-unf"  # unfiltered radiance
    # l2_cf = "l2-cf"  # cloud fraction
    # l2_ssw_toa = "l2-ssw-toa"  # SSW TOA flux
    # l2_ssw_surf = "l2-ssw-surf"  # SSW surface flux
    # l2_fir_toa = "l2-fir-toa"  # FIR TOA flux

    # Ancillary products
    anc_adm = "anc-adm"

    @classmethod
    def validate(cls, product_name: str) -> tuple["DataProductIdentifier", Optional[int]]:
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
                return product_id, int(product_name[idx + 1:])
            except ValueError:
                # assume that there is no chunk number, fall through to check that
                pass
        return DataProductIdentifier(product_name), None

    def dump(self, chunk_number: Optional[int] = None) -> str:
        """Convert the DataProductIdentifier to a string suitable for matching
        with a DAG key or in the processing orchestration system.

        The chunk_number can be specified when the data product represents
        a PDS file that is typically provided in 12 2-hour chunks per day.
        In that case, the chunk_number appears as a suffix to the orchestration
        product name.
        """
        return f"{self.value}-{chunk_number}" if chunk_number is not None else self.value


class ProcessingStepIdentifier(Enum):
    """Enumeration of processing step IDs used in AWS resource naming and processing orchestration

    In orchestration code, these are used as "NodeID" values to identify processing steps:
        The processing_step_node_id values used in libera_cdk deployment stackbuilder module
        and the node names in processing_system_dag.json must match these.
    They must also be passed to the ecr_upload module called by some libera_cdk integration tests.
    """
    l2cf = 'l2-cloud-fraction'
    l2_stf = 'l2-ssw-toa'
    adms = 'libera-adms'
    l2_surface_flux = 'l2-ssw-surface-flux'
    l2_firf = 'l2-far-ir-toa-flux'
    unfilt = 'l1c-unfiltered'
    spice_azel = 'spice-azel'
    spice_jpss = 'spice-jpss'
    l1b_rad = 'l1b-rad'
    l1b_cam = 'l1b-cam'
    l0_jpss_pds = 'l0-jpss'
    l0_azel_pds = 'l0-azel'
    l0_rad_pds = 'l0-rad'
    l0_cam_pds = 'l0-cam'
    l0_cr = 'l0-cr'
    cal_rad = 'cal-rad'
    cal_cam = 'cal-cam'

    @property
    def ecr_name(self) -> Union[str, None]:
        """Get the manually-configured ECR name for this processing step

        We name our ECRs in CDK because they are one of the few resources that humans will need to interact
        with on a regular basis.
        """
        if self.value.startswith("l0-"):
            # There is no ECR for the L0 processing steps. These are "dummy" processing steps used only for
            # purposes of orchestration management.
            return None
        return f"{self.value}-docker-repo"

    @classmethod
    def validate(cls, processing_step: str) -> tuple["ProcessingStepIdentifier", Optional[int]]:
        """Validate a processing step string used by the DAG or the orchestration system.

        If successful, returns a tuple containing the ProcessingStepIdentifier and the chunk_number,
        which can be None if the input string does not contain a valid chunk number.
        """
        if (idx := processing_step.rfind("-")) > 0:
            # the dash could be internal to the enum name, check that
            try:
                # check against the snake-case value
                product_id = ProcessingStepIdentifier(processing_step[:idx])
                # the int conversion could also fail with ValueError
                return product_id, int(processing_step[idx + 1:])
            except ValueError:
                # assume that there is no chunk number, fall through to check that
                pass
        return ProcessingStepIdentifier(processing_step), None

    def dump(self, chunk_number: Optional[int] = None) -> str:
        """Convert the ProcessingStepIdentifier to a string suitable for matching
        with a DAG key or in the processing orchestration system.

        The chunk_number can be specified when the data product represents
        a PDS file that is typically provided in 12 2-hour chunks per day.
        In that case, the chunk_number appears as a suffix to the orchestration
        step identifier
        """
        return f"{self.value}-{chunk_number}" if chunk_number is not None else self.value


class CkObject(Enum):
    """Enum of valid CK objects"""
    JPSS = "JPSS"
    AZROT = "AZROT"
    ELSCAN = "ELSCAN"

    @property
    def data_product_id(self) -> DataProductIdentifier:
        """DataProductIdentifier for CKs associated with this CK object"""
        _product_id_map = {
            CkObject.JPSS: DataProductIdentifier.spice_jpss_ck,
            CkObject.AZROT: DataProductIdentifier.spice_az_ck,
            CkObject.ELSCAN: DataProductIdentifier.spice_el_ck
        }
        return _product_id_map[self]

    @property
    def processing_step_id(self) -> ProcessingStepIdentifier:
        """ProcessingStepIdentifier for the processing step that produces CKs for this CK object"""
        _processing_step_id_map = {
            CkObject.JPSS: ProcessingStepIdentifier.spice_jpss,
            CkObject.AZROT: ProcessingStepIdentifier.spice_azel,
            CkObject.ELSCAN: ProcessingStepIdentifier.spice_azel
        }
        return _processing_step_id_map[self]


class SpkObject(Enum):
    """Enum of valid SPK objects"""
    JPSS = "JPSS"

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


class DataLevel(Enum):
    """Data product level"""
    L0 = "L0"
    SPICE = "SPICE"
    CAL = "CAL"
    L1B = 'L1B'
    L2 = 'L2'


class LiberaApid(Enum):
    """APIDs for L0 packets"""
    JPSS_ATTITUDE_EPHEMERIS = 11
    FILTERED_RADIOMETER = 1036
    FILTERED_AZEL = 1048
    CAMERA = 9999
