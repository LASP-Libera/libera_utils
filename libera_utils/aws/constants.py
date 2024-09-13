"""AWS ECR Repository/Algorithm names"""
from enum import Enum


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

    @property
    def ecr_name(self) -> str:
        """Get the manually-configured ECR name for this processing step

        We name our ECRs in CDK because they are one of the few resources that humans will need to interact
        with on a regular basis.
        """
        return f"{self.value}-docker-repo"


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
