"""AWS ECR Repository/Algorithm names"""
from enum import Enum


class DataProductIdentifier(Enum):
    """Enumeration of data product canonical IDs used in AWS resource naming
    These IDs refer to the data products (files) themselves, NOT the processing steps (since processing steps
    may produce multiple products).
    
    In general these names are of the form <level>-<source>-<type>
    # TODO: This enum is duplicated in libera_cdk in libera_lambda_runtime.constants
        When that code is stable, it should be moved here and libera_lambda_runtime should import it from here.
    """
    # L0 PDS files
    l0_rad_pds = "l0-rad-pds"
    l0_cam_pds = "l0-cam-pds"
    l0_az_pds = "l0-az-pds"
    l0_el_pds = "l0-el-pds"
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
    
    In orchestration code, these are used as "NodeID" values to identify processing steps
    # TODO: this enum is duplicated in libera_cdk in the libera_lambda_runtime.constants module for ease of development
        When that is working well, independent of libera_utils, this enum should be replaced with that code and the
        Lambda runtime package should import it from here. 2024-04-30
    """
    l2cf = 'l2-cloud-fraction'
    l2_stf = 'l2-ssw-toa'
    adms = 'libera-adms'
    l2_surface_flux = 'l2-ssw-surface-flux'
    l2_firf = 'l2-far-ir-toa-flux'
    unfilt = 'l1c-unfiltered'
    spice_az = 'libera-spice-az'
    spice_el = 'libera-spice-el'
    spice_jpss = 'jpss-spice'
    pds_ingest = 'l0-ingest-docker-repo'
    l1b_rad = 'l1b-rad'
