"""Constants used specifically for AWS resources and resource naming"""

from enum import StrEnum


class LiberaAccountSuffix(StrEnum):
    """Suffixes for the various account types"""

    STAGE = "-stage"
    PROD = "-prod"
    DEV = "-dev"


class LiberaDataBucketName(StrEnum):
    """Names of the data archive buckets"""

    L0_ARCHIVE_BUCKET = "libera-l0-data"
    L1A_ARCHIVE_BUCKET = "libera-l1a-data"
    SPICE_ARCHIVE_BUCKET = "libera-spice-kernels"
    AUXILIARY_ARCHIVE_BUCKET = "libera-auxiliary-data"
    CALIBRATION_ARCHIVE_BUCKET = "libera-calibration-data"
    L1B_ARCHIVE_BUCKET = "libera-l1b-data"
    L2_ARCHIVE_BUCKET = "libera-l2-data"
