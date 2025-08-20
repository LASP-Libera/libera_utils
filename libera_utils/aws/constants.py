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
    ANCILLARY_ARCHIVE_BUCKET = "libera-ancillary-data"
    L1B_ARCHIVE_BUCKET = "libera-l1b-data"
    L2_ARCHIVE_BUCKET = "libera-l2-data"
