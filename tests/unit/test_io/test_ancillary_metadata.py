"""Tests for reading an external ancillary file's version and revision from its own metadata"""

import datetime as dt
import shutil

import netCDF4
import pytest
from pyhdf import SD

from libera_utils.io.ancillary_metadata import AncillaryFileMetadata, read_ancillary_file_metadata
from libera_utils.io.filenaming import AbstractValidFilename

SSF_NAME = "CER_SSF_NOAA20-FM6-VIIRS_Edition1C_101103.2023010100.nc"
CLDPIX_NAME = "CER_CLDPIX_NOAA20-VIIRS_1P9test_000000.2020041015.nc"
NISE_NAME = "NISE_AMSR2_20260610.HDFEOS"

NISE_CORE_METADATA = """GROUP                  = INVENTORYMETADATA
  GROUP                  = ECSDATAGRANULE
    OBJECT                 = PRODUCTIONDATETIME
      NUM_VAL              = 1
      VALUE                = "2026-06-10T12:28:08.00Z"
    END_OBJECT             = PRODUCTIONDATETIME

    OBJECT                 = LOCALVERSIONID
      NUM_VAL              = 1
      VALUE                = "7"
    END_OBJECT             = LOCALVERSIONID
  END_GROUP              = ECSDATAGRANULE

  GROUP                  = COLLECTIONDESCRIPTIONCLASS
    OBJECT                 = VERSIONID
      NUM_VAL              = 1
      VALUE                = 1
    END_OBJECT             = VERSIONID
  END_GROUP              = COLLECTIONDESCRIPTIONCLASS
END_GROUP              = INVENTORYMETADATA
END
"""


def _upstream_copy(fmatch_real_data_path, tmp_path, name):
    """Copy a shortened sample granule into tmp_path under its upstream name, which the filename classes parse."""
    sample = fmatch_real_data_path / name.replace(".nc", "_subset.nc").replace(".HDFEOS", "_subset.HDFEOS")
    return shutil.copy(sample, tmp_path / name)


def _write_ceres(path, **global_attributes):
    """Write a netCDF file at path whose only content is the given global attributes."""
    with netCDF4.Dataset(path, "w") as dataset:
        dataset.setncatts(global_attributes)
    return path


def _write_nise(path, core_metadata=None):
    """Write an HDF4 file at path, with the core metadata attribute when given."""
    hdf = SD.SD(str(path), SD.SDC.WRITE | SD.SDC.CREATE)
    try:
        hdf.attr("title").set(SD.SDC.CHAR8, "NISE test file")
        if core_metadata is not None:
            hdf.attr("coremetadata.0").set(SD.SDC.CHAR8, core_metadata)
    finally:
        hdf.end()
    return path


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        pytest.param(
            SSF_NAME,
            AncillaryFileMetadata(version="Edition1C", revision=dt.datetime(2024, 8, 27, 19, 46, 15, tzinfo=dt.UTC)),
            id="ssf-date_created",
        ),
        pytest.param(
            CLDPIX_NAME,
            AncillaryFileMetadata(version="1P9test", revision=dt.datetime(2025, 7, 10, 19, 2, 50, tzinfo=dt.UTC)),
            id="cldpix-processTime-as-utc",
        ),
        pytest.param(
            NISE_NAME,
            AncillaryFileMetadata(version="1", revision=dt.datetime(2026, 6, 10, 12, 28, 8, tzinfo=dt.UTC)),
            id="nise-core-metadata",
        ),
    ],
)
def test_read_ancillary_file_metadata_from_sample_granules(fmatch_real_data_path, tmp_path, name, expected):
    """A real CERES SSF, CERES CLDPIX or NISE granule gives its version and its UTC production time."""
    filename = AbstractValidFilename.from_file_path(_upstream_copy(fmatch_real_data_path, tmp_path, name))

    metadata = read_ancillary_file_metadata(filename)

    assert metadata == expected
    assert metadata.revision.tzinfo is dt.UTC


def test_read_ancillary_file_metadata_converts_an_offset_to_utc(tmp_path):
    """A CERES date_created with a non-UTC offset is converted to UTC."""
    path = _write_ceres(tmp_path / SSF_NAME, date_created="2024-08-27T13:46:15-06:00")

    metadata = read_ancillary_file_metadata(AbstractValidFilename.from_file_path(path))

    assert metadata.revision == dt.datetime(2024, 8, 27, 19, 46, 15, tzinfo=dt.UTC)
    assert metadata.revision.tzinfo is dt.UTC


@pytest.mark.parametrize(
    ("name", "attributes", "attribute"),
    [
        pytest.param(SSF_NAME, {"title": "SSF"}, "date_created", id="ssf-missing"),
        pytest.param(SSF_NAME, {"date_created": "yesterday"}, "date_created", id="ssf-unparseable"),
        pytest.param(SSF_NAME, {"date_created": "2024-08-27T19:46:15"}, "date_created", id="ssf-no-zone"),
        pytest.param(CLDPIX_NAME, {"date_created": "2024-08-27T19:46:15Z"}, "processTime", id="cldpix-missing"),
        pytest.param(CLDPIX_NAME, {"processTime": "2025-07-10 19:02:50"}, "processTime", id="cldpix-unparseable"),
    ],
)
def test_read_ancillary_file_metadata_rejects_a_ceres_file_without_its_revision(tmp_path, name, attributes, attribute):
    """A CERES file whose revision attribute is missing or does not parse raises, naming the file and attribute."""
    path = _write_ceres(tmp_path / name, **attributes)

    with pytest.raises(ValueError, match=rf"{name}.*{attribute}"):
        read_ancillary_file_metadata(AbstractValidFilename.from_file_path(path))


@pytest.mark.parametrize(
    ("core_metadata", "attribute"),
    [
        pytest.param(None, "coremetadata.0", id="no-core-metadata"),
        pytest.param(
            NISE_CORE_METADATA.replace("PRODUCTIONDATETIME", "PRODUCTIONTIME"), "PRODUCTIONDATETIME", id="no-revision"
        ),
        pytest.param(NISE_CORE_METADATA.replace("= VERSIONID", "= COLLECTIONVERSION"), "VERSIONID", id="no-version"),
        pytest.param(
            NISE_CORE_METADATA.replace('"2026-06-10T12:28:08.00Z"', '"June 10"'),
            "PRODUCTIONDATETIME",
            id="unparseable-revision",
        ),
    ],
)
def test_read_ancillary_file_metadata_rejects_a_nise_file_without_its_metadata(tmp_path, core_metadata, attribute):
    """A NISE file whose core metadata, VERSIONID or PRODUCTIONDATETIME is missing or does not parse raises."""
    path = _write_nise(tmp_path / NISE_NAME, core_metadata)

    with pytest.raises(ValueError, match=rf"{NISE_NAME}.*{attribute}"):
        read_ancillary_file_metadata(AbstractValidFilename.from_file_path(path))


def test_read_ancillary_file_metadata_reads_the_versionid_not_localversionid(tmp_path):
    """NISE's version is the collection VERSIONID, not the LOCALVERSIONID beside it."""
    path = _write_nise(tmp_path / NISE_NAME, NISE_CORE_METADATA)

    metadata = read_ancillary_file_metadata(AbstractValidFilename.from_file_path(path))

    assert metadata == AncillaryFileMetadata(version="1", revision=dt.datetime(2026, 6, 10, 12, 28, 8, tzinfo=dt.UTC))


def test_read_ancillary_file_metadata_reads_an_s3_file(fmatch_real_data_path, tmp_path, write_file_to_s3):
    """A file on S3 is read through cloudpathlib's local copy."""
    local = _upstream_copy(fmatch_real_data_path, tmp_path, SSF_NAME)
    s3_path = write_file_to_s3(local, f"s3://ancillary-bucket/CERES-NOAA20-SSF/{SSF_NAME}")

    metadata = read_ancillary_file_metadata(AbstractValidFilename.from_file_path(s3_path))

    assert metadata.revision == dt.datetime(2024, 8, 27, 19, 46, 15, tzinfo=dt.UTC)
