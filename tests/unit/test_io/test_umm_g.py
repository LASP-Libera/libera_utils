"""Tests for UMM-G module"""

from datetime import UTC, datetime

import numpy as np
import pytest
import xarray as xr
from pydantic import ValidationError

from libera_utils.io.filenaming import LiberaDataProductFilename
from libera_utils.io.umm_g import (
    BaseFileType,
    DataGranuleType,
    DayNightFlagEnum,
    FileSizeUnitEnum,
    IdentifierType,
    IdentifierTypeEnum,
    OrbitCalculatedSpatialDomainType,
    QAFlagsType,
    QAStatsType,
    QualityFlagEnum,
    TemporalExtentType,
    UMMGDatasetTransformer,
    UMMGranule,
    VerticalSpatialDomainType,
    VerticalSpatialDomainTypeEnum,
    validate_iso_datetime,
)

# Valid LiberaDataProductFilename used across UMMGDatasetTransformer tests.
# GranuleUR is derived from filepath.path.stem, so assertions should reference this constant.
_TEST_FILEPATH = LiberaDataProductFilename("LIBERA_L1B_RAD-4CH_V1-0-0_20240101T000000_20240101T235959_R24001120000.nc")


def create_test_libera_dataset(
    n_samples: int = 100,
    include_all_metadata: bool = True,
    collection_short_name: str | None = None,
    collection_version: str | None = None,
) -> xr.Dataset:
    """
    Create a test xarray Dataset that mimics Libera data product structure.

    Parameters
    ----------
    n_samples : int, optional
        Number of sample data points to generate, by default 100
    include_all_metadata : bool, optional
        If True, includes all metadata fields. If False, creates minimal dataset
        to test warning generation, by default True
    collection_short_name : str | None, optional
        Collection short name, by default "LIBERA_L1A_RAD"
    collection_version : str | None, optional
        Collection version, by default "1.0"

    Returns
    -------
    xr.Dataset
        An xarray Dataset with Libera-style metadata and variables

    Notes
    -----
    This function generates a dataset structure that follows the Libera data product
    conventions defined in netcdf.py, including:
    - Static project metadata (Format, Conventions, Platform, Project info)
    - Dynamic product metadata (input_files)
    - Dynamic spatiotemporal metadata (temporal ranges, geolocation)
    - Science data variables (time_stamp, radiance, lat, lon, quality flags)
    """
    # Generate sample data
    time_data = np.arange(
        np.datetime64("2025-10-01"),
        np.datetime64("2025-10-01") + np.timedelta64(n_samples, "ns"),
        dtype="datetime64[ns]",
    )
    radiance_data = np.random.uniform(0, 1000, n_samples)
    lat_data = np.random.uniform(-90, 90, n_samples)
    lon_data = np.random.uniform(-180, 180, n_samples)
    qflags_data = np.random.randint(0, 10, n_samples, dtype=np.int32)

    # Create dataset with variables
    ds = xr.Dataset(
        {
            "time_stamp": (["n_samples"], time_data),
            "fil_rad": (["n_samples"], radiance_data),
            "lat": (["n_samples"], lat_data),
            "lon": (["n_samples"], lon_data),
            "q_flags": (["n_samples"], qflags_data),
        }
    )

    # Add variable-level attributes
    ds["time_stamp"].attrs = {
        "long_name": "Time",
        "units": "datetime64[ns]",
        "valid_range": [0, 1],
        "missing_value": -9999,
        "dtype": "datetime64",
    }
    ds["fil_rad"].attrs = {
        "long_name": "Filtered Radiance",
        "units": "W/(m^2*sr*nm)",
        "valid_range": [0, 1000],
        "missing_value": -9999,
        "dtype": "float",
    }
    ds["lat"].attrs = {
        "long_name": "Geolocation latitude",
        "units": "degrees",
        "valid_range": [-90, 90],
        "missing_value": -999,
        "dtype": "float",
    }
    ds["lon"].attrs = {
        "long_name": "Geolocation longitude",
        "units": "degrees",
        "valid_range": [-180, 180],
        "missing_value": -999,
        "dtype": "float",
    }
    ds["q_flags"].attrs = {
        "long_name": "Quality Flags",
        "units": "N/A",
        "valid_range": [0, 2147483647],
        "missing_value": -999,
        "dtype": "int32",
    }

    if include_all_metadata:
        # Static Project Metadata (from StaticProjectMetadata)
        ds.attrs["Format"] = "NetCDF-4"
        ds.attrs["Conventions"] = "CF-1.12"
        ds.attrs["ProjectLongName"] = "Libera"
        ds.attrs["ProjectShortName"] = "Libera"
        ds.attrs["PlatformLongName"] = "Joint Polar Satellite System 4"
        ds.attrs["PlatformShortName"] = "NOAA-22"

        # Dynamic Product Metadata (from DynamicProductMetadata)
        ds.attrs["input_files"] = ["LIBERA_L0_PKT_V1-0-0_20240101T000000_20240101T235959_R20240102T120000.nc"]

        # Dynamic Spatiotemporal Metadata (from DynamicSpatioTemporalMetadata)
        ds.attrs["ProductionDateTime"] = datetime.now(UTC).isoformat()
        ds.attrs["RangeBeginningDate"] = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC).isoformat()
        ds.attrs["RangeBeginningTime"] = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC).isoformat()
        ds.attrs["RangeEndingDate"] = datetime(2024, 1, 1, 23, 59, 59, tzinfo=UTC).isoformat()
        ds.attrs["RangeEndingTime"] = datetime(2024, 1, 1, 23, 59, 59, tzinfo=UTC).isoformat()

        # GPolygon - representing bounding box as polygon points
        # Format: list of dicts with latitude/longitude
        ds.attrs["GPolygon"] = str(
            [
                {"latitude": float(lat_data.min()), "longitude": float(lon_data.min())},
                {"latitude": float(lat_data.min()), "longitude": float(lon_data.max())},
                {"latitude": float(lat_data.max()), "longitude": float(lon_data.max())},
                {"latitude": float(lat_data.max()), "longitude": float(lon_data.min())},
                {"latitude": float(lat_data.min()), "longitude": float(lon_data.min())},  # Close polygon
            ]
        )

        # Collection Reference info
        ds.attrs["CollectionShortName"] = collection_short_name or "LIBERA_L1A_RAD"
        ds.attrs["CollectionVersion"] = collection_version or "1.0"

        # Additional CF and ACDD conventions
        ds.attrs["title"] = "Libera L1A Radiance Data"
        ds.attrs["institution"] = "NASA Langley Research Center"
        ds.attrs["source"] = "Libera Instrument"
        ds.attrs["history"] = f"Created {datetime.now(UTC).isoformat()}"
        ds.attrs["references"] = "https://libera.larc.nasa.gov"

        # Geospatial bounds (CF conventions)
        ds.attrs["geospatial_lat_min"] = float(lat_data.min())
        ds.attrs["geospatial_lat_max"] = float(lat_data.max())
        ds.attrs["geospatial_lon_min"] = float(lon_data.min())
        ds.attrs["geospatial_lon_max"] = float(lon_data.max())

        # Time coverage (CF conventions)
        ds.attrs["time_coverage_start"] = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC).isoformat()
        ds.attrs["time_coverage_end"] = datetime(2024, 1, 1, 23, 59, 59, tzinfo=UTC).isoformat()
    else:
        # Minimal metadata - only required for xarray
        ds.attrs["Format"] = "NetCDF-4"
        ds.attrs["Conventions"] = "CF-1.12"

    return ds


def test_validate_iso_datetime_with_z_suffix():
    """Test parsing ISO datetime with Z suffix"""
    result = validate_iso_datetime("2024-10-09T12:00:00Z")
    assert isinstance(result, datetime)
    assert result.year == 2024
    assert result.month == 10
    assert result.day == 9


def test_validate_iso_datetime_invalid_format():
    """Test that invalid datetime strings raise ValueError"""
    with pytest.raises(ValueError, match="Invalid ISO 8601 datetime format"):
        validate_iso_datetime("not-a-date")


def test_basefile_size_rules():
    with pytest.raises(ValidationError, match="If you provide Size, you must also use SizeUnit"):
        BaseFileType(
            Name="test.nc",
            Size=100.0,
        )

    with pytest.raises(ValidationError, match="If you provide Size, you must also use SizeInBytes"):
        BaseFileType(
            Name="test.nc",
            Size=100.0,
            SizeUnit=FileSizeUnitEnum.MB,
        )

    file = BaseFileType(
        Name="test.nc",
        Size=100.0,
        SizeUnit=FileSizeUnitEnum.MB,
        SizeInBytes=104857600,
    )
    assert file.Size == 100.0
    assert file.SizeUnit == FileSizeUnitEnum.MB


def test_temporal_extent_rules():
    with pytest.raises(
        ValidationError,
        match="Either RangeDateTime or SingleDateTime must be provided, but not both",
    ):
        TemporalExtentType(
            RangeDateTime={"BeginningDateTime": "2024-10-09T12:00:00Z"},
            SingleDateTime="2024-10-09T12:00:00Z",
        )
    with pytest.raises(
        ValidationError,
        match="Either RangeDateTime or SingleDateTime must be provided, but not both",
    ):
        TemporalExtentType()

    temporal = TemporalExtentType(RangeDateTime={"BeginningDateTime": "2024-10-09T12:00:00Z"})
    assert temporal.RangeDateTime is not None
    assert temporal.SingleDateTime is None


def test_vertical_spatial_domain_rules():
    """Test that providing both Value and MinimumValue raises error"""
    with pytest.raises(
        ValidationError,
        match="Either Value or both MinimumValue and MaximumValue must be provided",
    ):
        VerticalSpatialDomainType(
            Type=VerticalSpatialDomainTypeEnum.ALTITUDE,
            Value="1000",
            MinimumValue="500",
        )

    vertical = VerticalSpatialDomainType(Type=VerticalSpatialDomainTypeEnum.ALTITUDE, Value="1000")
    assert vertical.Value == "1000"
    assert vertical.MinimumValue is None


def test_orbit_begin_rules():
    with pytest.raises(
        ValidationError,
        match="Both BeginOrbitNumber and EndOrbitNumber must be provided together",
    ):
        OrbitCalculatedSpatialDomainType(BeginOrbitNumber=100)

    with pytest.raises(
        ValidationError,
        match="Cannot provide both OrbitNumber and BeginOrbitNumber/EndOrbitNumber",
    ):
        OrbitCalculatedSpatialDomainType(OrbitNumber=150, BeginOrbitNumber=100, EndOrbitNumber=200)

    with pytest.raises(ValidationError, match="At least one attribute value must be provided"):
        OrbitCalculatedSpatialDomainType()

    orbit = OrbitCalculatedSpatialDomainType(OrbitNumber=12345)
    assert orbit.OrbitNumber == 12345
    assert orbit.BeginOrbitNumber is None

    orbit = OrbitCalculatedSpatialDomainType(BeginOrbitNumber=100, EndOrbitNumber=200)
    assert orbit.BeginOrbitNumber == 100
    assert orbit.EndOrbitNumber == 200


def test_qa_stats_no_stats_rules():
    with pytest.raises(ValidationError, match="At least one QA statistic must be provided"):
        QAStatsType()

    qa = QAStatsType(QAPercentMissingData=2.5)
    assert qa.QAPercentMissingData == 2.5

    with pytest.raises(ValidationError, match="At least one quality flag value must be provided"):
        QAFlagsType()

    qa = QAFlagsType(AutomaticQualityFlag=QualityFlagEnum.PASSED)
    assert qa.AutomaticQualityFlag == QualityFlagEnum.PASSED


def test_data_granule_duplicate_identifiers():
    with pytest.raises(ValidationError, match="All identifiers must be unique"):
        DataGranuleType(
            DayNightFlag=DayNightFlagEnum.DAY,
            ProductionDateTime="2024-10-09T12:00:00Z",
            Identifiers=[
                IdentifierType(
                    Identifier="LIBERA_001",
                    IdentifierType=IdentifierTypeEnum.PRODUCER_GRANULE_ID,
                ),
                IdentifierType(
                    Identifier="LIBERA_001",
                    IdentifierType=IdentifierTypeEnum.PRODUCER_GRANULE_ID,
                ),
            ],
        )

    data_granule = DataGranuleType(
        DayNightFlag=DayNightFlagEnum.DAY,
        ProductionDateTime="2024-10-09T12:00:00Z",
        Identifiers=[
            IdentifierType(
                Identifier="LIBERA_001",
                IdentifierType=IdentifierTypeEnum.PRODUCER_GRANULE_ID,
            ),
            IdentifierType(
                Identifier="LIBERA_002",
                IdentifierType=IdentifierTypeEnum.PRODUCER_GRANULE_ID,
            ),
        ],
    )
    assert len(data_granule.Identifiers) == 2


def test_ummgranule_minimal_construction():
    granule = UMMGranule(
        GranuleUR="TEST_GRANULE_001",
        ProviderDates=[{"Date": "2024-10-09T12:00:00Z", "Type": "Create"}],
        CollectionReference={"ShortName": "TEST_COLLECTION", "Version": "001"},
    )
    assert granule.GranuleUR == "TEST_GRANULE_001"
    assert len(granule.ProviderDates) == 1


def test_ummgranule_serialization_roundtrip():
    original = UMMGranule(
        GranuleUR="TEST_GRANULE_001",
        ProviderDates=[{"Date": "2024-10-09T12:00:00Z", "Type": "Create"}],
        CollectionReference={"ShortName": "TEST_COLLECTION", "Version": "001"},
    )

    json_str = original.model_dump_json()
    parsed = UMMGranule.model_validate_json(json_str)

    assert parsed.GranuleUR == original.GranuleUR
    assert parsed.CollectionReference.ShortName == original.CollectionReference.ShortName


def test_umm_g_dataset_transformer_basic():
    """Test UMMGDatasetTransformer with a complete test dataset."""

    # Create a test dataset with complete metadata
    test_dataset = create_test_libera_dataset(
        n_samples=50,
        collection_short_name="TEST_COLLECTION",
        collection_version="1.0",
    )

    # Create transformer
    transformer = UMMGDatasetTransformer(test_dataset, _TEST_FILEPATH, log_warnings=False)

    # Transform to UMM granule
    umm_granule = transformer.umm_granule

    # Verify basic fields
    assert umm_granule.GranuleUR == _TEST_FILEPATH.path.stem
    assert umm_granule.CollectionReference.ShortName == "TEST_COLLECTION"
    assert umm_granule.CollectionReference.Version == "1.0"
    assert len(umm_granule.ProviderDates) > 0
    assert umm_granule.Platforms is not None
    assert umm_granule.Projects is not None


def test_umm_g_dataset_transformer_from_dataset_classmethod():
    """Test UMMGranule.from_dataset() classmethod."""
    # Create a test dataset
    test_dataset = create_test_libera_dataset(
        collection_short_name="LIBERA_L1A_RAD",
    )

    # Use the classmethod
    umm_granule = UMMGranule.from_dataset(test_dataset, _TEST_FILEPATH, log_warnings=True)

    # Verify the granule was created correctly
    assert isinstance(umm_granule, UMMGranule)
    assert umm_granule.GranuleUR == _TEST_FILEPATH.path.stem
    assert umm_granule.CollectionReference.ShortName == "LIBERA_L1A_RAD"


def test_umm_g_dataset_transformer_minimal_metadata():
    """Test UMMGDatasetTransformer with minimal metadata triggers warnings."""

    # Create a minimal dataset
    test_dataset = create_test_libera_dataset(n_samples=10, include_all_metadata=False)

    # Create transformer
    transformer = UMMGDatasetTransformer(test_dataset, _TEST_FILEPATH, log_warnings=False)

    # Transform to UMM granule
    umm_granule = transformer.umm_granule

    # Verify warnings were generated
    assert len(transformer.warnings) > 0

    # Verify basic structure still works
    assert umm_granule.GranuleUR is not None
    assert umm_granule.ProviderDates is not None


def test_umm_g_dataset_transformer_json_output():
    """Test that UMMGDatasetTransformer produces valid JSON output."""
    # Create a test dataset
    test_dataset = create_test_libera_dataset(collection_short_name="TEST_JSON")

    # Transform to UMM granule
    umm_granule = UMMGranule.from_dataset(test_dataset, _TEST_FILEPATH)

    # Verify JSON serialization works
    json_output = umm_granule.model_dump_json(indent=2)
    assert isinstance(json_output, str)
    assert _TEST_FILEPATH.path.stem in json_output
    assert "TEST_JSON" in json_output

    # Verify we can parse it back
    parsed = UMMGranule.model_validate_json(json_output)
    assert parsed.GranuleUR == _TEST_FILEPATH.path.stem


def test_granule_ur_equals_filename_stem():
    """Test that GranuleUR is set to the data product filename stem (i.e. filename without extension)."""

    test_dataset = create_test_libera_dataset()
    transformer = UMMGDatasetTransformer(test_dataset, _TEST_FILEPATH, log_warnings=False)

    assert transformer.umm_granule.GranuleUR == _TEST_FILEPATH.path.stem
