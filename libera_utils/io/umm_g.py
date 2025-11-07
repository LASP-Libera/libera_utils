"""File for describing pydantic model and utility functions for UMM Granules."""

import logging
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Annotated

import xarray as xr
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


def validate_iso_datetime(v):
    """Validate datetime format and convert to datetime object."""
    if isinstance(v, str):
        try:
            # Handle both 'Z' suffix and timezone formats
            if v.endswith("Z"):
                return datetime.fromisoformat(v.replace("Z", "+00:00"))
            else:
                return datetime.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Invalid ISO 8601 datetime format: {v}")
    elif isinstance(v, datetime):
        return v
    else:
        raise ValueError(f"Expected datetime string or datetime object, got {type(v)}")


# Custom DateTime type for UMM-G with ISO validation
ISODateTime = Annotated[
    datetime,
    Field(description="ISO 8601 datetime string (e.g., '2024-10-09T12:00:00Z')"),
    field_validator("*", mode="before")(validate_iso_datetime),
]

GranuleLocalityType = Annotated[str, Field(min_length=1, max_length=1024)]
LatitudeType = Annotated[
    float,
    Field(
        ge=-90,
        le=90,
        description="The latitude value of a spatially referenced point, in degrees. Latitude values range from -90 to 90.",
    ),
]
LongitudeType = Annotated[
    float,
    Field(
        ge=-180,
        le=180,
        description="The longitude value of a spatially referenced point, in degrees. Longitude values range from -180 to 180.",
    ),
]


class ProviderDateTypeEnum(str, Enum):
    CREATE = "Create"
    INSERT = "Insert"
    UPDATE = "Update"
    DELETE = "Delete"


class DayNightFlagEnum(str, Enum):
    DAY = "Day"
    NIGHT = "Night"
    BOTH = "Both"
    UNSPECIFIED = "Unspecified"


class FileSizeUnitEnum(str, Enum):
    KB = "KB"
    MB = "MB"
    GB = "GB"
    TB = "TB"
    PB = "PB"
    NA = "NA"


class ChecksumAlgorithmEnum(str, Enum):
    ADLER_32 = "Adler-32"
    BSD_CHECKSUM = "BSD checksum"
    FLETCHER_32 = "Fletcher-32"
    FLETCHER_64 = "Fletcher-64"
    MD5 = "MD5"
    POSIX = "POSIX"
    SHA_1 = "SHA-1"
    SHA_2 = "SHA-2"
    SHA_256 = "SHA-256"
    SHA_384 = "SHA-384"
    SHA_512 = "SHA-512"
    SM3 = "SM3"
    SYSV = "SYSV"


class FormatTypeEnum(str, Enum):
    NATIVE = "Native"
    SUPPORTED = "Supported"
    NA = "NA"


class IdentifierTypeEnum(str, Enum):
    PRODUCER_GRANULE_ID = "ProducerGranuleId"
    LOCAL_VERSION_ID = "LocalVersionId"
    FEATURE_ID = "FeatureId"
    CRID = "CRID"
    OTHER = "Other"


class OrbitDirectionEnum(str, Enum):
    ASCENDING = "A"
    DESCENDING = "D"


class VerticalSpatialDomainTypeEnum(str, Enum):
    ATMOSPHERE_LAYER = "Atmosphere Layer"
    PRESSURE = "Pressure"
    ALTITUDE = "Altitude"
    DEPTH = "Depth"


class VerticalUnitEnum(str, Enum):
    FATHOMS = "Fathoms"
    FEET = "Feet"
    HECTOPASCALS = "HectoPascals"
    KILOMETERS = "Kilometers"
    METERS = "Meters"
    MILLIBARS = "Millibars"
    POUNDS_PER_SQUARE_INCH = "PoundsPerSquareInch"
    ATMOSPHERE = "Atmosphere"
    INCHES_OF_MERCURY = "InchesOfMercury"
    INCHES_OF_WATER = "InchesOfWater"


class QualityFlagEnum(str, Enum):
    PASSED = "Passed"
    FAILED = "Failed"
    SUSPECT = "Suspect"
    UNDETERMINED = "Undetermined"


class OperationalFlagEnum(str, Enum):
    PASSED = "Passed"
    FAILED = "Failed"
    BEING_INVESTIGATED = "Being Investigated"
    NOT_INVESTIGATED = "Not Investigated"
    INFERRED_PASSED = "Inferred Passed"
    INFERRED_FAILED = "Inferred Failed"
    SUSPECT = "Suspect"
    UNDETERMINED = "Undetermined"


class ScienceFlagEnum(str, Enum):
    PASSED = "Passed"
    FAILED = "Failed"
    BEING_INVESTIGATED = "Being Investigated"
    NOT_INVESTIGATED = "Not Investigated"
    INFERRED_PASSED = "Inferred Passed"
    INFERRED_FAILED = "Inferred Failed"
    SUSPECT = "Suspect"
    HOLD = "Hold"
    UNDETERMINED = "Undetermined"


class TilingSystemNameEnum(str, Enum):
    CALIPSO = "CALIPSO"
    MISR = "MISR"
    MODIS_TILE_EASE = "MODIS Tile EASE"
    MODIS_TILE_SIN = "MODIS Tile SIN"
    SMAP_TILE_EASE = "SMAP Tile EASE"
    WELD_ALASKA_TILE = "WELD Alaska Tile"
    WELD_CONUS_TILE = "WELD CONUS Tile"
    WRS_1 = "WRS-1"
    WRS_2 = "WRS-2"


class ProjectionNameEnum(str, Enum):
    GEOGRAPHIC = "Geographic"
    MERCATOR = "Mercator"
    SPHERICAL_MERCATOR = "Spherical Mercator"
    SPACE_OBLIQUE_MERCATOR = "Space Oblique Mercator"
    UNIVERSAL_TRANSVERSE_MERCATOR = "Universal Transverse Mercator"
    MILITARY_GRID_REFERENCE = "Military Grid Reference"
    MODIS_SINUSOIDAL_SYSTEM = "MODIS Sinusoidal System"
    SINUSOIDAL = "Sinusoidal"
    LAMBERT_EQUAL_AREA = "Lambert Equal Area"
    NSIDC_EASE_GRID_NORTH_AND_SOUTH = "NSIDC EASE Grid North and South (Lambert EA)"
    NSIDC_EASE_GRID_GLOBAL = "NSIDC EASE Grid Global"
    EASE_GRID_2_0_N_POLAR = "EASE Grid 2.0 N. Polar"
    PLATE_CARREE = "Plate Carree"
    POLAR_STEREOGRAPHIC = "Polar Stereographic"
    WELD_ALBERS_EQUAL_AREA = "WELD Albers Equal Area"
    CANADIAN_ALBERS_EQUAL_AREA_CONIC = "Canadian Albers Equal Area Conic"
    LAMBERT_CONFORMAL_CONIC = "Lambert Conformal Conic"
    STATE_PLANE_COORDINATES = "State Plane Coordinates"
    ALBERS_EQUAL_AREA_CONIC = "Albers Equal Area Conic"
    TRANSVERSE_MERCATOR = "Transverse Mercator"
    LAMBERT_AZIMUTHAL_EQUAL_AREA = "Lambert Azimuthal Equal Area"
    UTM_NORTHERN_HEMISPHERE = "UTM Northern Hemisphere"
    NAD83_UTM_ZONE_17N = "NAD83 / UTM zone 17N"
    UTM_SOUTHERN_HEMISPHERE = "UTM Southern Hemisphere"
    CYLINDRICAL = "Cylindrical"


class RelatedUrlTypeEnum(str, Enum):
    DOWNLOAD_SOFTWARE = "DOWNLOAD SOFTWARE"
    EXTENDED_METADATA = "EXTENDED METADATA"
    GET_DATA = "GET DATA"
    GET_DATA_VIA_DIRECT_ACCESS = "GET DATA VIA DIRECT ACCESS"
    GET_RELATED_VISUALIZATION = "GET RELATED VISUALIZATION"
    GOTO_WEB_TOOL = "GOTO WEB TOOL"
    PROJECT_HOME_PAGE = "PROJECT HOME PAGE"
    USE_SERVICE_API = "USE SERVICE API"
    VIEW_RELATED_INFORMATION = "VIEW RELATED INFORMATION"


class RelatedUrlSubTypeEnum(str, Enum):
    BROWSE_IMAGE_SOURCE = "BROWSE IMAGE SOURCE"
    MOBILE_APP = "MOBILE APP"
    APPEARS = "APPEARS"
    DATA_COLLECTION_BUNDLE = "DATA COLLECTION BUNDLE"
    DATA_TREE = "DATA TREE"
    DATACAST_URL = "DATACAST URL"
    DIRECT_DOWNLOAD = "DIRECT DOWNLOAD"
    EOSDIS_DATA_POOL = "EOSDIS DATA POOL"
    EARTHDATA_SEARCH = "Earthdata Search"
    GIOVANNI = "GIOVANNI"
    GOLIVE_PORTAL = "GoLIVE Portal"
    ICEBRIDGE_PORTAL = "IceBridge Portal"
    LAADS = "LAADS"
    LANCE = "LANCE"
    MIRADOR = "MIRADOR"
    MODAPS = "MODAPS"
    NOAA_CLASS = "NOAA CLASS"
    NOMADS = "NOMADS"
    ORDER = "Order"
    PORTAL = "PORTAL"
    SUBSCRIBE = "Subscribe"
    USGS_EARTH_EXPLORER = "USGS EARTH EXPLORER"
    VERTEX = "VERTEX"
    VIRTUAL_COLLECTION = "VIRTUAL COLLECTION"
    MAP = "MAP"
    WORLDVIEW = "WORLDVIEW"
    LIVE_ACCESS_SERVER = "LIVE ACCESS SERVER (LAS)"
    MAP_VIEWER = "MAP VIEWER"
    SIMPLE_SUBSET_WIZARD = "SIMPLE SUBSET WIZARD (SSW)"
    SUBSETTER = "SUBSETTER"
    GRADS_DATA_SERVER = "GRADS DATA SERVER (GDS)"
    MAP_SERVICE = "MAP SERVICE"
    OPENDAP_DATA = "OPENDAP DATA"
    OPENSEARCH = "OpenSearch"
    SERVICE_CHAINING = "SERVICE CHAINING"
    TABULAR_DATA_STREAM = "TABULAR DATA STREAM (TDS)"
    THREDDS_DATA = "THREDDS DATA"
    WEB_COVERAGE_SERVICE = "WEB COVERAGE SERVICE (WCS)"
    WEB_FEATURE_SERVICE = "WEB FEATURE SERVICE (WFS)"
    WEB_MAP_SERVICE = "WEB MAP SERVICE (WMS)"
    WEB_MAP_TILE_SERVICE = "WEB MAP TILE SERVICE (WMTS)"
    ALGORITHM_DOCUMENTATION = "ALGORITHM DOCUMENTATION"
    ALGORITHM_THEORETICAL_BASIS_DOCUMENT = "ALGORITHM THEORETICAL BASIS DOCUMENT (ATBD)"
    ANOMALIES = "ANOMALIES"
    CASE_STUDY = "CASE STUDY"
    DATA_CITATION_POLICY = "DATA CITATION POLICY"
    DATA_QUALITY = "DATA QUALITY"
    DATA_RECIPE = "DATA RECIPE"
    DELIVERABLES_CHECKLIST = "DELIVERABLES CHECKLIST"
    GENERAL_DOCUMENTATION = "GENERAL DOCUMENTATION"
    HOW_TO = "HOW-TO"
    IMPORTANT_NOTICE = "IMPORTANT NOTICE"
    INSTRUMENT_SENSOR_CALIBRATION_DOCUMENTATION = "INSTRUMENT/SENSOR CALIBRATION DOCUMENTATION"
    MICRO_ARTICLE = "MICRO ARTICLE"
    PI_DOCUMENTATION = "PI DOCUMENTATION"
    PROCESSING_HISTORY = "PROCESSING HISTORY"
    PRODUCT_HISTORY = "PRODUCT HISTORY"
    PRODUCT_QUALITY_ASSESSMENT = "PRODUCT QUALITY ASSESSMENT"
    PRODUCT_USAGE = "PRODUCT USAGE"
    PRODUCTION_HISTORY = "PRODUCTION HISTORY"
    PUBLICATIONS = "PUBLICATIONS"
    READ_ME = "READ-ME"
    REQUIREMENTS_AND_DESIGN = "REQUIREMENTS AND DESIGN"
    SCIENCE_DATA_PRODUCT_SOFTWARE_DOCUMENTATION = "SCIENCE DATA PRODUCT SOFTWARE DOCUMENTATION"
    SCIENCE_DATA_PRODUCT_VALIDATION = "SCIENCE DATA PRODUCT VALIDATION"
    USER_FEEDBACK_PAGE = "USER FEEDBACK PAGE"
    USERS_GUIDE = "USER'S GUIDE"
    DMR_PLUS_PLUS = "DMR++"
    DMR_PLUS_PLUS_MISSING_DATA = "DMR++ MISSING DATA"


class MimeTypeEnum(str, Enum):
    APPLICATION_JSON = "application/json"
    APPLICATION_XML = "application/xml"
    APPLICATION_X_NETCDF = "application/x-netcdf"
    APPLICATION_X_HDFEOS = "application/x-hdfeos"
    APPLICATION_GML_XML = "application/gml+xml"
    APPLICATION_VND_GOOGLE_EARTH_KML_XML = "application/vnd.google-earth.kml+xml"
    IMAGE_GIF = "image/gif"
    IMAGE_TIFF = "image/tiff"
    IMAGE_BMP = "image/bmp"
    TEXT_CSV = "text/csv"
    TEXT_XML = "text/xml"
    APPLICATION_PDF = "application/pdf"
    APPLICATION_X_HDF = "application/x-hdf"
    APPLICATION_X_HDF5 = "application/x-hdf5"
    APPLICATION_OCTET_STREAM = "application/octet-stream"
    APPLICATION_VND_GOOGLE_EARTH_KMZ = "application/vnd.google-earth.kmz"
    IMAGE_JPEG = "image/jpeg"
    IMAGE_PNG = "image/png"
    IMAGE_VND_COLLADA_XML = "image/vnd.collada+xml"
    TEXT_HTML = "text/html"
    TEXT_PLAIN = "text/plain"
    APPLICATION_ZIP = "application/zip"
    APPLICATION_GZIP = "application/gzip"
    APPLICATION_TAR = "application/tar"
    APPLICATION_TAR_GZIP = "application/tar+gzip"
    APPLICATION_TAR_ZIP = "application/tar+zip"
    APPLICATION_VND_OPENDAP_DAP4_DMRPP_XML = "application/vnd.opendap.dap4.dmrpp+xml"
    NOT_PROVIDED = "Not provided"


class CollectionReferenceType(BaseModel):
    """A reference to a collection metadata record's short name and version, or entry title to which this granule metadata record belongs."""

    ShortName: str | None = Field(
        None, min_length=1, max_length=85, description="The collection's short name as per the UMM-C."
    )
    Version: str | None = Field(
        None, min_length=1, max_length=80, description="The collection's version as per the UMM-C."
    )
    EntryTitle: str | None = Field(
        None, min_length=1, max_length=1030, description="The collections entry title as per the UMM-C."
    )


class AccessConstraintsType(BaseModel):
    """Information about any physical constraints for accessing the data set."""

    Description: str | None = Field(
        None,
        min_length=1,
        max_length=4000,
        description="Free-text description of the constraint. In ECHO 10, this field is called RestrictionComment. Additional detailed instructions on how to access the granule data may be entered in this field.",
    )
    Value: float = Field(
        description="Numeric value that is used with Access Control Language (ACLs) to restrict access to this granule. For example, a provider might specify a granule level ACL that hides all granules with a value element set to 15. In ECHO, this field is called RestrictionFlag."
    )


class ChecksumType(BaseModel):
    """Allows the provider to provide a checksum value and checksum algorithm name to allow the user to calculate the checksum."""

    Value: str = Field(min_length=1, max_length=128, description="Describes the checksum value for a file.")
    Algorithm: ChecksumAlgorithmEnum = Field(
        description="The algorithm name by which the checksum was calculated. This allows the user to re-calculate the checksum to verify the integrity of the downloaded data."
    )


class BaseFileType(BaseModel):
    """This entity contains the basic characteristics of the file or files that comprise the granule."""

    Name: str = Field(min_length=1, max_length=1024, description="This field describes the name of the actual file.")
    SizeInBytes: int | None = Field(
        None,
        description="The size in Bytes of the volume of data contained in the granule. Bytes are defined as eight bits. Please use this element instead of or inclusive with the Size element. The issue with the size element is that if CMR data providers use a unit other than Bytes, end users don't know how the granule size was calculated. For example, if the unit was MegaBytes, the size could be calculated by using 1000xE2 Bytes (MegaBytes) or 1024xE2 Bytes (mebibytes) and therefore there is no systematic way to know the actual size of a granule by using the granule metadata record.",
    )
    Size: float | None = Field(
        None,
        description="The size of the volume of data contained in the granule. Please use the SizeInBytes element either instead of this one or inclusive of this one. The issue with the size element is that if CMR data providers use a unit other than Bytes, end users don't know how the granule size was calculated. For example, if the unit was MegaBytes, the size could be calculated by using 1000xE2 Bytes (MegaBytes) or 1024xE2 Bytes (mebibytes) and therefore there is no systematic way to know the actual size of a granule by using the granule metadata record.",
    )
    SizeUnit: FileSizeUnitEnum | None = Field(None, description="The unit of the file size.")
    Format: str | None = Field(
        None,
        min_length=1,
        max_length=80,
        description="This element defines a single format for a distributable artifact.",
    )

    MimeType: MimeTypeEnum | None = Field(None, description="The mime type of the resource.")
    Checksum: ChecksumType | None = Field(
        None, description="Allows the provider to provide the checksum value for the file."
    )

    @model_validator(mode="after")
    def validate_size_requirements(self):
        """If you use Size, you must also use SizeInBytes and SizeUnit."""
        if self.Size is not None and self.SizeUnit is None:
            raise ValueError("If you provide Size, you must also use SizeUnit.")

        if self.Size is not None and self.SizeInBytes is None:
            raise ValueError("If you provide Size, you must also use SizeInBytes.")
        return self


class FileType(BaseFileType):
    """This entity contains the characteristics of the file or files that comprise the granule, including the granule file format and file size."""

    FormatType: FormatTypeEnum | None = Field(
        None,
        description="Allows the provider to state whether the distributable item's format is its native format or another supported format.",
    )


class FilePackageType(BaseFileType):
    """This entity stores information about the file package for this granule. A file package is something like a tar or zip file."""

    Files: list[FileType] | None = Field(
        None,
        min_length=1,
        description="Allows the provider to add the list of the files that are included in this one.",
    )


class IdentifierType(BaseModel):
    """This entity contains the identifier for the granule, including the identifier name, identifier type, and identifier value."""

    Identifier: str = Field(min_length=1, max_length=1024, description="The identifier value.")
    IdentifierType: IdentifierTypeEnum = Field(description="The enumeration of known identifier types.")
    IdentifierName: str | None = Field(None, min_length=1, max_length=1024, description="The name of the identifier.")


class DataGranuleType(BaseModel):
    """This entity stores the basic descriptive characteristics associated with a granule."""

    ArchiveAndDistributionInformation: list[FilePackageType | FileType] | None = Field(
        None,
        min_length=1,
        description="A list of the file(s) or file package(s) that make up the granule. A file package is something like a tar or zip file.",
    )
    ReprocessingPlanned: str | None = Field(
        None,
        min_length=1,
        max_length=80,
        description="Granule level, stating what reprocessing may be performed on this granule.",
    )
    ReprocessingActual: str | None = Field(
        None,
        min_length=1,
        max_length=80,
        description="Granule level, stating what reprocessing has been performed on this granule.",
    )
    DayNightFlag: DayNightFlagEnum = Field(
        description="This attribute is used to identify if a granule was collected during the day, night (between  sunset and sunrise) or both."
    )
    ProductionDateTime: ISODateTime = Field(description="The date and time a specific granule was produced by a PGE.")
    Identifiers: list[IdentifierType] | None = Field(
        None, min_length=1, description="This holds any granule identifiers the provider wishes to provide."
    )

    @field_validator("Identifiers", mode="after")
    def validate_unique_identifiers(cls, v: list[IdentifierType] | None) -> list[IdentifierType] | None:
        if v is not None:
            seen = set()
            for item in v:
                key = (item.Identifier, item.IdentifierType, item.IdentifierName)
                if key in seen:
                    raise ValueError("All identifiers must be unique")
                seen.add(key)
        return v


class PGEVersionClassType(BaseModel):
    """This entity stores basic descriptive characteristics related to the Product Generation Executable associated with a granule."""

    PGEName: str | None = Field(
        None, min_length=1, max_length=1024, description="Name of product generation executable."
    )
    PGEVersion: str = Field(
        min_length=1,
        max_length=50,
        description="Version of the product generation executable that produced the granule.",
    )


class PointType(BaseModel):
    """A spatial point element consisting of a longitude and latitude."""

    Longitude: LongitudeType = Field(description="The longitude value of a spatially referenced point, in degrees.")
    Latitude: LatitudeType = Field(description="The latitude value of a spatially referenced point, in degrees.")


class BoundingRectangleType(BaseModel):
    """This entity contains the coordinates that define a bounding rectangle for a granule."""

    WestBoundingCoordinate: LongitudeType = Field(description="The western bounding coordinate of the rectangle.")
    NorthBoundingCoordinate: LatitudeType = Field(description="The northern bounding coordinate of the rectangle.")
    EastBoundingCoordinate: LongitudeType = Field(description="The eastern bounding coordinate of the rectangle.")
    SouthBoundingCoordinate: LatitudeType = Field(description="The southern bounding coordinate of the rectangle.")


class BoundaryType(BaseModel):
    """A boundary is a set of points connected by straight lines representing a polygon on the earth."""

    Points: list[PointType] = Field(
        min_length=3,
        description="A boundary is set of points connected by straight lines representing a polygon on the earth. It takes a minimum of three points to make a boundary. Points must be specified in counter-clockwise order and closed (the first and last vertices are the same).",
    )


class ExclusiveZoneType(BaseModel):
    """This entity contains boundaries for regions excluded from the main boundary of a GPolygon."""

    Boundaries: list[BoundaryType] = Field(
        min_length=1,
        description="Contains the excluded boundaries from the GPolygon.",
    )


class GPolygonType(BaseModel):
    """A GPolygon specifies an area on the earth represented by a main boundary with optional boundaries for regions excluded from the main boundary."""

    Boundary: BoundaryType = Field(description="The main boundary of the polygon representing the area on earth.")
    ExclusiveZone: ExclusiveZoneType | None = None


class LineType(BaseModel):
    """A line area contains at least two points representing the horizontal spatial coverage."""

    Points: list[PointType] = Field(
        min_length=2,
        description="A line area contains at least two points representing the horizontal spatial coverage.",
    )


class GeometryType(BaseModel):
    """This entity holds the geometry representing the spatial coverage information of a granule."""

    Points: list[PointType] | None = Field(
        None, min_length=1, description="The horizontal spatial coverage of a point."
    )
    BoundingRectangles: list[BoundingRectangleType] | None = Field(
        None,
        min_length=1,
        description="This entity holds the horizontal spatial coverage of a bounding box.",
    )
    GPolygons: list[GPolygonType] | None = Field(
        None,
        min_length=1,
        description="A GPolygon specifies an area on the earth represented by a main boundary with optional boundaries for regions excluded from the main boundary.",
    )
    Lines: list[LineType] | None = Field(
        None,
        min_length=1,
        description="This entity holds the horizontal spatial coverage of a line. A line area contains at least two points.",
    )

    @field_validator("BoundingRectangles", mode="after")
    def validate_bounding_rectangles(cls, bounding_rectangles: list[BoundingRectangleType] | None):
        """Ensure each bounding rectangle is unique."""
        if bounding_rectangles is not None:
            seen = set()
            for item in bounding_rectangles:
                key = (
                    item.WestBoundingCoordinate,
                    item.NorthBoundingCoordinate,
                    item.EastBoundingCoordinate,
                    item.SouthBoundingCoordinate,
                )
                if key in seen:
                    raise ValueError("All bounding rectangles must be unique")
                seen.add(key)
        return bounding_rectangles


class OrbitType(BaseModel):
    """This entity stores orbital coverage information of the granule."""

    AscendingCrossing: LongitudeType = Field(
        description="Equatorial crossing on the ascending pass in decimal degrees longitude. The convention we've been using is it's the first included ascending crossing if one is included, and the prior ascending crossing if none is included (e.g. descending half orbits)."
    )
    StartLatitude: LatitudeType = Field(description="Granule's starting latitude.")
    StartDirection: OrbitDirectionEnum = Field(description="Ascending or descending. Valid input: 'A' or 'D'")
    EndLatitude: LatitudeType = Field(description="Granule's ending latitude.")
    EndDirection: OrbitDirectionEnum = Field(description="Ascending or descending. Valid input: 'A' or 'D'")


class TrackPassTileType(BaseModel):
    """This entity contains pass and tile information for track-based spatial coverage."""

    Pass: int = Field(
        description="A pass number identifies a subset of a granule's spatial extent. This element holds a pass number that exists in the granule and will allow a user to search by pass number that is contained within a cycle number. While trying to keep this generic for all to use, this comes from a SWOT requirement where a pass represents a 1/2 orbit."
    )
    Tiles: list[str] | None = Field(
        None,
        min_length=1,
        description="A tile is a subset of a pass' spatial extent. This element holds a list of tile identifiers that exist in the granule and will allow a user to search by tile identifier that is contained within a pass number within a cycle number. Though intended to be generic, this comes from a SWOT mission requirement where a tile is a spatial extent that encompasses either a square scanning swath to the left or right of the ground track or a rectangle that includes a full scanning swath both to the left and right of the ground track.",
    )


class TrackType(BaseModel):
    """This entity stores track information of the granule based on orbital cycles and passes."""

    Cycle: int = Field(
        description="An integer that represents a specific set of orbital spatial extents defined by passes and tiles. Though intended to be generic, this comes from a SWOT mission requirement where each cycle represents a set of 1/2 orbits. Each 1/2 orbit is called a 'pass'. During science mode, a cycle represents 21 days of 14 full orbits or 588 passes."
    )
    Passes: list[TrackPassTileType] | None = Field(
        None,
        min_length=1,
        description="A pass number identifies a subset of a granule's spatial extent. This element holds a list of pass numbers and their tiles that exist in the granule. It will allow a user to search by pass number and its tiles that are contained with in a cycle number. While trying to keep this generic for all to use, this comes from a SWOT requirement where a pass represents a 1/2 orbit. This element will then hold a list of 1/2 orbits and their tiles that together represent the granule's spatial extent.",
    )


class HorizontalSpatialDomainType(BaseModel):
    """This entity stores the horizontal spatial coverage of a granule including geometry, orbit, and track information."""

    ZoneIdentifier: str | None = Field(
        None,
        min_length=1,
        max_length=80,
        description="The appropriate numeric or alpha code used to identify the various zones in the granule's grid coordinate system.",
    )
    Geometry: GeometryType | None = Field(
        None,
        description="This entity holds the geometry representing the spatial coverage information of a granule.",
    )
    Orbit: OrbitType | None = Field(
        None,
        description="This entity stores orbital coverage information of the granule. This coverage is an alternative way of expressing granule spatial coverage. This information supports orbital backtrack searching on a granule.",
    )
    Track: TrackType | None = Field(
        None,
        description="This element stores track information of the granule. Track information is used to allow a user to search for granules whose spatial extent is based on an orbital cycle, pass, and tile mapping. Though it is derived from the SWOT mission requirements, it is intended that this element type be generic enough so that other missions can make use of it. While track information is a type of spatial domain, it is expected that the metadata provider will provide geometry information that matches the spatial extent of the track information.",
    )


class RangeDateTimeType(BaseModel):
    """Stores the data acquisition start and end date/time for a granule."""

    BeginningDateTime: str = Field(description="The time when the temporal coverage period being described began.")
    EndingDateTime: str | None = Field(
        None,
        description="The time when the temporal coverage period being described ended.",
    )


class TemporalExtentType(BaseModel):
    """This class contains attributes which describe the temporal extent of a granule."""

    RangeDateTime: RangeDateTimeType | None = Field(
        None,
        description="Stores the data acquisition start and end date/time for a granule.",
    )
    SingleDateTime: str | None = Field(None, description="Stores the data acquisition date/time for a granule.")

    @model_validator(mode="after")
    def validate_temporal_extent(self):
        """Either RangeDateTime or SingleDateTime must be provided, but not both."""
        if (self.RangeDateTime is None and self.SingleDateTime is None) or (
            self.RangeDateTime is not None and self.SingleDateTime is not None
        ):
            raise ValueError("Either RangeDateTime or SingleDateTime must be provided, but not both.")
        return self


class VerticalSpatialDomainType(BaseModel):
    """This represents the domain value and type for the granule's vertical spatial domain."""

    Type: VerticalSpatialDomainTypeEnum = Field(
        description="Describes the type of the area of vertical space covered by the granule locality."
    )
    Value: str | None = Field(
        None,
        min_length=1,
        max_length=80,
        description="Describes the extent of the area of vertical space covered by the granule. Use this for Atmosphere profiles or for a specific value.",
    )
    MinimumValue: str | None = Field(
        None,
        min_length=1,
        max_length=80,
        description="Describes the extent of the area of vertical space covered by the granule. Use this and MaximumValue to represent a range of values (Min and Max).",
    )
    MaximumValue: str | None = Field(
        None,
        min_length=1,
        max_length=80,
        description="Describes the extent of the area of vertical space covered by the granule. Use this and MinimumValue to represent a range of values (Min and Max).",
    )
    Unit: VerticalUnitEnum | None = Field(None, description="Describes the unit of the vertical extent value.")

    @model_validator(mode="after")
    def validate_vertical_spatial_domain(self):
        """Either Value or both MinimumValue and MaximumValue must be provided."""
        if self.Value is not None and (self.MinimumValue is not None or self.MaximumValue is not None):
            raise ValueError("Either Value or both MinimumValue and MaximumValue must be provided.")
        return self


class SpatialExtentType(BaseModel):
    """This class contains attributes which describe the spatial extent of a granule."""

    GranuleLocalities: list[GranuleLocalityType] | None = Field(
        None,
        min_length=1,
        description="This entity stores information used at the granule level to describe the labeling of granules with compounded time/space text values and which are subsequently used to define more phenomenological-based granules, thus the locality type and description are contained.",
    )
    HorizontalSpatialDomain: HorizontalSpatialDomainType | None = Field(
        None,
        description="This represents the granule horizontal spatial domain information.",
    )
    VerticalSpatialDomains: list[VerticalSpatialDomainType] | None = Field(
        None,
        min_length=1,
        description="This represents the domain value and type for the granule's vertical spatial domain.",
    )


class OrbitCalculatedSpatialDomainType(BaseModel):
    """This entity is used to store the characteristics of the orbit calculated spatial domain."""

    OrbitalModelName: str | None = Field(
        None,
        min_length=1,
        max_length=80,
        description="The reference to the orbital model to be used to calculate the geo-location of this data in order to determine global spatial extent.",
    )
    OrbitNumber: int | None = Field(
        None,
        description="The orbit number to be used in calculating the spatial extent of this data.",
    )
    BeginOrbitNumber: int | None = Field(None, description="Orbit number at the start of the data granule.")
    EndOrbitNumber: int | None = Field(None, description="Orbit number at the end of the data granule.")
    EquatorCrossingLongitude: LongitudeType | None = Field(
        None,
        description="This attribute represents the terrestrial longitude of the descending equator crossing.",
    )
    EquatorCrossingDateTime: datetime | None = Field(
        None,
        description="This attribute represents the date and time of the descending equator crossing.",
    )

    @model_validator(mode="after")
    def validate_orbit_calculated_spatial_domain(self):
        if (self.BeginOrbitNumber is None) != (self.EndOrbitNumber is None):
            raise ValueError(
                "Both BeginOrbitNumber and EndOrbitNumber must be provided together if either is provided."
            )
        if self.OrbitNumber is not None and self.BeginOrbitNumber is not None:
            raise ValueError("Cannot provide both OrbitNumber and BeginOrbitNumber/EndOrbitNumber.")
        if not any(
            [
                self.OrbitalModelName,
                self.OrbitNumber,
                self.BeginOrbitNumber,
                self.EquatorCrossingLongitude,
                self.EquatorCrossingDateTime,
            ]
        ):
            raise ValueError("At least one attribute value must be provided.")

        if self.EquatorCrossingDateTime is not None:
            validate_iso_datetime(self.EquatorCrossingDateTime)

        return self


# TODO[LIBSDC-676]: Determine implementation for these values
class QAStatsType(BaseModel):
    """Represents the quality assurance statistics for a granule."""

    QAPercentMissingData: float | None = Field(
        None,
        ge=0,
        le=100,
        description="Granule level % missing data. This attribute can be repeated for individual parameters within a granule.",
    )
    QAPercentOutOfBoundsData: float | None = Field(
        None,
        ge=0,
        le=100,
        description="Granule level % out of bounds data. This attribute can be repeated for individual parameters within a granule.",
    )
    QAPercentInterpolatedData: float | None = Field(
        None,
        ge=0,
        le=100,
        description="Granule level % interpolated data. This attribute can be repeated for individual parameters within a granule.",
    )
    QAPercentCloudCover: float | None = Field(
        None,
        ge=0,
        le=100,
        description="This attribute is used to characterize the cloud cover amount of a granule. This attribute may be repeated for individual parameters within a granule. (Note - there may be more than one way to define a cloud or it's effects within a product containing several parameters; i.e. this attribute may be parameter specific).",
    )

    @model_validator(mode="after")
    def validate_qa_stats(self):
        if not any(
            [
                self.QAPercentMissingData,
                self.QAPercentOutOfBoundsData,
                self.QAPercentInterpolatedData,
                self.QAPercentCloudCover,
            ]
        ):
            raise ValueError("At least one QA statistic must be provided.")
        return self


class QAFlagsType(BaseModel):
    """Represents the quality assurance flags for a granule."""

    AutomaticQualityFlag: QualityFlagEnum | None = Field(
        None,
        description="The granule level flag applying generally to the granule and specifically to parameters the granule level. When applied to parameter, the flag refers to the quality of that parameter for the granule (as applicable). The parameters determining whether the flag is set are defined by the developer and documented in the Quality Flag Explanation.",
    )
    AutomaticQualityFlagExplanation: str | None = Field(
        None,
        min_length=1,
        max_length=2048,
        description="A text explanation of the criteria used to set automatic quality flag; including thresholds or other criteria.",
    )
    OperationalQualityFlag: OperationalFlagEnum | None = Field(
        None,
        description="The granule level flag applying both generally to a granule and specifically to parameters at the granule level. When applied to parameter, the flag refers to the quality of that parameter for the granule (as applicable). The parameters determining whether the flag is set are defined by the developers and documented in the QualityFlagExplanation.",
    )
    OperationalQualityFlagExplanation: str | None = Field(
        None,
        min_length=1,
        max_length=2048,
        description="A text explanation of the criteria used to set operational quality flag; including thresholds or other criteria.",
    )
    ScienceQualityFlag: ScienceFlagEnum | None = Field(
        None,
        description="Granule level flag applying to a granule, and specifically to parameters. When applied to parameter, the flag refers to the quality of that parameter for the granule (as applicable). The parameters determining whether the flag is set are defined by the developers and documented in the Quality Flag Explanation.",
    )
    ScienceQualityFlagExplanation: str | None = Field(
        None,
        min_length=1,
        max_length=2048,
        description="A text explanation of the criteria used to set science quality flag; including thresholds or other criteria.",
    )

    @model_validator(mode="after")
    def validate_qa_flags(self):
        if not any([self.AutomaticQualityFlag, self.OperationalQualityFlag, self.ScienceQualityFlag]):
            raise ValueError("At least one quality flag value must be provided.")


class MeasuredParameterType(BaseModel):
    """This entity contains the name of the geophysical parameter expressed in the data as well as associated quality flags and quality statistics."""

    ParameterName: str = Field(
        min_length=1,
        max_length=250,
        description="The measured science parameter expressed in the data granule.",
    )
    QAStats: QAStatsType | None = Field(None, description="The associated quality statistics.")
    QAFlags: QAFlagsType | None = Field(None, description="The associated quality flags.")


class CharacteristicType(BaseModel):
    """This entity contains the name and value for any granule specific characteristic."""

    Name: str = Field(min_length=1, max_length=80, description="The name of the characteristic.")
    Value: str = Field(min_length=1, max_length=80, description="The value of the characteristic.")


class InstrumentType(BaseModel):
    """This entity contains the instrument associated with the acquisition of the granule."""

    ShortName: str = Field(
        min_length=1,
        max_length=80,
        description="The short name of the instrument. This short name must match an instrument in the parent collection.",
    )
    Characteristics: list[CharacteristicType] | None = Field(
        None,
        min_length=1,
        description="This entity contains the name and value for any granule specific characteristics for the instrument/sensor associated with the granule.",
    )
    ComposedOf: list["InstrumentType"] | None = Field(
        None,
        min_length=1,
        description="A reference to a child instrument or sensor. The child instrument information is represented by child entities.",
    )
    OperationalModes: list[Annotated[str, Field(min_length=1, max_length=20)]] | None = Field(
        None,
        min_length=1,
        description="A list of modes the instrument can operate in. This is a granule level entity. The operational modes in the granule must exist in the referenced collection instrument.",
    )


class PlatformType(BaseModel):
    """A reference to a platform in the parent collection that is associated with the acquisition of the granule."""

    ShortName: str = Field(
        min_length=1,
        max_length=80,
        description="The short name of the platform. The platform must exist in the parent collection. For example, Platform types may include (but are not limited to): ADEOS-II, AEM-2, Terra, Aqua, Aura, BALLOONS, BUOYS, C-130, DEM, DMSP-F1, etc.",
    )
    Instruments: list[InstrumentType] | None = Field(
        None,
        min_length=1,
        description="References to the devices in the parent collection that were used to measure or record data, including direct human observation.",
    )


class ProjectType(BaseModel):
    """The name of the scientific program, field campaign, or project from which the data were collected."""

    ShortName: str = Field(
        min_length=1,
        max_length=40,
        description="The short name of the project. This short name must match a project in the parent collection.",
    )
    Campaigns: list[Annotated[str, Field(min_length=1, max_length=40)]] | None = Field(
        None,
        min_length=1,
        description="The campaign associated with the project. This campaign must exist in the parent collection project.",
    )


class AdditionalAttributeType(BaseModel):
    """Reference to an additional attribute in the parent collection with granule specific values."""

    Name: str = Field(
        min_length=1,
        max_length=80,
        description="The name of the additional attribute. This name must match an additional attribute name in the parent collection.",
    )
    Values: list[str] = Field(
        min_length=1,
        description="The values of the additional attribute. These values will override the values in the parent collection for this granule.",
    )


class TilingCoordinateType(BaseModel):
    """This entity stores the minimum and maximum values for a coordinate in the tiling identification system."""

    MinimumValue: float = Field(description="The minimum value for the coordinate in the tiling identification system.")
    MaximumValue: float | None = Field(
        None,
        description="The maximum value for the coordinate in the tiling identification system.",
    )


class TilingIdentificationSystemType(BaseModel):
    """This entity stores the tiling identification system for the granule."""

    TilingIdentificationSystemName: TilingSystemNameEnum = Field(
        description="The name of the tiling identification system. This name must match a tiling identification system name in the parent collection."
    )
    Coordinate1: TilingCoordinateType = Field(description="The first coordinate in the tiling identification system.")
    Coordinate2: TilingCoordinateType = Field(description="The second coordinate in the tiling identification system.")


class RelatedUrlType(BaseModel):
    """This element describes any data/service related URLs that provide additional context or access to the granule."""

    URL: str = Field(min_length=1, max_length=1024, description="The URL for the relevant resource.")
    Type: RelatedUrlTypeEnum = Field(
        description="The type of the related URL (e.g. GET DATA, VIEW RELATED INFORMATION)."
    )
    Subtype: RelatedUrlSubTypeEnum | None = Field(
        None,
        description="The subtype of the related URL. This further categorizes the URL Type.",
    )
    Description: str | None = Field(
        None,
        min_length=1,
        max_length=4000,
        description="Description of the web page at this URL.",
    )
    Format: str | None = Field(
        None,
        min_length=1,
        max_length=80,
        description="The format of the resource at the related URL.",
    )
    MimeType: MimeTypeEnum | None = Field(None, description="The mime type of the resource at the related URL.")
    Size: float | None = Field(None, description="The size of the resource at the related URL.")
    SizeUnit: FileSizeUnitEnum | None = Field(
        None, description="The unit of the size of the resource at the related URL."
    )

    @model_validator(mode="after")
    def validate_size_requirements(self):
        """If you use Size, you must also use SizeUnit."""
        if self.Size is not None and self.SizeUnit is None:
            raise ValueError("If you provide Size, you must also use SizeUnit.")
        return self


class MetadataSpecificationType(BaseModel):
    """This entity stores the metadata schema name, version, and schema URL for the metadata specification."""

    URL: str = Field(
        default="https://cdn.earthdata.nasa.gov/umm/granule/v1.6.6",
        description="The URL for the UMM-G schema.",
    )
    Name: str = Field(default="UMM_G", description="The name of the metadata schema.")
    Version: str = Field(default="1.6.6", description="The version of the metadata schema.")


class ProviderDateType(BaseModel):
    """This entity stores the date and type of a granule event such as creation, update, or deletion."""

    Date: ISODateTime = Field(description="This is the date that an event associated with the granule occurred.")
    Type: ProviderDateTypeEnum = Field(
        description="This is the type of event associated with the date. For example, Create or Update."
    )


class UMMGranule(BaseModel):
    """The Unified Metadata Model (UMM) for Granule (UMM-G) defines the metadata structure for describing individual data granules in NASA's Common Metadata Repository (CMR)."""

    # Required properties
    GranuleUR: str = Field(
        max_length=250,
        min_length=1,
        description="The Universal Reference ID of the granule referred by the data provider. This ID is unique per data provider.",
    )
    ProviderDates: list[ProviderDateType] = Field(
        min_length=1,
        max_length=4,
        description="Dates related to activities involving the the granule and the data provider database with the exception for Delete. For Create, Update, and Insert the date is the date that the granule file is created, updated, or inserted into the provider database by the provider. Delete is the date that the CMR should delete the granule metadata record from its repository.",
    )
    CollectionReference: CollectionReferenceType
    MetadataSpecification: MetadataSpecificationType = Field(default_factory=MetadataSpecificationType)

    # Optional properties
    AccessConstraints: AccessConstraintsType | None = Field(
        None,
        description="Allows the author to constrain access to the granule. Some words that may be used in this element's value include: Public, In-house, Limited, None. The value field is used for special ACL rules (Access Control Lists (http://en.wikipedia.org/wiki/Access_control_list)). For example it can be used to hide metadata when it isn't ready for public consumption.",
    )
    DataGranule: DataGranuleType | None = Field(
        None, description="This entity stores basic descriptive characteristics associated with a granule."
    )
    PGEVersionClass: PGEVersionClassType | None = Field(
        None,
        description="This entity stores basic descriptive characteristics related to the Product Generation Executable associated with a granule.",
    )
    TemporalExtent: TemporalExtentType | None = Field(
        None,
        description="This class contains attributes which describe the temporal extent of a granule. Temporal Extent includes either a Range Date Time, or a Single Date Time",
    )
    SpatialExtent: SpatialExtentType | None = Field(
        None,
        description="This class contains attributes which describe the spatial extent of a granule. Spatial Extent includes any or all of Granule Localities, Horizontal Spatial Domain, and Vertical Spatial Domain.",
    )
    OrbitCalculatedSpatialDomains: list[OrbitCalculatedSpatialDomainType] | None = Field(
        None,
        min_length=1,
        description="This entity is used to store the characteristics of the orbit calculated spatial domain to include the model name, orbit number, start and stop orbit number, equator crossing date and time, and equator crossing longitude.",
    )
    MeasuredParameters: list[MeasuredParameterType] | None = Field(
        None,
        min_length=1,
        description="This entity contains the name of the geophysical parameter expressed in the data as well as associated quality flags and quality statistics. The quality statistics element contains measures of quality for the granule. The parameters used to set these measures are not preset and will be determined by the data producer. Each set of measures can occur many times either for the granule as a whole or for individual parameters. The quality flags contain the science, operational and automatic quality flags which indicate the overall quality assurance levels of specific parameter values within a granule.",
    )
    Platforms: list[PlatformType] | None = Field(
        None,
        min_length=1,
        description="A reference to a platform in the parent collection that is associated with the acquisition of the granule. The platform must exist in the parent collection. For example, Platform types may include (but are not limited to): ADEOS-II, AEM-2, Terra, Aqua, Aura, BALLOONS, BUOYS, C-130, DEM, DMSP-F1,etc.",
    )
    Projects: list[ProjectType] | None = Field(
        None,
        min_length=1,
        description="The name of the scientific program, field campaign, or project from which the data were collected. This element is intended for the non-space assets such as aircraft, ground systems, balloons, sondes, ships, etc. associated with campaigns. This element may also cover a long term project that continuously creates new data sets — like MEaSUREs from ISCCP and NVAP or CMARES from MISR. Project also includes the Campaign sub-element to support multiple campaigns under the same project.",
    )
    AdditionalAttributes: list[AdditionalAttributeType] | None = Field(
        None,
        min_length=1,
        description="Reference to an additional attribute in the parent collection. The attribute reference may contain a granule specific value that will override the value in the parent collection for this granule. An attribute with the same name must exist in the parent collection.",
    )
    InputGranules: list[Annotated[str, Field(min_length=1, max_length=500)]] | None = Field(
        None,
        min_length=1,
        description="This entity contains the identification of the input granule(s) for a specific granule.",
    )
    TilingIdentificationSystem: TilingIdentificationSystemType | None = Field(
        None,
        description="This entity stores the tiling identification system for the granule. The tiling identification system information is an alternative way to express granule's spatial coverage based on a certain two dimensional coordinate system defined by the providers. The name must match the name in the parent collection.",
    )
    CloudCover: float | None = Field(
        None,
        description="A percentage value indicating how much of the area of a granule (the EOSDIS data unit) has been obscured by clouds. It is worth noting that there are many different measures of cloud cover within the EOSDIS data holdings and that the cloud cover parameter that is represented in the archive is dataset-specific.",
    )
    RelatedUrls: list[RelatedUrlType] | None = Field(
        None,
        min_length=1,
        description="This element describes any data/service related URLs that include project home pages, services, related data archives/servers, metadata extensions, direct links to online software packages, web mapping services, links to images, or other data.",
    )
    NativeProjectionNames: list[ProjectionNameEnum] | None = Field(
        None, description="Represents the native projection of the granule if the granule has a native projection."
    )
    GridMappingNames: list[Annotated[str, Field(min_length=1, max_length=1024)]] | None = Field(
        None, description="Represents the native grid mapping of the granule, if the granule is gridded."
    )

    @classmethod
    def from_dataset(cls, input_dataset: xr.Dataset, **kwargs):
        transformer = UMMGDatasetTransformer(input_dataset, **kwargs)
        return transformer.umm_granule


# TODO[LIBSDC-675]: Update this class with real dataset information
class UMMGDatasetTransformer:
    """
    CLass containing methods to go from an input dataset to UMMGranule objects.

    """

    def __init__(self, input_dataset: xr.Dataset, log_warnings: bool = False):
        """
        Create the transformer object from input_dataset.

        The UMM granule is stored in the attribute umm_granule.

        Parameters
        ----------
        input_dataset : xr.Dataset
            The input dataset read from a netCDF file to convert into a UMMGranule object.
        log_warnings : bool
            Indicates whether warnings while extracting values from the dataset should be logged out,
            or only stored in the "warnings" attribute. Note that this does not affect validation
            warnings or errors, only warnings from the transformation step within the class.

        """
        self.dataset_attrs = input_dataset.attrs
        filepath = input_dataset.encoding.get("source")
        self.science_variable_names = [var for var in input_dataset.data_vars]

        self.log_warnings = log_warnings
        self.warnings = []

        self.umm_granule = self._to_umm_granule(filepath)

    def _warn(self, message: str):
        if self.log_warnings:
            logger.warning(message)
        self.warnings.append(message)

    def extract_granule_ur(self) -> str:
        """Extract GranuleUR from dataset attributes."""
        granule_id = self.dataset_attrs.get("ProductID", None)
        granule_id = self.dataset_attrs.get("GranuleID", granule_id)
        if granule_id is None:
            self._warn("No GranuleID found in dataset attributes; using 'Unknown_GranuleID' as GranuleUR.")
            granule_id = "Unknown_GranuleID"
        return granule_id

    def extract_provider_dates(self) -> list[ProviderDateType]:
        """Extract provider dates from dataset attributes."""
        # Not totally sure what the correct dates for this in the dataset will be
        provider_datetime_attributes = ["ProductionDateTime"]
        output_dates = []
        for attribute in provider_datetime_attributes:
            if attribute in self.dataset_attrs:
                output_dates.append(
                    ProviderDateType(Date=self.dataset_attrs[attribute], Type=ProviderDateTypeEnum.CREATE)
                )

        if not output_dates:
            self._warn(f"Unable to find ProviderDate values from attributes {provider_datetime_attributes} in dataset.")
            output_dates.append(ProviderDateType(Date=datetime.now(UTC).isoformat(), Type=ProviderDateTypeEnum.CREATE))
        return output_dates

    def extract_collection_reference(self) -> CollectionReferenceType:
        """Extract collection reference from dataset attributes."""
        short_name = self.dataset_attrs.get("CollectionShortName", None)
        version = self.dataset_attrs.get("CollectionVersion", None)
        entry_title = self.dataset_attrs.get("EntryTitle", None)

        if version is None:
            version = self.dataset_attrs.get("version", None)

        return CollectionReferenceType(ShortName=short_name, Version=version, EntryTitle=entry_title)

    def extract_access_constraints(self) -> AccessConstraintsType | None:
        """Extract access constraints from dataset attributes."""
        return None

    def extract_data_granule(self, filepath: str | None = None) -> DataGranuleType | None:
        """Extract data granule information from dataset attributes."""
        file_type = None

        if filepath:
            # Get file path and information
            filepath = Path(filepath)

            # Get file size if file exists
            size_bytes = None
            if filepath.exists():
                size_bytes = filepath.stat().st_size
            else:
                self._warn(f"File {filepath} does not exist, cannot determine size")

            file_type = FileType(
                Name=filepath.name,
                SizeInBytes=size_bytes,
                Format=self.dataset_attrs.get("Format", "NetCDF-4"),
                MimeType=MimeTypeEnum.APPLICATION_X_NETCDF,
                FormatType=FormatTypeEnum.NATIVE,
            )

        production_datetime = self.dataset_attrs.get("ProductionDateTime", None)

        if file_type is not None:
            if production_datetime is None:
                production_datetime = datetime.now(UTC).isoformat()
                self._warn("No production datetime found while creating DataGranule; using current time.")

            data_granule = DataGranuleType(
                ArchiveAndDistributionInformation=[file_type],
                DayNightFlag=DayNightFlagEnum.UNSPECIFIED,
                ProductionDateTime=production_datetime,
            )
        else:
            data_granule = None

        return data_granule

    def extract_pge_version_class(self) -> PGEVersionClassType | None:
        """Extract PGE version information from dataset attributes."""
        return None

    def extract_temporal_extent(self) -> TemporalExtentType | None:
        """Extract temporal extent from dataset attributes."""
        temporal_extent = None
        start_date = self.dataset_attrs.get("RangeBeginningDate", None)
        end_date = self.dataset_attrs.get("RangeEndingDate", None)
        start_time = self.dataset_attrs.get("RangeBeginningTime", None)
        end_time = self.dataset_attrs.get("RangeEndingTime", None)

        production_datetime = self.dataset_attrs.get("ProductionDateTime", None)
        print(f"Production datetime: {production_datetime}")

        if start_date and start_time and end_date and end_time:
            beginning_datetime = f"{start_date}T{start_time}Z"
            ending_datetime = f"{end_date}T{end_time}Z"
            range = RangeDateTimeType(BeginningDateTime=beginning_datetime, EndingDateTime=ending_datetime)
            temporal_extent = TemporalExtentType(RangeDateTime=range)

        # Default to the range, but take the single time if it exists
        if production_datetime and temporal_extent is None:
            temporal_extent = TemporalExtentType(SingleDateTime=production_datetime)

        return temporal_extent

    def extract_spatial_extent(self) -> SpatialExtentType | None:
        """Extract spatial extent from dataset attributes."""
        spatial_extent = None

        # Process lat/lon data
        latitude_max = self.dataset_attrs.get("geospatial_lat_max", None)
        latitude_min = self.dataset_attrs.get("geospatial_lat_min", None)
        longitude_max = self.dataset_attrs.get("geospatial_lon_max", None)
        longitude_min = self.dataset_attrs.get("geospatial_lon_min", None)

        if None not in (latitude_max, latitude_min, longitude_max, longitude_min):
            horizontal_spatial_domain = HorizontalSpatialDomainType(
                BoundingRectangle=BoundingRectangleType(
                    WestBoundingCoordinate=float(longitude_min),
                    EastBoundingCoordinate=float(longitude_max),
                    NorthBoundingCoordinate=float(latitude_max),
                    SouthBoundingCoordinate=float(latitude_min),
                )
            )
            spatial_extent = SpatialExtentType(HorizontalSpatialDomain=horizontal_spatial_domain)

        return spatial_extent

    def extract_orbit_calculated_spatial_domains(self) -> list[OrbitCalculatedSpatialDomainType] | None:
        """Extract orbit calculated spatial domains from dataset attributes."""
        return None

    def extract_measured_parameters(self) -> list[MeasuredParameterType] | None:
        """Extract measured parameters from dataset variables."""
        # how do you determine science vs metadata variables?

        measured_parameters = []
        for var_name in self.science_variable_names:
            # QF may be in a different variable - will need to create a mapping in that
            # case
            measured_parameters.append(MeasuredParameterType(ParameterName=var_name))

        return measured_parameters if measured_parameters else None

    def extract_platforms(self) -> list[PlatformType] | None:
        """Extract platform information from dataset attributes."""
        if "PlatformShortName" in self.dataset_attrs:
            platform_name = self.dataset_attrs.get("PlatformShortName")
        else:
            self._warn("No Platform found, defaulting to 'NOAA-22' platform")
            platform_name = "NOAA-22"

        # Does Instruments include all the instruments, or only the ones used to collect the dataset?
        if "ProjectShortName" in self.dataset_attrs:
            project_name = self.dataset_attrs.get("ProjectShortName")
            instrument_list = [InstrumentType(ShortName=project_name)]
        else:
            self._warn("No Platform found, defaulting to 'Libera' instrument")
            instrument_list = [InstrumentType(ShortName="Libera")]

        platform = None
        if platform_name:
            platform = [PlatformType(ShortName=platform_name, Instruments=instrument_list)]
        return platform

    def extract_projects(self) -> list[ProjectType] | None:
        """Extract project information from dataset attributes."""
        if "ProjectShortName" in self.dataset_attrs:
            project_name = self.dataset_attrs.get("ProjectShortName")
        else:
            self._warn("No Project found, defaulting to 'Libera'")
            project_name = "Libera"
        project = [ProjectType(ShortName=project_name)]

        return project

    def extract_additional_attributes(self) -> list[AdditionalAttributeType] | None:
        """Extract additional attributes from dataset attributes."""
        return None

    def extract_input_granules(self) -> list[str] | None:
        """Extract input granules from dataset attributes."""

        return None

    def extract_tiling_identification_system(self) -> TilingIdentificationSystemType | None:
        """Extract tiling identification system from dataset attributes."""
        return None

    def extract_cloud_cover(self) -> float | None:
        """Extract cloud cover percentage from dataset attributes."""
        return None

    def extract_related_urls(self) -> list[RelatedUrlType] | None:
        """Extract related URLs from dataset attributes."""
        return None

    def extract_native_projection_names(self) -> list[ProjectionNameEnum] | None:
        """Extract native projection names from dataset attributes."""
        return None

    def extract_grid_mapping_names(self) -> list[str] | None:
        """Extract grid mapping names from dataset attributes."""
        return None

    def _to_umm_granule(self, filepath: str | None = None) -> UMMGranule:
        """Build complete UMM-G granule from dataset."""
        return UMMGranule(
            GranuleUR=self.extract_granule_ur(),
            ProviderDates=self.extract_provider_dates(),
            CollectionReference=self.extract_collection_reference(),
            AccessConstraints=self.extract_access_constraints(),
            DataGranule=self.extract_data_granule(filepath),
            PGEVersionClass=self.extract_pge_version_class(),
            TemporalExtent=self.extract_temporal_extent(),
            SpatialExtent=self.extract_spatial_extent(),
            OrbitCalculatedSpatialDomains=self.extract_orbit_calculated_spatial_domains(),
            MeasuredParameters=self.extract_measured_parameters(),
            Platforms=self.extract_platforms(),
            Projects=self.extract_projects(),
            AdditionalAttributes=self.extract_additional_attributes(),
            InputGranules=self.extract_input_granules(),
            TilingIdentificationSystem=self.extract_tiling_identification_system(),
            CloudCover=self.extract_cloud_cover(),
            RelatedUrls=self.extract_related_urls(),
            NativeProjectionNames=self.extract_native_projection_names(),
            GridMappingNames=self.extract_grid_mapping_names(),
        )
