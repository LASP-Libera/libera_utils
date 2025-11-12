"""Packet configurations for different LIBERA and JPSS packet types.

This module contains PacketConfiguration instances that define how to parse
different types of spacecraft and instrument packets into L1A datasets.
"""

from abc import ABC
from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum

import numpy as np

from libera_utils.constants import LiberaApid

# Registry for PacketConfiguration instances
_PACKET_CONFIG_REGISTRY: dict[LiberaApid, "PacketConfiguration"] = {}


def get_packet_config(apid: LiberaApid) -> "PacketConfiguration":
    """Get a PacketConfiguration instance by LiberaApid.

    Parameters
    ----------
    apid : LiberaApid
        The APID to look up the configuration for

    Returns
    -------
    PacketConfiguration
        The configuration for the given APID

    Raises
    ------
    KeyError
        If no configuration is registered for the given APID
    """
    if apid not in _PACKET_CONFIG_REGISTRY:
        raise KeyError(f"No PacketConfiguration registered for {apid}")
    return _PACKET_CONFIG_REGISTRY[apid]


def register_packet_config(cls: type["PacketConfiguration"]) -> type["PacketConfiguration"]:
    """Class decorator to register a PacketConfiguration subclass.

    Instantiates the class and registers it in the global registry.

    Parameters
    ----------
    cls : type[PacketConfiguration]
        The PacketConfiguration subclass to register

    Returns
    -------
    type[PacketConfiguration]
        The same class (unchanged)

    Raises
    ------
    ValueError
        If a configuration with the same APID is already registered
    """
    # Create an instance of the class
    instance = cls()

    # Register it
    if instance.packet_apid in _PACKET_CONFIG_REGISTRY:
        raise ValueError(f"PacketConfiguration for {instance.packet_apid} is already registered")
    _PACKET_CONFIG_REGISTRY[instance.packet_apid] = instance

    # Return the class (not the instance) to maintain decorator semantics
    return cls


@dataclass
class TimeFieldMapping:
    """Mapping of time field names to their roles in multipart timestamp conversion.

    This class defines which packet fields correspond to different time units
    (days, seconds, milliseconds, microseconds) and provides a property to
    generate the appropriate kwargs for the multipart_to_dt64 function.
    """

    day_field: str | None = None
    s_field: str | None = None
    ms_field: str | None = None
    us_field: str | None = None

    @property
    def multipart_kwargs(self) -> dict[str, str]:
        """Return kwargs dict for multipart_to_dt64 function.

        Returns
        -------
        dict[str, str]
            Dictionary with field parameter names as keys and field names as values,
            excluding any None values.
        """
        return {
            k: v
            for k, v in {
                "day_field": self.day_field,
                "s_field": self.s_field,
                "ms_field": self.ms_field,
                "us_field": self.us_field,
            }.items()
            if v is not None
        }


class SampleTimeSource(StrEnum):
    """Enumeration for sample timestamp sources."""

    ICIE = "ICIE"  # Libera main processor
    FPE = "FPE"  # Libera focal plane electronics
    JPSS = "JPSS"  # JPSS spacecraft system


@dataclass
class SampleGroup:
    """Configuration for a group of samples within a packet.

    This class defines how to parse a specific group of related samples
    that share timing characteristics within a packet.

    Attributes
    ----------
    name : str
        Name of the sample group (e.g., "AXIS_SAMPLE", "RAD_SAMPLE", "ADGPS"). This is used to name
        coordinates and dimensions.
    time_fields : TimeFieldMapping | None
        Mapping of time field patterns to their units for explicit per-sample timestamps.
        Use %i as placeholder for sample index in the field names.
    epoch_time_fields : TimeFieldMapping | None
        Mapping of time fields to units for a single epoch timestamp.
        Used with sample_period to calculate sample times.
    sample_period : timedelta | None
        Fixed time period between samples, used with epoch_time_fields.
    data_field_patterns : list[str]
        List of data field name patterns. Use %i for sample index if multiple samples.
    sample_count : int
        Number of samples per packet for this sample group.
    time_source : SampleTimeSource
        The source system for timestamps (e.g. ICIE, FPE, or SC).
    """

    name: str
    sample_count: int
    data_field_patterns: list[str]
    time_source: SampleTimeSource
    time_field_patterns: TimeFieldMapping | None = None
    epoch_time_fields: TimeFieldMapping | None = None
    sample_period: timedelta | None = None

    @property
    def sample_time_dimension(self) -> str:
        """Get the dimension name for this sample group.

        Returns
        -------
        str
            The dimension name, e.g. "AXIS_SAMPLE_ICIE_TIME"
        """
        return f"{self.name}_{self.time_source.value}_TIME"

    @property
    def sample_data_fields(self) -> list[str]:
        """Return the data field patterns with any %i placeholders removed
        (i.e. convert patterns to clean field names)

        For example, ICIE__AXIS_EL_FILT%i becomes ICIE__AXIS_EL_FILT
        """
        return [dfp.replace("%i", "") for dfp in self.data_field_patterns]

    def __post_init__(self):
        if self.sample_count < 1:
            raise ValueError("The sample_count must be > 0")
        if self.epoch_time_fields:
            if not self.sample_period:
                raise ValueError("You must provide sample_period for epoch_time_fields")
            # Check if any epoch time field names contain %i (they shouldn't)
            epoch_field_names = [
                f
                for f in [
                    self.epoch_time_fields.day_field,
                    self.epoch_time_fields.s_field,
                    self.epoch_time_fields.ms_field,
                    self.epoch_time_fields.us_field,
                ]
                if f is not None
            ]
            if any(["%i" in f for f in epoch_field_names]):
                raise ValueError("Epoch time fields should never contain %i as they are expected only once per packet")
        if self.epoch_time_fields and self.time_field_patterns:
            raise ValueError("Provide either epoch_time_fields or time_fields, not both")
        if not (self.epoch_time_fields or self.time_field_patterns):
            raise ValueError("Must provide one of either epoch_time_fields or time_fields")
        if self.sample_count > 1 and self.time_field_patterns:
            # Check if all time field names contain %i for multi-sample cases
            time_field_names = [
                f
                for f in [
                    self.time_field_patterns.day_field,
                    self.time_field_patterns.s_field,
                    self.time_field_patterns.ms_field,
                    self.time_field_patterns.us_field,
                ]
                if f is not None
            ]
            if not all(["%i" in f for f in time_field_names]):
                raise ValueError("Every time field must include %i when >1 samples are expected")


@dataclass
class AggregationGroup:
    """Configuration for aggregating multiple sequential fields into a single binary blob.

    This class defines how to combine multiple numbered fields (e.g., ICIE__WFOV_DATA_0
    through ICIE__WFOV_DATA_971) into a single bytes object per packet.

    Attributes
    ----------
    name : str
        Name for the aggregated variable (e.g., "ICIE__WFOV_DATA")
    field_pattern : str
        Pattern with %i placeholder for field index (e.g., "ICIE__WFOV_DATA_%i")
    field_count : int
        Expected number of fields to aggregate (e.g., 972)
    dtype : np.dtype
        Resulting numpy dtype for the aggregated data (e.g., np.dtype('|S972'))
    """

    name: str
    field_pattern: str
    field_count: int
    dtype: np.dtype = field(default_factory=lambda: np.dtype("object"))

    def __post_init__(self):
        if self.field_count < 1:
            raise ValueError("The field_count must be > 0")
        if "%i" not in self.field_pattern:
            raise ValueError("field_pattern must contain %i placeholder for field index")


@dataclass(frozen=True)
class PacketConfiguration(ABC):
    """Abstract base class for packet configurations.

    All packet configurations must subclass this and will be automatically
    registered when decorated with @register_packet_config.

    This class defines how to parse packets that may contain multiple groups
    of samples with their own timestamps, allowing for proper expansion and
    reshaping of the data.

    Attributes
    ----------
    packet_apid : LiberaApid
        The APID (Application Process Identifier) for the packet type
    packet_time_fields : TimeFieldMapping
        Mapping of packet timestamp fields to their time units for multipart_to_dt64 conversion.
    sample_groups : list[SampleGroup]
        List of sample group configurations for this packet type.
    aggregation_groups : list[AggregationGroup]
        List of aggregation group configurations for this packet type.
    packet_definition_config_key : str
        Configuration key to fetch the packet definition path from config.
        Defaults to "LIBERA_PACKET_DEFINITION".
    packet_time_source : SampleTimeSource
        The time source for packet timestamps.
    packet_generator_kwargs : dict
        Additional keyword arguments passed to the packet generator in space_packet_parser.
        Default is no additional kwargs.
    """

    packet_apid: LiberaApid
    packet_time_fields: TimeFieldMapping
    sample_groups: list[SampleGroup] = field(default_factory=list)
    aggregation_groups: list[AggregationGroup] = field(default_factory=list)
    packet_definition_config_key: str = "LIBERA_PACKET_DEFINITION"
    packet_time_source: SampleTimeSource = SampleTimeSource.ICIE
    packet_generator_kwargs: dict = field(default_factory=dict)

    @property
    def packet_time_coordinate(self) -> str:
        """Get the packet time coordinate name following the consistent pattern.

        Returns
        -------
        str
            The packet coordinate name, e.g. "PACKET_ICIE_TIME"
        """
        return f"PACKET_{self.packet_time_source.value}_TIME"

    def get_sample_group(self, name: str) -> SampleGroup:
        """Get a sample group by name"""
        for sg in self.sample_groups:
            if sg.name == name:
                return sg
        raise KeyError(f"No sample group with name {name}")


# Axis packet (azimuth and elevation encoders) contains many timestamped samples per packet
@register_packet_config
@dataclass(frozen=True)
class AxisPacketConfig(PacketConfiguration):
    """Configuration for ICIE Axis Sample packets."""

    packet_apid: LiberaApid = LiberaApid.icie_axis_sample
    packet_time_fields: TimeFieldMapping = field(
        default_factory=lambda: TimeFieldMapping(
            day_field="ICIE__TM_DAY_AXIS_SAMPLE",
            ms_field="ICIE__TM_MS_AXIS_SAMPLE",
            us_field="ICIE__TM_US_AXIS_SAMPLE",
        )
    )
    sample_groups: list[SampleGroup] = field(
        default_factory=lambda: [
            SampleGroup(
                name="AXIS_SAMPLE",
                time_field_patterns=TimeFieldMapping(
                    s_field="ICIE__AXIS_SAMPLE_TM_SEC%i",
                    us_field="ICIE__AXIS_SAMPLE_TM_SUB%i",
                ),
                data_field_patterns=[
                    "ICIE__AXIS_AZ_FILT%i",
                    "ICIE__AXIS_EL_FILT%i",
                ],
                sample_count=50,
                time_source=SampleTimeSource.ICIE,
            )
        ]
    )
    packet_definition_config_key: str = "LIBERA_PACKET_DEFINITION"
    packet_time_source: SampleTimeSource = SampleTimeSource.ICIE
    packet_generator_kwargs: dict = field(default_factory=lambda: {"skip_header_bytes": 8})


# Radiometer Sample Packet contains many periodic samples per packet for each of the radiometer channels
@register_packet_config
@dataclass(frozen=True)
class RadSamplePacketConfig(PacketConfiguration):
    """Configuration for ICIE Radiometer Sample packets."""

    packet_apid: LiberaApid = LiberaApid.icie_rad_sample
    packet_time_fields: TimeFieldMapping = field(
        default_factory=lambda: TimeFieldMapping(
            day_field="ICIE__TM_DAY_RAD_SAMPLE",
            ms_field="ICIE__TM_MS_RAD_SAMPLE",
            us_field="ICIE__TM_US_RAD_SAMPLE",
        )
    )
    sample_groups: list[SampleGroup] = field(
        default_factory=lambda: [
            SampleGroup(
                name="RAD_SAMPLE",
                epoch_time_fields=TimeFieldMapping(
                    s_field="ICIE__RAD_SAMP_START_HI",
                    us_field="ICIE__RAD_SAMP_START_LO",
                ),
                sample_period=timedelta(milliseconds=5),
                data_field_patterns=[
                    "ICIE__RAD_SAMPLE%i_0",
                    "ICIE__RAD_SAMPLE%i_1",
                    "ICIE__RAD_SAMPLE%i_2",
                    "ICIE__RAD_SAMPLE%i_3",
                ],
                sample_count=50,
                time_source=SampleTimeSource.FPE,
            )
        ]
    )
    packet_definition_config_key: str = "LIBERA_PACKET_DEFINITION"
    packet_time_source: SampleTimeSource = SampleTimeSource.ICIE
    packet_generator_kwargs: dict = field(default_factory=lambda: {"skip_header_bytes": 8})


# JPSS SC Position Packet config with separate sample groups for position/velocity and attitude
@register_packet_config
@dataclass(frozen=True)
class ScPosPacketConfig(PacketConfiguration):
    """Configuration for JPSS Spacecraft Position packets."""

    packet_apid: LiberaApid = LiberaApid.jpss_sc_pos
    packet_time_fields: TimeFieldMapping = field(
        default_factory=lambda: TimeFieldMapping(
            day_field="DAYS",
            ms_field="MSEC",
            us_field="USEC",
        )
    )
    sample_groups: list[SampleGroup] = field(
        default_factory=lambda: [
            SampleGroup(
                name="ADGPS",
                time_field_patterns=TimeFieldMapping(
                    day_field="ADAET1DAY",
                    ms_field="ADAET1MS",
                    us_field="ADAET1US",
                ),
                data_field_patterns=["ADGPSPOSX", "ADGPSPOSY", "ADGPSPOSZ", "ADGPSVELX", "ADGPSVELY", "ADGPSVELZ"],
                sample_count=1,
                time_source=SampleTimeSource.JPSS,
            ),
            SampleGroup(
                name="ADCFA",
                time_field_patterns=TimeFieldMapping(
                    day_field="ADAET2DAY",
                    ms_field="ADAET2MS",
                    us_field="ADAET2US",
                ),
                data_field_patterns=["ADCFAQ1", "ADCFAQ2", "ADCFAQ3", "ADCFAQ4"],
                sample_count=1,
                time_source=SampleTimeSource.JPSS,
            ),
        ]
    )
    packet_definition_config_key: str = "JPSS_GEOLOCATION_PACKET_DEFINITION"
    packet_time_source: SampleTimeSource = SampleTimeSource.JPSS


# WFOV Science Packet contains scientific data from the wide field of view camera
@register_packet_config
@dataclass(frozen=True)
class WfovSciPacketConfig(PacketConfiguration):
    """Configuration for ICIE WFOV Science packets."""

    packet_apid: LiberaApid = LiberaApid.icie_wfov_sci
    packet_time_fields: TimeFieldMapping = field(
        default_factory=lambda: TimeFieldMapping(
            day_field="ICIE__TM_DAY_WFOV_SCI",
            ms_field="ICIE__TM_MS_WFOV_SCI",
            us_field="ICIE__TM_US_WFOV_SCI",
        )
    )
    aggregation_groups: list[AggregationGroup] = field(
        default_factory=lambda: [
            AggregationGroup(
                name="ICIE__WFOV_DATA",
                field_pattern="ICIE__WFOV_DATA_%i",
                field_count=972,
                dtype=np.dtype("|S972"),
            )
        ]
    )
    packet_definition_config_key: str = "LIBERA_PACKET_DEFINITION"
    packet_time_source: SampleTimeSource = SampleTimeSource.ICIE
    packet_generator_kwargs: dict = field(default_factory=lambda: {"skip_header_bytes": 8})


# Nominal Housekeeping Packet contains routine telemetry data
@register_packet_config
@dataclass(frozen=True)
class NominalHkPacketConfig(PacketConfiguration):
    """Configuration for ICIE Nominal Housekeeping packets."""

    packet_apid: LiberaApid = LiberaApid.icie_nom_hk
    packet_time_fields: TimeFieldMapping = field(
        default_factory=lambda: TimeFieldMapping(
            day_field="ICIE__TM_DAY_NOM_HK",
            ms_field="ICIE__TM_MS_NOM_HK",
            us_field="ICIE__TM_US_NOM_HK",
        )
    )
    packet_definition_config_key: str = "LIBERA_PACKET_DEFINITION"
    packet_time_source: SampleTimeSource = SampleTimeSource.ICIE
    packet_generator_kwargs: dict = field(default_factory=lambda: {"skip_header_bytes": 8})


# Critical Housekeeping Packet contains important status information
@register_packet_config
@dataclass(frozen=True)
class CriticalHkPacketConfig(PacketConfiguration):
    """Configuration for ICIE Critical Housekeeping packets."""

    packet_apid: LiberaApid = LiberaApid.icie_crit_hk
    packet_time_fields: TimeFieldMapping = field(
        default_factory=lambda: TimeFieldMapping(
            day_field="ICIE__TM_DAY_CRIT_HK",
            ms_field="ICIE__TM_MS_CRIT_HK",
            us_field="ICIE__TM_US_CRIT_HK",
        )
    )
    packet_definition_config_key: str = "LIBERA_PACKET_DEFINITION"
    packet_time_source: SampleTimeSource = SampleTimeSource.ICIE
    packet_generator_kwargs: dict = field(default_factory=lambda: {"skip_header_bytes": 8})


# Temperature Housekeeping Packet contains thermal monitoring data
@register_packet_config
@dataclass(frozen=True)
class TempHkPacketConfig(PacketConfiguration):
    """Configuration for ICIE Temperature Housekeeping packets."""

    packet_apid: LiberaApid = LiberaApid.icie_temp_hk
    packet_time_fields: TimeFieldMapping = field(
        default_factory=lambda: TimeFieldMapping(
            day_field="ICIE__TM_DAY_TEMP_HK",
            ms_field="ICIE__TM_MS_TEMP_HK",
            us_field="ICIE__TM_US_TEMP_HK",
        )
    )
    packet_definition_config_key: str = "LIBERA_PACKET_DEFINITION"
    packet_time_source: SampleTimeSource = SampleTimeSource.ICIE
    packet_generator_kwargs: dict = field(default_factory=lambda: {"skip_header_bytes": 8})
