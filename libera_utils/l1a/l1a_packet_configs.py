"""Packet configurations for different LIBERA and JPSS packet types.

This module contains PacketConfiguration instances that define how to parse
different types of spacecraft and instrument packets into L1A datasets.

Configurations are loaded from a YAML file and cached for performance.
"""

from datetime import timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from libera_utils.constants import LiberaApid

# Registry for PacketConfiguration instances
_PACKET_CONFIG_REGISTRY: dict[int, "PacketConfiguration"] = {}


def get_packet_config(apid: LiberaApid | int) -> "PacketConfiguration":
    """Get a PacketConfiguration instance by LiberaApid.

    Configurations are lazily loaded from YAML on first access.

    Parameters
    ----------
    apid : LiberaApid | int
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
    try:
        return _PACKET_CONFIG_REGISTRY[apid]
    except KeyError:
        _load_configs_from_yaml()
        return _PACKET_CONFIG_REGISTRY[apid]


def _load_configs_from_yaml():
    """Load all packet configurations from YAML file and populate the registry."""
    yaml_path = Path(__file__).parent.parent / "data" / "l1a_processing_configs.yml"

    with open(yaml_path) as f:
        yaml_data = yaml.safe_load(f)

    for apid_name, config_data in yaml_data.items():
        config = PacketConfiguration(**config_data)
        _PACKET_CONFIG_REGISTRY[config.packet_apid] = config


class TimeFieldMapping(BaseModel):
    """Mapping of time field names to their roles in multipart timestamp conversion.

    This class defines which packet fields correspond to different time units
    (days, seconds, milliseconds, microseconds) and provides a property to
    generate the appropriate kwargs for the multipart_to_dt64 function.
    """

    model_config = ConfigDict(frozen=True)

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


class SampleGroup(BaseModel):
    """Configuration for a group of samples within a packet.

    This class defines how to parse a specific group of related samples
    that share timing characteristics within a packet.

    Attributes
    ----------
    name : str
        Name of the sample group (e.g., "AXIS_SAMPLE", "RAD_SAMPLE", "ADGPS"). This is used to name
        coordinates and dimensions.
    time_field_patterns : TimeFieldMapping | None
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

    model_config = ConfigDict(frozen=True)

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
        """Return the data field patterns with any %i placeholders removed.

        For example, ICIE__AXIS_EL_FILT%i becomes ICIE__AXIS_EL_FILT
        """
        return [dfp.replace("%i", "") for dfp in self.data_field_patterns]

    @field_validator("sample_period", mode="before")
    @classmethod
    def _convert_sample_period(cls, v: Any) -> timedelta | None:
        """Convert sample_period from microseconds (int) to timedelta."""
        if v is None:
            return None
        if isinstance(v, timedelta):
            return v
        if isinstance(v, int):
            return timedelta(microseconds=v)
        raise ValueError(f"sample_period must be int (microseconds) or timedelta, got {type(v)}")

    @field_validator("time_source", mode="before")
    @classmethod
    def _convert_time_source(cls, v: Any) -> SampleTimeSource:
        """Convert time_source from string to SampleTimeSource enum."""
        if isinstance(v, SampleTimeSource):
            return v
        if isinstance(v, str):
            return SampleTimeSource(v)
        raise ValueError(f"time_source must be str or SampleTimeSource, got {type(v)}")

    @model_validator(mode="after")
    def _validate_sample_group(self) -> "SampleGroup":
        """Validate SampleGroup constraints after all fields are set."""
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
            if any("%i" in f for f in epoch_field_names):
                raise ValueError("Epoch time fields should never contain %i as they are expected only once per packet")

        if self.epoch_time_fields and self.time_field_patterns:
            raise ValueError("Provide either epoch_time_fields or time_field_patterns, not both")

        if not (self.epoch_time_fields or self.time_field_patterns):
            raise ValueError("Must provide one of either epoch_time_fields or time_field_patterns")

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
            if not all("%i" in f for f in time_field_names):
                raise ValueError("Every time field must include %i when >1 samples are expected")

        return self


class AggregationGroup(BaseModel):
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

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    field_pattern: str
    field_count: int
    dtype: np.dtype = Field(default_factory=lambda: np.dtype("object"))

    @field_validator("dtype", mode="before")
    @classmethod
    def _convert_dtype(cls, v: Any) -> np.dtype:
        """Convert dtype from string to numpy dtype."""
        if isinstance(v, np.dtype):
            return v
        if isinstance(v, str):
            return np.dtype(v)
        raise ValueError(f"dtype must be str or np.dtype, got {type(v)}")

    @model_validator(mode="after")
    def _validate_aggregation_group(self) -> "AggregationGroup":
        """Validate AggregationGroup constraints after all fields are set."""
        if self.field_count < 1:
            raise ValueError("The field_count must be > 0")
        if "%i" not in self.field_pattern:
            raise ValueError("field_pattern must contain %i placeholder for field index")
        return self


class PacketConfiguration(BaseModel):
    """Base class for packet configurations.

    This class defines how to parse packets that may contain multiple groups
    of samples with their own timestamps, allowing for proper expansion and
    reshaping of the data.

    Configurations are loaded from YAML and cached for performance.

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
    """

    model_config = ConfigDict(frozen=True)

    packet_apid: LiberaApid
    packet_time_fields: TimeFieldMapping
    sample_groups: list[SampleGroup] = Field(default_factory=list)
    aggregation_groups: list[AggregationGroup] = Field(default_factory=list)
    packet_definition_config_key: str = "LIBERA_PACKET_DEFINITION"
    packet_time_source: SampleTimeSource = SampleTimeSource.ICIE

    @field_validator("packet_apid", mode="before")
    @classmethod
    def _convert_packet_apid(cls, v: Any) -> LiberaApid:
        """Convert packet_apid from string to LiberaApid enum."""
        if isinstance(v, LiberaApid):
            return v
        if isinstance(v, str):
            return LiberaApid[v]
        if isinstance(v, int):
            return LiberaApid(v)
        raise ValueError(f"packet_apid must be str, int, or LiberaApid, got {type(v)}")

    @field_validator("packet_time_source", mode="before")
    @classmethod
    def _convert_packet_time_source(cls, v: Any) -> SampleTimeSource:
        """Convert packet_time_source from string to SampleTimeSource enum."""
        if isinstance(v, SampleTimeSource):
            return v
        if isinstance(v, str):
            return SampleTimeSource(v)
        raise ValueError(f"packet_time_source must be str or SampleTimeSource, got {type(v)}")

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
        """Get a sample group by name.

        Parameters
        ----------
        name : str
            The name of the sample group to retrieve

        Returns
        -------
        SampleGroup
            The sample group with the given name

        Raises
        ------
        KeyError
            If no sample group with the given name exists
        """
        for sg in self.sample_groups:
            if sg.name == name:
                return sg
        raise KeyError(f"No sample group with name {name}")
