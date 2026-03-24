# L1A Processing

L1A processing takes data from CCSDS packets to L1A Dataset objects suitable for writing as NetCDF.

Each L1A data product contains a single packet type (APID), decoded and reformatted for easier usage in the processing system.
The structure of L1A products is controlled by L1A processing packet configurations, which instruct the L1A pipeline
how to restructure the packet data. For example, packets that contain multiple samples of a single data point
are restructured to pivot the samples into their own data array with a time dimension for the sample time.
Fields that appear only once per packet are left associated with the "PACKET" index dimension (no coordinate).

**Steps:**

1. Read packet data using Space Packet Parser
2. Fetch the L1A `PacketConfiguration` object to configure L1A processing
3. Create an `xr.Dataset` according to the `PacketConfiguration`, expanding multi-sample fields into
   sample-indexed arrays and aggregating binary blob fields as configured
4. Look up the L1A product definition path via `get_l1a_product_definition_path(apid)` and write the
   Dataset to NetCDF using `write_libera_data_product()`

## Configuration Overview

L1A processing draws on three layers of configuration:

- **Global runtime config** (`config.json`): controls file paths and behaviour flags such as
  `SKIP_PACKET_HEADER_BYTES`. All values can be overridden by environment variables of the same name.
- **L1A processing configs YAML** (`l1a_processing_configs.yml`): defines per-APID packet structure
  (sample groups, aggregation groups, time field mappings). Its path is set by the
  `LIBERA_L1A_PROCESSING_CONFIGS_PATH` config key.
- **XTCE packet definitions** (e.g. `icie_xtce_tlm.xml`): define the binary field layout at the
  bit-field level, consumed by Space Packet Parser. Each `PacketConfiguration` references the
  appropriate XTCE file via its `packet_definition_config_key`.

## Ground Testing Data

By default, `SKIP_PACKET_HEADER_BYTES` is `0` in `config.json`. This is correct for flight and
production data delivered through the SDC downlink pipeline.

Ground testing data generated from hardware-in-the-loop systems (e.g. Hydra/FSW) prepends an extra
**8-byte header** to each packet before the standard CCSDS primary header. To process this data, set
`SKIP_PACKET_HEADER_BYTES` to `8`. This is a **global** setting that affects all packets in a
processing run.

Set it via environment variable before running:

```bash
export SKIP_PACKET_HEADER_BYTES=8
```

In test code, override it with `monkeypatch`:

```python
monkeypatch.setenv("SKIP_PACKET_HEADER_BYTES", "8")
```

The value is read once per call to `parse_packets_to_l1a_dataset()` and forwarded to Space Packet
Parser as `skip_header_bytes`.

## L1A Packet Processing Configurations

Per-APID processing configurations are defined in `l1a_processing_configs.yml` (path resolved from
`LIBERA_L1A_PROCESSING_CONFIGS_PATH`). Each entry is keyed by the APID name from the `LiberaApid`
enum. Configurations are lazily loaded and cached on the first call to `get_packet_config()`.

```python
from libera_utils.constants import LiberaApid
from libera_utils.l1a.l1a_packet_configs import get_packet_config

config = get_packet_config(LiberaApid.icie_axis_sample)
print(config.packet_time_coordinate)                   # → "PACKET_ICIE_TIME"
print(config.sample_groups[0].sample_time_dimension)   # → "AXIS_SAMPLE_ICIE_TIME"
```

### PacketConfiguration

The top-level object describing how to process one packet type.

| Field                          | Type                     | Description                                                                          |
| ------------------------------ | ------------------------ | ------------------------------------------------------------------------------------ |
| `packet_apid`                  | `LiberaApid` str         | APID name matching the `LiberaApid` enum (e.g. `"icie_axis_sample"`)                 |
| `packet_time_fields`           | `TimeFieldMapping`       | Packet-level timestamp field names                                                   |
| `packet_time_source`           | `SampleTimeSource`       | Clock source for packet timestamps: `ICIE`, `FPE`, or `JPSS`                         |
| `packet_definition_config_key` | str                      | Config key for the XTCE definition file path. Defaults to `LIBERA_PACKET_DEFINITION` |
| `sample_groups`                | list[`SampleGroup`]      | Zero or more sample group configurations                                             |
| `aggregation_groups`           | list[`AggregationGroup`] | Zero or more aggregation group configurations                                        |

The computed property `packet_time_coordinate` returns `PACKET_{time_source}_TIME`
(e.g. `PACKET_ICIE_TIME` for `packet_time_source: "ICIE"`). This is the non-dimension coordinate
attached to the `PACKET` dimension in the output Dataset.

Minimal example — a housekeeping packet with no samples or aggregations:

```yaml
icie_nom_hk:
  packet_apid: "icie_nom_hk"
  packet_time_fields:
    day_field: "ICIE__TM_DAY_NOM_HK"
    ms_field: "ICIE__TM_MS_NOM_HK"
    us_field: "ICIE__TM_US_NOM_HK"
  packet_definition_config_key: "LIBERA_PACKET_DEFINITION"
  packet_time_source: "ICIE"
```

A JPSS packet using a different XTCE definition and time source:

```yaml
jpss_sc_pos:
  packet_apid: "jpss_sc_pos"
  packet_time_fields:
    day_field: "DAYS"
    ms_field: "MSEC"
    us_field: "USEC"
  packet_definition_config_key: "JPSS_GEOLOCATION_PACKET_DEFINITION"
  packet_time_source: "JPSS"
  sample_groups:
    - ... # see SampleGroup below
```

### TimeFieldMapping

CCSDS packet timestamps are split across multiple integer fields. `TimeFieldMapping` names which
XTCE packet fields hold each component. All fields are optional; at least one must be present.

| Field       | Description              |
| ----------- | ------------------------ |
| `day_field` | Days since mission epoch |
| `s_field`   | Seconds within the day   |
| `ms_field`  | Milliseconds (additive)  |
| `us_field`  | Microseconds (additive)  |

When `TimeFieldMapping` is used in `time_field_patterns` inside a `SampleGroup` with
`sample_count > 1`, each field name must include `%i` as a placeholder for the sample index
(e.g. `"ICIE__AXIS_SAMPLE_TM_SEC%i"`).

### SampleTimeSource

Identifies which subsystem clock provides the timestamps for a packet or sample group.

| Value  | System                                                               |
| ------ | -------------------------------------------------------------------- |
| `ICIE` | Instrument Control and Interface Electronics (Libera main processor) |
| `FPE`  | Focal Plane Electronics (Libera detector subsystem)                  |
| `JPSS` | JPSS spacecraft system clock                                         |

The `time_source` value is embedded in dimension names. A `SampleGroup` with
`name: "RAD_SAMPLE"` and `time_source: "FPE"` produces the output dimension `RAD_SAMPLE_FPE_TIME`.

### SampleGroup

A `SampleGroup` describes one set of related samples within a packet that share timing
characteristics. Each group produces a new dimension in the output Dataset (the sample time
dimension), with all configured data fields mapped to it.

| Field                 | Type               | Description                                                      |
| --------------------- | ------------------ | ---------------------------------------------------------------- |
| `name`                | str                | Group identifier; used to build the sample time dimension name   |
| `sample_count`        | int                | Number of samples per packet                                     |
| `data_field_patterns` | list[str]          | XTCE field name patterns; use `%i` for the sample index          |
| `time_source`         | `SampleTimeSource` | Clock source for sample timestamps                               |
| `time_field_patterns` | `TimeFieldMapping` | Per-sample timestamp fields — **Timing Mode A**                  |
| `epoch_time_fields`   | `TimeFieldMapping` | Single epoch timestamp per packet — **Timing Mode B**            |
| `sample_period`       | int (µs)           | Fixed period between samples in microseconds — **Timing Mode B** |

Exactly one timing mode must be used; providing both or neither is a configuration error.

The computed property `sample_time_dimension` returns `{name}_{time_source}_TIME`
(e.g. `AXIS_SAMPLE_ICIE_TIME`).

After sample expansion, data field names in the output Dataset have `%i` removed (trailing
underscores are also stripped). For example, `ICIE__AXIS_AZ_FILT%i` becomes `ICIE__AXIS_AZ_FILT`.

A `{name}_packet_index` variable (integer, dimensioned by the sample time dimension) is always
created alongside the expanded data. It maps each sample back to its originating packet index in the
`PACKET` dimension, enabling efficient joins between per-packet housekeeping and per-sample science
data.

#### Timing Mode A — Explicit Per-Sample Timestamps

Use this mode when each sample within a packet carries its own timestamp fields. All field names in
`time_field_patterns` must contain `%i`. The processor iterates `i` from `0` to `sample_count - 1`,
resolves each field name, and produces a flat time array of length `n_packets × sample_count`.

```yaml
# AXIS_SAMPLE: 50 Az/El encoder samples, each with its own ICIE timestamp
icie_axis_sample:
  packet_apid: "icie_axis_sample"
  packet_time_fields:
    day_field: "ICIE__TM_DAY_AXIS_SAMPLE"
    ms_field: "ICIE__TM_MS_AXIS_SAMPLE"
    us_field: "ICIE__TM_US_AXIS_SAMPLE"
  sample_groups:
    - name: "AXIS_SAMPLE"
      time_field_patterns:
        s_field: "ICIE__AXIS_SAMPLE_TM_SEC%i"
        us_field: "ICIE__AXIS_SAMPLE_TM_SUB%i"
      data_field_patterns:
        - "ICIE__AXIS_AZ_FILT%i"
        - "ICIE__AXIS_EL_FILT%i"
      sample_count: 50
      time_source: "ICIE"
  packet_definition_config_key: "LIBERA_PACKET_DEFINITION"
  packet_time_source: "ICIE"
```

#### Timing Mode B — Epoch + Periodic Sampling

Use this mode when samples are evenly spaced from a single epoch timestamp recorded per packet.
`epoch_time_fields` names the fields holding the epoch (no `%i` placeholders). `sample_period` is
specified as an integer in **microseconds**. Sample times are computed as
`epoch + i × sample_period` for `i` in `0..sample_count-1`.

Note that `packet_time_source` and `time_source` can differ. The packet-level timestamp (from ICIE)
and the sample-level timing (from FPE) are tracked independently. This cross-source pattern is
typical for science packets where the instrument main processor records a coarse packet time but the
detector subsystem provides precise per-sample timing.

```yaml
# RAD_SAMPLE: 50 radiometer samples at 5 ms intervals, FPE-timed
icie_rad_sample:
  packet_apid: "icie_rad_sample"
  packet_time_fields:
    day_field: "ICIE__TM_DAY_RAD_SAMPLE"
    ms_field: "ICIE__TM_MS_RAD_SAMPLE"
    us_field: "ICIE__TM_US_RAD_SAMPLE"
  sample_groups:
    - name: "RAD_SAMPLE"
      epoch_time_fields:
        s_field: "ICIE__RAD_SAMP_START_HI"
        us_field: "ICIE__RAD_SAMP_START_LO"
      sample_period: 5000 # microseconds (5 ms between samples)
      data_field_patterns:
        - "ICIE__RAD_SAMPLE%i_0"
        - "ICIE__RAD_SAMPLE%i_1"
        - "ICIE__RAD_SAMPLE%i_2"
        - "ICIE__RAD_SAMPLE%i_3"
      sample_count: 50
      time_source: "FPE"
  packet_definition_config_key: "LIBERA_PACKET_DEFINITION"
  packet_time_source: "ICIE"
```

### AggregationGroup

Some packets carry large binary payloads that XTCE decodes into many numbered scalar fields (e.g.
972 individual single-byte fields for WFOV camera data). `AggregationGroup` reassembles these into
a single bytes-typed variable per packet, reducing variable count and simplifying downstream access.

| Field           | Type            | Description                                                         |
| --------------- | --------------- | ------------------------------------------------------------------- |
| `name`          | str             | Output variable name in the Dataset                                 |
| `field_pattern` | str             | XTCE field name pattern with `%i` for the field index               |
| `field_count`   | int             | Number of fields to aggregate (indices `0` to `field_count-1`)      |
| `dtype`         | numpy dtype str | Target dtype, e.g. `\|S972` for a 972-byte fixed-length bytes value |

The total byte size of all aggregated fields must equal `dtype.itemsize`. A `ValueError` is raised
at processing time if there is a mismatch.

```yaml
# WFOV_SCI: 972 individual byte fields reassembled into one 972-byte blob per packet
icie_wfov_sci:
  packet_apid: "icie_wfov_sci"
  packet_time_fields:
    day_field: "ICIE__TM_DAY_WFOV_SCI"
    ms_field: "ICIE__TM_MS_WFOV_SCI"
    us_field: "ICIE__TM_US_WFOV_SCI"
  aggregation_groups:
    - name: "ICIE__WFOV_DATA"
      field_pattern: "ICIE__WFOV_DATA_%i"
      field_count: 972
      dtype: "|S972"
  packet_definition_config_key: "LIBERA_PACKET_DEFINITION"
  packet_time_source: "ICIE"
```

## L1A Product Structure

This varies by packet but there is some consistent behavior:

- Every L1A product has a `"PACKET"` index dimension that is simply an index of packets

  > **Note:** Space Packet Parser (SPP) internally creates datasets with a lowercase `"packet"`
  > dimension. The pipeline immediately renames this to `"PACKET"` to conform to the SDC naming
  > standard (`SDC_PACKET_DIMENSION = "PACKET"` in `packets.py`). All downstream code and product
  > definitions must use `"PACKET"`.

- Every L1A product has a packet time coordinate with dimension `"PACKET"`
- Fields appearing once per packet are associated with the `"PACKET"` index dimension
- Every sample set (possibly multiple) has a sample time coordinate that is a dimension coordinate (coordinate name == dimension name)
- Every sample variable has a dimension for its sample time
- Samples taken at the same time (possibly across multiple fields) are associated with the same sample time dimension
- Every sample group has a `{name}_packet_index` variable (integer, same dimension as the sample
  data) that maps each sample back to its originating packet index in the `PACKET` dimension.
  This enables efficient joins between per-packet metadata and per-sample science data.

For example, for `N` packets, the `AXIS_SAMPLE` packet containing Azimuth and Elevation mechanism data
comes down with 50 Az and El samples per packet (a sample group). It's L1A product has:

```yaml
coordinates:
  # Packet timestamp
  PACKET_ICIE_TIME:
    dtype: datetime64[ns]
    dimensions: ["PACKET"]
    attributes:
      long_name: Packet timestamp from ICIE main processor
    encoding:
      units: nanoseconds since 1958-01-01
      calendar: standard
      dtype: int64
  # Sample timestamp
  AXIS_SAMPLE_ICIE_TIME:
    dtype: datetime64[ns]
    dimensions: ["AXIS_SAMPLE_ICIE_TIME"]
    attributes:
      long_name: Azimuth and elevation encoder sample timestamp
    encoding:
      units: nanoseconds since 1958-01-01
      calendar: standard
      dtype: int64

variables:
  # There are more variables not listed here

  # Per packet checksum
  ICIE__AXIS_SAMPLE_CHECKSUM:
    dtype: uint32
    dimensions: ["PACKET"]
    attributes:
      long_name: ICIE axis sample packet checksum
  # Azimuth samples
  ICIE__AXIS_AZ_FILT:
    dtype: float32
    dimensions: ["AXIS_SAMPLE_ICIE_TIME"]
    attributes:
      long_name: ICIE azimuth axis filtered encoder reading
      units: radians
  # Elevation samples
  ICIE__AXIS_EL_FILT:
    dtype: float32
    dimensions: ["AXIS_SAMPLE_ICIE_TIME"]
    attributes:
      long_name: ICIE elevation axis filtered encoder reading
      units: radians
  # Packet index for each sample
  AXIS_SAMPLE_packet_index:
    dtype: int64
    dimensions: ["AXIS_SAMPLE_ICIE_TIME"]
    attributes:
      long_name: Packet index for axis sample data
      comment: Maps each axis sample to its originating packet index
```
