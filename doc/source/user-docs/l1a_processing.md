# L1A Processing

L1A processing takes data from CCSDS packets to L1A Dataset objects suitable for writing as NetCDF.

Each L1A data product contains a single packet type (APID), decoded and reformatted for easier usage in the processing system.
The structure of L1A products is controlled by L1A processing packet configurations, which instruct the L1A pipeline
how to restructure the packet data. For example, packets that contain multiple samples of a single data point
are restructured to pivot the samples into their own data array with a time dimension for the sample time.
Fields that appear only once per packet are left associated with the "packet" index dimension (no coordinate).

**Steps:**

1. Read packet data using Space Packet Parser
2. Fetch the L1A PacketConfiguration object to configure L1A processing
3. Create an xr.Dataset according to the PacketConfiguration
4. Write the Dataset to NetCDF

## L1A Product Structure

This varies by packet but there is some consistent behavior:

- Every L1A product has a "packet" index dimension that is simply an index of packets
- Every L1A product has a packet time coordinate with dimension "packet"
- Fields appearing once per packet are associated with the "packet" index dimension
- Every sample set (possibly multiple) has a sample time coordinate that is a dimension coordinate (coordinate name == dimension name)
- Every sample variable has dimension for its sample time
- Samples (possibly multiple) taken at the same time are associated with the same sample time dimension

For example, for `N` packets, the `AXIS_SAMPLE` packet containing Azimuth and Elevation mechanism data
comes down with 50 Az and El samples per packet (a sample group). It's L1A product has:

```yaml
coordinates:
  # Packet timestamp
  PACKET_ICIE_TIME:
    dtype: datetime64[ns]
    dimensions: ["packet"]
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
    dimensions: ["packet"]
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
