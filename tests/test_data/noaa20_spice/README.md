# NOAA-20 SPICE configuration

Test data, not package data. Selected by the `noaa20_environment` fixture in
`tests/plugins/spice_fixtures.py`.

NOAA-20 is not a Libera spacecraft and is not a configuration the pipeline can select. This family
is kept because it is the only thing that exercises two paths nothing else covers:

- kernel generation from **real decoded spacecraft telemetry** (`ADGPSPOS*`, `ADCFAQ*`, and
  multipart `ADAET1DAY/MS/US` times), where the JPSS-4 path uses simulated STK output, and the
  quaternion sign-flip convention in the spacecraft CK config;
- geolocation validated against **CERES**, the only truth here that is neither simulated nor
  Libera's own.

It also demonstrates that kernel generation is driven by configuration rather than hardcoded to
jpss4.

Its value decreases as real Libera data arrives and it is expected to be retired after launch. The
frame kernel declares no measured misalignments, so it cannot exercise the geometry that LIBSDC-806
introduced; use the jpss4 family for anything alignment-related.

These files lived in `libera_utils/data/spice/noaa20/` until LIBSDC-703 and were shipped in the
wheel, which implied a runtime configuration that no caller could reach. Moving them here dropped
the dead entries at the same time: the unused TLE-based SPK config, the Az/El mechanism CK configs,
and the five static offset configs (the static offset test now runs against the shipped jpss4 set).
