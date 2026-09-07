# Demuxed ground-test CCSDS fixtures

Single-APID CCSDS files in the demuxed ground naming convention,
`LIBERA_SDC_<apid>_ccsds_<yyyy>_<doy>_<hh>_<mm>_<ss>`. No record header precedes the CCSDS
primary header, so parse these with `skip_header_bytes=0`.

Derived from `../libera_ditl_packets/ccsds_2025_318_13_16_34` by stripping the 8-byte record
header from each packet and splitting by APID. The time fields in these basenames are the bin
start of the source capture, not the packet times inside — packet times are in 2028, matching
the DITL simulated clock.

| File                                      | APID                    | Packets | Exercises                                                               |
| ----------------------------------------- | ----------------------- | ------- | ----------------------------------------------------------------------- |
| `LIBERA_SDC_1036_ccsds_2025_318_13_00_00` | 1036 `icie_rad_sample`  | 40      | epoch + period data times                                               |
| `LIBERA_SDC_1048_ccsds_2025_318_13_00_00` | 1048 `icie_axis_sample` | 40      | per-sample data times                                                   |
| `LIBERA_SDC_1057_ccsds_2025_318_13_00_00` | 1057 `icie_nom_hk`      | 55      | packet-time-only APID (not data-time indexed)                           |
| `LIBERA_SDC_1040_ccsds_2025_318_13_00_00` | 1040 `icie_wfov_sci`    | 40      | WFOV with an in-window `SOP` (source packets 250-290)                   |
| `LIBERA_SDC_1040_ccsds_2025_318_13_20_00` | 1040 `icie_wfov_sci`    | 40      | WFOV with no `SOP`, so data times are unavailable (source packets 0-40) |

The two 1040 files carry the same APID and differ only in their bin field, as several bins per
day do in real data.

The WFOV-with-`SOP` file yields a camera data time of 2028-02-14T04:23:03, about 33 hours before
its own packet times. Camera images are downlinked well after they are taken, so a data time
earlier than the packet time is expected, not a fixture defect.
