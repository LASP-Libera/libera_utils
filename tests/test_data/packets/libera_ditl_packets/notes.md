# APID Contents

Day-in-the-Life (DITL) ground-test CCSDS telemetry. Files use an 8-byte record header per packet;
set `SKIP_PACKET_HEADER_BYTES=8` (or pass `skip_header_bytes=8` to Space Packet Parser) when
reading.

## `ccsds_2025_318_13_16_34`

DITL camera / WFOV science event (2025 DOY 318, ~13:16:34 UTC in the filename). Primary test
fixture for LIBSDC-679: `icie_wfov_sci` (1040) includes **1510** packets with **one duplicate**
`PACKET_ICIE_TIME` (`2028-02-15T13:17:03.095685`). L1A processing warns and deduplicates to
**1509** unique packet times.

This file requires multiple XTCE definitions to fully parse:

- ICIE packets (`icie_*`) → `LIBERA_PACKET_DEFINITION`
- PEV packets (`pev_*`, APID 215) → `LIBERA_PEV_PACKET_DEFINITION`
- PEC packets (`pec_*`, APID 116) → `LIBERA_PEC_PACKET_DEFINITION`

Several APIDs are present in the raw stream but are not defined in `LiberaApid` and do not parse
with the XTCE files above (notably APID <1000, 1006, 1008, 1200).

**Total packets**: 2813 (raw CCSDS headers, `skip_header_bytes=8`)

| APID | Packet Name       | Count | XTCE Definition              |
| ---- | ----------------- | ----- | ---------------------------- |
| 105  | UNKNOWN_APID_105  | 56    | —                            |
| 116  | UNKNOWN_APID_116  | 54    | LIBERA_PEC_PACKET_DEFINITION |
| 215  | UNKNOWN_APID_215  | 56    | LIBERA_PEV_PACKET_DEFINITION |
| 1000 | pev_sw_stat       | 1     | LIBERA_PEV_PACKET_DEFINITION |
| 1002 | pec_sw_stat       | 1     | LIBERA_PEC_PACKET_DEFINITION |
| 1006 | UNKNOWN_APID_1006 | 1     | —                            |
| 1008 | UNKNOWN_APID_1008 | 1     | —                            |
| 1017 | icie_seq_hk       | 1     | LIBERA_PACKET_DEFINITION     |
| 1019 | icie_fp_hk        | 1     | LIBERA_PACKET_DEFINITION     |
| 1026 | icie_log_msg      | 13    | LIBERA_PACKET_DEFINITION     |
| 1036 | icie_rad_sample   | 223   | LIBERA_PACKET_DEFINITION     |
| 1040 | icie_wfov_sci     | 1510  | LIBERA_PACKET_DEFINITION     |
| 1048 | icie_axis_sample  | 222   | LIBERA_PACKET_DEFINITION     |
| 1051 | icie_crit_hk      | 55    | LIBERA_PACKET_DEFINITION     |
| 1057 | icie_nom_hk       | 55    | LIBERA_PACKET_DEFINITION     |
| 1058 | UNKNOWN_APID_1058 | 1     | LIBERA_PACKET_DEFINITION     |
| 1059 | icie_ana_hk       | 55    | LIBERA_PACKET_DEFINITION     |
| 1060 | icie_temp_hk      | 1     | LIBERA_PACKET_DEFINITION     |
| 1200 | UNKNOWN_APID_1200 | 506   | —                            |
