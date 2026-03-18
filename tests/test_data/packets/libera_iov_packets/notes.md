# APID Contents

## `ccsds_2025_346_13_29_47`

IOV SWC (short wave cal) event. Contains PEV/PEC software status packets
(`pev_sw_stat`, `pec_sw_stat`) alongside routine ICIE science and housekeeping.

This file requires multiple XTCE definitions to fully parse:

- ICIE packets (`icie_*`) → `LIBERA_PACKET_DEFINITION`
- PEV packets (`pev_*`) → `LIBERA_PEV_PACKET_DEFINITION`
- PEC packets (`pec_*`) → `LIBERA_PEC_PACKET_DEFINITION`

**Total packets**: 3614 (ICIE: 3177, PEV: 222, PEC: 215)

| APID | Packet Name       | Count | XTCE Definition              |
| ---- | ----------------- | ----- | ---------------------------- |
| 112  | UNKNOWN_APID_112  | 1     | LIBERA_PEC_PACKET_DEFINITION |
| 115  | UNKNOWN_APID_115  | 242   | LIBERA_PEC_PACKET_DEFINITION |
| 116  | UNKNOWN_APID_116  | 243   | LIBERA_PEC_PACKET_DEFINITION |
| 212  | UNKNOWN_APID_212  | 13    | LIBERA_PEV_PACKET_DEFINITION |
| 215  | UNKNOWN_APID_215  | 243   | LIBERA_PEV_PACKET_DEFINITION |
| 216  | UNKNOWN_APID_216  | 243   | LIBERA_PEV_PACKET_DEFINITION |
| 217  | UNKNOWN_APID_217  | 243   | LIBERA_PEV_PACKET_DEFINITION |
| 218  | UNKNOWN_APID_218  | 243   | LIBERA_PEV_PACKET_DEFINITION |
| 1000 | pev_sw_stat       | 222   | LIBERA_PEV_PACKET_DEFINITION |
| 1002 | pec_sw_stat       | 215   | LIBERA_PEC_PACKET_DEFINITION |
| 1013 | icie_sw_stat      | 241   | LIBERA_PACKET_DEFINITION     |
| 1017 | icie_seq_hk       | 24    | LIBERA_PACKET_DEFINITION     |
| 1018 | UNKNOWN_APID_1018 | 2     | LIBERA_PACKET_DEFINITION     |
| 1019 | icie_fp_hk        | 24    | LIBERA_PACKET_DEFINITION     |
| 1026 | icie_log_msg      | 36    | LIBERA_PACKET_DEFINITION     |
| 1036 | icie_rad_sample   | 620   | LIBERA_PACKET_DEFINITION     |
| 1037 | icie_axis_hk      | 242   | LIBERA_PACKET_DEFINITION     |
| 1044 | icie_cal_sample   | 269   | LIBERA_PACKET_DEFINITION     |
| 1048 | icie_axis_sample  | 970   | LIBERA_PACKET_DEFINITION     |
| 1051 | icie_crit_hk      | 241   | LIBERA_PACKET_DEFINITION     |
| 1057 | icie_nom_hk       | 241   | LIBERA_PACKET_DEFINITION     |
| 1058 | UNKNOWN_APID_1058 | 1     | LIBERA_PACKET_DEFINITION     |
| 1059 | icie_ana_hk       | 241   | LIBERA_PACKET_DEFINITION     |
| 1060 | icie_temp_hk      | 25    | LIBERA_PACKET_DEFINITION     |
