# NOM-HK ObsID trim fixture

Compact subset of DITL NOM-HK for integration tests of `libera_utils.l1a.nom_hk_trim`.

**Source granule:**
`LIBERA_L1A_NOM-HK-DECODED_V5-8-5RC1_20280213T020114_20280213T040013_R26163174745.nc`
(DITL Full, orbits 316_02–316_04)

**Fixture file:**
`LIBERA_L1A_NOM-HK-DECODED_V5-8-5RC1_20280213T021705_20280213T040005_R26163174745.nc`

The fixture is regenerated rather than sliced from the published granule, so it carries the
global attributes the current parser writes. Parse the granule's `input_files` (raw DITL CCSDS,
`ccsds_2025_316_02_01_15` … `ccsds_2025_316_03_59_06`) with
`parse_packets_to_l1a_dataset(..., 1057, ground_data=True, skip_header_bytes=8)`, keep the packets
whose `PACKET_ICIE_TIME` is in the previous fixture, and write it with
`write_libera_data_product(..., strict=True)`. The filename keeps the source granule's version
and revision.

Keeps RAD cal runs with a 5-packet pad of surrounding non-cal ObsIDs:

| ObsID | CAL product   | Trimmed family ProductID    | Packets in run |
| ----: | ------------- | --------------------------- | -------------: |
|   257 | SWC-405NM     | NOM-HK-SWC-FAMILY-TRIMMED   |            236 |
|   385 | SOLAR-TOT-PRI | NOM-HK-SOLAR-FAMILY-TRIMMED |             81 |
|   386 | SOLAR-LW-PRI  | NOM-HK-SOLAR-FAMILY-TRIMMED |             81 |

ObsIDs 385 and 386 belong to the same trimmed family, so they produce two files sharing
`NOM-HK-SOLAR-FAMILY-TRIMMED` that differ only in their filename time ranges.

No camera/WFOV cal ObsIDs are present (see TODO[LIBSDC-567]).
