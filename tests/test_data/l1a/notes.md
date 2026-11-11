# NOM-HK ObsID trim fixture

Compact subset of DITL NOM-HK for integration tests of `libera_utils.l1a.nom_hk_trim`.

**Source granule:**
`LIBERA_L1A_NOM-HK-DECODED_V5-8-5RC1_20280213T020114_20280213T040013_R20260612T174745000000.nc`
(DITL Full, orbits 316_02–316_04)

**Fixture file:**
`LIBERA_L1A_NOM-HK-DECODED_V5-8-5RC1_20280213T021705_20280213T040005_R20260612T174745000000.nc`

Keeps RAD cal runs with a 5-packet pad of surrounding non-cal ObsIDs:

| ObsID | CAL product   | TRIMMED ProductID            | Packets in run |
| ----: | ------------- | ---------------------------- | -------------: |
|   257 | SWC-405NM     | NOM-HK-SWC-405NM-TRIMMED     |            236 |
|   385 | SOLAR-TOT-PRI | NOM-HK-SOLAR-TOT-PRI-TRIMMED |             81 |
|   386 | SOLAR-LW-PRI  | NOM-HK-SOLAR-LW-PRI-TRIMMED  |             81 |

No camera/WFOV cal ObsIDs are present (see TODO[LIBSDC-567]).
