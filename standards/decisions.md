# Decisions · libera_utils

Decisions that constrain this repository only. Cross-repository decisions are in the shared
corpus (`SHARED.md`), numbered in the same `D-` space; cite a local one as
`libera_utils/D-NNN` so a finding key is unambiguous.

Appended by the ratchet, from convention questions people answered in a pull request thread
or a ticket. One paragraph each, with where and when.

---

### libera_utils/D-001 · The ObsID registry is data, validated at import

_2026-09-02, PR #41_

`data/obsid_registry.csv` and `data/trim_family_inputs.csv` hold the registry, not a literal
inside `obsids.py`. A table in code cannot be validated as data and a table in a comment
cannot be used at all. The registry docstring names the upstream ICIE ObsID page — by name,
with no URL (R-014) — so the next reader knows there is a document to reconcile against.

### libera_utils/D-002 · A ProductID that reaches a filename is globally unique

_2026-09-02, PR #41_

Product identifier strings land in filenames, so two products that share a string collide on
disk and the second write clobbers the first. The registry key carries the instrument and
the ProductID does not, so any ObsID asserted on both the radiometer and the camera needs
its own pair of product identifiers. This is what the uniqueness invariant test guards.

### libera_utils/D-003 · Ground CCSDS filename times are bin starts, not data spans

_2026-09-09, PR #48_

The time fields in a ground capture's filename are the bin the file was cut from. They say
nothing about the packets inside it, and under DITL they disagree with the data by years.
Searchable data times come from File Metadata at ingest, never from the archive path. Any
code that reads a time from a filename must say which of the two it means (T-010, T-011,
T-012).

### libera_utils/D-004 · The NOAA-20 SPICE configuration is test data, not shipped config

_2026-09, LIBSDC-703, changelog 5.12.0_

It moved to `tests/test_data/noaa20_spice/`. Nothing in the pipeline could select it, and its
frame kernel declares none of the measured misalignments, so it cannot produce
flight-representative geometry. It is kept as test data because it is the only kernel
generation driven by real decoded spacecraft telemetry and the only geolocation validated
against a third-party product (CERES). A downstream package that pointed `LIBERA_KERNEL_DIR`
at it must vendor the configuration itself.
