# NISE AMSR-2 interchangeability with SSM/I-SSMIS — findings & implementation plan

## Status: IMPLEMENTED (2026-09-17)

Delivered across the fmatch stack (committed locally, **not pushed**). Chosen
approach: **full-faithful rename** + **commit + restack** (user-authorized).

- Reader (`readers/nsidc.py`) + tests + notebook 02 on **fmatch-2-readers-gridded**
  (`808ed91`); restacked downstream 3→9.
- Product-def YAMLs (5 files) + long_names + `(SSMIS)`→`(AMSR2)` + notebook 06 on
  **fmatch-6-product-definitions** (`f42c291`); notebooks 07/09 refreshed on
  fmatch-7/9 (`e3d10df`, `1a5cf49`).
- Parallel **fmatch-efficiency** patched directly (`25c6556`).
- Layer renames (to match the granule `data_grid_key`): `no_ice_or_snow`→
  `snow_free_land` (0), `dry_snow_on_land`→`snow_on_land` (103–104),
  `snow_ice_missing`→`open_ocean` (255 = ocean, previously mislabeled "missing");
  `INSTRUMENT` SSMIS→AMSR2.
- **Step 3 outcome:** the runner discovers NISE granules by *listing the `nise/`
  ancillary subdirectory* (`_runner.py`, filename-agnostic), so AMSR-2 files are
  already accepted — no glob change was needed.
- Backups: `backup/amsr-nise-20260917/*`. Tests: reader 36, fmatch-6 467,
  fmatch-9 & efficiency 604 each; 0 stale references on all nine branches.

The plan below is retained as the design record.

## Question

Three NISE granules were staged under `external_data/NSDIC/`:

| File | Product | Sensor |
| --- | --- | --- |
| `NISE_SSMISF18_20260111.HDFEOS` | NISE v5 | SSM/I-SSMIS (DMSP F18) |
| `NISE_SSMISF18_20260115.HDFEOS` | NISE v5 | SSM/I-SSMIS (DMSP F18) |
| `NISE_AMSR2_20260610.HDFEOS` | NISE_A2 v1 | AMSR-2 (GCOM-W1) |

Is the AMSR-2 file interchangeable with the SSM/I-SSMIS file that
[`NISEReader`](../libera_utils/footprint_matching/readers/nsidc.py) already reads?

## TL;DR

**Mechanically, yes — it already works, unmodified.** The AMSR-2 granule flows
through `NISEReader` end-to-end and produces byte-compatible output. Nothing
*fails* at runtime.

The gaps are **provenance and documentation**, not I/O:

1. `NISEReader.INSTRUMENT` is hardcoded to `"SSMIS"`, so every output variable's
   provenance is stamped `(SSMIS)`. **Production uses AMSR-2 only**, so this
   constant should simply become `"AMSR2"` — no per-granule derivation.
2. The reader's docstring and class docs explicitly disclaim AMSR-2
   (*"the AMSR-2 NISE_A2 variant is not read here"*), which is now false and
   misleading.
3. There is no file-discovery / filename-parsing layer on this branch, so no
   code has to learn the `NISE_AMSR2_*` naming pattern yet — but that layer
   (in the runner/product-assembly PR) will need to when it lands.

A separate, **pre-existing** correctness issue in the Extent-code decoding was
discovered along the way (affects SSM/I and AMSR-2 equally — see
[Appendix](#appendix-pre-existing-extent-code-mismatch-out-of-scope)).

## Evidence

### The two products are structurally identical on disk

Inspected with `pyhdf` (installed via the `fmatch` deps; HDF4 C libs already
present in this container):

| Property | SSMISF18 | AMSR2 | Match |
| --- | --- | --- | --- |
| SDS count | 4 | 4 | ✅ |
| SDS names | `Extent`, `Age` (×N/S hemi) | same | ✅ |
| Hemisphere dim labels | `YDim:Northern Hemisphere`, `…Southern…` | same | ✅ |
| Grid shape | 721 × 721 | 721 × 721 | ✅ |
| dtype | uint8 (HDF type 21) | uint8 | ✅ |
| `GridName` | `Northern/Southern Hemisphere` | same | ✅ |
| `UpperLeftPointMtrs` | `(-9036842.7625, 9036842.7625)` | same | ✅ |
| `LowerRightMtrs` | `(9036842.7625, -9036842.7625)` | same | ✅ |
| `Projection` | `GCTP_LAMAZ` (EASE-Grid, EPSG:3408/3409) | same | ✅ |
| `data_grid_key` (Extent encoding) | *(identical text)* | *(identical text)* | ✅ |

Every hardcoded default in `nsidc.py`
(`_DEFAULT_GRID_ROWS/COLS`, `_DEFAULT_RESOLUTION_M`, `_DEFAULT_X/Y_ORIGIN`,
`_DEFAULT_EPSG_NORTH/SOUTH`, the hemisphere labels, and the SDS name `Extent`)
matches the AMSR-2 file exactly.

### The reader ingests the AMSR-2 file unmodified

Running the **unchanged** `NISEReader` against both granules:

| Check | SSMISF18 | AMSR2 |
| --- | --- | --- |
| `_read_extent_sds(North/South)` → shapes | `(721,721)` / `(721,721)` | same |
| N/S SDS are distinct arrays | ✅ | ✅ |
| `_load_points` value array | `(5, 811786)` | `(5, 811786)` |
| point-cloud lat span | +90 … −90 | +90 … −90 |
| Arctic tile `(70–80 N)` finite cells | ✅ | ✅ |
| Antarctic tile `(80–70 S)` finite cells | ✅ | ✅ |
| Runtime warnings / errors | none | none |

Existing unit suite (`test_nsidc.py`): **32 passed, 3 skipped** (the 3 skipped
are the real-granule integration tests — see [staging note](#3-optional-test-coverage)).

## Where it "fails" (gaps, not crashes)

1. **Provenance mislabeling — the one substantive defect.**
   [`nsidc.py:253`](../libera_utils/footprint_matching/readers/nsidc.py#L253)
   pins `INSTRUMENT = "SSMIS"`. Per
   [`base.py`](../libera_utils/footprint_matching/readers/base.py#L76-L81),
   `INSTRUMENT` is the provenance tag recorded in every output variable's
   `long_name` as `"… ({INSTRUMENT})"`. Since the operational product is built
   from AMSR-2, this constant should read `"AMSR2"`. This is the only
   correctness gap that AMSR-vs-SSMI actually introduces.

2. **Documentation contradicts reality.** The module and class docstrings state
   the AMSR-2 variant is *not* read
   ([`nsidc.py:251-252`](../libera_utils/footprint_matching/readers/nsidc.py#L251-L252)
   and the class docstring). It reads fine; the docs are stale.

3. **No filename/date/instrument parsing exists on this branch.** The reader
   takes a `file_path` directly; the file-discovery + reader-dispatch layer
   lives in the runner/product-assembly PR that is not in
   `libera_utils/footprint_matching/` here. When that layer lands it must accept
   **both** naming patterns:
   - `NISE_SSMISF{ss}_{YYYYMMDD}.HDFEOS` (v5, `ss` = DMSP flight, e.g. `F18`)
   - `NISE_AMSR2_{YYYYMMDD}.HDFEOS` (NISE_A2 v1)

   and ideally derive the instrument label from the filename rather than a
   class constant (see plan).

## Implementation plan

Scope is small: the I/O already works. The work is making provenance correct
and the docs honest, plus test coverage proving the AMSR-2 path.

### 1. Set `INSTRUMENT = "AMSR2"`

Production uses AMSR-2 only, so the output product must reflect AMSR-2 — no
per-granule filename derivation is needed. This is a **one-line constant
change**:

- Change [`nsidc.py:253`](../libera_utils/footprint_matching/readers/nsidc.py#L253)
  from `INSTRUMENT = "SSMIS"` to `INSTRUMENT = "AMSR2"`, and update the adjacent
  comment ([`nsidc.py:251-252`](../libera_utils/footprint_matching/readers/nsidc.py#L251-L252))
  to say the reader targets the AMSR-2 NISE_A2 product.
- The class attribute keeps satisfying `base.__init_subclass__`'s
  registration-time `INSTRUMENT` check
  ([`base.py:156`](../libera_utils/footprint_matching/readers/base.py#L156)) and
  the `classmethod` provenance path unchanged — no structural change required.
- **Implication (accepted):** the two staged SSM/I example granules
  (`NISE_SSMISF18_*`) will now also be labeled `(AMSR2)` in provenance if read.
  That is fine — they exist only as regression/example data; the operational
  input is AMSR-2. No dual-sensor labeling is introduced.
- Confirm the eventual `long_name` builder (in the product-assembly PR, not this
  branch) reads this class attr; if so, the constant change is the whole fix.

### 2. Correct the documentation

- Update the module docstring: NISE_A2 / AMSR-2 **is** supported; drop the
  "not read here" disclaimers at
  [`nsidc.py:251-252`](../libera_utils/footprint_matching/readers/nsidc.py#L251-L252)
  and in the class docstring.
- Note that both products share the identical HDF-EOS4 layout, grid geometry,
  and Extent encoding, so one reader serves both; the only per-granule
  difference is the instrument provenance label.
- Keep both filename patterns in the "File naming" reference block (already
  present at [`nsidc.py:91-92`](../libera_utils/footprint_matching/readers/nsidc.py#L91-L92)).

### 3. Optional: file-discovery layer (only when that PR lands)

If/when the runner selects NISE files by glob, ensure the pattern matches
`NISE_*.HDFEOS` (both sensors), not `NISE_SSMISF*`. Prefer the AMSR-2 granule
only for dates ≥ its 2012 coverage start; otherwise the two are
substitutable. This item is a placeholder — no such code exists on
`fmatch-scene-id`.

### 4. Test coverage

- Add an AMSR-2 case to `TestNISEReaderRealGranule` asserting shape and
  both-hemisphere coverage against the AMSR-2 granule, plus a cheap unit
  assertion that `NISEReader.INSTRUMENT == "AMSR2"` (guards against a silent
  revert to `"SSMIS"`).
- **Staging note:** the integration test resolves the granule at
  `external_data/external_data/NSDIC/NISE_SSMISF18_20260111.HDFEOS`
  ([`test_nsidc.py:459-461`](../tests/unit/test_footprint_matching/test_readers/test_nsidc.py#L459-L461)),
  a **nested** `external_data/external_data/` path. The staged files are at
  `external_data/NSDIC/`, so these tests are currently **skipped**. Either fix
  the path constant or stage the granules where the test looks; otherwise new
  AMSR-2 assertions will silently skip too.
- No new synthetic fixture is needed — the SSM/I encoding fixtures already
  exercise the shared decode path.

### 5. Packaging note (not a blocker)

`pyhdf` is a **main** dependency in `pyproject.toml`
([lines 69-70](../pyproject.toml)) but was **not installed** in this venv; it
had to be added manually (`pip install pyhdf`) to read either granule. Worth a
one-line check that `poetry install` provisions it in dev/CI environments that
have the HDF4 C library, since neither the SSM/I nor the AMSR-2 path can run
without it.

## Estimated effort

A couple of hours. The substantive change is a one-line `INSTRUMENT` constant
(step 1) plus doc edits (step 2) and one or two test assertions (step 4). No
I/O, projection, or decoding code changes are required.

## Appendix: pre-existing Extent-code mismatch (out of scope)

Discovered while comparing the on-disk `data_grid_key` to the reader's decode
table — this affects **both** sensors equally and is **not** an
AMSR-interchangeability issue, but it is worth a separate ticket:

| Extent code | On-disk `data_grid_key` (both files) | Reader's treatment |
| --- | --- | --- |
| `0` | snow-free land | `no_ice_or_snow` = 1 |
| `103` | dry snow | `dry_snow_on_land` (103–110) |
| `104` | wet snow | folded into `dry_snow_on_land` |
| `105–251` | not used | 0 in all layers (mostly OK) |
| `252` | mixed pixels at coastlines | 0 in all layers (dropped) |
| `253` | suspect ice value | 0 in all layers (dropped) |
| `254` | corners (undefined) | dropped as off-Earth ✅ |
| `255` | **ocean** | `snow_ice_missing` = 1 ❌ |

The most consequential item: the reader's `snow_ice_missing` layer keys on
code 255, but the granules' own legend calls 255 **ocean**, not missing/fill.
If the embedded `data_grid_key` is authoritative, that layer is effectively an
ocean mask. This should be reconciled against the NISE v5 user guide in its own
issue, independently of AMSR-2 support.
