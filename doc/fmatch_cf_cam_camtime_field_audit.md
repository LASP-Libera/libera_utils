# Field-completeness audit: what FMATCH-CAM-CAMTIME reads off `PseudoFootprint`

**Question:** for FMATCH to read pseudo-footprints back from `CF-CAM-CAMTIME` (Option 1) instead
of segmenting, exactly which `PseudoFootprint` fields must the product store so FMATCH-CAM-CAMTIME
output is unchanged?

**Answer: all 13.** Every field of `PseudoFootprint` is consumed by the camtime assembly path,
either as a re-emitted output column or by the ancillary aggregation. Two fields have fidelity
traps (see below).

## Source of truth

`_assemble_camtime_dataset` ([product.py:1315](libera_utils/footprint_matching/product.py#L1315),
`fmatch-10` tip) is the only camtime consumer. It reads footprints via:

- `real_columns` ([product.py:1394-1408](libera_utils/footprint_matching/product.py#L1394)) — direct field reads re-emitted as output variables.
- `build_camtime_grid` ([product.py:1206](libera_utils/footprint_matching/product.py#L1206)) — `f.time` → `CAMERA_TIME` axis.
- `build_camtime_provenance` ([product.py:1253](libera_utils/footprint_matching/product.py#L1253)) — `f.slice_x/slice_y` → `camera_pixel_{x,y}_{min,max}`.
- `_merge_computed_variables` → `aggregate_external_variables` → `_footprint_geometry` ([product.py:279](libera_utils/footprint_matching/product.py#L279)) and the aggregation loop ([product.py:561-598](libera_utils/footprint_matching/product.py#L561)) — reads `f.bbox`, `f.latitude`, `f.longitude`, `f.viewing_zenith_angle`.
- `_merge_coverage_qa` ([product.py:947](libera_utils/footprint_matching/product.py#L947)) — reads `f.bbox.truncated`.

`PseudoFootprint` ([camera_segmentation.py:141-182](libera_utils/footprint_matching/camera_segmentation.py#L141)) has 13 fields.
`BoundingBox` is **already** a shared type in [types.py:99](libera_utils/footprint_matching/types.py#L99) — a 7-tuple `(lat_min, lat_max, lon_min, lon_max, wraps_dateline, is_polar, truncated)`. So only `PseudoFootprint` + `CameraFootprintQualityFlag` need relocating; `BoundingBox` does not.

## Full field map → CF-CAM-CAMTIME storage

| # | `PseudoFootprint` field | How FMATCH camtime uses it | CF-CAM-CAMTIME today | Action |
|---|---|---|---|---|
| 1 | `time` | `build_camtime_grid` → `CAMERA_TIME` axis | ✅ `CAMERA_TIME` | keep |
| 2 | `slice_x` (.start/.stop) | `build_camtime_provenance` → `camera_pixel_x_{min,max}` | ✅ stored | keep |
| 3 | `slice_y` (.start/.stop) | → `camera_pixel_y_{min,max}` | ✅ stored | keep |
| 4 | `center_ix` | output col `center_pixel_x` | ❌ | **add** |
| 5 | `center_iy` | output col `center_pixel_y` | ❌ | **add** |
| 6 | `latitude` | output col + aggregation boresight | ❌ | **add** |
| 7 | `longitude` | output col + aggregation boresight | ❌ | **add** |
| 8 | `altitude` | output col `altitude` (metres, surface height) | ❌ | **add** |
| 9 | `solar_zenith_angle` | output col + sunglint derive | ❌ | **add** |
| 10 | `viewing_zenith_angle` | output col + PSF weigher | ❌ | **add** |
| 11 | `relative_azimuth_angle` | output col + sunglint derive | ❌ | **add** |
| 12 | `bbox` (7-tuple) | output `psf_bbox_*` + **tile queries** + `truncated`→QA | ❌ | **add (see Trap A/B)** |
| 13 | `q_flags` | output `q_flags` (base bits) | ❌ | **add (see Trap B)** |

`off_limb` is read in aggregation ([product.py:562](libera_utils/footprint_matching/product.py#L562), [948](libera_utils/footprint_matching/product.py#L948)) but is **not** a `PseudoFootprint` field — camera footprints always read `False` via `getattr` default. No storage needed.

## Trap A — the bbox longitude in the FMATCH *output* is lossy; CF-CAM-CAMTIME must store it RAW

The output columns `psf_bbox_lon_{min,max}` are written through `_normalize_longitude()` to
`[-180, 180)` ([product.py:1403-1404](libera_utils/footprint_matching/product.py#L1403)). But the
aggregation loop passes the **full, un-normalized `footprint.bbox`** to
`tile_manager.get_data(key, footprint.bbox)` and `tile_manager.prefetch(...)`
([product.py:574,586](libera_utils/footprint_matching/product.py#L574)); the TileManager relies on
`wraps_dateline` / `is_polar` to split dateline-crossing and polar requests.

Normalization is **not invertible** for a dateline-wrapping box: raw `(170, 190)` (wraps, 20° wide)
and a `(170, -170)` normalized pair are indistinguishable from a `340°`-wide box. Therefore:

- **CF-CAM-CAMTIME must store the raw bbox** (un-normalized `lon_min`/`lon_max`, which may exceed
  ±180) **plus `wraps_dateline` and `is_polar`** (or store raw bounds and recompute both on read
  with the exact `geometry.bounding_box_from_points` rule). Do **not** reuse the normalized
  `psf_bbox_*` output convention as the CF product's bbox storage.
- On read, FMATCH reconstructs `BoundingBox(lat_min, lat_max, lon_min_raw, lon_max_raw,
  wraps_dateline, is_polar, truncated)`; its own output re-normalizes at write time, so the FMATCH
  output `psf_bbox_*` stays identical.

## Trap B — CF-CAM-CAMTIME `q_flags` must be the SEGMENTATION flags, not the FMATCH output flags

`PseudoFootprint.q_flags` is `CameraFootprintQualityFlag` (2 bits: `PARTIAL_COVERAGE`,
`CENTER_PIXEL_SUBSTITUTED`). FMATCH's **output** `q_flags` is *derived*: assembly seeds it with the
segmentation flags, then `_merge_coverage_qa` OR-s in coverage bits
(`PARTIAL_COVERAGE`/`INSUFFICIENT_COVERAGE`/`LIMB_TRUNCATED`/`OFF_LIMB` from `FmatchCoverageFlag`,
including `LIMB_TRUNCATED` **derived from `bbox.truncated`**).

So the round-trip must preserve the **base** segmentation `q_flags`, not the coverage-augmented
output, and must carry `bbox.truncated` **separately** (FMATCH re-derives `LIMB_TRUNCATED` itself).
Storing the final FMATCH output `q_flags` back into CF-CAM-CAMTIME would double-OR / corrupt the QA
on the next pass. Store:

- `q_flags` = raw `CameraFootprintQualityFlag` value (0–3), and
- `bbox.truncated` as its own boolean (part of the raw-bbox storage in Trap A).

## Resulting CF-CAM-CAMTIME additions

On the existing `(CAMERA_TIME, PSEUDOFOOTPRINT)` grid, add — mirroring the dtypes/units already
declared for the identically-named variables in `fmatch_cam_camtime.yml`
([lines 68-102, 382-444](libera_utils/data/product_definitions/fmatch_cam_camtime.yml#L68)) —
except bbox longitude/flags per Trap A:

- `latitude`, `longitude`, `altitude`, `solar_zenith_angle`, `viewing_zenith_angle`,
  `relative_azimuth_angle` — float, same as FMATCH output.
- `center_pixel_x`, `center_pixel_y` — int (`center_ix/iy`).
- `q_flags` — int, **base segmentation bits only** (Trap B).
- bbox: `psf_bbox_lat_min`, `psf_bbox_lat_max` (float); `psf_bbox_lon_min`, `psf_bbox_lon_max`
  stored **raw / un-normalized** (float); plus `bbox_wraps_dateline`, `bbox_is_polar`,
  `bbox_truncated` flags (or a single packed `bbox_flags` int). (Trap A/B.)

Already present and reused: `CAMERA_TIME`, `PSEUDOFOOTPRINT`, `camera_pixel_{x,y}_{min,max}`,
`cloud_fraction`, `cloud_fraction_standard_deviation`.

The geometry variables carry **real** segmentation-derived values (only `cloud_fraction*` remain
placeholder) — they must **not** be tagged PLACEHOLDER.

## Guards this dictates for the implementation

1. **Round-trip test** (unit): `footprints → generate_placeholder_cloud_fraction_camtime → write →
   read_cf_cam_camtime_pseudofootprints` reproduces **every** field, with explicit cases for a
   dateline-wrapping bbox and a polar bbox (Trap A) and a footprint whose `q_flags != 0` with
   `bbox.truncated=True` (Trap B).
2. **Runner regression test**: FMATCH-CAM-CAMTIME output for a fixed input day is value-identical
   before vs. after the source swap (segmentation relocated, not changed).
