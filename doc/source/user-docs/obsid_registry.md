# ObsID Registry

Libera's ICIE flight software tags certain telemetry packets with a small integer **ObsID**
identifying which calibration event or scan mode is running (gain/noise cal, a specific LED or
blackbody temperature, a solar diffuser face, a lunar scan, cross-track imaging, and so on).
`libera_utils.obsids` is the sole source of truth mapping those ObsIDs to the
`DataProductIdentifier`s they produce. This page explains why the registry is shaped the way it
is, and how it's actually used — both inside this package and by downstream algorithm repos.

## Why a registry, and why keyed by source

ObsIDs are reported in two separate NOM-HK telemetry fields, one per instrument:

| Source | NOM-HK variable       | Typical events                                              |
| ------ | --------------------- | ----------------------------------------------------------- |
| `RAD`  | `ICIE__SW_OBSID_RAD`  | Radiometer cal (gain, noise, SWC, LWC, solar, lunar, VIIRS) |
| `WFOV` | `ICIE__SW_OBSID_WFOV` | Camera cal (CT/RAPS video, darks, VIIRS lunar)              |

The numeric values are **not** globally unique — ObsID `256` means "SWC-365NM" on the radiometer
but "Darks of Darks" on the camera. Because of this, everything in `libera_utils.obsids` is keyed
by `(NomHkObsidSource, obsid)`, never by `obsid` alone. `NomHkObsidSource` is the enum that
disambiguates which telemetry field a given ObsID was read from.

## `ObsIdSpec` and `OBSID_REGISTRY`

Each known `(source, obsid)` pair maps to one `ObsIdSpec`:

```python
@dataclass(frozen=True)
class ObsIdSpec:
    obsid: int
    source: NomHkObsidSource
    kind: ObsIdKind
    description: str
    trimmed_product: DataProductIdentifier | None
    cal_product: DataProductIdentifier | None
```

- `kind` is one of `ObsIdKind.RAD_CAL`, `ObsIdKind.CAM_CAL`, or `ObsIdKind.SCIENCE`. Science/scan
  modes (e.g. Cross Track, RAP Scan, the Geo scans) are cataloged for documentation but never
  produce TRIMMED or CAL products — both product fields are `None`.
- `trimmed_product` is the L1A `NOM-HK-*-TRIMMED` `DataProductIdentifier` produced by the trim
  step (below) for this ObsID, or `None` for non-trim-eligible entries.
- `cal_product` is the `CAL`-level `DataProductIdentifier` a downstream cal-combine algorithm is
  expected to produce for this ObsID, or `None`.

`OBSID_REGISTRY` is the `dict[tuple[NomHkObsidSource, int], ObsIdSpec]` built from all of these.
Look entries up with `get_obsid_spec`, or iterate the trim-eligible subset with
`iter_trim_eligible` (optionally filtered by source):

```python
from libera_utils.obsids import NomHkObsidSource, get_obsid_spec, iter_trim_eligible

spec = get_obsid_spec(NomHkObsidSource.RAD, 256)
spec.cal_product        # DataProductIdentifier.cal_swc_365nm
spec.trimmed_product    # DataProductIdentifier.l1a_icie_nom_hk_swc_365nm_trimmed

# All RAD entries that produce a TRIMMED product (excludes science/scan modes):
rad_trim_eligible = list(iter_trim_eligible(NomHkObsidSource.RAD))
```

`get_obsid_spec` raises `KeyError` for a pair that isn't registered — there is no silent fallback,
so an unrecognized ObsID in telemetry surfaces immediately rather than being misfiled.

### VIIRS lunar: same trimmed data, different cal products

ObsIDs 513/514 (VIIRS lunar, positive/negative azimuth start) are registered under **both** `RAD`
and `WFOV`. Both sources share the same `trimmed_product` (it's the same underlying NOM-HK subset
either way), but each source has its own `cal_product`
(`cal_rad_viirs_lunar_pos_start` vs. `cal_wfov_viirs_lunar_pos_start`) because the radiometer and
camera cal-combine algorithms produce distinct CAL products from it. This is the general rule when
extending the registry: share a TRIMMED product across sources only when the underlying telemetry
subset is identical; keep CAL products distinct whenever downstream algorithms diverge.

## Consumer 1: the L1A trim step

`libera_utils.l1a.nom_hk_trim` is the first real consumer. After a daily `NOM-HK-DECODED` product
is produced, calibration pipelines need a per-ObsID subset of it. `find_obsid_runs` scans the
Dataset for contiguous runs of each trim-eligible ObsID (using `spec.source.value` to pick the
right NOM-HK field), and `write_trimmed_nom_hk_products` writes one `NOM-HK-*-TRIMMED` NetCDF per
run:

```python
from libera_utils.l1a.nom_hk_trim import write_trimmed_nom_hk_products

# After parse_packets_to_l1a_dataset(...) for NOM-HK:
trimmed_paths = write_trimmed_nom_hk_products(
    nom_hk_ds,
    output_dir,
    time_variable="PACKET_ICIE_TIME",
    add_archive_path_prefix=True,  # L1A preprocessor / ingest dropbox
)
```

TRIMMED products reuse the `NOM-HK-DECODED` variable schema; only `ProductID` (and thus the
filename product token) changes. Normal operations expect each calibration ObsID at most once per
day — if the same `(source, obsid)` appears in multiple disjoint runs, each run is written as a
separate file and a warning is logged, since that's unexpected outside of ground testing.

## Consumer 2: downstream cal-combine dispatch

The registry's second job is letting downstream algorithm repos (e.g. `libera_rad`) build their
own ObsID dispatch **without hand-maintaining a duplicate ObsID → product mapping**. Instead of a
locally-owned dict, a downstream repo derives its dispatch table directly from the registry, e.g.:

```python
from libera_utils.obsids import NomHkObsidSource, ObsIdKind, iter_trim_eligible

# Build once at import time: RAD-only, cal-eligible entries this algorithm supports
cal_event_by_obsid = {
    spec.obsid: spec
    for spec in iter_trim_eligible(NomHkObsidSource.RAD)
    if spec.kind is ObsIdKind.RAD_CAL
}
```

This is what makes the shared-ECR dispatch pattern possible: `ProcessingStepIdentifier` members
for radiometer cal-combine steps (`cal-gain`, `cal-noise`, `cal-swc-*`, `cal-lwc-*`, `cal-solar-*`) all set
`shared_ecr_name=CAL_RAD_SHARED_ECR_NAME`, so `ProcessingStepIdentifier.ecr_name` resolves every
one of them to the same `cal-rad-docker-repo` image. A Batch job reads which ObsID it was invoked
for from an environment variable (`LIBERA_CAL_OBSID`), looks it up in a dispatch table like the
one above, and runs the matching family algorithm — one container image instead of one per
calibration event. A downstream repo can also use `get_obsid_spec` at runtime to confirm the
ObsID(s) actually present in its input telemetry match what it was dispatched for, and fail closed
if they don't.

None of this requires the downstream repo to know about NOM-HK field names, ObsID collisions, or
product-naming conventions — that's exactly what registering a new ObsID here, once, is meant to
replace.

## Current coverage

- Radiometer (`RAD_CAL`) and camera (`CAM_CAL`) entries all have `trimmed_product` and
  `cal_product` set, so they're trim-eligible today.
- Lunar (ObsIDs 448/449) and VIIRS-lunar (513/514) cal products are registered, but no
  `ProcessingStepIdentifier` cal-combine steps exist for them yet — cal-combine support for those
  is deferred to a follow-on change.
- Camera `ProcessingStepIdentifier` cal-combine steps are deferred entirely; only the TRIMMED side
  (L1A preprocessing) is wired up for camera ObsIDs so far.
- Science/scan mode entries (Cross Track, RAP Scan, Along Track, Earth Target, the Geo scans) are
  catalog-only and never trim-eligible.

## Extending the registry

When a new calibration ObsID needs to be added:

1. Add its `DataProductIdentifier` member(s) (a `CAL` product, and a `NOM-HK-*-TRIMMED` product if
   it should be trim-eligible) in `libera_utils/constants.py`.
2. Register it in `OBSID_REGISTRY` via `_rad_cal`/`_cam_cal`/`_science` in `obsids.py`, with a
   plain-language `description` (it shows up in error messages and this documentation, so keep it
   accurate rather than terse).
3. Do this here, in `libera_utils`, first — not as a local dict in a downstream repo. Downstream
   dispatch tables should be derived from `iter_trim_eligible`/`get_obsid_spec`, matching the
   pattern in [Consumer 2](#consumer-2-downstream-cal-combine-dispatch) above.
