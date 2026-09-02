# Observation ID (ObsID) Registry

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
- `trimmed_product` is the L1A `NOM-HK-*-FAMILY-TRIMMED` `DataProductIdentifier` produced by the
  trim step (below) for this ObsID, or `None` for non-trim-eligible entries. It names a
  **calibration dependency family**, not a single ObsID — see below.
- `cal_product` is the `CAL`-level `DataProductIdentifier` a downstream cal-combine algorithm is
  expected to produce for this ObsID, or `None`.

`OBSID_REGISTRY` is the `dict[tuple[NomHkObsidSource, int], ObsIdSpec]` built from all of these.
Look entries up with `get_obsid_spec`, or iterate the trim-eligible subset with
`iter_trim_eligible` (optionally filtered by source):

```python
from libera_utils.obsids import NomHkObsidSource, get_obsid_spec, iter_trim_eligible

spec = get_obsid_spec(NomHkObsidSource.RAD, 256)
spec.cal_product        # DataProductIdentifier.cal_swc_365nm
spec.trimmed_product    # DataProductIdentifier.l1a_icie_nom_hk_swc_family_trimmed

# All RAD entries that produce a TRIMMED product (excludes science/scan modes):
rad_trim_eligible = list(iter_trim_eligible(NomHkObsidSource.RAD))
```

`get_obsid_spec` raises `KeyError` for a pair that isn't registered — there is no silent fallback,
so an unrecognized ObsID in telemetry surfaces immediately rather than being misfiled.

## Calibration dependency families

An ObsID does **not** get its own TRIMMED product. Several ObsIDs share one, because downstream
algorithms process every member of a group identically — `libera_rad` runs the same shortwave-LED
cal on ObsIDs 256 through 261 regardless of wavelength. What actually differs between processing
steps is the _set of input products a step depends on_, and that is what libera_cdk deploys against.
So the TRIMMED column names a **calibration dependency family**: one product per group of ObsIDs
sharing a dependency, and therefore one `ProcessingStepIdentifier` per family instead of one per
ObsID.

| Family ProductID                         | Source | ObsIDs        |
| ---------------------------------------- | ------ | ------------- |
| `NOM-HK-GAIN-FAMILY-TRIMMED`             | RAD    | 512, 515      |
| `NOM-HK-SWC-FAMILY-TRIMMED`              | RAD    | 256–261       |
| `NOM-HK-LWC-FAMILY-TRIMMED`              | RAD    | 320–324       |
| `NOM-HK-SOLAR-FAMILY-TRIMMED`            | RAD    | 384–395       |
| `NOM-HK-LUNAR-FAMILY-TRIMMED`            | RAD    | 448, 449      |
| `NOM-HK-RAD-VIIRS-LUNAR-FAMILY-TRIMMED`  | RAD    | 513, 514      |
| `NOM-HK-CT-VIDEO-FAMILY-TRIMMED`         | WFOV   | 129, 130, 131 |
| `NOM-HK-RAPS-VIDEO-FAMILY-TRIMMED`       | WFOV   | 133, 134, 135 |
| `NOM-HK-DARKS-FAMILY-TRIMMED`            | WFOV   | 256, 257, 258 |
| `NOM-HK-WFOV-VIIRS-LUNAR-FAMILY-TRIMMED` | WFOV   | 513, 514      |

`TRIM_FAMILIES` is the inverse view of that column, and `get_family_specs` is how a downstream step
enumerates the ObsIDs — and CAL products — it is responsible for:

```python
from libera_utils.constants import DataProductIdentifier
from libera_utils.obsids import get_family_specs

for spec in get_family_specs(DataProductIdentifier.l1a_icie_nom_hk_swc_family_trimmed):
    spec.obsid, spec.cal_product   # (256, cal_swc_365nm), (257, cal_swc_405nm), ...
```

Two invariants hold, and both are enforced at import time:

- **Each ObsID keeps its own CAL product.** A family groups inputs, not outputs, so a family step
  still maps every ObsID it sees to exactly one distinct CAL product. A duplicate `cal_product`
  cell raises `ValueError`.
- **A family never spans both NOM-HK ObsID fields.** Trimming scans one field (`spec.source.value`)
  at a time, so a family confined to `RAD` or to `WFOV` is what makes a trimmed file attributable
  to a source. ObsIDs 513/514 (VIIRS lunar) run on both instruments and are therefore registered as
  _two_ families, `NOM-HK-RAD-VIIRS-LUNAR-FAMILY-TRIMMED` and
  `NOM-HK-WFOV-VIIRS-LUNAR-FAMILY-TRIMMED`, each with its own CAL products.

The ObsID itself is never lost: a trimmed file still carries the `ICIE__SW_OBSID_RAD` /
`ICIE__SW_OBSID_WFOV` variable it was cut on, so a consumer recovers the exact ObsID from the data
rather than from the ProductID.

## Consumer 1: the L1A trim step

`libera_utils.l1a.nom_hk_trim` is the first real consumer. After a daily `NOM-HK-DECODED` product
is produced, calibration pipelines need a per-ObsID subset of it. `find_obsid_runs` scans the
Dataset for contiguous runs of each trim-eligible ObsID (using `spec.source.value` to pick the
right NOM-HK field), and `write_trimmed_nom_hk_products` writes one NetCDF per run, stamped with
that run's family ProductID:

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
filename product token) changes. Because the ProductID names a family, one day normally yields
_several_ files sharing a ProductID — six `NOM-HK-SWC-FAMILY-TRIMMED` granules for six LED ObsIDs,
say — told apart by their filename time ranges. Each file still covers exactly one ObsID run.

Normal operations expect each calibration ObsID at most once per day — if the same
`(source, obsid)` appears in multiple disjoint runs, each run is written as a separate file and a
warning is logged, since that's unexpected outside of ground testing. Different ObsIDs of one
family are not that case and do not warn.

## Consumer 2: downstream cal-combine dispatch

The registry's second job is letting downstream algorithm repos (e.g. `libera_rad`) build their
own ObsID dispatch **without hand-maintaining a duplicate ObsID → product mapping**. Instead of a
locally-owned dict, a downstream repo derives its dispatch table directly from the registry, e.g.:

```python
from libera_utils.constants import DataProductIdentifier
from libera_utils.obsids import get_family_specs

# Build once at import time: the ObsIDs this step is responsible for, and their CAL outputs
cal_event_by_obsid = {
    spec.obsid: spec
    for spec in get_family_specs(DataProductIdentifier.l1a_icie_nom_hk_swc_family_trimmed)
}
```

This is what makes the shared-ECR dispatch pattern possible: the radiometer cal-combine
`ProcessingStepIdentifier` members (`cal-gain-family`, `cal-swc-family`, `cal-lwc-family`,
`cal-solar-family`) all set `shared_ecr_name=CAL_RAD_SHARED_ECR_NAME`, so
`ProcessingStepIdentifier.ecr_name` resolves every one of them to the same `cal-rad-docker-repo`
image, and each step's `products` list the CAL products of its family's ObsIDs. A Batch
job reads the ObsID out of the trimmed input it was handed, and runs the matching algorithm. This leads to one container image, and one deployed step per family, instead of one per calibration event.

## The other L1A inputs a cal step needs

NOM-HK is the only product the L1A preprocessor trims. A calibration algorithm needs more than
NOM-HK — the shortwave LED cal, for instance, also needs PEV-SW, PEC-SW, RAD-SAMPLE, CAL-SAMPLE
and AXIS-SAMPLE — and those arrive as the **full daily L1A granules**. The cal container reads the
run's time range off the TRIMMED NOM-HK filename it was handed and subsets them itself.

Which products those are is stored in `libera_utils/data/trim_family_inputs.csv`.
This file maps each family to its L1A inputs, and the two catalog files are cross-checked at import, so a
family cannot exist in one without the other:

```python
from libera_utils.constants import DataProductIdentifier
from libera_utils.obsids import get_family_inputs

get_family_inputs(DataProductIdentifier.l1a_icie_nom_hk_swc_family_trimmed)
# (PEV-SW-STAT-DECODED, PEC-SW-STAT-DECODED,
#  RAD-SAMPLE-DECODED, CAL-SAMPLE-DECODED, AXIS-SAMPLE-DECODED)
```

That tuple, **plus the family's own TRIMMED product**, is what a `cal-*-family` node's
`input-products` should list in libera_cdk's `processing_system_dag.json`. The full-day
`NOM-HK-DECODED` granule is deliberately absent: the family's NOM-HK arrives already trimmed as
the TRIMMED product itself, so listing it here would stage a second, redundant NOM-HK input.
Families whose processing step is still deferred return an empty tuple — the dependency set is
undecided, not empty — and declaring it is part of closing `TODO[LIBSDC-811]`.

## Current coverage

- Radiometer (`RAD_CAL`) and camera (`CAM_CAL`) entries all have `trimmed_product` and
  `cal_product` set, so they're trim-eligible today.
- Four radiometer cal-combine steps exist: `cal-gain-family`, `cal-swc-family`, `cal-lwc-family`,
  and `cal-solar-family`.
- The lunar (448/449) and VIIRS-lunar (513/514) families are registered, but no
  `ProcessingStepIdentifier` cal-combine steps exist for them yet (`TODO[LIBSDC-811]`).
- Camera `ProcessingStepIdentifier` cal-combine steps are deferred entirely; only the TRIMMED side
  (L1A preprocessing) is wired up for camera ObsIDs so far.
- Science/scan mode entries (Cross Track, RAP Scan, Along Track, Earth Target, the Geo scans) are
  catalog-only and never trim-eligible.
- Camera cal ObsIDs are registered on `WFOV` only, and the loader enforces that. Science modes are
  dual-registered because both instruments assert them, but CT video (129-131), RAPS video
  (133-135) and the darks (256-258) are asserted only on `ICIE__SW_OBSID_WFOV` — ICIE has
  confirmed `ICIE__SW_OBSID_RAD` never carries those values. `get_obsid_spec(NomHkObsidSource.RAD,
129)` therefore raises `KeyError` by design; the asymmetry is not a missing row.

## Extending the registry

When a new calibration ObsID needs to be added:

1. Add its `CAL` `DataProductIdentifier` member in `libera_utils/constants.py`. If it belongs to an
   existing calibration dependency family, reuse that family's `NOM-HK-*-FAMILY-TRIMMED` product;
   only add a new TRIMMED member when the ObsID genuinely introduces a new input dependency, in
   which case add a `ProcessingStepIdentifier` for it too.
2. Add a row to `libera_utils/data/obsid_registry.csv` naming those two members, with a
   plain-language `description` (it shows up in error messages and this documentation, so keep it
   accurate rather than terse). Edit the file with a text editor or the `csv` module — never a
   spreadsheet app that may rewrite quoting, since descriptions contain commas. No Python change is
   needed for an ObsID joining an existing family.
3. If the ObsID joins the family of an existing cal step, add its CAL product to that step's
   `products` list so the step still declares everything it can emit. If instead you added a new
   TRIMMED family in step 1, add a matching row to
   `libera_utils/data/trim_family_inputs.csv` naming the L1A products that family's algorithm
   consumes — import fails if the two files disagree. Leave `required_inputs` empty if the
   dependency set is not settled yet.
4. Do this here, in `libera_utils`, first — not as a local dict in a downstream repo. Downstream
   dispatch tables should be derived from `get_family_specs`/`iter_trim_eligible`/`get_obsid_spec`,
   matching the pattern in [Consumer 2](#consumer-2-downstream-cal-combine-dispatch) above.
