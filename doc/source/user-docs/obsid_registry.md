# Observation ID (ObsID) Registry

Libera's ICIE flight software tags certain telemetry packets with a small integer **ObsID**
identifying which calibration event or scan mode is running (gain/noise cal, a specific LED or
blackbody temperature, a solar diffuser face, a lunar scan, cross-track imaging, and so on).
`libera_utils.obsids` is the sole source of truth mapping those ObsIDs to the
`DataProductIdentifier`s they produce. This page covers how the registry is structured and how it
is used, both inside this package and by downstream algorithm repos.

## ObsID sources

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
- `trimmed_product` is the L1A `NOM-HK-<CAL ProductID>-TRIMMED` `DataProductIdentifier` produced by
  the trim step (below) for this ObsID, or `None` for non-trim-eligible entries.
- `cal_product` is the `CAL`-level `DataProductIdentifier` this ObsID's combiner step produces, or
  `None`.

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

## One TRIMMED product, one combiner step, one CAL product per ObsID

Every calibration ObsID has its own TRIMMED NOM-HK product, its own CAL product, and one
`ProcessingStepIdentifier`, its **combiner**, that turns the first into the second. The three are
1:1:1, and named after the CAL product:

| ObsID    | TRIMMED ProductID               | Combiner step                | CAL ProductID    |
| -------- | ------------------------------- | ---------------------------- | ---------------- |
| RAD 256  | `NOM-HK-SWC-365NM-TRIMMED`      | `cal-combine-swc-365nm`      | `SWC-365NM`      |
| RAD 512  | `NOM-HK-GAIN-TRIMMED`           | `cal-combine-gain`           | `GAIN`           |
| WFOV 256 | `NOM-HK-DARKS-OF-DARKS-TRIMMED` | `cal-combine-darks-of-darks` | `DARKS-OF-DARKS` |

Go from a TRIMMED product to its ObsID and to its combiner step with `get_trimmed_obsid_spec` and
`get_combiner_step`; both raise `KeyError` for a product that is not a registered TRIMMED product:

```python
from libera_utils.constants import DataProductIdentifier
from libera_utils.obsids import get_combiner_step, get_trimmed_obsid_spec

trimmed = DataProductIdentifier.l1a_icie_nom_hk_swc_365nm_trimmed
get_trimmed_obsid_spec(trimmed).obsid   # 256
get_combiner_step(trimmed)              # ProcessingStepIdentifier.cal_combine_swc_365nm
```

Two invariants are enforced at import time: no two ObsIDs claim one TRIMMED product, and no two
claim one CAL product (either raises `ValueError`). A CAL product that no `ProcessingStepIdentifier`
produces also raises `ValueError` at import.

Every combiner step sets `shared_ecr_name=CAL_RAD_SHARED_ECR_NAME`, so
`ProcessingStepIdentifier.ecr_name` resolves all of them, radiometer and camera, to the one
`cal-rad-docker-repo` image.

### Calibration families

ObsIDs whose combiners take the same inputs and differ only in their TRIMMED product are often
called a **calibration family**: the six shortwave LED ObsIDs (256–261), the twelve solar diffuser
ObsIDs (384–395), and so on. A family is a way to talk about calibration events. It is not a
product or a processing step, and no catalog file records it; each combiner lists its inputs in
full (below).

## Calibration event files

A TRIMMED or CAL file holds one calibration **event**, not one day: an ObsID can occur more than
once a day, and each occurrence is its own file of the ObsID's product, told apart by the time
range in its filename. The event's start time, `utc_start` in the filename (`%Y%m%dT%H%M%S`, UTC),
identifies it.

`LiberaDataProductFilename.applicable_date`, and with it `archive_prefix`, dates an event file by
the UTC day the event **started**, whatever its length, so an event that crosses midnight is
archived under its start day. Every other product is dated by the midpoint of its time range.
`is_calibration_event_product` says which rule a ProductID follows.

## Consumer 1: the L1A trim step

`libera_utils.l1a.nom_hk_trim` is the first real consumer. After a daily `NOM-HK-DECODED` product
is produced, calibration pipelines need a per-ObsID subset of it. `find_obsid_runs` scans the
Dataset for contiguous runs of each trim-eligible ObsID (using `spec.source.value` to pick the
right NOM-HK field), and `write_trimmed_nom_hk_products` writes one NetCDF per run that starts on the requested day, stamped with
that ObsID's TRIMMED ProductID:

```python
from libera_utils.l1a.nom_hk_trim import write_trimmed_nom_hk_products

# After parse_packets_to_l1a_dataset(...) for NOM-HK:
trimmed_paths = write_trimmed_nom_hk_products(
    nom_hk_ds,
    output_dir,
    day=applicable_date,  # UTC day whose runs are written
    source_product_filename=daily_nom_hk_filename,  # the NOM-HK-DECODED granule for that day
    time_variable="PACKET_ICIE_TIME",
    add_archive_path_prefix=True,  # L1A preprocessor / ingest dropbox
)
```

TRIMMED products reuse the `NOM-HK-DECODED` variable schema; only `ProductID` (and thus the
filename product token) changes. A trimmed file also still carries the `ICIE__SW_OBSID_RAD` /
`ICIE__SW_OBSID_WFOV` variable it was cut on.

Only runs that start on `day` are written, and each is written whole, so a run that crosses
midnight belongs to the day it started and the next day leaves it out. Decode NOM-HK from at least
one packet before the day's first run (`day_window.day_bounds` gives a window starting `day_window.DAY_BUFFER` before the day)
through at least `nom_hk_trim.CALIBRATION_TRIM_PADDING` past the end of the day, so that every run
starting on the day is seen to start and end. An ObsID may recur within a day; each run is its
own file.

A run is a stretch of one ObsID: it ends where the ObsID changes. Packets lost inside it do not
end it, so the run stays one file. Two events of one ObsID whose boundary fell inside a gap are
one run until the missing packets arrive.

A run whose start or end is at a gap is written all the same. An edge is at a gap when the packets
beyond it are missing or the data stops there, so the file may not hold its event's true start or
end. The TRIMMED file carries no record of its gaps: the INFO line `write_trimmed_nom_hk_products`
logs for each file says whether its start and end are at a gap and how many packets are missing
inside it (counted modulo 16384, so a lower bound). When missing packets arrive later, the next
revision of the day writes the run again:

- an end that was at a gap moves later, under the same start time, so the new revision replaces
  the file;
- a start that was at a gap moves earlier, so the run is filed under a new start time and the
  earlier file stays archived with nothing replacing it;
- a gap that hid the boundary between two events of one ObsID becomes two files: the first keeps
  the start time and the second is new.

A run still going at the last packet of the data, when that packet is within 10 minutes of
`CALIBRATION_TRIM_PADDING` past the end of the day, lasts longer than the padding:
`write_trimmed_nom_hk_products` raises `ValueError` before writing any file for the day, and the
padding must be raised.

## Consumer 2: downstream combiners

The registry's second job is letting downstream algorithm repos (e.g. `libera_rad`) dispatch by
ObsID **without hand-maintaining a duplicate ObsID → product mapping**. A combiner handed a TRIMMED
file looks up its ObsID and CAL product from the registry:

```python
from libera_utils.io.filenaming import LiberaDataProductFilename
from libera_utils.obsids import get_trimmed_obsid_spec

spec = get_trimmed_obsid_spec(LiberaDataProductFilename(trimmed_path).data_product_id)
spec.obsid, spec.cal_product   # (256, cal_swc_365nm)
```

## The other L1A inputs a combiner needs

NOM-HK is the only product the L1A preprocessor trims. A combiner needs more than NOM-HK — the
shortwave LED cal, for instance, also needs PEV-SW, PEC-SW, RAD-SAMPLE, CAL-SAMPLE and AXIS-SAMPLE —
and those arrive as the **full daily L1A granules**. The combiner reads the event's time range off
the TRIMMED NOM-HK filename it was handed and subsets them itself.

Which products those are is stored in `libera_utils/data/cal_combiner_inputs.csv`, one row per
TRIMMED product. It is cross-checked at import against the ObsID catalog, so a TRIMMED product
cannot exist in one file without the other:

```python
from libera_utils.constants import DataProductIdentifier
from libera_utils.obsids import get_combiner_inputs

get_combiner_inputs(DataProductIdentifier.l1a_icie_nom_hk_swc_365nm_trimmed)
# (PEV-SW-STAT-DECODED, PEC-SW-STAT-DECODED,
#  RAD-SAMPLE-DECODED, CAL-SAMPLE-DECODED, AXIS-SAMPLE-DECODED)
```

That tuple, **plus the combiner's own TRIMMED product**, is what its node's `input-products`
should list in libera_cdk's `processing_system_dag.json`. A combiner takes its NOM-HK already
trimmed, so `required_inputs` normally omits the full-day `NOM-HK-DECODED` granule — listing it
would stage a second, redundant NOM-HK input. The lunar, VIIRS lunar and camera combiners, whose
dependency sets are still undecided (LIBSDC-811; the camera combiners with LIBSDC-901), return an
empty tuple.

## Coverage

- Every radiometer (`RAD_CAL`) and camera (`CAM_CAL`) entry has both `trimmed_product` and
  `cal_product` set, so all of them are trim-eligible, and each has a combiner step.
- Science/scan mode entries (Boot-Up, Safe Mode, Stowed, Cross Track, RAP Scan, Along Track, Earth
  Target, the Geo scans) are catalog-only and never trim-eligible.
- Camera cal ObsIDs are registered on `WFOV` only, and the loader enforces that. Science modes are
  dual-registered, since both instruments assert them, but CT video (129-131), RAPS video
  (133-135) and the darks (256-258) are asserted only on `ICIE__SW_OBSID_WFOV`, so
  `get_obsid_spec(NomHkObsidSource.RAD, 129)` raises `KeyError`.

## Extending the registry

When a new calibration ObsID needs to be added:

1. In `libera_utils/constants.py`, add its `CAL` `DataProductIdentifier` member, its
   `l1a_icie_nom_hk_<x>_trimmed` member (`NOM-HK-<CAL ProductID>-TRIMMED`), and its
   `cal_combine_<x>` `ProcessingStepIdentifier` producing only that CAL product, with
   `shared_ecr_name=CAL_RAD_SHARED_ECR_NAME`.
2. Add a row to `libera_utils/data/obsid_registry.csv` naming the TRIMMED and CAL members, with a
   plain-language `description` (it shows up in error messages and this documentation, so keep it
   accurate rather than terse). Edit the file with a text editor or the `csv` module — never a
   spreadsheet app that may rewrite quoting, since descriptions contain commas.
3. Add a row to `libera_utils/data/cal_combiner_inputs.csv` naming the L1A products its combiner
   consumes; import fails if the two files disagree. Leave `required_inputs` empty if the
   dependency set is not settled yet.
4. Do this here, in `libera_utils`, first — not as a local dict in a downstream repo. Downstream
   dispatch should be derived from `get_trimmed_obsid_spec`/`iter_trim_eligible`/`get_obsid_spec`,
   matching the pattern in [Consumer 2](#consumer-2-downstream-combiners) above.
