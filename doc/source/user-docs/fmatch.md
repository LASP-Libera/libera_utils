# Footprint Matching (FMATCH)

FMATCH matches each Libera footprint to the external, non-Libera datasets that describe the scene under it: ERA5
reanalysis, IGBP land cover, NISE snow and sea ice, VIIRS surface, cloud and aerosol products and, from the second
operational year, CERES CLDPIX and FLASHFlux. For every footprint it aggregates each source's native samples inside
the footprint's point spread function (PSF) to one value, and writes them with the footprint's geolocation and
viewing geometry to an FMATCH product. Scene ID, Camera Cloud Fraction and ADM binning read those products.

The code is in `libera_utils.footprint_matching`:

| Module                         | What it does                                                                    |
| ------------------------------ | ------------------------------------------------------------------------------- |
| `footprint_match_algorithm`    | The runner: reads an input manifest, writes the products and an output manifest |
| `product`                      | Product assembly: footprints, PSF aggregation, coverage QA, writing the NetCDF  |
| `readers`                      | One reader per external source, registered by its reader key                    |
| `sources`, `aggregation`       | Build the readers for a run; collapse a footprint's samples to one value        |
| `geometry`, `psf`, `weighting` | Footprint bounding boxes and the PSF weights of each sample                     |
| `types`                        | `OperationalMode`, footprint types and the quality flags                        |

## The four products

Each FMATCH product is an operational mode, `OperationalMode`, with its own product definition under
`libera_utils/data/product_definitions/fmatch_*.yml`. The modes differ in their time axis, their latency (which sets
the external sources available to them) and their inputs.

| Product                 | Processing step             | Time axis  | Latency         | Primary input    | Cloud fraction                 |
| ----------------------- | --------------------------- | ---------- | --------------- | ---------------- | ------------------------------ |
| `FMATCH-CAM`            | `aux-fmatch-cam`            | radiometer | near real time  | L1B `RAD-4CH`    | CF-CAM, matched by time        |
| `FMATCH-CAM-CAMTIME`    | `aux-fmatch-cam-camtime`    | camera     | near real time  | `CF-CAM-CAMTIME` | carried in `CF-CAM-CAMTIME`    |
| `FMATCH-IMAGER`         | `aux-fmatch-imager`         | radiometer | climate quality | L1B `RAD-4CH`    | none (from the imager readers) |
| `FMATCH-IMAGER-CAMTIME` | `aux-fmatch-imager-camtime` | camera     | climate quality | `CF-CAM-CAMTIME` | none (from the imager readers) |

The CAM products run from mission start. The IMAGER products need the RBSP CERES inputs (CLDPIX, FLASHFlux), so they
do not run in the first operational year.

The camera-timescale products rebuild their camera pseudo-footprints from the `CF-CAM-CAMTIME` product the Camera
Cloud Fraction algorithm writes. Those pseudo-footprints do not yet carry the spacecraft ephemeris the PSF is sized
from, so a camera-timescale run stops with a `ValueError` before it writes a product (LIBSDC-884).

## Running FMATCH

One command runs every product:

```shell
libera-utils fmatch <input manifest> [--processing_step_id <step>]
```

The `libera-utils-fmatch` target of the repository's `Dockerfile` builds an image whose entrypoint is
`libera-utils fmatch`, so one image serves all four processing steps. From Python, call
`footprint_match_algorithm.run_algorithm(manifest_path, processing_step_id=None)`.

### The input manifest

The manifest (see [Manifest Files](manifest_files.md)) lists every file the run reads in `files`, and two
`configuration` keys say what to do with them:

```json
{
  "manifest_type": "INPUT",
  "files": [
    "s3://dropbox/LIBERA_L1B_RAD-4CH_V1-0-0_20260611T000000_20260611T235959_R...nc",
    "s3://dropbox/LIBERA_L2_CF-CAM_V1-0-0_20260611T000000_20260611T235959_R...nc",
    "s3://ancillary/MCD12Q1.A2024001.h21v07.061.2025205222320.hdf",
    "s3://ancillary/NISE_AMSR2_20260610.HDFEOS",
    "s3://ancillary/CLDPROP_D3_VIIRS_NOAA20.A2026161.011.2026165000710.nc"
  ],
  "configuration": {
    "processing_step_id": "aux-fmatch-cam",
    "ancillary_inputs": {
      "IGBP": ["s3://ancillary/MCD12Q1.A2024001.h21v07.061.2025205222320.hdf"],
      "NISE": ["s3://ancillary/NISE_AMSR2_20260610.HDFEOS"],
      "VIIRS_CLDPROP_D3": [
        "s3://ancillary/CLDPROP_D3_VIIRS_NOAA20.A2026161.011.2026165000710.nc"
      ]
    }
  }
}
```

`processing_step_id`
: The `ProcessingStepIdentifier` value of the product to write, one of the four in the table above. It can also be
given with `--processing_step_id`. When both are given they must match, or the run raises
`ProcessingStepIdMismatchError`; when neither is, it raises `ValueError`.

`ancillary_inputs`
: Maps each reader key active for the product to the ancillary granules listed for it. Every path must also be in
`files`. The mapping must name exactly the readers the product uses (see
[Ancillary inputs](#ancillary-inputs)), each with at least one granule, or the run raises `AncillaryManifestError`.
When the key is absent, the runner reads the granules from a directory tree under `FMATCH_ANCILLARY_PATH`
instead.

The runner picks its Libera inputs out of `files` by their parsed product IDs: L1B `RAD-4CH` or `CF-CAM-CAMTIME` as
the primary input, and CF-CAM for FMATCH-CAM. Files whose names are not Libera product names, such as the ancillary
granules, are skipped there.

### Environment and configuration

| Name                         | Required | Meaning                                                                                |
| ---------------------------- | -------- | -------------------------------------------------------------------------------------- |
| `PROCESSING_PATH`            | yes      | The dropbox directory or S3 prefix the products and the output manifest are written to |
| `FMATCH_ANCILLARY_PATH`      | no       | Root of a staged ancillary tree, read only when the manifest has no `ancillary_inputs` |
| `FMATCH_WORKERS`             | no       | Worker processes for the PSF aggregation: a positive integer, or `auto` (the default)  |
| `FMATCH_WORKER_MEMORY_BYTES` | no       | Memory one worker is assumed to need when `FMATCH_WORKERS` is `auto`                   |

`FMATCH_WORKERS` and `FMATCH_WORKER_MEMORY_BYTES` are `config.json` keys; an environment variable of the same name
overrides them. With `auto`, the pool is the smaller of the CPUs the process may use (its
affinity set capped by its cgroup CPU quota) and the number of workers whose memory fits in what the process can
still use (the lower of its cgroup headroom and the host's `MemAvailable`) after a 512 MiB reserve, and at least one.
Where neither memory figure can be read, it is one worker. `config.json` gives the per-worker memory per product,
since the IMAGER products load the ERA5 pressure levels; an environment value applies to every product. The defaults
are estimates until they are measured on full-size inputs. A run with fewer than 50 footprints per worker stays in
the calling process.

When it runs in a forked pool, the PSF aggregation's result is identical to the single-process result.

### What a run does

1. Reads the input manifest and selects the product from `processing_step_id`.
2. Locates the ancillary granules: the manifest's `ancillary_inputs`, or the tree under `FMATCH_ANCILLARY_PATH`.
   When the manifest lists them and the variable is also set, the variable is ignored with a warning.
3. Selects the primary inputs from the manifest, and for FMATCH-CAM reads every CF-CAM input once.
4. Copies any S3 ancillary granule to a temporary local directory, because the readers open real files, then writes
   one FMATCH product per primary input.
5. Builds the output manifest from the input manifest.
6. Adds the written products to it.
7. Writes it to `PROCESSING_PATH`.

The run logs its steps as structured JSON; the input manifest and the ancillary inventory are logged as objects.

## Ancillary inputs

Each external source has a reader, registered under its reader key:

| Reader key         | Source                                         | Instrument |
| ------------------ | ---------------------------------------------- | ---------- |
| `ERA5_Surface`     | ERA5 single-level reanalysis                   | ECMWF      |
| `ERA5_Pressure`    | ERA5 pressure-level reanalysis                 | ECMWF      |
| `IGBP`             | MODIS MCD12Q1 land cover                       | MODIS      |
| `NISE`             | NISE snow and sea ice extent                   | AMSR2      |
| `VIIRS_BRDF`       | VIIRS BRDF (VJ143C1)                           | NOAA20     |
| `VIIRS_CLDPROP_D3` | VIIRS daily L3 cloud properties                | NOAA20     |
| `VIIRS_AOD`        | VIIRS Deep Blue aerosol optical depth and type | NOAA20     |
| `CLDPIX`           | CERES CLDPIX imager-pixel swath                | NOAA20     |
| `FLASHFlux`        | CERES FLASHFlux footprint swath                | NOAA20     |

Which readers a product uses is derived from each reader variable's `modes`, and is held in
`readers.registry.FMATCH_MODE_READERS`. To list them:

```python
from libera_utils.footprint_matching.readers.registry import ReaderRegistry
from libera_utils.footprint_matching.types import OperationalMode

ReaderRegistry.get_readers_for_mode(OperationalMode.CAM)
```

Each reader takes the granule or granules of its source for the period; a reader that reads one granule per period
raises when it is handed more.

### The directory tree

Without `ancillary_inputs` in the manifest, the runner looks under `FMATCH_ANCILLARY_PATH` for one subdirectory per
active reader, named by its reader key, holding that reader's granules:

```text
$FMATCH_ANCILLARY_PATH/
    ERA5_Surface/       ERA5 single-level granules
    ERA5_Pressure/      ERA5 pressure-level granules
    IGBP/               MCD12Q1 land-cover granules
    NISE/               NISE snow/ice granules
    VIIRS_BRDF/         VIIRS BRDF granules
    VIIRS_CLDPROP_D3/   VIIRS daily L3 cloud-property granules
    VIIRS_AOD/          VIIRS Deep Blue aerosol granules
    CLDPIX/             CERES CLDPIX granules    (IMAGER products)
    FLASHFlux/          CERES FLASHFlux granules (IMAGER products)
```

A run with neither `ancillary_inputs` nor `FMATCH_ANCILLARY_PATH` raises `ValueError`. A missing root, a missing
reader directory or an empty one raises `FileNotFoundError`. The tree may be local or in
S3. No SDC pipeline stages these granules yet, since Data Ingester does not handle them, so the tree is for local and
development runs; a processing step lists the granules in its manifest.

## Inputs by time axis

### Radiometer time: L1B RAD-4CH

The runner reads the footprint geolocation and viewing angles from L1B `RAD-4CH`, which the product carries unchanged
(`product.L1B_PASSTHROUGH_VARIABLES`), and the scan-reference geometry (subsatellite point, satellite radius,
cone-angle rate) that orients each footprint's PSF. A footprint with a non-finite value in any of those, other than
the cone-angle rate, is dropped: L1B writes them as fill where the boresight misses the Earth. A file with no usable
footprint raises `ValueError`.

FMATCH-CAM also needs CF-CAM, whose `Cloud_Fraction` (percent) it writes as `Cloud_Fraction_Camera`. CF-CAM carries
the L1B `radiometer_time` values unchanged, and its rows are matched to the footprints by exact time:

- a footprint with no CF-CAM row gets fill, since the camera sees only part of the radiometer track;
- a CF-CAM row outside the L1B granule's time span belongs to another granule and is ignored;
- a CF-CAM row with a value inside the span that matches no footprint raises `CloudFractionTimeMismatchError`, as do
  two rows at the same time, or no footprint matching any row;
- a missing CF-CAM input, or one without `Cloud_Fraction` or `radiometer_time`, raises
  `MissingCloudFractionInputError`.

### Camera time: CF-CAM-CAMTIME

The camera-timescale products read their pseudo-footprints, one per real cell of the `CAMERA_TIME` by
`PSEUDOFOOTPRINT` grid, from `CF-CAM-CAMTIME` (`product.pseudofootprints_from_camtime_dataset`). FMATCH-CAM-CAMTIME
also reads `Cloud_Fraction` from it. A missing pseudo-footprint variable raises `MissingCamtimeVariablesError`.

## What the product carries

Each column a product definition declares is filled one of three ways:

- **Pass-through**: the geolocation and viewing angles, from L1B or from the pseudo-footprints, and each footprint's
  PSF bounding box (`PSF_Bbox_*`).
- **Computed**: `Sunglint_Angle`; every external variable, named `<READER_KEY>_<variable>` (for example
  `ERA5_Surface_Wind_U10`) with a `long_name` ending in the reader's instrument; and the coverage columns
  `PSF_Coverage_Fraction` and `Quality_Flag`.
- **Fill**: a footprint with no usable sample for a variable gets `NaN`, or `-1` for an integer category
  (`product.FMATCH_INTEGER_FILL_VALUE`).

Coverage is the fraction of the PSF's energy backed by usable samples, the lowest across the active sources.
Following the CERES rule, a footprint is kept and flagged rather than discarded: coverage from 75% to below 95% sets
`PARTIAL_COVERAGE` in `Quality_Flag`, below 75% sets `INSUFFICIENT_COVERAGE`, a bounding box clipped at the limb sets
`LIMB_TRUNCATED`, and a view off the Earth's limb sets `OFF_LIMB` (`types.FmatchCoverageFlag`).

The product's `InputGranules` attribute names the primary input, any CF-CAM input and every ancillary granule it was
made from.

## Extending FMATCH

Reader configuration is code, not a configuration file: each reader declares its key, its source and its variables
as class attributes and field tables, and the FMATCH product definitions are written to match. Two tests in
`tests/unit/test_footprint_matching/test_fmatch_definitions.py` hold the two sides together:
`test_external_variables_match_readers` fails when a product definition lacks a reader variable or has a different
dtype, and `test_reader_instrument_is_recorded_in_long_name` fails when a `long_name` does not end with the reader's
`INSTRUMENT`. Run them after any of the changes below.

### Swap the satellite or instrument a source comes from

For example, moving the VIIRS readers from NOAA-20 to NOAA-21:

1. In each affected reader, set `INSTRUMENT` (`INSTRUMENT = "NOAA21"` in `aod.py`, `brdf.py`, `viirs.py`,
   `cldpix.py` and `flashflux.py`), and update the module docstring's product and file-naming lines.
2. In every FMATCH product definition, change the instrument in parentheses at the end of each affected `long_name`,
   for example `(NOAA20)` to `(NOAA21)`.
3. If the new product's file layout differs (group paths, variable names, dimension order, fill or geolocation
   conventions), update the reader's field table and its `_load_granule` or `_load_native`, and add a cut of a real
   granule under `tests/test_data/footprint_matching/` with a test that reads it.

The reader key and the product variable names do not change, so the manifest's `ancillary_inputs`, the staging tree
and downstream consumers are unaffected unless step 3 changes a variable.

### Add a variable to an existing reader

1. Add the field to the reader's table: `_CLDPIX_FIELDS` in `cldpix.py`, `_FLASHFLUX_BASE_FIELDS` or
   `_IMAGER_FLASHFLUX_SOURCES` in `flashflux.py`, `_D3_FIELDS` in `viirs.py`, or the reader's `VARIABLES` elsewhere.
   Give it its aggregation, the products that carry it in `modes`, its valid range with the document it comes from,
   and for a mode-aggregated code `n_categories` and an `int16` dtype.
2. Add the product variable to each product definition whose mode carries it, named `<READER_KEY>_<variable name>`
   exactly as the reader spells them (e.g. `ERA5_Surface_Wind_U10`), with a `long_name` ending in `(<INSTRUMENT>)` and
   its units. A continuous variable also gets a `..._Standard_Deviation` companion, and an integer category declares
   `_FillValue: -1`. A variable the reader stacks in `PRODUCT_STACKS` (ERA5's pressure levels) is published as its
   stack instead, named from `<reader key>_<stack name>` on the stack's dimension.
3. Extend the reader's synthetic fixture in `tests/plugins/footprint_matching_fixtures.py` and its tests.

### Add a reader or change which products a reader feeds

1. Write the reader as a subclass of `StaticReader`, `SteppedGridReader` or `PassReader` with a unique `READER_KEY`,
   and import it in `libera_utils/footprint_matching/readers/__init__.py` to register it.
2. Name the products it feeds in each of its variables' `modes`; `FMATCH_MODE_READERS` is derived from them.
3. Add its variables to those products' definitions, as above.
4. List its granules under its reader key in the manifest's `ancillary_inputs`, or stage them under
   `$FMATCH_ANCILLARY_PATH/<reader key>/`; the runner derives the readers it needs from the registry.

Renaming a reader key renames its product variables, its `ancillary_inputs` key and its staging directory, so it is a
change consumers and whatever writes the manifests must follow.

### Add a product variant

A variant needs:

1. an `OperationalMode` member and its product definition, `fmatch_<mode>.yml`, listed in
   `product.FMATCH_DEFINITION_FILENAMES`; a camera-timescale mode is also added to `product._CAMERA_TIMESCALE_MODES`,
   or it is assembled on the radiometer time axis;
2. the mode named in the `modes` of each reader variable it carries, which gives it its active readers and its
   external columns;
3. a `ProcessingStepIdentifier` member for the step that writes it, with the product it writes;
4. an `FmatchRunnerConfig` (the mode, its primary input product, its cloud-fraction input or `None`, its log label
   and, for a camera-timescale mode, its footprint loader) added to `RUNNER_CONFIGS` under that step.

The `--processing_step_id` choices and the steps `run_algorithm` accepts follow `RUNNER_CONFIGS`.
