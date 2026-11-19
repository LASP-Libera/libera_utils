# Footprint Matching (FMATCH)

FMATCH matches each Libera footprint to the external, non-Libera datasets that describe the scene under it: ERA5
reanalysis, IGBP land cover, NISE snow and sea ice, VIIRS surface, cloud and aerosol products and, from the second
operational year, CERES CLDPIX and SSF. For every footprint it aggregates each source's native samples inside
the footprint's point spread function (PSF) to one value, and writes them with the footprint's geolocation and
viewing geometry to an FMATCH product. Scene ID and ADM binning read those products.

The code is in `libera_utils.footprint_matching`:

| Module                      | What it does                                                                    |
| --------------------------- | ------------------------------------------------------------------------------- |
| `footprint_match_algorithm` | The runner: reads an input manifest, writes the products and an output manifest |
| `product`                   | Product assembly: footprints, PSF aggregation, coverage QA, writing the NetCDF  |
| `readers`                   | One reader per external source, registered by its reader key                    |
| `sources`                   | Builds the readers active in a run over its ancillary granules                  |
| `aggregation`               | Collapses a footprint's PSF-weighted samples from a source to one value         |
| `geometry`                  | Turns a footprint's viewing geometry into its latitude/longitude bounding box   |
| `psf`                       | The point spread function: its shape and the contour that sizes a footprint     |
| `weighting`                 | Weights each sample by the PSF and measures the footprint's coverage            |
| `types`                     | `OperationalMode`, footprint types and the quality flags                        |

## The four products

Each FMATCH product is an operational mode, `OperationalMode`, with its own product definition under
`libera_utils/data/product_definitions/fmatch_*.yml`. The modes differ in their time axis, their latency (which sets
the external sources available to them) and their inputs.

| Product                 | Processing step (`ProcessingStepIdentifier`) | Time axis  | Latency         | Libera input     | `Cloud_Fraction_Camera` from |
| ----------------------- | -------------------------------------------- | ---------- | --------------- | ---------------- | ---------------------------- |
| `FMATCH-CAM`            | `aux-fmatch-cam`                             | radiometer | near real time  | L1B `RAD-4CH`    | `CF-CAM`, matched by time    |
| `FMATCH-CAM-CAMTIME`    | `aux-fmatch-cam-camtime`                     | camera     | near real time  | `CF-CAM-CAMTIME` | `CF-CAM-CAMTIME`             |
| `FMATCH-IMAGER`         | `aux-fmatch-imager`                          | radiometer | climate quality | L1B `RAD-4CH`    | not carried                  |
| `FMATCH-IMAGER-CAMTIME` | `aux-fmatch-imager-camtime`                  | camera     | climate quality | `CF-CAM-CAMTIME` | not carried                  |

The IMAGER products carry no camera cloud fraction; their cloud information comes from the CERES and VIIRS readers.

The CAM products run from mission start. The IMAGER products need the RBSP CERES inputs (CLDPIX, SSF), so they
do not run in the first operational year; once those inputs exist, the IMAGER products are produced from mission
start by reprocessing.

The camera-timescale products read their camera pseudo-footprints from the `CF-CAM-CAMTIME` product the Camera
Cloud Fraction algorithm writes. Those pseudo-footprints do not yet carry the spacecraft ephemeris the PSF is sized
from, so a camera-timescale run stops with a `ValueError` before it writes a product (LIBSDC-943).

## Running FMATCH

One command runs every product:

```shell
libera-utils fmatch <input manifest> [--processing_step_id <step>]
```

The `libera-utils-fmatch` target of the repository's `Dockerfile` builds an image whose entrypoint is
`libera-utils fmatch`, so one image serves all four processing steps. From Python, call
`footprint_match_algorithm.run_algorithm(manifest_path, processing_step_id=None)`.

### The input manifest

The manifest (see [Manifest Files](manifest_files.md)) lists every file the run reads in `files`, and its
`configuration` names the product to write:

```json
{
  "manifest_type": "INPUT",
  "files": [
    "s3://dropbox/LIBERA_L1B_RAD-4CH_V1-0-0_20260611T000000_20260611T235959_R...nc",
    "s3://dropbox/LIBERA_L2_CF-CAM_V1-0-0_20260611T000000_20260611T235959_R...nc",
    "s3://ancillary/MODIS-IGBP_061_20250724_R20261020T120000000000.zip",
    "s3://ancillary/NISE_AMSR2_20260611.HDFEOS",
    "s3://ancillary/CLDPROP_D3_VIIRS_NOAA20.A2026162.011.2026166000710.nc"
  ],
  "configuration": {
    "processing_step_id": "aux-fmatch-cam"
  }
}
```

`processing_step_id`
: The `ProcessingStepIdentifier` value of the product to write, one of the four in the table above. It can also be
given with `--processing_step_id`. When both are given they must match, or the run raises
`ProcessingStepIdMismatchError` (from `libera_utils.io.manifest`); when neither is, it raises `ValueError`.

The runner parses every name in `files` once, as a Libera product or an external ancillary product (see the File
Naming page), and picks its inputs by product ID: L1B `RAD-4CH` or `CF-CAM-CAMTIME` as the Libera input, CF-CAM for
FMATCH-CAM, and for each reader the product uses, every granule of that reader's product (see
[Ancillary inputs](#ancillary-inputs)). A file of a product the run does not read is not used. A name that does not
parse, such as a loose MCD12Q1 tile rather than its version archive, raises `FmatchInputError`, as does a manifest
that lists no granule of a reader's product.

Every way a run's inputs can fail it, a missing or unusable input in the manifest or an input without a variable
FMATCH reads, raises `FmatchInputError`, a `ValueError`. The message names the input and what to change.

### Environment and configuration

| Name                         | Required | Meaning                                                                                |
| ---------------------------- | -------- | -------------------------------------------------------------------------------------- |
| `PROCESSING_PATH`            | yes      | The dropbox directory or S3 prefix the products and the output manifest are written to |
| `FMATCH_WORKERS`             | no       | Worker processes for the PSF aggregation: a positive integer, or `auto` (the default)  |
| `FMATCH_WORKER_MEMORY_BYTES` | no       | Memory one worker is assumed to need when `FMATCH_WORKERS` is `auto`                   |

`FMATCH_WORKERS` and `FMATCH_WORKER_MEMORY_BYTES` are `config.json` keys; an environment variable of the same name
overrides them. With `auto`, the pool is the smaller of the CPUs the process may use (its
affinity set capped by its cgroup CPU quota) and the number of workers whose memory fits in what the process can
still use (the lower of its cgroup headroom and the host's `MemAvailable`) after a 512 MiB reserve, and at least one.
Where neither memory figure can be read, it is one worker. `config.json` gives the per-worker memory per product,
since the IMAGER products load the ERA5 pressure levels; an environment value applies to every product. The 512 MiB
reserve, the per-product worker memory and the 50-footprint minimum are estimates until LIBSDC-928 measures
them on full-size inputs. A run with fewer than 50 footprints per worker stays in the calling process.

When it runs in a forked pool, the PSF aggregation's result is identical to the single-process result.

### What a run does

1. Reads the input manifest and selects the product from `processing_step_id`.
2. Sorts the manifest's files by product, and gives each reader the product uses the granules of its product.
3. Selects the Libera inputs from the manifest by product ID. For FMATCH-CAM it also reads every CF-CAM file once
   and matches each L1B footprint to its CF-CAM row by time.
4. Copies any S3 ancillary granule to a temporary local directory, because the readers open real files, and
   unpacks each version archive there (see [Version archives](#version-archives)), then writes the product
   `processing_step_id` selects, one file per Libera input.
5. Builds the output manifest from the input manifest.
6. Adds the written products to it.
7. Writes it to `PROCESSING_PATH`.

The run logs its steps as structured JSON; the input manifest and the ancillary inventory are logged as objects.

## Ancillary inputs

Each external source has a reader, registered under its reader key, that reads one ancillary product, its
`PRODUCT_ID`. The runner gives a reader every granule of that product listed in the manifest's `files`.

| Reader key         | Product (`PRODUCT_ID`)          | Source                                         | Instrument |
| ------------------ | ------------------------------- | ---------------------------------------------- | ---------- |
| `ERA5_Surface`     | `ERA5-SINGLE-LEVEL`             | ERA5 single-level reanalysis                   | ECMWF      |
| `ERA5_Pressure`    | `ERA5-PRESSURE-LEVEL`           | ERA5 pressure-level reanalysis                 | ECMWF      |
| `IGBP`             | `MODIS-IGBP`                    | MODIS MCD12Q1 land cover                       | MODIS      |
| `NISE`             | `NSIDC-ICE-SNOW`                | NISE snow and sea ice extent                   | AMSR2      |
| `VIIRS_BRDF`       | `VIIRS-NOAA20-BRDF`             | VIIRS BRDF (VJ143C1)                           | NOAA20     |
| `VIIRS_CLDPROP_D3` | `VIIRS-NOAA20-CLOUD-PROPERTIES` | VIIRS daily L3 cloud properties                | NOAA20     |
| `VIIRS_AOD`        | `VIIRS-NOAA20-AOD`              | VIIRS Deep Blue aerosol optical depth and type | NOAA20     |
| `CLDPIX`           | `CERES-NOAA20-CLDPIX`           | CERES CLDPIX imager-pixel swath                | NOAA20     |
| `CERES_SSF`        | `CERES-NOAA20-SSF`              | CERES SSF footprint swath                      | NOAA20     |

Which readers a product uses is derived from each reader variable's `modes`, and is held in
`readers.registry.FMATCH_MODE_READERS`. To list them:

```python
from libera_utils.footprint_matching.readers.registry import ReaderRegistry
from libera_utils.footprint_matching.types import OperationalMode

ReaderRegistry.get_readers_for_mode(OperationalMode.CAM)
```

Each reader takes the granule or granules of its source for the period; a reader that reads one granule per period
raises when it is handed more.

Before a product is written, the runner checks each granule's date against the Libera input's footprint times: a
granule whose own time span, read from its metadata (`AncillaryReader.granule_span`), does not overlap the footprint
times widened by its reader's `GRANULE_DATE_WINDOW` raises `FmatchInputError`, naming the granule, its reader and
the range it had to overlap. The window is one day either side for every reader, so a granule from the input's
previous or next day is accepted (LIBSDC-942). IGBP has none, since MCD12Q1 is an annual map whose newest year
is applied to later dates.

### Version archives

IGBP is version-partitioned: every tile of one map year of an MCD12Q1 collection is ingested as one zip, named
`<ProductID>_<version>_<production YYYYMMDD>_<revision>.zip`, where the version is the three-digit collection
(`AncillaryVersionArchiveFilename`; see the File Naming page). A manifest lists that one archive for `IGBP`, not its
tiles. The runner unpacks the archive once per run into the temporary local directory, keeping only the files named
as MCD12Q1 tiles, wherever they sit in the zip, and hands those tiles to the reader. An S3 archive is copied first and
its copy deleted once unpacked. One map year is about 1.5 GB of tiles, so the task needs that much free local
storage, and twice that while an S3 archive is copied.

The run raises `FmatchInputError` when a reader is given an archive alongside any other granule, or when an archive
is not a zip, holds no file of its product, holds a file in its product's form whose fields do not parse, holds two
files with one name, or holds files of more than one release (collection, or map year). A loose MCD12Q1 tile in
`files` does not parse as an ingested product name, so even a local or development run lists the archive.

## Inputs by time axis

### Radiometer time: L1B RAD-4CH

The runner reads the footprint geolocation and viewing angles from L1B `RAD-4CH`, which the product carries unchanged
(`product.L1B_PASSTHROUGH_VARIABLES`), and the scan-reference geometry (subsatellite point, satellite radius,
cone-angle rate) that orients each footprint's PSF. FMATCH writes no row for a footprint with a non-finite value in
any of those, other than the cone-angle rate; L1B writes them as fill where the boresight misses the Earth. A file
with no usable footprint raises `ValueError`.

FMATCH-CAM also needs CF-CAM, whose `Cloud_Fraction` (percent) it writes as `Cloud_Fraction_Camera`. CF-CAM carries
the L1B `radiometer_time` values unchanged, and its rows are matched to the footprints by exact time:

- a footprint with no CF-CAM row gets fill, since the camera sees only part of the radiometer track;
- a CF-CAM row outside the L1B granule's time span belongs to another granule and is ignored;
- a CF-CAM row at the time of a footprint FMATCH writes no row for is ignored;
- a CF-CAM row with a value inside the span that matches no L1B time raises `CloudFractionTimeMismatchError`, as do
  two rows at the same time, or no L1B time matching any row;
- a missing CF-CAM input, or one without `Cloud_Fraction` or `radiometer_time`, raises `FmatchInputError`.

### Camera time: CF-CAM-CAMTIME

The camera-timescale products read their pseudo-footprints, one per real cell of the `CAMERA_TIME` by
`PSEUDOFOOTPRINT` grid, from `CF-CAM-CAMTIME` (`product.pseudofootprints_from_camtime_dataset`). FMATCH-CAM-CAMTIME
also reads `Cloud_Fraction` from it. A missing pseudo-footprint variable raises `FmatchInputError`.

## What the product carries

Each variable a product definition declares is filled one of three ways:

- **Pass-through**: the geolocation and viewing angles, from L1B or from the pseudo-footprints, and each footprint's
  PSF bounding box (`PSF_Bbox_*`).
- **Computed**: `Sunglint_Angle`; every external variable, named `<READER_KEY>_<variable>` (for example
  `ERA5_Surface_Wind_U10`) with a `long_name` ending in the reader's instrument; and the coverage variables
  `PSF_Coverage_Fraction` and `Quality_Flag`.
- **Fill**: a footprint with no usable sample for a variable gets `NaN`, or `-1` for an integer category
  (`product.FMATCH_INTEGER_FILL_VALUE`).

Coverage is the fraction of the PSF's energy backed by usable samples, the lowest across the active sources.
Following the CERES coverage rule (Footprint Matching design doc, sections 2.4.2.7 and 2.8.1.1.3;
`aggregation.ACCEPT_COVERAGE_THRESHOLD` and `aggregation.PARTIAL_COVERAGE_THRESHOLD`), a footprint is kept and
flagged rather than discarded: coverage from 75% to below 95% sets
`PARTIAL_COVERAGE` in `Quality_Flag`, below 75% sets `INSUFFICIENT_COVERAGE`, a bounding box clipped at the limb sets
`LIMB_TRUNCATED`, and a view off the Earth's limb sets `OFF_LIMB` (`types.FmatchCoverageFlag`).

The product's `InputGranules` attribute names the Libera input, any CF-CAM input and every ancillary granule it was
made from. It is the product's record of its inputs until the UMM-G writer (LIBSDC-869) records them.

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
   `cldpix.py` and `ceres_ssf.py`), and update the module docstring's product and file-naming lines.
2. In every FMATCH product definition, change the instrument in parentheses at the end of each affected `long_name`,
   for example `(NOAA20)` to `(NOAA21)`.
3. If the new product's file layout differs (group paths, variable names, dimension order, fill or geolocation
   conventions), update the reader's field table and its `_load_granule` or `_load_native`, and add a cut of a real
   granule under `tests/test_data/footprint_matching/` with a test that reads it.

If the new source is a different ancillary product, set the readers' `PRODUCT_ID` to it, adding its
`DataProductIdentifier` and filename class if it has none. The reader key and the product variable names do not
change, so downstream consumers are unaffected unless step 3 changes a variable.

### Add a variable to an existing reader

1. Add the field to the reader's table: `_CLDPIX_FIELDS` in `cldpix.py`, `_SSF_BASE_FIELDS` or
   `_IMAGER_SSF_SOURCES` in `ceres_ssf.py`, `_D3_FIELDS` in `viirs.py`, or the reader's `VARIABLES` elsewhere.
   Give it its aggregation, the products that carry it in `modes`, its valid range with the document it comes from,
   and for a mode-aggregated code `n_categories` and an `int16` dtype.
2. Add the product variable to each product definition whose mode carries it, named `<READER_KEY>_<variable name>`
   exactly as the reader spells them (e.g. `ERA5_Surface_Wind_U10`), with a `long_name` ending in `(<INSTRUMENT>)` and
   its units. A continuous variable also gets a `..._Standard_Deviation` companion, and an integer category declares
   `_FillValue: -1`. A variable the reader stacks in `PRODUCT_STACKS` (ERA5's pressure levels) is published as its
   stack instead, named from `<reader key>_<stack name>` on the stack's dimension.
3. Extend the reader's synthetic fixture in `tests/plugins/footprint_matching_fixtures.py` and its tests.

### Add a reader or change which products a reader feeds

1. Write the reader as a subclass of `StaticReader`, `SteppedGridReader` or `PassReader` with a unique `READER_KEY`
   and the `DataLevel.ANC` product it reads as `PRODUCT_ID`, and import it in
   `libera_utils/footprint_matching/readers/__init__.py` to register it. A product with no `DataProductIdentifier` or
   filename class needs both first, so its granules' names parse.
2. Name the products it feeds in each of its variables' `modes`; `FMATCH_MODE_READERS` is derived from them.
3. Add its variables to those products' definitions, as above.
4. List its product's granules in the manifest's `files`; the runner derives the readers it needs from the registry
   and gives each the granules of its product.

Renaming a reader key renames its product variables, so it is a change consumers must follow.

### Add a product variant

A variant needs:

1. an `OperationalMode` member and its product definition, `fmatch_<mode>.yml`, listed in
   `product.FMATCH_DEFINITION_FILENAMES`; a camera-timescale mode is also added to `product._CAMERA_TIMESCALE_MODES`,
   or it is assembled on the radiometer time axis;
2. the mode named in the `modes` of each reader variable it carries, which gives it its active readers and its
   external variables;
3. a `ProcessingStepIdentifier` member for the step that writes it, with the product it writes;
4. an `FmatchRunnerConfig` (the mode, its Libera input product, its cloud-fraction input or `None`, its log label
   and, for a camera-timescale mode, its footprint loader) added to `RUNNER_CONFIGS` under that step.

The `--processing_step_id` choices and the steps `run_algorithm` accepts follow `RUNNER_CONFIGS`.
