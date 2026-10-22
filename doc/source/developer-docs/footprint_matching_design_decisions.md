# Footprint matching: design decisions and their consequences

This page records the decisions behind the footprint-matching (FMATCH) data path, from a
staged ancillary granule to one value per Libera footprint. It is written for two
readers: a scientist who needs to know what a product value means and where a choice
could bias it, and a developer who needs to know why the code is shaped as it is and
what to change if a choice is revisited. Each section states the decision, what it means
for the product, the alternatives considered, and the change that would reverse it.

The code is in `libera_utils/footprint_matching/`: the readers (`readers/`), the active
sources of a run (`sources.py`), the PSF weighers (`weighting.py`), the aggregation engine
(`aggregation.py`) and the orchestrator (`product.py`).

## 1. A value is weighted once, by the PSF, at its native location

**Decision.** A reader serves the native samples of its source inside a footprint's
bounding box: every pixel, CERES footprint or grid point at its own position, as a
`SampleSet`. The weigher evaluates the point spread function (PSF) at each sample and the
aggregation engine reduces the weighted samples to one statistic per variable. Nothing
averages, bins, re-grids or interpolates a value before the weigher sees it.

**Why.** An earlier design rasterized every source onto a regular 2 degree tile grid
(0.05 degree cells for 1 km sources, 0.2 degree for CERES FLASHFlux) and then PSF-weighted the
cells. That is two averaging stages. On a synthetic 1 km field the second stage reproduced
the PSF-weighted mean to within 2 percent, but reported one fifth of the true
within-footprint standard deviation (the variance between cell means, not between pixels)
and dropped a class that was 40 percent of every cell from the ranked IGBP scenes, because a
per-cell mode has no minority. The `_standard_deviation` and ranked-scene variables of the
FMATCH products were therefore wrong by construction.

**Consequence for interpretation.** The `<source>_<variable>_standard_deviation` variables
are the PSF-weighted spread of the source's native samples inside the footprint. The
ranked `surface_type_secondary` / `_tertiary` and `aerosol_type_*` variables rank classes
by PSF-weighted native area. A source with a native cell larger than the footprint (ERA5
at 0.25 degree, the 1 degree VIIRS composites) contributes one to four samples; its mean is
then close to a nearest-cell value and its standard deviation close to zero. That is a
property of the source, not of the processing.

**Alternatives.** Keep tiling and fix only the standard deviation (not possible: the
rasterizer is the second averaging stage). Pre-integrated PSF angular bins (CERES ATBD
4.4, Eq. 4.4-18): compatible with this design as a later optimization of the weigher.

**To reverse.** Not recommended. Any intermediate grid must be introduced in the reader's
`_load_native` and would reintroduce the biases above.

## 2. Each sample is matched in time, not only in space

**Decision.** A footprint carries its observation time (`RADIOMETER_TIME` or the camera
time) and each reader declares a `TimeMatch` policy applied after the geographic query:

| Policy         | Sources                                                                         | What is served                                                                                                                              |
| -------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `DAILY`        | IGBP land cover, NISE snow/ice, VIIRS BRDF, VIIRS Deep Blue AOD, VIIRS L3 cloud | Every sample in the box; the staged daily granule is the match                                                                              |
| `NEAREST_STEP` | ERA5 single-level and pressure-level                                            | The hourly step nearest the footprint time, no interpolation; none beyond `MAX_TIME_OFFSET` (1 h)                                           |
| `NEAREST_PASS` | CERES FLASHFlux, CERES CLDPIX                                                   | The samples within `PASS_WINDOW` (15 min) of the smallest sample-to-footprint offset; none when that offset exceeds `MAX_TIME_OFFSET` (3 h) |

**Why.** A 24 h staging period holds several observations of one ground location: two or
more swath passes, 24 reanalysis steps. Matching on location alone mixed them. The earlier
ERA5 readers served the file's first step to every footprint of the day, and the FLASHFlux and
CLDPIX readers read no time at all.

**Consequence for interpretation.** An ERA5 value is the analysis up to 30 minutes from
the footprint time. A CERES value comes from the pass nearest the Libera observation;
Libera and the CERES host are different spacecraft, so that pass can be tens of minutes
away, and the offset is not recorded in the product (see section 7). A footprint with no
source sample within the window gets fill for that source, zero coverage, and the
`INSUFFICIENT_COVERAGE` flag.

**Alternatives.** Linear interpolation between ERA5 steps (a transform; it would belong in
the weigher, not the reader, if the science team asks for it). A fixed window around the
footprint time instead of the nearest pass (admits the opposite pass at the poles where
passes are 100 minutes apart).

**To change.** The window and offset are class attributes on each reader
(`PASS_WINDOW`, `MAX_TIME_OFFSET`); the policy is `TIME_MATCH`. Their values are a science
decision and should be confirmed against real CERES and ERA5 inputs.

**Verified on granules.** On a CER_SSF_NOAA20-FM6-VIIRS granule (195 k footprints, one
hour) the time is `Time_and_Position/time`, CF days since 1970 in the gregorian calendar,
and loads in under a second. On a CER_CLDPIX_NOAA20-VIIRS granule (16 120 scanlines by
400 pixels, one hour) the time is `Observation_Time`, per pixel, in fractional UTC hours
of the day with no date; the reader dates it from the granule's `time_coverage_start`
attribute, which is safe because a granule covers one hour and never crosses midnight.
Both products are hourly, so a day is 24 granules per source, which the readers
concatenate. A fill or out-of-range time decodes to NaT and that sample is dropped.

## 3. Coverage is measured on a reference grid, with each sample covering its native cell

**Decision.** The weigher lays a fine reference grid over the PSF contour in its own frame
(angular for the CERES-faithful weigher, ground-plane for the radial stand-in) holding the
PSF energy of every cell. Each finite sample covers the square centered on it whose
half-width is the circumradius of the source's native cell, `RESOLUTION_KM / sqrt(2)`, so
a regular lattice of cells leaves no gap at any rotation relative to the reference axes.
Coverage is the share of the reference energy inside at least one such square. The
footprint's `psf_coverage_fraction` is the minimum over the active sources and drives the
CERES 75 / 95 percent flags. The reader's box query is widened by half a native cell so a
sample whose cell overlaps the contour's rim is served even when its center lies outside
the box; the PSF weight of such a sample is zero, so only the coverage changes.

The pass readers measure the cell size on each granule rather than trusting the
instrument's nominal value: the staged CLDPIX test granule carries 400 of the 3200 VIIRS
pixels per scanline (7.2 km across-track, 2 km along-track), and the SSF granule lays
footprints 12.6 km apart along a scan but 44 km apart between scans. Each reader takes
the larger of the two directions, so a complete but anisotropic swath scores complete
coverage, at the price of not seeing a gap narrower than that larger spacing.

**Why.** With native samples a swath gap simply has no samples, so the earlier ratio
(weight on finite cells over weight on all cells in the box) read 100 percent across a
gap. A denominator independent of what was served is needed.

**Consequence for interpretation.** Coverage now responds to swath gaps and to time
mismatches. A source whose cell is wider than the contour (the 1 degree VIIRS composites)
is complete as soon as one sample lies in the contour. ERA5 declares `RESOLUTION_KM = 28`,
the 0.25 degree spacing, so its cells abut and a footprint inside the grid is complete;
declaring a smaller value would leave gaps between samples and flag every footprint
partial. The circumradius over-credits the edge of a swath gap by about a fifth of a cell
(0.2 km for a 1 km imager). The reference grid steps are 0.02 degree of angle and 0.25 km
on the ground.

**Alternatives.** Keep the finite-over-all ratio (silently wrong across gaps). Count
samples against an expected count from the source resolution (density-dependent).

**To change.** `_covered_fraction` and the two `_*_reference` functions in `weighting.py`;
the per-source cell size is `RESOLUTION_KM`.

## 4. Samples are not area-weighted

**Decision.** A sample's weight is the PSF value at its location, with no factor for the
area the sample represents. This matches the CERES convolution (ATBD 4.4, Eq. 4.4-17),
which sums the PSF over imager pixels.

**Consequence for interpretation.** Where native sampling density varies inside a
footprint (imager bow-tie growth toward the scan edge, overlapping CERES footprints), denser
regions carry more total weight. The rasterizer hid this by averaging per cell; the native
path exposes it. The effect is small for the Libera footprint relative to a 1 km imager,
and absent for regular grids.

**To change.** Multiply the weights by a per-sample area in the weigher. The reader would
have to serve a per-sample area, which no source currently provides.

## 5. Coarse grids are point-sampled at their cell centers

**Decision.** A gridded source contributes the grid points inside the box, each weighted
by the PSF at its center. No area integration of the PSF over the cell and no spatial
interpolation of the field.

**Consequence for interpretation.** For ERA5 and the 1 degree VIIRS composites the
footprint value is a PSF-weighted combination of the one to four nearest grid values, not
a bilinear interpolation to the boresight. Two adjacent footprints can take different
values as the boresight crosses a cell boundary.

**To change.** A per-variable option in the weigher (integrate the PSF over the cell, or
interpolate the field). Both are transforms and belong in the weigher, not the reader.

## 6. Zeros in geometric means are dropped

**Decision.** `weighted_log_mean` (cloud optical depth, aerosol optical depth) uses the
positive samples only. The earlier rasterizer floored zeros to a small positive value
before its own geometric mean; that floor is gone with the rasterizer, so one convention
remains.

**Consequence for interpretation.** A clear sample (optical depth 0) does not pull the
geometric mean down; the mean describes the cloudy or aerosol-laden samples. A reviewer
on the gridded-readers pull request asked for exactly this, as a log mean over values
expected to be zero is numerically unstable. Whether a clear fraction should be reported
alongside is a product definition question.

**To change.** `_log_mean_from_partials` in `aggregation.py`.

## 7. The time offset of the matched sample is not a product variable

**Decision.** The product records which source was matched and its coverage, not how far
in time the matched pass or step was from the footprint.

**Consequence for interpretation.** A consumer cannot tell a 2 minute CERES match from a
170 minute one. The offset is bounded by `MAX_TIME_OFFSET`.

**To change.** Add a `<source>_time_offset` variable (signed minutes) to the product
definitions and have `aggregate_external_variables` fill it from the served samples'
times. Separable and recommended.

## 8. Software design choices

### One reader contract, three base classes

`StaticReader` loads its whole source once (`_load_native`) and the base class indexes it.
`SteppedGridReader` lists step times and loads one step on demand. `PassReader` reads each
granule's time span and holds only the granules within `MAX_TIME_OFFSET` of the current
footprint time, loading one when the footprints reach its window and releasing it when
they leave; a CLDPIX hour is 6.4 million pixels and a few hundred megabytes indexed, so a
day of 24 granules could not be held at once, and the window holds at most seven. All
three answer `samples_in(bbox, time)`. This keeps the per-reader code to "open the file, return native
samples" and puts the box query, the antimeridian split, the polar box and the time
selection in one place. The trade-off is that a stepped source reloads a step whenever
the nearest step changes, which is cheap only because footprints arrive in time order.

### A point cloud is sorted once into 1 degree bins

A point source (millions of CLDPIX pixels) is permuted once at load into latitude and
longitude bins held as offset ranges, and the unsorted arrays are released, so there is
one copy of the data. A box query touches only the bins it overlaps before the exact
box mask. The earlier design re-filtered the whole point cloud per tile.

### No tile cache, no prefetch

Every reader already held its whole source in memory; the tile cache, the merged-tile
cache and the weigher's per-tile geometry cache were further copies of the same data, and
the prefetch thread could only overlap one read with compute because the HDF5 stack is
serialized by a global lock. All of it is gone. Memory is one copy per source (one step
for ERA5, the granules in the time window for FLASHFlux and CLDPIX), and the per-footprint cost
is one box query plus one PSF evaluation per sample.

### A reader failure stops the run

A reader that cannot read its source raises, and the run stops. The earlier design caught
the exception and scored the footprint as partial coverage, which could publish a product
with a source silently missing. The runner still writes placeholder columns, by an explicit
opt-in, when no ancillary data is staged at all.

### What a granule list means per reader

Readers take the list of granules staged for their source. Stepped and pass readers
index them (hourly ERA5 files by step, hourly FLASHFlux and CLDPIX granules by time span); a
daily-composite reader reads exactly one and raises on more. IGBP concatenates its MODIS
tiles, so the staged tiles should be those under the day's orbits, not the global set. Staging must include the neighboring day's
granules for footprints near 00:00 UTC, or those footprints will match a pass up to a day
away or none; that is a staging concern outside this package.

### Parallelism: a forked process pool over along-track chunks

With `FMATCH_WORKERS` above one, `aggregate_external_variables` cuts the footprints into
four contiguous chunks per worker and runs them on a process pool. On Linux the pool is
forked after every static source is loaded, so the workers read the parent's indexed
sources through copy-on-write memory and the run holds one copy of each static source
regardless of the worker count. A stepped ERA5 source loads at most one step per worker,
and a pass-matched source (FLASHFlux, CLDPIX) loads the granules of its own chunk's time window
per worker, so the CLDPIX working set scales with the worker count: size the workers for
that. Each chunk is a
time-ordered along-track segment, so the per-chunk loop is the single-process loop and the
concatenated result is identical, which a test asserts array by array. Where forking is
unavailable (macOS, Windows) the workers are spawned and each builds its own sources from
the staged paths, at one copy per worker. Threads were not an option: the HDF5 stack is
not thread-safe and the earlier prefetch design had to serialize every read behind a
global lock. The single-process cost measured on a synthetic segment with three sources
is about 2 ms per footprint, two thirds of it in the weigher; on four cores the pool
runs the same segment 3.6 times faster with the parent's 2 GB of sources shared and under
0.7 GB touched per worker. Batch vectorization of the PSF evaluation across footprints is
the next lever if the pool is not enough.

## 9. Where configuration lives, and what to change for common updates

Reader configuration is code, not a configuration file: each reader declares its key, its
source and its variables as class attributes and field tables, and the product definitions
are written to match. Two tests in `tests/unit/test_footprint_matching/test_product.py`
hold the two sides together: `test_external_variables_match_readers` fails when a product
definition is missing a reader variable or has a different dtype, and
`test_reader_instrument_is_recorded_in_long_name` fails when a `long_name` does not carry
the reader's `INSTRUMENT`. Run them after any of the changes below.

### Swap the satellite or instrument a source comes from

For example, moving the VIIRS readers from NOAA-20 to NOAA-21:

1. In each affected reader under `libera_utils/footprint_matching/readers/`, set
   `INSTRUMENT` (for example `INSTRUMENT = "NOAA21"` in `aod.py`, `brdf.py`, `viirs.py`,
   `cldpix.py` and `ssf.py`), and update the module docstring's product and file-naming
   lines.
2. In every `libera_utils/data/product_definitions/fmatch_*.yml`, change the instrument
   in parentheses at the end of each affected `long_name`, for example `(NOAA20)` to
   `(NOAA21)`.
3. If the new product's file layout differs (group paths, variable names, dimension
   order, fill or geolocation conventions), update the reader's field table and
   `_load_granule` or `_load_native`, and add a cut of a real granule under
   `tests/test_data/footprint_matching/real/` with a test that reads it.
4. Stage the new granules under the same directory as before: the directory is named by
   the reader key, not the instrument.

The reader key and the product variable names do not change, so downstream consumers are
unaffected unless step 3 changes a variable.

### Add a variable to an existing reader

1. Add the field to the reader's table: `_CLDPIX_FIELDS` in `cldpix.py`,
   `_SSF_BASE_FIELDS` or `_IMAGER_SSF_SOURCES` in `ssf.py`, `_D3_GROUP_MAP` and
   `VARIABLES` in `viirs.py`, or the reader's `VARIABLES` elsewhere. Give it its
   aggregation, `required_mode` (or `only_modes`), its valid range with the document it
   comes from, and for a mode-aggregated code `n_categories` and an `int16` dtype.
2. Add the product variable `<reader key>_<variable name>` to each product definition
   whose mode carries it, with a `long_name` ending in `(<INSTRUMENT>)` and its units. A
   continuous variable also gets a `<reader key>_<variable name>_standard_deviation`
   companion.
3. Extend the reader's synthetic fixture in `tests/test_data/footprint_matching/fixtures.py`
   and its tests, and record the change under `## [Unreleased]` in `CHANGELOG.md`.

### Add a reader or change which products a reader feeds

1. Write the reader as a subclass of `StaticReader`, `SteppedGridReader` or `PassReader`
   with a unique `READER_KEY`; importing it in `readers/__init__.py` registers it.
2. Add its key to the products it feeds in `FMATCH_MODE_READERS` in `readers/registry.py`.
3. Add its variables to those products' definitions, as above.
4. Stage its granules under `$FMATCH_ANCILLARY_PATH/<reader key>/`; the runner derives the
   directories it reads from the registry.

Renaming a reader key renames its product variables and its staging directory, so it is a
change consumers and the staging pipeline must follow.

## 10. Open questions for the science team

- The time windows in section 2, and whether ERA5 should be interpolated in time.
- Whether a per-sample area weight (section 4) is wanted for the imager sources.
- Whether coarse-grid sources should be interpolated to the boresight (section 5).
- Whether a clear fraction should accompany each geometric-mean variable (section 6).
- Whether to record the matched time offset (section 7).
