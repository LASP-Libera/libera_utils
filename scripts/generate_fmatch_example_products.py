#!/usr/bin/env python
"""Generate one example (synthetic) NetCDF output file per FMATCH product definition.

WHY this script exists
----------------------
The footprint-matching (FMATCH) PSF aggregation engine that will eventually
*compute* the real values in these products is not implemented yet - the
producer functions in ``libera_utils/footprint_matching/product.py`` are still
``NotImplementedError`` stubs (see that module's "Milestone scope" docstring).

In the meantime, downstream consumers (Scene ID, the ADM binning algorithm, the
Camera Cloud Fraction algorithm - see ``instructions/documentation/data_products.md``)
need a concrete, *shape-and-format-correct* file to develop and test against. This
script produces exactly that: for every one of the five FMATCH operational modes
it writes a NetCDF file that fully conforms to the mode's product definition YAML.

Optional L1B geolocation pass-through
-------------------------------------
The radiometer footprint geolocation (``latitude``, ``longitude``), the
Sun-surface-sensor viewing angles (``solar_zenith_angle``,
``viewing_zenith_angle``, ``relative_azimuth_angle``), and the per-footprint
observation time (``RADIOMETER_TIME``) are NOT computed by the footprint-matching
engine - they come straight from the L1B Daily radiometer product (see the
"(a) Geolocation inputs (from L1B Daily)" block in each ``fmatch_*.yml``). So
unlike the truly-synthesized variables, these can be made *real* simply by
copying them out of an actual L1B file. When the caller passes ``--l1b-file``,
this script does exactly that for the three operational modes that are indexed on
radiometer time (FMATCH-CAM, FMATCH-IMAGER-FLASH, FMATCH-IMAGER): their time
coordinate, latitude/longitude, and viewing angles are passed through verbatim
from the L1B file, and every *other* variable is still random.

WHY only those variables (and only those three modes):
  * time/latitude/longitude/viewing-angles are the genuine L1B inputs the FMATCH
    product contract names; everything else is a product of the
    (not-yet-implemented) PSF aggregation / derived-geometry engine, so there is
    no real value to copy. (The derived ``sunglint_angle`` is computed from the
    viewing angles by the engine, so it stays synthetic too.)
  * The two camera-timescale modes (FMATCH-CAM-CAMTIME, FMATCH-IMAGER-CAMTIME)
    are indexed on ``CAMERA_TIME`` (the WFOV camera image cadence), NOT radiometer
    time, so the L1B radiometer timeline does not apply to them. They are left
    fully synthetic on purpose.

WHAT this script does NOT do
----------------------------
It does not run footprint matching. Every variable except the optional L1B
pass-through (time, latitude/longitude, and the viewing angles) is a synthetic
placeholder. The synthetic values are only guaranteed to:
  1. match the dtype declared in the product definition, and
  2. fall inside the variable's ``valid_range`` when one is declared (and inside a
     physically plausible range, chosen from the variable's ``units``, otherwise).
That is all the real consumers need from an *example* file - the structure,
dimensions, dtypes, attributes, and encoding are the contract; the magnitudes are
not. The contract is defined by the YAMLs in
``libera_utils/data/product_definitions/fmatch_*.yml`` and enforced by
``LiberaDataProductDefinition`` (``libera_utils/io/product_definition.py``).

HOW it works
------------
For each ``OperationalMode`` we:
  1. Load + validate the product definition with ``load_fmatch_definition(mode)``.
  2. Build a 1-D array for the time coordinate and for every data variable.
     Every FMATCH product is "SSF-style": a flat list of per-footprint records
     indexed by a single time coordinate (``RADIOMETER_TIME`` or ``CAMERA_TIME``),
     so every variable is 1-D along that one dimension. This keeps generation
     fully generic - we introspect the loaded definition rather than hard-coding
     any variable list, so the script stays correct as the YAMLs evolve. When an
     L1B file was supplied and the mode is radiometer-timed, the pass-through
     variables (time, latitude/longitude, viewing angles) are copied from the L1B
     file instead of synthesized, and every other variable is synthesized to match
     that (real) footprint count.
  3. Hand the data dict to ``write_libera_data_product`` which builds the Dataset,
     enforces conformance against the definition, generates the standardized
     Libera filename, and writes the NetCDF file.

Run it with::

    # Fully synthetic (original behavior):
    python scripts/generate_fmatch_example_products.py

    # Real time, latitude/longitude, and viewing angles from an L1B radiometer
    # file for the radiometer-timed modes (subsampled to 1000 footprints):
    python scripts/generate_fmatch_example_products.py \
        --l1b-file external_data/L1B/LIBERA_L1B_RAD-4CH_...nc \
        --n-footprints 1000

Output files are written to ``example_outputs/`` at the repository root by default
(override with ``--output-dir``). The Libera filename (product id, version, and
the time span taken from the time coordinate) is generated automatically by
``LiberaDataProductDefinition.generate_data_product_filename``.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import xarray as xr

from libera_utils.footprint_matching.product import (
    fmatch_time_variable,
    load_fmatch_definition,
)
from libera_utils.footprint_matching.types import OperationalMode
from libera_utils.io.netcdf import write_libera_data_product
from libera_utils.io.product_definition import LiberaVariableDefinition

# ---------------------------------------------------------------------------
# Tunables for the synthetic data.
# ---------------------------------------------------------------------------
# Number of synthetic footprints per file. Kept small (the real L1B day has
# ~600k footprints) so the example files stay tiny and quick to write/read - an
# example only needs enough records to exercise the format, not a full day.
N_FOOTPRINTS = 1000

# Time anchor for the synthetic footprint timeline. Borrowed from the example
# L1B RAD-4CH input file shipped under external_data/ (which starts
# 2028-02-12T03:39:45) so the example timestamps look like a real observation
# day. The radiometer samples at 100 Hz, hence the 10 ms cadence below. These
# only drive the start/end time embedded in the generated filename; the actual
# instant values are not otherwise meaningful.
BASE_TIME = np.datetime64("2028-02-12T03:39:45.000000000", "ns")
SAMPLE_CADENCE = np.timedelta64(10_000_000, "ns")  # 10 ms == 100 Hz

# A single deterministic RNG so re-running the script reproduces identical files,
# which makes the examples stable to diff and safe to commit/regenerate.
RNG = np.random.default_rng(seed=20280212)

# Mapping of FMATCH product-definition variable name -> source variable name in
# the L1B RAD-4CH file, for the variables we pass through verbatim when an L1B
# file is supplied. These are exactly the "(a) Geolocation inputs (from L1B
# Daily)" block of each fmatch_*.yml: the per-footprint geolocation and the
# Sun-surface-sensor viewing angles, all of which are genuine L1B quantities (as
# opposed to the PSF-aggregated / derived variables the engine would compute).
#
# Geolocation: we use the plain boresight ``Latitude``/``Longitude`` (the
# footprint boresight ground intersect) rather than the ``Terrain_Corrected_*``
# variants, matching the FMATCH definition's "Footprint boresight" long_name.
#
# Viewing angles: the FMATCH solar/viewing zenith and relative azimuth map to the
# L1B "_Surface" angles (geodetic angles at the Earth point), whose units
# (degrees) and ranges line up with the FMATCH definition - in particular L1B
# ``Relative_Azimuth_Surface`` spans [0, 360], matching relative_azimuth_angle's
# valid_range. (Note we do NOT pass through the derived ``sunglint_angle``; that
# is a computed product variable, not an L1B input.)
#
# Every variable listed here is declared float32 in the product definition, which
# is why load_l1b_passthrough can cast them all to float32 generically. The L1B
# time coordinate (``radiometer_time``) is handled separately because it maps to
# the product's time *coordinate* (RADIOMETER_TIME), not a data variable.
L1B_PASSTHROUGH_VARIABLES: dict[str, str] = {
    "latitude": "Latitude",
    "longitude": "Longitude",
    "solar_zenith_angle": "Solar_Zenith_Surface",
    "viewing_zenith_angle": "Viewing_Zenith_Surface",
    "relative_azimuth_angle": "Relative_Azimuth_Surface",
}
# Name of the time coordinate variable inside the L1B file. xarray decodes its
# CF "nanoseconds since 1958-01-01" units into datetime64[ns], which is exactly
# the dtype the FMATCH RADIOMETER_TIME coordinate declares.
L1B_TIME_VARIABLE = "radiometer_time"

# Physically-plausible fallback ranges keyed by the variable's declared ``units``.
# Used ONLY for variables that do not declare a ``valid_range`` in the product
# definition. The goal is realism, not correctness: e.g. a spacecraft altitude in
# meters should look like ~800 km, not ~0.5. Variables that DO declare a
# ``valid_range`` always use that range instead (see _value_range_for_variable).
# Any unit not listed here falls back to UNITLESS_DEFAULT_RANGE / int default.
UNITS_FALLBACK_RANGE: dict[str, tuple[float, float]] = {
    "meters": (7.0e5, 8.5e5),  # spacecraft altitude above the ellipsoid (~LEO)
    "km": (0.0, 20.0),  # e.g. cloud heights
    "hPa": (100.0, 1013.0),  # atmospheric pressure (cloud-top .. surface)
    "m/s": (-20.0, 20.0),  # wind components can be negative
    "K": (200.0, 320.0),  # brightness/temperature-like quantities
    "g/m^2": (0.0, 500.0),  # column water paths
    "percent": (0.0, 100.0),  # concentrations/fractions in percent
    "1": (0.0, 1.0),  # dimensionless fractions / BRDF kernel weights
}
# Last-resort range for a dimensionless float variable with no usable hint.
UNITLESS_DEFAULT_RANGE: tuple[float, float] = (0.0, 1.0)
# Last-resort inclusive range for an integer variable with no declared range.
INTEGER_DEFAULT_RANGE: tuple[int, int] = (0, 10)


def _value_range_for_variable(var_def: LiberaVariableDefinition) -> tuple[float, float]:
    """Pick a (low, high) range to draw synthetic values from for one variable.

    Preference order, most authoritative first:
      1. The variable's declared ``valid_range`` (the product definition's own
         constraint - honoring it guarantees in-range example values).
      2. A physically-plausible range chosen from the variable's ``units`` so the
         example values at least look like the real geophysical quantity.
      3. A generic dimensionless fallback.

    Parameters
    ----------
    var_def : LiberaVariableDefinition
        The loaded variable definition (carries ``attributes`` with optional
        ``valid_range`` and ``units``).

    Returns
    -------
    tuple[float, float]
        Inclusive low/high bounds to sample from.
    """
    attrs = var_def.attributes
    # 1. Honor an explicit valid_range when the definition declares one.
    valid_range = attrs.get("valid_range")
    if valid_range is not None:
        return float(valid_range[0]), float(valid_range[1])

    # 2. Otherwise, fall back to a realistic range based on the declared units.
    units = attrs.get("units")
    if units in UNITS_FALLBACK_RANGE:
        return UNITS_FALLBACK_RANGE[units]

    # 3. No range and no recognized units: use a safe dimensionless default for
    #    floats; integer columns are handled separately in _synthesize_variable.
    return UNITLESS_DEFAULT_RANGE


def _synthesize_variable(var_def: LiberaVariableDefinition, n: int) -> np.ndarray:
    """Create a 1-D synthetic data array for a single (non-time) variable.

    The returned array matches the variable's declared dtype exactly (the
    conformance checker requires this) and stays within the range chosen by
    :func:`_value_range_for_variable`.

    Parameters
    ----------
    var_def : LiberaVariableDefinition
        The variable definition to synthesize data for.
    n : int
        Number of footprints (length of the produced array). Passed in - rather
        than read from a module global - so synthesized variables always match
        the footprint count of the time coordinate, whether that count comes from
        the synthetic default or from a real (possibly subsampled) L1B file.

    Returns
    -------
    np.ndarray
        A length-``n`` array of the variable's dtype.
    """
    dtype = np.dtype(var_def.dtype)
    low, high = _value_range_for_variable(var_def)

    if np.issubdtype(dtype, np.integer):
        # Integers (e.g. IGBP surface_type int16 [1, 20], q_flags int64): draw
        # inclusive of both bounds. If no range was declared, _value_range_for_variable
        # returns the float UNITLESS default, so substitute the integer default.
        if (
            var_def.attributes.get("valid_range") is None
            and var_def.attributes.get("units") not in UNITS_FALLBACK_RANGE
        ):
            low, high = INTEGER_DEFAULT_RANGE
        # randint's high is exclusive, hence +1 to make the declared max attainable.
        values = RNG.integers(int(low), int(high) + 1, size=n)
    else:
        # Floats: uniform across the chosen range is plenty for an example file.
        values = RNG.uniform(low, high, size=n)

    # Cast to the exact dtype the product definition declares so conformance
    # checking (which compares dtype strings) passes without an auto-cast warning.
    return values.astype(dtype)


def _synthesize_time_coordinate(n: int) -> np.ndarray:
    """Create the monotonically increasing per-footprint time coordinate.

    Every FMATCH product is indexed by a single time coordinate whose dtype is
    ``datetime64[ns]``. ``write_libera_data_product`` requires the time variable
    to be ``datetime64`` and uses its first/last elements to stamp the output
    filename's start/end times, so the array must be sorted ascending.

    Parameters
    ----------
    n : int
        Number of footprints (length of the produced array).

    Returns
    -------
    np.ndarray
        ``datetime64[ns]`` array of length ``n`` at 100 Hz cadence.
    """
    offsets = np.arange(n, dtype="int64")
    return BASE_TIME + offsets * SAMPLE_CADENCE


def load_l1b_passthrough(l1b_file: Path, n_footprints: int | None) -> dict[str, np.ndarray]:
    """Read the real per-footprint L1B inputs that FMATCH passes through verbatim.

    Pulls every quantity the FMATCH product contract takes straight from L1B Daily
    - the ``RADIOMETER_TIME`` coordinate plus all variables in
    :data:`L1B_PASSTHROUGH_VARIABLES` (footprint ``latitude``/``longitude`` and
    the solar/viewing zenith and relative-azimuth angles) - so they can be copied
    into the example products instead of being synthesized.

    Two cleanups are applied to make the result a drop-in for the product:

    1. Drop NaN rows. The L1B geolocation/angles are NaN wherever the boresight
       has no valid Earth intersection (e.g. the first samples of the file and any
       gaps). Those rows carry no usable values, so we keep only footprints where
       *every* pass-through variable is finite (logical-AND of the per-variable
       finite masks). In this RAD-4CH file the geolocation and the "_Surface"
       angles share the same gaps, but AND-ing all of them is robust if they ever
       diverge.
    2. Optional even subsample. A full L1B day is ~600k footprints; an example
       file only needs enough records to exercise the format. When
       ``n_footprints`` is given (and smaller than the finite pool), we take an
       evenly-spaced subsample across the whole day - via ``np.linspace`` integer
       indices - so the example still spans the real time/space range rather than
       just the first slice of the orbit.

    Parameters
    ----------
    l1b_file : Path
        Path to the L1B RAD-4CH NetCDF file to read pass-through variables from.
    n_footprints : int | None
        Target number of footprints. ``None`` keeps every finite footprint;
        otherwise the finite pool is evenly subsampled to this many rows (or kept
        whole if it already has fewer than ``n_footprints``).

    Returns
    -------
    dict[str, np.ndarray]
        Mapping keyed by FMATCH variable name: ``"RADIOMETER_TIME"``
        (datetime64[ns]) plus each key of :data:`L1B_PASSTHROUGH_VARIABLES`
        (float32), all the same length.
    """
    # Open with default decoding so the CF-encoded time coordinate
    # ("nanoseconds since 1958-01-01") is decoded into datetime64[ns] for us.
    with xr.open_dataset(l1b_file) as l1b:
        radiometer_time = l1b[L1B_TIME_VARIABLE].values
        # Read every pass-through variable, keyed by its FMATCH (output) name.
        passthrough = {fmatch_name: l1b[l1b_name].values for fmatch_name, l1b_name in L1B_PASSTHROUGH_VARIABLES.items()}

    # (1) Keep only footprints where every pass-through variable is finite. Start
    #     from an all-True mask and AND in each variable's finite mask so a NaN in
    #     ANY variable drops that footprint.
    finite = np.ones(radiometer_time.shape, dtype=bool)
    for values in passthrough.values():
        finite &= np.isfinite(values)

    radiometer_time = radiometer_time[finite]
    passthrough = {name: values[finite] for name, values in passthrough.items()}

    # (2) Optionally subsample evenly across the surviving footprints. linspace
    #     with endpoint=True picks indices spread from the first to the last row,
    #     preserving the full day's time/geographic span at lower resolution.
    if n_footprints is not None and n_footprints < radiometer_time.size:
        indices = np.linspace(0, radiometer_time.size - 1, num=n_footprints, dtype="int64")
        radiometer_time = radiometer_time[indices]
        passthrough = {name: values[indices] for name, values in passthrough.items()}

    # Cast to the exact dtypes the FMATCH definition declares so conformance
    # checking passes without an auto-cast. Every pass-through variable is float32
    # in the definition (see L1B_PASSTHROUGH_VARIABLES) and the decoded time is
    # datetime64[ns]; the casts are belt-and-braces over already-correct dtypes.
    result: dict[str, np.ndarray] = {"RADIOMETER_TIME": radiometer_time.astype("datetime64[ns]")}
    result.update({name: values.astype(np.float32) for name, values in passthrough.items()})
    return result


def build_synthetic_data(
    mode: OperationalMode,
    n_footprints: int | None,
    l1b_passthrough: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], str]:
    """Build the full {name: array} data dict for one operational mode.

    Introspects the loaded product definition so the set of variables is always
    derived from the YAML - never hard-coded here. This is what keeps the script
    correct across all five modes (and any future variable additions).

    When ``l1b_passthrough`` is supplied AND the mode is radiometer-timed (i.e.
    not a CAMERA_TIME mode), the ``RADIOMETER_TIME`` coordinate and every
    pass-through variable (latitude/longitude and the viewing angles) are copied
    from the real L1B data rather than synthesized; every other variable is still
    synthesized, to a length that matches the (real) L1B footprint count.
    Camera-timescale modes ignore the L1B data entirely and stay fully synthetic
    (see the module docstring for why).

    Parameters
    ----------
    mode : OperationalMode
        The FMATCH mode whose product definition drives the data shape.
    n_footprints : int | None
        Number of synthetic footprints to generate. Used for the synthetic path
        (no L1B, or camera-timescale modes); ignored when L1B pass-through drives
        the footprint count. ``None`` falls back to ``N_FOOTPRINTS``.
    l1b_passthrough : dict[str, np.ndarray] | None
        Real pass-through arrays from :func:`load_l1b_passthrough`, or ``None``
        for the original fully-synthetic behavior.

    Returns
    -------
    tuple[dict[str, np.ndarray], str]
        The data dict (keyed by coordinate/variable name) and the name of the
        time coordinate to pass to ``write_libera_data_product``.
    """
    definition = load_fmatch_definition(mode)
    time_variable = fmatch_time_variable(mode)

    # Pass-through applies only when we have L1B data AND the mode is indexed on
    # radiometer time (RADIOMETER_TIME). Camera-timescale modes use CAMERA_TIME,
    # which the L1B radiometer timeline does not describe, so they stay synthetic.
    use_l1b = l1b_passthrough is not None and time_variable == "RADIOMETER_TIME"

    data: dict[str, np.ndarray] = {}

    if use_l1b:
        # Copy every real pass-through variable (time coordinate + lat/lon +
        # viewing angles) straight in. The footprint count for this file is
        # therefore the real (post-filter, post-subsample) L1B length, and every
        # synthesized variable below is generated to match it.
        data.update(l1b_passthrough)
        n = data[time_variable].size
    else:
        # Original synthetic path: build the single time coordinate from the
        # 100 Hz cadence anchor. Fall back to N_FOOTPRINTS when no count is given.
        n = N_FOOTPRINTS if n_footprints is None else n_footprints
        data[time_variable] = _synthesize_time_coordinate(n)

    # Every data variable is 1-D along the time dimension; synthesize each, but
    # skip any variable already populated from the L1B pass-through above.
    for var_name, var_def in definition.variables.items():
        if var_name in data:
            continue
        data[var_name] = _synthesize_variable(var_def, n)

    return data, time_variable


def generate_example_product(
    mode: OperationalMode,
    output_dir: Path,
    n_footprints: int | None,
    l1b_passthrough: dict[str, np.ndarray] | None = None,
) -> Path:
    """Generate and write one example NetCDF file for a single operational mode.

    Parameters
    ----------
    mode : OperationalMode
        The FMATCH mode to generate an example for.
    output_dir : Path
        Directory to write the file into. Created if it does not exist.
    n_footprints : int | None
        Number of synthetic footprints; ``None`` falls back to ``N_FOOTPRINTS``.
        Ignored for radiometer-timed modes when ``l1b_passthrough`` is supplied
        (the real L1B footprint count is used instead).
    l1b_passthrough : dict[str, np.ndarray] | None
        Real pass-through arrays from :func:`load_l1b_passthrough`, or ``None`` for
        fully synthetic output.

    Returns
    -------
    Path
        Path to the written NetCDF file.
    """
    definition = load_fmatch_definition(mode)
    data, time_variable = build_synthetic_data(mode, n_footprints, l1b_passthrough)

    # Supply the three required *dynamic* product attributes that every FMATCH
    # definition leaves null (algorithm_version, date_created, input_files - see
    # the `attributes:` block in each fmatch_*.yml). These must be provided at
    # write time or conformance checking fails on null attributes.
    dynamic_product_attributes = {
        # Semantic version of the (synthetic) producer. Must match the semver regex.
        "algorithm_version": "0.1.0",
        "date_created": datetime.now(UTC).isoformat(),
        # Provenance string. Real files list the L1B + ancillary inputs; this is
        # an example, so we mark it as synthetic to avoid implying real inputs.
        "input_files": "SYNTHETIC EXAMPLE - no real input files were used",
    }

    written = write_libera_data_product(
        definition,
        data,
        output_path=output_dir,
        time_variable=time_variable,
        dynamic_product_attributes=dynamic_product_attributes,
    )
    return Path(str(written.path))


def main() -> None:
    """Generate one example NetCDF per FMATCH product definition."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output-dir",
        type=Path,
        # Default to <repo-root>/example_outputs. This file lives in
        # <repo-root>/scripts/, so the repo root is its parent's parent.
        default=Path(__file__).resolve().parent.parent / "example_outputs",
        help="Directory to write the example NetCDF files into (created if absent).",
    )
    parser.add_argument(
        "--l1b-file",
        type=Path,
        default=None,
        help=(
            "Optional L1B RAD-4CH NetCDF file. When given, the radiometer-timed "
            "products (FMATCH-CAM, FMATCH-IMAGER-FLASH, FMATCH-IMAGER) copy their "
            "RADIOMETER_TIME, latitude/longitude, and solar/viewing zenith and "
            "relative-azimuth angles from this file; all other variables stay "
            "synthetic, and the camera-timescale products stay fully synthetic. "
            "When omitted, every product is fully synthetic (original behavior)."
        ),
    )
    parser.add_argument(
        "--n-footprints",
        type=int,
        default=None,
        help=(
            "Number of footprints per file. Without --l1b-file this is the synthetic "
            f"record count (default {N_FOOTPRINTS}). With --l1b-file it evenly "
            "subsamples the real footprints for the radiometer-timed products "
            "(default: keep every valid L1B footprint)."
        ),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load the real pass-through arrays once (if requested) and reuse them for
    # every radiometer-timed mode, so we only open/filter the large L1B file once.
    l1b_passthrough: dict[str, np.ndarray] | None = None
    if args.l1b_file is not None:
        print(f"Reading L1B pass-through inputs from: {args.l1b_file}")
        l1b_passthrough = load_l1b_passthrough(args.l1b_file, args.n_footprints)
        print(f"  Using {l1b_passthrough['RADIOMETER_TIME'].size} valid L1B footprints for radiometer-timed modes.")

    print(f"Writing FMATCH example products to: {args.output_dir}")
    for mode in OperationalMode:
        path = generate_example_product(mode, args.output_dir, args.n_footprints, l1b_passthrough)
        # Flag which geolocation source each mode used so the output is self-explaining.
        # Pass-through only happens for radiometer-timed modes when an L1B file was given.
        used_l1b = l1b_passthrough is not None and fmatch_time_variable(mode) == "RADIOMETER_TIME"
        source = "L1B lat/lon/angles" if used_l1b else "synthetic"
        print(f"  [{mode.value:>22}] ({source:>16}) -> {path.name}")
    print("Done.")


if __name__ == "__main__":
    main()
