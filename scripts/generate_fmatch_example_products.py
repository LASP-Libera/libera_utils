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

WHAT this script does NOT do
----------------------------
It does not run footprint matching. The numbers inside each file are synthetic
placeholders. They are only guaranteed to:
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
  2. Synthesize a 1-D array for the time coordinate and for every data variable.
     Every FMATCH product is "SSF-style": a flat list of per-footprint records
     indexed by a single time coordinate (``RADIOMETER_TIME`` or ``CAMERA_TIME``),
     so every variable is 1-D along that one dimension. This keeps generation
     fully generic - we introspect the loaded definition rather than hard-coding
     any variable list, so the script stays correct as the YAMLs evolve.
  3. Hand the data dict to ``write_libera_data_product`` which builds the Dataset,
     enforces conformance against the definition, generates the standardized
     Libera filename, and writes the NetCDF file.

Run it with::

    python scripts/generate_fmatch_example_products.py

Output files are written to ``example_outputs/`` at the repository root by default
(override with ``--output-dir``). The Libera filename (product id, version, and
the time span taken from the time coordinate) is generated automatically by
``LiberaDataProductDefinition.generate_data_product_filename``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

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


def _synthesize_variable(var_def: LiberaVariableDefinition) -> np.ndarray:
    """Create a 1-D synthetic data array for a single (non-time) variable.

    The returned array matches the variable's declared dtype exactly (the
    conformance checker requires this) and stays within the range chosen by
    :func:`_value_range_for_variable`.

    Parameters
    ----------
    var_def : LiberaVariableDefinition
        The variable definition to synthesize data for.

    Returns
    -------
    np.ndarray
        A length-``N_FOOTPRINTS`` array of the variable's dtype.
    """
    dtype = np.dtype(var_def.dtype)
    low, high = _value_range_for_variable(var_def)

    if np.issubdtype(dtype, np.integer):
        # Integers (e.g. IGBP surface_type int16 [1, 20], q_flags int64): draw
        # inclusive of both bounds. If no range was declared, _value_range_for_variable
        # returns the float UNITLESS default, so substitute the integer default.
        if var_def.attributes.get("valid_range") is None and var_def.attributes.get("units") not in UNITS_FALLBACK_RANGE:
            low, high = INTEGER_DEFAULT_RANGE
        # randint's high is exclusive, hence +1 to make the declared max attainable.
        values = RNG.integers(int(low), int(high) + 1, size=N_FOOTPRINTS)
    else:
        # Floats: uniform across the chosen range is plenty for an example file.
        values = RNG.uniform(low, high, size=N_FOOTPRINTS)

    # Cast to the exact dtype the product definition declares so conformance
    # checking (which compares dtype strings) passes without an auto-cast warning.
    return values.astype(dtype)


def _synthesize_time_coordinate() -> np.ndarray:
    """Create the monotonically increasing per-footprint time coordinate.

    Every FMATCH product is indexed by a single time coordinate whose dtype is
    ``datetime64[ns]``. ``write_libera_data_product`` requires the time variable
    to be ``datetime64`` and uses its first/last elements to stamp the output
    filename's start/end times, so the array must be sorted ascending.

    Returns
    -------
    np.ndarray
        ``datetime64[ns]`` array of length ``N_FOOTPRINTS`` at 100 Hz cadence.
    """
    offsets = np.arange(N_FOOTPRINTS, dtype="int64")
    return BASE_TIME + offsets * SAMPLE_CADENCE


def build_synthetic_data(mode: OperationalMode) -> tuple[dict[str, np.ndarray], str]:
    """Build the full {name: array} data dict for one operational mode.

    Introspects the loaded product definition so the set of variables is always
    derived from the YAML - never hard-coded here. This is what keeps the script
    correct across all five modes (and any future variable additions).

    Parameters
    ----------
    mode : OperationalMode
        The FMATCH mode whose product definition drives the data shape.

    Returns
    -------
    tuple[dict[str, np.ndarray], str]
        The data dict (keyed by coordinate/variable name) and the name of the
        time coordinate to pass to ``write_libera_data_product``.
    """
    definition = load_fmatch_definition(mode)
    time_variable = fmatch_time_variable(mode)

    data: dict[str, np.ndarray] = {}

    # The single time coordinate (the product definition declares exactly one
    # coordinate, the per-footprint observation time).
    data[time_variable] = _synthesize_time_coordinate()

    # Every data variable is 1-D along that time dimension; synthesize each.
    for var_name, var_def in definition.variables.items():
        data[var_name] = _synthesize_variable(var_def)

    return data, time_variable


def generate_example_product(mode: OperationalMode, output_dir: Path) -> Path:
    """Generate and write one example NetCDF file for a single operational mode.

    Parameters
    ----------
    mode : OperationalMode
        The FMATCH mode to generate an example for.
    output_dir : Path
        Directory to write the file into. Created if it does not exist.

    Returns
    -------
    Path
        Path to the written NetCDF file.
    """
    definition = load_fmatch_definition(mode)
    data, time_variable = build_synthetic_data(mode)

    # Supply the three required *dynamic* product attributes that every FMATCH
    # definition leaves null (algorithm_version, date_created, input_files - see
    # the `attributes:` block in each fmatch_*.yml). These must be provided at
    # write time or conformance checking fails on null attributes.
    dynamic_product_attributes = {
        # Semantic version of the (synthetic) producer. Must match the semver regex.
        "algorithm_version": "0.1.0",
        "date_created": datetime.now(timezone.utc).isoformat(),
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
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing FMATCH example products to: {args.output_dir}")
    for mode in OperationalMode:
        path = generate_example_product(mode, args.output_dir)
        print(f"  [{mode.value:>22}] -> {path.name}")
    print("Done.")


if __name__ == "__main__":
    main()
