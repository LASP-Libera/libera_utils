"""Regenerate the frozen dynamic SPICE kernels in ``tests/test_data/dynamic_kernels``.

These four kernels are test fixtures, not products. They stand in for the output of the
libera_utils SPICE processing step so that geolocation tests exercise *geolocation* rather
than kernel generation, mirroring production: the L1B container is handed kernels from S3 and
never builds them.

Source data
-----------
The three CSVs under ``tests/test_data/tier1_geo`` provided by Jake Fernandez, describing a
simulated JPSS-4 overpass of the Libya-4 calibration site on 2028-01-02:

``JPSS-4_Fixed_Ephemeris_And_Attitude.csv``
    Spacecraft position, velocity and attitude quaternions.
``Libya-4_access003.csv``
    STK access output. ``Command Azimuth (deg)`` is the ideal commanded azimuth that points
    the radiometer at Libya-4.
``libera_el_cross_track_scan_profile_72deg_2021-10.csv``
    A single +/-72 degree cross-track elevation sweep, 5.087 s long at 1 kHz, tiled to fill
    the window.

Coverage
--------
The mechanism CKs cover a 60 s window centred on closest approach to Libya-4 (00:21:32,
ground range 147.2 km), which holds 11.8 complete cross-track sweeps. The full 926 s access
is 182 repeats of the same 5.087 s profile and carries no scan geometry the first sweep does
not, so the long window only inflated the ELSCAN CK (59.4 MB at 926 s versus ~3.8 MB here).
The spacecraft SPK/CK keep their full span: they are small, and the margin keeps ephemeris
valid at the window edges.

Why the encoder correction is inverted here
-------------------------------------------
``create_kernel_from_l1a`` applies the forward encoder correction, because in flight it
receives raw encoder telemetry. Jake's angles are already ideal commanded pointing, so this
script applies the inverse first to synthesise the raw telemetry that would have produced
them. The round trip is exact to machine precision (see ``TestEncoderCorrection`` in
``tests/unit/test_kernel_maker.py``), so the CKs encode Jake's commanded angles -- about the
*measured* axes of rotation read from the production frame kernel.

Determinism
-----------
Kernel *content* is reproducible: NAIF kernels are read from the repo's bundled
``tests/test_data/spice`` rather than downloaded, so output does not depend on what NAIF
published today. Filenames are not, and deliberately so -- the ``R`` field is the creation
timestamp, exactly as in production, so every run yields a new name. That matters because
:class:`~libera_utils.libera_spice.spice_utils.KernelFileCache` keys on basename and returns
any cached copy younger than its timeout without inspecting content: a fixture regenerated
under a pinned name would be silently served stale from a developer's cache. A drift guard
should therefore compare kernel content per product type, not filenames.

Usage
-----
``python tests/fixture_generation/generate_dynamic_kernels.py``

Regenerating these kernels changes the geometry downstream tests see. ``libera_rad`` carries
a byte-identical copy of this fixture set and pins its own time ranges to the old 926 s
window, so a regeneration here needs a matching libera_rad update.
"""

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_DATA = REPO_ROOT / "tests" / "test_data"
TIER1_GEO = TEST_DATA / "tier1_geo"
OUTPUT_DIR = TEST_DATA / "dynamic_kernels"

SC_CSV = TIER1_GEO / "JPSS-4_Fixed_Ephemeris_And_Attitude.csv"
AZ_CSV = TIER1_GEO / "Libya-4_access003.csv"
EL_CSV = TIER1_GEO / "libera_el_cross_track_scan_profile_72deg_2021-10.csv"

#: Mechanism CK coverage: 60 s centred on closest approach to Libya-4 (2028-01-02 00:21:32).
WINDOW_START = pd.Timestamp("2028-01-02 00:21:02")
WINDOW_END = pd.Timestamp("2028-01-02 00:22:02")

AZ_COMMANDED_COLUMN = "Command Azimuth (deg)"
EL_PROFILE_COLUMN = "El Angle [deg]"


def _configure_spice_environment() -> None:
    """Point kernel lookup at the repo's bundled NAIF kernels and load the test LSK.

    Set before importing libera_utils so config resolution picks it up, and so nothing in this
    script reaches the NAIF server.
    """
    os.environ["GENERIC_KERNEL_DIR"] = str(TEST_DATA / "spice")
    os.environ["LEAPSECOND_FILE_ENV"] = str(TEST_DATA / "spice")


def spacecraft_frame() -> tuple[pd.DataFrame, list[datetime]]:
    """Spacecraft ephemeris/attitude dataframe over the CSV's full span."""
    from libera_utils import kernel_maker

    return kernel_maker.create_jpss_kernel_dataframe_from_csv(SC_CSV)


def azimuth_frame() -> tuple[pd.DataFrame, list[datetime]]:
    """Azimuth mechanism dataframe over the fixture window, as synthetic raw encoder telemetry."""
    from curryer import spicetime

    from libera_utils import kernel_maker

    commanded = pd.read_csv(AZ_CSV, index_col=0)
    timetags = pd.to_datetime(commanded.index.values)
    in_window = (timetags >= WINDOW_START) & (timetags <= WINDOW_END)
    if not in_window.any():
        raise ValueError(f"No azimuth samples in {WINDOW_START}..{WINDOW_END}; check {AZ_CSV.name}")

    timetags = timetags[in_window]
    ideal_az = np.deg2rad(commanded[AZ_COMMANDED_COLUMN].to_numpy()[in_window])

    frame = pd.DataFrame(
        {
            kernel_maker.AZ_ENCODER_FIELD: kernel_maker.uncorrect_azimuth(ideal_az),
            "AXIS_SAMPLE_ICIE_ET": spicetime.adapt(timetags, "dt64", "et"),
        }
    )
    return frame, [timetags[0].to_pydatetime(), timetags[-1].to_pydatetime()]


def elevation_frame() -> tuple[pd.DataFrame, list[datetime]]:
    """Elevation mechanism dataframe: the cross-track sweep tiled across the fixture window.

    Tiling is anchored to the start of the *full* azimuth series, not to the window, so the
    sweep is at the same phase it would be in the full-access scenario. The sweep traverses
    +/-72 degrees in 5.087 s, so a fraction of a second of phase error moves the ground point
    by hundreds of km; anchoring here is what makes this fixture a shorter window of Jake's
    scenario rather than a different one.
    """
    from curryer import spicetime

    from libera_utils import kernel_maker

    profile = pd.read_csv(EL_CSV, index_col=0)
    offsets = pd.to_timedelta(profile.index.values, "sec")
    sweep_values = profile[EL_PROFILE_COLUMN].to_numpy()
    sweep_duration = offsets[-1] - offsets[0]

    access_start = pd.to_datetime(pd.read_csv(AZ_CSV, index_col=0).index.values)[0]
    n_sweeps = int(np.ceil((WINDOW_END - access_start) / sweep_duration))
    timetags = []
    values = []
    sweep_start = access_start
    for _ in range(n_sweeps):
        timetags.append(sweep_start + offsets)
        values.append(sweep_values)
        sweep_start = timetags[-1][-1]

    timetags = pd.to_datetime(np.hstack(timetags))
    values = np.hstack(values)
    in_window = (timetags >= WINDOW_START) & (timetags <= WINDOW_END)
    timetags = timetags[in_window]
    ideal_el = np.deg2rad(values[in_window])

    frame = pd.DataFrame(
        {
            kernel_maker.EL_ENCODER_FIELD: kernel_maker.uncorrect_elevation(ideal_el),
            "AXIS_SAMPLE_ICIE_ET": spicetime.adapt(timetags, "dt64", "et"),
        }
    )
    return frame, [timetags[0].to_pydatetime(), timetags[-1].to_pydatetime()]


def _write_kernel(frame: pd.DataFrame, utc_range: list[datetime], identifier: str, output_dir: Path) -> Path:
    """Drive the production ``create_kernel_from_l1a`` path for one kernel type.

    Only the L1A read is stubbed -- the CSV-derived frame stands in for what an L1A granule
    would yield. Everything downstream of that (encoder correction, mechanism quaternions,
    filenaming, the curryer ``make_kernel`` call) runs unmodified.
    """
    from libera_utils import kernel_maker

    with mock.patch(
        "libera_utils.kernel_maker.create_kernel_dataframe_from_l1a",
        side_effect=lambda **_: (frame.copy(), utc_range),
    ):
        return kernel_maker.create_kernel_from_l1a(
            l1a_data=xr.Dataset(),
            kernel_identifier=identifier,
            output_dir=output_dir,
            overwrite=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Where to write the kernels (default: tests/test_data/dynamic_kernels).",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not clear the output directory first. Leaves stale kernels in place.",
    )
    args = parser.parse_args()

    _configure_spice_environment()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.keep_existing:
        for stale in args.output_dir.iterdir():
            if stale.is_file():
                stale.unlink()

    builders = {
        "JPSS-SPK": spacecraft_frame,
        "JPSS-CK": spacecraft_frame,
        "AZROT-CK": azimuth_frame,
        "ELSCAN-CK": elevation_frame,
    }

    for identifier, builder in builders.items():
        frame, utc_range = builder()
        written = _write_kernel(frame, utc_range, identifier, args.output_dir)
        size_mb = Path(str(written)).stat().st_size / 1e6
        print(f"{identifier:>10s}  {len(frame):>7d} samples  {size_mb:7.2f} MB  {Path(str(written)).name}")

    total_mb = sum(f.stat().st_size for f in args.output_dir.iterdir() if f.is_file()) / 1e6
    print(f"\nTotal fixture size: {total_mb:.2f} MB in {args.output_dir}")
    if shutil.which("msopck") is None:
        print("WARNING: msopck was not on PATH; CK generation may have silently produced nothing.")


if __name__ == "__main__":
    main()
