"""Geolocation validated against the CERES public data product.

Validation provenance
---------------------
This is the test that satisfied the **Tier-0 geolocation** internal validation. The "Tier-0"
label is historical and is being retired (LIBSDC-703); the test remains the record of that
validation and its tolerances must not be loosened without re-running it.

What it establishes: geolocating from checked-in NOAA-20 kernels reproduces the lon/lat in the
CERES public product to a pinned error distribution. CERES is the only geolocation reference
available to this repo that is neither simulated nor derived from Libera's own geometry, which is
why the NOAA-20 configuration is retained even as JPSS-4 becomes the operational one.

Source data
-----------
Expected lon/lat are a 2-minute extract from
``CER_BDS_NOAA20-FM6_Edition1_100111.20210409.hdf``, saved as CSV. The kernels are the checked-in
outputs of the Tier-0 kernel creation tests rather than being regenerated here, so a failure
points at geolocation rather than at kernel generation.
"""

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from curryer import meta
from curryer import spicierpy as sp
from curryer.compute import constants, spatial

from libera_utils.config import config

# Mark test module as integration tests
pytestmark = pytest.mark.integration


#: SPICE bodies the geolocation frame chain needs ephemeris for: the spacecraft, the three
#: structural elements, the radiometer, and the WFOV camera. Asserted rather than a file count so
#: that a kernel serving the wrong body -- or a leftover from a superseded frame layout -- fails
#: here instead of silently standing in for the body it happens to share an ID with.
EXPECTED_SPK_BODIES = {-143013, -143013001, -143013002, -143013003, -143013010, -143013011}


@pytest.fixture
def noaa20_kernels(test_data_path):
    """The NOAA-20 test kernels."""
    data_dir = test_data_path / "tier0_geo"
    spk_files = sorted(data_dir.glob("*.bsp"))
    ck_files = sorted(data_dir.glob("*.bc"))

    provided_bodies = {int(body) for spk in spk_files for body in sp.spkobj(str(spk))}
    assert provided_bodies == EXPECTED_SPK_BODIES, (
        f"tier0_geo SPKs provide {sorted(provided_bodies)}, expected {sorted(EXPECTED_SPK_BODIES)}"
    )
    return spk_files + ck_files


@pytest.fixture
def noaa20_expected(test_data_path):
    """Load the NOAA-20 lon/lat test data."""
    input_file = test_data_path / "tier0_geo" / "CER_BDS_NOAA20-FM6_Edition1_100111.20210409.lonlat.2min.csv"
    input_data = pd.read_csv(input_file, index_col=0)
    return input_data


def test_geolocate_noaa20_against_ceres(
    noaa20_environment, curryer_lsk, noaa20_kernels, noaa20_expected, spice_test_data_path, test_data_path
):
    """Geolocated lon/lat from the NOAA-20 kernels match the CERES product."""
    # Load meta kernel details.
    mkrn = meta.MetaKernel.from_json(
        config.get("LIBERA_KERNEL_META"),
        relative=True,
        sds_dir=spice_test_data_path,
    )

    # Use the expected data times as the geolocation times.
    noaa20_expected = noaa20_expected.set_index("UGPS")
    ugps_times = noaa20_expected.index.values

    # This validates the ellipsoid-intersection math against CERES under nominal Libera geometry.
    # No frame-kernel substitution is needed: the NOAA-20 configuration selected by
    # noaa20_environment carries no measured misalignments at all (its FK defines none), so it is
    # already the nominal geometry this comparison requires. The measured misalignments are
    # validated separately in test_los_alignment.py.
    with sp.ext.load_kernel([mkrn.sds_kernels, mkrn.mission_kernels, noaa20_kernels]):
        # Geolocate to the ellipsoid.
        ellips_lla_df, sc_xyz_df, ellips_qf_ds = spatial.compute_ellipsoid_intersection(
            ugps_times,
            sp.obj.Body("LIBERA_RAD", frame=True),
            give_geodetic_output=True,
            give_lat_lon_in_degrees=True,
        )

        # Sanity checks.
        assert noaa20_expected.shape[0] == 7092
        assert noaa20_expected.shape[0] == ellips_lla_df.shape[0]
        npt.assert_equal(noaa20_expected.index.values, ellips_lla_df.index.values)

        # Will be non-zero if kernels are missing. Routine doesn't throw errors.
        assert (ellips_qf_ds == 0).all()
        assert np.isfinite(ellips_lla_df["lon"].values).sum() == 7092
        assert np.isfinite(ellips_lla_df["lat"].values).sum() == 7092

        # Check that they are within a reasonable area on the globe.
        lon2 = ellips_lla_df["lon"].values
        lat2 = ellips_lla_df["lat"].values

        # Longitude spans about 170.5 East to 135.9 West (crossing date line)
        npt.assert_allclose(lon2[lon2 >= 0].min(), 170.5586222055922)  # Degrees
        npt.assert_allclose(lon2[lon2 >= 0].max(), 179.97238438946204)
        npt.assert_allclose(lon2[lon2 < 0].min(), -179.9975928633903)
        npt.assert_allclose(lon2[lon2 < 0].max(), -135.8650860432574)

        # Latitude spans 13.9 South to 27.6 South.
        npt.assert_allclose(lat2.min(), -27.56581994224719)  # Degrees
        npt.assert_allclose(lat2.max(), -13.909055339072571)

        # Verify scan-like behavior. Every 197 samples the longitude stepping
        # changes directions.
        dlon = lon2
        dlon[dlon < 0] = (360 + dlon)[dlon < 0]  # Range 0 to 360.
        dlon = dlon[1:] - dlon[:-1]
        (lon_flip,) = np.where((dlon > 0)[:-1] != (dlon > 0)[1:])
        npt.assert_equal(lon_flip[1:] - lon_flip[:-1], 197)

        # Use Haversine formula to compute error distance on a sphere.
        lon1 = np.deg2rad(noaa20_expected["LON"].values)
        lat1 = np.deg2rad(noaa20_expected["LAT"].values)
        lon2 = np.deg2rad(lon2)
        lat2 = np.deg2rad(lat2)

        error = (
            2
            * constants.WGS84_SEMI_MAJOR_AXIS_KM
            * np.arcsin(
                np.sqrt((1 - np.cos(lat2 - lat1) + np.cos(lat1) * np.cos(lat2) * (1 - np.cos(lon2 - lon1))) / 2)
            )
        )
        min_error, max_error = np.nanmin(error), np.nanmax(error)
        med_error, mean_error = np.nanmedian(error), np.nanmean(error)

        # KM to deg (at equator!) is 1/111.32.
        print(f"\nEl[all] {'Min Error':>16s}: {min_error / 111.32: 12.6f} (deg), {min_error:.3f} (km)")
        print(f"El[all] {'Median Error':>16s}: {med_error / 111.32: 12.6f} (deg), {med_error:.3f} (km)")
        print(f"El[all] {'Mean Error':>16s}: {mean_error / 111.32: 12.6f} (deg), {mean_error:.3f} (km)")
        print(f"El[all] {'Max Error':>16s}: {max_error / 111.32: 12.6f} (deg), {max_error:.3f} (km)")

        npt.assert_allclose(min_error, 0.02131926023473517)  # KM
        npt.assert_allclose(med_error, 0.10573680728898222)  # KM
        npt.assert_allclose(mean_error, 0.17991876488800304)  # KM
        npt.assert_allclose(max_error, 3.347837260941865)  # KM

        # For reference, the median inter-sample distance (resolution) is ~14 KM!!!
        lon2 = lon1[1:]
        lat2 = lat1[1:]
        lon1 = lon1[:-1]
        lat1 = lat1[:-1]

        error = (
            2
            * constants.WGS84_SEMI_MAJOR_AXIS_KM
            * np.arcsin(
                np.sqrt((1 - np.cos(lat2 - lat1) + np.cos(lat1) * np.cos(lat2) * (1 - np.cos(lon2 - lon1))) / 2)
            )
        )
        npt.assert_allclose(np.median(error), 13.612881278739977)  # KM
