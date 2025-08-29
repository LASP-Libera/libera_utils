"""Tier-0 geolocation test case.

Goal: Geolocation from Tier 0 SPICE and validate against CERES. Reusable
code in libera_utils designed for reuse by L1bCam and L1bRad.

Preparation:
    - Depends on the SPICE kernels that were created during the tier-0 kernels
    test. It does not recreate those results, instead it uses files that were
    checked into the repo.
    - The expected longitude and latitude values were taken from the CERES
    public data product: CER_BDS_NOAA20-FM6_Edition1_100111.20210409.hdf
        - For the unit test, a 2-minute chunk was extracted and saved to a CSV,
        checked into the repo.

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


@pytest.fixture
def noaa20_kernels(test_data_path):
    """The NOAA-20 test kernels."""
    data_dir = test_data_path / "tier0_geo"
    kernels = sorted(list(data_dir.glob("*.bsp")) + list(data_dir.glob("*.bc")))
    assert len(kernels) == 12
    return kernels


@pytest.fixture
def noaa20_expected(test_data_path):
    """Load the NOAA-20 lon/lat test data."""
    input_file = test_data_path / "tier0_geo" / "CER_BDS_NOAA20-FM6_Edition1_100111.20210409.lonlat.2min.csv"
    input_data = pd.read_csv(input_file, index_col=0)
    return input_data


def test_geolocate_noaa20(curryer_lsk, noaa20_kernels, noaa20_expected, spice_test_data_path):
    """Tier-0 test for geolocating points from kernels."""
    # Load meta kernel details.
    mkrn = meta.MetaKernel.from_json(
        config.get("LIBERA_KERNEL_META"),
        relative=True,
        sds_dir=spice_test_data_path,
    )

    # Use the expected data times as the geolocation times.
    noaa20_expected = noaa20_expected.set_index("UGPS")
    ugps_times = noaa20_expected.index.values

    with sp.ext.load_kernel([mkrn.sds_kernels, mkrn.mission_kernels, noaa20_kernels]):
        # Geolocate to the ellipsoid.
        ellips_lla_df, sc_xyz_df, ellips_qf_ds = spatial.instrument_intersect_ellipsoid(
            ugps_times, sp.obj.Body("LIBERA_SW_RAD", frame=True), geodetic=True, degrees=True
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
