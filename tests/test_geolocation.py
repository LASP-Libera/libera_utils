"""Tests for geolocation module"""
# Installed
import pytest
import numpy as np
# Local
from libera_sdp import geolocation as geo
from libera_sdp import spiceutil

np.set_printoptions(precision=64)


@pytest.mark.parametrize(
    ('from_frame', 'to_frame', 'et', 'expected_rotation'),
    [
        ('J2000', 'ECLIPJ2000',
         99,  # Irrelevant ET
         [[1., 0., 0.],
          [0., 0.9174820620691818, 0.3977771559319137],
          [0., -0.3977771559319137, 0.9174820620691818]]),
        ('EARTH_FIXED', 'JPSS_SPACECRAFT',  # Dynamic based on JPSS CK
         [671202069.186, 671202969.186],  # 2021-04-09T01:00:00, 2021-04-09T01:15:00
         [[[0.49842718476170034, 0.41282178408346193, -0.7623309754146643],
           [-0.12388779000147064, -0.8364013692590833, -0.533933109097087],
           [-0.858033890344127, 0.3605702762121111, -0.36574160131607175]],
          [[0.9533045535986435, 0.06794106186872627, -0.29426933275528566],
           [0.05675578990562619, -0.9973095088366656, -0.04639530036714099],
           [-0.29662974968852934, 0.02752736268004445, -0.9545957447546064]]]
         ),
        ('EARTH_FIXED', 'ITRF93',  # Aliased
         99,  # Irrelevant ET
         np.eye(3)),
        ('JPSS_SPACECRAFT', 'LIBERA_MOUNT_PLATE',  # Fixed offset in FK
         99,  # Irrelevant ET
         np.eye(3)),
    ]
)
def test_vec_pxform(from_frame, to_frame, et, expected_rotation,
                    furnish_testing_kernels):
    """Test vectorized form of the pxform SPICE function
    Note: This test will need to be updated when we update the reference frame offsets in the frame kernel
    and when we change the JPSS test kernels."""
    rotation = geo.vec_pxform(from_frame, to_frame, et)
    assert np.allclose(rotation, expected_rotation, atol=1e-16)


@pytest.mark.parametrize(
    ('target', 'et', 'frame', 'observer', 'expected_spoint', 'expected_trgepc', 'expected_obs_tgt_vec'),
    [
        ('EARTH', 671202069.186, 'EARTH_FIXED', 'JPSS',  # 2021-04-09T01:00:00
         [-5924.69221986335, -258.12245062133303, 2339.8978174477925],
         671202069.1832341,
         [769.9013027778929, 33.542469994191606, -306.1140491597953]
         ),
        ('EARTH', [671202069.186, 671202969.186], 'EARTH_FIXED', 'JPSS',  # 2021-04-09T01:00:00, 2021-04-09T01:15:00
         [[-5924.69221986335, -258.12245062133303, 2339.8978174477925],
          [-1670.8876960739055, 868.4422364857616, 6073.382565056971]],
         [6.712020691832341e+08, 6.712029691832025e+08],
         [[769.9013027778929, 33.542469994191606, -306.1140491597953],
          [219.03353146673703, -113.84246253012918, -801.5139623581053]]
         ),
    ]
)
def test_vec_subpnt(furnish_testing_kernels,
                    target, et, frame, observer,
                    expected_spoint, expected_trgepc, expected_obs_tgt_vec):
    """Test vectorized form of the subpnt SPICE function"""
    method = 'NEAR POINT/ELLIPSOID'
    abcorr = 'LT+S'
    spoint, trgepc, obs_tgt_vec = geo.vec_subpnt(method, target, et, frame, abcorr, observer)
    print(spoint, trgepc, obs_tgt_vec)
    assert np.allclose(spoint, expected_spoint, atol=1e-16)
    assert np.allclose(trgepc, expected_trgepc, atol=1e-16)
    assert np.allclose(obs_tgt_vec, expected_obs_tgt_vec, atol=1e-16)


@pytest.mark.parametrize(
    ('target', 'et', 'frame', 'observer', 'expected_spoint', 'expected_trgepc', 'expected_srfvec'),
    [
        ('EARTH', 671202069.186, 'EARTH_FIXED', 'JPSS',  # 2021-04-09T01:00:00
         [-6118.725079952674, 1592.5369757804303, 837.2167778120764],
         671202069.1770792,
         [575.8648253277091, 1884.0588695988738, -1808.614922220075]
         ),
        ('EARTH', [671202069.186, 671202969.186], 'EARTH_FIXED', 'JPSS',  # 2021-04-09T01:00:00, 2021-04-09T01:15:00
         [[-6118.725079952674, 1592.5369757804303, 837.2167778120764],
          [-6001.391366604171, 1989.360418552823, 837.6422527689641]],
         [6.712020691770792e+08, 6.712029691614056e+08],
         [[575.8648253277091, 1884.0588695988738, -1808.614922220075],
          [-4111.264056392862, 1007.0774276852109, -6037.176700204616]]
         ),
    ]
)
def test_vec_subslr(furnish_testing_kernels,
                    target, et, frame, observer,
                    expected_spoint, expected_trgepc, expected_srfvec):
    """Test vectorized form of the subslr SPICE function"""
    method = 'NEAR POINT/ELLIPSOID'
    abcorr = 'LT+S'
    spoint, trgepc, srfvec = geo.vec_subslr(method, target, et, frame, abcorr, observer)
    assert np.allclose(spoint, expected_spoint, atol=1e-16)
    assert np.allclose(trgepc, expected_trgepc, atol=1e-16)
    assert np.allclose(srfvec, expected_srfvec, atol=1e-16)


def test_get_earth_radii(furnish_test_pck):
    """Test function that retrieves Earth radii values from the SPICE variable pool"""
    re, rp, flat = geo.get_earth_radii()
    assert re == 6378.1366
    assert rp == 6356.7519
    assert flat == 0.0033528131084554717


@pytest.mark.parametrize(
    ('target', 'et', 'frame', 'observer', 'normalize', 'x_expected', 'v_expected', 'lt_expected'),
    [
        (spiceutil.SpiceBody.JPSS,
         671202069.186,  # 2021-04-09T01:00:00
         spiceutil.SpiceFrame.EARTH_FIXED, spiceutil.SpiceBody.EARTH,
         True,
         [-6694.613946694387, -291.66107948757536, 2646.041752848919],
         [2.6289067240007404, 1.8164855012053074, 6.82444215353862],
         0.02403154161898258),
        (spiceutil.SpiceBody.JPSS,
         [671202069.186, 671202969.186],  # 2021-04-09T01:00:00, 2021-04-09T01:15:00
         spiceutil.SpiceFrame.EARTH_FIXED, spiceutil.SpiceBody.EARTH,
         True,
         [[-6694.613946694387, -291.66107948757536, 2646.041752848919],
          [-1889.9076314428785, 982.2901645209304, 6874.926264639398]],
         [[2.6289067240007404, 1.8164855012053074, 6.82444215353862],
          [7.24627132024457, 0.7050369062423633, 1.8874721285946148]],
         [0.02403154161898258, 0.02400763844180673])
    ]
)
def test_target_position(furnish_testing_kernels,
                         target, et, frame, observer, normalize,
                         x_expected, v_expected, lt_expected):
    """Test the function that calculates object position and velocity relative to an observer"""
    x, v, lt = geo.target_position(target, et, frame, observer)
    print(x, v, lt)
    assert isinstance(x, np.ndarray)
    assert isinstance(v, np.ndarray)
    assert x.dtype == v.dtype == np.float64
    assert np.allclose(x, x_expected, atol=1e-16)
    assert np.allclose(v, v_expected, atol=1e-16)
    assert np.allclose(lt, lt_expected, atol=1e-16)


@pytest.mark.parametrize(
    ('target', 'et', 'frame', 'observer',
     'lon_expected', 'lat_expected', 'spoint_alt_expected', 'observer_alt_expected'),
    [
        (spiceutil.SpiceBody.EARTH,
         671202069.186,  # 2021-04-09T01:00:00
         spiceutil.SpiceFrame.EARTH_FIXED, spiceutil.SpiceBody.JPSS,
         182.49460063928385, 21.66436881194109, -0.0, 829.2336304511722),
        (spiceutil.SpiceBody.EARTH,
         [671202069.186, 671202969.186],  # 2021-04-09T01:00:00, 2021-04-09T01:15:00
         spiceutil.SpiceFrame.EARTH_FIXED, spiceutil.SpiceBody.JPSS,
         [182.49460063928385, 152.536589920107],
         [21.66436881194109, 72.88225824594679],
         [-0., -0.],
         [829.2336304511722, 838.6914717316978]),
    ]
)
def test_sub_observer_point(furnish_testing_kernels,
                            target, et, frame, observer,
                            lon_expected, lat_expected, spoint_alt_expected, observer_alt_expected):
    """Test function that calculates the point on a planetary body directly under an observing object"""
    spoint, observer_alt = geo.sub_observer_point(target, et, frame, observer)
    # We convert to lon lat because the original test values were constructed that way
    lon, lat, spoint_alt = geo.cartesian_to_planetographic(spoint)
    print(lon, lat, spoint_alt, observer_alt)
    assert np.allclose(lon, lon_expected, atol=1e-16)
    assert np.allclose(lat, lat_expected, atol=1e-16)
    assert np.allclose(spoint_alt, spoint_alt_expected, atol=1e-16)
    assert np.allclose(observer_alt, observer_alt_expected, atol=1e-16)


@pytest.mark.parametrize(
    ('target', 'et', 'frame', 'observer',
     'lon_expected', 'lat_expected', 'alt_expected'),
    [
        (spiceutil.SpiceBody.EARTH,
         671202069.186,  # 2021-04-09T01:00:00
         spiceutil.SpiceFrame.EARTH_FIXED, spiceutil.SpiceBody.JPSS,
         165.41115061997294, 7.593299184359046, 0.0),
        (spiceutil.SpiceBody.EARTH,
         [671202069.186, 671202969.186],  # 2021-04-09T01:00:00, 2021-04-09T01:15:00
         spiceutil.SpiceFrame.EARTH_FIXED, spiceutil.SpiceBody.JPSS,
         [165.41115061997294, 161.660508386243],
         [7.593299184359046, 7.5971804272542585],
         [0., -0.]),
    ]
)
def test_sub_solar_point(furnish_testing_kernels,
                         target, et, frame, observer,
                         lon_expected, lat_expected, alt_expected):
    """Test function that calculates the position of the sub solar point"""
    spoint, trgepc, srfvec = geo.sub_solar_point(target, et, frame, observer)
    # We convert to lon lat because the original test values were constructed that way
    lon, lat, spoint_alt = geo.cartesian_to_planetographic(spoint)
    print(lon, lat, spoint_alt)
    assert np.allclose(lon, lon_expected, atol=1e-16)
    assert np.allclose(lat, lat_expected, atol=1e-16)
    assert np.allclose(spoint_alt, alt_expected, atol=1e-16)


@pytest.mark.parametrize(
    ('from_frame', 'to_frame', 'et', 'position', 'normalize', 'expectation'),
    [
        (spiceutil.SpiceFrame.J2000, spiceutil.SpiceFrame.J2000, 671202069.186, np.array([10., 20., 30.]), True,
         np.array([0.2672612419124244, 0.5345224838248488, 0.8017837257372731])),
        (spiceutil.SpiceFrame.J2000, spiceutil.SpiceFrame.J2000, [671202069.186, 671202069.186],
         np.array([[10., 20., 30.],
                   [5., 5., 5.]]),
         True,
         np.array([[0.2672612419124244, 0.5345224838248488, 0.8017837257372731],
                   [0.5773502691896257, 0.5773502691896257, 0.5773502691896257]])),
        (spiceutil.SpiceFrame.ITRF93, spiceutil.SpiceFrame.J2000, [671202069.186, 671202969.186],
         np.array([[10., 20., 30.],
                   [5., 5., 5.]]),
         False,
         np.array([[2.2683090580865497, -22.25119822761605, 29.99564887533421],
                   [-1.0971633505613465, -6.983777245285373, 5.002308264233546]])),
    ]
)
def test_frame_transform(furnish_testing_kernels,
                         from_frame, to_frame, et, position, normalize,
                         expectation):
    """Test function that transforms vectors between reference frames (possibly dynamic frames)"""
    result = geo.frame_transform(from_frame, to_frame, et, position, normalize)
    assert np.allclose(result, expectation, atol=1e-16)


@pytest.mark.parametrize(
    ('v1', 'v2', 'degrees', 'expected_angle'),
    [
        (np.array([1, 0, 0]), np.array([0, 1, 0]), True, 90.0),
        (np.array([1, 0, 0]), np.array([0, 1, 0]), False, np.pi/2),
        (np.array([1, 1, 0]), np.array([0, 1, 0]), True, 45.0),
        (np.array([[1, 1, 0], [1, 0, 1]]), np.array([[0, 1, 0], [0, 1, 0]]), True, [45.0, 90.0]),
    ]
)
def test_angle_between(v1, v2, degrees, expected_angle):
    """Test function that calculates the angle between two (sets of) vectors"""
    angle = geo.angle_between(v1, v2, degrees)
    assert np.allclose(angle, expected_angle, atol=1e-16)


@pytest.mark.parametrize(
    ('v', 'degrees', 'expected_lon', 'expected_lat', 'expected_alt'),
    [
        (np.array([0, 0, 6356.7519]), True, 0.0, 90.0, 0.0),
        (np.array([0, 0, -6356.7519]), True, 0.0, -90.0, 0.0),
        (np.array([6378.1366, 0, 0]), False, 0.0, 0.0, 0.0),
        (np.array([0, 6378.1366, 0]), False, np.pi/2, 0.0, 0.0),
        (np.array([0, -6378.1366, 0]), False, 3*np.pi/2, 0.0, 0.0),
    ]
)
def test_cartesian_to_planetographic(furnish_testing_kernels,
                                     v, degrees, expected_lon, expected_lat, expected_alt):
    """Test function that converts cartesian coordinates to planetographic coordinates"""
    lon, lat, alt = geo.cartesian_to_planetographic(v, degrees=degrees)
    print(lon, lat, alt)
    assert np.allclose(lon, expected_lon, atol=1e-16)
    assert np.allclose(lat, expected_lat, atol=1e-16)
    assert np.allclose(alt, expected_alt, atol=1e-16)


@pytest.mark.parametrize(
    ('sc_location', 'look_vector', 'look_frame', 'et', 'expected_pnear', 'expected_dist'),
    [
        (np.array([0, 0, 2*6356.7519]), np.array([0, 0, -1]), spiceutil.SpiceFrame.ITRF93, None,
         np.array([0, 0, 6356.7519]), 0.0),
        (np.array([[0, 0, 2*6356.7519], [2*6378.1366, 0, 0]]), np.array([[0, 0, -1], [-1, 0, 0]]),
         spiceutil.SpiceFrame.ITRF93, None,
         np.array([[0, 0, 6356.7519], [6378.1366, 0, 0]]), [0.0, 0.0])
    ]
)
def test_surface_intercept_point(furnish_testing_kernels,
                                 sc_location, look_vector, look_frame, et, expected_pnear, expected_dist):
    """Test function that finds the near point to the Earth ellipsoid"""
    if et:
        pnear, dist = geo.surface_intercept_point(sc_location, look_vector, look_frame, et=et)
    else:
        pnear, dist = geo.surface_intercept_point(sc_location, look_vector, look_frame)

    print(pnear, dist)
    assert np.allclose(pnear, expected_pnear, atol=1e-16)
    assert np.allclose(dist, expected_dist, atol=1e-16)
