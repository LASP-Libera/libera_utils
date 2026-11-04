"""Unit tests for FMATCH product assembly and writing (product.py).

Covers the assembly entry points (``assemble_fmatch_dataset`` / ``write_fmatch_product``) for both
timescales: the L1B pass-through columns, ``cloud_fraction_camera``, their failure paths, the integer
fill, the derived sun glint angle, the footprint geometry the PSF weighers read, the coverage QA bits,
assembly with the real ERA5 surface reader, and the small helpers whose edge cases decide a product
value (longitude wrapping, the viewing zenith angle read).
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from libera_utils.constants import DataProductIdentifier
from libera_utils.footprint_matching import product
from libera_utils.footprint_matching.product import (
    _RADIOMETER_L1B_VARIABLES,
    CAMTIME_BBOX_TRUNCATED_BIT,
    CAMTIME_BBOX_WRAPS_DATELINE_BIT,
    CAMTIME_PSEUDOFOOTPRINT_VARIABLES,
    L1B_PASSTHROUGH_VARIABLES,
    MissingCamtimeVariablesError,
    _assemble_camtime_dataset,
    _assemble_radiometer_dataset,
    _fill_value_for,
    _footprint_geometry,
    _internal_name_view,
    _merge_coverage_qa,
    _normalize_longitude,
    assemble_fmatch_dataset,
    build_radiometer_footprints,
    compute_derived_viewing_geometry,
    fmatch_time_variable,
    load_fmatch_definition,
    pseudofootprints_from_camtime_dataset,
    write_fmatch_product,
)
from libera_utils.footprint_matching.readers.base import AncillaryReader, StaticReader
from libera_utils.footprint_matching.readers.registry import ReaderRegistry
from libera_utils.footprint_matching.types import (
    BoundingBox,
    CameraFootprintQualityFlag,
    FmatchCoverageFlag,
    GridSamples,
    OperationalMode,
    PseudoFootprint,
    RadiometerFootprint,
    TimeMatch,
    VariableSpec,
)
from libera_utils.footprint_matching.weighting import AngularPSFWeigher
from libera_utils.io.product_definition import LiberaVariableDefinition
from libera_utils.io.product_variable_names import product_variable_name
from tests.plugins.footprint_matching_fixtures import (
    FAKE_GRID_LATS,
    FAKE_GRID_LONS,
    FakeMeanReader,
    fake_radiometer_footprint,
    make_cf_cam_camtime_dataset,
    make_era5_surface_netcdf_fixture,
)

_PARTIAL = int(FmatchCoverageFlag.PARTIAL_COVERAGE)
_INSUFFICIENT = int(FmatchCoverageFlag.INSUFFICIENT_COVERAGE)
_TRUNCATED = int(FmatchCoverageFlag.LIMB_TRUNCATED)
_OFF_LIMB = int(FmatchCoverageFlag.OFF_LIMB)

# CF-CAM values, in percent, for the three footprints of _l1b_inputs and _pseudo_footprints.
_CLOUD_FRACTION = np.array([0.0, 50.0, 100.0])

RADIOMETER_MODES = tuple(mode for mode in OperationalMode if fmatch_time_variable(mode) == "RADIOMETER_TIME")


def _l1b_inputs(n: int = 3) -> dict[str, np.ndarray]:
    """L1B pass-through inputs for ``n`` nadir footprints over the fake readers' grid near (0.5, 0.5)."""
    return {
        "RADIOMETER_TIME": np.datetime64("2028-01-01T00:00:00", "ns") + np.arange(n) * np.timedelta64(1, "s"),
        "latitude": np.full(n, 0.5),
        "longitude": np.full(n, 0.5),
        "altitude": np.zeros(n),
        "solar_zenith_angle": np.full(n, 30.0),
        "viewing_zenith_angle": np.zeros(n),
        "relative_azimuth_angle": np.full(n, 90.0),
        "subsatellite_latitude": np.full(n, 0.5),
        "subsatellite_longitude": np.full(n, 0.5),
        "satellite_radius_km": np.full(n, 7213.0),
        "cone_angle_rate": np.full(n, -1.0),
    }


def _pseudo_footprint(
    time: np.datetime64, *, center_ix: int, center_iy: int, slice_x: slice, slice_y: slice
) -> PseudoFootprint:
    """A camera pseudo-footprint over the fake readers' grid near (0.5, 0.5)."""
    return PseudoFootprint(
        time=time,
        slice_x=slice_x,
        slice_y=slice_y,
        center_ix=center_ix,
        center_iy=center_iy,
        latitude=0.5,
        longitude=0.5,
        altitude=0.0,
        solar_zenith_angle=30.0,
        viewing_zenith_angle=10.0,
        relative_azimuth_angle=90.0,
        bbox=BoundingBox(0.2, 0.8, 0.2, 0.8),
        q_flags=CameraFootprintQualityFlag(0),
    )


def _pseudo_footprints() -> list[PseudoFootprint]:
    """Two images: the first segmented into two subsections, the second into one (one padded cell)."""
    first = np.datetime64("2028-01-01T00:00:00", "ns")
    second = np.datetime64("2028-01-01T00:00:10", "ns")
    return [
        _pseudo_footprint(first, center_ix=0, center_iy=0, slice_x=slice(0, 4), slice_y=slice(0, 4)),
        _pseudo_footprint(first, center_ix=6, center_iy=2, slice_x=slice(4, 8), slice_y=slice(0, 4)),
        _pseudo_footprint(second, center_ix=1, center_iy=1, slice_x=slice(0, 4), slice_y=slice(0, 4)),
    ]


class TestRadiometerPassThrough:
    """Radiometer-timescale modes carry the L1B columns through and merge cloud_fraction_camera."""

    @pytest.mark.parametrize("mode", RADIOMETER_MODES)
    def test_assembles_conformant_dataset(self, mode):
        dataset = assemble_fmatch_dataset(
            mode, _l1b_inputs(), algorithm_version="1.0.0", input_files="l1b.nc", allow_placeholders=True
        )
        assert load_fmatch_definition(mode).check_dataset_conformance(dataset, strict=True) == []

    @pytest.mark.parametrize("mode", RADIOMETER_MODES)
    def test_l1b_columns_are_carried_through_verbatim(self, mode):
        """The L1B-derived columns are real values, not placeholders."""
        l1b_inputs = _l1b_inputs()
        dataset = assemble_fmatch_dataset(mode, l1b_inputs, allow_placeholders=True)

        for name in _RADIOMETER_L1B_VARIABLES:
            np.testing.assert_allclose(dataset[product_variable_name(name)].values, l1b_inputs[name], rtol=1e-6)
        np.testing.assert_array_equal(dataset["RADIOMETER_TIME"].values, l1b_inputs["RADIOMETER_TIME"])

    @pytest.mark.parametrize("mode", RADIOMETER_MODES)
    def test_altitude_is_the_l1b_altitude(self, mode):
        """altitude is copied from L1B ``Altitude``, the boresight surface height in meters, not a placeholder."""
        assert L1B_PASSTHROUGH_VARIABLES["altitude"] == "Altitude"
        l1b_inputs = _l1b_inputs()
        l1b_inputs["altitude"] = np.array([0.0, 150.0, 4000.0], dtype=np.float32)
        dataset = assemble_fmatch_dataset(mode, l1b_inputs, allow_placeholders=True)

        np.testing.assert_array_equal(dataset[product_variable_name("altitude")].values, l1b_inputs["altitude"])

    def test_cloud_fraction_camera_is_merged_when_declared(self):
        """FMATCH-CAM declares cloud_fraction_camera, so supplied values become real, not placeholder."""
        values = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        dataset = assemble_fmatch_dataset(
            OperationalMode.CAM, _l1b_inputs(), cloud_fraction_camera=values, allow_placeholders=True
        )

        np.testing.assert_allclose(dataset[product_variable_name("cloud_fraction_camera")].values, values, rtol=1e-6)


class TestCloudFractionCamera:
    """cloud_fraction_camera is required for the CAM modes, refused for IMAGER, and one value per footprint."""

    @pytest.mark.parametrize(
        ("mode", "inputs"),
        [
            pytest.param(OperationalMode.CAM, _l1b_inputs, id="radiometer"),
            pytest.param(OperationalMode.CAM_CAMTIME, _pseudo_footprints, id="camtime"),
        ],
    )
    def test_omitted_raises(self, mode, inputs):
        with pytest.raises(ValueError, match="declares Cloud_Fraction_Camera but no values were given"):
            assemble_fmatch_dataset(mode, inputs(), allow_placeholders=False)

    def test_omitted_with_placeholders_is_fill(self):
        dataset = assemble_fmatch_dataset(OperationalMode.CAM, _l1b_inputs(), allow_placeholders=True)
        assert np.isnan(dataset[product_variable_name("cloud_fraction_camera")].values).all()

    @pytest.mark.parametrize("mode", [OperationalMode.IMAGER, OperationalMode.IMAGER_CAMTIME])
    def test_values_for_an_imager_mode_raise(self, mode):
        inputs = _pseudo_footprints() if mode is OperationalMode.IMAGER_CAMTIME else _l1b_inputs()
        with pytest.raises(ValueError, match="does not declare that variable"):
            assemble_fmatch_dataset(mode, inputs, cloud_fraction_camera=np.zeros(3), allow_placeholders=True)

    @pytest.mark.parametrize(
        ("mode", "inputs"),
        [
            pytest.param(OperationalMode.CAM, _l1b_inputs, id="radiometer"),
            pytest.param(OperationalMode.CAM_CAMTIME, _pseudo_footprints, id="camtime"),
        ],
    )
    def test_wrong_length_raises(self, mode, inputs):
        with pytest.raises(ValueError, match="2 values for 3 footprints"):
            assemble_fmatch_dataset(mode, inputs(), cloud_fraction_camera=np.zeros(2), allow_placeholders=True)


class TestRadiometerAssemblyRaises:
    """_assemble_radiometer_dataset rejects inputs it cannot build a product from."""

    @staticmethod
    def _without(name: str) -> dict[str, np.ndarray]:
        inputs = _l1b_inputs()
        del inputs[name]
        return inputs

    @staticmethod
    def _short(name: str) -> dict[str, np.ndarray]:
        inputs = _l1b_inputs()
        inputs[name] = inputs[name][:-1]
        return inputs

    @pytest.mark.parametrize(
        ("inputs", "match"),
        [
            pytest.param(lambda: TestRadiometerAssemblyRaises._without("relative_azimuth_angle"), "missing", id="key"),
            pytest.param(lambda: _l1b_inputs(0), "zero footprints", id="empty"),
            pytest.param(lambda: TestRadiometerAssemblyRaises._short("latitude"), "inconsistent", id="length"),
        ],
    )
    def test_bad_inputs_raise(self, inputs, match):
        with pytest.raises(ValueError, match=match):
            assemble_fmatch_dataset(OperationalMode.CAM, inputs(), allow_placeholders=True)

    def test_camera_timescale_mode_raises(self):
        with pytest.raises(ValueError, match="camera-timescale mode"):
            _assemble_radiometer_dataset(_l1b_inputs(), mode=OperationalMode.CAM_CAMTIME)


class TestCamtimeAssemblyRaises:
    """_assemble_camtime_dataset rejects a radiometer mode and an empty footprint list."""

    def test_zero_pseudo_footprints_raise(self):
        with pytest.raises(ValueError, match="zero pseudo-footprints"):
            assemble_fmatch_dataset(OperationalMode.CAM_CAMTIME, [])

    def test_radiometer_mode_raises(self):
        with pytest.raises(ValueError, match="not a camera-timescale mode"):
            _assemble_camtime_dataset(_pseudo_footprints(), mode=OperationalMode.CAM)


class TestAncillaryInputs:
    """Without ancillary inputs, assembly raises unless placeholders are asked for."""

    def test_no_ancillary_inputs_raise(self):
        with pytest.raises(ValueError, match="allow_placeholders"):
            assemble_fmatch_dataset(OperationalMode.CAM, _l1b_inputs(), cloud_fraction_camera=_CLOUD_FRACTION)

    def test_writer_raises_without_ancillary_inputs(self, tmp_path):
        with pytest.raises(ValueError, match="allow_placeholders"):
            write_fmatch_product(
                OperationalMode.CAM,
                _l1b_inputs(),
                tmp_path,
                algorithm_version="1.0.0",
                input_files="x",
                cloud_fraction_camera=_CLOUD_FRACTION,
            )

    def test_allow_placeholders_writes_fills(self):
        dataset = assemble_fmatch_dataset(OperationalMode.CAM, _l1b_inputs(), allow_placeholders=True)
        # Integer categories take the -1 fill, not 0 (a real category); q_flags 0 means no flag set.
        np.testing.assert_array_equal(dataset[product_variable_name("igbp_surface_type")].values, [-1, -1, -1])
        np.testing.assert_array_equal(dataset[product_variable_name("q_flags")].values, [0, 0, 0])
        assert np.isnan(dataset[product_variable_name("psf_coverage_fraction")].values).all()


class TestFillValueFor:
    """_fill_value_for gives the declared fill, 0 for a bitmask and NaN for a float; an undeclared integer raises."""

    @staticmethod
    def _definition(dtype: str, attributes: dict) -> LiberaVariableDefinition:
        return LiberaVariableDefinition(dtype=dtype, dimensions=["RADIOMETER_TIME"], attributes=attributes)

    def test_declared_fill_wins(self):
        assert _fill_value_for("x", self._definition("int16", {"long_name": "x", "_FillValue": -999}))[1] == -999

    def test_undeclared_integer_raises(self):
        with pytest.raises(
            ValueError, match="Integer variable surface_type declares neither _FillValue nor flag_masks"
        ):
            _fill_value_for("surface_type", self._definition("int16", {"long_name": "surface type"}))

    def test_bitmask_takes_zero(self):
        assert _fill_value_for("flags", self._definition("int32", {"long_name": "flags", "flag_masks": [1, 4]}))[1] == 0

    def test_float_takes_nan(self):
        assert np.isnan(_fill_value_for("x", self._definition("float32", {"long_name": "x"}))[1])


class TestNormalizeLongitude:
    """Only longitudes above 180 wrap, so a full-span box keeps its 180 bound."""

    @pytest.mark.parametrize(
        ("lon_min", "lon_max", "expected"),
        [
            (-180.0, 180.0, (-180.0, 180.0)),
            (170.0, 180.0, (170.0, 180.0)),
            (179.9, 180.2, (179.9, -179.8)),
        ],
    )
    def test_box_bounds(self, lon_min, lon_max, expected):
        assert (_normalize_longitude(lon_min), _normalize_longitude(lon_max)) == pytest.approx(expected)


class TestFootprintGeometry:
    """The PSF geometry is read from the L1B ephemeris, never assumed."""

    def test_missing_viewing_zenith_raises(self):
        footprint = fake_radiometer_footprint(
            BoundingBox(0.2, 0.8, 0.2, 0.8), 0.5, 0.5, viewing_zenith_angle=float("nan")
        )
        with pytest.raises(ValueError, match="viewing zenith angle"):
            _footprint_geometry(footprint)

    def test_radiometer_footprint_uses_its_satellite_radius(self):
        (footprint,) = build_radiometer_footprints(_l1b_inputs(1))
        geom = _footprint_geometry(footprint)
        assert geom.satellite_radius_km == 7213.0
        assert geom.subsatellite_lat_deg == 0.5

    def test_pseudo_footprint_without_ephemeris_raises(self):
        # A camera PseudoFootprint's ``altitude`` is the center-pixel surface height in
        # meters, not a spacecraft position, so the PSF has nothing to be sized from.
        with pytest.raises(ValueError, match="no L1B ephemeris"):
            _footprint_geometry(_pseudo_footprints()[0])

    def test_other_footprint_type_raises(self):
        with pytest.raises(TypeError, match="RadiometerFootprint or PseudoFootprint"):
            _footprint_geometry(object())


class TestSunglint:
    """The L1B relative azimuth puts the Sun at 180 (CERES BDS origin), so RAA 0 is specular."""

    @pytest.mark.parametrize(
        ("sza", "vza", "raa", "expected"),
        [
            pytest.param(30.0, 30.0, 0.0, 0.0, id="specular"),
            pytest.param(30.0, 30.0, 180.0, 60.0, id="hot-spot"),
            pytest.param(40.0, 10.0, 0.0, 30.0, id="forward-scatter"),
            pytest.param(40.0, 10.0, 180.0, 50.0, id="backscatter"),
            pytest.param(30.0, 0.0, 90.0, 30.0, id="nadir"),
        ],
    )
    def test_glint_angle(self, sza, vza, raa, expected):
        glint = compute_derived_viewing_geometry(np.array([sza]), np.array([vza]), np.array([raa]))
        assert glint["sunglint_angle"][0] == pytest.approx(expected, abs=1e-6)

    def test_nan_input_propagates(self):
        glint = compute_derived_viewing_geometry(np.array([np.nan]), np.array([0.0]), np.array([0.0]))
        assert np.isnan(glint["sunglint_angle"][0])


class TestMergeCoverageQa:
    """Coverage below 75% is insufficient, 75-95% partial; truncation and off-limb add their own bits."""

    @staticmethod
    def _footprint(*, truncated: bool = False, off_limb: bool = False) -> RadiometerFootprint:
        bbox = None if off_limb else BoundingBox(0.2, 0.8, 0.2, 0.8, truncated=truncated)
        return fake_radiometer_footprint(bbox, 0.5, 0.5, off_limb=off_limb)

    def test_coverage_bits(self):
        definition = _internal_name_view(load_fmatch_definition(OperationalMode.CAM))
        footprints = [
            self._footprint(),
            self._footprint(),
            self._footprint(),
            self._footprint(truncated=True),
            self._footprint(off_limb=True),
        ]
        data: dict[str, np.ndarray] = {}
        _merge_coverage_qa(data, definition, np.array([0.5, 0.8, 1.0, 1.0, 0.0]), footprints)
        np.testing.assert_array_equal(
            data["q_flags"], [_INSUFFICIENT, _PARTIAL, 0, _TRUNCATED, _OFF_LIMB | _INSUFFICIENT]
        )
        np.testing.assert_allclose(data["psf_coverage_fraction"], [0.5, 0.8, 1.0, 1.0, 0.0])

    @pytest.mark.parametrize(
        ("coverage", "bits"),
        [
            pytest.param(0.749, _INSUFFICIENT, id="below-accept"),
            pytest.param(0.75, _PARTIAL, id="accept-edge"),
            pytest.param(0.949, _PARTIAL, id="below-clean"),
            pytest.param(0.95, 0, id="clean-edge"),
        ],
    )
    def test_threshold_edges(self, coverage, bits):
        """Coverage exactly at 75% is accepted (partial) and exactly at 95% is clean."""
        assert product.ACCEPT_COVERAGE_THRESHOLD == 0.75
        assert product.PARTIAL_COVERAGE_THRESHOLD == 0.95
        data: dict[str, np.ndarray] = {}
        _merge_coverage_qa(
            data,
            _internal_name_view(load_fmatch_definition(OperationalMode.CAM)),
            np.array([coverage]),
            [self._footprint()],
        )
        assert data["q_flags"][0] == bits

    def test_existing_flags_are_kept(self):
        definition = _internal_name_view(load_fmatch_definition(OperationalMode.CAM))
        data = {"q_flags": np.array([_TRUNCATED], dtype=np.int32)}
        _merge_coverage_qa(data, definition, np.array([0.8]), [self._footprint()])
        assert data["q_flags"][0] == _TRUNCATED | _PARTIAL


class _WestHalfReader(StaticReader):
    """Serves the fake grid's western columns only, so it covers well under 75% of a footprint box."""

    READER_KEY = "_fake_west_half"
    INSTRUMENT = "TEST"
    RESOLUTION_KM = 25.0
    TIME_MATCH = TimeMatch.DAILY
    VARIABLES = (
        VariableSpec(name="field", dtype="float32", aggregation="weighted_mean", modes=frozenset(OperationalMode)),
    )

    def _load_native(self) -> GridSamples:
        lons = FAKE_GRID_LONS[:2]
        return GridSamples(
            lat_axis=FAKE_GRID_LATS,
            lon_axis=lons,
            data=np.full((1, FAKE_GRID_LATS.size, lons.size), 1.0, dtype=np.float32),
        )


class TestCoverageOverSources:
    """A footprint's coverage is the minimum over its sources, and that minimum sets the QA bits."""

    def test_worst_covered_source_sets_coverage(self, fake_sources, set_mode_readers, tmp_path):
        half_path = tmp_path / "west.bin"
        half_path.touch()
        half = _WestHalfReader([half_path])
        footprints = [fake_radiometer_footprint(BoundingBox(0.2, 0.8, 0.2, 0.8), 0.5, 0.5)]

        set_mode_readers(OperationalMode.CAM, (FakeMeanReader.READER_KEY,))
        _, full_coverage = product.aggregate_external_variables(
            OperationalMode.CAM,
            footprints,
            {FakeMeanReader.READER_KEY: fake_sources[FakeMeanReader.READER_KEY]},
            return_coverage=True,
        )
        set_mode_readers(OperationalMode.CAM, (_WestHalfReader.READER_KEY,))
        _, half_coverage = product.aggregate_external_variables(
            OperationalMode.CAM, footprints, {_WestHalfReader.READER_KEY: half}, return_coverage=True
        )
        assert half_coverage[0] < product.ACCEPT_COVERAGE_THRESHOLD < full_coverage[0]

        set_mode_readers(OperationalMode.CAM, (FakeMeanReader.READER_KEY, _WestHalfReader.READER_KEY))
        _, coverage = product.aggregate_external_variables(
            OperationalMode.CAM,
            footprints,
            {FakeMeanReader.READER_KEY: fake_sources[FakeMeanReader.READER_KEY], _WestHalfReader.READER_KEY: half},
            return_coverage=True,
        )
        np.testing.assert_allclose(coverage, half_coverage)

        data: dict[str, np.ndarray] = {}
        _merge_coverage_qa(data, _internal_name_view(load_fmatch_definition(OperationalMode.CAM)), coverage, footprints)
        assert data["q_flags"][0] & _INSUFFICIENT


class TestCamtimeAssembly:
    def test_grid_padding_takes_fills(self):
        cloud_fraction = np.array([0.1, 0.2, 0.3])
        dataset = assemble_fmatch_dataset(
            OperationalMode.CAM_CAMTIME,
            _pseudo_footprints(),
            cloud_fraction_camera=cloud_fraction,
            allow_placeholders=True,
        )
        assert dataset[product_variable_name("latitude")].shape == (2, 2)
        # Row 1 holds the second image's single subsection; its second cell is padding.
        np.testing.assert_array_equal(dataset[product_variable_name("center_pixel_x")].values, [[0, 6], [1, -1]])
        np.testing.assert_array_equal(dataset[product_variable_name("camera_pixel_x_max")].values, [[3, 7], [3, -1]])
        assert np.isnan(dataset[product_variable_name("latitude")].values[1, 1])
        np.testing.assert_allclose(
            dataset[product_variable_name("cloud_fraction_camera")].values, [[0.1, 0.2], [0.3, np.nan]]
        )
        np.testing.assert_array_equal(dataset[product_variable_name("igbp_surface_type")].values, np.full((2, 2), -1))


class TestIntegerAggregatePlacement:
    """An aggregated integer column takes the -1 fill where a footprint had no usable data."""

    @staticmethod
    def _aggregate_with_one_gap(monkeypatch):
        def fake_aggregate(mode, footprints, sources=None, **kwargs):
            values = np.full(len(footprints), 3.0)
            values[0] = np.nan
            return {"igbp_surface_type": values}, np.ones(len(footprints))

        monkeypatch.setattr(product, "aggregate_external_variables", fake_aggregate)

    def test_radiometer_column(self, fake_sources, monkeypatch):
        self._aggregate_with_one_gap(monkeypatch)
        dataset = assemble_fmatch_dataset(
            OperationalMode.CAM, _l1b_inputs(), sources=fake_sources, cloud_fraction_camera=_CLOUD_FRACTION
        )
        assert dataset[product_variable_name("igbp_surface_type")].dtype == np.int16
        np.testing.assert_array_equal(dataset[product_variable_name("igbp_surface_type")].values, [-1, 3, 3])

    def test_camtime_grid(self, fake_sources, monkeypatch):
        self._aggregate_with_one_gap(monkeypatch)
        dataset = assemble_fmatch_dataset(
            OperationalMode.CAM_CAMTIME,
            _pseudo_footprints(),
            sources=fake_sources,
            cloud_fraction_camera=_CLOUD_FRACTION,
        )
        # The first footprint's gap and the padded cell both take -1.
        np.testing.assert_array_equal(dataset[product_variable_name("igbp_surface_type")].values, [[-1, 3], [3, -1]])


class TestRadiometerAggregation:
    def test_defaults_to_angular_weigher(self, fake_sources, monkeypatch):
        calls = []
        original = AngularPSFWeigher.weight_field

        def spy(self, *args, **kwargs):
            calls.append(type(self))
            return original(self, *args, **kwargs)

        monkeypatch.setattr(AngularPSFWeigher, "weight_field", spy)
        assemble_fmatch_dataset(
            OperationalMode.CAM, _l1b_inputs(), sources=fake_sources, cloud_fraction_camera=_CLOUD_FRACTION
        )
        assert calls
        assert set(calls) == {AngularPSFWeigher}


class TestWriteFmatchProduct:
    def test_round_trip(self, tmp_path, fake_sources):
        filename = write_fmatch_product(
            OperationalMode.CAM,
            _l1b_inputs(),
            tmp_path,
            algorithm_version="1.0.0",
            input_files="x",
            sources=fake_sources,
            cloud_fraction_camera=_CLOUD_FRACTION,
        )
        assert filename.data_product_id == DataProductIdentifier.aux_fmatch_cam

        with xr.open_dataset(filename.path, mask_and_scale=False) as written:
            coverage = written[product_variable_name("psf_coverage_fraction")].values
            assert np.all((coverage >= 0.0) & (coverage <= 1.0))
            expected = np.where(
                coverage < product.ACCEPT_COVERAGE_THRESHOLD,
                _INSUFFICIENT,
                np.where(coverage < product.PARTIAL_COVERAGE_THRESHOLD, _PARTIAL, 0),
            )
            np.testing.assert_array_equal(written[product_variable_name("q_flags")].values, expected)
            # The fake readers feed no IGBP data, so the column is the -1 integer fill on disk, declared as such.
            np.testing.assert_array_equal(written[product_variable_name("igbp_surface_type")].values, [-1, -1, -1])
            assert written[product_variable_name("igbp_surface_type")].attrs["_FillValue"] == -1

    @pytest.mark.parametrize(
        "mode",
        [OperationalMode.CAM, OperationalMode.IMAGER, OperationalMode.CAM_CAMTIME, OperationalMode.IMAGER_CAMTIME],
    )
    def test_written_product_carries_only_product_names(self, tmp_path, mode):
        """The written file holds exactly the definition's product names, never an internal one.

        Assembly fills internal snake_case columns; each is written under its product name. The camera-timescale
        products also carry the Camera_Pixel_* non-dimension coordinates, which must reopen as coordinates.
        """
        camtime = mode in (OperationalMode.CAM_CAMTIME, OperationalMode.IMAGER_CAMTIME)
        filename = write_fmatch_product(
            mode,
            _pseudo_footprints() if camtime else _l1b_inputs(),
            tmp_path,
            algorithm_version="1.0.0",
            input_files="x",
            cloud_fraction_camera=_CLOUD_FRACTION
            if mode in (OperationalMode.CAM, OperationalMode.CAM_CAMTIME)
            else None,
            allow_placeholders=True,
        )

        definition = load_fmatch_definition(mode)
        with xr.open_dataset(filename.path, mask_and_scale=False) as written:
            assert set(written.data_vars) == set(definition.variables)
            assert set(written.coords) == set(definition.coordinates)


def _l1b_at(lat: float, lon: float, *, sza: float = 30.0, vza: float = 5.0, raa: float = 90.0) -> dict:
    """A one-footprint L1B pass-through dict at a chosen boresight location."""
    return {
        "RADIOMETER_TIME": np.array([np.datetime64("2026-06-11T00:00:00", "ns")]),
        "latitude": np.array([lat], dtype=np.float32),
        "longitude": np.array([lon], dtype=np.float32),
        "altitude": np.array([0.0], dtype=np.float32),
        "solar_zenith_angle": np.array([sza], dtype=np.float32),
        "viewing_zenith_angle": np.array([vza], dtype=np.float32),
        "relative_azimuth_angle": np.array([raa], dtype=np.float32),
        # The satellite sits just north of the boresight, consistent with a small VZA.
        "subsatellite_latitude": np.array([lat + 0.5], dtype=np.float64),
        "subsatellite_longitude": np.array([lon], dtype=np.float64),
        "satellite_radius_km": np.array([7213.0]),
        "cone_angle_rate": np.array([-1.0]),
    }


class TestAssemblyWithAggregation:
    """Supplying readers makes the external columns carry real aggregated values."""

    def _era5_sources(self, tmp_path) -> dict[str, AncillaryReader]:
        """The real ERA5 surface reader, the one CAM reader for these tests, over a constant-wind grid.

        The grid is fine (0.1 deg) so a grid point falls inside the PSF ground radius
        (~26 km) of the test boresight; u10 == 2.5 and v10 == -1.5 everywhere, so the
        PSF-weighted mean is exactly those constants regardless of the weight kernel. The
        fixture's one hourly step is the L1B time of ``_l1b_at``, so the step matches.
        """
        era5_cls = ReaderRegistry.get("era5_surface")
        era5_file = make_era5_surface_netcdf_fixture(
            tmp_path,
            lat_min=0.0,
            lat_max=2.0,
            lon_min=10.0,
            lon_max=12.0,
            n_lat=21,
            n_lon=21,
            times=[np.datetime64("2026-06-11T00:00:00", "ns")],
        )
        return {"era5_surface": era5_cls([era5_file])}

    @pytest.fixture(autouse=True)
    def _cam_uses_era5_surface_only(self, set_mode_readers):
        set_mode_readers(OperationalMode.CAM, ("era5_surface",))

    def test_radiometer_assembly_carries_real_era5_values(self, tmp_path):
        # Footprint sits inside the ERA5 grid so every PSF cell has data.
        l1b = _l1b_at(lat=1.0, lon=11.0)
        dataset = assemble_fmatch_dataset(
            OperationalMode.CAM,
            l1b,
            sources=self._era5_sources(tmp_path),
            cloud_fraction_camera=np.zeros(1),
            algorithm_version="1.0.0",
            input_files="l1b.nc,era5.nc",
        )

        # Weighted mean of a constant field is that constant.
        np.testing.assert_allclose(dataset[product_variable_name("era5_surface_wind_u10")].values, [2.5], rtol=1e-4)
        np.testing.assert_allclose(dataset[product_variable_name("era5_surface_wind_v10")].values, [-1.5], rtol=1e-4)
        # A constant field has zero within-footprint spread.
        np.testing.assert_allclose(
            dataset[product_variable_name("era5_surface_wind_u10_standard_deviation")].values, [0.0], atol=1e-4
        )
        # Fully covered -> coverage ~1 and no coverage QA bits set.
        assert dataset[product_variable_name("psf_coverage_fraction")].values[0] > 0.99
        assert int(dataset[product_variable_name("q_flags")].values[0]) == 0
        # igbp is not among the readers set for the test, so its column stays the -1 integer category fill.
        assert int(dataset[product_variable_name("igbp_surface_type")].values[0]) == -1
        # The product still conforms strictly with the computed columns in place.
        assert load_fmatch_definition(OperationalMode.CAM).check_dataset_conformance(dataset, strict=True) == []

    def test_footprint_off_the_grid_is_flagged_insufficient_coverage(self, tmp_path):
        # Boresight far from the ERA5 grid -> no samples -> zero coverage.
        l1b = _l1b_at(lat=1.0, lon=100.0)
        dataset = assemble_fmatch_dataset(
            OperationalMode.CAM, l1b, sources=self._era5_sources(tmp_path), cloud_fraction_camera=np.zeros(1)
        )

        assert np.isnan(dataset[product_variable_name("era5_surface_wind_u10")].values[0])
        np.testing.assert_allclose(dataset[product_variable_name("psf_coverage_fraction")].values, [0.0])
        assert int(dataset[product_variable_name("q_flags")].values[0]) & _INSUFFICIENT


class TestPseudofootprintsFromCamtime:
    """Camera pseudo-footprints are rebuilt from the CF-CAM-CAMTIME grid the Camera Cloud Fraction algorithm writes."""

    def test_rebuilds_one_footprint_per_real_cell_in_row_major_order(self):
        dataset = make_cf_cam_camtime_dataset(n_images=2, n_pixels_x=2, n_pixels_y=2)

        footprints = pseudofootprints_from_camtime_dataset(dataset)

        assert len(footprints) == 8
        assert [f.time for f in footprints] == [dataset["CAMERA_TIME"].values[i] for i in (0, 0, 0, 0, 1, 1, 1, 1)]
        assert [(f.center_ix, f.center_iy) for f in footprints[:4]] == [(0, 0), (0, 1), (1, 0), (1, 1)]

    def test_inclusive_pixel_bounds_become_half_open_slices(self):
        footprint = pseudofootprints_from_camtime_dataset(make_cf_cam_camtime_dataset(n_images=1))[0]

        assert footprint.slice_x == slice(0, 1)
        assert footprint.slice_y == slice(0, 1)

    def test_padded_cells_are_skipped(self):
        dataset = make_cf_cam_camtime_dataset(n_images=2, n_pixels_x=2, n_pixels_y=2, n_padded=3)

        assert len(pseudofootprints_from_camtime_dataset(dataset)) == 5

    def test_bbox_and_quality_flags_are_decoded(self):
        dataset = make_cf_cam_camtime_dataset(n_images=1, n_pixels_x=1, n_pixels_y=2)
        dataset["psf_bbox_flags"].values[0, 1] = CAMTIME_BBOX_WRAPS_DATELINE_BIT | CAMTIME_BBOX_TRUNCATED_BIT
        dataset["q_flags"].values[0, 1] = int(CameraFootprintQualityFlag.CENTER_PIXEL_SUBSTITUTED)

        clean, flagged = pseudofootprints_from_camtime_dataset(dataset)

        assert (clean.bbox.wraps_dateline, clean.bbox.is_polar, clean.bbox.truncated) == (False, False, False)
        assert (flagged.bbox.wraps_dateline, flagged.bbox.is_polar, flagged.bbox.truncated) == (True, False, True)
        assert clean.q_flags == CameraFootprintQualityFlag(0)
        assert flagged.q_flags == CameraFootprintQualityFlag.CENTER_PIXEL_SUBSTITUTED

    def test_missing_variable_raises_naming_it(self):
        dataset = make_cf_cam_camtime_dataset().drop_vars(["psf_bbox_flags", "camera_pixel_x_min"])

        with pytest.raises(MissingCamtimeVariablesError, match="camera_pixel_x_min, psf_bbox_flags"):
            pseudofootprints_from_camtime_dataset(dataset)

    @pytest.mark.parametrize("mode", [OperationalMode.CAM_CAMTIME, OperationalMode.IMAGER_CAMTIME])
    def test_every_required_name_is_one_the_fmatch_camtime_products_declare(self, mode):
        """FMATCH copies these straight into its camera-timescale products, so each must be declared there."""
        definition = load_fmatch_definition(mode)
        declared = {*definition.variables, *definition.coordinates}
        assert set(CAMTIME_PSEUDOFOOTPRINT_VARIABLES) - declared - {"psf_bbox_flags"} == set()
