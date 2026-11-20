"""Unit tests for CeresSsfReader (CERES SSF footprint reader).

Uses synthetic CERES SSF NetCDF4 fixtures written by the ``ceres_ssf_file_factory`` fixture. The fixture
clusters five footprints near lat ≈ 10–11°, lon ≈ −10° (stored as 350° in the
0..360 convention) plus one far-away footprint, so tests can verify longitude
normalization, fill handling, layer selection, and time matching.

The real-granule tests read a cut of ``CER_SSF_NOAA20-FM6-VIIRS_Edition1C_103103.2026061100.nc``.
"""

import numpy as np
import pytest

from libera_utils.footprint_matching.readers.ceres_ssf import CeresSsfReader
from libera_utils.footprint_matching.types import BoundingBox
from tests.plugins.footprint_matching_fixtures import SWATH_FIXTURE_TIME
from tests.unit.test_footprint_matching.test_readers.grid_helpers import raw_and_usable, served_at

# Box that contains the default footprint cluster (lon −10° == 350°).
_BBOX = BoundingBox(10.0, 12.0, -11.0, -9.0)
_T = SWATH_FIXTURE_TIME


def _values(reader: CeresSsfReader, bbox: BoundingBox = _BBOX, time=_T) -> np.ndarray:
    """The ``(n_var, n)`` value rows of the samples in ``bbox`` at ``time``."""
    return reader.samples_in(bbox, time).values


def _var_index(name: str) -> int:
    return [v.name for v in CeresSsfReader.VARIABLES].index(name)


def _finite_values(data: np.ndarray, name: str) -> np.ndarray:
    arr = data[_var_index(name)]
    return np.sort(arr[np.isfinite(arr)])


def _ramp(start: float, step: float) -> list[float]:
    return [start + step * i for i in range(5)]


# The five clustered footprints' values for every field, sorted, from the fixture defaults: a layered field's upper
# layer differs from its lower one, and each coverage field's lower and upper entries differ from its other two. Two
# of the five shortwave ADM codes are the int16 fill.
_CLUSTER_VALUES = {
    "Clear_Coverage": _ramp(10.0, 10.0),
    "Cloud_Optical_Depth_Lower": [1.0, 2.0, 4.0, 8.0, 16.0],
    "Cloud_Water_Particle_Radius_Lower": _ramp(5.0, 1.0),
    "Cloud_Ice_Particle_Radius_Lower": _ramp(20.0, 5.0),
    "Cloud_Classification": [1001.0, 1001.0, 1001.0, 1001.0, 1191.0],
    "Shortwave_ADM_Type": [50.0, 50.0, 50.0],
    "Longwave_ADM_Type": [50.0] * 5,
    "Layer_Coverage_Lower": _ramp(10.0, 5.0),
    "Layer_Coverage_Upper": _ramp(15.0, 5.0),
    "Cloud_Coverage_Multilayer_Lower": _ramp(5.0, 3.0),
    "Cloud_Coverage_Multilayer_Upper": _ramp(8.0, 3.0),
    "Cloud_Top_Pressure_Lower": _ramp(300.0, 10.0),
    "Cloud_Top_Pressure_Upper": _ramp(250.0, 10.0),
    "Cloud_Base_Pressure_Lower": _ramp(800.0, 10.0),
    "Cloud_Base_Pressure_Upper": _ramp(700.0, 10.0),
    "Cloud_Particle_Phase_37um_Lower": _ramp(1.0, 0.05),
    "Cloud_Particle_Phase_37um_Upper": _ramp(1.5, 0.05),
    "Cloud_Optical_Depth_Upper": [2.0, 3.0, 5.0, 9.0, 17.0],
    "Cloud_Water_Particle_Radius_Upper": _ramp(6.0, 1.0),
    "Cloud_Ice_Particle_Radius_Upper": _ramp(22.0, 5.0),
    "Imager_Dark_Target_Land_055um_Corrected_Aerosol_Optical_Depth": _ramp(0.20, 0.10),
    "Imager_Deep_Blue_Ocean_055um_Aerosol_Optical_Depth": _ramp(0.25, 0.10),
    "Surface_Albedo": _ramp(0.10, 0.05),
    "TOA_Incoming_Solar_Radiation": _ramp(1360.0, 1.0),
}


class TestCeresSsfReaderSamples:
    def test_cluster_is_served_as_time_stamped_footprints(self, ceres_ssf_file_factory):
        """``CeresSsfReader.samples_in`` serves each footprint as its own float32 sample, at its position and time.

        Each footprint is a native sample for the PSF weigher, tagged ``"CERES_SSF"`` with its own observation time, which
        the nearest-pass selection relies on. SSF stores longitude in 0..360 while boxes use -180..180, so without
        normalization every western-hemisphere footprint would be missed.

        Reads the default fixture over the cluster box at -11..-9 degrees and asserts the values are float32
        ``(len(VARIABLES), 5)``, every longitude is -10 degrees (stored as 350), the source is ``"CERES_SSF"``,
        ``timestamp_source`` is None and every sample time is the fixture time.
        """
        samples = CeresSsfReader([ceres_ssf_file_factory()]).samples_in(_BBOX, _T)
        assert samples.values.shape == (len(CeresSsfReader.VARIABLES), 5)
        assert samples.values.dtype == np.float32
        assert np.allclose(samples.lons, -10.0, atol=1e-4)
        assert samples.source == "CERES_SSF"
        assert samples.timestamp_source is None
        assert np.all(samples.times == _T)

    def test_every_field_serves_its_source_values(self, ceres_ssf_file_factory):
        """``CeresSsfReader.samples_in`` serves every field in its own row, from its own layer or condition, fill as NaN.

        Values are matched to variables by row, so a row out of ``VARIABLES`` order publishes one field under another's
        name. A layered field read from the wrong layer reports the wrong cloud, a coverage read from the wrong
        condition reports clear sky, overlap or single-layer cloud as a cloud layer, a 1-D field indexed as layered
        serves the wrong footprint, and a fill left as a number votes in the mode of a code field.

        Reads the default fixture over the cluster box once and asserts, for every variable, that its sorted finite
        values equal the fixture's (``_CLUSTER_VALUES``): the shortwave ADM row keeps three of its five codes.
        """
        data = _values(CeresSsfReader([ceres_ssf_file_factory()]))
        assert set(_CLUSTER_VALUES) == {v.name for v in CeresSsfReader.VARIABLES}
        for name, expected in _CLUSTER_VALUES.items():
            np.testing.assert_allclose(_finite_values(data, name), expected, rtol=1e-5, err_msg=name)


class TestCeresSsfReaderTimeMatching:
    def test_nearest_pass_is_served(self, ceres_ssf_file_factory):
        """``CeresSsfReader.samples_in`` serves only the pass nearest the query time when two passes cover the box.

        Two passes over the same place can both fall inside ``MAX_TIME_OFFSET``; mixing the farther one into a Libera
        footprint would average clouds and aerosols observed hours apart into one match.

        Builds a fixture with three footprints at the fixture time and three two hours later at the same location, so
        both passes are within the three-hour ``MAX_TIME_OFFSET`` of either query, queries 20 minutes after each pass,
        and asserts each query returns only that pass's three samples and clear coverages.
        """
        # Six footprints: the first three at the fixture time, the last three two hours
        # later, all at the same cluster location. A footprint near either pass gets that
        # pass's three footprints only.
        lats = np.array([10.2, 10.4, 10.6, 10.2, 10.4, 10.6], dtype=np.float32)
        lons = np.full(6, 350.0, dtype=np.float32)
        late_time = _T + np.timedelta64(2, "h")
        times = np.array([_T, _T, _T, late_time, late_time, late_time], dtype="datetime64[ns]")
        reader = CeresSsfReader([ceres_ssf_file_factory(lats=lats, lons_0360=lons, times=times)])
        near_early = reader.samples_in(_BBOX, _T + np.timedelta64(20, "m"))
        assert near_early.n_samples == 3
        assert np.allclose(_finite_values(near_early.values, "Clear_Coverage"), [10.0, 20.0, 30.0], atol=1e-4)
        near_late = reader.samples_in(_BBOX, late_time + np.timedelta64(20, "m"))
        assert near_late.n_samples == 3
        assert np.allclose(_finite_values(near_late.values, "Clear_Coverage"), [40.0, 50.0, 60.0], atol=1e-4)


class TestCeresSsfReaderSpacing:
    def test_served_resolution_is_the_measured_footprint_spacing(self, ceres_ssf_file_factory):
        """``CeresSsfReader.samples_in`` serves the granule's measured footprint spacing as ``resolution_km``.

        The coverage measure credits each sample with this spacing; using the nominal value for a sparser swath would
        overstate how completely SSF covers a Libera footprint.

        Reads the default fixture, whose footprints lie 0.2 degrees (about 22 km) apart along one scan, and asserts
        ``resolution_km`` is 22.2 km within 3 percent.
        """
        # The default fixture lays five footprints 0.2 deg (~22 km) apart along one scan,
        # so the measured along-scan spacing, not the nominal 20 km, is served.
        reader = CeresSsfReader([ceres_ssf_file_factory()])
        samples = reader.samples_in(_BBOX, _T)
        assert samples.resolution_km == pytest.approx(22.2, rel=0.03)


class TestCeresSsfReaderGeolocationChecks:
    @pytest.mark.parametrize("units", ["degrees", None])
    def test_latitude_units_other_than_degrees_north_raises(self, units, ceres_ssf_file_factory):
        """``CeresSsfReader.samples_in`` raises ``ValueError`` when footprint latitude is not in ``degrees_north``.

        The reader assumes a true -90..90 latitude; a granule using another convention, such as a colatitude, would
        place every footprint at the wrong latitude without an error.

        Parametrized over a ``"degrees"`` units attribute and no units attribute; asserts the error names
        ``instrument_fov_latitude`` and its units.
        """
        path = ceres_ssf_file_factory(latitude_units=units)
        with pytest.raises(ValueError, match="instrument_fov_latitude has units"):
            CeresSsfReader([path]).samples_in(_BBOX, _T)

    @pytest.mark.parametrize(
        ("attribute", "name"),
        [
            ("_FillValue", "instrument_fov_latitude"),
            ("_FillValue", "instrument_fov_longitude"),
            ("_FillValue", "time"),
            ("_FillValue", "clear_coverage"),
            ("_FillValue", "cloud_classification"),
            ("valid_range", "instrument_fov_latitude"),
            ("valid_range", "instrument_fov_longitude"),
            ("valid_range", "time"),
            ("valid_range", "clear_coverage"),
            ("valid_range", "cloud_ice_particle_radius_37um_mean"),
        ],
    )
    def test_field_without_fill_value_or_valid_range_raises(self, attribute, name, ceres_ssf_file_factory):
        """``CeresSsfReader.samples_in`` raises ``ValueError`` when a field it reads has no ``_FillValue`` or ``valid_range``.

        The reader masks each field with its own fill value and valid range; with either missing, fill sentinels would
        pass through as real numbers, or a retrieval could not be told from an out-of-range value.

        Parametrized over each attribute on the geolocation fields, the footprint time, a continuous field, a code
        field and a layered field, each written without it; asserts the error names the field and the attribute.
        """
        omit = {"omit_fill_value": (name,)} if attribute == "_FillValue" else {"omit_valid_range": (name,)}
        path = ceres_ssf_file_factory(**omit)
        with pytest.raises(ValueError, match=f"{name} has no {attribute} attribute"):
            CeresSsfReader([path]).samples_in(_BBOX, _T)


class TestCeresSsfReaderValidRange:
    def test_valid_range_comes_from_the_file(self, ceres_ssf_file_factory):
        """``CeresSsfReader`` range-checks each field with the ``valid_range`` the file declares.

        A hard-coded range that differs from the file silently drops valid retrievals or publishes invalid ones; the
        real Edition1C granule declares the 3.7 µm ice radius [0, 225] µm and water radius [0, 40] µm.

        Writes lower-layer ice radii of 150 µm (inside the file's range) and water radii of 50 µm (outside it) on the
        five clustered footprints, and asserts all five ice radii are served and no water radius is.
        """
        n = 6
        path = ceres_ssf_file_factory(
            cloud_ice_particle_radius_lower=np.full(n, 150.0, dtype=np.float32),
            cloud_water_particle_radius_lower=np.full(n, 50.0, dtype=np.float32),
        )
        data = _values(CeresSsfReader([path]))
        assert np.allclose(_finite_values(data, "Cloud_Ice_Particle_Radius_Lower"), 150.0)
        assert _finite_values(data, "Cloud_Ice_Particle_Radius_Lower").size == 5
        assert _finite_values(data, "Cloud_Water_Particle_Radius_Lower").size == 0

    def test_longitude_valid_range_comes_from_the_file(self, ceres_ssf_file_factory):
        """``CeresSsfReader`` range-checks footprint longitude with the ``valid_range`` the file declares.

        Geolocation is held to the file's own range like every other field, so a footprint outside it is dropped
        rather than placed on the map.

        Writes ``instrument_fov_longitude`` with a ``valid_range`` of [0, 340], which excludes the cluster's 350
        degrees, and asserts no footprint is served in the cluster box.
        """
        path = ceres_ssf_file_factory(longitude_valid_range=(0.0, 340.0))
        assert CeresSsfReader([path]).samples_in(_BBOX, _T).n_samples == 0


class TestCeresSsfReaderLayoutChecks:
    def test_renamed_layer_dimension_raises(self, ceres_ssf_file_factory):
        """``CeresSsfReader`` raises ``ValueError`` naming the field and ``product_version`` when a layer axis is renamed.

        The reader picks a cloud layer by position on the second axis; in a layout whose second axis means something
        else, it would serve the wrong quantity under the layer's name.

        Builds the fixture with the layered fields' second dimension named ``Layers`` and asserts a ``ValueError``
        naming the first layered field read and ``Edition1C``.
        """
        path = ceres_ssf_file_factory(layer_dim_name="Layers")
        with pytest.raises(ValueError, match=r"read as Cloud_Optical_Depth_Lower\) has dimensions.*'Edition1C'"):
            CeresSsfReader([path]).samples_in(_BBOX, _T)

    def test_missing_variable_raises(self, ceres_ssf_file_factory):
        """``CeresSsfReader`` raises ``ValueError``, not ``KeyError``, naming a variable the granule lacks.

        A granule in another layout should fail with a message saying what is missing and what to do, rather than an
        opaque ``KeyError`` from inside the netCDF library.

        Builds the fixture without ``Scene_Type/shortwave_adm_type`` and asserts a ``ValueError`` naming it and
        ``Edition1C``.
        """
        path = ceres_ssf_file_factory(omit_variables=(("Scene_Type", "shortwave_adm_type"),))
        with pytest.raises(ValueError, match=r"no variable Scene_Type/shortwave_adm_type .*'Edition1C'"):
            CeresSsfReader([path]).samples_in(_BBOX, _T)

    def test_missing_group_raises(self, ceres_ssf_file_factory):
        """``CeresSsfReader`` raises ``ValueError``, not ``KeyError``, naming a group the granule lacks.

        A granule without ``Multilayer_Footprint_Area`` should fail with a message naming the missing group, not an
        opaque ``KeyError``, so the reader of the error knows the granule is in a layout the reader does not read.

        Builds the fixture with the group's one variable left out, so the group is not written, and asserts a
        ``ValueError`` naming the group.
        """
        omit = (("Multilayer_Footprint_Area", "cloud_coverage_multilayer"),)
        path = ceres_ssf_file_factory(omit_variables=omit)
        with pytest.raises(ValueError, match="no group 'Multilayer_Footprint_Area'"):
            CeresSsfReader([path]).samples_in(_BBOX, _T)


class TestCeresSsfReaderRealSubset:
    """A real CER_SSF_NOAA20-FM6-VIIRS Edition1C granule (2026-06-11 00 UTC), cut to three blocks of footprints.

    The blocks are spread across the hour (start, middle, end) with their values unchanged; see
    tests/test_data/footprint_matching/README.md. The tests pin the file conventions the reader relies on: the time
    variable and its encoding, the geolocation conventions, and that a query at a footprint's own time serves its pass
    with the values the file holds there.
    """

    @pytest.fixture
    def ssf_path(self, fmatch_real_data_path):
        return fmatch_real_data_path / "CER_SSF_NOAA20-FM6-VIIRS_Edition1C_103103.2026061100_subset.nc"

    def test_every_footprint_has_a_time_and_the_times_span_one_hour(self, ssf_path):
        """``CeresSsfReader._load_granule`` decodes a time for every footprint of a real SSF granule, spanning the hour.

        The pass matching selects SSF footprints by their observation time, so a footprint with an undecoded time would
        never match, and a time variable decoded with the wrong encoding would place the whole granule at the wrong
        hour.

        Loads the real SSF cut and asserts every time is set (no NaT), the times span between 55 min and 1 h, the values
        array holds one row per declared variable and one column per footprint, and every longitude lies in [-180, 180).
        """
        reader = CeresSsfReader([ssf_path])
        points = reader._load_granule(ssf_path)
        assert points.times is not None
        assert np.isnat(points.times).sum() == 0
        span = points.times.max() - points.times.min()
        assert np.timedelta64(55, "m") < span <= np.timedelta64(1, "h")
        assert points.values.shape == (len(CeresSsfReader.VARIABLES), points.lats.size)
        assert np.all((points.lons >= -180.0) & (points.lons < 180.0))

    def test_query_at_a_footprint_time_serves_its_pass(self, ssf_path):
        """``CeresSsfReader.samples_in`` serves a real SSF footprint's own pass when queried at its time and place.

        This is the end-to-end path a Libera footprint takes: if the geolocation or time conventions of the file were
        read wrong, a query at a known SSF footprint would find nothing, or a pass from the wrong time, and the matched
        footprint would carry no or wrong SSF values.

        Takes the first footprint from the middle of the real SSF cut whose layer coverages, clear coverage and
        multilayer coverages are all valid in the file and whose four layer coverages all differ, queries a 1-degree
        box around it at its time and asserts samples are served, all within ``PASS_WINDOW`` of that time; that the
        sample at that footprint serves the lower-cloud and upper-cloud ``layers_coverages``, ``clear_coverage`` and
        the multi-lower and multi-upper ``cloud_coverage_multilayer`` the file stores for it, so a wrong group or
        condition index fails; and that the same box four hours later serves none.
        """
        reader = CeresSsfReader([ssf_path])
        points = reader._load_granule(ssf_path)
        layers, layers_ok = raw_and_usable(ssf_path, "Cloudy_Footprint_Area", "layers_coverages")
        clear, clear_ok = raw_and_usable(ssf_path, "Clear_Footprint_Area", "clear_coverage")
        multilayer, multilayer_ok = raw_and_usable(ssf_path, "Multilayer_Footprint_Area", "cloud_coverage_multilayer")
        distinct = np.array([np.unique(row).size == row.size for row in layers])
        usable = layers_ok.all(axis=1) & distinct & clear_ok & multilayer_ok.all(axis=1) & ~np.isnat(points.times)
        middle = points.lats.size // 2
        i = middle + int(np.flatnonzero(usable[middle:])[0])
        box = BoundingBox(points.lats[i] - 0.5, points.lats[i] + 0.5, points.lons[i] - 0.5, points.lons[i] + 0.5)
        samples = reader.samples_in(box, points.times[i])
        assert samples.n_samples > 0
        assert np.all(np.abs(samples.times - points.times[i]) <= CeresSsfReader.PASS_WINDOW)
        at_i = served_at(samples, points.lats[i], points.lons[i], points.times[i])
        assert at_i.size > 0
        names = [v.name for v in CeresSsfReader.VARIABLES]
        expected = {
            "Layer_Coverage_Lower": layers[i, 1],
            "Layer_Coverage_Upper": layers[i, 2],
            "Clear_Coverage": clear[i],
            "Cloud_Coverage_Multilayer_Lower": multilayer[i, 2],
            "Cloud_Coverage_Multilayer_Upper": multilayer[i, 3],
        }
        for name, value in expected.items():
            assert np.allclose(samples.values[names.index(name), at_i], value), name
        # Four hours away nothing is served.
        assert reader.samples_in(box, points.times[i] + np.timedelta64(4, "h")).n_samples == 0

    def test_granule_span_is_the_granule_date(self, fmatch_real_ancillary_granules):
        """``CeresSsfReader.granule_span`` dates the real granule from its ``time_coverage_start`` / ``time_coverage_end`` attributes: the hour ``2026061100`` in its name.

        The FMATCH runner rejects a granule whose span lies more than a day from the input's footprint times, so a
        misread span would reject the right granule or accept one from the wrong day. Read without constructing the
        reader, as the runner reads it.
        """
        start, end = CeresSsfReader.granule_span(fmatch_real_ancillary_granules["CERES_SSF"])
        assert start == np.datetime64("2026-06-11T00:00:00", "ns")
        assert end == np.datetime64("2026-06-11T00:59:59.999999", "ns")
