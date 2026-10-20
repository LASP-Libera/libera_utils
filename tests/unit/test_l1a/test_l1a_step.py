"""Tests for the L1A processing step (libera_utils.l1a.l1a_step).

The L0 input is the demuxed ground-test fixture ``LIBERA_SDC_1057_ccsds_2025_318_13_00_00`` (named by the ground
convention, not the L0 PDS one): 55 NOM-HK packets with packet times on 2028-02-15 13:16-13:17 UTC.
"""

from datetime import UTC, date, datetime, time, timedelta

import pytest

from libera_utils.io.filenaming import LiberaDataProductFilename
from libera_utils.io.manifest import Manifest, ManifestType
from libera_utils.l1a.l1a_step import run_l1a_step, step_apid

DAY = date(2028, 2, 15)
NOM_HK_FIXTURE = "LIBERA_SDC_1057_ccsds_2025_318_13_00_00"
FIXTURE_SPAN = ["2028-02-15T13:16:32+00:00", "2028-02-15T13:17:27+00:00"]
WHOLE_BUFFERED_DAY = ["2028-02-14T23:00:00+00:00", "2028-02-16T01:00:00+00:00"]


def _manifest(tmp_path, files, *, force: bool, spans: list[str]) -> Manifest:
    """Write an input manifest as the step's Manifest Creator does, and return its path."""
    manifest = Manifest(manifest_type=ManifestType.INPUT, files=[], configuration={})
    start = datetime.combine(DAY, time.min, tzinfo=UTC)
    manifest.add_desired_time_range(start_datetime=start, end_datetime=start + timedelta(days=1) - timedelta(seconds=1))
    manifest.add_files(*files)
    manifest.configuration["force"] = force
    manifest.configuration["l0_spans"] = {file.name: spans for file in files}
    manifest.write(tmp_path)
    return manifest.filename.path


@pytest.mark.parametrize(
    ("force", "spans"),
    [
        pytest.param(True, FIXTURE_SPAN, id="forced"),
        pytest.param(False, WHOLE_BUFFERED_DAY, id="coverage-gate-passes"),
    ],
)
def test_run_l1a_step_writes_the_days_product(tmp_path, test_ditl_demux_path, force, spans):
    """The step decodes the day's L0 into the APID's L1A product and lists it in the output manifest."""
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    manifest_path = _manifest(tmp_path, [test_ditl_demux_path / NOM_HK_FIXTURE], force=force, spans=spans)

    output = run_l1a_step(manifest_path, "l1a-nom-hk", output_dir)

    (entry,) = output.files
    written = LiberaDataProductFilename(entry.filename)
    assert written.data_product_id.value == "NOM-HK-DECODED"
    assert written.applicable_date == DAY
    assert output.manifest_type == ManifestType.OUTPUT
    assert output.filename.path.parent == output_dir


def test_run_l1a_step_fails_coverage_gate_without_force(tmp_path, test_ditl_demux_path):
    """Without force, a day whose files fail the APID's coverage gate is not combined."""
    manifest_path = _manifest(tmp_path, [test_ditl_demux_path / NOM_HK_FIXTURE], force=False, spans=FIXTURE_SPAN)

    with pytest.raises(ValueError, match="fails its coverage gate without force"):
        run_l1a_step(manifest_path, "l1a-nom-hk", tmp_path)


def test_run_l1a_step_refuses_ground_and_flight_together(tmp_path, test_ditl_demux_path):
    """A day of L1A comes from ground-test captures or from flight L0 files, not both."""
    flight = tmp_path / "P1571057AAAAAAAAAAAAAA28046000000001.PDS"
    flight.write_bytes(b"flight packets")
    manifest_path = _manifest(tmp_path, [test_ditl_demux_path / NOM_HK_FIXTURE, flight], force=True, spans=FIXTURE_SPAN)

    with pytest.raises(ValueError, match="mixes ground-test captures and flight L0 files"):
        run_l1a_step(manifest_path, "l1a-nom-hk", tmp_path)


@pytest.mark.parametrize(
    ("step_name", "apid"),
    [("l1a-rad-sample", 1036), ("l1a-wfov-sci", 1040), ("l1a-axis-sample", 1048), ("l1a-nom-hk", 1057)],
)
def test_step_apid(step_name, apid):
    """Each L1A step decodes the APID of its product."""
    assert step_apid(step_name) == apid


def test_step_apid_rejects_a_step_without_an_apid():
    """A step whose product is not an APID's L1A product is not an L1A step."""
    with pytest.raises(ValueError, match="does not produce an L1A product of an APID"):
        step_apid("spice-jpss")
