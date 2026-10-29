"""Unit tests for the end-to-end test harness in tests/e2e/harness.py."""

import hashlib
import json
import logging
import subprocess
from datetime import date, timedelta

import boto3
import numpy as np
import pytest

from libera_utils.constants import LiberaApid
from libera_utils.l1a.day_coverage import TimeAxisCoverage
from libera_utils.l1a.day_window import trim_l1a_to_day_window
from libera_utils.l1a.l1a_packet_configs import CoverageMode, DayCoveragePolicy
from tests.e2e.harness import (
    GOLDEN_MANIFEST_PATH,
    LogFindings,
    PeakMemorySampler,
    RunReport,
    assert_coverage_meets_policy,
    assert_day_window_invariants,
    golden_apid,
    golden_root_from_env,
    in_window_data_times,
    load_golden_manifest,
    output_dir_from_env,
    run_commit,
    run_prefix,
    stage_golden_inputs,
    upload_run_outputs,
)
from tests.e2e.upload_outputs import main as upload_outputs_main
from tests.e2e.upload_outputs import report_commit
from tests.unit.test_l1a.test_day_window import DAY, WINDOW_END, WINDOW_START, _packet_dataset, _sample_dataset

KEY_A = "l1a/ditl2/LIBERA_SDC_1057_ccsds_2026_193_00_00_00"
KEY_B = "l1a/ditl2/LIBERA_SDC_1057_ccsds_2026_193_02_00_00"
SECOND = timedelta(seconds=1)


def _write_manifest(path, rows, header="key,sha256"):
    path.write_text("\n".join([header, *(f"{key},{sha}" for key, sha in rows)]) + "\n")
    return path


def _golden_root(root, contents):
    """Write ``contents`` (bytes by key) under ``root`` and return the matching manifest."""
    for key, data in contents.items():
        (root / key).parent.mkdir(parents=True, exist_ok=True)
        (root / key).write_bytes(data)
    return {key: hashlib.sha256(data).hexdigest() for key, data in contents.items()}


def test_committed_manifest_holds_the_golden_day():
    manifest = load_golden_manifest(GOLDEN_MANIFEST_PATH)
    apids = {golden_apid(key) for key in manifest}
    assert len(manifest) == 116
    assert apids == {
        LiberaApid.pev_sw_stat,
        LiberaApid.pec_sw_stat,
        LiberaApid.icie_rad_sample,
        LiberaApid.icie_wfov_sci,
        LiberaApid.icie_axis_sample,
        LiberaApid.icie_crit_hk,
        LiberaApid.icie_nom_hk,
        LiberaApid.icie_temp_hk,
    }


@pytest.mark.parametrize(
    ("header", "rows", "match"),
    [
        ("name,digest", [(KEY_A, "0" * 64)], "header"),
        ("key,sha256", [(KEY_A, "0" * 64), (KEY_A, "1" * 64)], "twice"),
        ("key,sha256", [("l1a/ditl2/LIBERA_SDC_1017_ccsds_2026_193_00_00_00", "0" * 64)], "no L1A product definition"),
    ],
    ids=["bad_header", "repeated_key", "apid_without_definition"],
)
def test_load_golden_manifest_rejects(tmp_path, header, rows, match):
    with pytest.raises(ValueError, match=match):
        load_golden_manifest(_write_manifest(tmp_path / "manifest.csv", rows, header))


def test_stage_local_root_reads_in_place(tmp_path):
    manifest = _golden_root(tmp_path, {KEY_A: b"a", KEY_B: b"b"})
    staged = stage_golden_inputs(str(tmp_path), manifest, tmp_path / "staging")
    assert staged == {KEY_A: tmp_path / KEY_A, KEY_B: tmp_path / KEY_B}
    assert not (tmp_path / "staging").exists()


def test_stage_names_every_missing_object(tmp_path):
    manifest = {KEY_A: "0" * 64, KEY_B: "0" * 64}
    with pytest.raises(FileNotFoundError, match=f"2 golden input .*{KEY_A}, {KEY_B}") as excinfo:
        stage_golden_inputs(str(tmp_path), manifest, tmp_path / "staging")
    assert str(tmp_path) not in str(excinfo.value)


def test_stage_names_an_object_that_differs_from_the_manifest(tmp_path):
    manifest = _golden_root(tmp_path, {KEY_A: b"a", KEY_B: b"b"})
    (tmp_path / KEY_B).write_bytes(b"changed")
    with pytest.raises(ValueError, match=f"1 golden input .*{KEY_B}"):
        stage_golden_inputs(str(tmp_path), manifest, tmp_path / "staging")


def test_stage_copies_s3_objects(tmp_path, write_file_to_s3):
    source = tmp_path / "source"
    manifest = _golden_root(source, {KEY_A: b"a", KEY_B: b"b"})
    for key in manifest:
        write_file_to_s3(source / key, f"s3://golden-bucket/root/{key}")

    staged = stage_golden_inputs("s3://golden-bucket/root", manifest, tmp_path / "staging")
    assert staged == {KEY_A: tmp_path / "staging" / KEY_A, KEY_B: tmp_path / "staging" / KEY_B}
    assert staged[KEY_A].read_bytes() == b"a"
    assert staged[KEY_B].read_bytes() == b"b"


def test_stage_names_a_missing_s3_object(tmp_path, write_file_to_s3):
    source = tmp_path / "source"
    manifest = _golden_root(source, {KEY_A: b"a", KEY_B: b"b"})
    write_file_to_s3(source / KEY_A, f"s3://golden-bucket/root/{KEY_A}")
    with pytest.raises(FileNotFoundError, match=f"1 golden input .*{KEY_B}") as excinfo:
        stage_golden_inputs("s3://golden-bucket/root", manifest, tmp_path / "staging")
    assert "golden-bucket" not in str(excinfo.value)


def test_stage_names_an_s3_object_that_differs_from_the_manifest(tmp_path, write_file_to_s3):
    source = tmp_path / "source"
    manifest = _golden_root(source, {KEY_A: b"a", KEY_B: b"b"})
    (source / KEY_B).write_bytes(b"changed")
    for key in manifest:
        write_file_to_s3(source / key, f"s3://golden-bucket/root/{key}")
    with pytest.raises(ValueError, match=f"1 golden input .*{KEY_B}"):
        stage_golden_inputs("s3://golden-bucket/root", manifest, tmp_path / "staging")


@pytest.mark.parametrize(
    ("environ", "error", "match"),
    [
        ({}, ValueError, "LIBERA_E2E_GOLDEN_ROOT is not set"),
        ({"LIBERA_E2E_GOLDEN_ROOT": ""}, ValueError, "LIBERA_E2E_GOLDEN_ROOT is not set"),
        ({"LIBERA_E2E_GOLDEN_ROOT": "/golden", "AWS_ACCESS_KEY_ID": "testing"}, RuntimeError, "on their own"),
    ],
    ids=["unset", "empty", "mock_credentials"],
)
def test_golden_root_from_env_rejects(environ, error, match):
    with pytest.raises(error, match=match):
        golden_root_from_env(environ)


def test_golden_root_from_env():
    assert golden_root_from_env({"LIBERA_E2E_GOLDEN_ROOT": "s3://bucket/root"}) == "s3://bucket/root"


@pytest.mark.parametrize(
    ("environ", "match"),
    [({}, "is not set"), ({"LIBERA_E2E_OUTPUT_DIR": "s3://bucket/out"}, "is an S3 URI")],
    ids=["unset", "s3"],
)
def test_output_dir_from_env_rejects(environ, match):
    with pytest.raises(ValueError, match=match):
        output_dir_from_env(environ)


def test_output_dir_from_env_rejects_a_directory_with_outputs(tmp_path):
    (tmp_path / "logs").mkdir()
    with pytest.raises(FileExistsError, match="already holds a run's outputs"):
        output_dir_from_env({"LIBERA_E2E_OUTPUT_DIR": str(tmp_path)})


@pytest.mark.parametrize("make_dir", [True, False], ids=["empty", "absent"])
def test_output_dir_from_env(tmp_path, make_dir):
    output = tmp_path / "out"
    if make_dir:
        output.mkdir()
    assert output_dir_from_env({"LIBERA_E2E_OUTPUT_DIR": str(output)}) == output


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def test_run_commit_prefers_github_sha(tmp_path):
    assert run_commit({"GITHUB_SHA": "abc123"}, tmp_path) == "abc123"


def test_run_commit_reads_the_local_head(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(
        tmp_path,
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@t",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-q",
        "--allow-empty",
        "-m",
        "init",
    )
    head = _git(tmp_path, "rev-parse", "HEAD")
    assert run_commit({}, tmp_path) == head
    (tmp_path / "untracked.txt").write_text("change")
    assert run_commit({}, tmp_path) == f"{head}-dirty"


def test_run_commit_outside_a_repository(tmp_path):
    assert run_commit({}, tmp_path) is None


def test_upload_run_outputs_keeps_kinds_under_the_run_and_tags_them(tmp_path, create_mock_bucket):
    for relative in ("products/a.nc", "logs/icie_nom_hk.log", "reports/run_report.json", "other/ignored.txt"):
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_text(relative)
    create_mock_bucket("e2e-bucket")

    uploaded = upload_run_outputs(tmp_path, "s3://e2e-bucket/test_run_output", "abc123/42-1")

    assert uploaded == [
        "abc123/42-1/products/a.nc",
        "abc123/42-1/logs/icie_nom_hk.log",
        "abc123/42-1/reports/run_report.json",
    ]
    s3 = boto3.client("s3")
    for relative in uploaded:
        key = f"test_run_output/{relative}"
        tags = s3.get_object_tagging(Bucket="e2e-bucket", Key=key)["TagSet"]
        assert tags == [{"Key": "kind", "Value": relative.split("/")[2]}]


def test_upload_run_outputs_at_the_bucket_root(tmp_path, create_mock_bucket):
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "run_report.json").write_text("{}")
    create_mock_bucket("e2e-bucket")
    upload_run_outputs(tmp_path, "s3://e2e-bucket", "abc123/42-1")
    assert boto3.client("s3").head_object(Bucket="e2e-bucket", Key="abc123/42-1/reports/run_report.json")


def test_upload_run_outputs_rejects_a_local_root(tmp_path):
    with pytest.raises(ValueError, match="not an s3:// URI"):
        upload_run_outputs(tmp_path, str(tmp_path), "abc123/42-1")


def _report(output_dir, commit):
    report = output_dir / "reports" / "run_report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"header": {"commit": commit}, "apids": {}}))


def test_run_prefix_leads_with_the_repository():
    assert run_prefix("abc123", "42-1") == "libera_utils/abc123/42-1"


def test_report_commit(tmp_path):
    _report(tmp_path, "abc123-dirty")
    assert report_commit(tmp_path) == "abc123-dirty"


def test_report_commit_rejects_a_missing_report(tmp_path):
    with pytest.raises(FileNotFoundError, match="No run report"):
        report_commit(tmp_path)


def test_report_commit_rejects_a_report_with_no_commit(tmp_path):
    _report(tmp_path, None)
    with pytest.raises(ValueError, match="records no commit"):
        report_commit(tmp_path)


@pytest.mark.parametrize(
    ("extra_args", "run_id"), [([], "local-run_7"), (["--run-id", "42-1"], "42-1")], ids=["local", "ci"]
)
def test_upload_outputs_files_the_run_under_repository_and_commit(tmp_path, create_mock_bucket, extra_args, run_id):
    output_dir = tmp_path / "run_7"
    _report(output_dir, "abc123")
    create_mock_bucket("e2e-bucket")
    upload_outputs_main([str(output_dir), "s3://e2e-bucket/test_run_output", *extra_args])
    key = f"test_run_output/libera_utils/abc123/{run_id}/reports/run_report.json"
    assert boto3.client("s3").head_object(Bucket="e2e-bucket", Key=key)


def test_peak_memory_sampler_sees_an_allocation():
    with PeakMemorySampler() as baseline:
        pass
    # The allocation is still held when the sampler exits, and exit takes a sample.
    with PeakMemorySampler() as sampler:
        block = np.ones(200 * 2**20 // 8)
    del block
    assert sampler.peak_bytes - baseline.peak_bytes > 150 * 2**20


def test_run_report_rewrites_on_every_record(tmp_path):
    path = tmp_path / "reports" / "run_report.json"
    report = RunReport(path, {"golden_day": date(2026, 7, 12), "absent_apids": ["jpss_sc_pos"]})
    assert json.loads(path.read_text())["apids"] == {}

    coverage = TimeAxisCoverage(1.0, 0.5, 0.25, timedelta(seconds=1), timedelta(seconds=3), 86400)
    report.record(LiberaApid.icie_nom_hk, {"status": "passed", "data_time_coverage": coverage, "total_s": 1.5})
    written = json.loads(path.read_text())
    assert written["header"]["golden_day"] == "2026-07-12"
    assert written["apids"]["icie_nom_hk"]["data_time_coverage"]["max_gap"] == 3.0


def test_run_report_rejects_a_value_json_cannot_hold(tmp_path):
    report = RunReport(tmp_path / "run_report.json", {})
    with pytest.raises(TypeError, match="set is not JSON serializable"):
        report.record(LiberaApid.icie_nom_hk, {"status": {"passed"}})


def test_run_report_step_summary(tmp_path):
    report = RunReport(tmp_path / "run_report.json", {"golden_day": date(2026, 7, 12), "absent_apids": ["jpss_sc_pos"]})
    report.record(
        LiberaApid.icie_nom_hk,
        {"status": "passed", "packets_in": 10, "packets_out": 8, "peak_memory_bytes": 2**30, "total_s": 1.5},
    )
    summary = tmp_path / "summary.md"
    summary.write_text("earlier step\n")
    report.write_step_summary(summary)
    text = summary.read_text()
    assert text.startswith("earlier step\n")
    assert "absent from the golden input: jpss_sc_pos" in text
    assert "| icie_nom_hk | passed | 10 | 8 | 1.00 | 1.5 |" in text


def test_log_findings_collects_warnings_and_order_info(caplog):
    caplog.set_level(logging.INFO, logger="libera_utils")
    with LogFindings() as findings:
        logging.getLogger("libera_utils.l1a.packets").warning("Duplicate coordinates detected (%d)", 2)
        logging.getLogger("libera_utils.l1a.packets").info("Parsing packets")
        logging.getLogger("libera_utils.l1a.packet_ordering").info("Packet order on 'PACKET_ICIE_TIME'")
        logging.getLogger("space_packet_parser").warning("Not a libera_utils logger")
    logging.getLogger("libera_utils.l1a.packets").warning("After the collector closed")
    assert findings.findings == [
        {"level": "WARNING", "logger": "libera_utils.l1a.packets", "message": "Duplicate coordinates detected (2)"},
        {
            "level": "INFO",
            "logger": "libera_utils.l1a.packet_ordering",
            "message": "Packet order on 'PACKET_ICIE_TIME'",
        },
    ]


def test_run_report_step_summary_lists_log_findings(tmp_path):
    report = RunReport(tmp_path / "run_report.json", {})
    findings = [
        {"level": "WARNING", "logger": "libera_utils.l1a.packets", "message": "Duplicates ['a'\n 'b']"},
        {"level": "INFO", "logger": "libera_utils.l1a.packet_ordering", "message": "Packet order"},
    ]
    report.record(LiberaApid.icie_nom_hk, {"status": "passed", "log_findings": findings})
    summary = tmp_path / "summary.md"
    report.write_step_summary(summary)
    text = summary.read_text()
    assert "| passed |  |  |  |  |  | 1 |" in text
    assert "- WARNING `libera_utils.l1a.packets`: Duplicates ['a' 'b']" in text
    assert "- INFO `libera_utils.l1a.packet_ordering`: Packet order" in text


FRACTION_POLICY = DayCoveragePolicy(mode=CoverageMode.FRACTION, day_coverage_frac=0.9, buffer_coverage_frac=0.6)
PRESENCE_POLICY = DayCoveragePolicy(mode=CoverageMode.PRESENCE)


def _coverage(day_frac=1.0, left_frac=1.0, right_frac=1.0, n_times=100):
    return TimeAxisCoverage(day_frac, left_frac, right_frac, SECOND, SECOND, n_times)


@pytest.mark.parametrize(
    ("coverage", "policy"),
    [
        (_coverage(day_frac=0.9, left_frac=0.6, right_frac=0.6), FRACTION_POLICY),
        (_coverage(day_frac=0.0, left_frac=0.0, right_frac=0.0, n_times=1), PRESENCE_POLICY),
    ],
    ids=["fraction_at_thresholds", "presence"],
)
def test_coverage_meets_policy(coverage, policy):
    assert_coverage_meets_policy(coverage, policy)


@pytest.mark.parametrize(
    ("coverage", "policy", "match"),
    [
        (_coverage(day_frac=0.72), FRACTION_POLICY, r"day core 0\.7200 < 0\.9"),
        (_coverage(right_frac=0.0), FRACTION_POLICY, r"right buffer 0\.0000 < 0\.6"),
        (_coverage(left_frac=0.5, right_frac=0.5), FRACTION_POLICY, "left buffer .*; right buffer"),
        (_coverage(n_times=0), PRESENCE_POLICY, "requires presence"),
    ],
    ids=["short_day", "short_right_buffer", "both_buffers_named", "presence_empty"],
)
def test_coverage_short_of_policy(coverage, policy, match):
    with pytest.raises(AssertionError, match=match):
        assert_coverage_meets_policy(coverage, policy)


def _packet_times():
    """Packets from one minute before the window to one minute after it, one every five minutes."""
    start = WINDOW_START - timedelta(minutes=1)
    return [start + timedelta(minutes=5 * i) for i in range(int((WINDOW_END - start) / timedelta(minutes=5)) + 2)]


def test_invariants_hold_for_a_trimmed_day():
    parsed = _packet_dataset(_packet_times())
    assert_day_window_invariants(trim_l1a_to_day_window(parsed, day=DAY), in_window_data_times(parsed, DAY), DAY)


def test_invariants_reject_a_repeated_data_time():
    times = _packet_times()
    times[5] = times[4]
    parsed = _packet_dataset(times)
    with pytest.raises(AssertionError, match="not unique"):
        assert_day_window_invariants(trim_l1a_to_day_window(parsed, day=DAY), in_window_data_times(parsed, DAY), DAY)


def test_invariants_reject_a_lost_in_window_packet():
    parsed = _packet_dataset(_packet_times())
    trimmed = trim_l1a_to_day_window(parsed, day=DAY)
    with pytest.raises(AssertionError, match="1 lost"):
        assert_day_window_invariants(trimmed.drop_isel(PACKET=[3]), in_window_data_times(parsed, DAY), DAY)


def test_invariants_reject_a_packet_outside_the_window():
    parsed = _packet_dataset(_packet_times())
    with pytest.raises(AssertionError, match="2 packet"):
        assert_day_window_invariants(parsed, in_window_data_times(parsed, DAY), DAY)


def test_invariants_keep_a_packet_straddling_the_window_start():
    packet_times = [WINDOW_START - SECOND, WINDOW_START + timedelta(hours=1)]
    samples = [WINDOW_START - 2 * SECOND, WINDOW_START + SECOND, WINDOW_START + timedelta(hours=1)]
    parsed = _sample_dataset({"RAD_SAMPLE": (samples, [0, 0, 1])}, packet_times)
    trimmed = trim_l1a_to_day_window(parsed, day=DAY)
    assert trimmed.sizes["PACKET"] == 2
    assert_day_window_invariants(trimmed, in_window_data_times(parsed, DAY), DAY)


def test_invariants_reject_axes_that_differ_from_the_input():
    parsed = _packet_dataset(_packet_times())
    with pytest.raises(AssertionError, match="differ from the input's"):
        assert_day_window_invariants(parsed, {"RAD_SAMPLE_TIME": np.array([])}, DAY)


def test_invariants_reject_a_product_with_no_data_time_axis():
    empty = _packet_dataset([])
    with pytest.raises(AssertionError, match="no data-time axis"):
        assert_day_window_invariants(empty, {}, DAY)
