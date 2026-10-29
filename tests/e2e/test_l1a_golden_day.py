"""Full-day L1A processing of the DITL2 golden day, one daily product per L1A APID it holds.

Run with ``pytest -m e2e tests/e2e``; the developer testing docs describe the environment it needs.
"""

from time import perf_counter

import pytest
import xarray as xr

from libera_utils.constants import LiberaApid
from libera_utils.io.netcdf import write_libera_data_product
from libera_utils.l1a.day_coverage import measure_time_axis_coverage
from libera_utils.l1a.day_window import check_data_times, trim_l1a_to_day_window
from libera_utils.l1a.ground_ccsds import GROUND_CCSDS_SKIP_HEADER_BYTES
from libera_utils.l1a.l1a_packet_configs import get_l1a_product_definition_path, get_packet_config
from libera_utils.l1a.packet_slicing import PACKET_DIM, data_time_axes, packet_time_coordinate
from libera_utils.l1a.packets import parse_packets_to_l1a_dataset
from tests.e2e.harness import (
    GOLDEN_DAY,
    GOLDEN_MANIFEST_PATH,
    LogFindings,
    PeakMemorySampler,
    assert_coverage_meets_policy,
    assert_day_window_invariants,
    golden_apid,
    in_window_data_times,
    load_golden_manifest,
)

pytestmark = pytest.mark.e2e

# One worker parses in-process, so the sampled peak is the whole parse's memory.
PARSE_MAX_WORKERS = 1

GOLDEN_APIDS = sorted({golden_apid(key) for key in load_golden_manifest(GOLDEN_MANIFEST_PATH)})


@pytest.mark.parametrize("apid", GOLDEN_APIDS, ids=lambda apid: apid.name)
def test_l1a_golden_day(apid: LiberaApid, golden_inputs, e2e_output_dir, e2e_task_log, run_report):
    packet_files = sorted(path for key, path in golden_inputs.items() if golden_apid(key) == apid)
    entry = {"status": "failed", "input_files": len(packet_files)}
    memory = PeakMemorySampler()
    log_findings = LogFindings()
    run_start = perf_counter()
    try:
        with memory, log_findings:
            start = perf_counter()
            parsed = parse_packets_to_l1a_dataset(
                packet_files,
                apid,
                ground_data=True,
                skip_header_bytes=GROUND_CCSDS_SKIP_HEADER_BYTES,
                max_workers=PARSE_MAX_WORKERS,
            )
            entry["parse_s"] = perf_counter() - start
            entry["packets_in"] = parsed.sizes[PACKET_DIM]
            expected_in_window = in_window_data_times(parsed, GOLDEN_DAY)

            start = perf_counter()
            day = trim_l1a_to_day_window(parsed, day=GOLDEN_DAY)
            del parsed
            entry["trim_s"] = perf_counter() - start
            entry["packets_out"] = day.sizes[PACKET_DIM]
            check_data_times(day)

            axes = data_time_axes(day)
            assert len(axes) == 1, f"Expected one data-time axis to name the product by, found {axes}"
            time_variable = axes[0]
            packet_time = packet_time_coordinate(day)
            entry["data_time_axis"] = time_variable
            entry["data_time_coverage"] = measure_time_axis_coverage(day[time_variable].values, day=GOLDEN_DAY)
            if packet_time != time_variable:
                entry["packet_time_coverage"] = measure_time_axis_coverage(day[packet_time].values, day=GOLDEN_DAY)

            start = perf_counter()
            written = write_libera_data_product(
                get_l1a_product_definition_path(apid),
                day,
                e2e_output_dir / "products",
                time_variable=time_variable,
                strict=True,
            )
            del day
            entry["write_s"] = perf_counter() - start
            entry["product"] = written.path.name

        with xr.open_dataset(written.path) as product:
            assert_day_window_invariants(product, expected_in_window, GOLDEN_DAY)
        assert_coverage_meets_policy(entry["data_time_coverage"], get_packet_config(apid).day_coverage)
        entry["status"] = "passed"
    except Exception as err:
        entry["error"] = f"{type(err).__name__}: {err}"
        raise
    finally:
        entry["peak_memory_bytes"] = memory.peak_bytes
        entry["log_findings"] = log_findings.findings
        entry["total_s"] = round(perf_counter() - run_start, 1)
        run_report.record(apid, entry)
