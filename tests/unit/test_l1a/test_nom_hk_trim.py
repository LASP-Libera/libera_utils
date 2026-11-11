"""Unit tests for libera_utils.l1a.nom_hk_trim."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from libera_utils.constants import DataProductIdentifier
from libera_utils.io.product_definition import DATE_CREATED_ATTRIBUTE, LiberaDataProductDefinition
from libera_utils.l1a.nom_hk_trim import (
    RUN_END_BEFORE_GAP_ATTRIBUTE,
    RUN_MISSING_PACKETS_ATTRIBUTE,
    RUN_START_AFTER_GAP_ATTRIBUTE,
    find_obsid_runs,
    get_trimmed_nom_hk_product_definition,
    write_trimmed_nom_hk_products,
)
from libera_utils.l1a.packet_ordering import SEQUENCE_COUNTER_MODULUS
from libera_utils.obsids import NomHkObsidSource
from libera_utils.version import version as libera_utils_version

_DAY = date(2028, 2, 13)

_PARENT_GRANULE = (
    "/some/dir/LIBERA_L1A_NOM-HK-DECODED_V5-10-0_20280213T020000_20280213T040000_R20280213T120000000000.nc"
)


def _synthetic_nom_hk(
    rad_obsids: list[int],
    wfov_obsids: list[int] | None = None,
    *,
    start: str = "2028-02-13T02:00:00",
    sequence_counter: list[int] | None = None,
) -> xr.Dataset:
    """Build a minimal NOM-HK-shaped Dataset for run-detection unit tests."""
    n = len(rad_obsids)
    if wfov_obsids is None:
        wfov_obsids = [0] * n
    assert len(wfov_obsids) == n
    if sequence_counter is None:
        sequence_counter = list(range(100, 100 + n))
    times = np.datetime64(start) + np.arange(n) * np.timedelta64(1, "s")
    return xr.Dataset(
        {
            "ICIE__SW_OBSID_RAD": ("PACKET", np.asarray(rad_obsids, dtype=np.uint16)),
            "ICIE__SW_OBSID_WFOV": ("PACKET", np.asarray(wfov_obsids, dtype=np.uint16)),
            "SRC_SEQ_CTR": ("PACKET", np.asarray(sequence_counter, dtype=np.uint16)),
        },
        coords={"PACKET_ICIE_TIME": ("PACKET", times)},
        attrs={
            "ProductID": "NOM-HK-DECODED",
            "algorithm_version": "5.10.0",
            "input_files": ["fake.bin"],
            "date_created": "2028-01-01T00:00:00+00:00",
        },
    )


class TestGetTrimmedNomHkProductDefinition:
    """Product-definition ProductID override."""

    def test_overrides_product_id(self):
        definition = get_trimmed_nom_hk_product_definition(DataProductIdentifier.l1a_icie_nom_hk_swc_365nm_trimmed)
        assert definition.attributes["ProductID"] == "NOM-HK-SWC-365NM-TRIMMED"

    def test_declares_the_gap_flags_as_dynamic_attributes(self):
        definition = get_trimmed_nom_hk_product_definition(DataProductIdentifier.l1a_icie_nom_hk_swc_365nm_trimmed)
        assert definition.attributes[RUN_START_AFTER_GAP_ATTRIBUTE] is None
        assert definition.attributes[RUN_END_BEFORE_GAP_ATTRIBUTE] is None
        assert definition.attributes[RUN_MISSING_PACKETS_ATTRIBUTE] is None

    def test_rejects_non_trimmed_product(self):
        with pytest.raises(ValueError, match="is not a TRIMMED NOM-HK product in the ObsID registry"):
            get_trimmed_nom_hk_product_definition(DataProductIdentifier.l1a_icie_nom_hk_decoded)

    def test_callers_get_independent_definitions(self):
        """Stamping one ProductID must not leak into another's definition."""
        swc = get_trimmed_nom_hk_product_definition(DataProductIdentifier.l1a_icie_nom_hk_swc_365nm_trimmed)
        gain = get_trimmed_nom_hk_product_definition(DataProductIdentifier.l1a_icie_nom_hk_gain_trimmed)
        assert swc is not gain
        assert swc.attributes["ProductID"] == "NOM-HK-SWC-365NM-TRIMMED"
        assert gain.attributes["ProductID"] == "NOM-HK-GAIN-TRIMMED"


class TestFindObsidRuns:
    """Contiguous ObsID run detection."""

    def test_rad_and_wfov_in_same_dataset(self):
        # RAD: 257 run, then pad, then 385; WFOV: 256 (darks) in the middle
        ds = _synthetic_nom_hk(
            rad_obsids=[128, 257, 257, 257, 128, 385, 385, 128],
            wfov_obsids=[0, 0, 0, 0, 256, 256, 0, 0],
        )
        runs = find_obsid_runs(ds)
        by_key = {(spec.source, spec.obsid): pkt_slice for spec, pkt_slice in runs}
        assert (NomHkObsidSource.RAD, 257) in by_key
        assert (NomHkObsidSource.RAD, 385) in by_key
        assert (NomHkObsidSource.WFOV, 256) in by_key
        assert by_key[(NomHkObsidSource.RAD, 257)] == slice(1, 4)
        assert by_key[(NomHkObsidSource.RAD, 385)] == slice(5, 7)
        assert by_key[(NomHkObsidSource.WFOV, 256)] == slice(4, 6)

    def test_science_obsid_not_returned(self):
        ds = _synthetic_nom_hk(rad_obsids=[136, 136, 136], wfov_obsids=[136, 136, 136])
        assert find_obsid_runs(ds) == []

    def test_rad_vs_wfov_256_are_distinct(self):
        ds = _synthetic_nom_hk(
            rad_obsids=[256, 256, 128],
            wfov_obsids=[256, 256, 0],
        )
        runs = find_obsid_runs(ds)
        products = {spec.trimmed_product for spec, _ in runs}
        assert DataProductIdentifier.l1a_icie_nom_hk_swc_365nm_trimmed in products
        assert DataProductIdentifier.l1a_icie_nom_hk_darks_of_darks_trimmed in products

    def test_source_filter(self):
        ds = _synthetic_nom_hk(
            rad_obsids=[257, 257],
            wfov_obsids=[129, 129],
        )
        rad_only = find_obsid_runs(ds, source=NomHkObsidSource.RAD)
        assert len(rad_only) == 1
        assert rad_only[0][0].obsid == 257

    def test_backward_packet_time_keeps_one_run(self):
        """The counter advances by one across a backward packet-time step, so the run stays one slice."""
        ds = _synthetic_nom_hk(rad_obsids=[257, 257, 257])
        times = ds["PACKET_ICIE_TIME"].values.copy()
        times[1] = times[0] - np.timedelta64(2, "s")
        inverted = ds.assign_coords(PACKET_ICIE_TIME=("PACKET", times))
        runs = find_obsid_runs(inverted)
        assert len(runs) == 1
        assert runs[0][1] == slice(0, 3)

    def test_backward_sequence_counter_raises(self):
        ds = _synthetic_nom_hk(rad_obsids=[257, 257, 257], sequence_counter=[100, 102, 101])
        with pytest.raises(ValueError, match="SRC_SEQ_CTR 102 -> 101") as excinfo:
            find_obsid_runs(ds)
        assert "Decode its L0 files again" in str(excinfo.value)

    def test_missing_packet_time_raises(self):
        ds = _synthetic_nom_hk(rad_obsids=[257, 257]).drop_vars("PACKET_ICIE_TIME")
        with pytest.raises(ValueError, match="PACKET_ICIE_TIME"):
            find_obsid_runs(ds)


class TestWriteTrimmedNomHkProducts:
    """High-level writer orchestration (write_libera_data_product mocked)."""

    def test_recurring_obsid_writes_each_run_without_warning(self, tmp_path, caplog):
        """An ObsID may recur within a day; each run is its own file, and recurrence is not warned about."""
        ds = _synthetic_nom_hk(rad_obsids=[128, 257, 257, 128, 257, 257, 128])
        mock_filename = MagicMock()
        mock_filename.path.name = "LIBERA_L1A_NOM-HK-SWC-405NM-TRIMMED_fake.nc"

        with (
            patch(
                "libera_utils.l1a.nom_hk_trim.write_libera_data_product",
                return_value=mock_filename,
            ) as mock_write,
            caplog.at_level("WARNING"),
        ):
            written = write_trimmed_nom_hk_products(
                ds, tmp_path, day=_DAY, source_product_filename=_PARENT_GRANULE, strict=False
            )

        assert len(written) == 2
        assert mock_write.call_count == 2
        assert not [r for r in caplog.records if r.levelname == "WARNING"]

    def test_writes_distinct_products_for_rad_and_wfov(self, tmp_path):
        ds = _synthetic_nom_hk(
            rad_obsids=[128, 257, 257, 128],
            wfov_obsids=[0, 256, 256, 0],
        )
        mock_filename = MagicMock()
        mock_filename.path.name = "fake.nc"
        with patch(
            "libera_utils.l1a.nom_hk_trim.write_libera_data_product",
            return_value=mock_filename,
        ) as mock_write:
            write_trimmed_nom_hk_products(ds, tmp_path, day=_DAY, source_product_filename=_PARENT_GRANULE, strict=False)

        product_ids = {call.args[1].attrs["ProductID"] for call in mock_write.call_args_list}
        assert product_ids == {"NOM-HK-SWC-405NM-TRIMMED", "NOM-HK-DARKS-OF-DARKS-TRIMMED"}

    def test_two_obsids_get_two_product_ids(self, tmp_path):
        """Two ObsIDs whose combiners take the same inputs still write their own ProductIDs."""
        ds = _synthetic_nom_hk(rad_obsids=[128, 256, 256, 128, 257, 257, 128])
        mock_filename = MagicMock()
        mock_filename.path.name = "fake.nc"
        with patch(
            "libera_utils.l1a.nom_hk_trim.write_libera_data_product",
            return_value=mock_filename,
        ) as mock_write:
            written = write_trimmed_nom_hk_products(
                ds, tmp_path, day=_DAY, source_product_filename=_PARENT_GRANULE, strict=False
            )

        assert len(written) == 2
        product_ids = [call.args[1].attrs["ProductID"] for call in mock_write.call_args_list]
        assert product_ids == ["NOM-HK-SWC-365NM-TRIMMED", "NOM-HK-SWC-405NM-TRIMMED"]
        # Each file still covers exactly one ObsID, recoverable from the retained ObsID field
        obsids_per_file = [
            np.unique(call.args[1]["ICIE__SW_OBSID_RAD"].values).tolist() for call in mock_write.call_args_list
        ]
        assert obsids_per_file == [[256], [257]]

    def test_input_files_names_the_parent_granule(self, tmp_path):
        """input_files must list the parent NOM-HK-DECODED granule, not the L0 packet files."""
        ds = _synthetic_nom_hk(rad_obsids=[128, 257, 257, 128])
        mock_filename = MagicMock()
        mock_filename.path.name = "fake.nc"
        with patch(
            "libera_utils.l1a.nom_hk_trim.write_libera_data_product",
            return_value=mock_filename,
        ) as mock_write:
            write_trimmed_nom_hk_products(ds, tmp_path, day=_DAY, source_product_filename=_PARENT_GRANULE, strict=False)

        written_attrs = mock_write.call_args_list[0].args[1].attrs
        assert written_attrs["input_files"] == [
            "LIBERA_L1A_NOM-HK-DECODED_V5-10-0_20280213T020000_20280213T040000_R20280213T120000000000.nc"
        ]
        # The parent's L0 provenance must not leak through
        assert "fake.bin" not in written_attrs["input_files"]

    def test_parent_date_created_is_dropped(self, tmp_path):
        """The parent's date_created is not handed to the writer, which sets the trimmed file's own."""
        ds = _synthetic_nom_hk(rad_obsids=[128, 257, 257, 128])
        assert DATE_CREATED_ATTRIBUTE in ds.attrs
        mock_filename = MagicMock()
        mock_filename.path.name = "fake.nc"
        with patch(
            "libera_utils.l1a.nom_hk_trim.write_libera_data_product",
            return_value=mock_filename,
        ) as mock_write:
            write_trimmed_nom_hk_products(ds, tmp_path, day=_DAY, source_product_filename=_PARENT_GRANULE, strict=False)

        assert DATE_CREATED_ATTRIBUTE not in mock_write.call_args_list[0].args[1].attrs

    def test_algorithm_version_is_restamped(self, tmp_path):
        ds = _synthetic_nom_hk(rad_obsids=[128, 257, 257, 128])
        mock_filename = MagicMock()
        mock_filename.path.name = "fake.nc"
        with patch(
            "libera_utils.l1a.nom_hk_trim.write_libera_data_product",
            return_value=mock_filename,
        ) as mock_write:
            write_trimmed_nom_hk_products(ds, tmp_path, day=_DAY, source_product_filename=_PARENT_GRANULE, strict=False)

        assert mock_write.call_args_list[0].args[1].attrs["algorithm_version"] == libera_utils_version()

    def test_definition_is_built_once_per_product_not_once_per_run(self, tmp_path):
        """Three runs of two ObsIDs cost two definitions, not three."""
        # 256 twice (disjoint runs) and 512 once
        ds = _synthetic_nom_hk(rad_obsids=[128, 256, 256, 128, 512, 128, 256, 128])
        mock_filename = MagicMock()
        mock_filename.path.name = "LIBERA_L1A_NOM-HK-SWC-365NM-TRIMMED_fake.nc"

        with (
            patch("libera_utils.l1a.nom_hk_trim.write_libera_data_product", return_value=mock_filename),
            patch(
                "libera_utils.l1a.nom_hk_trim.get_trimmed_nom_hk_product_definition",
                wraps=get_trimmed_nom_hk_product_definition,
            ) as mock_definition,
            patch(
                "libera_utils.l1a.nom_hk_trim.LiberaDataProductDefinition.from_yaml",
                wraps=LiberaDataProductDefinition.from_yaml,
            ) as mock_from_yaml,
        ):
            written = write_trimmed_nom_hk_products(
                ds, tmp_path, day=_DAY, source_product_filename=_PARENT_GRANULE, strict=False
            )

        assert len(written) == 3, "one file per contiguous ObsID run"
        assert mock_definition.call_count == 2, "definition built once per ProductID, not once per run"
        assert mock_from_yaml.call_count == 2, "definition YAML parsed once per ProductID, not once per run"


def _write_runs(ds: xr.Dataset, tmp_path, *, day: date = _DAY) -> tuple[list, MagicMock]:
    """Run the writer on ``ds`` with the file write mocked; return what it returned and the mock."""
    mock_filename = MagicMock()
    mock_filename.path.name = "fake.nc"
    with patch("libera_utils.l1a.nom_hk_trim.write_libera_data_product", return_value=mock_filename) as mock_write:
        written = write_trimmed_nom_hk_products(ds, tmp_path, day=day, source_product_filename=_PARENT_GRANULE)
    return written, mock_write


class TestDaySelection:
    """Only runs that start on the day are written, each flagged at an edge that follows a gap."""

    def test_run_crossing_midnight_is_written_whole_by_its_start_day(self, tmp_path):
        ds = _synthetic_nom_hk(rad_obsids=[128, 257, 257, 257, 257, 128], start="2028-02-13T23:59:58")
        written, mock_write = _write_runs(ds, tmp_path)

        assert len(written) == 1
        run = mock_write.call_args.args[1]
        assert run.attrs["ProductID"] == "NOM-HK-SWC-405NM-TRIMMED"
        assert run["PACKET_ICIE_TIME"].values.min() == np.datetime64("2028-02-13T23:59:59")
        assert run["PACKET_ICIE_TIME"].values.max() == np.datetime64("2028-02-14T00:00:02")

    def test_next_day_does_not_write_a_run_that_started_the_day_before(self, tmp_path):
        ds = _synthetic_nom_hk(rad_obsids=[128, 257, 257, 257, 257, 128], start="2028-02-13T23:59:58")
        written, mock_write = _write_runs(ds, tmp_path, day=date(2028, 2, 14))

        assert written == []
        mock_write.assert_not_called()

    def test_run_from_the_day_before_at_the_first_packet_is_left_out_without_error(self, tmp_path):
        """Its start is not in the data, but it started before the day, so it is not this day's run."""
        ds = _synthetic_nom_hk(rad_obsids=[257, 257, 128, 256, 256, 128], start="2028-02-13T23:59:58")
        written, mock_write = _write_runs(ds, tmp_path, day=date(2028, 2, 14))

        assert len(written) == 1
        assert mock_write.call_args.args[1].attrs["ProductID"] == "NOM-HK-SWC-365NM-TRIMMED"

    def test_run_starting_after_the_day_is_left_out_even_when_cut_off(self, tmp_path):
        ds = _synthetic_nom_hk(rad_obsids=[128, 256, 256, 128, 257, 257], start="2028-02-13T23:59:57")
        written, mock_write = _write_runs(ds, tmp_path)

        assert len(written) == 1
        assert mock_write.call_args.args[1].attrs["ProductID"] == "NOM-HK-SWC-365NM-TRIMMED"

    def test_no_calibration_obsid_writes_nothing_without_error(self, tmp_path):
        ds = _synthetic_nom_hk(rad_obsids=[128, 136, 136, 128])
        written, mock_write = _write_runs(ds, tmp_path)

        assert written == []
        mock_write.assert_not_called()

    @pytest.mark.parametrize(
        ("rad_obsids", "start"),
        [
            pytest.param([128, 256, 256, 128], "2028-02-12T23:59:59", id="run-at-the-day-start-is-kept"),
            pytest.param([128, 256, 257, 257, 128], "2028-02-13T23:59:58", id="run-at-the-next-day-start-is-left-out"),
        ],
    )
    def test_day_is_half_open(self, tmp_path, rad_obsids, start):
        """Packets are 1 s apart, so the 256 run starts at 00:00:00 of the day, or the 257 run at 00:00:00 of the next."""
        ds = _synthetic_nom_hk(rad_obsids=rad_obsids, start=start)
        _, mock_write = _write_runs(ds, tmp_path)

        assert [call.args[1].attrs["ProductID"] for call in mock_write.call_args_list] == ["NOM-HK-SWC-365NM-TRIMMED"]

    def test_missing_time_variable_raises_before_writing(self, tmp_path):
        ds = _synthetic_nom_hk(rad_obsids=[128, 256, 256, 128])
        with (
            patch("libera_utils.l1a.nom_hk_trim.write_libera_data_product") as mock_write,
            pytest.raises(ValueError, match="missing time variable 'NO_SUCH_TIME'"),
        ):
            write_trimmed_nom_hk_products(
                ds, tmp_path, day=_DAY, source_product_filename=_PARENT_GRANULE, time_variable="NO_SUCH_TIME"
            )

        mock_write.assert_not_called()

    def test_counter_steps_by_one_across_the_wrap(self, tmp_path):
        """A run edge where the 14-bit counter wraps to 0 is a step of one, not a gap."""
        top = SEQUENCE_COUNTER_MODULUS - 1
        ds = _synthetic_nom_hk(rad_obsids=[128, 257, 257, 128], sequence_counter=[top - 1, top, 0, 1])
        written, _ = _write_runs(ds, tmp_path)

        assert len(written) == 1

    @pytest.mark.parametrize(
        ("rad_obsids", "sequence_counter", "start_after_gap", "end_before_gap"),
        [
            pytest.param([257, 257, 128, 256, 256, 128], None, 1, 0, id="start-at-first-packet"),
            pytest.param([128, 256, 256, 128, 257, 257], None, 0, 1, id="data-stops-early-at-last-packet"),
            pytest.param([128, 256, 256, 128, 257, 257, 128], [1, 2, 3, 4, 6, 7, 8], 1, 0, id="gap-before-start"),
            pytest.param([128, 256, 256, 128, 257, 257, 128], [1, 2, 3, 4, 5, 6, 8], 0, 1, id="gap-after-end"),
        ],
    )
    def test_run_edge_at_a_gap_is_written_and_flagged(
        self, tmp_path, rad_obsids, sequence_counter, start_after_gap, end_before_gap
    ):
        """The 257 run is written with its gap edge flagged; the whole 256 run carries no flag."""
        ds = _synthetic_nom_hk(rad_obsids=rad_obsids, sequence_counter=sequence_counter)
        _, mock_write = _write_runs(ds, tmp_path)

        flags = {
            call.args[1].attrs["ProductID"]: (
                call.args[1].attrs[RUN_START_AFTER_GAP_ATTRIBUTE],
                call.args[1].attrs[RUN_END_BEFORE_GAP_ATTRIBUTE],
            )
            for call in mock_write.call_args_list
        }
        assert flags == {
            "NOM-HK-SWC-365NM-TRIMMED": (0, 0),
            "NOM-HK-SWC-405NM-TRIMMED": (start_after_gap, end_before_gap),
        }

    def test_gaps_inside_a_run_are_one_run_counting_the_missing_packets(self, tmp_path):
        """Packets lost inside one ObsID's stretch leave it one run, with its edges unflagged and the losses counted."""
        ds = _synthetic_nom_hk(rad_obsids=[128, 257, 257, 257, 257, 128], sequence_counter=[1, 2, 3, 5, 9, 10])
        written, mock_write = _write_runs(ds, tmp_path)

        assert len(written) == 1
        run = mock_write.call_args.args[1]
        assert run.sizes["PACKET"] == 4
        assert run.attrs[RUN_START_AFTER_GAP_ATTRIBUTE] == 0
        assert run.attrs[RUN_END_BEFORE_GAP_ATTRIBUTE] == 0
        assert run.attrs[RUN_MISSING_PACKETS_ATTRIBUTE] == 4

    def test_whole_run_counts_no_missing_packets(self, tmp_path):
        ds = _synthetic_nom_hk(rad_obsids=[128, 257, 257, 128], sequence_counter=[1, 2, 3, 4])
        _, mock_write = _write_runs(ds, tmp_path)

        assert mock_write.call_args.args[1].attrs[RUN_MISSING_PACKETS_ATTRIBUTE] == 0

    def test_gap_across_the_counter_wrap_is_counted(self, tmp_path):
        top = SEQUENCE_COUNTER_MODULUS - 1
        ds = _synthetic_nom_hk(rad_obsids=[128, 257, 257, 128], sequence_counter=[top - 1, top, 2, 3])
        _, mock_write = _write_runs(ds, tmp_path)

        assert mock_write.call_args.args[1].attrs[RUN_MISSING_PACKETS_ATTRIBUTE] == 2

    def test_find_obsid_runs_does_not_split_at_a_counter_gap(self):
        ds = _synthetic_nom_hk(rad_obsids=[128, 257, 257, 257, 128], sequence_counter=[1, 2, 3, 5, 6])
        assert [pkt_slice for _, pkt_slice in find_obsid_runs(ds)] == [slice(1, 4)]

    @pytest.mark.parametrize(
        ("last_time", "raises"),
        [
            pytest.param("2028-02-14T01:50:00", True, id="data-reaches-the-padding"),
            pytest.param("2028-02-14T01:49:59", False, id="data-stops-before-the-padding"),
        ],
    )
    def test_run_still_going_at_the_padding_raises(self, tmp_path, last_time, raises):
        """Data within DAY_BUFFER of 2 h past the day reaches the padding, so a run still going there outlasts it."""
        ds = _synthetic_nom_hk(rad_obsids=[128, 257, 257, 257])
        times = np.array(
            ["2028-02-13T23:00:00", "2028-02-13T23:00:01", "2028-02-14T00:30:00", last_time], dtype="datetime64[us]"
        )
        ds = ds.assign_coords(PACKET_ICIE_TIME=("PACKET", times))

        if raises:
            with pytest.raises(ValueError, match="the run lasts longer than CALIBRATION_TRIM_PADDING"):
                _write_runs(ds, tmp_path)
        else:
            _, mock_write = _write_runs(ds, tmp_path)
            assert mock_write.call_args.args[1].attrs[RUN_END_BEFORE_GAP_ATTRIBUTE] == 1

    def test_rejected_day_calls_no_write(self, tmp_path):
        """The whole 256 run is not written when the 257 run on the same day outlasts the padding."""
        ds = _synthetic_nom_hk(rad_obsids=[128, 256, 256, 128, 257, 257])
        times = np.datetime64("2028-02-13T23:00:00") + np.arange(5) * np.timedelta64(1, "s")
        times = np.append(times, np.datetime64("2028-02-14T02:00:00")).astype("datetime64[us]")
        ds = ds.assign_coords(PACKET_ICIE_TIME=("PACKET", times))
        mock_filename = MagicMock()
        with (
            patch("libera_utils.l1a.nom_hk_trim.write_libera_data_product", return_value=mock_filename) as mock_write,
            pytest.raises(ValueError, match="the run lasts longer than CALIBRATION_TRIM_PADDING"),
        ):
            write_trimmed_nom_hk_products(ds, tmp_path, day=_DAY, source_product_filename=_PARENT_GRANULE)

        mock_write.assert_not_called()
