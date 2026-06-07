"""Unit tests for WFOV image FSW and FPGA metadata extraction."""

import struct

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from libera_utils.l1a.wfov_image_metadata import (
    CAMERA_TIME_COORD,
    FPGA_HEADER_SIZE,
    FSW_HEADER_SIZE,
    FSW_TIMESTAMP_MIN_SIZE,
    SOP_FPGA_MIN_SIZE,
    TIMESTAMP_SECONDS_OFFSET,
    TIMESTAMP_SUBSECONDS_OFFSET,
    assess_wfov_image_completeness,
    build_wfov_camera_metadata_dataset,
    extract_fpga_metadata_from_blob,
    extract_fsw_metadata_from_blob,
    find_qualifying_sop_indices,
    swap_32bit_words,
)
from libera_utils.time import multipart_to_dt64


def _build_fsw_blob(
    timestamp_seconds: int = 2212630896,
    timestamp_subseconds: int = 49631,
    azimuth_angle: float = 1.25,
) -> bytes:
    blob = bytearray(FSW_HEADER_SIZE)
    blob[0] = FSW_HEADER_SIZE
    blob[1] = 0x0A  # bitmask_id=1, img_mode=1
    blob[2] = 3
    blob[3] = 0
    struct.pack_into(">H", blob, 4, 42)
    blob[6] = 1
    blob[7] = 2
    struct.pack_into(">I", blob, 8, 0x12345678)
    struct.pack_into(">I", blob, TIMESTAMP_SECONDS_OFFSET, timestamp_seconds)
    struct.pack_into(">I", blob, TIMESTAMP_SUBSECONDS_OFFSET, timestamp_subseconds)
    struct.pack_into(">H", blob, 20, 0xABCD)
    struct.pack_into(">H", blob, 22, 0x00EF)
    struct.pack_into(">I", blob, 24, 1000)
    struct.pack_into(">I", blob, 28, 2000)
    struct.pack_into(">f", blob, 32, azimuth_angle)
    return bytes(blob)


def _expected_datetime64(timestamp_seconds: int, timestamp_subseconds: int) -> np.datetime64:
    dt = multipart_to_dt64(
        {"timestamp_seconds": timestamp_seconds, "timestamp_subseconds": timestamp_subseconds},
        s_field="timestamp_seconds",
        us_field="timestamp_subseconds",
    )
    return np.datetime64(pd.Timestamp(dt).to_datetime64(), "us")


def _encode_fpga_block(
    header_meta: dict | None = None,
    footer_meta: dict | None = None,
    status_meta: dict | None = None,
) -> bytes:
    header_meta = header_meta or {}
    footer_meta = footer_meta or {}
    status_meta = status_meta or {}

    swapped = bytearray(FPGA_HEADER_SIZE)
    header = bytearray(49)
    struct.pack_into("<I", header, 0, header_meta.get("image_length", 12345))
    header[4] = header_meta.get("flags", 2)
    header[5] = header_meta.get("frame_id", 7)
    struct.pack_into("<Q", header, 6, header_meta.get("tag", 0xAABBCCDDEEFF0011))
    struct.pack_into("<I", header, 14, header_meta.get("actual_exp_time_1", 500) & 0xFFFFFF)
    struct.pack_into("<H", header, 17, header_meta.get("temperature", 250))
    header[19] = header_meta.get("gain", 3)
    struct.pack_into("<H", header, 20, header_meta.get("width", 2048))
    struct.pack_into("<H", header, 22, header_meta.get("height", 2048))
    struct.pack_into("<H", header, 24, header_meta.get("offset_x", 0))
    struct.pack_into("<H", header, 26, header_meta.get("offset_y", 0))
    header[28] = header_meta.get("readout", 1)
    struct.pack_into("<I", header, 29, header_meta.get("actual_exp_time_2", 600) & 0xFFFFFF)
    struct.pack_into("<I", header, 32, header_meta.get("delta", 10) & 0xFFFFFF)
    struct.pack_into("<I", header, 35, header_meta.get("exposure_step", 20) & 0xFFFFFF)
    header[38] = header_meta.get("nr_slopes", 4)
    struct.pack_into("<I", header, 39, header_meta.get("kp1", 11) & 0xFFFFFF)
    struct.pack_into("<I", header, 42, header_meta.get("kp2", 12) & 0xFFFFFF)
    header[45] = header_meta.get("vlow_3", 13)
    header[46] = header_meta.get("vlow_2", 14)
    header[47] = header_meta.get("exp_seq", 15)
    header[48] = header_meta.get("footer_size", 8)

    for i, value in enumerate(header):
        swapped[2 + 2 * i] = value

    footer = bytearray(18)
    struct.pack_into("<I", footer, 0, footer_meta.get("pixel_sum", 999))
    struct.pack_into("<I", footer, 4, footer_meta.get("dark", 111) & 0xFFFFFF)
    struct.pack_into("<I", footer, 7, footer_meta.get("white", 222) & 0xFFFFFF)
    struct.pack_into("<I", footer, 10, footer_meta.get("footer_delta", 333))
    struct.pack_into("<I", footer, 14, footer_meta.get("crc", 0xDEADBEEF))

    for i, value in enumerate(footer):
        swapped[100 + 2 * i] = value

    status = 0
    for bit, key in enumerate(
        ("sync_error", "pid_error", "size_error", "eop_error", "eep_error", "crc_error", "drop_error")
    ):
        status |= (status_meta.get(key, 0) & 1) << bit
    struct.pack_into("<I", swapped, 136, status)

    raw = bytearray(FPGA_HEADER_SIZE)
    for i in range(0, FPGA_HEADER_SIZE, 4):
        raw[i : i + 4] = swapped[i : i + 4][::-1]
    return bytes(raw)


def _make_wfov_packet_dataset(
    rows: list[tuple[str, int, int, bytes]],
) -> xr.Dataset:
    flags = np.array([row[0] for row in rows], dtype="S8")
    offsets = np.array([row[1] for row in rows], dtype=np.uint32)
    lengths = np.array([row[2] for row in rows], dtype=np.uint32)
    data = np.array([row[3] for row in rows], dtype="S972")
    n_packets = len(rows)
    base_time = np.datetime64("2028-01-01T00:00:00", "us")
    packet_times = np.array([base_time + np.timedelta64(i, "s") for i in range(n_packets)], dtype="datetime64[us]")

    return xr.Dataset(
        {
            "ICIE__MEM_DUMP_FLAGS_WFOV": (("PACKET",), flags),
            "ICIE__MEM_DUMP_OFFSET_WFOV": (("PACKET",), offsets),
            "ICIE__MEM_DUMP_LENGTH_WFOV": (("PACKET",), lengths),
            "ICIE__WFOV_DATA": (("PACKET",), data),
        },
        coords={"PACKET_ICIE_TIME": ("PACKET", packet_times)},
    )


class TestSwap32BitWords:
    def test_reverses_each_word(self):
        data = b"\x01\x02\x03\x04\xaa\xbb\xcc\xdd"
        assert bytes(swap_32bit_words(data)) == b"\x04\x03\x02\x01\xdd\xcc\xbb\xaa"


class TestExtractFswMetadataFromBlob:
    def test_decodes_full_header(self):
        blob = _build_fsw_blob(2212630896, 49631, 1.25)
        meta = extract_fsw_metadata_from_blob(blob)
        assert meta["fsw_length"] == FSW_HEADER_SIZE
        assert meta["bitmask_id"] == 1
        assert meta["img_mode"] == 1
        assert meta["cadence"] == 42
        assert meta["timestamp_seconds"] == 2212630896
        assert meta["timestamp_subseconds"] == 49631
        assert meta["rad_obs_id"] == 0xABCD
        assert meta["cam_obs_id"] == 0x00EF
        assert meta["commanded_exp_time_1"] == 1000
        assert meta["commanded_exp_time_2"] == 2000
        assert meta["azimuth_angle"] == pytest.approx(1.25)

    def test_rejects_short_blob(self):
        with pytest.raises(ValueError, match="Blob too small for full FSW header"):
            extract_fsw_metadata_from_blob(b"\x00" * (FSW_HEADER_SIZE - 1))


class TestExtractFpgaMetadataFromBlob:
    def test_decodes_fpga_block_from_full_sop_slice(self):
        fpga_block = _encode_fpga_block(
            header_meta={"image_length": 12345, "width": 2048, "height": 2048},
            footer_meta={"pixel_sum": 999, "crc": 0xDEADBEEF},
            status_meta={"sync_error": 1, "crc_error": 1},
        )
        blob = _build_fsw_blob() + fpga_block
        assert len(blob) == SOP_FPGA_MIN_SIZE

        meta = extract_fpga_metadata_from_blob(blob)
        assert meta["image_length"] == 12345
        assert meta["width"] == 2048
        assert meta["height"] == 2048
        assert meta["pixel_sum"] == 999
        assert meta["crc"] == 0xDEADBEEF
        assert meta["sync_error"] == 1
        assert meta["crc_error"] == 1

    def test_rejects_short_blob(self):
        with pytest.raises(ValueError, match="Blob too small for FPGA block"):
            extract_fpga_metadata_from_blob(_build_fsw_blob())


class TestFindQualifyingSopIndices:
    def test_returns_zero_offset_sops_in_order(self):
        flags = np.array(["MOP", "SOP", "SOP", "SOP"], dtype="S8")
        offsets = np.array([0, 512, 0, 0], dtype=np.uint32)
        assert np.array_equal(find_qualifying_sop_indices(flags, offsets), np.array([2, 3]))


class TestAssessWfovImageCompleteness:
    def test_complete_image(self):
        flags = np.array(["SOP", "MOP", "EOP"], dtype="S8")
        offsets = np.array([0, 100, 200], dtype=np.uint32)
        lengths = np.array([100, 100, 50], dtype=np.uint32)
        complete = assess_wfov_image_completeness(flags, offsets, lengths)
        assert complete.tolist() == [True, False, False]

    def test_missing_eop(self):
        flags = np.array(["SOP", "MOP"], dtype="S8")
        offsets = np.array([0, 100], dtype=np.uint32)
        lengths = np.array([100, 100], dtype=np.uint32)
        complete = assess_wfov_image_completeness(flags, offsets, lengths)
        assert complete.tolist() == [False, False]

    def test_offset_gap(self):
        flags = np.array(["SOP", "MOP", "EOP"], dtype="S8")
        offsets = np.array([0, 100, 250], dtype=np.uint32)
        lengths = np.array([100, 100, 50], dtype=np.uint32)
        complete = assess_wfov_image_completeness(flags, offsets, lengths)
        assert complete.tolist() == [False, False, False]


class TestBuildWfovCameraMetadataDataset:
    def test_camera_time_from_fsw_timestamps(self):
        blob = _build_fsw_blob(100, 1) + _encode_fpga_block()
        ds = _make_wfov_packet_dataset([("SOP", 0, len(blob), blob.ljust(972, b"\x00"))])
        camera_ds = build_wfov_camera_metadata_dataset(ds)

        assert camera_ds.sizes[CAMERA_TIME_COORD] == 1
        np.testing.assert_equal(camera_ds[CAMERA_TIME_COORD].values[0], _expected_datetime64(100, 1))
        assert camera_ds["WFOV_FSW_PARSE_VALID"].values[0]
        assert camera_ds["WFOV_FPGA_PARSE_VALID"].values[0]

    def test_partial_length_sop_decodes_fsw_only(self):
        blob = _build_fsw_blob(200, 2).ljust(100, b"\x00")
        ds = _make_wfov_packet_dataset([("SOP", 0, 100, blob.ljust(972, b"\x00"))])
        camera_ds = build_wfov_camera_metadata_dataset(ds)

        assert camera_ds["WFOV_FSW_PARSE_VALID"].values[0]
        assert not camera_ds["WFOV_FPGA_PARSE_VALID"].values[0]

    def test_short_sop_marks_fsw_invalid(self):
        blob = b"\x00" * (FSW_TIMESTAMP_MIN_SIZE - 1)
        ds = _make_wfov_packet_dataset([("SOP", 0, len(blob), blob.ljust(972, b"\x00"))])
        camera_ds = build_wfov_camera_metadata_dataset(ds)

        assert not camera_ds["WFOV_FSW_PARSE_VALID"].values[0]
        assert not camera_ds["WFOV_FPGA_PARSE_VALID"].values[0]
        assert np.isnat(camera_ds[CAMERA_TIME_COORD].values[0])

    def test_preserves_packet_order_not_acquisition_time_order(self):
        later_blob = _build_fsw_blob(300, 3)
        earlier_blob = _build_fsw_blob(100, 1)
        ds = _make_wfov_packet_dataset(
            [
                ("SOP", 0, len(later_blob), later_blob.ljust(972, b"\x00")),
                ("SOP", 0, len(earlier_blob), earlier_blob.ljust(972, b"\x00")),
            ]
        )
        camera_ds = build_wfov_camera_metadata_dataset(ds)

        np.testing.assert_equal(
            camera_ds[CAMERA_TIME_COORD].values,
            np.array([_expected_datetime64(300, 3), _expected_datetime64(100, 1)], dtype="datetime64[us]"),
        )
        np.testing.assert_array_equal(camera_ds["CAMERA_PACKET_INDEX"].values, np.array([0, 1], dtype=np.int32))

    def test_image_complete_flag(self):
        blob = _build_fsw_blob(100, 1)
        ds = _make_wfov_packet_dataset(
            [
                ("SOP", 0, len(blob), blob.ljust(972, b"\x00")),
                ("MOP", len(blob), len(blob), blob.ljust(972, b"\x00")),
                ("EOP", 2 * len(blob), len(blob), blob.ljust(972, b"\x00")),
            ]
        )
        camera_ds = build_wfov_camera_metadata_dataset(ds)
        assert camera_ds["WFOV_IMAGE_COMPLETE"].values[0]

    def test_ignores_non_qualifying_sops(self):
        blob = _build_fsw_blob(100, 1)
        ds = _make_wfov_packet_dataset([("SOP", 512, len(blob), blob.ljust(972, b"\x00"))])
        camera_ds = build_wfov_camera_metadata_dataset(ds)
        assert camera_ds.sizes[CAMERA_TIME_COORD] == 0
