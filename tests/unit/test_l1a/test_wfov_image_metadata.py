"""Unit tests for WFOV image stitching, metadata extraction, and L1A enhancement."""

import struct

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from libera_utils.l1a.wfov_image_metadata import (
    BLOB_BYTE_COORD,
    CAMERA_TIME_COORD,
    COMPLETE_IMAGE_COUNT_ATTR,
    CRC_ERROR_COUNT_ATTR,
    FPGA_HEADER_SIZE,
    FPGA_TRAILING_FOOTER_SIZE,
    FSW_HEADER_SIZE,
    PACKET_IMAGE_ID_VAR,
    SOP_FPGA_MIN_SIZE,
    TIMESTAMP_SECONDS_OFFSET,
    TIMESTAMP_SUBSECONDS_OFFSET,
    WFOV_IMAGE_BLOB_LENGTH_VAR,
    WFOV_IMAGE_BLOB_VAR,
    encode_trailing_footer_bytes,
    enhance_wfov_l1a_dataset,
    extract_compressed_payload,
    extract_fpga_metadata_from_blob,
    extract_fsw_metadata_from_blob,
    extract_trailing_footer_from_blob,
    stitch_wfov_images,
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


def _build_complete_image_blob(
    payload: bytes,
    *,
    timestamp_seconds: int = 100,
    timestamp_subseconds: int = 1,
    trailing_footer: bytes | None = None,
) -> bytes:
    trailing_footer = trailing_footer or encode_trailing_footer_bytes(
        pixel_sum=111,
        dark=222,
        white=333,
        sync_error=1,
        crc_error=1,
    )
    return _build_fsw_blob(timestamp_seconds, timestamp_subseconds) + _encode_fpga_block() + payload + trailing_footer


def _pad_packet(blob: bytes, packet_len: int | None = None) -> bytes:
    packet_len = packet_len or len(blob)
    return blob.ljust(972, b"\x00")


def _split_full_blob_for_packets(full_blob: bytes, payload_split: int) -> tuple[bytes, bytes, bytes]:
    """Split a stitched blob into SOP, MOP, and EOP packet payloads."""
    payload_start = SOP_FPGA_MIN_SIZE
    payload_end = len(full_blob) - FPGA_TRAILING_FOOTER_SIZE
    sop_blob = full_blob[: payload_start + payload_split]
    mop_blob = full_blob[payload_start + payload_split : payload_end]
    eop_blob = full_blob[payload_end:]
    return sop_blob, mop_blob, eop_blob


def _complete_rows(blob: bytes) -> list[tuple[str, int, int, bytes]]:
    """Build minimal SOP+EOP rows that stitch to ``blob``."""
    return [
        ("SOP", 0, len(blob), _pad_packet(blob)),
        ("EOP", len(blob), 0, _pad_packet(b"\x00" * 972)),
    ]


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


class TestExtractCompressedPayload:
    def test_extracts_payload_between_fpga_block_and_trailing_footer(self):
        payload = b"\xff\xd8\xff\xe0" + b"\x00" * 10 + b"\x00"
        raw_blob = _build_complete_image_blob(payload)
        assert extract_compressed_payload(raw_blob) == payload

    def test_payload_ending_in_null_bytes_unchanged(self):
        payload = b"\xaa\xbb\xcc\x00\x00\x00"
        raw_blob = _build_complete_image_blob(payload)
        assert extract_compressed_payload(raw_blob) == payload


class TestTrailingFooterDecode:
    def test_round_trip(self):
        footer = encode_trailing_footer_bytes(
            pixel_sum=0x12345678,
            dark=0xABCDEF,
            white=0x112233,
            sync_error=1,
            pid_error=0,
            spare=1,
        )
        raw_blob = _build_complete_image_blob(b"\x01\x02", trailing_footer=footer)
        decoded = extract_trailing_footer_from_blob(raw_blob)
        assert decoded["pixel_sum"] == 0x12345678
        assert decoded["dark"] == 0xABCDEF
        assert decoded["sync_error"] == 1
        assert decoded["spare"] == 1
        assert decoded["delta"] == 333
        assert decoded["crc"] == 0xDEADBEEF
        assert decoded["white"] == 222


class TestStitchWfovImages:
    def test_multi_packet_stitch(self):
        payload = b"\xde\xad\xbe\xef"
        full_blob = _build_complete_image_blob(payload)
        sop_blob, mop_blob, eop_blob = _split_full_blob_for_packets(full_blob, payload_split=2)
        ds = _make_wfov_packet_dataset(
            [
                ("SOP", 0, len(sop_blob), _pad_packet(sop_blob)),
                ("MOP", len(sop_blob), len(mop_blob), _pad_packet(mop_blob)),
                ("EOP", len(sop_blob) + len(mop_blob), len(eop_blob), _pad_packet(eop_blob)),
            ]
        )
        stitched, stats = stitch_wfov_images(
            ds["ICIE__MEM_DUMP_FLAGS_WFOV"].values,
            ds["ICIE__MEM_DUMP_OFFSET_WFOV"].values,
            ds["ICIE__MEM_DUMP_LENGTH_WFOV"].values,
            ds["ICIE__WFOV_DATA"].values,
        )
        assert stats.n_complete_images == 1
        assert len(stitched) == 1
        assert extract_compressed_payload(stitched[0].raw_blob) == payload

    def test_orphan_eop_increments_missing_sop_or_eop(self):
        ds = _make_wfov_packet_dataset([("EOP", 0, 10, _pad_packet(b"\x00" * 10))])
        _, stats = stitch_wfov_images(
            ds["ICIE__MEM_DUMP_FLAGS_WFOV"].values,
            ds["ICIE__MEM_DUMP_OFFSET_WFOV"].values,
            ds["ICIE__MEM_DUMP_LENGTH_WFOV"].values,
            ds["ICIE__WFOV_DATA"].values,
        )
        assert stats.n_missing_sop_or_eop == 1
        assert stats.n_bad_images == 0

    def test_sop_without_eop_increments_missing_sop_or_eop(self):
        blob = _build_complete_image_blob(b"\x01")
        ds = _make_wfov_packet_dataset([("SOP", 0, len(blob), _pad_packet(blob))])
        _, stats = stitch_wfov_images(
            ds["ICIE__MEM_DUMP_FLAGS_WFOV"].values,
            ds["ICIE__MEM_DUMP_OFFSET_WFOV"].values,
            ds["ICIE__MEM_DUMP_LENGTH_WFOV"].values,
            ds["ICIE__WFOV_DATA"].values,
        )
        assert stats.n_missing_sop_or_eop == 1

    def test_non_zero_sop_offset_counts_bad_image(self):
        blob = _build_complete_image_blob(b"\x01")
        ds = _make_wfov_packet_dataset([("SOP", 512, len(blob), _pad_packet(blob))])
        _, stats = stitch_wfov_images(
            ds["ICIE__MEM_DUMP_FLAGS_WFOV"].values,
            ds["ICIE__MEM_DUMP_OFFSET_WFOV"].values,
            ds["ICIE__MEM_DUMP_LENGTH_WFOV"].values,
            ds["ICIE__WFOV_DATA"].values,
        )
        assert stats.n_bad_images == 1
        assert stats.n_complete_images == 0

    def test_offset_gap_counts_bad_image(self):
        blob = _build_complete_image_blob(b"\x01")
        ds = _make_wfov_packet_dataset(
            [
                ("SOP", 0, len(blob), _pad_packet(blob)),
                ("MOP", len(blob), len(blob), _pad_packet(blob)),
                ("EOP", len(blob) + 100, len(blob), _pad_packet(blob)),
            ]
        )
        _, stats = stitch_wfov_images(
            ds["ICIE__MEM_DUMP_FLAGS_WFOV"].values,
            ds["ICIE__MEM_DUMP_OFFSET_WFOV"].values,
            ds["ICIE__MEM_DUMP_LENGTH_WFOV"].values,
            ds["ICIE__WFOV_DATA"].values,
        )
        assert stats.n_bad_images == 1
        assert stats.n_complete_images == 0

    def test_new_sop_aborts_prior_collection(self):
        blob = _build_complete_image_blob(b"\x01")
        ds = _make_wfov_packet_dataset(
            [
                ("SOP", 0, len(blob), _pad_packet(blob)),
                ("MOP", len(blob), len(blob), _pad_packet(blob)),
                ("SOP", 0, len(blob), _pad_packet(blob)),
            ]
        )
        _, stats = stitch_wfov_images(
            ds["ICIE__MEM_DUMP_FLAGS_WFOV"].values,
            ds["ICIE__MEM_DUMP_OFFSET_WFOV"].values,
            ds["ICIE__MEM_DUMP_LENGTH_WFOV"].values,
            ds["ICIE__WFOV_DATA"].values,
        )
        assert stats.n_missing_sop_or_eop == 2


class TestEnhanceWfovL1aDataset:
    def test_complete_sequence_creates_camera_time_and_zeros_packets(self):
        payload = b"\xca\xfe"
        blob = _build_complete_image_blob(payload)
        ds = _make_wfov_packet_dataset(_complete_rows(blob))
        original_bytes = bytes(ds["ICIE__WFOV_DATA"].values[0])

        enhanced = enhance_wfov_l1a_dataset(ds)
        assert enhanced.sizes[CAMERA_TIME_COORD] == 1
        length = int(enhanced[WFOV_IMAGE_BLOB_LENGTH_VAR].values[0])
        blob_bytes = enhanced[WFOV_IMAGE_BLOB_VAR].values[0, :length].tobytes()
        assert blob_bytes == payload
        assert enhanced[PACKET_IMAGE_ID_VAR].values.tolist() == [0, 0]
        assert bytes(enhanced["ICIE__WFOV_DATA"].values[0]) != original_bytes
        assert np.all(enhanced["ICIE__WFOV_DATA"].values[0].view(np.uint8) == 0)
        assert enhanced.attrs[COMPLETE_IMAGE_COUNT_ATTR] == 1
        assert enhanced.attrs[CRC_ERROR_COUNT_ATTR] == 0

    def test_crc_error_count_and_warning(self, caplog):
        import logging

        payload = b"\xca\xfe"
        fpga_block = _encode_fpga_block(status_meta={"crc_error": 1})
        trailing_footer = encode_trailing_footer_bytes()
        blob = _build_fsw_blob() + fpga_block + payload + trailing_footer
        ds = _make_wfov_packet_dataset(_complete_rows(blob))

        with caplog.at_level(logging.WARNING):
            enhanced = enhance_wfov_l1a_dataset(ds)

        assert enhanced.attrs[CRC_ERROR_COUNT_ATTR] == 1
        assert enhanced["WFOV_FPGA_CRC_ERROR"].values[0] == 1
        assert "FPGA CRC errors" in caplog.text

    def test_incomplete_sequence_preserves_packet_data(self):
        blob = _build_complete_image_blob(b"\x01")
        ds = _make_wfov_packet_dataset([("SOP", 0, len(blob), _pad_packet(blob))])
        original_bytes = bytes(ds["ICIE__WFOV_DATA"].values[0])

        enhanced = enhance_wfov_l1a_dataset(ds)
        assert enhanced.sizes[CAMERA_TIME_COORD] == 0
        assert enhanced[PACKET_IMAGE_ID_VAR].values.tolist() == [-1]
        assert bytes(enhanced["ICIE__WFOV_DATA"].values[0]) == original_bytes

    def test_camera_time_from_fsw_timestamps(self):
        blob = _build_complete_image_blob(b"\x01", timestamp_seconds=100, timestamp_subseconds=1)
        ds = _make_wfov_packet_dataset(_complete_rows(blob))
        enhanced = enhance_wfov_l1a_dataset(ds)

        np.testing.assert_equal(enhanced[CAMERA_TIME_COORD].values[0], _expected_datetime64(100, 1))
        assert enhanced["WFOV_FSW_PARSE_VALID"].values[0]
        assert enhanced["WFOV_FPGA_PARSE_VALID"].values[0]

    def test_multi_packet_complete_image(self):
        payload = b"\x11\x22\x33\x44"
        full_blob = _build_complete_image_blob(payload)
        sop_blob, mop_blob, eop_blob = _split_full_blob_for_packets(full_blob, payload_split=2)
        ds = _make_wfov_packet_dataset(
            [
                ("SOP", 0, len(sop_blob), _pad_packet(sop_blob)),
                ("MOP", len(sop_blob), len(mop_blob), _pad_packet(mop_blob)),
                ("EOP", len(sop_blob) + len(mop_blob), len(eop_blob), _pad_packet(eop_blob)),
            ]
        )
        enhanced = enhance_wfov_l1a_dataset(ds)
        assert enhanced.sizes[CAMERA_TIME_COORD] == 1
        length = int(enhanced[WFOV_IMAGE_BLOB_LENGTH_VAR].values[0])
        assert enhanced[WFOV_IMAGE_BLOB_VAR].values[0, :length].tobytes() == payload
        assert enhanced[PACKET_IMAGE_ID_VAR].values.tolist() == [0, 0, 0]
        assert BLOB_BYTE_COORD in enhanced.dims

    def test_preserves_packet_order_not_acquisition_time_order(self):
        later_blob = _build_complete_image_blob(b"\x01", timestamp_seconds=300, timestamp_subseconds=3)
        earlier_blob = _build_complete_image_blob(b"\x02", timestamp_seconds=100, timestamp_subseconds=1)
        ds = _make_wfov_packet_dataset(
            [
                *_complete_rows(later_blob),
                *_complete_rows(earlier_blob),
            ]
        )
        enhanced = enhance_wfov_l1a_dataset(ds)
        np.testing.assert_equal(
            enhanced[CAMERA_TIME_COORD].values,
            np.array([_expected_datetime64(300, 3), _expected_datetime64(100, 1)], dtype="datetime64[us]"),
        )
        np.testing.assert_array_equal(enhanced["CAMERA_PACKET_INDEX"].values, np.array([0, 2], dtype=np.int32))

    def test_stitch_with_trailing_null_padding_in_packet_buffer(self):
        payload = b"\xaa\xbb\x00\x00"
        blob = _build_complete_image_blob(payload)
        rows = _complete_rows(blob)
        rows[0] = ("SOP", rows[0][1], rows[0][2], _pad_packet(blob, packet_len=len(blob)))
        ds = _make_wfov_packet_dataset(rows)
        enhanced = enhance_wfov_l1a_dataset(ds)
        length = int(enhanced[WFOV_IMAGE_BLOB_LENGTH_VAR].values[0])
        assert enhanced[WFOV_IMAGE_BLOB_VAR].values[0, :length].tobytes() == payload
