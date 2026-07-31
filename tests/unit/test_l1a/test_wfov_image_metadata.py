"""Unit tests for WFOV image stitching, metadata extraction, and L1A enhancement."""

import struct

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from libera_utils.l1a.wfov_image_metadata import (
    BLOB_BYTE_COORD,
    CAMERA_TIME_COORD,
    ERROR_FLAGGED_IMAGE_COUNT_ATTR,
    FIRST_IMAGE_INCOMPLETE_ATTR,
    FOOTER_MISMATCH_COUNT_ATTR,
    FPGA_HEADER_SIZE,
    FPGA_TRAILING_FOOTER_SIZE,
    FSW_HEADER_SIZE,
    HEADER_PARSE_ERROR_COUNT_ATTR,
    LAST_IMAGE_INCOMPLETE_ATTR,
    PACKET_COUNT_NOT_USED_IN_IMAGES_ATTR,
    PACKET_IMAGE_ID_VAR,
    VALID_FOOTER_BYTES,
    WFOV_COMPRESSED_IMAGE_LENGTH_VAR,
    WFOV_COMPRESSED_IMAGE_VAR,
    WFOV_HEADER_PARSE_VALID_VAR,
    WFOV_HEADER_SIZE,
    enhance_wfov_l1a_dataset,
    extract_compressed_payload,
    extract_wfov_header_metadata_from_blob,
    is_valid_footer_from_blob,
    stitch_wfov_images,
    swap_32bit_words,
)
from libera_utils.time import multipart_to_dt64

# Byte offsets of the timestamp fields within the 36-byte FSW header; not exported by the module
# since production code reads the header sequentially, but fixed here to keep test blobs aligned.
_TIMESTAMP_SECONDS_OFFSET = 12
_TIMESTAMP_SUBSECONDS_OFFSET = 16


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
    struct.pack_into(">I", blob, _TIMESTAMP_SECONDS_OFFSET, timestamp_seconds)
    struct.pack_into(">I", blob, _TIMESTAMP_SUBSECONDS_OFFSET, timestamp_subseconds)
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


def _packet_rows(ds: xr.Dataset) -> list:
    """Build the ``list[bytes]`` argument ``stitch_wfov_images`` expects.

    Mirrors production's conversion in ``enhance_wfov_l1a_dataset``: a uint8 view rather than
    ``ndarray.tolist()``, since the fixed-width ``|S`` dtype silently strips trailing null bytes on
    conversion to plain Python ``bytes``, which would corrupt any packet whose valid payload run
    itself ends in zero bytes.
    """
    data_var = ds["ICIE__WFOV_DATA"]
    width = data_var.dtype.itemsize
    n_packets = ds.sizes["PACKET"]
    return [row.tobytes() for row in data_var.values.view(np.uint8).reshape(n_packets, width)]


def _build_complete_image_blob(
    payload: bytes,
    *,
    timestamp_seconds: int = 100,
    timestamp_subseconds: int = 1,
    footer_bytes: bytes | None = None,
    status_meta: dict | None = None,
) -> bytes:
    footer_bytes = VALID_FOOTER_BYTES if footer_bytes is None else footer_bytes
    fpga_block = _encode_fpga_block(status_meta=status_meta)
    return _build_fsw_blob(timestamp_seconds, timestamp_subseconds) + fpga_block + payload + footer_bytes


def _pad_packet(blob: bytes, packet_len: int | None = None) -> bytes:
    packet_len = packet_len or len(blob)
    return blob.ljust(972, b"\x00")


def _split_full_blob_for_packets(full_blob: bytes, payload_split: int) -> tuple[bytes, bytes, bytes]:
    """Split a stitched blob into SOP, MOP, and EOP packet payloads."""
    payload_start = WFOV_HEADER_SIZE
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


class TestExtractWfovHeaderMetadataFromBlob:
    def test_decodes_full_header_as_one_unit(self):
        fpga_block = _encode_fpga_block(
            header_meta={"image_length": 12345, "width": 2048, "height": 2048},
            footer_meta={"pixel_sum": 999, "crc": 0xDEADBEEF},
            status_meta={"sync_error": 1, "crc_error": 1},
        )
        blob = _build_fsw_blob(2212630896, 49631, 1.25) + fpga_block
        assert len(blob) == WFOV_HEADER_SIZE

        meta = extract_wfov_header_metadata_from_blob(blob)

        # FSW fields
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

        # Image header / footer / FPGA status fields, all decoded together
        assert meta["image_length"] == 12345
        assert meta["width"] == 2048
        assert meta["height"] == 2048
        assert meta["pixel_sum"] == 999
        assert meta["crc"] == 0xDEADBEEF
        assert meta["sync_error"] == 1
        assert meta["crc_error"] == 1

    def test_rejects_blob_too_short_for_whole_header(self):
        # One byte short of the combined FSW+FPGA header; no partial FSW-only decode is attempted.
        with pytest.raises(ValueError, match="Blob too small for WFOV header"):
            extract_wfov_header_metadata_from_blob(b"\x00" * (WFOV_HEADER_SIZE - 1))

    def test_rejects_blob_with_only_fsw_bytes(self):
        with pytest.raises(ValueError, match="Blob too small for WFOV header"):
            extract_wfov_header_metadata_from_blob(_build_fsw_blob())


class TestExtractCompressedPayload:
    def test_extracts_payload_between_fpga_block_and_trailing_footer(self):
        payload = b"\xff\xd8\xff\xe0" + b"\x00" * 10 + b"\x00"
        raw_blob = _build_complete_image_blob(payload)
        assert extract_compressed_payload(raw_blob) == payload

    def test_payload_ending_in_null_bytes_unchanged(self):
        payload = b"\xaa\xbb\xcc\x00\x00\x00"
        raw_blob = _build_complete_image_blob(payload)
        assert extract_compressed_payload(raw_blob) == payload


class TestIsValidFooterFromBlob:
    def test_valid_footer_bytes_return_true(self):
        raw_blob = _build_complete_image_blob(b"\x01\x02", footer_bytes=VALID_FOOTER_BYTES)
        assert is_valid_footer_from_blob(raw_blob) is True

    def test_mismatched_footer_bytes_return_false(self):
        raw_blob = _build_complete_image_blob(b"\x01\x02", footer_bytes=b"\x00" * FPGA_TRAILING_FOOTER_SIZE)
        assert is_valid_footer_from_blob(raw_blob) is False

    def test_too_short_blob_returns_false(self):
        assert is_valid_footer_from_blob(b"\x00" * 10) is False


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
            _packet_rows(ds),
        )
        assert len(stitched) == 1
        assert stats.n_packets_not_used_in_images == 0
        assert stitched[0].payload == payload

    def test_clean_window_has_no_edge_truncation(self):
        # A window that opens exactly on a qualifying SOP and closes exactly on its EOP has
        # neither edge truncated -- a stitched image existing at all doesn't by itself say
        # anything about truncation; only where the window's boundaries fell relative to SOP/EOP
        # does.
        blob = _build_complete_image_blob(b"\x01")
        ds = _make_wfov_packet_dataset(_complete_rows(blob))
        stitched, stats = stitch_wfov_images(
            ds["ICIE__MEM_DUMP_FLAGS_WFOV"].values,
            ds["ICIE__MEM_DUMP_OFFSET_WFOV"].values,
            ds["ICIE__MEM_DUMP_LENGTH_WFOV"].values,
            _packet_rows(ds),
        )
        assert len(stitched) == 1
        assert stats.first_image_incomplete is False
        assert stats.last_image_incomplete is False

    def test_leading_fragment_flags_first_image_incomplete(self):
        # An EOP with no preceding SOP anywhere in the window: the image it belongs to started
        # before this window. Expected edge truncation, so it's excluded from
        # n_packets_not_used_in_images and instead flagged via first_image_incomplete.
        ds = _make_wfov_packet_dataset([("EOP", 0, 10, _pad_packet(b"\x00" * 10))])
        stitched, stats = stitch_wfov_images(
            ds["ICIE__MEM_DUMP_FLAGS_WFOV"].values,
            ds["ICIE__MEM_DUMP_OFFSET_WFOV"].values,
            ds["ICIE__MEM_DUMP_LENGTH_WFOV"].values,
            _packet_rows(ds),
        )
        assert len(stitched) == 0
        assert stats.n_packets_not_used_in_images == 0
        assert stats.first_image_incomplete is True
        assert stats.last_image_incomplete is False

    def test_dangling_sop_flags_last_image_incomplete(self):
        # An SOP that never reaches its EOP because the window ends first. Expected edge
        # truncation, so it's excluded from n_packets_not_used_in_images and instead flagged via
        # last_image_incomplete.
        blob = _build_complete_image_blob(b"\x01")
        ds = _make_wfov_packet_dataset([("SOP", 0, len(blob), _pad_packet(blob))])
        stitched, stats = stitch_wfov_images(
            ds["ICIE__MEM_DUMP_FLAGS_WFOV"].values,
            ds["ICIE__MEM_DUMP_OFFSET_WFOV"].values,
            ds["ICIE__MEM_DUMP_LENGTH_WFOV"].values,
            _packet_rows(ds),
        )
        assert len(stitched) == 0
        assert stats.n_packets_not_used_in_images == 0
        assert stats.first_image_incomplete is False
        assert stats.last_image_incomplete is True

    def test_non_zero_sop_offset_counts_unused_packet(self):
        blob = _build_complete_image_blob(b"\x01")
        ds = _make_wfov_packet_dataset([("SOP", 512, len(blob), _pad_packet(blob))])
        stitched, stats = stitch_wfov_images(
            ds["ICIE__MEM_DUMP_FLAGS_WFOV"].values,
            ds["ICIE__MEM_DUMP_OFFSET_WFOV"].values,
            ds["ICIE__MEM_DUMP_LENGTH_WFOV"].values,
            _packet_rows(ds),
        )
        assert len(stitched) == 0
        # This packet is flagged SOP (just with a bad offset), not a pre-SOP fragment, so it's a
        # genuine anomaly rather than edge truncation.
        assert stats.n_packets_not_used_in_images == 1
        assert stats.first_image_incomplete is False

    def test_offset_gap_discards_all_collected_packets(self):
        blob = _build_complete_image_blob(b"\x01")
        ds = _make_wfov_packet_dataset(
            [
                ("SOP", 0, len(blob), _pad_packet(blob)),
                ("MOP", len(blob), len(blob), _pad_packet(blob)),
                ("EOP", len(blob) + 100, len(blob), _pad_packet(blob)),
            ]
        )
        stitched, stats = stitch_wfov_images(
            ds["ICIE__MEM_DUMP_FLAGS_WFOV"].values,
            ds["ICIE__MEM_DUMP_OFFSET_WFOV"].values,
            ds["ICIE__MEM_DUMP_LENGTH_WFOV"].values,
            _packet_rows(ds),
        )
        assert len(stitched) == 0
        # SOP + MOP + the mismatched EOP itself are all discarded together.
        assert stats.n_packets_not_used_in_images == 3

    def test_new_sop_aborts_prior_collection_and_counts_unused_packets(self):
        blob = _build_complete_image_blob(b"\x01")
        ds = _make_wfov_packet_dataset(
            [
                ("SOP", 0, len(blob), _pad_packet(blob)),
                ("MOP", len(blob), len(blob), _pad_packet(blob)),
                ("SOP", 0, len(blob), _pad_packet(blob)),
            ]
        )
        stitched, stats = stitch_wfov_images(
            ds["ICIE__MEM_DUMP_FLAGS_WFOV"].values,
            ds["ICIE__MEM_DUMP_OFFSET_WFOV"].values,
            ds["ICIE__MEM_DUMP_LENGTH_WFOV"].values,
            _packet_rows(ds),
        )
        assert len(stitched) == 0
        # First SOP+MOP (2 packets) abandoned when the second SOP arrives -- a genuine anomaly.
        # The second SOP is then left dangling with no EOP once the stream ends, but that's
        # expected trailing truncation (last_image_incomplete), not counted here.
        assert stats.n_packets_not_used_in_images == 2
        assert stats.last_image_incomplete is True

    def test_mop_while_seeking_after_discard_counts_as_unused_packet(self):
        # Once at least one SOP has been seen, a stray MOP encountered while re-seeking (e.g.
        # after a discarded collection, before the next SOP) is a genuine mid-stream gap -- not
        # leading-edge truncation -- and must be counted rather than silently dropped.
        blob = _build_complete_image_blob(b"\x01")
        ds = _make_wfov_packet_dataset(
            [
                ("SOP", 0, len(blob), _pad_packet(blob)),
                ("MOP", len(blob) + 999, len(blob), _pad_packet(blob)),  # offset gap -> discard SOP+MOP
                ("MOP", 0, 5, _pad_packet(b"\x00" * 5)),  # stray MOP while SEEKING
            ]
        )
        stitched, stats = stitch_wfov_images(
            ds["ICIE__MEM_DUMP_FLAGS_WFOV"].values,
            ds["ICIE__MEM_DUMP_OFFSET_WFOV"].values,
            ds["ICIE__MEM_DUMP_LENGTH_WFOV"].values,
            _packet_rows(ds),
        )
        assert len(stitched) == 0
        assert stats.n_packets_not_used_in_images == 3
        assert stats.first_image_incomplete is False
        assert stats.last_image_incomplete is False

    def test_error_flagged_and_footer_mismatch_stats(self):
        payload = b"\xca\xfe"
        blob = _build_complete_image_blob(payload, status_meta={"crc_error": 1}, footer_bytes=b"\x00" * 8)
        ds = _make_wfov_packet_dataset(_complete_rows(blob))
        stitched, stats = stitch_wfov_images(
            ds["ICIE__MEM_DUMP_FLAGS_WFOV"].values,
            ds["ICIE__MEM_DUMP_OFFSET_WFOV"].values,
            ds["ICIE__MEM_DUMP_LENGTH_WFOV"].values,
            _packet_rows(ds),
        )
        assert len(stitched) == 1
        assert stats.n_error_flagged_images == 1
        assert stats.n_footer_mismatches == 1
        assert stats.n_header_parse_errors == 0

    def test_packet_rows_are_freed_once_consumed(self):
        # Every row -- whether folded into a complete image or dropped -- must be released so its
        # memory can be freed as soon as its fate is decided, not held until the whole stream is
        # processed.
        blob = _build_complete_image_blob(b"\x01")
        ds = _make_wfov_packet_dataset(_complete_rows(blob))
        rows = _packet_rows(ds)
        stitched, _ = stitch_wfov_images(
            ds["ICIE__MEM_DUMP_FLAGS_WFOV"].values,
            ds["ICIE__MEM_DUMP_OFFSET_WFOV"].values,
            ds["ICIE__MEM_DUMP_LENGTH_WFOV"].values,
            rows,
        )
        assert len(stitched) == 1
        assert rows == [None, None]


class TestEnhanceWfovL1aDataset:
    def test_complete_sequence_creates_camera_time_and_drops_wfov_data(self):
        payload = b"\xca\xfe"
        blob = _build_complete_image_blob(payload)
        ds = _make_wfov_packet_dataset(_complete_rows(blob))

        enhanced = enhance_wfov_l1a_dataset(ds)
        assert enhanced.sizes[CAMERA_TIME_COORD] == 1
        length = int(enhanced[WFOV_COMPRESSED_IMAGE_LENGTH_VAR].values[0])
        blob_bytes = enhanced[WFOV_COMPRESSED_IMAGE_VAR].values[0, :length].tobytes()
        assert blob_bytes == payload
        assert enhanced[PACKET_IMAGE_ID_VAR].values.tolist() == [0, 0]
        assert "ICIE__WFOV_DATA" not in enhanced.data_vars
        assert enhanced.attrs[PACKET_COUNT_NOT_USED_IN_IMAGES_ATTR] == 0
        assert enhanced.attrs[ERROR_FLAGGED_IMAGE_COUNT_ATTR] == 0
        assert enhanced.attrs[FOOTER_MISMATCH_COUNT_ATTR] == 0
        assert enhanced.attrs[HEADER_PARSE_ERROR_COUNT_ATTR] == 0
        assert enhanced.attrs[FIRST_IMAGE_INCOMPLETE_ATTR] == 0
        assert enhanced.attrs[LAST_IMAGE_INCOMPLETE_ATTR] == 0

    def test_error_flagged_image_count_and_warning(self, caplog):
        import logging

        payload = b"\xca\xfe"
        blob = _build_complete_image_blob(payload, status_meta={"crc_error": 1})
        ds = _make_wfov_packet_dataset(_complete_rows(blob))

        with caplog.at_level(logging.WARNING):
            enhanced = enhance_wfov_l1a_dataset(ds)

        assert enhanced.attrs[ERROR_FLAGGED_IMAGE_COUNT_ATTR] == 1
        assert enhanced["WFOV_FPGA_STATUS_CRC_ERROR"].values[0] == 1
        assert "FPGA status errors flagged" in caplog.text

    def test_incomplete_sequence_drops_wfov_data_and_flags_last_incomplete(self):
        blob = _build_complete_image_blob(b"\x01")
        ds = _make_wfov_packet_dataset([("SOP", 0, len(blob), _pad_packet(blob))])

        enhanced = enhance_wfov_l1a_dataset(ds)
        assert enhanced.sizes[CAMERA_TIME_COORD] == 0
        assert enhanced[PACKET_IMAGE_ID_VAR].values.tolist() == [-1]
        assert "ICIE__WFOV_DATA" not in enhanced.data_vars
        assert enhanced.attrs[PACKET_COUNT_NOT_USED_IN_IMAGES_ATTR] == 0
        assert enhanced.attrs[FIRST_IMAGE_INCOMPLETE_ATTR] == 0
        assert enhanced.attrs[LAST_IMAGE_INCOMPLETE_ATTR] == 1

    def test_camera_time_from_fsw_timestamps(self):
        blob = _build_complete_image_blob(b"\x01", timestamp_seconds=100, timestamp_subseconds=1)
        ds = _make_wfov_packet_dataset(_complete_rows(blob))
        enhanced = enhance_wfov_l1a_dataset(ds)

        np.testing.assert_equal(enhanced[CAMERA_TIME_COORD].values[0], _expected_datetime64(100, 1))
        assert enhanced[WFOV_HEADER_PARSE_VALID_VAR].values[0]

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
        length = int(enhanced[WFOV_COMPRESSED_IMAGE_LENGTH_VAR].values[0])
        assert enhanced[WFOV_COMPRESSED_IMAGE_VAR].values[0, :length].tobytes() == payload
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
        length = int(enhanced[WFOV_COMPRESSED_IMAGE_LENGTH_VAR].values[0])
        assert enhanced[WFOV_COMPRESSED_IMAGE_VAR].values[0, :length].tobytes() == payload
