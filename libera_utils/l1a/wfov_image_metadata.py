"""WFOV camera image stitching, metadata extraction, and L1A CAMERA_TIME enhancement."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from io import BytesIO

import numpy as np
import pandas as pd
import xarray as xr

from libera_utils.time import multipart_to_dt64

CAMERA_TIME_COORD = "CAMERA_TIME"
BLOB_BYTE_COORD = "BLOB_BYTE"
CAMERA_PACKET_INDEX_VAR = "CAMERA_PACKET_INDEX"
PACKET_IMAGE_ID_VAR = "PACKET_IMAGE_ID"
WFOV_FSW_PARSE_VALID_VAR = "WFOV_FSW_PARSE_VALID"
WFOV_FPGA_PARSE_VALID_VAR = "WFOV_FPGA_PARSE_VALID"
WFOV_IMAGE_BLOB_VAR = "WFOV_IMAGE_BLOB"
WFOV_IMAGE_BLOB_LENGTH_VAR = "WFOV_IMAGE_BLOB_LENGTH"
WFOV_CRC_VALID_VAR = "WFOV_CRC_VALID"
WFOV_CRC_VALID_NOT_VALIDATED = np.int8(-1)

FSW_HEADER_SIZE = 36
FPGA_HEADER_SIZE = 140
FPGA_TRAILING_FOOTER_SIZE = 8
SOP_FPGA_MIN_SIZE = FSW_HEADER_SIZE + FPGA_HEADER_SIZE
MIN_STITCHED_BLOB_SIZE = SOP_FPGA_MIN_SIZE + FPGA_TRAILING_FOOTER_SIZE
FSW_TIMESTAMP_MIN_SIZE = 20
TIMESTAMP_SECONDS_OFFSET = 12
TIMESTAMP_SUBSECONDS_OFFSET = 16
PACKET_DATA_WIDTH = 972

MEM_DUMP_FLAGS_VAR = "ICIE__MEM_DUMP_FLAGS_WFOV"
MEM_DUMP_OFFSET_VAR = "ICIE__MEM_DUMP_OFFSET_WFOV"
MEM_DUMP_LENGTH_VAR = "ICIE__MEM_DUMP_LENGTH_WFOV"
WFOV_DATA_VAR = "ICIE__WFOV_DATA"

FSW_FIELDS = (
    "fsw_length",
    "jpeg_bypass",
    "bitmask_disable",
    "testpattern",
    "bitmask_id",
    "img_mode",
    "pixel_mask_id",
    "simulator",
    "cadence",
    "image_total",
    "image_count",
    "flash_write_pointer",
    "timestamp_seconds",
    "timestamp_subseconds",
    "rad_obs_id",
    "cam_obs_id",
    "commanded_exp_time_1",
    "commanded_exp_time_2",
    "azimuth_angle",
)

FPGA_HEADER_FIELDS = (
    "image_length",
    "flags",
    "frame_id",
    "tag",
    "actual_exp_time_1",
    "temperature",
    "gain",
    "width",
    "height",
    "offset_x",
    "offset_y",
    "readout",
    "actual_exp_time_2",
    "delta",
    "exposure_step",
    "nr_slopes",
    "kp1",
    "kp2",
    "vlow_3",
    "vlow_2",
    "exp_seq",
    "footer_size",
)

FPGA_FOOTER_FIELDS = (
    "pixel_sum",
    "dark",
    "white",
    "footer_delta",
    "crc",
)

FPGA_STATUS_FIELDS = (
    "sync_error",
    "pid_error",
    "size_error",
    "eop_error",
    "eep_error",
    "crc_error",
    "drop_error",
)

TRAILING_FOOTER_VALUE_FIELDS = (
    "pixel_sum",
    "dark",
    "white",
    "delta",
    "crc",
)

TRAILING_FOOTER_STATUS_FIELDS = (
    "spare",
    "drop_error",
    "crc_error",
    "eep_error",
    "eop_error",
    "size_error",
    "pid_error",
    "sync_error",
)

DATETIME_USEC_DTYPE = np.dtype("datetime64[us]")

FSW_FIELD_DTYPES: dict[str, np.dtype] = {
    "fsw_length": np.dtype("uint8"),
    "jpeg_bypass": np.dtype("uint8"),
    "bitmask_disable": np.dtype("uint8"),
    "testpattern": np.dtype("uint8"),
    "bitmask_id": np.dtype("uint8"),
    "img_mode": np.dtype("uint8"),
    "pixel_mask_id": np.dtype("uint8"),
    "simulator": np.dtype("uint8"),
    "cadence": np.dtype("uint16"),
    "image_total": np.dtype("uint8"),
    "image_count": np.dtype("uint8"),
    "flash_write_pointer": np.dtype("uint32"),
    "timestamp_seconds": np.dtype("uint32"),
    "timestamp_subseconds": np.dtype("uint32"),
    "rad_obs_id": np.dtype("uint16"),
    "cam_obs_id": np.dtype("uint16"),
    "commanded_exp_time_1": np.dtype("uint32"),
    "commanded_exp_time_2": np.dtype("uint32"),
    "azimuth_angle": np.dtype("float32"),
}

FPGA_FIELD_DTYPES: dict[str, np.dtype] = {
    "image_length": np.dtype("uint32"),
    "flags": np.dtype("uint8"),
    "frame_id": np.dtype("uint8"),
    "tag": np.dtype("uint64"),
    "actual_exp_time_1": np.dtype("uint32"),
    "temperature": np.dtype("uint16"),
    "gain": np.dtype("uint8"),
    "width": np.dtype("uint16"),
    "height": np.dtype("uint16"),
    "offset_x": np.dtype("uint16"),
    "offset_y": np.dtype("uint16"),
    "readout": np.dtype("uint8"),
    "actual_exp_time_2": np.dtype("uint32"),
    "delta": np.dtype("uint32"),
    "exposure_step": np.dtype("uint32"),
    "nr_slopes": np.dtype("uint8"),
    "kp1": np.dtype("uint32"),
    "kp2": np.dtype("uint32"),
    "vlow_3": np.dtype("uint8"),
    "vlow_2": np.dtype("uint8"),
    "exp_seq": np.dtype("uint8"),
    "footer_size": np.dtype("uint8"),
    "pixel_sum": np.dtype("uint32"),
    "dark": np.dtype("uint32"),
    "white": np.dtype("uint32"),
    "footer_delta": np.dtype("uint32"),
    "crc": np.dtype("uint32"),
    "sync_error": np.dtype("uint8"),
    "pid_error": np.dtype("uint8"),
    "size_error": np.dtype("uint8"),
    "eop_error": np.dtype("uint8"),
    "eep_error": np.dtype("uint8"),
    "crc_error": np.dtype("uint8"),
    "drop_error": np.dtype("uint8"),
}

TRAILING_FOOTER_FIELD_DTYPES: dict[str, np.dtype] = {
    "pixel_sum": np.dtype("uint32"),
    "dark": np.dtype("uint32"),
    "white": np.dtype("uint32"),
    "delta": np.dtype("uint32"),
    "crc": np.dtype("uint32"),
    "spare": np.dtype("uint8"),
    "drop_error": np.dtype("uint8"),
    "crc_error": np.dtype("uint8"),
    "eep_error": np.dtype("uint8"),
    "eop_error": np.dtype("uint8"),
    "size_error": np.dtype("uint8"),
    "pid_error": np.dtype("uint8"),
    "sync_error": np.dtype("uint8"),
}


@dataclass(frozen=True)
class StitchedImage:
    """One complete WFOV image stitched from SOP through EOP."""

    image_id: int
    sop_index: int
    eop_index: int
    raw_blob: bytes


@dataclass
class StitchStats:
    """Counters for WFOV image stitching quality metrics."""

    n_missing_sop_or_eop: int = 0
    n_bad_images: int = 0
    n_complete_images: int = 0
    n_unexpected_eop: int = dataclass_field(default=0, repr=False)
    n_images_discarded_sop: int = dataclass_field(default=0, repr=False)
    n_images_discarded_gap: int = dataclass_field(default=0, repr=False)


def swap_32bit_words(data: bytes) -> bytearray:
    """Swap 32-bit words in the data."""
    result = bytearray(len(data))
    for i in range(0, len(data), 4):
        result[i : i + 4] = data[i : i + 4][::-1]
    return result


def extract_fsw_metadata_from_blob(blob_bytes: bytes) -> dict:
    """Extract FSW metadata from the start of a WFOV image blob.

    Byte layout matches ``read_fsw_metadata`` in libera_cam ``metadata_parser.py``.
    Requires at least ``FSW_HEADER_SIZE`` bytes for a complete header.
    """
    if len(blob_bytes) < FSW_HEADER_SIZE:
        raise ValueError(f"Blob too small for full FSW header: {len(blob_bytes)} bytes (minimum {FSW_HEADER_SIZE})")

    with BytesIO(blob_bytes[:FSW_HEADER_SIZE]) as file:
        metadata: dict = {}
        metadata["fsw_length"] = struct.unpack("B", file.read(1))[0]

        second_byte = struct.unpack("B", file.read(1))[0]
        metadata["jpeg_bypass"] = (second_byte >> 7) & 1
        metadata["bitmask_disable"] = (second_byte >> 6) & 1
        metadata["testpattern"] = (second_byte >> 5) & 1
        metadata["bitmask_id"] = (second_byte >> 3) & 0x03
        metadata["img_mode"] = (second_byte >> 1) & 0x03

        metadata["pixel_mask_id"] = struct.unpack("B", file.read(1))[0]
        metadata["simulator"] = struct.unpack("B", file.read(1))[0]
        metadata["cadence"] = struct.unpack(">H", file.read(2))[0]
        metadata["image_total"] = struct.unpack("B", file.read(1))[0]
        metadata["image_count"] = struct.unpack("B", file.read(1))[0]
        metadata["flash_write_pointer"] = struct.unpack(">I", file.read(4))[0]
        metadata["timestamp_seconds"] = struct.unpack(">I", file.read(4))[0]
        metadata["timestamp_subseconds"] = struct.unpack(">I", file.read(4))[0]
        metadata["rad_obs_id"] = struct.unpack(">H", file.read(2))[0]
        metadata["cam_obs_id"] = struct.unpack(">H", file.read(2))[0]
        metadata["commanded_exp_time_1"] = struct.unpack(">I", file.read(4))[0]
        metadata["commanded_exp_time_2"] = struct.unpack(">I", file.read(4))[0]
        metadata["azimuth_angle"] = struct.unpack(">f", file.read(4))[0]

    return metadata


def extract_fpga_metadata_from_blob(blob_bytes: bytes) -> dict:
    """Extract FPGA header, internal footer, and status metadata from a WFOV image blob.

    Expects a full SOP slice from image start with at least ``SOP_FPGA_MIN_SIZE`` bytes
    (FSW header plus FPGA block).
    """
    if len(blob_bytes) < SOP_FPGA_MIN_SIZE:
        raise ValueError(f"Blob too small for FPGA block: {len(blob_bytes)} bytes (minimum {SOP_FPGA_MIN_SIZE})")

    fpga_bytes = blob_bytes[FSW_HEADER_SIZE:SOP_FPGA_MIN_SIZE]
    if len(fpga_bytes) != FPGA_HEADER_SIZE:
        raise ValueError(f"Expected {FPGA_HEADER_SIZE} FPGA bytes, got {len(fpga_bytes)}")

    data = swap_32bit_words(fpga_bytes)
    header = data[2:100][::2]
    footer = data[100:136][::2]

    combined: dict = {}

    combined["image_length"] = int.from_bytes(header[0:4], byteorder="little")
    combined["flags"] = int.from_bytes(header[4:5], byteorder="little")
    combined["frame_id"] = int.from_bytes(header[5:6], byteorder="little")
    combined["tag"] = int.from_bytes(header[6:14], byteorder="little")
    combined["actual_exp_time_1"] = int.from_bytes(header[14:17], byteorder="little")
    combined["temperature"] = int.from_bytes(header[17:19], byteorder="little")
    combined["gain"] = int.from_bytes(header[19:20], byteorder="little")
    combined["width"] = int.from_bytes(header[20:22], byteorder="little")
    combined["height"] = int.from_bytes(header[22:24], byteorder="little")
    combined["offset_x"] = int.from_bytes(header[24:26], byteorder="little")
    combined["offset_y"] = int.from_bytes(header[26:28], byteorder="little")
    combined["readout"] = int.from_bytes(header[28:29], byteorder="little")
    combined["actual_exp_time_2"] = int.from_bytes(header[29:32], byteorder="little")
    combined["delta"] = int.from_bytes(header[32:35], byteorder="little")
    combined["exposure_step"] = int.from_bytes(header[35:38], byteorder="little")
    combined["nr_slopes"] = int.from_bytes(header[38:39], byteorder="little")
    combined["kp1"] = int.from_bytes(header[39:42], byteorder="little")
    combined["kp2"] = int.from_bytes(header[42:45], byteorder="little")
    combined["vlow_3"] = int.from_bytes(header[45:46], byteorder="little")
    combined["vlow_2"] = int.from_bytes(header[46:47], byteorder="little")
    combined["exp_seq"] = int.from_bytes(header[47:48], byteorder="little")
    combined["footer_size"] = int.from_bytes(header[48:49], byteorder="little")

    combined["pixel_sum"] = int.from_bytes(footer[0:4], byteorder="little")
    combined["dark"] = int.from_bytes(footer[4:7], byteorder="little")
    combined["white"] = int.from_bytes(footer[7:10], byteorder="little")
    combined["footer_delta"] = int.from_bytes(footer[10:14], byteorder="little")
    combined["crc"] = int.from_bytes(footer[14:18], byteorder="little")

    fpga_status = int.from_bytes(data[136:140], byteorder="little")
    combined["sync_error"] = (fpga_status >> 0) & 0x01
    combined["pid_error"] = (fpga_status >> 1) & 0x01
    combined["size_error"] = (fpga_status >> 2) & 0x01
    combined["eop_error"] = (fpga_status >> 3) & 0x01
    combined["eep_error"] = (fpga_status >> 4) & 0x01
    combined["crc_error"] = (fpga_status >> 5) & 0x01
    combined["drop_error"] = (fpga_status >> 6) & 0x01

    return combined


def extract_compressed_payload(raw_blob: bytes) -> bytes:
    """Extract compressed JPEG-LS payload from a full stitched NAND image blob.

    Layout matches libera_cam ``extract_dict_from_bytearray`` slicing:
    ``[FSW 36][FPGA 140][payload][trailing footer 8]``.
    """
    if len(raw_blob) < MIN_STITCHED_BLOB_SIZE:
        raise ValueError(
            f"Blob too small for compressed payload extraction: {len(raw_blob)} bytes "
            f"(minimum {MIN_STITCHED_BLOB_SIZE})"
        )

    fpga_header_end = FSW_HEADER_SIZE + FPGA_HEADER_SIZE
    footer_start = len(raw_blob) - FPGA_TRAILING_FOOTER_SIZE
    if footer_start < fpga_header_end:
        raise ValueError("File structure invalid: overlapping headers and footers (negative payload size).")

    return raw_blob[fpga_header_end:footer_start]


def _decode_trailing_footer_bits(footer_bytes: bytes) -> dict:
    """Decode the 8-byte trailing NAND footer from a stitched image blob.

    Byte layout: ``pixel_sum`` (uint32 LE), ``dark`` (24-bit LE), status byte (8 flags).
    ``white``, ``delta``, and ``crc`` are filled from the FPGA internal footer when available.
    """
    if len(footer_bytes) != FPGA_TRAILING_FOOTER_SIZE:
        raise ValueError(f"Expected {FPGA_TRAILING_FOOTER_SIZE} trailing footer bytes, got {len(footer_bytes)}")

    pixel_sum = int.from_bytes(footer_bytes[0:4], byteorder="little")
    dark = footer_bytes[4] | (footer_bytes[5] << 8) | (footer_bytes[6] << 16)
    status_byte = footer_bytes[7]
    decoded: dict[str, int] = {
        "pixel_sum": pixel_sum,
        "dark": dark,
        "white": 0,
        "delta": 0,
        "crc": 0,
    }
    for bit, field_name in enumerate(TRAILING_FOOTER_STATUS_FIELDS):
        decoded[field_name] = (status_byte >> bit) & 0x01
    return decoded


def encode_trailing_footer_bytes(
    *,
    pixel_sum: int = 0,
    dark: int = 0,
    white: int = 0,
    delta: int = 0,
    crc: int = 0,
    spare: int = 0,
    drop_error: int = 0,
    crc_error: int = 0,
    eep_error: int = 0,
    eop_error: int = 0,
    size_error: int = 0,
    pid_error: int = 0,
    sync_error: int = 0,
) -> bytes:
    """Pack trailing footer fields into 8 bytes for unit tests."""
    _ = (white, delta, crc)
    status_values = {
        "spare": spare,
        "drop_error": drop_error,
        "crc_error": crc_error,
        "eep_error": eep_error,
        "eop_error": eop_error,
        "size_error": size_error,
        "pid_error": pid_error,
        "sync_error": sync_error,
    }
    status_byte = 0
    for bit, field_name in enumerate(TRAILING_FOOTER_STATUS_FIELDS):
        status_byte |= (status_values[field_name] & 0x01) << bit
    footer = bytearray(FPGA_TRAILING_FOOTER_SIZE)
    footer[0:4] = (pixel_sum & 0xFFFFFFFF).to_bytes(4, byteorder="little")
    footer[4] = dark & 0xFF
    footer[5] = (dark >> 8) & 0xFF
    footer[6] = (dark >> 16) & 0xFF
    footer[7] = status_byte
    return bytes(footer)


def extract_trailing_footer_from_blob(raw_blob: bytes) -> dict:
    """Decode trailing footer metadata from the last 8 bytes of a stitched image blob."""
    if len(raw_blob) < MIN_STITCHED_BLOB_SIZE:
        raise ValueError(
            f"Blob too small for trailing footer decode: {len(raw_blob)} bytes (minimum {MIN_STITCHED_BLOB_SIZE})"
        )

    decoded = _decode_trailing_footer_bits(raw_blob[-FPGA_TRAILING_FOOTER_SIZE:])
    try:
        fpga_meta = extract_fpga_metadata_from_blob(raw_blob)
    except (ValueError, struct.error, IndexError):
        fpga_meta = {}

    if fpga_meta:
        decoded["white"] = fpga_meta.get("white", 0)
        decoded["delta"] = fpga_meta.get("footer_delta", fpga_meta.get("delta", 0))
        decoded["crc"] = fpga_meta.get("crc", 0)

    return decoded


def validate_wfov_image_crc(compressed_payload: bytes, metadata: dict) -> int:
    """Validate WFOV image CRC over the compressed payload.

    Returns
    -------
    int
        ``1`` if CRC matches, ``0`` if mismatch, ``-1`` if validation is not performed.
    """
    _ = (compressed_payload, metadata)
    # TODO[LIBSDC-747]: Validate the CRC algorithm. Might be proprietary and we won't be able to reproduce it.
    return WFOV_CRC_VALID_NOT_VALIDATED


def _stitch_packet_range(
    packet_data: np.ndarray,
    lengths: np.ndarray,
    start_index: int,
    end_index: int,
) -> bytes:
    """Stitch packet slices from ``start_index`` through ``end_index`` inclusive."""
    packet_width = packet_data.dtype.itemsize
    packet_rows = packet_data[start_index : end_index + 1].view(np.uint8).reshape(-1, packet_width)
    packet_lengths = lengths[start_index : end_index + 1]
    parts = [packet_rows[p_idx, : packet_lengths[p_idx]].tobytes() for p_idx in range(packet_rows.shape[0])]
    return b"".join(parts)


def stitch_wfov_images(
    flags: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
    packet_data: np.ndarray,
) -> tuple[list[StitchedImage], StitchStats]:
    """Stitch complete WFOV images from mem-dump packet streams.

    State machine matches ``reassemble_image_blobs`` in libera_cam ``read_l1a_cam_data.py``.
    """
    stats = StitchStats()
    stitched_images: list[StitchedImage] = []
    image_id = 0

    state = "SEEKING"
    start_index = -1
    expected_offset = 0

    for i in range(len(flags)):
        flag = flags[i]
        offset = int(offsets[i])
        length = int(lengths[i])

        if flag == b"SOP":
            if state == "COLLECTING":
                stats.n_images_discarded_sop += 1
                stats.n_missing_sop_or_eop += 1

            state = "COLLECTING"
            start_index = i
            if offset != 0:
                stats.n_images_discarded_sop += 1
                stats.n_bad_images += 1
                state = "SEEKING"
                continue
            expected_offset = length

        elif state == "COLLECTING":
            if offset != expected_offset:
                stats.n_images_discarded_gap += 1
                stats.n_bad_images += 1
                state = "SEEKING"
                continue

            expected_offset += length

            if flag == b"EOP":
                raw_blob = _stitch_packet_range(packet_data, lengths, start_index, i)
                stitched_images.append(
                    StitchedImage(
                        image_id=image_id,
                        sop_index=start_index,
                        eop_index=i,
                        raw_blob=raw_blob,
                    )
                )
                image_id += 1
                stats.n_complete_images += 1
                state = "SEEKING"
        elif flag == b"EOP":
            stats.n_unexpected_eop += 1
            stats.n_missing_sop_or_eop += 1

    if state == "COLLECTING":
        stats.n_missing_sop_or_eop += 1

    return stitched_images, stats


def _fsw_timestamps_to_datetime64(timestamp_seconds: int, timestamp_subseconds: int) -> np.datetime64:
    meta = {"timestamp_seconds": timestamp_seconds, "timestamp_subseconds": timestamp_subseconds}
    dt = multipart_to_dt64(meta, s_field="timestamp_seconds", us_field="timestamp_subseconds")
    if isinstance(dt, pd.Series):
        dt = dt.iloc[0]
    return np.datetime64(pd.Timestamp(dt).to_datetime64(), "us")


def _extract_partial_fsw_timestamps(blob_bytes: bytes) -> tuple[int, int]:
    timestamp_seconds = struct.unpack(">I", blob_bytes[TIMESTAMP_SECONDS_OFFSET : TIMESTAMP_SECONDS_OFFSET + 4])[0]
    timestamp_subseconds = struct.unpack(
        ">I", blob_bytes[TIMESTAMP_SUBSECONDS_OFFSET : TIMESTAMP_SUBSECONDS_OFFSET + 4]
    )[0]
    return timestamp_seconds, timestamp_subseconds


def _parse_sop_row(
    blob_bytes: bytes,
) -> tuple[np.datetime64, dict, dict, bool, bool]:
    """Parse one SOP slice and return camera time, FSW dict, FPGA dict, and validity flags."""
    length = len(blob_bytes)
    fsw_meta: dict = {}
    fpga_meta: dict = {}
    fsw_valid = False
    fpga_valid = False
    camera_time = np.datetime64("NaT", "us")

    if length >= FSW_HEADER_SIZE:
        try:
            fsw_meta = extract_fsw_metadata_from_blob(blob_bytes)
            fsw_valid = True
            camera_time = _fsw_timestamps_to_datetime64(
                fsw_meta["timestamp_seconds"],
                fsw_meta["timestamp_subseconds"],
            )
        except (ValueError, struct.error):
            pass
    elif length >= FSW_TIMESTAMP_MIN_SIZE:
        try:
            _extract_partial_fsw_timestamps(blob_bytes)
        except struct.error:
            pass

    if length >= SOP_FPGA_MIN_SIZE:
        try:
            fpga_meta = extract_fpga_metadata_from_blob(blob_bytes)
            fpga_valid = True
        except (ValueError, struct.error, IndexError):
            pass

    return camera_time, fsw_meta, fpga_meta, fsw_valid, fpga_valid


def _fsw_fill_value(field: str):
    dtype = FSW_FIELD_DTYPES[field]
    if dtype.kind == "f":
        return dtype.type(np.nan)
    return dtype.type(0)


def _fpga_fill_value(field: str):
    return FPGA_FIELD_DTYPES[field].type(0)


def _trailing_fill_value(field: str):
    return TRAILING_FOOTER_FIELD_DTYPES[field].type(0)


def _build_camera_dataset(stitched_images: list[StitchedImage]) -> xr.Dataset:
    """Build CAMERA_TIME coordinate and per-image metadata for complete stitched images."""
    n_images = len(stitched_images)
    if n_images == 0:
        return xr.Dataset(coords={CAMERA_TIME_COORD: (CAMERA_TIME_COORD, np.array([], dtype=DATETIME_USEC_DTYPE))})

    payloads = [extract_compressed_payload(image.raw_blob) for image in stitched_images]
    payload_lengths = np.array([len(payload) for payload in payloads], dtype=np.uint32)
    max_payload_length = int(payload_lengths.max())
    blob_array = np.zeros((n_images, max_payload_length), dtype=np.uint8)
    for row, payload in enumerate(payloads):
        blob_array[row, : len(payload)] = np.frombuffer(payload, dtype=np.uint8)

    camera_times = np.full(n_images, np.datetime64("NaT", "us"), dtype=DATETIME_USEC_DTYPE)
    packet_indices = np.zeros(n_images, dtype=np.int32)
    fsw_parse_valid = np.zeros(n_images, dtype=bool)
    fpga_parse_valid = np.zeros(n_images, dtype=bool)
    crc_valid = np.full(n_images, WFOV_CRC_VALID_NOT_VALIDATED, dtype=np.int8)

    fsw_arrays = {field: np.zeros(n_images, dtype=FSW_FIELD_DTYPES[field]) for field in FSW_FIELDS}
    fpga_arrays = {
        field: np.zeros(n_images, dtype=FPGA_FIELD_DTYPES[field])
        for field in (*FPGA_HEADER_FIELDS, *FPGA_FOOTER_FIELDS, *FPGA_STATUS_FIELDS)
    }
    trailing_arrays = {
        field: np.zeros(n_images, dtype=TRAILING_FOOTER_FIELD_DTYPES[field])
        for field in (*TRAILING_FOOTER_VALUE_FIELDS, *TRAILING_FOOTER_STATUS_FIELDS)
    }

    for row, image in enumerate(stitched_images):
        raw_blob = image.raw_blob
        camera_time, fsw_meta, fpga_meta, fsw_valid, fpga_valid = _parse_sop_row(raw_blob)
        trailing_meta: dict = {}
        try:
            trailing_meta = extract_trailing_footer_from_blob(raw_blob)
        except ValueError:
            pass

        camera_times[row] = camera_time
        packet_indices[row] = image.sop_index
        fsw_parse_valid[row] = fsw_valid
        fpga_parse_valid[row] = fpga_valid
        crc_valid[row] = validate_wfov_image_crc(payloads[row], {**fsw_meta, **fpga_meta, **trailing_meta})

        for field in FSW_FIELDS:
            fsw_arrays[field][row] = fsw_meta.get(field, _fsw_fill_value(field))

        for field in (*FPGA_HEADER_FIELDS, *FPGA_FOOTER_FIELDS, *FPGA_STATUS_FIELDS):
            fpga_arrays[field][row] = fpga_meta.get(field, _fpga_fill_value(field))

        for field in (*TRAILING_FOOTER_VALUE_FIELDS, *TRAILING_FOOTER_STATUS_FIELDS):
            trailing_arrays[field][row] = trailing_meta.get(field, _trailing_fill_value(field))

    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        CAMERA_PACKET_INDEX_VAR: ((CAMERA_TIME_COORD,), packet_indices),
        WFOV_FSW_PARSE_VALID_VAR: ((CAMERA_TIME_COORD,), fsw_parse_valid),
        WFOV_FPGA_PARSE_VALID_VAR: ((CAMERA_TIME_COORD,), fpga_parse_valid),
        WFOV_IMAGE_BLOB_VAR: ((CAMERA_TIME_COORD, BLOB_BYTE_COORD), blob_array),
        WFOV_IMAGE_BLOB_LENGTH_VAR: ((CAMERA_TIME_COORD,), payload_lengths),
        WFOV_CRC_VALID_VAR: ((CAMERA_TIME_COORD,), crc_valid),
    }

    for field in FSW_FIELDS:
        data_vars[f"WFOV_FSW_{field.upper()}"] = ((CAMERA_TIME_COORD,), fsw_arrays[field])
    for field in (*FPGA_HEADER_FIELDS, *FPGA_FOOTER_FIELDS, *FPGA_STATUS_FIELDS):
        data_vars[f"WFOV_FPGA_{field.upper()}"] = ((CAMERA_TIME_COORD,), fpga_arrays[field])
    for field in (*TRAILING_FOOTER_VALUE_FIELDS, *TRAILING_FOOTER_STATUS_FIELDS):
        data_vars[f"WFOV_TRAILING_FOOTER_{field.upper()}"] = ((CAMERA_TIME_COORD,), trailing_arrays[field])

    coords = {
        CAMERA_TIME_COORD: (CAMERA_TIME_COORD, camera_times),
        BLOB_BYTE_COORD: (BLOB_BYTE_COORD, np.arange(max_payload_length, dtype=np.int64)),
    }
    return xr.Dataset(data_vars, coords=coords)


def _apply_packet_deduplication(packet_ds: xr.Dataset, stitched_images: list[StitchedImage]) -> xr.Dataset:
    """Zero packet payloads for complete images and assign ``PACKET_IMAGE_ID``."""
    n_packets = packet_ds.sizes["PACKET"]
    packet_image_id = np.full(n_packets, -1, dtype=np.int32)

    for image in stitched_images:
        packet_image_id[image.sop_index : image.eop_index + 1] = image.image_id

    packet_ds = packet_ds.copy(deep=False)
    if stitched_images:
        packet_data = packet_ds[WFOV_DATA_VAR].values.copy()
        packet_width = packet_data.dtype.itemsize
        packet_data_uint8 = packet_data.view(np.uint8).reshape(n_packets, packet_width)
        for image in stitched_images:
            for packet_index in range(image.sop_index, image.eop_index + 1):
                packet_data_uint8[packet_index, :] = 0
        packet_ds[WFOV_DATA_VAR] = (("PACKET",), packet_data)
    packet_ds[PACKET_IMAGE_ID_VAR] = (("PACKET",), packet_image_id)
    return packet_ds


def enhance_wfov_l1a_dataset(packet_ds: xr.Dataset) -> xr.Dataset:
    """Stitch complete WFOV images, deduplicate packet data, and attach CAMERA_TIME metadata."""
    required_vars = [MEM_DUMP_FLAGS_VAR, MEM_DUMP_OFFSET_VAR, MEM_DUMP_LENGTH_VAR, WFOV_DATA_VAR]
    missing = [name for name in required_vars if name not in packet_ds]
    if missing:
        raise ValueError(f"Missing required WFOV variables: {missing}")

    flags = packet_ds[MEM_DUMP_FLAGS_VAR].values
    offsets = packet_ds[MEM_DUMP_OFFSET_VAR].values
    lengths = packet_ds[MEM_DUMP_LENGTH_VAR].values
    packet_data = packet_ds[WFOV_DATA_VAR].values

    stitched_images, stats = stitch_wfov_images(flags, offsets, lengths, packet_data)
    packet_ds = _apply_packet_deduplication(packet_ds, stitched_images)
    camera_ds = _build_camera_dataset(stitched_images)

    packet_ds = packet_ds.merge(camera_ds)
    packet_ds.attrs["n_missing_sop_or_eop"] = stats.n_missing_sop_or_eop
    packet_ds.attrs["n_bad_images"] = stats.n_bad_images
    packet_ds.attrs["n_complete_images"] = stats.n_complete_images
    return packet_ds
