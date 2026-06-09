"""WFOV camera image FSW and FPGA metadata extraction for L1A CAMERA_TIME."""

import struct
from io import BytesIO

import numpy as np
import pandas as pd
import xarray as xr

from libera_utils.time import multipart_to_dt64

CAMERA_TIME_COORD = "CAMERA_TIME"
CAMERA_PACKET_INDEX_VAR = "CAMERA_PACKET_INDEX"
WFOV_FSW_PARSE_VALID_VAR = "WFOV_FSW_PARSE_VALID"
WFOV_FPGA_PARSE_VALID_VAR = "WFOV_FPGA_PARSE_VALID"
WFOV_IMAGE_COMPLETE_VAR = "WFOV_IMAGE_COMPLETE"
FSW_HEADER_SIZE = 36
FPGA_HEADER_SIZE = 140
SOP_FPGA_MIN_SIZE = FSW_HEADER_SIZE + FPGA_HEADER_SIZE
FSW_TIMESTAMP_MIN_SIZE = 20
TIMESTAMP_SECONDS_OFFSET = 12
TIMESTAMP_SUBSECONDS_OFFSET = 16

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

    Expects ``blob_bytes`` to begin at the FPGA block (offset 36 from image start) or to be a
    full SOP slice with at least ``SOP_FPGA_MIN_SIZE`` bytes.
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


def find_qualifying_sop_indices(flags: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Return packet indices for SOP packets with zero mem-dump offset, in packet order."""
    flags_str = flags.astype(str)
    return np.flatnonzero((flags_str == "SOP") & (offsets == 0))


def assess_wfov_image_completeness(
    flags: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
) -> np.ndarray:
    """Return per-packet bool indicating whether a qualifying SOP starts a complete image.

    State machine matches ``reassemble_image_blobs`` in libera_cam ``read_l1a_cam_data.py``.
    """
    flags_str = flags.astype(str)
    n_packets = len(flags_str)
    complete = np.zeros(n_packets, dtype=bool)

    for i in range(n_packets):
        if flags_str[i] != "SOP" or offsets[i] != 0:
            continue

        expected_offset = lengths[i]
        j = i + 1
        while j < n_packets:
            if offsets[j] != expected_offset:
                break
            expected_offset += lengths[j]
            if flags_str[j] == "EOP":
                complete[i] = True
                break
            if flags_str[j] == "SOP":
                break
            j += 1

    return complete


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


def _sop_blob_bytes(packet_data: np.ndarray, sop_index: int, length: int) -> bytes:
    """Extract SOP bytes without numpy fixed-width string null stripping."""
    packet_width = packet_data.dtype.itemsize
    row = packet_data.view(np.uint8).reshape(-1, packet_width)[sop_index]
    return row[:length].tobytes()


def _build_metadata_arrays(
    sop_indices: np.ndarray,
    packet_data: np.ndarray,
    lengths: np.ndarray,
    image_complete: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    n_images = sop_indices.size
    camera_times = np.full(n_images, np.datetime64("NaT", "us"), dtype=DATETIME_USEC_DTYPE)
    packet_indices = np.zeros(n_images, dtype=np.int32)
    fsw_parse_valid = np.zeros(n_images, dtype=bool)
    fpga_parse_valid = np.zeros(n_images, dtype=bool)
    image_complete_flags = np.zeros(n_images, dtype=bool)

    fsw_arrays = {field: np.zeros(n_images, dtype=FSW_FIELD_DTYPES[field]) for field in FSW_FIELDS}
    fpga_arrays = {
        field: np.zeros(n_images, dtype=FPGA_FIELD_DTYPES[field])
        for field in (*FPGA_HEADER_FIELDS, *FPGA_FOOTER_FIELDS, *FPGA_STATUS_FIELDS)
    }

    for row, sop_index in enumerate(sop_indices):
        sop_index = int(sop_index)
        blob = _sop_blob_bytes(packet_data, sop_index, int(lengths[sop_index]))
        camera_time, fsw_meta, fpga_meta, fsw_valid, fpga_valid = _parse_sop_row(blob)

        camera_times[row] = camera_time
        packet_indices[row] = sop_index
        fsw_parse_valid[row] = fsw_valid
        fpga_parse_valid[row] = fpga_valid
        image_complete_flags[row] = bool(image_complete[sop_index])

        for field in FSW_FIELDS:
            if field in fsw_meta:
                fsw_arrays[field][row] = fsw_meta[field]
            else:
                fsw_arrays[field][row] = _fsw_fill_value(field)

        for field in (*FPGA_HEADER_FIELDS, *FPGA_FOOTER_FIELDS, *FPGA_STATUS_FIELDS):
            if field in fpga_meta:
                fpga_arrays[field][row] = fpga_meta[field]
            else:
                fpga_arrays[field][row] = _fpga_fill_value(field)

    arrays: dict[str, np.ndarray] = {
        CAMERA_PACKET_INDEX_VAR: packet_indices,
        WFOV_FSW_PARSE_VALID_VAR: fsw_parse_valid,
        WFOV_FPGA_PARSE_VALID_VAR: fpga_parse_valid,
        WFOV_IMAGE_COMPLETE_VAR: image_complete_flags,
    }
    for field in FSW_FIELDS:
        arrays[f"WFOV_FSW_{field.upper()}"] = fsw_arrays[field]
    for field in (*FPGA_HEADER_FIELDS, *FPGA_FOOTER_FIELDS, *FPGA_STATUS_FIELDS):
        arrays[f"WFOV_FPGA_{field.upper()}"] = fpga_arrays[field]

    return camera_times, arrays


def build_wfov_camera_metadata_dataset(packet_ds: xr.Dataset) -> xr.Dataset:
    """Build CAMERA_TIME coordinate and per-image metadata from WFOV L1A packet data."""
    required_vars = [MEM_DUMP_FLAGS_VAR, MEM_DUMP_OFFSET_VAR, MEM_DUMP_LENGTH_VAR, WFOV_DATA_VAR]
    missing = [name for name in required_vars if name not in packet_ds]
    if missing:
        raise ValueError(f"Missing required WFOV variables: {missing}")

    flags = packet_ds[MEM_DUMP_FLAGS_VAR].values
    offsets = packet_ds[MEM_DUMP_OFFSET_VAR].values
    lengths = packet_ds[MEM_DUMP_LENGTH_VAR].values
    packet_data = packet_ds[WFOV_DATA_VAR].values

    sop_indices = find_qualifying_sop_indices(flags, offsets)
    image_complete = assess_wfov_image_completeness(flags, offsets, lengths)

    if sop_indices.size == 0:
        return xr.Dataset(coords={CAMERA_TIME_COORD: (CAMERA_TIME_COORD, np.array([], dtype=DATETIME_USEC_DTYPE))})

    camera_times, arrays = _build_metadata_arrays(sop_indices, packet_data, lengths, image_complete)

    data_vars = {name: ((CAMERA_TIME_COORD,), values) for name, values in arrays.items()}
    return xr.Dataset(data_vars, coords={CAMERA_TIME_COORD: (CAMERA_TIME_COORD, camera_times)})
