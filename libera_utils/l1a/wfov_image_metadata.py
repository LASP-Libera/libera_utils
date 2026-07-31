"""WFOV camera image stitching, metadata extraction, and L1A CAMERA_TIME enhancement."""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from io import BytesIO

import numpy as np
import pandas as pd
import xarray as xr

from libera_utils.time import multipart_to_dt64

logger = logging.getLogger(__name__)

CAMERA_TIME_COORD = "CAMERA_TIME"
BLOB_BYTE_COORD = "BLOB_BYTE"
CAMERA_PACKET_INDEX_VAR = "CAMERA_PACKET_INDEX"
PACKET_IMAGE_ID_VAR = "PACKET_IMAGE_ID"
WFOV_HEADER_PARSE_VALID_VAR = "WFOV_HEADER_PARSE_VALID"
WFOV_COMPRESSED_IMAGE_VAR = "WFOV_COMPRESSED_IMAGE"
WFOV_COMPRESSED_IMAGE_LENGTH_VAR = "WFOV_COMPRESSED_IMAGE_LENGTH"

PACKET_COUNT_NOT_USED_IN_IMAGES_ATTR = "PacketCountNotUsedInImages"
ERROR_FLAGGED_IMAGE_COUNT_ATTR = "ErrorFlaggedImageCount"
FOOTER_MISMATCH_COUNT_ATTR = "FooterMismatchCount"
HEADER_PARSE_ERROR_COUNT_ATTR = "HeaderParseErrorCount"

FSW_HEADER_SIZE = 36
FPGA_HEADER_SIZE = 140
FPGA_TRAILING_FOOTER_SIZE = 8
# The FSW and FPGA blocks are only ever decoded together (see extract_wfov_header_metadata_from_blob),
# so this is the one size that matters for validating a header: either all of it is present or none of it is.
WFOV_HEADER_SIZE = FSW_HEADER_SIZE + FPGA_HEADER_SIZE
MIN_STITCHED_BLOB_SIZE = WFOV_HEADER_SIZE + FPGA_TRAILING_FOOTER_SIZE

MEM_DUMP_FLAGS_VAR = "ICIE__MEM_DUMP_FLAGS_WFOV"
MEM_DUMP_OFFSET_VAR = "ICIE__MEM_DUMP_OFFSET_WFOV"
MEM_DUMP_LENGTH_VAR = "ICIE__MEM_DUMP_LENGTH_WFOV"
WFOV_DATA_VAR = "ICIE__WFOV_DATA"

VALID_FOOTER_BYTES = b"\xff\xfd\x00\x00\xff\xfc\x00\x01"

DATETIME_USEC_DTYPE = np.dtype("datetime64[us]")

# Each dict below is the single source of truth for both field naming/order and storage dtype
# for one metadata block; iterate the dict directly (or `tuple(...)`) wherever the field-name
# list is needed instead of keeping a parallel names-only tuple in sync by hand.
FSW_HEADER_FIELD_DTYPES: dict[str, np.dtype] = {
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

IMAGE_HEADER_FIELD_DTYPES: dict[str, np.dtype] = {
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
}

IMAGE_FOOTER_FIELD_DTYPES: dict[str, np.dtype] = {
    "pixel_sum": np.dtype("uint32"),
    "dark": np.dtype("uint32"),
    "white": np.dtype("uint32"),
    "footer_delta": np.dtype("uint32"),
    "crc": np.dtype("uint32"),
}

FPGA_STATUS_FIELD_DTYPES: dict[str, np.dtype] = {
    "sync_error": np.dtype("uint8"),
    "pid_error": np.dtype("uint8"),
    "size_error": np.dtype("uint8"),
    "eop_error": np.dtype("uint8"),
    "eep_error": np.dtype("uint8"),
    "crc_error": np.dtype("uint8"),
    "drop_error": np.dtype("uint8"),
}

# Combined view of every field decoded from the 140-byte FPGA block (image header + image footer +
# status) — the region of the WFOV header that isn't the FSW block.
FPGA_BLOCK_FIELD_DTYPES: dict[str, np.dtype] = {
    **IMAGE_HEADER_FIELD_DTYPES,
    **IMAGE_FOOTER_FIELD_DTYPES,
    **FPGA_STATUS_FIELD_DTYPES,
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

    n_packets_not_used_in_images: int = 0
    n_error_flagged_images: int = 0
    n_footer_mismatches: int = 0
    n_header_parse_errors: int = 0


def swap_32bit_words(data: bytes) -> bytearray:
    """Swap 32-bit words in the data."""
    result = bytearray(len(data))
    for i in range(0, len(data), 4):
        result[i : i + 4] = data[i : i + 4][::-1]
    return result


def extract_wfov_header_metadata_from_blob(blob_bytes: bytes) -> dict:
    """Decode the full WFOV image header (FSW block + FPGA block) as a single atomic unit.

    The FSW and FPGA sub-blocks are only ever meaningful together, so this checks the combined
    ``WFOV_HEADER_SIZE`` length once rather than validating each sub-block independently: given a
    stitched image blob, either there are enough bytes for the whole header or there aren't. The
    fields decoded here span four output categories (``WFOV_FSW_HEADER_*``, ``WFOV_IMAGE_HEADER_*``,
    ``WFOV_IMAGE_FOOTER_*``, ``WFOV_FPGA_STATUS_*``) purely for data-product naming; on the wire it's
    one 176-byte object. Byte layout matches ``read_fsw_metadata`` and the FPGA block handling in
    libera_cam ``metadata_parser.py`` / ``read_l1a_cam_data.py``.
    """
    if len(blob_bytes) < WFOV_HEADER_SIZE:
        raise ValueError(f"Blob too small for WFOV header: {len(blob_bytes)} bytes (minimum {WFOV_HEADER_SIZE})")

    metadata = _decode_fsw_header_bytes(blob_bytes[:FSW_HEADER_SIZE])
    metadata.update(_decode_fpga_block_bytes(blob_bytes[FSW_HEADER_SIZE:WFOV_HEADER_SIZE]))
    return metadata


def _decode_fsw_header_bytes(fsw_bytes: bytes) -> dict:
    """Decode the 36-byte FSW header. Byte layout matches ``read_fsw_metadata`` in libera_cam ``metadata_parser.py``."""
    with BytesIO(fsw_bytes) as file:
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


def _decode_fpga_block_bytes(fpga_bytes: bytes) -> dict:
    """Decode the 140-byte FPGA block (image header + image footer + status flags).

    Byte layout matches the FPGA block handling in libera_cam ``read_l1a_cam_data.py``.
    """
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

    footer_start = len(raw_blob) - FPGA_TRAILING_FOOTER_SIZE
    if footer_start < WFOV_HEADER_SIZE:
        raise ValueError("File structure invalid: overlapping headers and footers (negative payload size).")

    return raw_blob[WFOV_HEADER_SIZE:footer_start]


def is_valid_footer_from_blob(raw_blob: bytes) -> dict:
    """Decode trailing footer metadata from a stitched image blob. Compares against a known hex string and returns False if it doesn't match."""
    if len(raw_blob) < MIN_STITCHED_BLOB_SIZE:
        logger.warning(
            f"Blob too small for trailing footer decode: {len(raw_blob)} bytes (minimum {MIN_STITCHED_BLOB_SIZE})"
        )
        return False

    footer_bytes = raw_blob[-FPGA_TRAILING_FOOTER_SIZE:]
    if footer_bytes != VALID_FOOTER_BYTES:
        return False

    return True


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
                # Prior SOP never reached an EOP; every packet it collected goes unused.
                stats.n_packets_not_used_in_images += i - start_index

            state = "COLLECTING"
            start_index = i
            if offset != 0:
                stats.n_packets_not_used_in_images += 1
                state = "SEEKING"
                continue
            expected_offset = length

        elif state == "COLLECTING":
            if offset != expected_offset:
                # Offset gap: discard this packet plus everything collected since start_index.
                stats.n_packets_not_used_in_images += i - start_index + 1
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
                try:
                    header_meta = extract_wfov_header_metadata_from_blob(raw_blob)
                except (ValueError, struct.error, IndexError):
                    header_meta = None

                if header_meta is None:
                    stats.n_header_parse_errors += 1
                elif any(header_meta[field] for field in FPGA_STATUS_FIELD_DTYPES):
                    stats.n_error_flagged_images += 1

                if not is_valid_footer_from_blob(raw_blob):
                    stats.n_footer_mismatches += 1

                image_id += 1
                state = "SEEKING"
        elif flag == b"EOP":
            # Orphan EOP with no preceding SOP.
            stats.n_packets_not_used_in_images += 1

    if state == "COLLECTING":
        # Stream ended mid-collection; the dangling SOP never got its EOP.
        stats.n_packets_not_used_in_images += len(flags) - start_index

    return stitched_images, stats


def _fsw_timestamps_to_datetime64(timestamp_seconds: int, timestamp_subseconds: int) -> np.datetime64:
    meta = {"timestamp_seconds": timestamp_seconds, "timestamp_subseconds": timestamp_subseconds}
    dt = multipart_to_dt64(meta, s_field="timestamp_seconds", us_field="timestamp_subseconds")
    if isinstance(dt, pd.Series):
        dt = dt.iloc[0]
    return np.datetime64(pd.Timestamp(dt).to_datetime64(), "us")


def _parse_sop_row(blob_bytes: bytes) -> tuple[np.datetime64, dict, bool]:
    """Parse one SOP slice and return camera time, the decoded header dict, and its validity."""
    header_meta: dict = {}
    header_valid = False
    camera_time = np.datetime64("NaT", "us")

    try:
        header_meta = extract_wfov_header_metadata_from_blob(blob_bytes)
        header_valid = True
        camera_time = _fsw_timestamps_to_datetime64(
            header_meta["timestamp_seconds"],
            header_meta["timestamp_subseconds"],
        )
    except (ValueError, struct.error, IndexError):
        pass

    return camera_time, header_meta, header_valid


def _field_fill_value(dtype: np.dtype):
    """Default value for a metadata field on rows where header parsing failed."""
    if dtype.kind == "f":
        return dtype.type(np.nan)
    return dtype.type(0)


# (output name prefix, field dtype dict) for each of the four data-product metadata categories
# decoded from the single WFOV header blob (see extract_wfov_header_metadata_from_blob).
_HEADER_METADATA_CATEGORIES = (
    ("WFOV_FSW_HEADER_", FSW_HEADER_FIELD_DTYPES),
    ("WFOV_IMAGE_HEADER_", IMAGE_HEADER_FIELD_DTYPES),
    ("WFOV_IMAGE_FOOTER_", IMAGE_FOOTER_FIELD_DTYPES),
    ("WFOV_FPGA_STATUS_", FPGA_STATUS_FIELD_DTYPES),
)


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
    header_parse_valid = np.zeros(n_images, dtype=bool)

    category_arrays = [
        {field: np.zeros(n_images, dtype=dtype) for field, dtype in field_dtypes.items()}
        for _, field_dtypes in _HEADER_METADATA_CATEGORIES
    ]

    for row, image in enumerate(stitched_images):
        raw_blob = image.raw_blob
        camera_time, header_meta, header_valid = _parse_sop_row(raw_blob)

        camera_times[row] = camera_time
        packet_indices[row] = image.sop_index
        header_parse_valid[row] = header_valid

        for arrays, (_, field_dtypes) in zip(category_arrays, _HEADER_METADATA_CATEGORIES):
            for field, dtype in field_dtypes.items():
                arrays[field][row] = header_meta.get(field, _field_fill_value(dtype))

    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        CAMERA_PACKET_INDEX_VAR: ((CAMERA_TIME_COORD,), packet_indices),
        WFOV_HEADER_PARSE_VALID_VAR: ((CAMERA_TIME_COORD,), header_parse_valid),
        WFOV_COMPRESSED_IMAGE_VAR: ((CAMERA_TIME_COORD, BLOB_BYTE_COORD), blob_array),
        WFOV_COMPRESSED_IMAGE_LENGTH_VAR: ((CAMERA_TIME_COORD,), payload_lengths),
    }

    for arrays, (prefix, field_dtypes) in zip(category_arrays, _HEADER_METADATA_CATEGORIES):
        for field in field_dtypes:
            data_vars[f"{prefix}{field.upper()}"] = ((CAMERA_TIME_COORD,), arrays[field])

    coords = {
        CAMERA_TIME_COORD: (CAMERA_TIME_COORD, camera_times),
        BLOB_BYTE_COORD: (BLOB_BYTE_COORD, np.arange(max_payload_length, dtype=np.int64)),
    }
    return xr.Dataset(data_vars, coords=coords)


def _zero_stitched_packet_payloads(packet_ds: xr.Dataset, stitched_images: list[StitchedImage]) -> xr.Dataset:
    """Zero out per-packet payloads once folded into a stitched image, and tag ``PACKET_IMAGE_ID``.

    This is unrelated to packet-row deduplication (handled earlier by ``packets._drop_duplicates``
    on packet timestamps). Here, no packets are dropped; instead the raw ``ICIE__WFOV_DATA`` slices
    for packets contributing to a *complete* stitched image are zeroed, since that image content is
    now stored once on ``CAMERA_TIME`` as ``WFOV_COMPRESSED_IMAGE`` and would otherwise be duplicated
    in the file. ``PACKET_IMAGE_ID`` traces each packet back to its image (``-1`` if not part of one).
    """
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
    """Stitch complete WFOV images, zero out redundant packet payloads, and attach CAMERA_TIME metadata."""
    required_vars = [MEM_DUMP_FLAGS_VAR, MEM_DUMP_OFFSET_VAR, MEM_DUMP_LENGTH_VAR, WFOV_DATA_VAR]
    missing = [name for name in required_vars if name not in packet_ds]
    if missing:
        raise ValueError(f"Missing required WFOV variables: {missing}")

    flags = packet_ds[MEM_DUMP_FLAGS_VAR].values
    offsets = packet_ds[MEM_DUMP_OFFSET_VAR].values
    lengths = packet_ds[MEM_DUMP_LENGTH_VAR].values
    packet_data = packet_ds[WFOV_DATA_VAR].values

    stitched_images, stats = stitch_wfov_images(flags, offsets, lengths, packet_data)
    packet_ds = _zero_stitched_packet_payloads(packet_ds, stitched_images)
    camera_ds = _build_camera_dataset(stitched_images)

    packet_ds = packet_ds.merge(camera_ds)
    packet_ds.attrs[PACKET_COUNT_NOT_USED_IN_IMAGES_ATTR] = stats.n_packets_not_used_in_images
    packet_ds.attrs[ERROR_FLAGGED_IMAGE_COUNT_ATTR] = stats.n_error_flagged_images
    packet_ds.attrs[FOOTER_MISMATCH_COUNT_ATTR] = stats.n_footer_mismatches
    packet_ds.attrs[HEADER_PARSE_ERROR_COUNT_ATTR] = stats.n_header_parse_errors
    if stats.n_error_flagged_images:
        logger.warning(
            "WFOV images with FPGA status errors flagged: %d of %d",
            stats.n_error_flagged_images,
            len(stitched_images),
        )
    return packet_ds
