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

# These flag *expected* truncation at the edges of a chunked packet window (the first packet in a
# window is rarely a qualifying SOP, and the last is rarely an EOP), deliberately kept separate from
# PACKET_COUNT_NOT_USED_IN_IMAGES_ATTR, which reflects genuine mid-stream anomalies only.
FIRST_IMAGE_INCOMPLETE_ATTR = "FirstImageIncomplete"
LAST_IMAGE_INCOMPLETE_ATTR = "LastImageIncomplete"

FSW_HEADER_SIZE = 36
FPGA_HEADER_SIZE = 140
FPGA_TRAILING_FOOTER_SIZE = 8

# The FSW and FPGA blocks are only ever decoded together (see _extract_wfov_header_metadata_from_blob),
# so this is the one size that matters for validating a header: either all of it is present or none of it is.
WFOV_HEADER_SIZE = FSW_HEADER_SIZE + FPGA_HEADER_SIZE
MIN_STITCHED_BLOB_SIZE = WFOV_HEADER_SIZE + FPGA_TRAILING_FOOTER_SIZE

MEM_DUMP_FLAGS_VAR = "ICIE__MEM_DUMP_FLAGS_WFOV"
MEM_DUMP_OFFSET_VAR = "ICIE__MEM_DUMP_OFFSET_WFOV"
MEM_DUMP_LENGTH_VAR = "ICIE__MEM_DUMP_LENGTH_WFOV"
WFOV_DATA_VAR = "ICIE__WFOV_DATA"

# This magical footer byte sequence provided to us by FSW. We speculate that it is a lower level output
# and we are assuming any issues with an image that would change these values would cause errors before
# the data could reach us, therefore our system should only see this byte stream in production.
# We validate this sequence of bytes against the footer and any mismatch results in an incrementing of
# the FooterMismatchCount.
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


@dataclass
class _StitchedImage:
    """One complete WFOV image stitched from SOP through EOP, already parsed once.

    Only the pieces ``_build_camera_dataset`` actually needs are kept — not the joined raw blob —
    since header/footer/payload parsing happens exactly once, at stitch time.

    Deliberately *not* ``frozen=True``: ``_build_camera_dataset`` clears ``payload`` as soon as each
    one has been copied into its output array, so the two never hold the same bytes at once. Because
    the destination is lazily committed (``np.zeros`` only commits pages as they are written), the
    growing output and the shrinking payload list cancel and peak stays at ~1× the image volume
    instead of 2×. Re-freezing this class would silently double peak memory over a full granule.
    """

    image_id: int
    sop_index: int
    eop_index: int
    payload: bytes
    header_meta: dict | None
    footer_valid: bool


@dataclass
class _StitchStats:
    """Counters for WFOV image stitching quality metrics."""

    n_packets_not_used_in_images: int = 0
    n_error_flagged_images: int = 0
    n_footer_mismatches: int = 0
    n_header_parse_errors: int = 0
    first_image_incomplete: bool = False
    last_image_incomplete: bool = False


def _swap_32bit_words(data: bytes) -> bytearray:
    """Swap each 4-byte word end-for-end (FPGA block endianness prep).

    Parameters
    ----------
    data : bytes
        Input byte string; length should be a multiple of 4.

    Returns
    -------
    bytearray
        Copy of ``data`` with each 32-bit word byte-reversed.
    """
    result = bytearray(len(data))
    for i in range(0, len(data), 4):
        result[i : i + 4] = data[i : i + 4][::-1]
    return result


def _extract_wfov_header_metadata_from_blob(blob_bytes: bytes) -> dict:
    """Decode the full WFOV image header (FSW block + FPGA block) as a single atomic unit.

    The FSW and FPGA sub-blocks are only ever meaningful together, so this checks the combined
    ``WFOV_HEADER_SIZE`` length once rather than validating each sub-block independently: given a
    stitched image blob, either there are enough bytes for the whole header or there aren't. The
    fields decoded here span four output categories (``WFOV_FSW_HEADER_*``, ``WFOV_IMAGE_HEADER_*``,
    ``WFOV_IMAGE_FOOTER_*``, ``WFOV_FPGA_STATUS_*``) purely for data-product naming; on the wire it's
    one 176-byte object. Byte layout matches ``read_fsw_metadata`` and the FPGA block handling in
    libera_cam ``metadata_parser.py`` / ``read_l1a_cam_data.py``.

    Parameters
    ----------
    blob_bytes : bytes
        Stitched NAND image blob (or any prefix long enough to contain the 176-byte header).

    Returns
    -------
    dict
        Flat mapping of FSW + FPGA field names to decoded Python scalars.

    Raises
    ------
    ValueError
        If ``blob_bytes`` is shorter than ``WFOV_HEADER_SIZE``.
    """
    if len(blob_bytes) < WFOV_HEADER_SIZE:
        raise ValueError(f"Blob too small for WFOV header: {len(blob_bytes)} bytes (minimum {WFOV_HEADER_SIZE})")

    metadata = _decode_fsw_header_bytes(blob_bytes[:FSW_HEADER_SIZE])
    metadata.update(_decode_fpga_block_bytes(blob_bytes[FSW_HEADER_SIZE:WFOV_HEADER_SIZE]))
    return metadata


def _decode_fsw_header_bytes(fsw_bytes: bytes) -> dict:
    """Decode the 36-byte FSW header.

    Byte layout matches ``read_fsw_metadata`` in libera_cam ``metadata_parser.py``.

    Parameters
    ----------
    fsw_bytes : bytes
        Exactly ``FSW_HEADER_SIZE`` bytes of FSW header.

    Returns
    -------
    dict
        Decoded FSW header fields.
    """
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

    Parameters
    ----------
    fpga_bytes : bytes
        Exactly ``FPGA_HEADER_SIZE`` bytes of FPGA block.

    Returns
    -------
    dict
        Decoded image-header, image-footer, and FPGA status fields.
    """
    data = _swap_32bit_words(fpga_bytes)
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


def _extract_compressed_payload(raw_blob: bytes) -> bytes:
    """Extract compressed JPEG-LS payload from a full stitched NAND image blob.

    Layout matches libera_cam ``extract_dict_from_bytearray`` slicing:
    ``[FSW 36][FPGA 140][payload][trailing footer 8]``.

    Parameters
    ----------
    raw_blob : bytes
        Complete stitched image blob including headers and trailing footer.

    Returns
    -------
    bytes
        Compressed payload bytes between the WFOV header and trailing footer.

    Raises
    ------
    ValueError
        If the blob is too small to hold the WFOV header and the trailing footer.
    """
    if len(raw_blob) < MIN_STITCHED_BLOB_SIZE:
        raise ValueError(
            f"Blob too small for compressed payload extraction: {len(raw_blob)} bytes "
            f"(minimum {MIN_STITCHED_BLOB_SIZE})"
        )

    footer_start = len(raw_blob) - FPGA_TRAILING_FOOTER_SIZE
    return raw_blob[WFOV_HEADER_SIZE:footer_start]


def _is_valid_footer_from_blob(raw_blob: bytes) -> bool:
    """Return whether the trailing 8-byte NAND footer matches ``VALID_FOOTER_BYTES``.

    Parameters
    ----------
    raw_blob : bytes
        Complete stitched image blob (or any buffer whose last 8 bytes are the footer).

    Returns
    -------
    bool
        ``True`` if the blob is long enough and the trailing footer matches the expected magic.
    """
    if len(raw_blob) < MIN_STITCHED_BLOB_SIZE:
        logger.warning(
            f"Blob too small for trailing footer decode: {len(raw_blob)} bytes (minimum {MIN_STITCHED_BLOB_SIZE})"
        )
        return False

    footer_bytes = raw_blob[-FPGA_TRAILING_FOOTER_SIZE:]
    if footer_bytes != VALID_FOOTER_BYTES:
        return False

    return True


def _stitch_wfov_images(
    flags: np.ndarray,
    offsets: np.ndarray,
    lengths: np.ndarray,
    packet_rows_u8: np.ndarray,
) -> tuple[list[_StitchedImage], _StitchStats]:
    """Stitch complete WFOV images from mem-dump packet streams.

    State machine matches ``reassemble_image_blobs`` in libera_cam ``read_l1a_cam_data.py``.

    Every packet-stream window handed to this function is expected to start mid-image (the first
    packet is usually not a qualifying ``SOP``) and end mid-image (the last is usually not an
    ``EOP``) — chunked processing always has both edges truncated. That truncation is *expected*
    and is reported via ``_StitchStats.first_image_incomplete`` / ``last_image_incomplete``, kept
    deliberately separate from ``n_packets_not_used_in_images``, which is reserved for genuine
    anomalies regardless of position: offset gaps, orphan EOPs, aborted collections, unsupported
    ``SINGLE`` packets, and images discarded for an undecodable header.

    An image is emitted only if it is structurally complete (``SOP`` through ``EOP`` with contiguous
    offsets) *and* its 176-byte header decodes, so every returned ``_StitchedImage`` has a real
    acquisition time. Structurally complete images whose header is too short are counted in
    ``n_header_parse_errors`` and dropped rather than emitted with no time.

    ``packet_rows_u8`` is read only, and is deliberately taken as a 2-D ``uint8`` view over the
    caller's packet array rather than as a list of per-packet ``bytes``. Materializing that list
    would allocate a second full-size copy of the entire packet stream up front, which coexists
    with the source array for the whole pass. Converting to ``bytes`` one packet at a time inside
    the join below keeps the transient bounded by the size of a single image instead.

    Parameters
    ----------
    flags : numpy.ndarray
        Per-packet mem-dump flags (``SOP`` / ``MOP`` / ``EOP`` / …), typically ``|S8``.
    offsets : numpy.ndarray
        Per-packet byte offsets within the reassembled image.
    lengths : numpy.ndarray
        Per-packet valid payload lengths.
    packet_rows_u8 : numpy.ndarray
        Read-only ``(n_packets, packet_width)`` ``uint8`` view of the per-packet payload bytes.

    Returns
    -------
    tuple[list[_StitchedImage], _StitchStats]
        Complete stitched images (already header/footer/payload parsed) and quality counters.
    """
    stats = _StitchStats()
    stitched_images: list[_StitchedImage] = []
    image_id = 0

    state = "SEEKING"
    start_index = -1
    expected_offset = 0
    seen_first_sop = False

    for i in range(len(flags)):
        flag = flags[i]
        offset = int(offsets[i])
        length = int(lengths[i])

        if not seen_first_sop:
            if flag in (b"SOP", b"SINGLE"):
                # SINGLE also marks an image boundary, so let it reach the warning below rather
                # than being swallowed here as leading-edge truncation.
                seen_first_sop = True
            else:
                # Leading fragment: belongs to an image whose SOP fell outside this window.
                stats.first_image_incomplete = True
                continue

        if flag == b"SINGLE":
            # SINGLE (ICIE__MEM_DUMP_FLAGS_WFOV_Type enum value 4) means a whole image fit in one
            # packet. Not expected during normal operations. Handled in its own
            # branch — before the SOP chain, so a SINGLE arriving mid-collection still aborts the
            # in-progress image instead of being absorbed as a data packet by the COLLECTING
            # branch below — and warned, so the cause is never mistaken for an
            # offset gap. No dedicated counter: the packet lands in n_packets_not_used_in_images
            # and keeps PACKET_IMAGE_ID == -1, which is the per-packet record of what was dropped.
            logger.warning(
                "Unsupported SINGLE-flagged WFOV packet at index %d (offset=%d, length=%d); dropping. "
                "This is not expected during normal operations.",
                i,
                offset,
                length,
            )
            if state == "COLLECTING":
                # A SINGLE starts a new image, so whatever was being collected is orphaned.
                stats.n_packets_not_used_in_images += i - start_index
            stats.n_packets_not_used_in_images += 1
            state = "SEEKING"
            continue

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
                raw_blob = b"".join(packet_rows_u8[k, : int(lengths[k])].tobytes() for k in range(start_index, i + 1))

                try:
                    header_meta = _extract_wfov_header_metadata_from_blob(raw_blob)
                except (ValueError, struct.error, IndexError):
                    # Only reachable when the stitched blob is shorter than WFOV_HEADER_SIZE
                    logger.warning(
                        "WFOV image starting at packet %d has an undecodable header "
                        "(%d stitched bytes, minimum %d); discarding image.",
                        start_index,
                        len(raw_blob),
                        WFOV_HEADER_SIZE,
                    )
                    stats.n_header_parse_errors += 1
                    stats.n_packets_not_used_in_images += i - start_index + 1
                    state = "SEEKING"
                    continue

                if any(header_meta[field] for field in FPGA_STATUS_FIELD_DTYPES):
                    stats.n_error_flagged_images += 1

                footer_valid = _is_valid_footer_from_blob(raw_blob)
                if not footer_valid:
                    stats.n_footer_mismatches += 1

                try:
                    payload = _extract_compressed_payload(raw_blob)
                except ValueError:
                    payload = b""

                stitched_images.append(
                    _StitchedImage(
                        image_id=image_id,
                        sop_index=start_index,
                        eop_index=i,
                        payload=payload,
                        header_meta=header_meta,
                        footer_valid=footer_valid,
                    )
                )
                image_id += 1
                state = "SEEKING"
        elif flag == b"EOP":
            # Orphan EOP with no preceding SOP.
            stats.n_packets_not_used_in_images += 1
        else:
            # MOP (or any other non-SOP/EOP flag) seen while SEEKING: a genuine mid-stream gap
            # once the leading phase is over (e.g. after a discarded collection, before the next SOP).
            stats.n_packets_not_used_in_images += 1

    if state == "COLLECTING":
        # Stream ended mid-collection; the dangling SOP never got its EOP. Expected trailing
        # truncation, not an anomaly.
        stats.last_image_incomplete = True

    return stitched_images, stats


def _fsw_timestamps_to_datetime64(timestamp_seconds: int, timestamp_subseconds: int) -> np.datetime64:
    """Convert the FSW header's split image timestamp to a ``datetime64[us]``.

    The 1958-01-01 CCSDS epoch (``multipart_to_dt64``'s default) and the treatment of
    ``timestamp_subseconds`` as a whole number of microseconds both match libera_cam
    ``read_l1a_cam_data.py`` and have been confirmed correct for this field.

    This conversion cannot fail. Both inputs are ``uint32``, so the widest possible result is
    1958-01-01 + 2**32 s + 2**32 us = 2094-02-06T07:39:49.967295, comfortably inside the
    ``datetime64[us]`` range. No input produces ``NaT`` or raises.

    Parameters
    ----------
    timestamp_seconds : int
        Whole seconds since the CCSDS epoch, from the FSW header.
    timestamp_subseconds : int
        Microseconds to add to ``timestamp_seconds``.

    Returns
    -------
    numpy.datetime64
        Image acquisition time at microsecond resolution.
    """
    meta = {"timestamp_seconds": timestamp_seconds, "timestamp_subseconds": timestamp_subseconds}
    dt = multipart_to_dt64(meta, s_field="timestamp_seconds", us_field="timestamp_subseconds")
    if isinstance(dt, pd.Series):
        dt = dt.iloc[0]
    return np.datetime64(pd.Timestamp(dt).to_datetime64(), "us")


def _field_fill_value(dtype: np.dtype):
    """Default value for a metadata field on rows where header parsing failed."""
    if dtype.kind == "f":
        return dtype.type(np.nan)
    return dtype.type(0)


# (output name prefix, field dtype dict) for each of the four data-product metadata categories
# decoded from the single WFOV header blob (see _extract_wfov_header_metadata_from_blob).
_HEADER_METADATA_CATEGORIES = (
    ("WFOV_FSW_HEADER_", FSW_HEADER_FIELD_DTYPES),
    ("WFOV_IMAGE_HEADER_", IMAGE_HEADER_FIELD_DTYPES),
    ("WFOV_IMAGE_FOOTER_", IMAGE_FOOTER_FIELD_DTYPES),
    ("WFOV_FPGA_STATUS_", FPGA_STATUS_FIELD_DTYPES),
)


def _build_camera_dataset(stitched_images: list[_StitchedImage]) -> xr.Dataset:
    """Build CAMERA_TIME coordinate and per-image metadata for complete stitched images.

    Parameters
    ----------
    stitched_images : list[_StitchedImage]
        Complete images from ``_stitch_wfov_images``.

    Returns
    -------
    xarray.Dataset
        Dataset on ``CAMERA_TIME`` with compressed payloads and decoded header fields. Empty if
        ``stitched_images`` is empty.
    """
    n_images = len(stitched_images)
    if n_images == 0:
        return xr.Dataset(coords={CAMERA_TIME_COORD: (CAMERA_TIME_COORD, np.array([], dtype=DATETIME_USEC_DTYPE))})

    payload_lengths = np.array([len(image.payload) for image in stitched_images], dtype=np.uint32)
    max_payload_length = int(payload_lengths.max())
    blob_array = np.zeros((n_images, max_payload_length), dtype=np.uint8)
    for row, image in enumerate(stitched_images):
        if image.payload:
            blob_array[row, : len(image.payload)] = np.frombuffer(image.payload, dtype=np.uint8)
        # Release each payload the moment blob_array owns its bytes, so the two never hold the same
        # image at once; see _StitchedImage on why this class stays mutable. Nothing downstream reads
        # payload — enhance_wfov_l1a_dataset only uses sop_index/eop_index/image_id and the count.
        image.payload = b""

    camera_times = np.full(n_images, np.datetime64("NaT", "us"), dtype=DATETIME_USEC_DTYPE)
    packet_indices = np.zeros(n_images, dtype=np.int32)
    header_parse_valid = np.zeros(n_images, dtype=bool)

    category_arrays = [
        {field: np.zeros(n_images, dtype=dtype) for field, dtype in field_dtypes.items()}
        for _, field_dtypes in _HEADER_METADATA_CATEGORIES
    ]

    for row, image in enumerate(stitched_images):
        header_meta = image.header_meta
        packet_indices[row] = image.sop_index
        header_parse_valid[row] = header_meta is not None

        if header_meta is not None:
            camera_times[row] = _fsw_timestamps_to_datetime64(
                header_meta["timestamp_seconds"],
                header_meta["timestamp_subseconds"],
            )

        for arrays, (_, field_dtypes) in zip(category_arrays, _HEADER_METADATA_CATEGORIES):
            for field, dtype in field_dtypes.items():
                arrays[field][row] = (header_meta or {}).get(field, _field_fill_value(dtype))

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


def enhance_wfov_l1a_dataset(packet_ds: xr.Dataset) -> xr.Dataset:
    """Stitch complete WFOV images, drop raw packet payloads, and attach CAMERA_TIME metadata.

    Raises if no image completes, before dropping anything, so a granule that produces nothing
    keeps its raw payload for diagnosis instead of being silently emptied.

    ``ICIE__WFOV_DATA`` is dropped from the returned dataset entirely, for every packet regardless
    of completeness: content folded into a complete image is already duplicated (compressed) on
    ``CAMERA_TIME`` as ``WFOV_COMPRESSED_IMAGE``, and content that never completed an image (edge
    truncation or a mid-stream anomaly) cannot be assembled from this granule alone — the raise
    above guarantees at least one image did complete before any payload is discarded, and
    ``PacketCountNotUsedInImages`` plus ``PACKET_IMAGE_ID == -1`` record what was dropped.
    ``PACKET_IMAGE_ID`` remains the only
    per-packet trace-back to a stitched image (``-1`` if none). Dropping the whole variable, rather
    than zeroing it row by row, avoids a second full-size copy of the ~GB-scale raw array: zeroing
    in place would corrupt the caller's dataset, since ``copy(deep=False)`` shares the buffer. Note
    that ``drop_vars`` here does not itself free that buffer — ``parse_packets_to_l1a_dataset``
    still holds the original ``packet_ds`` across this call, so the array survives until the caller
    rebinds on return.

    Parameters
    ----------
    packet_ds : xarray.Dataset
        APID 1040 packet-level L1A dataset containing mem-dump flag/offset/length fields and
        ``ICIE__WFOV_DATA``.

    Returns
    -------
    xarray.Dataset
        Input dataset merged with ``CAMERA_TIME`` metadata, ``PACKET_IMAGE_ID``, and file-level
        stitch quality attributes; ``ICIE__WFOV_DATA`` is absent.

    Raises
    ------
    ValueError
        If required WFOV packet variables are missing from ``packet_ds``, or if no complete image
        could be stitched from the packet stream. In the latter case ``packet_ds`` is left
        untouched, so the caller still holds the raw payloads.
    """
    required_vars = [MEM_DUMP_FLAGS_VAR, MEM_DUMP_OFFSET_VAR, MEM_DUMP_LENGTH_VAR, WFOV_DATA_VAR]
    missing = [name for name in required_vars if name not in packet_ds]
    if missing:
        raise ValueError(f"Missing required WFOV variables: {missing}")

    flags = packet_ds[MEM_DUMP_FLAGS_VAR].values
    offsets = packet_ds[MEM_DUMP_OFFSET_VAR].values
    lengths = packet_ds[MEM_DUMP_LENGTH_VAR].values
    n_packets = packet_ds.sizes["PACKET"]

    # Hand the stitcher a zero-copy uint8 view of the packet array, not a list of per-packet bytes:
    # building that list would allocate a second full-size copy of the whole stream that
    # lives alongside the original for the entire pass. A uint8 view rather than ndarray.tolist()
    # because the fixed-width |S dtype silently strips trailing null bytes on conversion to Python
    # bytes, which would corrupt any packet whose valid payload run (per ICIE__MEM_DUMP_LENGTH_WFOV)
    # itself ends in zero bytes.
    packet_width = packet_ds[WFOV_DATA_VAR].dtype.itemsize
    packet_rows_u8 = packet_ds[WFOV_DATA_VAR].values.view(np.uint8).reshape(n_packets, packet_width)

    stitched_images, stats = _stitch_wfov_images(flags, offsets, lengths, packet_rows_u8)

    # Fail if no images were successfully made
    if not stitched_images:
        raise ValueError(
            f"No complete WFOV images could be produced from {n_packets} APID 1040 packets. "
            f"packets_not_used={stats.n_packets_not_used_in_images}, "
            f"header_parse_errors={stats.n_header_parse_errors}, "
            f"first_image_incomplete={stats.first_image_incomplete}, "
            f"last_image_incomplete={stats.last_image_incomplete}"
        )

    del packet_rows_u8  # a view over the same buffer as WFOV_DATA_VAR; drop it before dropping the var
    packet_ds = packet_ds.drop_vars(WFOV_DATA_VAR)

    packet_image_id = np.full(n_packets, -1, dtype=np.int32)
    for image in stitched_images:
        packet_image_id[image.sop_index : image.eop_index + 1] = image.image_id
    packet_ds[PACKET_IMAGE_ID_VAR] = (("PACKET",), packet_image_id)

    camera_ds = _build_camera_dataset(stitched_images)

    packet_ds = packet_ds.merge(camera_ds)
    packet_ds.attrs[PACKET_COUNT_NOT_USED_IN_IMAGES_ATTR] = stats.n_packets_not_used_in_images
    packet_ds.attrs[ERROR_FLAGGED_IMAGE_COUNT_ATTR] = stats.n_error_flagged_images
    packet_ds.attrs[FOOTER_MISMATCH_COUNT_ATTR] = stats.n_footer_mismatches
    packet_ds.attrs[HEADER_PARSE_ERROR_COUNT_ATTR] = stats.n_header_parse_errors
    # Stored as 0/1 rather than Python bool: NetCDF (via h5netcdf) has no boolean attribute type.
    packet_ds.attrs[FIRST_IMAGE_INCOMPLETE_ATTR] = int(stats.first_image_incomplete)
    packet_ds.attrs[LAST_IMAGE_INCOMPLETE_ATTR] = int(stats.last_image_incomplete)
    if stats.n_error_flagged_images:
        logger.warning(
            "WFOV images with FPGA status errors flagged: %d of %d",
            stats.n_error_flagged_images,
            len(stitched_images),
        )
    return packet_ds
