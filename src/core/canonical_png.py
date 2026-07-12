"""Small deterministic PNG encoder/validator for authoritative GA8 rasters.

The writer uses filter 0 and stored DEFLATE blocks, so exact PNG bytes do not
depend on Pillow, platform fonts, compression heuristics, or zlib versions.
Only the fixed chunk sequence used by Digital Rubbing packages is accepted.
"""

from __future__ import annotations

import binascii
import json
import struct
from typing import Any, Mapping
import zlib

import numpy as np

from .canonical_json import CanonicalJSONError, canonical_json_bytes


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_METADATA_KEYWORD = b"ArchMeshRubbing"
MAX_CANONICAL_PNG_BYTES = 64 * 1024 * 1024
MAX_CANONICAL_PNG_PIXELS = 8_000_000
MAX_CANONICAL_PNG_METADATA_BYTES = 256 * 1024
_PNG_CHUNK_ORDER = (b"IHDR", b"sRGB", b"pHYs", b"iTXt", b"IDAT", b"IEND")


class CanonicalPNGError(ValueError):
    """PNG bytes violate the authoritative deterministic subset."""


def _chunk(chunk_type: bytes, payload: bytes) -> bytes:
    crc = binascii.crc32(chunk_type)
    crc = binascii.crc32(payload, crc) & 0xFFFFFFFF
    return (
        struct.pack(">I", len(payload))
        + chunk_type
        + payload
        + struct.pack(">I", crc)
    )


def _stored_zlib_stream(payload: bytes) -> bytes:
    output = bytearray(b"\x78\x01")
    if not payload:
        output.extend(b"\x01\x00\x00\xff\xff")
    else:
        offset = 0
        while offset < len(payload):
            block = payload[offset : offset + 65_535]
            offset += len(block)
            output.append(1 if offset == len(payload) else 0)
            length = len(block)
            output.extend(struct.pack("<HH", length, 0xFFFF ^ length))
            output.extend(block)
    output.extend(struct.pack(">I", zlib.adler32(payload) & 0xFFFFFFFF))
    return bytes(output)


def _decode_stored_zlib_stream(payload: bytes, *, maximum_output: int) -> bytes:
    if len(payload) < 11 or payload[:2] != b"\x78\x01":
        raise CanonicalPNGError("PNG IDAT is not the canonical stored zlib stream")
    offset = 2
    result = bytearray()
    saw_final = False
    while not saw_final:
        if offset + 5 > len(payload) - 4:
            raise CanonicalPNGError("PNG IDAT stored block is truncated")
        header = payload[offset]
        offset += 1
        if header not in {0, 1}:
            raise CanonicalPNGError("PNG IDAT uses a non-canonical DEFLATE block")
        saw_final = header == 1
        length, complement = struct.unpack_from("<HH", payload, offset)
        offset += 4
        if complement != (0xFFFF ^ length):
            raise CanonicalPNGError("PNG IDAT stored block length check failed")
        if offset + length > len(payload) - 4:
            raise CanonicalPNGError("PNG IDAT stored block payload is truncated")
        result.extend(payload[offset : offset + length])
        offset += length
        if len(result) > maximum_output:
            raise CanonicalPNGError("PNG decompressed raster exceeds its safety limit")
    if offset != len(payload) - 4:
        raise CanonicalPNGError("PNG IDAT contains trailing DEFLATE data")
    expected_adler = struct.unpack_from(">I", payload, offset)[0]
    if expected_adler != (zlib.adler32(result) & 0xFFFFFFFF):
        raise CanonicalPNGError("PNG IDAT Adler-32 does not match")
    return bytes(result)


def _metadata_bytes(metadata: Mapping[str, Any]) -> bytes:
    if not isinstance(metadata, Mapping):
        raise CanonicalPNGError("PNG metadata must be an object")
    try:
        encoded = canonical_json_bytes(metadata)
    except CanonicalJSONError as exc:
        raise CanonicalPNGError(str(exc)) from exc
    if not encoded or len(encoded) > MAX_CANONICAL_PNG_METADATA_BYTES:
        raise CanonicalPNGError("PNG metadata byte length is outside the safety limit")
    return encoded


def encode_canonical_ga8_png(
    pixels: np.ndarray,
    *,
    pixels_per_meter: int,
    metadata: Mapping[str, Any],
) -> bytes:
    array = np.asarray(pixels)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 2:
        raise CanonicalPNGError("PNG pixels must be an HxWx2 uint8 GA8 array")
    height = int(array.shape[0])
    width = int(array.shape[1])
    if width <= 0 or height <= 0 or width * height > MAX_CANONICAL_PNG_PIXELS:
        raise CanonicalPNGError("PNG dimensions are outside the safety limit")
    if (
        isinstance(pixels_per_meter, bool)
        or not isinstance(pixels_per_meter, int)
        or pixels_per_meter <= 0
        or pixels_per_meter > 2**31 - 1
    ):
        raise CanonicalPNGError("pixels_per_meter must be a positive 31-bit integer")
    metadata_payload = _metadata_bytes(metadata)
    contiguous = np.ascontiguousarray(array)
    scanlines = bytearray()
    for row in contiguous:
        scanlines.append(0)
        scanlines.extend(row.tobytes(order="C"))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 4, 0, 0, 0)
    phys = struct.pack(">IIB", pixels_per_meter, pixels_per_meter, 1)
    international_text = (
        PNG_METADATA_KEYWORD
        + b"\x00"
        + b"\x00\x00"
        + b"\x00"
        + b"\x00"
        + metadata_payload
    )
    encoded = b"".join(
        (
            PNG_SIGNATURE,
            _chunk(b"IHDR", ihdr),
            _chunk(b"sRGB", b"\x00"),
            _chunk(b"pHYs", phys),
            _chunk(b"iTXt", international_text),
            _chunk(b"IDAT", _stored_zlib_stream(bytes(scanlines))),
            _chunk(b"IEND", b""),
        )
    )
    if len(encoded) > MAX_CANONICAL_PNG_BYTES:
        raise CanonicalPNGError("canonical PNG exceeds its byte safety limit")
    return encoded


def _strict_json_object(payload: bytes) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise CanonicalPNGError(f"PNG metadata contains invalid constant {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CanonicalPNGError(f"PNG metadata contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CanonicalPNGError("PNG metadata is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise CanonicalPNGError("PNG metadata must decode to an object")
    if _metadata_bytes(value) != payload:
        raise CanonicalPNGError("PNG metadata is not RFC 8785 canonical JSON")
    return value


def decode_canonical_ga8_png(
    png_bytes: bytes,
) -> tuple[np.ndarray, int, dict[str, Any]]:
    if not isinstance(png_bytes, bytes) or not png_bytes.startswith(PNG_SIGNATURE):
        raise CanonicalPNGError("payload is not a PNG file")
    if len(png_bytes) <= len(PNG_SIGNATURE) or len(png_bytes) > MAX_CANONICAL_PNG_BYTES:
        raise CanonicalPNGError("PNG byte length is outside the safety limit")
    offset = len(PNG_SIGNATURE)
    chunks: list[tuple[bytes, bytes]] = []
    while offset < len(png_bytes):
        if offset + 12 > len(png_bytes):
            raise CanonicalPNGError("PNG chunk is truncated")
        length = struct.unpack_from(">I", png_bytes, offset)[0]
        chunk_type = png_bytes[offset + 4 : offset + 8]
        end = offset + 12 + length
        if end > len(png_bytes):
            raise CanonicalPNGError("PNG chunk payload is truncated")
        payload = png_bytes[offset + 8 : offset + 8 + length]
        observed_crc = struct.unpack_from(">I", png_bytes, offset + 8 + length)[0]
        expected_crc = binascii.crc32(chunk_type)
        expected_crc = binascii.crc32(payload, expected_crc) & 0xFFFFFFFF
        if observed_crc != expected_crc:
            raise CanonicalPNGError("PNG chunk CRC does not match")
        chunks.append((chunk_type, payload))
        offset = end
    if tuple(chunk_type for chunk_type, _payload in chunks) != _PNG_CHUNK_ORDER:
        raise CanonicalPNGError("PNG chunk sequence is not canonical")
    chunk_map = {chunk_type: payload for chunk_type, payload in chunks}
    ihdr = chunk_map[b"IHDR"]
    if len(ihdr) != 13:
        raise CanonicalPNGError("PNG IHDR length is invalid")
    width, height, bit_depth, color_type, compression, filtering, interlace = (
        struct.unpack(">IIBBBBB", ihdr)
    )
    if (
        width <= 0
        or height <= 0
        or width * height > MAX_CANONICAL_PNG_PIXELS
        or (bit_depth, color_type, compression, filtering, interlace)
        != (8, 4, 0, 0, 0)
    ):
        raise CanonicalPNGError("PNG IHDR is outside the canonical GA8 contract")
    if chunk_map[b"sRGB"] != b"\x00":
        raise CanonicalPNGError("PNG sRGB rendering intent is invalid")
    phys = chunk_map[b"pHYs"]
    if len(phys) != 9:
        raise CanonicalPNGError("PNG pHYs length is invalid")
    x_ppm, y_ppm, unit = struct.unpack(">IIB", phys)
    if x_ppm <= 0 or x_ppm != y_ppm or unit != 1:
        raise CanonicalPNGError("PNG pHYs must declare equal pixels per metre")
    text = chunk_map[b"iTXt"]
    prefix = PNG_METADATA_KEYWORD + b"\x00\x00\x00\x00\x00"
    if not text.startswith(prefix):
        raise CanonicalPNGError("PNG iTXt metadata header is invalid")
    metadata_payload = text[len(prefix) :]
    if not metadata_payload or len(metadata_payload) > MAX_CANONICAL_PNG_METADATA_BYTES:
        raise CanonicalPNGError("PNG metadata byte length is outside the safety limit")
    metadata = _strict_json_object(metadata_payload)
    expected_scanline_bytes = height * (1 + width * 2)
    raw = _decode_stored_zlib_stream(
        chunk_map[b"IDAT"],
        maximum_output=expected_scanline_bytes,
    )
    if len(raw) != expected_scanline_bytes:
        raise CanonicalPNGError("PNG decompressed raster length is invalid")
    rows = np.frombuffer(raw, dtype=np.uint8).reshape(height, 1 + width * 2)
    if np.any(rows[:, 0] != 0):
        raise CanonicalPNGError("PNG uses a non-canonical row filter")
    pixels = np.ascontiguousarray(rows[:, 1:].reshape(height, width, 2)).copy()
    expected_png = encode_canonical_ga8_png(
        pixels,
        pixels_per_meter=x_ppm,
        metadata=metadata,
    )
    if expected_png != png_bytes:
        raise CanonicalPNGError("PNG bytes are not the canonical raster derivative")
    pixels.setflags(write=False)
    return pixels, x_ppm, metadata


__all__ = [
    "CanonicalPNGError",
    "MAX_CANONICAL_PNG_BYTES",
    "MAX_CANONICAL_PNG_METADATA_BYTES",
    "MAX_CANONICAL_PNG_PIXELS",
    "PNG_METADATA_KEYWORD",
    "PNG_SIGNATURE",
    "decode_canonical_ga8_png",
    "encode_canonical_ga8_png",
]
