from __future__ import annotations

import hashlib
import io
import struct
import unittest

import numpy as np
from PIL import Image

from src.core.canonical_png import (
    CanonicalPNGError,
    PNG_SIGNATURE,
    decode_canonical_ga8_png,
    encode_canonical_ga8_png,
)


def _pixels() -> np.ndarray:
    pixels = np.zeros((2, 3, 2), dtype=np.uint8)
    pixels[:, :, 0] = np.array([[255, 128, 0], [10, 20, 30]], dtype=np.uint8)
    pixels[:, :, 1] = np.array([[255, 255, 0], [0, 255, 255]], dtype=np.uint8)
    return pixels


def _chunk_types(payload: bytes) -> list[bytes]:
    offset = len(PNG_SIGNATURE)
    result: list[bytes] = []
    while offset < len(payload):
        length = struct.unpack_from(">I", payload, offset)[0]
        result.append(payload[offset + 4 : offset + 8])
        offset += 12 + length
    return result


class TestCanonicalPNG(unittest.TestCase):
    def test_exact_bytes_round_trip_scale_and_pillow_compatibility(self):
        metadata = {"a": 1, "한글": "값"}
        encoded = encode_canonical_ga8_png(
            _pixels(),
            pixels_per_meter=10_000,
            metadata=metadata,
        )
        self.assertEqual(
            _chunk_types(encoded),
            [b"IHDR", b"sRGB", b"pHYs", b"iTXt", b"IDAT", b"IEND"],
        )
        self.assertEqual(
            hashlib.sha256(encoded).hexdigest(),
            "f5111659e69de013fa8dcd7a622d3ef1ccb7382d1c3c72e6e3b2e1f5e9accaf2",
        )
        decoded, ppm, decoded_metadata = decode_canonical_ga8_png(encoded)
        np.testing.assert_array_equal(decoded, _pixels())
        self.assertEqual(ppm, 10_000)
        self.assertEqual(decoded_metadata, metadata)

        image = Image.open(io.BytesIO(encoded))
        self.assertEqual(image.mode, "LA")
        self.assertEqual(image.size, (3, 2))
        self.assertAlmostEqual(image.info["dpi"][0], 254.0, places=2)

    def test_output_has_no_time_exif_icc_or_noncanonical_text_chunks(self):
        encoded = encode_canonical_ga8_png(
            _pixels(),
            pixels_per_meter=20_000,
            metadata={"format": "test"},
        )
        for forbidden in (b"tIME", b"eXIf", b"iCCP", b"tEXt", b"zTXt"):
            self.assertNotIn(forbidden, _chunk_types(encoded))

    def test_crc_trailing_chunk_and_noncanonical_stream_are_rejected(self):
        encoded = encode_canonical_ga8_png(
            _pixels(),
            pixels_per_meter=10_000,
            metadata={"a": 1},
        )
        corrupted = bytearray(encoded)
        corrupted[-5] ^= 1
        with self.assertRaisesRegex(CanonicalPNGError, "CRC"):
            decode_canonical_ga8_png(bytes(corrupted))

        with self.assertRaisesRegex(CanonicalPNGError, "chunk"):
            decode_canonical_ga8_png(encoded + b"trailing")

        idat = encoded.find(b"IDAT")
        self.assertGreater(idat, 0)
        corrupted = bytearray(encoded)
        corrupted[idat + 5] = 0x09
        with self.assertRaises(CanonicalPNGError):
            decode_canonical_ga8_png(bytes(corrupted))

    def test_invalid_pixels_scale_and_metadata_fail_closed(self):
        with self.assertRaisesRegex(CanonicalPNGError, "HxWx2"):
            encode_canonical_ga8_png(
                np.zeros((2, 2), dtype=np.uint8),
                pixels_per_meter=10_000,
                metadata={"a": 1},
            )
        with self.assertRaisesRegex(CanonicalPNGError, "pixels_per_meter"):
            encode_canonical_ga8_png(
                _pixels(),
                pixels_per_meter=0,
                metadata={"a": 1},
            )
        with self.assertRaisesRegex(CanonicalPNGError, "canonicalized"):
            encode_canonical_ga8_png(
                _pixels(),
                pixels_per_meter=10_000,
                metadata={"value": float("nan")},
            )


if __name__ == "__main__":
    unittest.main()
