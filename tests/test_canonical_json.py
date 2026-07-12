from __future__ import annotations

from types import MappingProxyType
import unittest

import numpy as np

from src.core.canonical_json import (
    CanonicalJSONError,
    canonical_json_bytes,
    canonical_json_sha256,
)


class TestRFC8785CanonicalJSON(unittest.TestCase):
    def test_cross_language_number_golden(self):
        value = {
            "numbers": [333333333.33333329, 1e30, 4.5, 2e-3, 1e-27, -0.0],
            "small": 1e-7,
            "large": 1e20,
        }
        expected = (
            b'{"large":100000000000000000000,'
            b'"numbers":[333333333.3333333,1e+30,4.5,0.002,1e-27,0],'
            b'"small":1e-7}'
        )

        self.assertEqual(canonical_json_bytes(value), expected)
        self.assertEqual(
            canonical_json_sha256(value),
            "75b41ce1b2c489310fef1bb33e445fdacebd31d44717e5c7986d628654a78208",
        )

    def test_mapping_order_numpy_scalars_and_signed_zero_are_semantic(self):
        first = MappingProxyType(
            {"z": np.float64(-0.0), "a": (np.int64(1), np.float64(1e-7))}
        )
        second = {"a": [1, 1e-7], "z": 0.0}

        self.assertEqual(canonical_json_bytes(first), b'{"a":[1,1e-7],"z":0}')
        self.assertEqual(canonical_json_sha256(first), canonical_json_sha256(second))

    def test_i_json_domain_and_structure_limits_fail_closed(self):
        with self.assertRaises(CanonicalJSONError):
            canonical_json_bytes({"unsafe_integer": 9007199254740992})
        with self.assertRaises(CanonicalJSONError):
            canonical_json_bytes({"nan": float("nan")})
        with self.assertRaises(CanonicalJSONError):
            canonical_json_bytes({1: "non-string key"})


if __name__ == "__main__":
    unittest.main()
