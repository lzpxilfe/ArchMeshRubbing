import unittest
from unittest.mock import patch

import numpy as np

from src.core.flattener import flatten_with_method
from src.core.mesh_loader import MeshData


class TestFlattenerPolicyFallback(unittest.TestCase):
    def test_section_failure_uses_policy_fallback_chain(self):
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        mesh = MeshData(vertices=vertices, faces=faces, unit="cm")

        fake_uv = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=np.float64,
        )

        with patch(
            "src.core.flattener.sectionwise_cylindrical_parameterization",
            return_value=(fake_uv, {"sectionwise": True, "sectionwise_fallback": True, "sectionwise_reason": "forced"}),
        ):
            result = flatten_with_method(mesh, method="section")

        meta = dict(getattr(result, "meta", {}) or {})
        self.assertEqual(str(meta.get("flatten_method")), "area")
        self.assertEqual(str(meta.get("fallback_from")), "section")
        self.assertEqual(str(meta.get("fallback_used_method")), "area")
        self.assertEqual(str(meta.get("fallback_reason")), "forced")


if __name__ == "__main__":
    unittest.main()
