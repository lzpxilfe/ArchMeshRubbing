from __future__ import annotations

import unittest

import numpy as np

from src.core.artifact_document import GEOMETRY_HASH_SCOPE_V1
from src.core.geometry_identity import (
    GeometryIdentityError,
    canonical_geometry_sha256,
    mesh_geometry_sha256,
)
from src.core.mesh_loader import MeshData


class TestGeometryIdentity(unittest.TestCase):
    def setUp(self) -> None:
        self.vertices = np.array(
            [[-0.0, 0.0, 1.0], [2.0, 0.0, 1.0], [0.0, 3.0, 1.0]],
            dtype=np.float64,
        )
        self.faces = np.array([[0, 1, 2]], dtype=np.int32)

    def test_v1_digest_has_a_fixed_golden(self) -> None:
        self.assertEqual(
            canonical_geometry_sha256(self.vertices, self.faces),
            "950e0f1f4df15821932e2406680086f1d1a7945d0b92346fc3754844598c9b9c",
        )

    def test_endianness_layout_and_signed_zero_are_canonical(self) -> None:
        expected = canonical_geometry_sha256(self.vertices, self.faces)
        big_endian_vertices = np.asfortranarray(self.vertices.astype(">f8"))
        big_endian_faces = np.asfortranarray(self.faces.astype(">i4"))
        positive_zero = self.vertices.copy()
        positive_zero[0, 0] = 0.0

        self.assertEqual(
            canonical_geometry_sha256(big_endian_vertices, big_endian_faces),
            expected,
        )
        self.assertEqual(
            canonical_geometry_sha256(positive_zero, self.faces),
            expected,
        )

    def test_vertex_order_and_triangle_winding_remain_identity_bearing(self) -> None:
        expected = canonical_geometry_sha256(self.vertices, self.faces)
        self.assertNotEqual(
            canonical_geometry_sha256(self.vertices[[1, 0, 2]], self.faces),
            expected,
        )
        self.assertNotEqual(
            canonical_geometry_sha256(self.vertices, self.faces[:, ::-1]),
            expected,
        )

    def test_mesh_helper_and_invalid_inputs(self) -> None:
        mesh = MeshData(vertices=self.vertices, faces=self.faces)
        self.assertEqual(
            mesh_geometry_sha256(mesh),
            canonical_geometry_sha256(self.vertices, self.faces),
        )

        invalid_cases = (
            (np.zeros((3, 2)), self.faces),
            (np.array([[np.nan, 0.0, 0.0]]), np.zeros((0, 3), dtype=np.int32)),
            (self.vertices, np.array([[0.0, 1.0, 2.0]])),
            (self.vertices, np.array([[0, 1, 3]], dtype=np.int32)),
            (self.vertices, np.array([[-1, 1, 2]], dtype=np.int32)),
        )
        for vertices, faces in invalid_cases:
            with self.subTest(vertices=vertices, faces=faces):
                with self.assertRaises(GeometryIdentityError):
                    canonical_geometry_sha256(vertices, faces)

        with self.assertRaisesRegex(GeometryIdentityError, "unsupported"):
            canonical_geometry_sha256(
                self.vertices,
                self.faces,
                scope=f"{GEOMETRY_HASH_SCOPE_V1}-future",
            )


if __name__ == "__main__":
    unittest.main()
