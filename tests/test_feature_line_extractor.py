import unittest

import numpy as np

from src.core.feature_line_extractor import extract_sharp_edges
from src.core.mesh_loader import MeshData


class TestFeatureLineExtractor(unittest.TestCase):
    def test_shared_edge_face_pair_mapping(self):
        # Two coplanar triangles sharing edge (0, 2).
        vertices = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        mesh = MeshData(vertices=vertices, faces=faces, unit="mm")

        res = extract_sharp_edges(mesh, angle_deg=0.0, include_boundary=False)

        self.assertEqual(int(res.edges.shape[0]), 1)
        self.assertEqual(tuple(map(int, res.edges[0].tolist())), (0, 2))
        pair = tuple(sorted(map(int, res.face_pairs[0].tolist())))
        self.assertEqual(pair, (0, 1))
        self.assertAlmostEqual(float(res.dihedral_deg[0]), 0.0, places=6)

    def test_boundary_edges_count(self):
        # Same mesh as above: 4 boundary edges + 1 internal shared edge.
        vertices = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        mesh = MeshData(vertices=vertices, faces=faces, unit="mm")

        res = extract_sharp_edges(mesh, angle_deg=170.0, include_boundary=True)
        edges = {tuple(map(int, row.tolist())) for row in res.edges}

        self.assertEqual(len(edges), 4)
        self.assertNotIn((0, 2), edges)
        self.assertTrue(all(int(v) == -1 for v in res.face_pairs[:, 1].tolist()))
        self.assertTrue(np.all(res.dihedral_deg >= 170.0))


if __name__ == "__main__":
    unittest.main()

