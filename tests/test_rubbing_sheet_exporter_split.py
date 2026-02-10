import unittest

import numpy as np

from src.core.mesh_loader import MeshData
from src.core.rubbing_sheet_exporter import RubbingSheetExporter


class TestRubbingSheetExporterSplit(unittest.TestCase):
    def test_split_outer_inner_does_not_duplicate_full_mesh_on_planar_input(self):
        # Planar mesh: all face normals point to +Z, so sign split cannot produce both sides.
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
        mesh = MeshData(vertices=vertices, faces=faces, unit="cm")

        exporter = RubbingSheetExporter()
        outer, inner = exporter._split_outer_inner(mesh, threshold=0.2)

        # Regression guard: avoid exporting the same full mesh for both sides.
        self.assertFalse(outer.n_faces == mesh.n_faces and inner.n_faces == mesh.n_faces)
        self.assertGreaterEqual(outer.n_faces, 0)
        self.assertGreaterEqual(inner.n_faces, 0)


if __name__ == "__main__":
    unittest.main()
