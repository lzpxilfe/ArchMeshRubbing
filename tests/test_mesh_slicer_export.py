import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from src.core.mesh_slicer import MeshSlicer


class TestMeshSlicerExport(unittest.TestCase):
    def _make_box(self):
        mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
        mesh.metadata["unit"] = "cm"
        return mesh

    def test_export_contours_svg_writes_file(self):
        slicer = MeshSlicer(self._make_box())
        contours = [
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            )
        ]

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "contours.svg"
            saved = slicer.export_contours_svg(
                contours,
                str(out),
                unit="cm",
                title="Test Contours",
                desc="Unit test export",
            )

            self.assertEqual(saved, str(out))
            self.assertTrue(out.exists())
            text = out.read_text(encoding="utf-8")
            self.assertIn("<svg", text)
            self.assertIn("<polyline", text)
            self.assertIn("<title>Test Contours</title>", text)

    def test_export_contours_svg_empty_returns_none(self):
        slicer = MeshSlicer(self._make_box())
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "empty.svg"
            saved = slicer.export_contours_svg([], str(out))
            self.assertIsNone(saved)
            self.assertFalse(out.exists())

    def test_export_slice_svg_still_works(self):
        slicer = MeshSlicer(self._make_box())
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "slice.svg"
            saved = slicer.export_slice_svg(0.0, str(out), unit="cm")
            self.assertEqual(saved, str(out))
            self.assertTrue(out.exists())
            text = out.read_text(encoding="utf-8")
            self.assertIn("Cross Section at Z=", text)
            self.assertIn("<polyline", text)

