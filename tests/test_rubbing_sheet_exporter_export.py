import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from src.core.mesh_loader import MeshData
from src.core.rubbing_sheet_exporter import RubbingSheetExporter, SheetExportOptions


class _FakeRubbing:
    def __init__(self, width_real: float, height_real: float):
        self.width_real = float(width_real)
        self.height_real = float(height_real)

    def to_pil_image(self):
        return Image.new("L", (8, 8), color=180)


class TestRubbingSheetExporterExport(unittest.TestCase):
    def _make_planar_mesh(self) -> MeshData:
        vertices = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        faces = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        return MeshData(vertices=vertices, faces=faces, unit="cm")

    def test_export_keeps_single_available_side(self):
        mesh = self._make_planar_mesh()
        exporter = RubbingSheetExporter()

        def fake_top_view_group(*_args, **_kwargs):
            return '<g id="top_view"></g>', 10.0, 5.0

        def fake_flatten_and_rub(side_mesh, **_kwargs):
            n_faces = int(getattr(side_mesh, "n_faces", 0) or 0)
            self.assertGreater(n_faces, 0)
            return object(), _FakeRubbing(width_real=12.0, height_real=6.0)

        exporter._build_top_view_group = fake_top_view_group  # type: ignore[method-assign]
        exporter._flatten_and_rub = fake_flatten_and_rub  # type: ignore[method-assign]

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "sheet.svg"
            saved = exporter.export(mesh, out, options=SheetExportOptions(include_labels=True))
            text = Path(saved).read_text(encoding="utf-8")

        self.assertIn('id="outer_rubbing"', text)
        self.assertNotIn('id="inner_rubbing"', text)
        self.assertIn("외면 탁본", text)
        self.assertNotIn("내면 탁본", text)
        self.assertIn('id="top_measurement"', text)

    def test_export_single_surface_mode_uses_single_group(self):
        mesh = self._make_planar_mesh()
        exporter = RubbingSheetExporter()
        calls = {"count": 0}

        def fake_top_view_group(*_args, **_kwargs):
            return '<g id="top_view"></g>', 8.0, 4.0

        def fake_flatten_and_rub(side_mesh, **_kwargs):
            calls["count"] += 1
            n_faces = int(getattr(side_mesh, "n_faces", 0) or 0)
            self.assertGreater(n_faces, 0)
            return object(), _FakeRubbing(width_real=10.0, height_real=5.0)

        exporter._build_top_view_group = fake_top_view_group  # type: ignore[method-assign]
        exporter._flatten_and_rub = fake_flatten_and_rub  # type: ignore[method-assign]

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "sheet_single.svg"
            saved = exporter.export(
                mesh,
                out,
                options=SheetExportOptions(include_labels=True, single_surface_label="선택 표면 탁본"),
            )
            text = Path(saved).read_text(encoding="utf-8")

        self.assertEqual(calls["count"], 1)
        self.assertIn('id="surface_rubbing"', text)
        self.assertNotIn('id="outer_rubbing"', text)
        self.assertNotIn('id="inner_rubbing"', text)
        self.assertIn("선택 표면 탁본", text)
        self.assertIn('id="top_measurement"', text)

    def test_flatten_and_rub_keeps_explicit_axis_and_initial_method(self):
        mesh = self._make_planar_mesh()
        exporter = RubbingSheetExporter()
        captured = {}

        def fake_flatten_with_method(mesh_arg, **kwargs):
            captured["mesh"] = mesh_arg
            captured.update(kwargs)

            class _Flat:
                width = 12.0
                height = 6.0
                original_mesh = mesh_arg

            return _Flat()

        class _FakeVisualizer:
            def __init__(self, default_dpi=300):
                self.default_dpi = int(default_dpi)

            def generate_rubbing(self, flattened, **kwargs):
                captured["rubbing_width_pixels"] = int(kwargs.get("width_pixels", 0))
                return _FakeRubbing(width_real=flattened.width, height_real=flattened.height)

        with patch("src.core.rubbing_sheet_exporter.flatten_with_method", side_effect=fake_flatten_with_method):
            with patch("src.core.rubbing_sheet_exporter.SurfaceVisualizer", _FakeVisualizer):
                exporter._flatten_and_rub(
                    mesh,
                    svg_unit="mm",
                    unit_scale=1.0,
                    options=SheetExportOptions(
                        dpi=300,
                        flatten_method="section",
                        flatten_initial_method="section",
                        cylinder_axis=(1.0, 0.0, 0.0),
                        cylinder_radius=4.5,
                    ),
                )

        self.assertEqual(captured.get("method"), "section")
        self.assertEqual(captured.get("initial_method"), "section")
        self.assertEqual(captured.get("cylinder_axis"), (1.0, 0.0, 0.0))
        self.assertEqual(captured.get("cylinder_radius"), 4.5)
        self.assertGreater(int(captured.get("rubbing_width_pixels", 0)), 0)


if __name__ == "__main__":
    unittest.main()
