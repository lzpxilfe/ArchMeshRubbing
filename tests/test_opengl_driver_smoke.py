from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from PyQt6.QtGui import QSurfaceFormat

from src.gui.opengl_context import (
    OPENGL_MINIMUM_DEPTH_BITS,
    OPENGL_MINIMUM_VERSION,
    compatibility_surface_format,
)
from src.gui.opengl_driver_smoke import (
    PROBE_BASE_WORLD_MM,
    PROBE_GAP_WIDTH_MM,
    PROBE_STEP_HEIGHT_MM,
    connected_component_sizes,
    probe_geometry,
    write_report,
)

ROOT = Path(__file__).resolve().parents[1]


def test_compatibility_surface_format_matches_fixed_function_viewport() -> None:
    surface_format = compatibility_surface_format()

    assert (
        surface_format.majorVersion(),
        surface_format.minorVersion(),
    ) == OPENGL_MINIMUM_VERSION
    assert (
        surface_format.renderableType()
        == QSurfaceFormat.RenderableType.OpenGL
    )
    assert (
        surface_format.profile()
        == QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile
    )
    assert surface_format.depthBufferSize() == OPENGL_MINIMUM_DEPTH_BITS
    assert surface_format.stencilBufferSize() == 8
    assert surface_format.samples() == 0


def test_interactive_app_installs_surface_contract_before_qapplication() -> None:
    source = (ROOT / "app_interactive.py").read_text(encoding="utf-8-sig")
    install = source.index("install_compatibility_surface_format()")
    application = source.index("app = QApplication(sys.argv)", install)

    assert install < application


def test_viewport_preserves_depth_attachments_for_post_paint_picking() -> None:
    source = (ROOT / "src" / "gui" / "viewport_3d.py").read_text(
        encoding="utf-8"
    )

    assert (
        "self.setUpdateBehavior(QOpenGLWidget.UpdateBehavior.PartialUpdate)"
        in source
    )


def test_probe_geometry_exposes_absolute_float32_precision_regression() -> None:
    vertices, faces = probe_geometry()

    assert vertices.shape == (8, 3)
    assert faces.shape == (4, 3)
    assert vertices.dtype == np.float64
    assert faces.dtype == np.int32
    assert float(np.max(np.abs(vertices))) >= 1_000_000_000.0
    assert np.array_equal(
        np.unique(vertices[:, 2] - PROBE_BASE_WORLD_MM[2]),
        np.array([0.0, PROBE_STEP_HEIGHT_MM], dtype=np.float64),
    )
    assert float(vertices[4, 0] - vertices[1, 0]) == PROBE_GAP_WIDTH_MM

    # An absolute float32 upload collapses all sub-mm vertices at this offset.
    # The actual driver smoke must therefore pass through the relative VBO path.
    assert np.unique(vertices.astype(np.float32), axis=0).shape == (1, 3)

    vertices[0, 0] = 0.0
    faces[0, 0] = 7
    fresh_vertices, fresh_faces = probe_geometry()
    assert fresh_vertices[0, 0] != 0.0
    assert fresh_faces[0, 0] == 0


def test_connected_component_sizes_uses_four_neighbours() -> None:
    mask = np.array(
        [
            [True, True, False, False],
            [False, True, False, True],
            [False, False, True, True],
        ],
        dtype=bool,
    )

    assert connected_component_sizes(mask) == [3, 3]

    with pytest.raises(ValueError, match="2D"):
        connected_component_sizes(np.zeros((2, 2, 2), dtype=bool))


def test_write_report_is_atomic_json_and_rejects_overwrite(tmp_path) -> None:
    path = tmp_path / "driver-smoke.json"
    report = {
        "ok": True,
        "metrics": {
            "origin": PROBE_BASE_WORLD_MM,
            "step_mm": np.float64(PROBE_STEP_HEIGHT_MM),
        },
    }

    write_report(path, report)

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "metrics": {
            "origin": PROBE_BASE_WORLD_MM.tolist(),
            "step_mm": PROBE_STEP_HEIGHT_MM,
        },
        "ok": True,
    }
    assert not list(tmp_path.glob(".driver-smoke.json.*.tmp"))
    with pytest.raises(FileExistsError, match="already exists"):
        write_report(path, report)


def test_write_report_concurrent_publication_has_one_winner(tmp_path) -> None:
    path = tmp_path / "driver-smoke.json"

    def publish(marker: str) -> str:
        try:
            write_report(path, {"marker": marker})
        except FileExistsError:
            return "lost"
        return marker

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(publish, ("first", "second")))

    winner = json.loads(path.read_text(encoding="utf-8"))["marker"]
    assert winner in {"first", "second"}
    assert outcomes.count("lost") == 1
    assert winner in outcomes
    assert not list(tmp_path.glob(".driver-smoke.json.*.tmp"))


def test_driver_smoke_rejects_qt_offscreen_without_touching_gl() -> None:
    environment = dict(os.environ)
    environment["QT_QPA_PLATFORM"] = "offscreen"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.gui.opengl_driver_smoke",
            "--qt-platform",
            "offscreen",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=15,
    )

    assert completed.returncode == 1
    report = json.loads(completed.stdout)
    assert report["ok"] is False
    assert report["error"]["type"] == "DriverSmokeFailure"
    assert len(report["source"]["commit"]) in {40, 64}
    assert report["source"]["source_tree"] in {"clean", "dirty"}
    assert len(report["source"]["runtime_lock_sha256"]) == 64
    assert report["tested_at_utc"].endswith("Z")
    assert report["checks"][-1] == {
        "actual": "offscreen",
        "id": "qt.native_platform",
        "ok": False,
        "requested": "offscreen",
    }
