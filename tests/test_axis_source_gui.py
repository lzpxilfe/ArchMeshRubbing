"""Which line the two circles make an axis of is the archaeologist's call.

`commit_axis_alignment` has taken an `axis_source` since the flat-dish work,
but the window always sent the default, so a dish or a warped bowl could
only be stood up by a script.  These hold the chooser to what it promises:
three sources, the ordinary one first, and the one that is chosen is the one
that is committed.
"""

from __future__ import annotations

import hashlib
import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QCoreApplication, QEvent  # noqa: E402
from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from app_interactive import MainWindow  # noqa: E402
from src.core.artifact_axis_alignment import (  # noqa: E402
    AXIS_SOURCE_CENTER_LINE,
    AXIS_SOURCE_CIRCLE_NORMALS,
    AXIS_SOURCE_STANDING_ON_FOOT,
)
from src.core.artifact_session import ArtifactSession  # noqa: E402
from src.core.mesh_import_recipe import current_mesh_import_recipe  # noqa: E402
from src.core.mesh_loader import MeshData, SourceFingerprint  # noqa: E402


#: One application for the module.  A QApplication that goes out of scope is
#: garbage collected, and it takes every widget with it.
_APP: QApplication | None = None


def _window() -> MainWindow:
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    assert _APP is not None
    return MainWindow()


def _drop(window: MainWindow) -> None:
    app = QApplication.instance()
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    if app is not None:
        app.processEvents()


def _session() -> ArtifactSession:
    payload = b"axis-source-gui"
    mesh = MeshData(
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64),
        faces=np.array([[0, 1, 2]], dtype=np.int32),
        unit="mm",
        source_identity=SourceFingerprint(
            sha256=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
            mtime_ns=1,
            original_name="axis.ply",
            format="ply",
        ),
        source_format="ply",
        source_import_recipe=current_mesh_import_recipe("ply"),
    )
    return ArtifactSession.create_from_source(
        mesh,
        resolved_source_path="/source/axis.ply",
        unit="mm",
        axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
        handedness="right",
        software_version="test",
        operator="pytest",
        created_at="2026-09-09T00:00:00Z",
        document_id="artifact:axis-source",
        metadata_revision_id="metadata:axis-source",
        align_revision_id="align:identity",
    )


def test_the_three_sources_are_offered_with_the_ordinary_one_first() -> None:
    window = _window()
    try:
        combo = window.section_panel.combo_axis_source
        offered = [combo.itemData(index) for index in range(combo.count())]
        assert offered == [
            AXIS_SOURCE_CENTER_LINE,
            AXIS_SOURCE_CIRCLE_NORMALS,
            AXIS_SOURCE_STANDING_ON_FOOT,
        ]
        assert combo.currentData() == AXIS_SOURCE_CENTER_LINE
        # Each one names the artifact it is for; "circle_plane_normals/v1"
        # tells an archaeologist nothing about which pot it is for.
        assert "납작한" in combo.itemText(1)
        assert "뒤틀린" in combo.itemText(2)
    finally:
        _drop(window)


def test_the_chosen_source_is_the_one_committed() -> None:
    """A chooser whose value the handler drops is worse than no chooser: the
    archaeologist would believe the dish was stood up the way they asked."""

    window = _window()
    try:
        panel = window.section_panel
        combo = panel.combo_axis_source
        combo.setCurrentIndex(combo.findData(AXIS_SOURCE_CIRCLE_NORMALS))
        panel.combo_axis_top_record.addItem("위", "record:diameter:top")
        panel.combo_axis_top_record.setCurrentIndex(panel.combo_axis_top_record.count() - 1)
        panel.combo_axis_bottom_record.addItem("아래", "record:diameter:bottom")
        panel.combo_axis_bottom_record.setCurrentIndex(
            panel.combo_axis_bottom_record.count() - 1
        )

        session = _session()
        window._artifact_session = session
        # What the handler does after the commit - reading the new revision's
        # QC for the status line - is not what this test is about.
        aligned = SimpleNamespace(
            document=SimpleNamespace(
                active_align_revision_id="align:axis",
                align_revision_index={
                    "align:axis": SimpleNamespace(
                        qc={
                            "axis_tilt_corrected_deg": 1.25,
                            "center_separation_mm": 84.0,
                            "circle_normal_disagreement_deg": 0.5,
                        }
                    )
                },
            )
        )
        committed = Mock(return_value=aligned)
        with (
            patch.object(MainWindow, "_artifact_workbench_controller", return_value=Mock()),
            patch.object(MainWindow, "_refresh_native_record_selectors"),
            patch.object(
                QMessageBox,
                "question",
                return_value=QMessageBox.StandardButton.Yes,
            ),
            patch.object(ArtifactSession, "commit_axis_alignment", committed),
        ):
            window.on_axis_align_requested()

        assert committed.call_count == 1
        assert committed.call_args.kwargs["axis_source"] == AXIS_SOURCE_CIRCLE_NORMALS
        assert committed.call_args.kwargs["top_record_id"] == "record:diameter:top"
        assert committed.call_args.kwargs["bottom_record_id"] == "record:diameter:bottom"
        assert "회전축 정치 완료" in window.status_info.text()
    finally:
        _drop(window)
