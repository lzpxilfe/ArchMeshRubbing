from __future__ import annotations

from contextlib import ExitStack
import hashlib
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import QCoreApplication, QEvent, QStandardPaths, Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QMessageBox, QPushButton

from app_interactive import (
    MainWindow,
    TaskThread,
    UnitSelectionDialog,
    _native_cutline_frame,
    _mesh_source_payload,
    _validate_project_source_declarations,
    _verify_loaded_project_source,
)
from src.application.artifact_exports import ArtifactExportState
from src.application.artifact_measurements import (
    MeasurementOperationState,
)
from src.application.artifact_workflow_progress import (
    ArtifactWorkflowProgress,
    ArtifactWorkflowStep,
    ArtifactWorkflowStepProgress,
    REQUIRED_CUTLINE_VIEWS,
    REQUIRED_SIX_VIEWS,
)
from src.core.artifact_session import ArtifactSession, ArtifactSessionError
from src.application.artifact_workbench import (
    ConfirmedSourceMetadata,
    ProjectionTransition,
    RecordBindingTransition,
    WorkflowBusyError,
    WorkflowStage,
    WorkflowTransitionKind,
)
from src.core.artifact_rubbing_export import (
    ArtifactRubbingExportError,
    validate_rubbing_export_package,
)
from src.core.artifact_rubbing_extractor import (
    DEFAULT_RUBBING_BLACK_POINT_UM,
    DEFAULT_RUBBING_DEPTH_QUANTIZATION_UM,
    DEFAULT_RUBBING_INK_STRENGTH_PERCENT,
    DEFAULT_RUBBING_MARGIN_UM,
    DEFAULT_RUBBING_PIXELS_PER_MM,
    DEFAULT_RUBBING_POLARITY,
    DEFAULT_RUBBING_REFERENCE_RADIUS_UM,
    commit_artifact_rubbing,
    compute_artifact_rubbing,
)
from src.core.artifact_rubbing_record import rubbing_receipt_from_record
from src.core.artifact_vector_export import validate_vector_export_package
from src.core.artifact_outline_extractor import compute_artifact_outline
from src.core.artifact_vector_extractor import (
    commit_vector_computation,
    compute_artifact_cutline,
    extract_cutline_geometry,
)
from src.core.artifact_vector_record import PlanarFrame
from src.core.project_file import (
    ARTIFACT_PAYLOAD_TYPE,
    MIGRATION_MARKER_NAME,
    ProjectFormatError,
    ProjectSerializationError,
    UnsupportedPayloadError,
)
from src.core.mesh_loader import MeshData
from src.core.source_identity import SourceFingerprint, SourceVerificationStatus
from src.core.tile_form_model import TileInterpretationState
from src.gui.viewport_3d import SceneObject, Viewport3D


def _fingerprint(payload: bytes, *, name: str = "artifact.ply") -> SourceFingerprint:
    return SourceFingerprint(
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        mtime_ns=1,
        original_name=name,
        format="ply",
    )


def _artifact_session() -> ArtifactSession:
    payload = b"gui-native-artifact"
    mesh = MeshData(
        vertices=np.array(
            [[1.0, 2.0, 3.0], [4.0, 2.0, 3.0], [1.0, 6.0, 3.0]],
            dtype=np.float64,
        ),
        faces=np.array([[0, 1, 2]], dtype=np.int32),
        unit="cm",
        source_identity=_fingerprint(payload),
        source_format="ply",
    )
    session = ArtifactSession.create_from_source(
        mesh,
        resolved_source_path="/source/gui-native-artifact.ply",
        unit="cm",
        axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
        handedness="right",
        software_version="test",
        operator="pytest",
        created_at="2026-07-11T00:00:00Z",
        document_id="artifact:gui-smoke",
        metadata_revision_id="metadata:gui-smoke",
        align_revision_id="align:identity",
    )
    return session


def _artifact_box_session() -> ArtifactSession:
    vertices = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ],
        dtype=np.int32,
    )
    payload = b"gui-native-box"
    mesh = MeshData(
        vertices=vertices,
        faces=faces,
        unit="cm",
        source_identity=_fingerprint(payload, name="box.ply"),
        source_format="ply",
    )
    session = ArtifactSession.create_from_source(
        mesh,
        resolved_source_path="/source/box.ply",
        unit="cm",
        axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
        handedness="right",
        software_version="test",
        operator="pytest",
        created_at="2026-07-11T00:00:00Z",
        document_id="artifact:gui-box",
        metadata_revision_id="metadata:gui-box",
        align_revision_id="align:gui-box-baseline",
    )
    return session.commit_preview(
        translation_mm=(0.0, 0.0, 0.0),
        rotation_deg=(0.0, 0.0, 0.0),
        scale=1.0,
        pivot_mm=(0.0, 0.0, 0.0),
        operator="pytest",
        created_at="2026-07-11T00:00:00Z",
        revision_id="align:gui-box",
    )
def _projected_scene_object(session: ArtifactSession) -> SceneObject:
    projection = session.materialize()
    obj = SceneObject(projection.mesh, "native artifact")
    obj._amr_projection_preserves_origin = True
    obj._amr_artifact_projection_snapshot = projection.snapshot
    obj._amr_preview_pivot_mm = projection.mesh.centroid.copy()
    return obj


def _capture_measurement_publication(
    window: MainWindow,
    captured: dict[str, object],
):
    """Simulate the controller half of the patched GUI scene publisher."""

    def capture(candidate: ArtifactSession, **kwargs) -> None:
        transition = kwargs.get("workflow_transition")
        assert isinstance(transition, RecordBindingTransition)
        assert transition.candidate_session is candidate
        controller = window._artifact_workbench_controller()
        activation = controller.activate_record_binding(transition)
        window._artifact_session = candidate
        obj = window.viewport.selected_obj
        assert obj is not None
        obj.compare_and_swap_artifact_binding(
            transition.expected_snapshot,
            transition.candidate_snapshot,
        )
        window.current_mesh = obj.mesh
        controller.finalize_record_binding(activation)
        captured["session"] = candidate
        captured["transition"] = transition
        captured["kwargs"] = {
            key: value
            for key, value in kwargs.items()
            if key != "workflow_transition"
        }

    return capture


def _reloaded_source_mesh(
    session: ArtifactSession,
    *,
    fingerprint: SourceFingerprint | None = None,
) -> MeshData:
    source = session.source_mesh
    return MeshData(
        vertices=np.asarray(source.vertices, dtype=np.float64).copy(),
        faces=np.asarray(source.faces, dtype=np.int32).copy(),
        unit=str(source.unit),
        source_identity=fingerprint or source.source_identity,
        source_format=str(source.source_format),
    )


def test_main_window_constructs_offscreen() -> None:
    """Exercise imports and widget wiring without claiming an OpenGL render."""
    QStandardPaths.setTestModeEnabled(True)
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    window = MainWindow()
    try:
        assert window.viewport is not None
        assert window.centralWidget() is not None
    finally:
        # Avoid MainWindow.closeEvent(), which intentionally asks the user to
        # confirm application exit and would block an offscreen test runner.
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_native_workflow_buttons_show_record_counts_and_green_completion() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    window = MainWindow()
    complete = ArtifactWorkflowProgress(
        align_ready=True,
        cutline=ArtifactWorkflowStepProgress(
            step=ArtifactWorkflowStep.CUTLINE,
            required_views=REQUIRED_CUTLINE_VIEWS,
            completed_views=REQUIRED_CUTLINE_VIEWS,
            enabled=True,
        ),
        outline=ArtifactWorkflowStepProgress(
            step=ArtifactWorkflowStep.OUTLINE,
            required_views=REQUIRED_SIX_VIEWS,
            completed_views=REQUIRED_SIX_VIEWS,
            enabled=True,
        ),
        rubbing=ArtifactWorkflowStepProgress(
            step=ArtifactWorkflowStep.DIGITAL_RUBBING,
            required_views=REQUIRED_SIX_VIEWS,
            completed_views=REQUIRED_SIX_VIEWS,
            enabled=True,
        ),
    )
    try:
        panel = window.section_panel
        panel.native_group.setEnabled(True)
        panel.apply_native_workflow_progress(complete)
        for button, suffix in (
            (panel.btn_native_cutline, "(3/3)"),
            (panel.btn_native_outline, "(6/6)"),
            (panel.btn_native_rubbing, "(6/6)"),
        ):
            assert button.isEnabled()
            assert button.text().endswith(suffix)
            assert button.property("workflowComplete") is True

        panel.apply_native_workflow_progress(ArtifactWorkflowProgress.empty())
        for button in (
            panel.btn_native_cutline,
            panel.btn_native_outline,
            panel.btn_native_rubbing,
        ):
            assert not button.isEnabled()
            assert button.property("workflowComplete") is False
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_late_task_finished_signal_cannot_clear_new_thread_or_dialog() -> None:
    class FakeSignal:
        def __init__(self) -> None:
            self.callbacks = []

        def connect(self, callback) -> None:
            self.callbacks.append(callback)

        def emit(self, *args) -> None:
            for callback in tuple(self.callbacks):
                callback(*args)

    class FakeThread:
        def __init__(self) -> None:
            self.done = FakeSignal()
            self.failed = FakeSignal()
            self.finished = FakeSignal()
            self.running = False
            self.deleted = False

        def isRunning(self) -> bool:
            return self.running

        def start(self) -> None:
            self.running = True

        def deleteLater(self) -> None:
            self.deleted = True

    class FakeDialog:
        instances = []

        def __init__(self, *_args) -> None:
            self.closed = False
            self.instances.append(self)

        def setWindowTitle(self, _value) -> None:
            pass

        def setWindowModality(self, _value) -> None:
            pass

        def setCancelButton(self, _value) -> None:
            pass

        def setMinimumDuration(self, _value) -> None:
            pass

        def show(self) -> None:
            pass

        def close(self) -> None:
            self.closed = True

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    window = MainWindow()
    first = FakeThread()
    second = FakeThread()
    try:
        with (
            patch("app_interactive.QProgressDialog", FakeDialog),
            patch.object(window, "_status_task_begin"),
            patch.object(window, "_status_task_end"),
        ):
            assert window._start_task(
                title="first",
                label="first",
                thread=first,  # type: ignore[arg-type]
                on_done=Mock(),
            )
            first_dialog = window._task_dialog
            first.running = False
            first.done.emit("done")

            assert window._start_task(
                title="second",
                label="second",
                thread=second,  # type: ignore[arg-type]
                on_done=Mock(),
            )
            second_dialog = window._task_dialog
            assert second_dialog is not first_dialog

            first.finished.emit()
            assert first.deleted
            assert window._task_thread is second
            assert window._task_dialog is second_dialog
            assert second_dialog is not None and not second_dialog.closed

            second.running = False
            second.done.emit("done")
            second.finished.emit()
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_start_task_exposes_one_shot_cooperative_cancel_without_terminating_thread() -> None:
    class FakeSignal:
        def __init__(self) -> None:
            self.callbacks = []

        def connect(self, callback) -> None:
            self.callbacks.append(callback)

        def emit(self, *args) -> None:
            for callback in tuple(self.callbacks):
                callback(*args)

    class FakeThread:
        def __init__(self) -> None:
            self.done = FakeSignal()
            self.failed = FakeSignal()
            self.finished = FakeSignal()
            self.started = False
            self.deleted = False

        def isRunning(self) -> bool:
            return self.started

        def start(self) -> None:
            self.started = True

        def deleteLater(self) -> None:
            self.deleted = True

    class FakeDialog:
        instance = None

        def __init__(self, label, cancel_text, *_args) -> None:
            self.label = label
            self.cancel_text = cancel_text
            self.canceled = FakeSignal()
            self.cancel_button = cancel_text
            self.closed = False
            self.show_count = 0
            self.__class__.instance = self

        def setWindowTitle(self, _value) -> None:
            pass

        def setWindowModality(self, _value) -> None:
            pass

        def setCancelButton(self, value) -> None:
            self.cancel_button = value

        def setMinimumDuration(self, _value) -> None:
            pass

        def setLabelText(self, value) -> None:
            self.label = value

        def show(self) -> None:
            self.show_count += 1

        def close(self) -> None:
            self.closed = True

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    window = MainWindow()
    thread = FakeThread()
    cancelled = Mock()
    failed = Mock()
    try:
        with patch("app_interactive.QProgressDialog", FakeDialog):
            assert window._start_task(
                title="cooperative",
                label="working",
                thread=thread,  # type: ignore[arg-type]
                on_done=Mock(),
                on_failed=failed,
                on_cancel_requested=cancelled,
            )
        dialog = FakeDialog.instance
        assert dialog is not None
        assert dialog.cancel_text == "취소"
        dialog.canceled.emit()
        dialog.canceled.emit()
        cancelled.assert_called_once_with()
        assert dialog.cancel_button is None
        assert "안전한 계산 경계" in dialog.label
        assert thread.started
        assert not thread.deleted

        thread.failed.emit("MeasurementCancelledError: user_cancelled")
        failed.assert_called_once_with("MeasurementCancelledError: user_cancelled")
        assert dialog.closed
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_start_task_success_close_does_not_emit_a_spurious_cancel_request() -> None:
    class FakeSignal:
        def __init__(self) -> None:
            self.callbacks = []

        def connect(self, callback) -> None:
            self.callbacks.append(callback)

        def emit(self, *args) -> None:
            for callback in tuple(self.callbacks):
                callback(*args)

    class FakeThread:
        def __init__(self) -> None:
            self.done = FakeSignal()
            self.failed = FakeSignal()
            self.finished = FakeSignal()
            self.running = False

        def isRunning(self) -> bool:
            return self.running

        def start(self) -> None:
            self.running = True

        def deleteLater(self) -> None:
            pass

    class FakeDialog:
        instance = None

        def __init__(self, *_args) -> None:
            self.canceled = FakeSignal()
            self.signals_blocked = False
            self.closed = False
            self.__class__.instance = self

        def setWindowTitle(self, _value) -> None:
            pass

        def setWindowModality(self, _value) -> None:
            pass

        def setCancelButton(self, _value) -> None:
            pass

        def setMinimumDuration(self, _value) -> None:
            pass

        def show(self) -> None:
            pass

        def blockSignals(self, blocked: bool) -> bool:
            previous = self.signals_blocked
            self.signals_blocked = blocked
            return previous

        def close(self) -> None:
            self.closed = True
            if not self.signals_blocked:
                self.canceled.emit()

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    window = MainWindow()
    thread = FakeThread()
    cancelled = Mock()
    done = Mock()
    try:
        with patch("app_interactive.QProgressDialog", FakeDialog):
            assert window._start_task(
                title="cooperative",
                label="working",
                thread=thread,  # type: ignore[arg-type]
                on_done=done,
                on_cancel_requested=cancelled,
            )
        thread.running = False
        thread.done.emit("result")

        dialog = FakeDialog.instance
        assert dialog is not None and dialog.closed
        done.assert_called_once_with("result")
        cancelled.assert_not_called()
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_cancelled_task_dialog_resists_escape_and_close_until_worker_finishes() -> None:
    class FakeSignal:
        def __init__(self) -> None:
            self.callbacks = []

        def connect(self, callback) -> None:
            self.callbacks.append(callback)

        def emit(self, *args) -> None:
            for callback in tuple(self.callbacks):
                callback(*args)

    class FakeThread:
        def __init__(self) -> None:
            self.done = FakeSignal()
            self.failed = FakeSignal()
            self.finished = FakeSignal()
            self.running = False
            self.deleted = False

        def isRunning(self) -> bool:
            return self.running

        def start(self) -> None:
            self.running = True

        def deleteLater(self) -> None:
            self.deleted = True

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    window = MainWindow()
    thread = FakeThread()
    cancelled = Mock()
    failed = Mock()
    try:
        assert window._start_task(
            title="cooperative",
            label="working",
            thread=thread,  # type: ignore[arg-type]
            on_done=Mock(),
            on_failed=failed,
            on_cancel_requested=cancelled,
        )
        dialog = window._task_dialog
        assert dialog is not None and dialog.isVisible()

        cancel_button = dialog.findChild(QPushButton)
        assert cancel_button is not None
        QTest.mouseClick(cancel_button, Qt.MouseButton.LeftButton)
        app.processEvents()
        cancelled.assert_called_once_with()
        assert dialog.isVisible()

        QTest.keyClick(dialog, Qt.Key.Key_Escape)
        app.processEvents()
        cancelled.assert_called_once_with()
        assert dialog.isVisible()

        assert not dialog.close()
        app.processEvents()
        cancelled.assert_called_once_with()
        assert dialog.isVisible()

        thread.running = False
        thread.failed.emit("MeasurementCancelledError: user_cancelled")
        app.processEvents()
        failed.assert_called_once_with("MeasurementCancelledError: user_cancelled")
        assert not dialog.isVisible()

        thread.finished.emit()
        assert thread.deleted
        assert window._task_thread is None
        assert window._task_dialog is None
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_source_metadata_dialog_requires_explicit_bijective_axis_confirmation() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    dialog = UnitSelectionDialog()
    try:
        dialog.combo.setCurrentIndex(0)
        dialog.axis_combos["source_x"].setCurrentText("+X")
        dialog.axis_combos["source_y"].setCurrentText("+Y")
        dialog.axis_combos["source_z"].setCurrentText("+Z")
        assert not dialog.ok_btn.isEnabled()
        assert dialog.get_source_metadata()["confirmation_status"] == "unconfirmed"

        dialog.confirm_metadata.setChecked(True)
        assert dialog.ok_btn.isEnabled()
        metadata = dialog.get_source_metadata()
        assert metadata == {
            "unit": "mm",
            "axes": {"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
            "handedness": "right",
            "confirmation_status": "confirmed",
        }

        dialog.axis_combos["source_z"].setCurrentText("+Y")
        assert not dialog.ok_btn.isEnabled()
        invalid = dialog.get_source_metadata()
        assert invalid["handedness"] == "unknown"
        assert invalid["confirmation_status"] == "unconfirmed"
    finally:
        dialog.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_project_source_helpers_preserve_import_identity_and_detect_relocation() -> None:
    expected = _fingerprint(b"same bytes", name="old.ply")
    actual = _fingerprint(b"same bytes", name="new.ply")
    mesh = SimpleNamespace(source_identity=actual, source_format="obj")

    payload = _mesh_source_payload(mesh, "/path/that/does/not/need/to/exist/new.ply")
    assert payload["identity"] == actual.to_dict()
    assert payload["binding_status"] == "captured_at_import"
    assert payload["parse_format"] == "obj"

    state = {
        "objects": [
            {
                "mesh": {
                    "path": "/old/location/old.ply",
                    "source": {
                        "identity": expected.to_dict(),
                        "binding_status": "captured_at_import",
                        "parse_format": "obj",
                    },
                }
            }
        ]
    }
    _validate_project_source_declarations(state, migrated_from_v1=False)
    verification, binding = _verify_loaded_project_source(
        mesh,
        state["objects"][0],
        "/new/location/new.ply",
        migrated_from_v1=False,
    )
    assert verification.status is SourceVerificationStatus.VERIFIED
    assert verification.relocated
    assert binding == "captured_at_import"

    with pytest.raises(ProjectFormatError):
        _validate_project_source_declarations(
            {"objects": [{"mesh": {"path": "missing-identity.ply"}}]},
            migrated_from_v1=False,
        )
    with pytest.raises(ProjectSerializationError):
        _mesh_source_payload(SimpleNamespace(source_identity=None), "external.ply")


def test_unpersisted_bake_blocks_legacy_project_snapshot() -> None:
    window_like = SimpleNamespace(
        viewport=SimpleNamespace(
            objects=[SimpleNamespace(_amr_has_unpersisted_bake=True)]
        )
    )

    with pytest.raises(ProjectSerializationError, match="재현 가능하게 저장"):
        MainWindow._collect_project_state(window_like)


def test_native_save_requires_exact_document_projection_geometry() -> None:
    session = _artifact_session()
    obj = _projected_scene_object(session)
    window_like = SimpleNamespace(
        viewport=SimpleNamespace(
            objects=[obj],
            cut_lines=[[], []],
            line_section_contours=[],
        )
    )

    MainWindow._validate_native_scene_for_save(window_like, session)

    obj.mesh.vertices[0, 0] += 0.001
    with pytest.raises(ProjectSerializationError, match="canonical projection"):
        MainWindow._validate_native_scene_for_save(window_like, session)

    obj = _projected_scene_object(session)
    obj._amr_has_unpersisted_bake = True
    window_like.viewport.objects = [obj]
    with pytest.raises(ProjectSerializationError, match="vertex bake"):
        MainWindow._validate_native_scene_for_save(window_like, session)

    obj = _projected_scene_object(session)
    obj.tile_interpretation_state = TileInterpretationState(note="unsaved hypothesis")
    window_like.viewport.objects = [obj]
    with pytest.raises(ProjectSerializationError, match="record로 승격"):
        MainWindow._validate_native_scene_for_save(window_like, session)

    obj.tile_interpretation_state = TileInterpretationState()
    window_like.viewport.line_profile = [(0.0, 1.0)]
    with pytest.raises(ProjectSerializationError, match="record로 승격"):
        MainWindow._validate_native_scene_for_save(window_like, session)


def test_viewport_refuses_destructive_bake_of_native_projection() -> None:
    session = _artifact_session()
    obj = _projected_scene_object(session)
    before = obj.mesh.vertices.copy()

    with pytest.raises(RuntimeError, match="cannot be destructively baked"):
        Viewport3D.bake_object_transform(SimpleNamespace(), obj)

    np.testing.assert_array_equal(obj.mesh.vertices, before)


def test_native_cutline_frames_are_explicit_right_handed_mm_planes() -> None:
    cases = {
        "top": ((0.0, 0.0, 12.5), (0.0, 0.0, 1.0)),
        "front": ((0.0, 12.5, 0.0), (0.0, -1.0, 0.0)),
        "right": ((12.5, 0.0, 0.0), (1.0, 0.0, 0.0)),
    }
    for view, (origin, normal) in cases.items():
        frame = _native_cutline_frame(view, 12.5)
        assert frame.origin_world_mm == origin
        assert frame.normal_world == normal
        np.testing.assert_allclose(
            np.cross(frame.u_axis_world, frame.v_axis_world),
            frame.normal_world,
            rtol=0.0,
            atol=1e-12,
        )

    with pytest.raises(ValueError, match="unsupported"):
        _native_cutline_frame("perspective", 0.0)


def test_viewport_native_vector_preview_is_derived_world_projection_only() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    session = _artifact_box_session()
    projection = session.materialize()
    frame = PlanarFrame(
        origin_world_mm=(0.0, 0.0, 0.0),
        u_axis_world=(0.0, 1.0, 0.0),
        v_axis_world=(0.0, 0.0, 1.0),
        normal_world=(1.0, 0.0, 0.0),
    )
    payload = extract_cutline_geometry(
        projection.mesh.vertices,
        projection.mesh.faces,
        frame,
    ).payload
    viewport = Viewport3D()
    try:
        with patch.object(viewport, "update"):
            viewport.set_native_vector_preview(payload, record_id="record:right")
        assert viewport.native_vector_preview_record_id == "record:right"
        assert len(viewport.native_vector_preview_world) == 1
        world, closed = viewport.native_vector_preview_world[0]
        assert closed
        np.testing.assert_allclose(world[:, 0], 0.0, rtol=0.0, atol=1e-12)
        assert payload.frame.origin_world_mm == (0.0, 0.0, 0.0)
        with patch.object(viewport, "update"):
            viewport.set_native_vector_preview(None)
        assert viewport.native_vector_preview_world == []
    finally:
        viewport.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_viewport_ctrl_z_requests_main_window_authority_undo() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    viewport = Viewport3D()
    requested: list[bool] = []
    viewport.undoRequested.connect(lambda: requested.append(True))
    event = QKeyEvent(
        QEvent.Type.KeyPress,
        Qt.Key.Key_Z,
        Qt.KeyboardModifier.ControlModifier,
    )
    try:
        with patch.object(viewport, "undo") as legacy_undo:
            viewport.keyPressEvent(event)
        assert requested == [True]
        legacy_undo.assert_not_called()
    finally:
        viewport.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_projection_reset_fences_late_cut_section_and_roi_worker_callbacks() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    viewport = Viewport3D()
    old_obj = object()
    old_cut_worker = Mock()
    old_roi_worker = Mock()
    viewport.objects = [old_obj]
    viewport.selected_index = 0
    old_generation = viewport._projection_generation
    viewport._cut_section_thread = old_cut_worker
    viewport._roi_edges_thread = old_roi_worker
    viewport._cut_section_timer.start(1_000)
    viewport._roi_edges_timer.start(1_000)

    try:
        viewport._reset_projection_transients()

        assert viewport._projection_generation == old_generation + 1
        assert viewport._cut_section_thread is None
        assert viewport._roi_edges_thread is None
        assert not viewport._cut_section_timer.isActive()
        assert not viewport._roi_edges_timer.isActive()

        new_obj = object()
        new_cut_worker = Mock()
        new_roi_worker = Mock()
        viewport.objects = [new_obj]
        viewport.selected_index = 0
        viewport._cut_section_thread = new_cut_worker
        viewport._roi_edges_thread = new_roi_worker
        current_generation = viewport._projection_generation

        with (
            patch.object(viewport, "_on_cut_section_computed") as cut_computed,
            patch.object(viewport, "_on_cut_section_failed") as cut_failed,
            patch.object(viewport, "_on_roi_edges_computed") as roi_computed,
            patch.object(viewport, "_on_roi_edges_failed") as roi_failed,
        ):
            viewport._dispatch_cut_section_computed(
                old_cut_worker,
                old_generation,
                old_obj,
                {"index": 0},
            )
            viewport._dispatch_cut_section_failed(
                old_cut_worker,
                old_generation,
                old_obj,
                "stale",
            )
            viewport._dispatch_roi_edges_computed(
                old_roi_worker,
                old_generation,
                old_obj,
                {"x1": []},
            )
            viewport._dispatch_roi_edges_failed(
                old_roi_worker,
                old_generation,
                old_obj,
                "stale",
            )
            viewport._dispatch_cut_section_finished(
                old_cut_worker,
                old_generation,
                old_obj,
            )
            viewport._dispatch_roi_edges_finished(
                old_roi_worker,
                old_generation,
                old_obj,
            )

            cut_computed.assert_not_called()
            cut_failed.assert_not_called()
            roi_computed.assert_not_called()
            roi_failed.assert_not_called()
            assert viewport._cut_section_thread is new_cut_worker
            assert viewport._roi_edges_thread is new_roi_worker
            old_cut_worker.deleteLater.assert_called_once()
            old_roi_worker.deleteLater.assert_called_once()

            viewport._dispatch_cut_section_computed(
                new_cut_worker,
                current_generation,
                new_obj,
                {"index": 0},
            )
            viewport._dispatch_cut_section_failed(
                new_cut_worker,
                current_generation,
                new_obj,
                "current",
            )
            viewport._dispatch_roi_edges_computed(
                new_roi_worker,
                current_generation,
                new_obj,
                {"x1": []},
            )
            viewport._dispatch_roi_edges_failed(
                new_roi_worker,
                current_generation,
                new_obj,
                "current",
            )

            cut_computed.assert_called_once_with({"index": 0})
            cut_failed.assert_called_once_with("current")
            roi_computed.assert_called_once_with({"x1": []})
            roi_failed.assert_called_once_with("current")

        viewport._dispatch_cut_section_finished(
            new_cut_worker,
            current_generation,
            new_obj,
        )
        viewport._dispatch_roi_edges_finished(
            new_roi_worker,
            current_generation,
            new_obj,
        )
        assert viewport._cut_section_thread is None
        assert viewport._roi_edges_thread is None
        new_cut_worker.deleteLater.assert_called_once()
        new_roi_worker.deleteLater.assert_called_once()
    finally:
        viewport.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_artifact_scene_projection_can_skip_destructive_centroid_centering() -> None:
    vertices = np.array(
        [[10.0, 20.0, 30.0], [12.0, 20.0, 30.0], [10.0, 24.0, 30.0]],
        dtype=np.float64,
    )
    mesh = MeshData(vertices=vertices.copy(), faces=np.array([[0, 1, 2]], dtype=np.int32))
    viewport_like = SimpleNamespace(
        objects=[],
        selected_index=-1,
        update_vbo=Mock(return_value=True),
        update_grid_scale=Mock(),
        camera=SimpleNamespace(fit_to_bounds=Mock(), pan_offset=None),
        meshLoaded=SimpleNamespace(emit=Mock()),
        selectionChanged=SimpleNamespace(emit=Mock()),
        update=Mock(),
    )

    Viewport3D.add_mesh_object(
        viewport_like,
        mesh,
        name="canonical-mm",
        center_at_origin=False,
    )

    np.testing.assert_array_equal(mesh.vertices, vertices)
    assert len(viewport_like.objects) == 1
    assert viewport_like.objects[0]._amr_projection_preserves_origin is True
    np.testing.assert_array_equal(
        viewport_like.objects[0].get_world_bounds(),
        np.array([[10.0, 20.0, 30.0], [12.0, 24.0, 30.0]]),
    )


def test_prepared_scene_vbo_failure_is_detached_and_success_swaps_once() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    mesh = MeshData(
        vertices=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        faces=np.array([[0, 1, 2]], dtype=np.int32),
    )
    viewport = Viewport3D()
    old = SceneObject(mesh, "old")
    old.cleanup = Mock()
    viewport.objects = [old]
    viewport.selected_index = 0
    emitted_meshes: list[object] = []
    emitted_selections: list[int] = []
    viewport.meshLoaded.connect(emitted_meshes.append)
    viewport.selectionChanged.connect(emitted_selections.append)
    try:
        with patch.object(viewport, "update_vbo", return_value=False):
            with pytest.raises(RuntimeError, match="VBO upload failed"):
                viewport.prepare_mesh_object(mesh, "failed")
        assert viewport.objects == [old]
        assert emitted_meshes == []
        assert emitted_selections == []

        with (
            patch.object(viewport, "update_vbo", return_value=True),
            patch.object(SceneObject, "cleanup") as invalid_cleanup,
        ):
            with pytest.raises(RuntimeError, match="invalid prepared object"):
                viewport.prepare_mesh_object(mesh, "invalid")
        invalid_cleanup.assert_called_once()
        assert viewport.objects == [old]
        assert emitted_meshes == []
        assert emitted_selections == []

        def upload(prepared: SceneObject) -> bool:
            prepared.vbo_id = 77
            prepared.vertex_count = 3
            return True

        with patch.object(viewport, "update_vbo", side_effect=upload):
            prepared = viewport.prepare_mesh_object(
                mesh,
                "native",
                artifact_binding="snapshot",
            )
        viewport.curvature_pick_mode = True
        viewport.slice_enabled = True
        viewport.roi_enabled = True
        viewport.cut_lines_enabled = True
        viewport.line_section_enabled = True
        viewport._surface_magnetic_dist = np.ones((2, 2), dtype=np.float32)
        viewport._cached_modelview = np.eye(4, dtype=np.float64)
        previous = viewport.swap_prepared_scene([prepared], fit_camera=False)

        assert previous == [old]
        assert viewport.objects == [prepared]
        assert viewport.selected_index == 0
        assert prepared._amr_artifact_projection_snapshot == "snapshot"
        assert not viewport.curvature_pick_mode
        assert not viewport.slice_enabled
        assert not viewport.roi_enabled
        assert not viewport.cut_lines_enabled
        assert not viewport.line_section_enabled
        assert viewport._surface_magnetic_dist is None
        assert viewport._cached_modelview is None
        assert emitted_meshes == [mesh]
        assert emitted_selections == [0]
        viewport.cleanup_scene_objects(previous)
        old.cleanup.assert_called_once()
    finally:
        prepared_obj = viewport.objects[0] if viewport.objects else None
        if prepared_obj is not None:
            prepared_obj.vbo_id = None
        viewport.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_native_project_save_never_serializes_legacy_ui_state() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    session = _artifact_session()
    obj = _projected_scene_object(session)
    window = MainWindow()
    window._artifact_session = session
    window.viewport.objects = [obj]
    window.viewport.selected_index = 0
    try:
        with (
            patch("app_interactive.save_amr_artifact_project") as save_native,
            patch.object(
                window,
                "_collect_project_state",
                side_effect=AssertionError("legacy serializer must not run"),
            ) as collect_legacy,
            patch("app_interactive._safe_git_info", return_value=("abc123", False)),
        ):
            assert window._write_project("/tmp/native-artifact.amr")

        collect_legacy.assert_not_called()
        save_native.assert_called_once()
        assert save_native.call_args.args[:2] == (
            "/tmp/native-artifact.amr",
            session.document,
        )
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_native_align_commit_creates_revision_without_destructive_bake() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    session = _artifact_session()
    obj = _projected_scene_object(session)
    obj.translation = np.array([5.0, -2.0, 1.0], dtype=np.float64)
    obj.rotation = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    window = MainWindow()
    window._artifact_session = session
    window._current_project_path = "/tmp/native.amr"
    window.viewport.objects = [obj]
    window.viewport.selected_index = 0
    captured: dict[str, object] = {}
    try:
        def capture(candidate, **kwargs):
            captured["session"] = candidate
            captured["kwargs"] = kwargs

        with (
            patch.object(window, "_publish_artifact_session_projection", side_effect=capture),
            patch.object(window.viewport, "bake_object_transform") as destructive_bake,
        ):
            window.on_bake_all_clicked()

        destructive_bake.assert_not_called()
        candidate = captured["session"]
        assert isinstance(candidate, ArtifactSession)
        assert candidate.document.active_align_revision_id != session.document.active_align_revision_id
        active = candidate.document.align_revision_index[
            candidate.document.active_align_revision_id
        ]
        assert active.parent_id == session.document.active_align_revision_id
        assert active.recipe["convention"] == "delta @ parent"
        assert isinstance(captured["kwargs"], dict)
        kwargs = dict(captured["kwargs"])
        transition = kwargs.pop("workflow_transition")
        assert isinstance(transition, ProjectionTransition)
        assert transition.candidate_session is candidate
        assert kwargs == {
            "project_path": "/tmp/native.amr",
            "fit_camera": False,
            "status_text": "✅ 정치 확정 | 새 Align revision 생성",
        }
        np.testing.assert_array_equal(
            session.materialize().mesh.vertices,
            _artifact_session().materialize().mesh.vertices,
        )
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_initial_identity_requires_explicit_align_before_measurement() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    initial = _artifact_session()
    obj = _projected_scene_object(initial)
    window = MainWindow()
    window._artifact_session = initial
    window.viewport.objects = [obj]
    window.viewport.selected_index = 0
    captured: dict[str, object] = {}
    try:
        window._sync_native_cutline_controls(reset_offset=True)
        assert window._native_workflow_stage() is WorkflowStage.ALIGN_REQUIRED
        assert not window.section_panel.btn_native_cutline.isEnabled()
        assert not window.section_panel.btn_native_outline.isEnabled()
        assert not window.section_panel.btn_native_rubbing.isEnabled()
        assert window.section_panel.btn_native_cutline.text().endswith("(0/3)")
        assert window.section_panel.btn_native_outline.text().endswith("(0/6)")
        assert window.section_panel.btn_native_rubbing.text().endswith("(0/6)")
        with pytest.raises(ArtifactSessionError, match="explicit Align confirmation"):
            window._compute_and_commit_native_cutline(view="top", offset_mm=0.0)

        def capture(candidate, **kwargs):
            captured["session"] = candidate
            captured["transition"] = kwargs["workflow_transition"]

        with patch.object(
            window,
            "_publish_artifact_session_projection",
            side_effect=capture,
        ):
            window.on_bake_all_clicked()

        confirmed = captured["session"]
        transition = captured["transition"]
        assert isinstance(confirmed, ArtifactSession)
        assert isinstance(transition, ProjectionTransition)
        assert transition.candidate_session is confirmed
        assert confirmed.document.active_align_revision_id != "align:identity"
        active = confirmed.document.align_revision_index[
            confirmed.document.active_align_revision_id
        ]
        assert active.parent_id == "align:identity"
        assert active.recipe["translation_mm"] == (0.0, 0.0, 0.0)
        assert active.recipe["rotation_deg"] == (0.0, 0.0, 0.0)

        window._artifact_session = confirmed
        window.viewport.objects = [_projected_scene_object(confirmed)]
        window.viewport.selected_index = 0
        window._sync_native_cutline_controls(reset_offset=True)
        assert window._native_workflow_stage() is WorkflowStage.MEASUREMENT_READY
        assert window.section_panel.btn_native_cutline.isEnabled()
        assert not window.section_panel.btn_native_outline.isEnabled()
        assert not window.section_panel.btn_native_rubbing.isEnabled()
        assert window.section_panel.btn_native_cutline.text().endswith("(0/3)")
        assert window.section_panel.btn_native_outline.text().endswith("(0/6)")
        assert window.section_panel.btn_native_rubbing.text().endswith("(0/6)")
        assert not window.section_panel.btn_native_cutline.property(
            "workflowComplete"
        )
        assert not window.section_panel.btn_native_outline.property(
            "workflowComplete"
        )
        assert not window.section_panel.btn_native_rubbing.property(
            "workflowComplete"
        )
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_cutline_progress_reopens_and_tracks_align_stale_restore() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    session = _artifact_box_session()
    for index, view in enumerate(REQUIRED_CUTLINE_VIEWS, start=1):
        computation = compute_artifact_cutline(
            session,
            _native_cutline_frame(view, 0.0),
        )
        session = commit_vector_computation(
            session,
            computation,
            record_id=f"record:workflow-cutline:{view}",
            created_at=f"2026-07-11T00:00:0{index}Z",
            operator="pytest",
        )
    completed_align_id = session.document.active_align_revision_id
    assert completed_align_id is not None

    window = MainWindow()
    window._artifact_session = session
    window.viewport.objects = [_projected_scene_object(session)]
    window.viewport.selected_index = 0
    try:
        window._sync_native_cutline_controls(reset_offset=True)
        panel = window.section_panel
        assert panel.btn_native_cutline.text().endswith("(3/3)")
        assert panel.btn_native_cutline.property("workflowComplete") is True
        assert panel.btn_native_outline.isEnabled()
        assert not panel.btn_native_rubbing.isEnabled()

        changed = session.commit_preview(
            translation_mm=(0.25, 0.0, 0.0),
            rotation_deg=(0.0, 0.0, 0.0),
            scale=1.0,
            operator="pytest",
            created_at="2026-07-11T00:00:04Z",
            revision_id="align:workflow-progress-changed",
        )
        window._artifact_session = changed
        window.viewport.objects = [_projected_scene_object(changed)]
        window.viewport.selected_index = 0
        window._sync_native_cutline_controls(reset_offset=True)
        assert panel.btn_native_cutline.text().endswith("(0/3)")
        assert panel.btn_native_cutline.property("workflowComplete") is False
        assert not panel.btn_native_outline.isEnabled()

        restored = changed.activate_align(completed_align_id)
        window._artifact_session = restored
        window.viewport.objects = [_projected_scene_object(restored)]
        window.viewport.selected_index = 0
        window._sync_native_cutline_controls(reset_offset=True)
        assert panel.btn_native_cutline.text().endswith("(3/3)")
        assert panel.btn_native_cutline.property("workflowComplete") is True
        assert panel.btn_native_outline.isEnabled()
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_native_cutline_command_commits_record_previews_and_exports_offline() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    session = _artifact_box_session()
    obj = _projected_scene_object(session)
    window = MainWindow()
    window._artifact_session = session
    window._current_project_path = "/tmp/gui-box.amr"
    window.viewport.objects = [obj]
    window.viewport.selected_index = 0
    captured: dict[str, object] = {}
    try:
        with (
            patch.object(
                window,
                "_publish_artifact_session_projection",
                side_effect=_capture_measurement_publication(window, captured),
            ),
            patch.object(window.viewport, "set_native_vector_preview") as preview,
            patch.object(window, "_sync_native_cutline_controls"),
        ):
            record_id = window._compute_and_commit_native_cutline(
                view="top",
                offset_mm=0.0,
                record_id="record:gui-cutline",
                created_at="2026-07-11T00:00:01Z",
                operator="pytest",
            )

        assert record_id == "record:gui-cutline"
        committed = captured["session"]
        assert isinstance(committed, ArtifactSession)
        record = committed.document.record_index[record_id]
        assert record.type == "vector.cutline.v1"
        assert record.align_revision_id == "align:gui-box"
        assert record.qc["bounds_mm"] == (-10.0, -10.0, 10.0, 10.0)
        assert captured["kwargs"] == {
            "project_path": "/tmp/gui-box.amr",
            "fit_camera": False,
            "expected_new_record_ids": ("record:gui-cutline",),
            "status_text": "✅ Top Cutline 기록 | 1개 경로 | canonical mm",
        }
        preview.assert_called_once()
        assert preview.call_args.kwargs["record_id"] == record_id
        np.testing.assert_array_equal(
            session.source_mesh.vertices,
            _artifact_box_session().source_mesh.vertices,
        )

        window._artifact_session = committed
        with tempfile.TemporaryDirectory() as temporary:
            package = window._export_native_vector_record(
                Path(temporary) / "gui-cutline.amr-vector",
                record_id=record_id,
            )
            verified = validate_vector_export_package(package)
        assert verified.vector_payload_sha256 in record.geometry_ref
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_record_append_rebinds_live_document_without_rebuilding_the_vbo() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    session = _artifact_box_session()
    obj = _projected_scene_object(session)
    obj.vbo_id = 777
    obj.vertex_count = int(obj.mesh.n_vertices)
    window = MainWindow()
    window._artifact_session = session
    window._current_project_path = "/tmp/gui-record-rebind.amr"
    window.current_mesh = obj.mesh
    window.current_filepath = session.resolved_source_path
    window.viewport.objects = [obj]
    window.viewport.selected_index = 0
    window.viewport.undo_stack = [("preserve", object())]
    window._flattened_cache["preserve"] = object()
    objects_identity = window.viewport.objects
    mesh_identity = obj.mesh
    undo_identity = window.viewport.undo_stack
    cache_identity = window._flattened_cache
    generation = window.viewport._projection_generation
    try:
        with (
            patch.object(window.viewport, "prepare_mesh_object") as prepare,
            patch.object(window.viewport, "swap_prepared_scene") as swap,
            patch.object(window.viewport, "cleanup_scene_objects") as cleanup,
        ):
            record_id = window._compute_and_commit_native_cutline(
                view="top",
                offset_mm=0.0,
                record_id="record:gui-rebind",
                created_at="2026-07-11T00:00:01Z",
                operator="pytest",
            )

        prepare.assert_not_called()
        swap.assert_not_called()
        cleanup.assert_not_called()
        committed = window._artifact_session
        assert isinstance(committed, ArtifactSession)
        assert record_id in committed.document.record_index
        assert window._artifact_workbench_controller().session is committed
        assert window.viewport.objects is objects_identity
        assert window.viewport.selected_obj is obj
        assert obj.mesh is mesh_identity
        assert window.current_mesh is mesh_identity
        assert obj.vbo_id == 777
        assert obj.vertex_count == mesh_identity.n_vertices
        assert window.viewport._projection_generation == generation
        assert window.viewport.undo_stack is undo_identity
        assert window._flattened_cache is cache_identity
        assert "preserve" in window._flattened_cache
        binding = obj._amr_artifact_projection_snapshot
        assert binding == committed.projection_snapshot()
        assert binding != session.projection_snapshot()
        assert binding.has_same_render_projection(session.projection_snapshot())
        window._validate_native_scene_for_save(committed)
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_retryable_native_publication_is_queued_and_reuses_exact_result() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    session = _artifact_box_session()
    window = MainWindow()
    window._artifact_session = session
    window.viewport.objects = [_projected_scene_object(session)]
    window.viewport.selected_index = 0
    window._sync_native_cutline_controls(reset_offset=True)
    controller = window._artifact_measurement_controller()
    item = controller.begin_cutline(
        _native_cutline_frame("top", 0.0),
        record_id="record:gui-retry",
        created_at="2026-07-11T00:00:01Z",
        operator="pytest",
    )
    result = controller.execute(item)
    captured: dict[str, object] = {}
    capture = _capture_measurement_publication(window, captured)
    attempt_count = 0

    def fail_once(candidate: ArtifactSession, **kwargs) -> None:
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count == 1:
            raise RuntimeError("temporary scene preparation failure")
        capture(candidate, **kwargs)

    try:
        with patch.object(
            window,
            "_publish_artifact_session_projection",
            side_effect=fail_once,
        ):
            with pytest.raises(RuntimeError, match="temporary scene"):
                window._publish_native_measurement_result(item, result)

            assert controller.summary(item).state is MeasurementOperationState.RUNNING
            assert window._pending_native_measurement_publications[item.id] == (
                item,
                result,
            )
            assert window.section_panel.btn_native_measurement_retry.isEnabled()
            with pytest.raises(ProjectSerializationError, match="실측 작업"):
                window._validate_native_scene_for_save(session)

            window.on_native_measurement_retry_requested()

        assert controller.summary(item).state is MeasurementOperationState.COMPLETED
        assert window._pending_native_measurement_publications == {}
        assert not window.section_panel.btn_native_measurement_retry.isEnabled()
        committed = captured["session"]
        assert isinstance(committed, ArtifactSession)
        assert committed.document.record_index[item.record_id].id == item.record_id
        assert attempt_count == 2
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_record_binding_cas_failure_rolls_back_and_retries_without_vbo_swap() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    session = _artifact_box_session()
    obj = _projected_scene_object(session)
    window = MainWindow()
    window._artifact_session = session
    window.current_mesh = obj.mesh
    window.current_filepath = session.resolved_source_path
    window.viewport.objects = [obj]
    window.viewport.selected_index = 0
    window._sync_native_cutline_controls(reset_offset=True)
    controller = window._artifact_measurement_controller()
    item = controller.begin_cutline(
        _native_cutline_frame("top", 0.0),
        record_id="record:gui-binding-retry",
        created_at="2026-07-11T00:00:01Z",
        operator="pytest",
    )
    result = controller.execute(item)
    original_cas = obj.compare_and_swap_artifact_binding
    calls = 0

    def fail_once(expected, candidate) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected binding CAS failure")
        original_cas(expected, candidate)

    try:
        with (
            patch.object(
                obj,
                "compare_and_swap_artifact_binding",
                side_effect=fail_once,
            ),
            patch.object(window.viewport, "prepare_mesh_object") as prepare,
            patch.object(window.viewport, "swap_prepared_scene") as swap,
        ):
            with pytest.raises(RuntimeError, match="binding CAS"):
                window._publish_native_measurement_result(item, result)

            assert controller.summary(item).state is MeasurementOperationState.RUNNING
            assert window._artifact_session is session
            assert window._artifact_workbench_controller().session is session
            assert obj._amr_artifact_projection_snapshot == session.projection_snapshot()
            assert window.section_panel.btn_native_measurement_retry.isEnabled()

            window.on_native_measurement_retry_requested()

        prepare.assert_not_called()
        swap.assert_not_called()
        assert calls == 2
        assert controller.summary(item).state is MeasurementOperationState.COMPLETED
        assert window._artifact_session is not session
        assert item.record_id in window._artifact_session.document.record_index
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_pending_open_retry_button_reenables_after_open_is_cancelled() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    session = _artifact_box_session()
    window = MainWindow()
    window._artifact_session = session
    window.viewport.objects = [_projected_scene_object(session)]
    window.viewport.selected_index = 0
    window._sync_native_cutline_controls(reset_offset=True)
    controller = window._artifact_measurement_controller()
    item = controller.begin_cutline(
        _native_cutline_frame("top", 0.0),
        record_id="record:gui-pending-open",
        created_at="2026-07-11T00:00:01Z",
        operator="pytest",
    )
    result = controller.execute(item)
    workbench = window._artifact_workbench_controller()
    ticket = workbench.begin_new_import(
        "/source/replacement.ply",
        ConfirmedSourceMetadata(
            unit="cm",
            source_x="+X",
            source_y="+Y",
            source_z="+Z",
            handedness="right",
        ),
        software_version="test",
        operator="pytest",
    )
    window._artifact_load_ticket = ticket
    window._artifact_load_active = True
    captured: dict[str, object] = {}
    try:
        with pytest.raises(WorkflowBusyError, match="Open request is pending"):
            window._publish_native_measurement_result(item, result)
        assert controller.summary(item).state is MeasurementOperationState.RUNNING
        assert not window.section_panel.btn_native_measurement_retry.isEnabled()

        window._clear_artifact_pending_load(cancel_workbench=True)

        assert workbench.snapshot.pending_load is None
        assert window.section_panel.btn_native_measurement_retry.isEnabled()
        with patch.object(
            window,
            "_publish_artifact_session_projection",
            side_effect=_capture_measurement_publication(window, captured),
        ):
            window.on_native_measurement_retry_requested()
        assert controller.summary(item).state is MeasurementOperationState.COMPLETED
        assert window._pending_native_measurement_publications == {}
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_reopened_project_requires_explicit_durable_record_selection() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    base = _artifact_box_session()
    vector_record_id = "record:reopened-cutline"
    vector_computation = compute_artifact_cutline(
        base,
        _native_cutline_frame("top", 0.0),
    )
    with_vector = commit_vector_computation(
        base,
        vector_computation,
        record_id=vector_record_id,
        created_at="2026-07-11T00:00:01Z",
        operator="pytest",
    )
    outline_record_id = "record:reopened-outline"
    outline_computation = compute_artifact_outline(
        with_vector,
        "right",
        precision_grid_mm=0.01,
    )
    with_vectors = commit_vector_computation(
        with_vector,
        outline_computation,
        record_id=outline_record_id,
        created_at="2026-07-11T00:00:02Z",
        operator="pytest",
    )
    rubbing_record_id = "record:reopened-rubbing"
    rubbing_computation = compute_artifact_rubbing(
        with_vectors,
        "top",
        pixels_per_mm=2,
        margin_um=0,
        reference_radius_um=500,
        depth_quantization_um=10,
        black_point_um=100,
        ink_strength_percent=100,
        relief_polarity="bidirectional",
    )
    reopened = commit_artifact_rubbing(
        with_vectors,
        rubbing_computation,
        record_id=rubbing_record_id,
        created_at="2026-07-11T00:00:03Z",
        operator="pytest",
    )
    with (
        patch("app_interactive.DEFAULT_RUBBING_MEMORY_BUDGET_BYTES", 1),
        patch.object(
            ArtifactSession,
            "materialize",
            side_effect=AssertionError("budget rejection must precede materialize"),
        ) as materialize,
        pytest.raises(ArtifactRubbingExportError, match="1 GiB memory budget"),
    ):
        MainWindow._recompute_native_rubbing_record(
            reopened,
            reopened.document.record_index[rubbing_record_id],
        )
    materialize.assert_not_called()
    window = MainWindow()
    window._artifact_session = reopened
    window.viewport.objects = [_projected_scene_object(reopened)]
    window.viewport.selected_index = 0
    window.current_mesh = window.viewport.selected_obj.mesh
    try:
        window._sync_native_cutline_controls(reset_offset=True)

        vector_combo = window.section_panel.combo_native_vector_record
        rubbing_combo = window.section_panel.combo_native_rubbing_record
        assert vector_combo.count() == 3
        assert rubbing_combo.count() == 2
        assert vector_combo.currentData() is None
        assert rubbing_combo.currentData() is None
        assert {vector_combo.itemData(index) for index in range(1, 3)} == {
            vector_record_id,
            outline_record_id,
        }
        assert rubbing_combo.itemData(1) == rubbing_record_id
        assert window._current_native_vector_record() is None
        assert window._current_native_rubbing_record() is None
        assert window.section_panel.btn_native_cutline.isEnabled()
        assert not window.section_panel.btn_native_outline.isEnabled()
        assert not window.section_panel.btn_native_rubbing.isEnabled()
        assert window.section_panel.btn_native_cutline.text().endswith("(1/3)")
        assert window.section_panel.btn_native_outline.text().endswith("(1/6)")
        assert window.section_panel.btn_native_rubbing.text().endswith("(1/6)")
        assert not window.section_panel.btn_native_vector_export.isEnabled()
        assert not window.section_panel.btn_native_rubbing_export.isEnabled()

        controller = window._artifact_measurement_controller()
        budget_owner = controller.begin_cutline(
            _native_cutline_frame("top", 0.0),
            record_id="record:preview-budget-owner",
            created_at="2026-07-11T00:00:04Z",
            operator="pytest",
        )
        with patch.object(window, "_start_task", return_value=True) as blocked_task:
            rubbing_combo.setCurrentIndex(1)
        blocked_task.assert_not_called()
        assert rubbing_combo.currentData() is None
        with pytest.raises(ArtifactRubbingExportError, match="raster memory budget"):
            window._export_native_rubbing_record(
                "/tmp/blocked-rubbing.amr-rubbing",
                record_id=rubbing_record_id,
            )
        controller.cancel(budget_owner)

        vector_combo.setCurrentIndex(vector_combo.findData(outline_record_id))
        selected_vector = window._current_native_vector_record()
        assert selected_vector is not None
        assert selected_vector.id == outline_record_id
        assert window.viewport.native_vector_preview_record_id == outline_record_id
        assert (
            window._native_vector_preview_document_id
            == reopened.document.document_id
        )
        assert window.section_panel.btn_native_vector_export.isEnabled()

        with patch.object(window, "_start_task", return_value=True) as start_task:
            rubbing_combo.setCurrentIndex(1)
        preview_thread = start_task.call_args.kwargs["thread"]
        assert isinstance(preview_thread, TaskThread)
        assert preview_thread._task_name == "native_rubbing_record_preview"
        assert window._current_native_rubbing_record() is None
        assert not window.section_panel.btn_native_rubbing_export.isEnabled()

        raster = preview_thread._fn()
        start_task.call_args.kwargs["on_done"](raster)
        selected_rubbing = window._current_native_rubbing_record()
        assert selected_rubbing is not None
        assert selected_rubbing.id == rubbing_record_id
        assert window._native_rubbing_preview_record_id == rubbing_record_id
        assert (
            window._native_rubbing_preview_document_id
            == reopened.document.document_id
        )
        assert window.section_panel.label_native_rubbing_preview.pixmap() is not None
        assert window.section_panel.btn_native_rubbing_export.isEnabled()

        rubbing_combo.setCurrentIndex(0)
        with patch.object(window, "_start_task", return_value=True) as late_task:
            rubbing_combo.setCurrentIndex(1)
        late_preview_done = late_task.call_args.kwargs["on_done"]

        unrelated_computation = compute_artifact_cutline(
            reopened,
            _native_cutline_frame("front", 0.0),
        )
        appended = commit_vector_computation(
            reopened,
            unrelated_computation,
            record_id="record:preview-unrelated-append",
            created_at="2026-07-11T00:00:05Z",
            operator="pytest",
        )
        binding = window._artifact_workbench_controller().prepare_record_commit(
            reopened,
            appended,
            expected_new_record_ids=("record:preview-unrelated-append",),
        )
        window._publish_artifact_session_projection(
            appended,
            project_path=None,
            fit_camera=False,
            status_text="same-Align append",
            workflow_transition=binding,
        )
        assert rubbing_combo.currentData() == rubbing_record_id
        assert (
            window._native_rubbing_preview_pending_record_id
            == rubbing_record_id
        )
        assert window._current_native_rubbing_record() is None
        assert not window.section_panel.btn_native_rubbing_export.isEnabled()

        late_preview_done(raster)
        assert window._current_native_rubbing_record().id == rubbing_record_id
        assert window._native_rubbing_preview_pending_record_id is None
        assert window.section_panel.btn_native_rubbing_export.isEnabled()

        rubbing_combo.setCurrentIndex(0)
        with patch.object(window, "_start_task", return_value=True) as stale_task:
            rubbing_combo.setCurrentIndex(
                rubbing_combo.findData(rubbing_record_id)
            )
        stale_preview_done = stale_task.call_args.kwargs["on_done"]

        stale = appended.commit_preview(
            translation_mm=(1.0, 0.0, 0.0),
            rotation_deg=(0.0, 0.0, 0.0),
            scale=1.0,
            operator="pytest",
            created_at="2026-07-11T00:00:06Z",
            revision_id="align:reopened-stale",
        )
        window._artifact_session = stale
        window.viewport.objects = [_projected_scene_object(stale)]
        window.viewport.selected_index = 0
        window._sync_native_cutline_controls(reset_offset=False)
        assert vector_combo.count() == 1
        assert rubbing_combo.count() == 1
        assert vector_combo.currentData() is None
        assert rubbing_combo.currentData() is None
        assert window._current_native_vector_record() is None
        assert window._current_native_rubbing_record() is None
        assert not window.section_panel.btn_native_vector_export.isEnabled()
        assert not window.section_panel.btn_native_rubbing_export.isEnabled()

        stale_preview_done(raster)
        assert window._current_native_rubbing_record() is None
        assert rubbing_combo.currentData() is None
        assert not window.section_panel.btn_native_rubbing_export.isEnabled()
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_reopened_rubbing_preview_ignores_late_same_record_request() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    base = _artifact_box_session()
    record_id = "record:rubbing:request-token"
    computation = compute_artifact_rubbing(
        base,
        "top",
        pixels_per_mm=2,
        margin_um=0,
        reference_radius_um=500,
        depth_quantization_um=10,
        black_point_um=100,
        ink_strength_percent=100,
        relief_polarity="bidirectional",
    )
    session = commit_artifact_rubbing(
        base,
        computation,
        record_id=record_id,
        created_at="2026-07-11T00:00:07Z",
        operator="pytest",
    )
    obj = _projected_scene_object(session)
    window = MainWindow()
    window._artifact_session = session
    window.viewport.objects = [obj]
    window.viewport.selected_index = 0
    window.current_mesh = obj.mesh
    try:
        window._sync_native_cutline_controls(reset_offset=True)
        combo = window.section_panel.combo_native_rubbing_record
        record_index = combo.findData(record_id)
        assert record_index >= 1

        with patch.object(window, "_start_task", return_value=True) as first_task:
            combo.setCurrentIndex(record_index)
        first_done = first_task.call_args.kwargs["on_done"]
        first_token = window._native_rubbing_preview_pending_token
        assert first_token is not None

        combo.setCurrentIndex(0)
        with patch.object(window, "_start_task", return_value=True) as second_task:
            combo.setCurrentIndex(combo.findData(record_id))
        second_done = second_task.call_args.kwargs["on_done"]
        second_token = window._native_rubbing_preview_pending_token
        assert second_token is not None
        assert second_token is not first_token

        first_done(computation.raster)
        assert window._native_rubbing_preview_pending_token is second_token
        assert window._native_rubbing_preview_pending_record_id == record_id
        assert window._native_rubbing_preview_record_id is None
        assert combo.currentData() == record_id
        assert not window.section_panel.btn_native_rubbing_export.isEnabled()
        assert "재계산 중" in window.status_info.text()

        second_done(computation.raster)
        assert window._native_rubbing_preview_pending_token is None
        assert window._current_native_rubbing_record().id == record_id
        assert window.section_panel.btn_native_rubbing_export.isEnabled()
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_native_outline_ui_exposes_six_views_and_explicit_mm_grid() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    window = MainWindow()
    try:
        combo = window.section_panel.combo_native_outline_view
        assert combo.count() == 6
        assert [combo.itemData(index) for index in range(combo.count())] == [
            "top",
            "bottom",
            "front",
            "back",
            "right",
            "left",
        ]
        assert window.section_panel.spin_native_outline_grid.value() == pytest.approx(0.01)
        assert window.section_panel.spin_native_outline_grid.suffix() == " mm"
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_cutline_and_outline_handlers_dispatch_computation_to_task_threads() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    session = _artifact_box_session()
    window = MainWindow()
    window._artifact_session = session
    window.viewport.objects = [_projected_scene_object(session)]
    window.viewport.selected_index = 0
    controller = Mock()
    cutline_item = object()
    outline_item = object()
    controller.begin_cutline.return_value = cutline_item
    controller.begin_outline.return_value = outline_item
    controller.execute.side_effect = ["cutline-result", "outline-result"]
    try:
        with (
            patch.object(
                window,
                "_artifact_measurement_controller",
                return_value=controller,
            ),
            patch.object(
                window,
                "_native_record_workflow_progress",
                return_value=SimpleNamespace(
                    outline=SimpleNamespace(enabled=True),
                ),
            ),
            patch.object(window, "_start_task", return_value=True) as start_task,
        ):
            window.on_native_cutline_requested()
            cutline_thread = start_task.call_args.kwargs["thread"]
            assert isinstance(cutline_thread, TaskThread)
            assert cutline_thread._task_name == "native_cutline"
            assert cutline_thread._fn() == "cutline-result"

            window.on_native_outline_requested()
            outline_thread = start_task.call_args.kwargs["thread"]
            assert isinstance(outline_thread, TaskThread)
            assert outline_thread._task_name == "native_outline"
            assert outline_thread._fn() == "outline-result"

        controller.execute.assert_has_calls(
            [
                call(cutline_item),
                call(outline_item),
            ]
        )
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_downstream_measurement_handlers_enforce_record_derived_prerequisites() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    session = _artifact_box_session()
    window = MainWindow()
    window._artifact_session = session
    window.viewport.objects = [_projected_scene_object(session)]
    window.viewport.selected_index = 0
    controller = Mock()
    try:
        with (
            patch.object(
                window,
                "_artifact_measurement_controller",
                return_value=controller,
            ),
            patch.object(window, "_start_task") as start_task,
            patch.object(QMessageBox, "warning") as warning,
        ):
            window.on_native_outline_requested()
            window.on_native_rubbing_requested()

        controller.begin_outline.assert_not_called()
        controller.begin_rubbing.assert_not_called()
        start_task.assert_not_called()
        assert warning.call_count == 2
        assert session.document.records == ()
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_native_cutline_user_cancel_is_quiet_and_never_publishes_a_record() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    session = _artifact_box_session()
    window = MainWindow()
    window._artifact_session = session
    window.viewport.objects = [_projected_scene_object(session)]
    window.viewport.selected_index = 0
    try:
        with patch.object(window, "_start_task", return_value=True) as start_task:
            window.on_native_cutline_requested()
        callbacks = start_task.call_args.kwargs
        cancel_requested = callbacks["on_cancel_requested"]
        assert callable(cancel_requested)
        cancel_requested()
        controller = window._artifact_measurement_controller()
        assert controller.active_summaries == ()
        assert "취소 요청됨" in window.status_info.text()

        with (
            patch.object(window, "_publish_artifact_session_projection") as publish,
            patch.object(QMessageBox, "warning") as warning,
        ):
            callbacks["on_done"](object())
            callbacks["on_failed"]("StaleMeasurementOperationError: cancelled")
        publish.assert_not_called()
        warning.assert_not_called()
        assert "취소됨" in window.status_info.text()
        assert session.document.records == ()
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_native_outline_command_commits_verified_record_and_closed_preview() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    session = _artifact_box_session()
    obj = _projected_scene_object(session)
    window = MainWindow()
    window._artifact_session = session
    window._current_project_path = "/tmp/gui-outline.amr"
    window.viewport.objects = [obj]
    window.viewport.selected_index = 0
    captured: dict[str, object] = {}
    try:
        with (
            patch.object(
                window,
                "_publish_artifact_session_projection",
                side_effect=_capture_measurement_publication(window, captured),
            ),
            patch.object(window.viewport, "set_native_vector_preview") as preview,
            patch.object(window, "_sync_native_cutline_controls"),
        ):
            record_id = window._compute_and_commit_native_outline(
                view="right",
                precision_grid_mm=0.01,
                record_id="record:gui-outline",
                created_at="2026-07-11T00:00:02Z",
                operator="pytest",
            )

        assert record_id == "record:gui-outline"
        committed = captured["session"]
        assert isinstance(committed, ArtifactSession)
        record = committed.document.record_index[record_id]
        assert record.type == "vector.outline.v1"
        assert record.recipe["view"] == "right"
        assert record.recipe["precision_grid_mm"] == 0.01
        assert record.qc["outline_topology"]["topology_valid"] is True
        assert captured["kwargs"] == {
            "project_path": "/tmp/gui-outline.amr",
            "fit_camera": False,
            "expected_new_record_ids": ("record:gui-outline",),
            "status_text": (
                "✅ Right Outline 기록 | 1개 성분 · 0개 구멍 | grid 0.01 mm"
            ),
        }
        preview.assert_called_once()
        preview_payload = preview.call_args.args[0]
        assert preview_payload.kind.value == "outline"
        assert all(path.closed for path in preview_payload.paths)
        assert preview.call_args.kwargs["record_id"] == record_id
        np.testing.assert_array_equal(
            session.source_mesh.vertices,
            _artifact_box_session().source_mesh.vertices,
        )
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_native_rubbing_ui_exposes_physical_recipe_and_six_views() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    window = MainWindow()
    try:
        panel = window.section_panel
        combo = panel.combo_native_rubbing_view
        assert [combo.itemData(index) for index in range(combo.count())] == [
            "top",
            "bottom",
            "front",
            "back",
            "right",
            "left",
        ]
        assert panel.spin_native_rubbing_pixels_per_mm.value() == (
            DEFAULT_RUBBING_PIXELS_PER_MM
        )
        assert panel.spin_native_rubbing_pixels_per_mm.suffix() == " px/mm"
        assert panel.spin_native_rubbing_margin_um.value() == DEFAULT_RUBBING_MARGIN_UM
        assert panel.spin_native_rubbing_reference_radius_um.value() == (
            DEFAULT_RUBBING_REFERENCE_RADIUS_UM
        )
        assert panel.spin_native_rubbing_depth_quantization_um.value() == (
            DEFAULT_RUBBING_DEPTH_QUANTIZATION_UM
        )
        assert panel.spin_native_rubbing_black_point_um.value() == (
            DEFAULT_RUBBING_BLACK_POINT_UM
        )
        assert panel.spin_native_rubbing_strength.value() == (
            DEFAULT_RUBBING_INK_STRENGTH_PERCENT
        )
        assert panel.combo_native_rubbing_polarity.currentData() == (
            DEFAULT_RUBBING_POLARITY
        )
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_native_rubbing_command_commits_previews_and_recomputes_offline_export() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    session = _artifact_box_session()
    window = MainWindow()
    window._artifact_session = session
    window._current_project_path = "/tmp/gui-rubbing.amr"
    window.viewport.objects = [_projected_scene_object(session)]
    window.viewport.selected_index = 0
    captured: dict[str, object] = {}
    options = {
        "view": "top",
        "pixels_per_mm": 2,
        "margin_um": 0,
        "reference_radius_um": 500,
        "depth_quantization_um": 10,
        "black_point_um": 100,
        "ink_strength_percent": 100,
        "relief_polarity": "bidirectional",
    }
    try:
        with (
            patch.object(
                window,
                "_publish_artifact_session_projection",
                side_effect=_capture_measurement_publication(window, captured),
            ),
            patch.object(window, "_preview_native_rubbing") as preview,
            patch.object(window, "_sync_native_cutline_controls"),
        ):
            record_id = window._compute_and_commit_native_rubbing(
                options=options,
                record_id="record:gui-rubbing",
                created_at="2026-07-11T00:00:03Z",
                operator="pytest",
            )

        assert record_id == "record:gui-rubbing"
        committed = captured["session"]
        assert isinstance(committed, ArtifactSession)
        record = committed.document.record_index[record_id]
        assert record.type == "raster.digital_rubbing.v1"
        assert record.recipe["view"] == "top"
        assert record.recipe["pixel_policy"]["pixels_per_mm"] == 2
        receipt = rubbing_receipt_from_record(record)
        assert receipt["pixels_per_meter"] == 2_000
        assert captured["kwargs"] == {
            "project_path": "/tmp/gui-rubbing.amr",
            "fit_camera": False,
            "expected_new_record_ids": ("record:gui-rubbing",),
            "status_text": "✅ Top Digital Rubbing 기록 | 40×40 px · ink 0 px",
        }
        preview.assert_called_once()
        assert preview.call_args.args[1] == record_id

        window._artifact_session = committed
        raster = window._recompute_native_rubbing_record(committed, record)
        window._preview_native_rubbing(committed, record_id, raster)
        assert window._native_rubbing_preview_record_id == record_id
        assert window.section_panel.label_native_rubbing_preview.pixmap() is not None
        with tempfile.TemporaryDirectory() as temporary:
            package = window._export_native_rubbing_record(
                Path(temporary) / "gui-rubbing.amr-rubbing",
                record_id=record_id,
            )
            verified = validate_rubbing_export_package(package)
        assert verified.raster_sha256 == receipt["raster_sha256"]

        stale = committed.commit_preview(
            translation_mm=(0.5, 0.0, 0.0),
            rotation_deg=(0.0, 0.0, 0.0),
            scale=1.0,
            operator="pytest",
            created_at="2026-07-11T00:00:04Z",
            revision_id="align:rubbing-export-stale",
        )
        window._artifact_session = stale
        window.current_mesh = None
        window.viewport.objects = [_projected_scene_object(stale)]
        window.viewport.selected_index = 0
        window.current_mesh = window.viewport.selected_obj.mesh
        window._sync_native_cutline_controls(reset_offset=False)
        assert not window.section_panel.btn_native_rubbing_export.isEnabled()
        with tempfile.TemporaryDirectory() as temporary:
            with pytest.raises(ArtifactRubbingExportError, match="READY.*FRESH"):
                window._export_native_rubbing_record(
                    Path(temporary) / "stale.amr-rubbing",
                    record_id=record_id,
                )
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_native_vector_export_worker_stages_before_gui_final_publish(
    tmp_path: Path,
) -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    base = _artifact_box_session()
    record_id = "record:gui-vector-export-worker"
    computation = compute_artifact_cutline(base, _native_cutline_frame("top", 0.0))
    session = commit_vector_computation(
        base,
        computation,
        record_id=record_id,
        created_at="2026-07-11T00:00:07Z",
        operator="pytest",
    )
    obj = _projected_scene_object(session)
    window = MainWindow()
    window._artifact_session = session
    window.viewport.objects = [obj]
    window.viewport.selected_index = 0
    window.current_mesh = obj.mesh
    window.current_filepath = session.resolved_source_path
    window._preview_native_vector_record(session, record_id)
    destination = tmp_path / "worker.amr-vector"
    try:
        with (
            patch(
                "app_interactive.QFileDialog.getSaveFileName",
                return_value=(str(destination), ""),
            ),
            patch.object(window, "_start_task", return_value=True) as start_task,
        ):
            window.on_native_vector_export_requested()

        thread = start_task.call_args.kwargs["thread"]
        assert isinstance(thread, TaskThread)
        assert thread._task_name == "export_native_vector"
        result = thread._fn()
        assert not destination.exists()
        assert result.staging_directory.is_dir()

        with patch.object(QMessageBox, "warning") as warning:
            start_task.call_args.kwargs["on_done"](result)
        warning.assert_not_called()
        validate_vector_export_package(destination)
        assert not result.staging_directory.exists()
        assert (
            window._artifact_export_controller().summary(result.operation_id).state
            is ArtifactExportState.COMPLETED
        )
        assert "SVG" in window.status_info.text()
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_native_vector_export_start_failure_releases_ready_reservation(
    tmp_path: Path,
) -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    base = _artifact_box_session()
    record_id = "record:gui-vector-export-start-failure"
    computation = compute_artifact_cutline(base, _native_cutline_frame("top", 0.0))
    session = commit_vector_computation(
        base,
        computation,
        record_id=record_id,
        created_at="2026-07-11T00:00:08Z",
        operator="pytest",
    )
    obj = _projected_scene_object(session)
    window = MainWindow()
    window._artifact_session = session
    window.viewport.objects = [obj]
    window.viewport.selected_index = 0
    window.current_mesh = obj.mesh
    window.current_filepath = session.resolved_source_path
    window._preview_native_vector_record(session, record_id)
    destination = tmp_path / "start-failure.amr-vector"
    controller = window._artifact_export_controller()
    try:
        with (
            patch(
                "app_interactive.QFileDialog.getSaveFileName",
                return_value=(str(destination), ""),
            ),
            patch.object(
                window,
                "_start_task",
                side_effect=RuntimeError("injected thread start failure"),
            ),
            patch.object(controller, "cancel", wraps=controller.cancel) as cancel,
            patch.object(QMessageBox, "warning") as warning,
        ):
            window.on_native_vector_export_requested()

        cancel.assert_called_once()
        work_item = cancel.call_args.args[0]
        assert cancel.call_args.kwargs["reason"] == "task_start_failed"
        assert controller.summary(work_item).state is ArtifactExportState.CANCELLED
        assert controller.active_summaries == ()
        assert not destination.exists()
        assert "시작 실패" in window.status_info.text()
        warning.assert_called_once()
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_native_rubbing_export_worker_stages_before_gui_final_publish(
    tmp_path: Path,
) -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    base = _artifact_box_session()
    record_id = "record:gui-rubbing-export-worker"
    computation = compute_artifact_rubbing(
        base,
        "top",
        pixels_per_mm=2,
        margin_um=0,
        reference_radius_um=500,
        depth_quantization_um=10,
        black_point_um=100,
        ink_strength_percent=100,
        relief_polarity="bidirectional",
    )
    session = commit_artifact_rubbing(
        base,
        computation,
        record_id=record_id,
        created_at="2026-07-11T00:00:08Z",
        operator="pytest",
    )
    obj = _projected_scene_object(session)
    window = MainWindow()
    window._artifact_session = session
    window.viewport.objects = [obj]
    window.viewport.selected_index = 0
    window.current_mesh = obj.mesh
    window.current_filepath = session.resolved_source_path
    window._preview_native_rubbing(session, record_id, computation.raster)
    destination = tmp_path / "worker.amr-rubbing"
    try:
        with (
            patch(
                "app_interactive.QFileDialog.getSaveFileName",
                return_value=(str(destination), ""),
            ),
            patch.object(window, "_start_task", return_value=True) as start_task,
        ):
            window.on_native_rubbing_export_requested()

        thread = start_task.call_args.kwargs["thread"]
        assert isinstance(thread, TaskThread)
        assert thread._task_name == "export_native_digital_rubbing"
        result = thread._fn()
        assert not destination.exists()
        assert result.staging_directory.is_dir()

        with patch.object(QMessageBox, "warning") as warning:
            start_task.call_args.kwargs["on_done"](result)
        warning.assert_not_called()
        validate_rubbing_export_package(destination)
        assert not result.staging_directory.exists()
        assert (
            window._artifact_export_controller().summary(result.operation_id).state
            is ArtifactExportState.COMPLETED
        )
        assert "PNG" in window.status_info.text()
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_native_vector_export_pending_open_discards_stage_without_publishing(
    tmp_path: Path,
) -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    base = _artifact_box_session()
    record_id = "record:gui-vector-export-open-race"
    computation = compute_artifact_cutline(base, _native_cutline_frame("top", 0.0))
    session = commit_vector_computation(
        base,
        computation,
        record_id=record_id,
        created_at="2026-07-11T00:00:09Z",
        operator="pytest",
    )
    obj = _projected_scene_object(session)
    window = MainWindow()
    window._artifact_session = session
    window.viewport.objects = [obj]
    window.viewport.selected_index = 0
    window.current_mesh = obj.mesh
    window.current_filepath = session.resolved_source_path
    window._preview_native_vector_record(session, record_id)
    destination = tmp_path / "open-race.amr-vector"
    workbench = window._artifact_workbench_controller()
    try:
        with (
            patch(
                "app_interactive.QFileDialog.getSaveFileName",
                return_value=(str(destination), ""),
            ),
            patch.object(window, "_start_task", return_value=True) as start_task,
        ):
            window.on_native_vector_export_requested()
        result = start_task.call_args.kwargs["thread"]._fn()
        ticket = workbench.begin_new_import(
            "/source/replacement.ply",
            ConfirmedSourceMetadata(
                unit="cm",
                source_x="+X",
                source_y="+Y",
                source_z="+Z",
                handedness="right",
            ),
            software_version="test",
            operator="pytest",
        )

        with patch.object(QMessageBox, "warning") as warning:
            start_task.call_args.kwargs["on_done"](result)
        warning.assert_called_once()
        assert not destination.exists()
        assert not result.staging_directory.exists()
        assert (
            window._artifact_export_controller().summary(result.operation_id).state
            is ArtifactExportState.CANCELLED
        )
        workbench.cancel_load(ticket)
    finally:
        if workbench.snapshot.pending_load is not None:
            workbench.cancel_load(workbench.snapshot.pending_load)
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_native_rubbing_export_pending_open_discards_stage_without_publishing(
    tmp_path: Path,
) -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    base = _artifact_box_session()
    record_id = "record:gui-rubbing-export-open-race"
    computation = compute_artifact_rubbing(
        base,
        "top",
        pixels_per_mm=2,
        margin_um=0,
        reference_radius_um=500,
        depth_quantization_um=10,
        black_point_um=100,
        ink_strength_percent=100,
        relief_polarity="bidirectional",
    )
    session = commit_artifact_rubbing(
        base,
        computation,
        record_id=record_id,
        created_at="2026-07-11T00:00:09Z",
        operator="pytest",
    )
    obj = _projected_scene_object(session)
    window = MainWindow()
    window._artifact_session = session
    window.viewport.objects = [obj]
    window.viewport.selected_index = 0
    window.current_mesh = obj.mesh
    window.current_filepath = session.resolved_source_path
    window._preview_native_rubbing(session, record_id, computation.raster)
    destination = tmp_path / "rubbing-open-race.amr-rubbing"
    workbench = window._artifact_workbench_controller()
    try:
        with (
            patch(
                "app_interactive.QFileDialog.getSaveFileName",
                return_value=(str(destination), ""),
            ),
            patch.object(window, "_start_task", return_value=True) as start_task,
        ):
            window.on_native_rubbing_export_requested()
        result = start_task.call_args.kwargs["thread"]._fn()
        ticket = workbench.begin_new_import(
            "/source/replacement-rubbing.ply",
            ConfirmedSourceMetadata(
                unit="cm",
                source_x="+X",
                source_y="+Y",
                source_z="+Z",
                handedness="right",
            ),
            software_version="test",
            operator="pytest",
        )

        with patch.object(QMessageBox, "warning") as warning:
            start_task.call_args.kwargs["on_done"](result)
        warning.assert_called_once()
        assert not destination.exists()
        assert not result.staging_directory.exists()
        assert (
            window._artifact_export_controller().summary(result.operation_id).state
            is ArtifactExportState.CANCELLED
        )
        workbench.cancel_load(ticket)
    finally:
        if workbench.snapshot.pending_load is not None:
            workbench.cancel_load(workbench.snapshot.pending_load)
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_native_rubbing_handler_uses_worker_and_late_result_cannot_overwrite_session() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    session = _artifact_box_session()
    window = MainWindow()
    window._artifact_session = session
    window._current_project_path = "/tmp/gui-rubbing-late.amr"
    window.viewport.objects = [_projected_scene_object(session)]
    window.viewport.selected_index = 0
    try:
        with (
            patch.object(
                window,
                "_native_record_workflow_progress",
                return_value=SimpleNamespace(
                    rubbing=SimpleNamespace(enabled=True),
                ),
            ),
            patch.object(window, "_start_task", return_value=True) as start_task,
        ):
            window.on_native_rubbing_requested()
        assert start_task.call_count == 1
        thread = start_task.call_args.kwargs["thread"]
        assert isinstance(thread, TaskThread)
        result = thread._fn()

        controller = window._artifact_workbench_controller()
        align = controller.prepare_align_commit(
            translation_mm=(1.0, 0.0, 0.0),
            rotation_deg=(0.0, 0.0, 0.0),
            scale=1.0,
            pivot_mm=(0.0, 0.0, 0.0),
            operator="pytest",
            created_at="2026-07-11T00:00:05Z",
            revision_id="align:late-rubbing",
        )
        assert align is not None
        activation = controller.activate_projection(align)
        controller.finalize_projection(activation)
        window._artifact_session = align.candidate_session

        with (
            patch.object(window, "_publish_artifact_session_projection") as publish,
            patch.object(QMessageBox, "warning") as warning,
        ):
            start_task.call_args.kwargs["on_done"](result)
        publish.assert_not_called()
        warning.assert_not_called()
        assert "결과 폐기" in window.status_info.text()
        assert window._artifact_session.document.records == ()
        assert window._artifact_session.document.active_align_revision_id == (
            "align:late-rubbing"
        )
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_native_rubbing_and_flattened_exports_cannot_bypass_verified_path() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    session = _artifact_box_session()
    window = MainWindow()
    window._artifact_session = session
    window.viewport.objects = [_projected_scene_object(session)]
    window.viewport.selected_index = 0
    try:
        with (
            patch.object(QMessageBox, "warning") as warning,
            patch.object(window, "_start_task") as start_task,
        ):
            for export_type in (
                "review_sheet",
                "flat_svg",
                "rubbing",
                "rubbing_digital",
                "rubbing_view_cyl",
            ):
                window.on_export_requested({"type": export_type})
        assert warning.call_count == 5
        start_task.assert_not_called()
        assert "legacy" in window.status_info.text()
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_native_profile_exports_cannot_bypass_verified_vector_path() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    session = _artifact_box_session()
    window = MainWindow()
    window._artifact_session = session
    window.viewport.objects = [_projected_scene_object(session)]
    window.viewport.selected_index = 0
    try:
        with (
            patch.object(QMessageBox, "warning"),
            patch.object(window, "export_2d_profile_package") as legacy_package,
            patch.object(window, "export_2d_profile") as legacy_single,
        ):
            window.on_export_requested({"type": "profile_2d_package"})
            window.on_export_requested({"type": "profile_2d", "view": "top"})
        legacy_package.assert_not_called()
        legacy_single.assert_not_called()
        assert "legacy" in window.status_info.text()

        with (
            patch.object(QMessageBox, "warning"),
            patch("app_interactive.ProfileExportThread") as export_thread,
        ):
            window.export_2d_profile_package()
            window.export_2d_profile("top")
        export_thread.assert_not_called()
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_native_undo_cancels_preview_then_activates_parent_revision() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    initial = _artifact_session()
    committed = initial.commit_preview(
        translation_mm=(4.0, 0.0, 0.0),
        rotation_deg=(0.0, 0.0, 15.0),
        scale=1.0,
        pivot_mm=(10.0, 20.0, 30.0),
        operator="pytest",
        created_at="2026-07-11T00:00:01Z",
        revision_id="align:committed",
    )
    obj = _projected_scene_object(committed)
    obj.translation = np.array([9.0, 0.0, 0.0], dtype=np.float64)
    window = MainWindow()
    window._artifact_session = committed
    window.viewport.objects = [obj]
    window.viewport.selected_index = 0
    try:
        with patch.object(window, "_publish_artifact_session_projection") as publish:
            window.undo_last_action()
        publish.assert_not_called()
        np.testing.assert_array_equal(obj.translation, [0.0, 0.0, 0.0])
        assert window._artifact_session is committed

        captured: dict[str, ArtifactSession] = {}

        def capture(candidate, **_kwargs):
            captured["session"] = candidate

        with patch.object(
            window,
            "_publish_artifact_session_projection",
            side_effect=capture,
        ):
            window.undo_last_action()
        restored = captured["session"]
        assert restored.document.active_align_revision_id == initial.document.active_align_revision_id
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_native_rubbing_worker_accepts_a_new_record_sorted_before_existing_ids() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    base = _artifact_box_session()
    computation = compute_artifact_rubbing(
        base,
        "top",
        pixels_per_mm=2,
        margin_um=0,
        reference_radius_um=500,
        depth_quantization_um=10,
        black_point_um=100,
        ink_strength_percent=100,
        relief_polarity="bidirectional",
    )
    existing = commit_artifact_rubbing(
        base,
        computation,
        record_id="record:rubbing:z-existing",
        created_at="2026-07-11T00:00:03Z",
        operator="pytest",
    )
    window = MainWindow()
    window._artifact_session = existing
    window._current_project_path = "/tmp/gui-rubbing-sorted.amr"
    window.viewport.objects = [_projected_scene_object(existing)]
    window.viewport.selected_index = 0
    captured: dict[str, object] = {}
    try:
        with (
            patch.object(
                window,
                "_publish_artifact_session_projection",
                side_effect=_capture_measurement_publication(window, captured),
            ),
            patch.object(window, "_preview_native_rubbing"),
            patch.object(window, "_sync_native_cutline_controls"),
        ):
            record_id = window._compute_and_commit_native_rubbing(
                options={
                    "view": "top",
                    "pixels_per_mm": 2,
                    "margin_um": 0,
                    "reference_radius_um": 500,
                    "depth_quantization_um": 10,
                    "black_point_um": 100,
                    "ink_strength_percent": 100,
                    "relief_polarity": "bidirectional",
                },
                record_id="record:rubbing:a-new",
                created_at="2026-07-11T00:00:04Z",
                operator="pytest",
            )

        assert record_id == "record:rubbing:a-new"
        committed = captured["session"]
        assert isinstance(committed, ArtifactSession)
        assert tuple(record.id for record in committed.document.records) == (
            "record:rubbing:a-new",
            "record:rubbing:z-existing",
        )
        assert captured["kwargs"]["expected_new_record_ids"] == (
            "record:rubbing:a-new",
        )
        assert window._latest_native_rubbing_record().id == "record:rubbing:a-new"
        assert window._current_native_rubbing_record() is None
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_native_projection_is_published_before_scene_swap_notifications() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    initial = _artifact_session()
    candidate = initial.commit_preview(
        translation_mm=(3.0, 2.0, 1.0),
        rotation_deg=(0.0, 0.0, 5.0),
        scale=1.0,
        operator="pytest",
        created_at="2026-07-11T00:00:01Z",
        revision_id="align:candidate",
    )
    old_obj = _projected_scene_object(initial)
    window = MainWindow()
    window._artifact_session = initial
    window.viewport.objects = [old_obj]
    window.viewport.selected_index = 0
    transition = window._artifact_workbench_controller().prepare_session_commit(
        initial,
        candidate,
        kind=WorkflowTransitionKind.ALIGN_COMMIT,
        project_path="/tmp/candidate.amr",
    )
    prepared_holder: dict[str, SceneObject] = {}

    def prepare(mesh, name, *, artifact_binding):
        prepared = SceneObject(mesh, name)
        prepared._amr_artifact_projection_snapshot = artifact_binding
        prepared_holder["obj"] = prepared
        return prepared

    def swap(objects, *, selected_index, fit_camera):
        assert window._artifact_session is candidate
        assert window.current_mesh is objects[0].mesh
        assert objects[0]._amr_artifact_projection_snapshot == candidate.projection_snapshot()
        previous = list(window.viewport.objects)
        window.viewport.objects = list(objects)
        window.viewport.selected_index = selected_index
        assert fit_camera is False
        return previous

    try:
        with (
            patch.object(window.viewport, "prepare_mesh_object", side_effect=prepare),
            patch.object(window.viewport, "validate_prepared_scene"),
            patch.object(window.viewport, "swap_prepared_scene", side_effect=swap),
            patch.object(window.viewport, "cleanup_scene_objects") as cleanup,
            patch.object(window.scene_panel, "update_list"),
            patch.object(window, "sync_transform_panel"),
            patch.object(window, "_sync_tile_panel"),
        ):
            window._publish_artifact_session_projection(
                candidate,
                project_path="/tmp/candidate.amr",
                fit_camera=False,
                status_text="published",
                workflow_transition=transition,
            )

        assert window._artifact_session is candidate
        assert window.viewport.objects == [prepared_holder["obj"]]
        cleanup.assert_called_once_with([old_obj])
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_native_projection_publish_failure_restores_exact_live_authority() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    initial = _artifact_session()
    candidate = initial.commit_preview(
        translation_mm=(3.0, 2.0, 1.0),
        rotation_deg=(0.0, 0.0, 5.0),
        scale=1.0,
        operator="pytest",
        created_at="2026-07-11T00:00:01Z",
        revision_id="align:will-fail",
    )
    old_obj = _projected_scene_object(initial)
    old_obj.translation = np.array([7.0, 8.0, 9.0], dtype=np.float64)
    old_obj.rotation = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    old_cleanup = Mock()
    old_obj.cleanup = old_cleanup
    prepared = _projected_scene_object(candidate)
    prepared_cleanup = Mock()
    prepared.cleanup = prepared_cleanup
    window = MainWindow()
    window._artifact_session = initial
    window.viewport.objects = [old_obj]
    window.viewport.selected_index = 0
    window.current_mesh = old_obj.mesh
    window.current_filepath = "/source/live.ply"
    window._current_project_path = "/projects/live.amr"
    window._project_load_failed = True
    transition = window._artifact_workbench_controller().prepare_session_commit(
        initial,
        candidate,
        kind=WorkflowTransitionKind.ALIGN_COMMIT,
        project_path="/projects/candidate.amr",
    )
    window._flattened_cache["sentinel"] = object()
    window._flatten_recommendation_cache[1] = ((), {})
    emitted_meshes: list[object] = []
    emitted_selections: list[int] = []
    window.viewport.meshLoaded.connect(emitted_meshes.append)
    window.viewport.selectionChanged.connect(emitted_selections.append)
    try:
        with (
            patch.object(window.viewport, "prepare_mesh_object", return_value=prepared),
            patch.object(
                window.viewport,
                "validate_prepared_scene",
                side_effect=RuntimeError("injected VBO validation failure"),
            ),
            patch.object(window.scene_panel, "update_list"),
            patch.object(window, "sync_transform_panel"),
            patch.object(window, "_sync_tile_panel"),
        ):
            with pytest.raises(RuntimeError, match="injected VBO"):
                window._publish_artifact_session_projection(
                    candidate,
                    project_path="/projects/candidate.amr",
                    fit_camera=False,
                    status_text="must not publish",
                    workflow_transition=transition,
                )

        assert window._artifact_session is initial
        assert window.viewport.objects == [old_obj]
        assert window.viewport.objects[0] is old_obj
        np.testing.assert_array_equal(old_obj.translation, [7.0, 8.0, 9.0])
        np.testing.assert_array_equal(old_obj.rotation, [1.0, 2.0, 3.0])
        assert window.current_mesh is old_obj.mesh
        assert window.current_filepath == "/source/live.ply"
        assert window._current_project_path == "/projects/live.amr"
        assert window._project_load_failed
        assert "sentinel" in window._flattened_cache
        assert 1 in window._flatten_recommendation_cache
        assert emitted_meshes == []
        assert emitted_selections == []
        old_cleanup.assert_not_called()
        prepared_cleanup.assert_called_once()
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_scene_swap_failure_after_tentative_authority_rolls_back_workbench() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    initial = _artifact_session()
    candidate = initial.commit_preview(
        translation_mm=(3.0, 0.0, 0.0),
        rotation_deg=(0.0, 0.0, 0.0),
        scale=1.0,
        operator="pytest",
        created_at="2026-07-11T00:00:01Z",
        revision_id="align:swap-failure",
    )
    old_obj = _projected_scene_object(initial)
    prepared = _projected_scene_object(candidate)
    prepared.cleanup = Mock()
    window = MainWindow()
    window._artifact_session = initial
    window._current_project_path = "/projects/live.amr"
    window.current_mesh = old_obj.mesh
    window.current_filepath = "/source/live.ply"
    window.viewport.objects = [old_obj]
    window.viewport.selected_index = 0
    controller = window._artifact_workbench_controller()
    transition = controller.prepare_session_commit(
        initial,
        candidate,
        kind=WorkflowTransitionKind.ALIGN_COMMIT,
        project_path="/projects/candidate.amr",
    )

    def fail_after_authority(*_args, **_kwargs):
        assert window._artifact_session is candidate
        assert controller.session is candidate
        raise RuntimeError("injected post-authority scene swap failure")

    try:
        with (
            patch.object(window.viewport, "prepare_mesh_object", return_value=prepared),
            patch.object(window.viewport, "validate_prepared_scene"),
            patch.object(window.viewport, "swap_prepared_scene", side_effect=fail_after_authority),
            patch.object(window.scene_panel, "update_list"),
            patch.object(window, "sync_transform_panel"),
            patch.object(window, "_sync_tile_panel"),
        ):
            with pytest.raises(RuntimeError, match="post-authority"):
                window._publish_artifact_session_projection(
                    candidate,
                    project_path="/projects/candidate.amr",
                    fit_camera=False,
                    status_text="must roll back",
                    workflow_transition=transition,
                )

        assert window._artifact_session is initial
        assert controller.session is initial
        assert controller.snapshot.failure is not None
        assert controller.snapshot.authority_epoch >= 3
        assert window.viewport.objects == [old_obj]
        assert window.current_mesh is old_obj.mesh
        assert window.current_filepath == "/source/live.ply"
        assert window._current_project_path == "/projects/live.amr"
        prepared.cleanup.assert_called_once()
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_record_binding_finalize_uncertainty_keeps_reopen_required_banner() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    session = _artifact_box_session()
    obj = _projected_scene_object(session)
    window = MainWindow()
    window._artifact_session = session
    window.viewport.objects = [obj]
    window.viewport.selected_index = 0
    window.current_mesh = obj.mesh
    window.current_filepath = session.resolved_source_path
    controller = window._artifact_workbench_controller()
    try:
        with patch.object(window, "_start_task", return_value=True) as start_task:
            window.on_native_cutline_requested()
        result = start_task.call_args.kwargs["thread"]._fn()

        with (
            patch.object(
                controller,
                "finalize_record_binding",
                side_effect=RuntimeError("injected record finalize uncertainty"),
            ),
            patch.object(QMessageBox, "warning") as warning,
            patch.object(QMessageBox, "critical") as critical,
        ):
            start_task.call_args.kwargs["on_done"](result)

        warning.assert_not_called()
        critical.assert_called_once()
        assert "다시 여세요" in critical.call_args.args[2]
        assert window._artifact_authority_faulted
        assert window._project_load_failed
        assert window._current_project_path is None
        assert controller.snapshot.failure is not None
        assert controller.snapshot.failure.fatal
        assert not controller.snapshot.can_save
        assert not controller.snapshot.can_measure
        assert "문서 권위 복원 실패" in window.status_info.text()
        assert "다시 여세요" in window.status_info.text()
        assert window._artifact_session is session
        assert obj._amr_artifact_projection_snapshot == session.projection_snapshot()
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


@pytest.mark.parametrize("fault_stage", ["rollback", "restore", "finalize"])
def test_uncertain_projection_recovery_faults_authority_and_blocks_writes(
    fault_stage: str,
) -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    initial = _artifact_session()
    candidate = initial.commit_preview(
        translation_mm=(4.0, 0.0, 0.0),
        rotation_deg=(0.0, 0.0, 0.0),
        scale=1.0,
        operator="pytest",
        created_at="2026-07-11T00:00:01Z",
        revision_id=f"align:fault-{fault_stage}",
    )
    old_obj = _projected_scene_object(initial)
    prepared = _projected_scene_object(candidate)
    prepared.cleanup = Mock()
    window = MainWindow()
    window._artifact_session = initial
    window._current_project_path = "/projects/live.amr"
    window.current_mesh = old_obj.mesh
    window.current_filepath = "/source/live.ply"
    window.viewport.objects = [old_obj]
    window.viewport.selected_index = 0
    controller = window._artifact_workbench_controller()
    transition = controller.prepare_session_commit(
        initial,
        candidate,
        kind=WorkflowTransitionKind.ALIGN_COMMIT,
        project_path="/projects/candidate.amr",
    )

    def swap_or_fail(*_args, **_kwargs):
        if fault_stage == "finalize":
            window.viewport.objects = [prepared]
            window.viewport.selected_index = 0
            return [old_obj]
        raise RuntimeError("injected scene swap failure")

    try:
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(window.viewport, "prepare_mesh_object", return_value=prepared)
            )
            stack.enter_context(patch.object(window.viewport, "validate_prepared_scene"))
            stack.enter_context(
                patch.object(
                    window.viewport,
                    "swap_prepared_scene",
                    side_effect=swap_or_fail,
                )
            )
            if fault_stage == "rollback":
                stack.enter_context(
                    patch.object(
                        controller,
                        "rollback_projection",
                        side_effect=RuntimeError("injected rollback failure"),
                    )
                )
            elif fault_stage == "restore":
                stack.enter_context(
                    patch.object(
                        window,
                        "_restore_live_scene_after_failed_swap",
                        side_effect=RuntimeError("injected restore failure"),
                    )
                )
            else:
                stack.enter_context(
                    patch.object(
                        controller,
                        "finalize_projection",
                        side_effect=RuntimeError("injected finalize failure"),
                    )
                )

            with pytest.raises(RuntimeError, match="injected"):
                window._publish_artifact_session_projection(
                    candidate,
                    project_path="/projects/candidate.amr",
                    fit_camera=False,
                    status_text="must fault closed",
                    workflow_transition=transition,
                )

        assert window._artifact_authority_faulted
        assert window._project_load_failed
        assert window._current_project_path is None
        assert controller.snapshot.failure is not None
        assert controller.snapshot.failure.fatal
        assert not controller.snapshot.can_save
        assert not controller.snapshot.can_measure
        assert window._write_project("/tmp/must-not-write.amr") is False
        with pytest.raises(ArtifactSessionError, match="faulted"):
            window._require_native_measurement_session(old_obj)
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_artifact_source_finish_rejects_byte_mismatch_before_projection_publish() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    initial = _artifact_session()
    old_obj = _projected_scene_object(initial)
    changed_fingerprint = SourceFingerprint(
        sha256="f" * 64,
        size_bytes=initial.source_mesh.source_identity.size_bytes,
        mtime_ns=2,
        original_name="lookalike.ply",
        format="ply",
    )
    changed_mesh = _reloaded_source_mesh(initial, fingerprint=changed_fingerprint)
    window = MainWindow()
    window._artifact_session = initial
    window.viewport.objects = [old_obj]
    window.viewport.selected_index = 0
    window.current_mesh = old_obj.mesh
    window.current_filepath = "/source/live.ply"
    window._current_project_path = "/projects/live.amr"
    window._artifact_load_active = True
    window._artifact_pending_document = initial.document
    window._artifact_pending_project_path = "/projects/candidate.amr"
    try:
        with (
            patch.object(window, "_publish_artifact_session_projection") as publish,
            patch("app_interactive.QMessageBox.critical") as critical,
        ):
            window._finish_artifact_source_loaded(
                changed_mesh,
                "/relocated/lookalike.ply",
            )

        publish.assert_not_called()
        critical.assert_called_once()
        assert "source bytes" in critical.call_args.args[2]
        assert window._artifact_session is initial
        assert window.viewport.objects == [old_obj]
        assert window.current_mesh is old_obj.mesh
        assert window.current_filepath == "/source/live.ply"
        assert window._current_project_path == "/projects/live.amr"
        assert not window._artifact_load_active
        assert window._artifact_pending_document is None
        assert window._artifact_pending_project_path is None
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_artifact_source_finish_success_clears_pending_transaction_state() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    initial = _artifact_session()
    reloaded = _reloaded_source_mesh(initial)
    window = MainWindow()
    window._artifact_load_active = True
    window._artifact_pending_document = initial.document
    window._artifact_pending_project_path = "/projects/native.amr"
    window._artifact_pending_source_metadata = None
    captured: dict[str, object] = {}
    try:
        def capture(session, **kwargs):
            captured["session"] = session
            captured["kwargs"] = kwargs

        with patch.object(
            window,
            "_publish_artifact_session_projection",
            side_effect=capture,
        ):
            window._finish_artifact_source_loaded(
                reloaded,
                "/relocated/native.raw-scan",
            )

        rebound = captured["session"]
        assert isinstance(rebound, ArtifactSession)
        assert rebound.document is initial.document
        assert rebound.resolved_source_path == "/relocated/native.raw-scan"
        assert captured["kwargs"]["project_path"] == "/projects/native.amr"
        assert not window._artifact_load_active
        assert window._artifact_pending_document is None
        assert window._artifact_pending_project_path is None
        assert window._artifact_pending_source_metadata is None
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_native_destructive_alignment_tools_fail_closed() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    session = _artifact_session()
    obj = _projected_scene_object(session)
    window = MainWindow()
    window._artifact_session = session
    window.viewport.objects = [obj]
    window.viewport.selected_index = 0
    before = obj.mesh.vertices.copy()
    try:
        with (
            patch("app_interactive.QMessageBox.warning") as warning,
            patch.object(window.viewport, "bake_object_transform") as destructive_bake,
            patch.object(window.viewport, "update_vbo") as update_vbo,
        ):
            window.fit_ground_plane()
            window.bake_and_center()
            window.start_floor_picking()

        assert warning.call_count == 3
        destructive_bake.assert_not_called()
        update_vbo.assert_not_called()
        np.testing.assert_array_equal(obj.mesh.vertices, before)
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_drop_event_routes_through_explicit_native_open_contract() -> None:
    filepath = "/tmp/dropped-artifact.ply"
    window_like = SimpleNamespace(open_file_path=Mock())
    accept = Mock()
    event = SimpleNamespace(
        mimeData=lambda: SimpleNamespace(
            urls=lambda: [SimpleNamespace(toLocalFile=lambda: filepath)]
        ),
        acceptProposedAction=accept,
    )

    MainWindow.dropEvent(window_like, event)

    window_like.open_file_path.assert_called_once_with(filepath, prompt_unit=True)
    accept.assert_called_once()


def test_failed_worker_start_rolls_back_ticketed_artifact_open() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    window = MainWindow()
    metadata = {
        "unit": "mm",
        "axes": {"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
        "handedness": "right",
        "confirmation_status": "confirmed",
    }
    try:
        with patch.object(window, "_start_async_load", return_value=False):
            window._start_artifact_source_import("/source/artifact.ply", metadata)

        assert not window._artifact_load_active
        assert window._artifact_load_ticket is None
        assert window._artifact_pending_source_metadata is None
        assert window._artifact_workbench.snapshot.phase.value == "initial"
        assert window._artifact_workbench.snapshot.pending_load is None
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_stale_mesh_callbacks_cannot_touch_new_worker_or_request() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    window = MainWindow()
    old_owner = Mock()
    new_owner = Mock()
    window._mesh_load_thread = new_owner
    window._mesh_load_request_id = "mesh-load:new"
    try:
        with (
            patch.object(window, "_on_mesh_load_thread_loaded") as loaded,
            patch.object(window, "_on_mesh_load_thread_failed") as failed,
        ):
            window._dispatch_mesh_load_success(
                old_owner,
                "mesh-load:old",
                None,
                object(),
                "/source/old.ply",
            )
            window._dispatch_mesh_load_failure(
                old_owner,
                "mesh-load:old",
                None,
                "late failure",
            )
            window._dispatch_mesh_load_finished(old_owner, "mesh-load:old")
            loaded.assert_not_called()
            failed.assert_not_called()

            assert window._mesh_load_thread is new_owner
            assert window._mesh_load_request_id == "mesh-load:new"
            old_owner.deleteLater.assert_called_once()

            window._dispatch_mesh_load_success(
                new_owner,
                "mesh-load:new",
                None,
                object(),
                "/source/new.ply",
            )
            loaded.assert_called_once()
    finally:
        window._mesh_load_thread = None
        window._mesh_load_request_id = None
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_direct_legacy_load_cannot_create_hybrid_native_scene() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    session = _artifact_session()
    old_obj = _projected_scene_object(session)
    window = MainWindow()
    window._artifact_session = session
    window.viewport.objects = [old_obj]
    window.viewport.selected_index = 0
    window.current_mesh = old_obj.mesh
    window.current_filepath = "/source/native.ply"
    try:
        with (
            patch.object(window, "_start_async_load") as start_load,
            patch("app_interactive.QMessageBox.warning") as warning,
        ):
            assert window.load_mesh("/source/legacy.ply", 2.0) is False
        warning.assert_called_once()
        start_load.assert_not_called()

        with (
            patch.object(window.viewport, "add_mesh_object") as add_mesh,
            patch("app_interactive.QMessageBox.critical") as critical,
        ):
            window._on_mesh_load_thread_loaded(object(), "/source/legacy.ply")
        critical.assert_called_once()
        add_mesh.assert_not_called()
        assert window._artifact_session is session
        assert window.viewport.objects == [old_obj]
        assert window.current_mesh is old_obj.mesh
        assert window.current_filepath == "/source/native.ply"
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_project_source_mismatch_is_blocked_before_scene_mutation() -> None:
    QStandardPaths.setTestModeEnabled(True)
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    expected = _fingerprint(b"expected")
    actual = _fingerprint(b"changed")
    obj_state = {
        "mesh": {
            "path": "/original/artifact.ply",
            "source": {
                "identity": expected.to_dict(),
                "binding_status": "captured_at_import",
            },
        }
    }
    mesh = SimpleNamespace(source_identity=actual)
    sentinel_mesh = object()

    window = MainWindow()
    window._project_load_active = True
    window._project_load_current = obj_state
    window._project_load_from_legacy = False
    window.current_mesh = sentinel_mesh
    window.current_filepath = "unchanged"
    try:
        with (
            patch.object(window, "_abort_project_source_load") as abort,
            patch.object(window.viewport, "add_mesh_object") as add_mesh,
        ):
            window._on_mesh_load_thread_loaded(mesh, "/candidate/artifact.ply")

        abort.assert_called_once()
        verification = abort.call_args.args[0]
        assert verification.status is SourceVerificationStatus.MISMATCH
        add_mesh.assert_not_called()
        assert window.current_mesh is sentinel_mesh
        assert window.current_filepath == "unchanged"
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_verified_project_source_is_staged_without_touching_live_scene() -> None:
    QStandardPaths.setTestModeEnabled(True)
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    identity = _fingerprint(b"verified")
    obj_state = {
        "name": "Staged artifact",
        "mesh": {
            "path": "/original/artifact.ply",
            "source": {
                "identity": identity.to_dict(),
                "binding_status": "captured_at_import",
                "parse_format": "ply",
            },
        },
    }
    mesh = SimpleNamespace(source_identity=identity)
    sentinel_mesh = object()

    window = MainWindow()
    window._project_load_active = True
    window._project_load_current = obj_state
    window._project_load_from_legacy = False
    window.current_mesh = sentinel_mesh
    window.current_filepath = "live-scene-source"
    try:
        with patch.object(window.viewport, "add_mesh_object") as add_mesh:
            window._on_mesh_load_thread_loaded(mesh, "/original/artifact.ply")

        add_mesh.assert_not_called()
        assert window.current_mesh is sentinel_mesh
        assert window.current_filepath == "live-scene-source"
        assert len(window._project_staged_objects) == 1
        assert window._project_staged_objects[0][0] is mesh
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_project_open_preserves_scene_on_invalid_container_and_forces_v1_save_as() -> None:
    QStandardPaths.setTestModeEnabled(True)
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    window = MainWindow()
    try:
        with (
            patch("app_interactive.load_amr_project", side_effect=ProjectFormatError("corrupt")),
            patch.object(window.viewport, "clear_scene") as clear_scene,
            patch("app_interactive.QMessageBox.critical"),
        ):
            window.open_project_path("corrupt.amr")
        clear_scene.assert_not_called()

        legacy_doc = {
            "state": {
                "objects": [
                    {
                        "mesh": {
                            "path": "/legacy/artifact.ply",
                            "source_scale_factor": 1.0,
                            "source": {
                                "identity": None,
                                "binding_status": "legacy_unverified",
                            },
                        }
                    }
                ]
            },
            MIGRATION_MARKER_NAME: {
                "from_version": 1,
                "to_version": 2,
                "status": "legacy_unverified",
                "runtime_only": True,
                "requires_save_as": True,
            },
        }
        with (
            patch("app_interactive.load_amr_project", return_value=legacy_doc),
            patch.object(window.viewport, "clear_scene"),
            patch.object(window, "_start_next_project_object_load") as start_load,
        ):
            window.open_project_path("/projects/legacy.amr")

        start_load.assert_called_once()
        assert window._current_project_path is None
        assert window._project_requires_save_as
        assert window._legacy_project_path == "/projects/legacy.amr"

        legacy_object = SimpleNamespace(
            mesh=SimpleNamespace(n_faces=0),
            translation=[0.0, 0.0, 0.0],
            rotation=[0.0, 0.0, 0.0],
            scale=1.0,
            fixed_state_valid=True,
            fixed_translation=[1.0, 2.0, 3.0],
            fixed_rotation=[10.0, 20.0, 30.0],
            fixed_scale=2.0,
        )
        window._apply_loaded_object_state(
            legacy_object,
            {
                "alignment": {"status": "legacy_unverifiable"},
                "transform": {"fixed_state_valid": True},
            },
        )
        assert legacy_object._amr_alignment_status == "legacy_unverifiable"
        assert legacy_object.fixed_state_valid is False
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_project_open_dispatches_artifact_payload_to_native_loader() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    document = _artifact_session().document
    window = MainWindow()
    try:
        with (
            patch(
                "app_interactive.load_amr_project",
                side_effect=UnsupportedPayloadError(
                    ARTIFACT_PAYLOAD_TYPE,
                    "1.0.0",
                ),
            ),
            patch(
                "app_interactive.load_amr_artifact_project",
                return_value=document,
            ) as load_native,
            patch.object(window, "_start_artifact_project_load") as start_native,
        ):
            window.open_project_path("/projects/native.amr")

        load_native.assert_called_once_with("/projects/native.amr")
        start_native.assert_called_once_with(document, "/projects/native.amr")
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_project_open_cannot_overlap_an_active_artifact_import() -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    window = MainWindow()
    window._artifact_load_active = True
    try:
        with (
            patch("app_interactive.load_amr_project") as load_project,
            patch.object(QMessageBox, "information") as information,
        ):
            window.open_project_path("/projects/overlap.amr")

        load_project.assert_not_called()
        information.assert_called_once()
        assert not window._project_load_active
    finally:
        window._artifact_load_active = False
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_failed_project_verification_discards_staging_and_preserves_live_save_target() -> None:
    QStandardPaths.setTestModeEnabled(True)
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    expected = _fingerprint(b"expected")
    actual = _fingerprint(b"changed")
    verification, _binding = _verify_loaded_project_source(
        SimpleNamespace(source_identity=actual),
        {
            "mesh": {
                "path": "/project/artifact.ply",
                "source": {
                    "identity": expected.to_dict(),
                    "binding_status": "captured_at_import",
                    "parse_format": "ply",
                },
            }
        },
        "/candidate/artifact.ply",
        migrated_from_v1=False,
    )
    assert verification.status is SourceVerificationStatus.MISMATCH

    window = MainWindow()
    window._current_project_path = "/project/current-work.amr"
    window._project_pending_path = "/project/failed-attempt.amr"
    window._project_previous_context = {
        "current_project_path": "/project/current-work.amr",
        "requires_save_as": False,
        "legacy_project_path": None,
        "has_legacy_bindings": False,
    }
    window._project_staged_objects = [(object(), "staged", {}, verification, "captured_at_import")]
    window._project_load_active = True
    try:
        with patch("app_interactive.QMessageBox.critical"):
            window._abort_project_source_load(
                verification,
                message="mismatch",
            )

        assert window._current_project_path == "/project/current-work.amr"
        assert window._project_pending_path is None
        assert not window._project_load_failed
        assert window._project_staged_objects == []

        with (
            patch("app_interactive.QMessageBox.warning") as warning,
            patch.object(window, "_write_project", return_value=True) as write_project,
        ):
            window.save_project()
        warning.assert_not_called()
        write_project.assert_called_once_with("/project/current-work.amr")
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_failed_replacement_load_preserves_preexisting_save_block() -> None:
    QStandardPaths.setTestModeEnabled(True)
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    window = MainWindow()
    window._project_load_failed = False
    window._project_load_active = True
    window._project_pending_path = "/projects/replacement.amr"
    window._project_previous_context = {
        "current_project_path": None,
        "requires_save_as": False,
        "legacy_project_path": None,
        "has_legacy_bindings": False,
        "load_failed": True,
    }
    verification = SimpleNamespace(status=SourceVerificationStatus.MISMATCH)
    try:
        with patch("app_interactive.QMessageBox.critical"):
            window._abort_project_source_load(
                verification,
                message="replacement mismatch",
            )

        assert window._project_load_failed
        assert window._project_pending_path is None

        with (
            patch("app_interactive.QMessageBox.warning") as warning,
            patch("app_interactive.QFileDialog.getSaveFileName") as save_dialog,
        ):
            window.save_project_as()
        warning.assert_called_once()
        save_dialog.assert_not_called()
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_project_materialization_failure_restores_live_scene_and_save_target() -> None:
    QStandardPaths.setTestModeEnabled(True)
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    window = MainWindow()
    native_session = _artifact_session()
    window._artifact_session = native_session
    old_cleanup = Mock()
    new_cleanup = Mock()
    old_object = SimpleNamespace(cleanup=old_cleanup)
    new_object = SimpleNamespace(cleanup=new_cleanup, vertex_count=3, vbo_id=101)
    old_mesh = object()

    window.viewport.objects = [old_object]
    window.viewport.selected_index = 0
    window.current_mesh = old_mesh
    window.current_filepath = "/sources/live.ply"
    window._current_project_path = "/projects/live.amr"
    window._project_pending_path = "/projects/replacement.amr"
    window._project_previous_context = {
        "current_project_path": "/projects/live.amr",
        "requires_save_as": False,
        "legacy_project_path": None,
        "has_legacy_bindings": False,
    }
    window._project_load_active = True
    window._project_load_state = {"objects": [{"name": "first"}, {"name": "second"}]}
    window._project_staged_objects = [
        (object(), "/sources/first.ply", {"name": "first"}, object(), "captured_at_import"),
        (object(), "/sources/second.ply", {"name": "second"}, object(), "captured_at_import"),
    ]
    previous_undo_stack = [{"obj": old_object, "mesh_vertices": object()}]
    window.viewport.undo_stack = previous_undo_stack

    add_count = 0

    def add_then_fail(_mesh: object, *, name: str) -> None:
        nonlocal add_count
        add_count += 1
        if add_count == 1:
            assert name == "first"
            window.viewport.objects.append(new_object)
            window.viewport.selected_index = 0
            return
        raise RuntimeError("injected VBO/materialization failure")

    signals_were_blocked = window.viewport.blockSignals(True)
    try:
        with (
            patch.object(window.viewport, "add_mesh_object", side_effect=add_then_fail),
            patch.object(window, "_apply_loaded_object_state"),
            patch.object(window.scene_panel, "update_list"),
            patch.object(window, "sync_transform_panel"),
            patch.object(window, "_sync_tile_panel"),
            patch("app_interactive.QMessageBox.critical") as critical,
        ):
            window._finish_project_load()

        assert window.viewport.objects == [old_object]
        assert window.viewport.objects[0] is old_object
        assert window.viewport.selected_index == 0
        assert window.current_mesh is old_mesh
        assert window.current_filepath == "/sources/live.ply"
        assert window._artifact_session is native_session
        assert window._current_project_path == "/projects/live.amr"
        assert window._project_pending_path is None
        assert not window._project_load_failed
        assert window.viewport.undo_stack is previous_undo_stack
        old_cleanup.assert_not_called()
        new_cleanup.assert_called_once()
        critical.assert_called_once()
        assert "기존 scene을 복원" in critical.call_args.args[2]
    finally:
        window.viewport.blockSignals(signals_were_blocked)
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_project_vbo_failure_rolls_back_instead_of_committing_invisible_scene() -> None:
    QStandardPaths.setTestModeEnabled(True)
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    window = MainWindow()
    native_session = _artifact_session()
    window._artifact_session = native_session
    old_cleanup = Mock()
    first_cleanup = Mock()
    failed_cleanup = Mock()
    old_object = SimpleNamespace(cleanup=old_cleanup)
    first_new = SimpleNamespace(cleanup=first_cleanup, vertex_count=3, vbo_id=201)
    failed_new = SimpleNamespace(cleanup=failed_cleanup, vertex_count=0, vbo_id=None)
    old_mesh = object()

    window.viewport.objects = [old_object]
    window.viewport.selected_index = 0
    window.viewport.undo_stack = [{"obj": old_object}]
    window.current_mesh = old_mesh
    window.current_filepath = "/sources/live.ply"
    window._current_project_path = "/projects/live.amr"
    window._project_pending_path = "/projects/replacement.amr"
    window._project_previous_context = {
        "current_project_path": "/projects/live.amr",
        "requires_save_as": False,
        "legacy_project_path": None,
        "has_legacy_bindings": False,
    }
    window._project_load_active = True
    window._project_load_state = {"objects": [{"name": "first"}, {"name": "second"}]}
    window._project_staged_objects = [
        (object(), "/sources/first.ply", {"name": "first"}, object(), "captured_at_import"),
        (object(), "/sources/second.ply", {"name": "second"}, object(), "captured_at_import"),
    ]

    objects_to_add = iter((first_new, failed_new))

    def add_object(_mesh: object, *, name: str) -> None:
        assert name in {"first", "second"}
        window.viewport.objects.append(next(objects_to_add))
        window.viewport.selected_index = len(window.viewport.objects) - 1

    signals_were_blocked = window.viewport.blockSignals(True)
    try:
        with (
            patch.object(window.viewport, "add_mesh_object", side_effect=add_object),
            patch.object(window.viewport, "update_vbo", return_value=False) as update_vbo,
            patch.object(window, "_apply_loaded_object_state"),
            patch.object(window.scene_panel, "update_list"),
            patch.object(window, "sync_transform_panel"),
            patch.object(window, "_sync_tile_panel"),
            patch("app_interactive.QMessageBox.critical"),
        ):
            window._finish_project_load()

        update_vbo.assert_called_once_with(failed_new)
        assert window.viewport.objects == [old_object]
        assert window.current_mesh is old_mesh
        assert window.current_filepath == "/sources/live.ply"
        assert window._artifact_session is native_session
        assert window._current_project_path == "/projects/live.amr"
        assert not window._project_load_failed
        old_cleanup.assert_not_called()
        first_cleanup.assert_called_once()
        failed_cleanup.assert_called_once()
    finally:
        window.viewport.blockSignals(signals_were_blocked)
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_successful_project_swap_releases_old_scene_and_undo_history() -> None:
    QStandardPaths.setTestModeEnabled(True)
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    window = MainWindow()
    window._artifact_session = _artifact_session()
    old_cleanup = Mock()
    new_cleanup = Mock()
    old_object = SimpleNamespace(cleanup=old_cleanup)
    new_object = SimpleNamespace(cleanup=new_cleanup, vertex_count=3, vbo_id=301)
    staged_mesh = object()

    window.viewport.objects = [old_object]
    window.viewport.selected_index = 0
    window.viewport.undo_stack = [{"obj": old_object, "mesh_vertices": object()}]
    window.current_mesh = object()
    window.current_filepath = "/sources/live.ply"
    window._current_project_path = "/projects/live.amr"
    window._project_pending_path = "/projects/replacement.amr"
    window._project_previous_context = {
        "current_project_path": "/projects/live.amr",
        "requires_save_as": False,
        "legacy_project_path": None,
        "has_legacy_bindings": False,
    }
    window._project_load_active = True
    window._project_load_state = {"objects": [{"name": "replacement"}]}
    window._project_staged_objects = [
        (
            staged_mesh,
            "/sources/replacement.ply",
            {"name": "replacement"},
            object(),
            "captured_at_import",
        )
    ]

    def add_object(_mesh: object, *, name: str) -> None:
        assert _mesh is staged_mesh
        assert name == "replacement"
        window.viewport.objects.append(new_object)
        window.viewport.selected_index = 0

    signals_were_blocked = window.viewport.blockSignals(True)
    try:
        with (
            patch.object(window.viewport, "add_mesh_object", side_effect=add_object),
            patch.object(window, "_apply_loaded_object_state"),
            patch.object(window, "_apply_project_state"),
            patch.object(window.scene_panel, "update_list"),
            patch.object(window, "sync_transform_panel"),
            patch.object(window, "_sync_tile_panel"),
        ):
            window._finish_project_load()

        assert window.viewport.objects == [new_object]
        assert window.viewport.undo_stack == []
        assert window.current_mesh is staged_mesh
        assert window.current_filepath == "/sources/replacement.ply"
        assert window._artifact_session is None
        assert window._current_project_path == "/projects/replacement.amr"
        assert not window._project_load_failed
        old_cleanup.assert_called_once()
        new_cleanup.assert_not_called()
    finally:
        window.viewport.blockSignals(signals_were_blocked)
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()


def test_update_vbo_reports_failure_and_scene_cleanup_invalidates_buffer() -> None:
    QStandardPaths.setTestModeEnabled(True)
    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    mesh = SimpleNamespace(
        vertices=np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        faces=np.asarray([[0, 1, 2]], dtype=np.int32),
        n_vertices=3,
        normals=np.asarray([[0, 0, 1]] * 3, dtype=np.float32),
        face_normals=np.asarray([[0, 0, 1]], dtype=np.float32),
    )
    obj = SceneObject(mesh, "failure fixture")
    window = MainWindow()
    try:
        with (
            patch("src.gui.viewport_3d.glGenBuffers", return_value=0),
            patch("src.gui.viewport_3d.glBindBuffer"),
        ):
            assert window.viewport.update_vbo(obj) is False
        assert obj.vbo_id is None
        assert obj.vertex_count == 0

        obj.vbo_id = 77
        obj.vertex_count = 3
        with patch("src.gui.viewport_3d.glDeleteBuffers") as delete_buffers:
            obj.cleanup()
            obj.cleanup()
        delete_buffers.assert_called_once_with(1, [77])
        assert obj.vbo_id is None
        assert obj.vertex_count == 0
    finally:
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()
