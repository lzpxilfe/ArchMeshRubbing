"""Showing the scanner's colour, and taking it off again.

There is no GL context in a test run, so these exercise the decisions the
viewport makes around the driver - what gets uploaded, what is refused, what
is remembered - by calling the methods on a stand-in object, the way the
other viewport tests do.
"""

from __future__ import annotations

import os
from types import MethodType, SimpleNamespace

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QCoreApplication, QEvent  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from app_interactive import MainWindow  # noqa: E402
from src.gui.viewport_3d import Viewport3D  # noqa: E402


def _mesh(*, textured: bool = True, vertices: int = 3):
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    return SimpleNamespace(
        faces=faces,
        texture=np.zeros((4, 4, 3), dtype=np.uint8) if textured else None,
        uv_coords=np.zeros((vertices, 2), dtype=np.float64) if textured else None,
    )


def _object(*, textured: bool = True, visible: bool = True, vertex_count: int = 3):
    return SimpleNamespace(
        name="obj",
        mesh=_mesh(textured=textured),
        visible=visible,
        vertex_count=vertex_count,
        texture_id=None,
        uv_vbo_id=None,
        texture_upload_failed=False,
    )


def _viewport(objects, *, show_texture: bool = True):
    fake = SimpleNamespace(
        objects=list(objects),
        show_texture=show_texture,
        updated=0,
        released=[],
    )
    fake.update = lambda: setattr(fake, "updated", fake.updated + 1)
    fake.context = lambda: None
    fake.makeCurrent = lambda: None
    fake.doneCurrent = lambda: None
    fake._max_texture_side = lambda: 4096
    fake.set_show_texture = MethodType(Viewport3D.set_show_texture, fake)
    fake.textured_object_count = MethodType(Viewport3D.textured_object_count, fake)
    fake.upload_texture = MethodType(Viewport3D.upload_texture, fake)
    fake.release_texture = MethodType(Viewport3D.release_texture, fake)
    return fake


def test_turning_the_texture_off_gives_the_driver_its_memory_back() -> None:
    """A 4k image per mesh is real memory; switching to the silhouette view
    should not keep paying for pictures nobody is looking at."""

    first, second = _object(), _object()
    first.texture_id, first.uv_vbo_id = 7, 8
    viewport = _viewport([first, second])

    viewport.set_show_texture(False)

    assert viewport.show_texture is False
    assert first.texture_id is None and first.uv_vbo_id is None
    assert viewport.updated == 1


def test_turning_it_back_on_gives_a_failed_upload_one_more_try() -> None:
    """The archaeologist asking again is a reason to try again - otherwise a
    mesh that failed once is grey for the rest of the session with no way to
    say why it will not come back."""

    obj = _object()
    obj.texture_upload_failed = True
    viewport = _viewport([obj], show_texture=False)

    viewport.set_show_texture(True)

    assert viewport.show_texture is True
    assert obj.texture_upload_failed is False


def test_setting_what_is_already_set_does_not_redraw() -> None:
    viewport = _viewport([_object()])
    viewport.set_show_texture(True)
    assert viewport.updated == 0


def test_only_shown_meshes_with_both_halves_are_counted() -> None:
    """The count is what the status line reports, so it has to mean 'meshes
    that would actually show a pattern', not 'meshes that are loaded'."""

    viewport = _viewport(
        [
            _object(),
            _object(visible=False),
            _object(textured=False),
        ]
    )
    assert viewport.textured_object_count() == 1


def test_a_mesh_with_no_texture_is_not_a_failure() -> None:
    """Most meshes have no texture.  That is a fact about the mesh, not an
    error, and it must not put the object in the failed state that stops it
    ever being tried again."""

    obj = _object(textured=False)
    viewport = _viewport([obj])

    assert viewport.upload_texture(obj) is False
    assert obj.texture_upload_failed is False


def test_nothing_is_uploaded_while_the_texture_is_off() -> None:
    obj = _object()
    viewport = _viewport([obj], show_texture=False)
    assert viewport.upload_texture(obj) is False
    assert obj.texture_upload_failed is False


def test_an_already_uploaded_object_is_not_uploaded_again() -> None:
    obj = _object()
    obj.texture_id, obj.uv_vbo_id = 3, 4
    viewport = _viewport([obj])
    assert viewport.upload_texture(obj) is True


def test_uvs_that_do_not_match_the_geometry_are_refused_not_drawn() -> None:
    """The geometry buffer and the uv buffer are read corner for corner.  A
    uv buffer of a different length paints the pattern onto whichever
    triangles happen to line up - a wrong picture with no error, which is
    the one outcome a measured drawing cannot afford."""

    obj = _object(vertex_count=99)  # geometry says 99 corners, the mesh has 3
    viewport = _viewport([obj])

    assert viewport.upload_texture(obj) is False
    assert obj.texture_upload_failed is True
    # And it is not retried every frame after that.
    assert viewport.upload_texture(obj) is False


def test_the_window_offers_the_texture_as_a_way_of_looking() -> None:
    """It sits with the backdrop and the shadow: a viewing choice, checkable,
    and nothing it does reaches a record or a drawing."""

    app = QApplication.instance() or QApplication([])
    assert app is not None
    window = MainWindow()
    try:
        action = window.action_show_texture
        assert action.isCheckable()
        assert action.isChecked() is True
        assert window.viewport.show_texture is True

        action.setChecked(False)
        assert window.viewport.show_texture is False
        assert "껐습니다" in window.status_info.text()

        action.setChecked(True)
        assert window.viewport.show_texture is True
        # Nothing is open, so the window says so rather than appearing to work.
        assert "텍스처와 UV" in window.status_info.text()
    finally:
        # Never close(): MainWindow.closeEvent asks the user to confirm, and
        # an offscreen runner has nobody to answer it.
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()
