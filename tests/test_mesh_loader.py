from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from src.core.mesh_loader import MeshData


def _repeated_triangle_mesh(face_count: int) -> MeshData:
    return MeshData(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        faces=np.tile(
            np.array([[0, 1, 2]], dtype=np.int32),
            (face_count, 1),
        ),
        unit="mm",
    )


def test_large_mesh_surface_area_uses_every_face_deterministically() -> None:
    mesh = _repeated_triangle_mesh(1_000_001)

    with mock.patch(
        "src.core.mesh_loader.np.random.choice",
        side_effect=AssertionError("surface area must not sample randomly"),
    ):
        first = mesh.surface_area

    assert first == pytest.approx(500_000.5, rel=0.0, abs=1e-9)
    assert mesh.surface_area == first
    assert mesh.surface_area_status == "exact"


def test_surface_area_chunking_has_a_fixed_small_working_set(monkeypatch) -> None:
    mesh = _repeated_triangle_mesh(11)
    observed_chunk_sizes: list[int] = []
    real_cross = np.cross

    def recording_cross(left, right):
        observed_chunk_sizes.append(int(left.shape[0]))
        return real_cross(left, right)

    monkeypatch.setattr("src.core.mesh_loader._SURFACE_AREA_CHUNK_FACES", 4)
    monkeypatch.setattr("src.core.mesh_loader.np.cross", recording_cross)

    assert mesh.surface_area == pytest.approx(5.5)
    assert observed_chunk_sizes == [4, 4, 3]
    assert mesh.surface_area_status == "exact"


def test_surface_area_reports_unavailable_after_memory_error(monkeypatch) -> None:
    mesh = _repeated_triangle_mesh(1)

    def out_of_memory(*_args, **_kwargs):
        raise MemoryError

    monkeypatch.setattr("src.core.mesh_loader.np.cross", out_of_memory)

    assert mesh.surface_area == -1.0
    assert mesh.surface_area_status == "unavailable"


def test_empty_mesh_has_exact_zero_surface_area() -> None:
    mesh = MeshData(
        vertices=np.zeros((0, 3), dtype=np.float64),
        faces=np.zeros((0, 3), dtype=np.int32),
        unit="mm",
    )

    assert mesh.surface_area == 0.0
    assert mesh.surface_area_status == "exact"
