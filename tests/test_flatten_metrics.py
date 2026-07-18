from __future__ import annotations

import numpy as np

from src.core.flatten_metrics import compute_face_distortion
from src.core.mesh_loader import MeshData


def test_distortion_detects_opposite_edge_shear() -> None:
    height = np.sqrt(3.0) * 0.5
    mesh = MeshData(
        vertices=np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, height, 0.0],
            ],
            dtype=np.float64,
        ),
        faces=np.asarray([[0, 1, 2]], dtype=np.int32),
        unit="mm",
    )
    uv = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [-0.5, height],
        ],
        dtype=np.float64,
    )

    distortion = compute_face_distortion(mesh, uv)

    assert distortion.shape == (1,)
    assert float(distortion[0]) > 0.4


def test_distortion_is_zero_for_rigid_planar_embedding() -> None:
    vertices = np.asarray(
        [
            [10.0, 20.0, 3.0],
            [12.0, 20.0, 3.0],
            [10.5, 21.5, 3.0],
        ],
        dtype=np.float64,
    )
    mesh = MeshData(
        vertices=vertices,
        faces=np.asarray([[0, 1, 2]], dtype=np.int32),
        unit="mm",
    )
    uv = vertices[:, :2] - vertices[0, :2]

    distortion = compute_face_distortion(mesh, uv)

    np.testing.assert_allclose(distortion, np.zeros((1,)), atol=1e-12, rtol=0.0)
