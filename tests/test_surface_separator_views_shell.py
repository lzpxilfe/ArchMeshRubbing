import unittest

import numpy as np

from src.core.mesh_loader import MeshData
from src.core.surface_separator import SurfaceSeparator


def _rot_matrix_xyz(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])
    cx, sx = float(np.cos(rx)), float(np.sin(rx))
    cy, sy = float(np.cos(ry)), float(np.sin(ry))
    cz, sz = float(np.cos(rz)), float(np.sin(rz))
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rot_x @ rot_y @ rot_z


def _make_wrinkled_shell(
    *,
    nx: int = 28,
    ny: int = 18,
    thickness: float = 1.0,
    wrinkle_amp: float = 1.1,
    wrinkle_sigma: float = 0.25,
    rotate_xyz_deg: tuple[float, float, float] = (27.0, -31.0, 19.0),
) -> tuple[MeshData, set[int], set[int], set[int], np.ndarray]:
    xs = np.linspace(-1.0, 1.0, int(nx) + 1, dtype=np.float64)
    ys = np.linspace(-0.7, 0.7, int(ny) + 1, dtype=np.float64)
    xg, yg = np.meshgrid(xs, ys, indexing="xy")

    outer_z = np.full_like(xg, 0.5 * float(thickness))
    rr2 = (xg / float(wrinkle_sigma)) ** 2 + (yg / float(wrinkle_sigma)) ** 2
    bump = float(wrinkle_amp) * np.exp(-0.5 * rr2)
    inner_z = -0.5 * float(thickness) + bump

    outer = np.stack([xg, yg, outer_z], axis=-1).reshape(-1, 3)
    inner = np.stack([xg, yg, inner_z], axis=-1).reshape(-1, 3)

    v_off = int(outer.shape[0])
    vertices = np.vstack([outer, inner]).astype(np.float64, copy=False)

    def out_idx(ix: int, iy: int) -> int:
        return int(iy) * (int(nx) + 1) + int(ix)

    def in_idx(ix: int, iy: int) -> int:
        return v_off + out_idx(ix, iy)

    faces: list[list[int]] = []

    # Outer sheet
    for iy in range(int(ny)):
        for ix in range(int(nx)):
            a = out_idx(ix, iy)
            b = out_idx(ix + 1, iy)
            c = out_idx(ix + 1, iy + 1)
            d = out_idx(ix, iy + 1)
            faces.append([a, b, c])
            faces.append([a, c, d])
    outer_face_count = len(faces)

    # Inner sheet
    for iy in range(int(ny)):
        for ix in range(int(nx)):
            a = in_idx(ix, iy)
            b = in_idx(ix, iy + 1)
            c = in_idx(ix + 1, iy + 1)
            d = in_idx(ix + 1, iy)
            faces.append([a, b, c])
            faces.append([a, c, d])
    inner_face_count = len(faces) - outer_face_count

    # Side walls (migu-like)
    for ix in range(int(nx)):
        # y min
        o0 = out_idx(ix, 0)
        o1 = out_idx(ix + 1, 0)
        i0 = in_idx(ix, 0)
        i1 = in_idx(ix + 1, 0)
        faces.append([o0, i1, o1])
        faces.append([o0, i0, i1])
        # y max
        o0 = out_idx(ix, int(ny))
        o1 = out_idx(ix + 1, int(ny))
        i0 = in_idx(ix, int(ny))
        i1 = in_idx(ix + 1, int(ny))
        faces.append([o0, o1, i1])
        faces.append([o0, i1, i0])

    for iy in range(int(ny)):
        # x min
        o0 = out_idx(0, iy)
        o1 = out_idx(0, iy + 1)
        i0 = in_idx(0, iy)
        i1 = in_idx(0, iy + 1)
        faces.append([o0, o1, i1])
        faces.append([o0, i1, i0])
        # x max
        o0 = out_idx(int(nx), iy)
        o1 = out_idx(int(nx), iy + 1)
        i0 = in_idx(int(nx), iy)
        i1 = in_idx(int(nx), iy + 1)
        faces.append([o0, i1, o1])
        faces.append([o0, i0, i1])

    faces_arr = np.asarray(faces, dtype=np.int32)

    rot = _rot_matrix_xyz(*rotate_xyz_deg)
    vertices = (rot @ vertices.T).T

    mesh = MeshData(vertices=vertices, faces=faces_arr, unit="mm")
    true_outer = set(range(0, outer_face_count))
    true_inner = set(range(outer_face_count, outer_face_count + inner_face_count))
    true_migu = set(range(outer_face_count + inner_face_count, int(faces_arr.shape[0])))
    true_axis = rot @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    true_axis = true_axis / (float(np.linalg.norm(true_axis)) + 1e-12)
    return mesh, true_outer, true_inner, true_migu, true_axis


class TestSurfaceSeparatorViewsShell(unittest.TestCase):
    def test_reference_direction_tracks_shell_normal(self):
        mesh, _true_outer, _true_inner, _true_migu, true_axis = _make_wrinkled_shell()

        sep = SurfaceSeparator()
        ref = np.asarray(sep._estimate_reference_direction(mesh), dtype=np.float64).reshape(3)
        ref = ref / (float(np.linalg.norm(ref)) + 1e-12)

        self.assertGreaterEqual(abs(float(np.dot(ref, true_axis))), 0.90)

    def test_views_separates_wrinkled_inner_outer(self):
        mesh, true_outer, true_inner, _true_migu, _true_axis = _make_wrinkled_shell()

        sep = SurfaceSeparator()
        res = sep.auto_detect_surfaces(mesh, method="views", return_submeshes=False)

        pred_outer = set(map(int, res.outer_face_indices))
        pred_inner = set(map(int, res.inner_face_indices))

        self.assertTrue(pred_outer.isdisjoint(pred_inner))
        outer_recall = len(pred_outer & true_outer) / max(1, len(true_outer))
        inner_recall = len(pred_inner & true_inner) / max(1, len(true_inner))
        self.assertGreaterEqual(outer_recall, 0.95)
        self.assertGreaterEqual(inner_recall, 0.95)


if __name__ == "__main__":
    unittest.main()
