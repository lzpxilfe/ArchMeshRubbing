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


def _make_thin_cylinder_patch(
    *,
    r_outer: float = 50.0,
    thickness: float = 5.0,
    length: float = 200.0,
    theta0: float = -0.5 * np.pi,
    theta1: float = 0.5 * np.pi,
    n_theta: int = 24,
    n_len: int = 10,
    rotate_xyz_deg: tuple[float, float, float] = (25.0, -40.0, 15.0),
    translate: tuple[float, float, float] = (13.0, -7.0, 5.0),
) -> tuple[MeshData, set[int], set[int], set[int]]:
    r_inner = float(r_outer) - float(thickness)
    if r_inner <= 0.0:
        raise ValueError("thickness too large")

    thetas = np.linspace(float(theta0), float(theta1), int(n_theta) + 1, dtype=np.float64)
    ys = np.linspace(0.0, float(length), int(n_len) + 1, dtype=np.float64)

    outer = []
    inner = []
    for y in ys:
        for th in thetas:
            outer.append([r_outer * float(np.cos(th)), float(y), r_outer * float(np.sin(th))])
            inner.append([r_inner * float(np.cos(th)), float(y), r_inner * float(np.sin(th))])
    outer = np.asarray(outer, dtype=np.float64)
    inner = np.asarray(inner, dtype=np.float64)

    v_off = int(outer.shape[0])
    vertices = np.vstack([outer, inner])

    def out_idx(iy: int, it: int) -> int:
        return int(iy) * (int(n_theta) + 1) + int(it)

    def in_idx(iy: int, it: int) -> int:
        return v_off + out_idx(iy, it)

    faces: list[list[int]] = []

    # Outer surface
    for iy in range(int(n_len)):
        for it in range(int(n_theta)):
            a = out_idx(iy, it)
            b = out_idx(iy, it + 1)
            c = out_idx(iy + 1, it + 1)
            d = out_idx(iy + 1, it)
            faces.append([a, b, c])
            faces.append([a, c, d])

    outer_face_count = len(faces)

    # Inner surface (orientation doesn't matter for radius-based separation)
    for iy in range(int(n_len)):
        for it in range(int(n_theta)):
            a = in_idx(iy, it)
            b = in_idx(iy + 1, it)
            c = in_idx(iy + 1, it + 1)
            d = in_idx(iy, it + 1)
            faces.append([a, b, c])
            faces.append([a, c, d])

    inner_face_count = len(faces) - outer_face_count

    # Side walls at theta endpoints (migu-like thickness faces)
    for iy in range(int(n_len)):
        # theta = theta0 (it=0)
        o0 = out_idx(iy, 0)
        o1 = out_idx(iy + 1, 0)
        i0 = in_idx(iy, 0)
        i1 = in_idx(iy + 1, 0)
        faces.append([o0, o1, i1])
        faces.append([o0, i1, i0])

        # theta = theta1 (it=n_theta)
        o0 = out_idx(iy, int(n_theta))
        o1 = out_idx(iy + 1, int(n_theta))
        i0 = in_idx(iy, int(n_theta))
        i1 = in_idx(iy + 1, int(n_theta))
        faces.append([o0, i1, o1])
        faces.append([o0, i0, i1])

    faces_arr = np.asarray(faces, dtype=np.int32)

    # Apply a fixed rotation+translation so the separator can't rely on world axes.
    rot = _rot_matrix_xyz(*rotate_xyz_deg)
    vertices = (rot @ vertices.T).T + np.asarray(translate, dtype=np.float64).reshape(1, 3)

    mesh = MeshData(vertices=vertices, faces=faces_arr, unit="mm")

    true_outer = set(range(0, outer_face_count))
    true_inner = set(range(outer_face_count, outer_face_count + inner_face_count))
    true_migu = set(range(outer_face_count + inner_face_count, int(faces_arr.shape[0])))
    return mesh, true_outer, true_inner, true_migu


class TestSurfaceSeparatorCylinder(unittest.TestCase):
    def test_cylinder_method_separates_outer_inner_migu(self):
        mesh, true_outer, true_inner, true_migu = _make_thin_cylinder_patch()

        sep = SurfaceSeparator()
        res = sep.auto_detect_surfaces(mesh, method="cylinder", return_submeshes=False)

        self.assertTrue(bool(res.meta.get("cylinder_ok", False)))

        pred_outer = set(map(int, res.outer_face_indices))
        pred_inner = set(map(int, res.inner_face_indices))
        pred_migu = set(map(int, res.migu_face_indices)) if res.migu_face_indices is not None else set()

        self.assertTrue(pred_outer.isdisjoint(pred_inner))
        self.assertTrue(pred_outer.isdisjoint(pred_migu))
        self.assertTrue(pred_inner.isdisjoint(pred_migu))

        outer_recall = len(pred_outer & true_outer) / max(1, len(true_outer))
        inner_recall = len(pred_inner & true_inner) / max(1, len(true_inner))
        migu_recall = len(pred_migu & true_migu) / max(1, len(true_migu))

        self.assertGreaterEqual(outer_recall, 0.95)
        self.assertGreaterEqual(inner_recall, 0.95)
        self.assertGreaterEqual(migu_recall, 0.80)

    def test_auto_prefers_cylinder_when_valid(self):
        mesh, true_outer, true_inner, _true_migu = _make_thin_cylinder_patch()

        sep = SurfaceSeparator()
        res = sep.auto_detect_surfaces(mesh, method="auto", return_submeshes=False)

        self.assertEqual(str(res.meta.get("method")), "cylinder_radius")
        self.assertTrue(bool(res.meta.get("cylinder_ok", False)))

        pred_outer = set(map(int, res.outer_face_indices))
        pred_inner = set(map(int, res.inner_face_indices))

        outer_recall = len(pred_outer & true_outer) / max(1, len(true_outer))
        inner_recall = len(pred_inner & true_inner) / max(1, len(true_inner))
        self.assertGreaterEqual(outer_recall, 0.90)
        self.assertGreaterEqual(inner_recall, 0.90)

