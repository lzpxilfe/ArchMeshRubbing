"""General-purpose flattening engines: ARAP, LSCM, and Tutte helpers."""

from __future__ import annotations

import logging

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu, spsolve

from .flatten_metrics import compute_face_distortion
from .flatten_types import FlattenedMesh
from .flatten_utils import (
    _log_ignored_exception,
    _smooth_uv_laplacian,
    extract_submesh_with_mapping,
    sanitize_mesh,
)
from .logging_utils import log_once
from .mesh_loader import MeshData

_LOGGER = logging.getLogger(__name__)


class ARAPFlattener:
    """ARAP-based mesh flattening."""

    def __init__(self, max_iterations: int = 50, tolerance: float = 1e-6):
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def flatten(
        self,
        mesh: MeshData,
        boundary_type: str = "free",
        initial_method: str = "lscm",
        *,
        initial_uv: np.ndarray | None = None,
        smooth_iters: int = 0,
        smooth_strength: float = 0.15,
        pack_compact: bool = True,
    ) -> FlattenedMesh:
        if mesh is None:
            raise ValueError("mesh is None")

        mesh = self._sanitize_mesh(mesh)
        if mesh.n_vertices == 0 or mesh.n_faces == 0:
            return FlattenedMesh(
                uv=np.zeros((mesh.n_vertices, 2), dtype=np.float64),
                faces=mesh.faces,
                original_mesh=mesh,
                distortion_per_face=np.zeros((mesh.n_faces,), dtype=np.float64),
                scale=1.0,
            )

        components = self._face_connected_components(mesh.faces)
        if len(components) <= 1:
            if initial_uv is None:
                initial_uv = self._safe_initial_parameterization(mesh, initial_method)
            else:
                initial_uv = np.asarray(initial_uv, dtype=np.float64)

            optimized_uv = self._arap_optimize(mesh, initial_uv, boundary_type)
            optimized_uv = self._orient_uv_pca(optimized_uv)

            if int(smooth_iters) > 0 and float(smooth_strength) > 0:
                try:
                    edge_i, edge_j, edge_w = self._compute_cotangent_edge_weights(mesh)
                    anchors = self._pick_anchor_pair(mesh, np.arange(mesh.n_vertices, dtype=np.int32))
                    optimized_uv = self._smooth_uv_laplacian(
                        optimized_uv,
                        edge_i,
                        edge_j,
                        edge_w,
                        iterations=int(smooth_iters),
                        strength=float(smooth_strength),
                        fixed_indices=anchors,
                    )
                except Exception:
                    _log_ignored_exception()

            distortion = self._compute_distortion(mesh, optimized_uv)
            return FlattenedMesh(
                uv=optimized_uv,
                faces=mesh.faces,
                original_mesh=mesh,
                distortion_per_face=distortion,
                scale=1.0,
            )

        components = sorted(components, key=lambda a: int(a.size), reverse=True)
        uv_all = np.zeros((mesh.n_vertices, 2), dtype=np.float64)
        gap = 1.0
        try:
            ext = np.asarray(mesh.extents, dtype=np.float64).reshape(-1)
            if ext.size >= 1 and np.isfinite(ext).all():
                gap = float(max(0.5, 0.005 * float(np.max(ext))))
        except Exception:
            gap = 1.0

        packed: list[tuple[np.ndarray, np.ndarray, float, float]] = []
        for face_indices in components:
            comp_mesh, vmap = extract_submesh_with_mapping(mesh, face_indices)
            if comp_mesh.n_vertices == 0 or comp_mesh.n_faces == 0:
                continue

            if initial_uv is None:
                comp_initial = self._safe_initial_parameterization(comp_mesh, initial_method)
            else:
                try:
                    comp_initial = np.asarray(initial_uv, dtype=np.float64)[vmap]
                except Exception:
                    comp_initial = self._safe_initial_parameterization(comp_mesh, initial_method)
            comp_uv = self._arap_optimize(comp_mesh, comp_initial, boundary_type)
            comp_uv = np.asarray(comp_uv, dtype=np.float64)
            if comp_uv.ndim != 2 or comp_uv.shape[0] == 0 or comp_uv.shape[1] < 2:
                continue
            comp_uv = comp_uv[:, :2].copy()
            comp_uv = self._orient_uv_pca(comp_uv)

            if int(smooth_iters) > 0 and float(smooth_strength) > 0:
                try:
                    edge_i, edge_j, edge_w = self._compute_cotangent_edge_weights(comp_mesh)
                    anchors = self._pick_anchor_pair(comp_mesh, np.arange(comp_mesh.n_vertices, dtype=np.int32))
                    comp_uv = self._smooth_uv_laplacian(
                        comp_uv,
                        edge_i,
                        edge_j,
                        edge_w,
                        iterations=int(smooth_iters),
                        strength=float(smooth_strength),
                        fixed_indices=anchors,
                    )
                except Exception:
                    _log_ignored_exception()

            finite = np.all(np.isfinite(comp_uv), axis=1)
            if np.any(finite):
                min_uv = comp_uv[finite].min(axis=0)
                max_uv = comp_uv[finite].max(axis=0)
            else:
                min_uv = np.zeros((2,), dtype=np.float64)
                max_uv = np.zeros((2,), dtype=np.float64)
            comp_uv[~finite] = 0.0
            comp_uv -= min_uv
            width = float(max_uv[0] - min_uv[0]) if np.any(finite) else 0.0
            height = float(max_uv[1] - min_uv[1]) if np.any(finite) else 0.0
            packed.append((comp_uv, vmap, width, height))

        if packed:
            packed.sort(key=lambda x: float(x[2] * x[3]), reverse=True)
            max_w = max(p[2] for p in packed)
            total_area = sum(float(p[2] * p[3]) for p in packed)
            row_limit = max(max_w, float(np.sqrt(total_area)) * (1.25 if pack_compact else 2.0))
            cursor_x = 0.0
            cursor_y = 0.0
            row_h = 0.0
            for comp_uv, vmap, width, height in packed:
                if cursor_x > 0.0 and (cursor_x + width) > row_limit:
                    cursor_x = 0.0
                    cursor_y += row_h + gap
                    row_h = 0.0
                comp_uv[:, 0] += cursor_x
                comp_uv[:, 1] += cursor_y
                uv_all[vmap] = comp_uv
                cursor_x += width + gap
                row_h = max(row_h, height)

        distortion = self._compute_distortion(mesh, uv_all)
        return FlattenedMesh(
            uv=uv_all,
            faces=mesh.faces,
            original_mesh=mesh,
            distortion_per_face=distortion,
            scale=1.0,
        )

    def _sanitize_mesh(self, mesh: MeshData) -> MeshData:
        return sanitize_mesh(mesh)

    def _safe_initial_parameterization(self, mesh: MeshData, initial_method: str) -> np.ndarray:
        method = str(initial_method or "lscm").lower().strip()

        try:
            max_verts = int(getattr(self, "_initial_param_max_vertices", 200000))
        except Exception:
            max_verts = 200000
        try:
            max_faces = int(getattr(self, "_initial_param_max_faces", 400000))
        except Exception:
            max_faces = 400000
        n_verts = int(getattr(mesh, "n_vertices", 0) or 0)
        n_faces = int(getattr(mesh, "n_faces", 0) or 0)
        if n_verts > max_verts or n_faces > max_faces:
            log_once(
                _LOGGER,
                "flattener:initial_param_fallback_large",
                logging.INFO,
                "Initial UV fallback to PCA projection (verts=%d, faces=%d)",
                n_verts,
                n_faces,
            )
            basis = self._compute_reference_basis(mesh)
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            return (basis @ vertices.T).T

        try:
            if method == "tutte":
                return self._tutte_parameterization(mesh)
            if method == "lscm":
                return self._lscm_parameterization(mesh)
        except Exception:
            _LOGGER.debug("Initial UV parameterization failed (method=%s)", method, exc_info=True)

        try:
            return self._tutte_parameterization(mesh)
        except Exception:
            _LOGGER.debug("Fallback Tutte parameterization failed", exc_info=True)
        try:
            return self._lscm_parameterization(mesh)
        except Exception:
            _LOGGER.debug("Fallback LSCM parameterization failed", exc_info=True)

        basis = self._compute_reference_basis(mesh)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        return np.asarray((basis @ vertices.T).T, dtype=np.float64)

    def _orient_uv_pca(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, dtype=np.float64)
        if uv.ndim != 2 or uv.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.float64)

        out = uv[:, :2].copy()
        finite = np.all(np.isfinite(out), axis=1)
        if np.count_nonzero(finite) < 2:
            out[~finite] = 0.0
            return out

        pts = out[finite]
        mean = pts.mean(axis=0)
        centered = pts - mean
        cov = centered.T @ centered
        if cov.shape == (2, 2) and centered.shape[0] > 0:
            cov = cov / float(centered.shape[0])

        try:
            evals, evecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            out[~finite] = 0.0
            return out

        order = np.argsort(evals)[::-1]
        axes = evecs[:, order]
        for k in range(2):
            axis = axes[:, k]
            idx = int(np.argmax(np.abs(axis)))
            if axis[idx] < 0:
                axes[:, k] *= -1
        if float(np.linalg.det(axes)) < 0:
            axes[:, 1] *= -1

        out[finite] = (pts - mean) @ axes
        out[~finite] = 0.0
        return out

    def _face_connected_components(self, faces: np.ndarray) -> list[np.ndarray]:
        faces = np.asarray(faces, dtype=np.int32)
        if faces.ndim != 2 or faces.shape[0] == 0:
            return []

        faces = faces[:, :3].astype(np.int32, copy=False)
        m = int(faces.shape[0])

        try:
            max_faces = int(getattr(self, "_component_split_max_faces", 400000))
        except Exception:
            max_faces = 400000
        if m > max_faces:
            log_once(
                _LOGGER,
                "flattener:skip_face_components",
                logging.INFO,
                "Skipping face connected-components split (faces=%d > %d)",
                m,
                int(max_faces),
            )
            return [np.arange(m, dtype=np.int32)]

        parent = np.arange(m, dtype=np.int32)
        rank = np.zeros(m, dtype=np.int8)

        def find(x: int) -> int:
            x = int(x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = int(parent[x])
            return x

        def union(a_idx: int, b_idx: int) -> None:
            ra = find(a_idx)
            rb = find(b_idx)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] = np.int8(rank[ra] + 1)

        edge_to_face: dict[tuple[int, int], int] = {}
        for fi, face in enumerate(faces):
            a, b, c = int(face[0]), int(face[1]), int(face[2])
            for u, v in ((a, b), (b, c), (c, a)):
                if u > v:
                    u, v = v, u
                key = (u, v)
                prev = edge_to_face.get(key)
                if prev is None:
                    edge_to_face[key] = int(fi)
                else:
                    union(int(fi), int(prev))

        groups: dict[int, list[int]] = {}
        for fi in range(m):
            root = find(fi)
            groups.setdefault(root, []).append(fi)
        return [np.asarray(g, dtype=np.int32) for g in groups.values()]

    def _lscm_parameterization(self, mesh: MeshData) -> np.ndarray:
        n = mesh.n_vertices
        m = mesh.n_faces
        vertices = mesh.vertices
        faces = mesh.faces

        boundary = np.asarray(mesh.get_boundary_vertices(), dtype=np.int32).reshape(-1)
        candidates = boundary if boundary.size >= 2 else np.arange(n, dtype=np.int32)
        fixed_verts = self._pick_anchor_pair(mesh, candidates)
        dist = float(np.linalg.norm(vertices[fixed_verts[1]] - vertices[fixed_verts[0]]))
        if not np.isfinite(dist) or dist < 1e-9:
            dist = 1.0
        fixed_pos = np.array([[0.0, 0.0], [dist, 0.0]], dtype=np.float64)

        rows = []
        cols = []
        vals_real = []

        for fi, face in enumerate(faces):
            v0, v1, v2 = vertices[face]
            e1 = v1 - v0
            e2 = v2 - v0
            len_e1 = np.linalg.norm(e1)
            if len_e1 < 1e-10:
                continue

            x1 = len_e1
            y1 = 0.0
            x2 = np.dot(e2, e1) / len_e1
            e1_normalized = e1 / len_e1
            perp = e2 - x2 * e1_normalized
            y2 = np.linalg.norm(perp)
            area = 0.5 * x1 * y2
            if area < 1e-10:
                continue

            sqrt_area = np.sqrt(area)
            z0 = complex(0, 0)
            z1 = complex(x1, y1)
            z2 = complex(x2, y2)
            w0 = (z2 - z1) / (2 * sqrt_area)
            w1 = (z0 - z2) / (2 * sqrt_area)
            w2 = (z1 - z0) / (2 * sqrt_area)

            for vi, w in zip(face, [w0, w1, w2]):
                rows.extend([2 * fi, 2 * fi, 2 * fi + 1, 2 * fi + 1])
                cols.extend([2 * vi, 2 * vi + 1, 2 * vi, 2 * vi + 1])
                vals_real.extend([w.real, -w.imag, w.imag, w.real])

        A = sparse.coo_matrix((vals_real, (rows, cols)), shape=(2 * m, 2 * n)).tocsr()
        free_mask = np.ones(n, dtype=bool)
        free_mask[fixed_verts] = False
        free_indices = np.where(free_mask)[0]

        free_cols = []
        for idx in free_indices:
            free_cols.extend([2 * idx, 2 * idx + 1])
        A_free = A[:, free_cols]

        fixed_cols = []
        for idx in fixed_verts:
            fixed_cols.extend([2 * idx, 2 * idx + 1])
        A_fixed = A[:, fixed_cols]
        b_fixed = np.zeros(len(fixed_verts) * 2)
        for i, (_idx, pos) in enumerate(zip(fixed_verts, fixed_pos)):
            b_fixed[2 * i] = pos[0]
            b_fixed[2 * i + 1] = pos[1]

        b = -A_fixed @ b_fixed
        AtA = A_free.T @ A_free
        Atb = A_free.T @ b
        AtA += sparse.eye(AtA.shape[0]) * 1e-8
        x_free = spsolve(AtA.tocsr(), Atb)

        uv = np.zeros((n, 2))
        for i, idx in enumerate(fixed_verts):
            uv[idx] = fixed_pos[i]
        for i, idx in enumerate(free_indices):
            uv[idx, 0] = x_free[2 * i]
            uv[idx, 1] = x_free[2 * i + 1]
        return uv

    def _tutte_parameterization(self, mesh: MeshData) -> np.ndarray:
        n = mesh.n_vertices
        loops = mesh.get_boundary_loops()
        boundary = loops[0] if loops else mesh.get_boundary_vertices()
        if loops:
            boundary = max(loops, key=lambda a: int(a.size))
        if len(boundary) < 3:
            return self._lscm_parameterization(mesh)

        verts = np.asarray(mesh.vertices, dtype=np.float64)
        b_pts = verts[boundary]
        if b_pts.shape[0] >= 2:
            diffs = b_pts[(np.arange(len(boundary)) + 1) % len(boundary)] - b_pts
            perim = float(np.linalg.norm(diffs, axis=1).sum())
        else:
            perim = 0.0
        radius = perim / (2.0 * np.pi) if perim > 1e-12 else 1.0

        angles = np.linspace(0, 2 * np.pi, len(boundary), endpoint=False)
        boundary_uv = np.column_stack([np.cos(angles), np.sin(angles)]) * radius

        is_boundary = np.zeros(n, dtype=bool)
        is_boundary[boundary] = True
        adjacency = [set() for _ in range(n)]
        for face in mesh.faces:
            for i in range(3):
                adjacency[face[i]].add(face[(i + 1) % 3])
                adjacency[face[i]].add(face[(i + 2) % 3])

        rows, cols, vals = [], [], []
        b = np.zeros((n, 2))
        for i in range(n):
            if is_boundary[i]:
                rows.append(i)
                cols.append(i)
                vals.append(1.0)
                idx = np.where(boundary == i)[0][0]
                b[i] = boundary_uv[idx]
            else:
                neighbors = list(adjacency[i])
                rows.append(i)
                cols.append(i)
                vals.append(1.0)
                if neighbors:
                    for j in neighbors:
                        rows.append(i)
                        cols.append(j)
                        vals.append(-1.0 / len(neighbors))

        L = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        uv = np.zeros((n, 2))
        uv[:, 0] = np.asarray(spsolve(L, b[:, 0])).ravel()
        uv[:, 1] = np.asarray(spsolve(L, b[:, 1])).ravel()
        return uv

    def _arap_optimize(self, mesh: MeshData, initial_uv: np.ndarray, boundary_type: str) -> np.ndarray:
        n = mesh.n_vertices
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        if n <= 0 or vertices.ndim != 2 or vertices.shape[0] == 0 or faces.ndim != 2 or faces.shape[0] == 0:
            return np.zeros((n, 2), dtype=np.float64)

        uv = np.asarray(initial_uv, dtype=np.float64)
        if uv.ndim != 2 or uv.shape[0] != n or uv.shape[1] < 2:
            uv = np.zeros((n, 2), dtype=np.float64)
        else:
            uv = uv[:, :2].copy()

        reference_basis = self._compute_reference_basis(mesh)
        try:
            max_verts = int(getattr(self, "_arap_max_vertices", 150000))
        except Exception:
            max_verts = 150000
        try:
            max_faces = int(getattr(self, "_arap_max_faces", 300000))
        except Exception:
            max_faces = 300000
        if n > max_verts or int(faces.shape[0]) > max_faces:
            log_once(
                _LOGGER,
                "flattener:arap_skip_large",
                logging.INFO,
                "Skipping ARAP optimize (verts=%d, faces=%d)",
                int(n),
                int(faces.shape[0]),
            )
            return uv

        edge_i, edge_j, edge_w = self._compute_cotangent_edge_weights(mesh)
        if edge_i.size == 0:
            return (reference_basis @ vertices.T).T

        edge_vec_3d = vertices[edge_j] - vertices[edge_i]
        bt = str(boundary_type or "free").lower().strip()
        if bt in {"fixed_circle", "fixed_rect"}:
            boundary = np.asarray(mesh.get_boundary_vertices(), dtype=np.int32).reshape(-1)
        else:
            boundary = np.zeros((0,), dtype=np.int32)
        fixed_indices, fixed_uv, uv = self._build_fixed_constraints(
            mesh,
            uv,
            boundary_type=bt,
            boundary=boundary,
        )
        fixed_indices = np.asarray(fixed_indices, dtype=np.int32).reshape(-1)
        fixed_uv = np.asarray(fixed_uv, dtype=np.float64).reshape(-1, 2)
        if fixed_indices.size > 0:
            uv[fixed_indices] = fixed_uv

        L = self._build_laplacian(n, edge_i, edge_j, edge_w).tocsc()
        all_idx = np.arange(n, dtype=np.int32)
        fixed_mask = np.zeros(n, dtype=bool)
        if fixed_indices.size > 0:
            fixed_mask[fixed_indices] = True
        free_indices = all_idx[~fixed_mask]
        if free_indices.size == 0:
            return uv

        L_ff = L[free_indices][:, free_indices].tocsc()
        reg = 1e-8
        L_ff_reg = L_ff + sparse.eye(L_ff.shape[0], format="csc") * reg
        try:
            lu = splu(L_ff_reg)
            use_lu = True
        except Exception:
            lu = None
            use_lu = False
            L_ff_reg = L_ff_reg.tocsr()

        L_fc = None
        if fixed_indices.size > 0:
            try:
                L_fc = L[free_indices][:, fixed_indices].tocsc()
            except Exception:
                L_fc = None

        max_iter = int(max(0, self.max_iterations))
        tol = float(self.tolerance)
        for _ in range(max_iter):
            rotations = self._compute_local_rotations(
                vertices,
                uv,
                edge_i=edge_i,
                edge_j=edge_j,
                edge_w=edge_w,
                edge_vec_3d=edge_vec_3d,
                reference_basis=reference_basis,
            )
            b = self._compute_global_rhs(
                rotations,
                edge_i=edge_i,
                edge_j=edge_j,
                edge_w=edge_w,
                edge_vec_3d=edge_vec_3d,
                n_vertices=n,
            )
            rhs = b[free_indices].copy()
            if fixed_indices.size > 0 and L_fc is not None:
                rhs -= L_fc @ uv[fixed_indices]

            new_uv = uv.copy()
            if use_lu and lu is not None:
                new_uv[free_indices, 0] = np.asarray(lu.solve(rhs[:, 0])).ravel()
                new_uv[free_indices, 1] = np.asarray(lu.solve(rhs[:, 1])).ravel()
            else:
                new_uv[free_indices, 0] = np.asarray(spsolve(L_ff_reg, rhs[:, 0])).ravel()
                new_uv[free_indices, 1] = np.asarray(spsolve(L_ff_reg, rhs[:, 1])).ravel()

            diff = new_uv[free_indices] - uv[free_indices]
            if fixed_indices.size > 0:
                new_uv[fixed_indices] = fixed_uv

            try:
                max_delta = float(np.max(np.abs(diff))) if diff.size else 0.0
                scale_ref = float(np.max(np.abs(uv[free_indices]))) if uv[free_indices].size else 1.0
            except Exception:
                max_delta = 0.0
                scale_ref = 1.0
            uv = new_uv
            if max_delta <= tol * max(1.0, scale_ref):
                break

        return uv

    def _pick_anchor_pair(self, mesh: MeshData, candidates: np.ndarray) -> np.ndarray:
        candidates = np.asarray(candidates, dtype=np.int32).reshape(-1)
        n = int(mesh.n_vertices)
        if n <= 0:
            return np.array([0, 0], dtype=np.int32)
        if candidates.size < 2:
            a = int(candidates[0]) if candidates.size == 1 else 0
            b = 1 if n > 1 else a
            return np.array([a, b], dtype=np.int32)

        verts = np.asarray(mesh.vertices, dtype=np.float64)
        a0 = int(candidates[0])
        d0 = np.linalg.norm(verts[candidates] - verts[a0], axis=1)
        b = int(candidates[int(np.argmax(d0))])
        d1 = np.linalg.norm(verts[candidates] - verts[b], axis=1)
        a = int(candidates[int(np.argmax(d1))])
        if a == b:
            for cand in candidates:
                if int(cand) != b:
                    a = int(cand)
                    break
        return np.array([b, a], dtype=np.int32)

    def _align_uv_to_anchors(self, uv: np.ndarray, a: int, b: int, target_dist: float) -> np.ndarray:
        uv = np.asarray(uv, dtype=np.float64)
        if uv.ndim != 2 or uv.shape[0] == 0 or uv.shape[1] < 2:
            return np.zeros((int(uv.shape[0]) if uv.ndim == 2 else 0, 2), dtype=np.float64)

        a = int(a)
        b = int(b)
        if a < 0 or b < 0 or a >= uv.shape[0] or b >= uv.shape[0]:
            return uv[:, :2].copy()

        p0 = uv[a, :2].copy()
        p1 = uv[b, :2].copy()
        v = p1 - p0
        norm = float(np.linalg.norm(v))
        if not np.isfinite(norm) or norm < 1e-12:
            out = uv[:, :2].copy()
            out[a] = (0.0, 0.0)
            out[b] = (float(target_dist), 0.0)
            return out

        angle = float(np.arctan2(v[1], v[0]))
        c = float(np.cos(-angle))
        s = float(np.sin(-angle))
        rot = np.array([[c, -s], [s, c]], dtype=np.float64)
        scale = float(target_dist) / norm if np.isfinite(target_dist) and float(target_dist) > 0 else 1.0
        out = (uv[:, :2] - p0) @ rot.T
        out *= scale
        out[a] = (0.0, 0.0)
        out[b] = (float(target_dist), 0.0)
        return out

    def _build_fixed_constraints(
        self,
        mesh: MeshData,
        uv: np.ndarray,
        *,
        boundary_type: str,
        boundary: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = int(mesh.n_vertices)
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        boundary_type = str(boundary_type or "free").lower().strip()
        boundary = np.asarray(boundary, dtype=np.int32).reshape(-1)

        if boundary_type == "fixed_circle" and boundary.size >= 3:
            b_pts = verts[boundary]
            diffs = b_pts[(np.arange(len(boundary)) + 1) % len(boundary)] - b_pts
            perim = float(np.linalg.norm(diffs, axis=1).sum())
            radius = perim / (2.0 * np.pi) if perim > 1e-12 else 1.0
            angles = np.linspace(0.0, 2.0 * np.pi, len(boundary), endpoint=False)
            boundary_uv = np.column_stack([np.cos(angles), np.sin(angles)]) * radius
            new_uv = np.asarray(uv, dtype=np.float64).copy()
            new_uv[boundary] = boundary_uv
            return boundary.copy(), boundary_uv.astype(np.float64, copy=False), new_uv

        if boundary_type == "fixed_rect" and boundary.size >= 4:
            b_pts = verts[boundary]
            seg = b_pts[(np.arange(len(boundary)) + 1) % len(boundary)] - b_pts
            seg_len = np.linalg.norm(seg, axis=1)
            perim = float(seg_len.sum())
            if perim <= 1e-12:
                return boundary.copy(), uv[boundary][:, :2].copy(), np.asarray(uv, dtype=np.float64).copy()

            basis = self._compute_reference_basis(mesh)
            proj = (basis @ b_pts.T).T
            ext = proj.max(axis=0) - proj.min(axis=0)
            w = float(max(ext[0], 1e-6))
            h = float(max(ext[1], 1e-6))
            rect_perim = 2.0 * (w + h)
            scale = perim / rect_perim if rect_perim > 1e-12 else 1.0
            w *= scale
            h *= scale

            cum = np.concatenate([[0.0], np.cumsum(seg_len)])
            t = cum[:-1] / perim * (2.0 * (w + h))
            out = np.zeros((len(boundary), 2), dtype=np.float64)
            for i, ti in enumerate(t):
                if ti < w:
                    out[i] = (ti, 0.0)
                elif ti < w + h:
                    out[i] = (w, ti - w)
                elif ti < 2.0 * w + h:
                    out[i] = (w - (ti - (w + h)), h)
                else:
                    out[i] = (0.0, h - (ti - (2.0 * w + h)))

            new_uv = np.asarray(uv, dtype=np.float64).copy()
            new_uv[boundary] = out
            return boundary.copy(), out, new_uv

        candidates = boundary if boundary.size >= 2 else np.arange(n, dtype=np.int32)
        anchors = self._pick_anchor_pair(mesh, candidates)
        dist = float(np.linalg.norm(verts[anchors[1]] - verts[anchors[0]]))
        if not np.isfinite(dist) or dist < 1e-9:
            dist = 1.0
        new_uv = self._align_uv_to_anchors(uv, anchors[0], anchors[1], dist)
        fixed_uv = new_uv[anchors][:, :2].copy()
        return anchors.astype(np.int32, copy=False), fixed_uv, new_uv

    def _compute_reference_basis(self, mesh: MeshData) -> np.ndarray:
        vertices = mesh.vertices.astype(np.float64, copy=False)
        centered = vertices - vertices.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = np.argsort(eigenvalues)[::-1]
        axes = eigenvectors[:, order]
        u = axes[:, 0].copy()
        v = axes[:, 1].copy()
        for axis in (u, v):
            idx = int(np.argmax(np.abs(axis)))
            if axis[idx] < 0:
                axis *= -1
        return np.vstack([u, v])

    def _compute_cotangent_edge_weights(self, mesh: MeshData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        if vertices.ndim != 2 or vertices.shape[0] == 0 or faces.ndim != 2 or faces.shape[0] == 0 or faces.shape[1] < 3:
            return (
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.float64),
            )

        f = faces[:, :3].astype(np.int32, copy=False)
        a = f[:, 0]
        b = f[:, 1]
        c = f[:, 2]
        va = vertices[a]
        vb = vertices[b]
        vc = vertices[c]
        eps = 1e-10

        u0 = va - vc
        v0 = vb - vc
        cot_c = np.einsum("ij,ij->i", u0, v0) / (np.linalg.norm(np.cross(u0, v0), axis=1) + eps)
        u1 = vb - va
        v1 = vc - va
        cot_a = np.einsum("ij,ij->i", u1, v1) / (np.linalg.norm(np.cross(u1, v1), axis=1) + eps)
        u2 = vc - vb
        v2 = va - vb
        cot_b = np.einsum("ij,ij->i", u2, v2) / (np.linalg.norm(np.cross(u2, v2), axis=1) + eps)

        edge_i = np.concatenate([a, b, c]).astype(np.int32, copy=False)
        edge_j = np.concatenate([b, c, a]).astype(np.int32, copy=False)
        edge_w = (0.5 * np.concatenate([cot_c, cot_a, cot_b])).astype(np.float64, copy=False)
        finite = np.isfinite(edge_w)
        if not bool(np.all(finite)):
            edge_w = edge_w.copy()
            edge_w[~finite] = 0.0

        ii = np.minimum(edge_i, edge_j)
        jj = np.maximum(edge_i, edge_j)
        keep = ii != jj
        ii = ii[keep]
        jj = jj[keep]
        edge_w = edge_w[keep]
        if ii.size == 0:
            return (
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.float64),
            )

        order = np.lexsort((jj, ii))
        ii = ii[order]
        jj = jj[order]
        edge_w = edge_w[order]
        n = int(vertices.shape[0])
        key = ii.astype(np.int64) * np.int64(n) + jj.astype(np.int64)
        starts = np.r_[0, np.nonzero(key[1:] != key[:-1])[0] + 1]
        w_sum = np.add.reduceat(edge_w, starts)
        ii_u = ii[starts]
        jj_u = jj[starts]
        w_sum = np.maximum(w_sum, 1e-6)
        return (
            np.asarray(ii_u, dtype=np.int32),
            np.asarray(jj_u, dtype=np.int32),
            np.asarray(w_sum, dtype=np.float64),
        )

    def _smooth_uv_laplacian(self, *args, **kwargs) -> np.ndarray:
        return _smooth_uv_laplacian(*args, **kwargs)

    def _build_laplacian(
        self,
        n: int,
        edge_i: np.ndarray,
        edge_j: np.ndarray,
        edge_w: np.ndarray,
    ) -> sparse.csr_matrix:
        n = int(n)
        edge_i = np.asarray(edge_i, dtype=np.int32).reshape(-1)
        edge_j = np.asarray(edge_j, dtype=np.int32).reshape(-1)
        edge_w = np.asarray(edge_w, dtype=np.float64).reshape(-1)
        if n <= 0 or edge_i.size == 0:
            return sparse.csr_matrix((n, n), dtype=np.float64)

        m = int(min(edge_i.size, edge_j.size, edge_w.size))
        edge_i = edge_i[:m]
        edge_j = edge_j[:m]
        edge_w = edge_w[:m]
        rows = np.concatenate([edge_i, edge_j, edge_i, edge_j])
        cols = np.concatenate([edge_j, edge_i, edge_i, edge_j])
        vals = np.concatenate([-edge_w, -edge_w, edge_w, edge_w])
        return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)

    def _compute_local_rotations(
        self,
        vertices: np.ndarray,
        uv: np.ndarray,
        *,
        edge_i: np.ndarray,
        edge_j: np.ndarray,
        edge_w: np.ndarray,
        edge_vec_3d: np.ndarray,
        reference_basis: np.ndarray,
    ) -> np.ndarray:
        vertices = np.asarray(vertices, dtype=np.float64)
        uv = np.asarray(uv, dtype=np.float64)
        n = int(vertices.shape[0]) if vertices.ndim == 2 else 0
        if n <= 0:
            return np.zeros((0, 2, 3), dtype=np.float64)

        edge_i = np.asarray(edge_i, dtype=np.int32).reshape(-1)
        edge_j = np.asarray(edge_j, dtype=np.int32).reshape(-1)
        edge_w = np.asarray(edge_w, dtype=np.float64).reshape(-1)
        edge_vec_3d = np.asarray(edge_vec_3d, dtype=np.float64).reshape(-1, 3)
        default_R = np.asarray(reference_basis, dtype=np.float64).reshape(2, 3)
        if edge_i.size == 0:
            return np.tile(default_R.reshape(1, 2, 3), (n, 1, 1))

        m = int(min(edge_i.size, edge_j.size, edge_w.size, edge_vec_3d.shape[0]))
        ei = edge_i[:m]
        ej = edge_j[:m]
        w = edge_w[:m]
        e3 = edge_vec_3d[:m]
        e_uv = uv[ej, :2] - uv[ei, :2]
        w0 = (w * e_uv[:, 0]).astype(np.float64, copy=False)
        w1 = (w * e_uv[:, 1]).astype(np.float64, copy=False)
        c0 = (w0[:, None] * e3).astype(np.float64, copy=False)
        c1 = (w1[:, None] * e3).astype(np.float64, copy=False)
        contrib = np.concatenate([c0, c1], axis=1)

        s_flat = np.zeros((n, 6), dtype=np.float64)
        np.add.at(s_flat, ei, contrib)
        np.add.at(s_flat, ej, contrib)
        s_mat = s_flat.reshape(n, 2, 3)

        try:
            U, _s, Vt = np.linalg.svd(s_mat, full_matrices=True)
        except np.linalg.LinAlgError:
            return np.tile(default_R.reshape(1, 2, 3), (n, 1, 1))

        M = np.zeros((2, 3), dtype=np.float64)
        M[0, 0] = 1.0
        M[1, 1] = 1.0
        rotations = (U @ M) @ Vt
        deg = np.bincount(np.concatenate([ei, ej]).astype(np.int64, copy=False), minlength=n)
        rotations[deg == 0] = default_R
        return rotations

    def _compute_global_rhs(
        self,
        rotations: np.ndarray,
        *,
        edge_i: np.ndarray,
        edge_j: np.ndarray,
        edge_w: np.ndarray,
        edge_vec_3d: np.ndarray,
        n_vertices: int,
    ) -> np.ndarray:
        n = int(n_vertices)
        b = np.zeros((n, 2), dtype=np.float64)
        if n <= 0:
            return b

        edge_i = np.asarray(edge_i, dtype=np.int32).reshape(-1)
        edge_j = np.asarray(edge_j, dtype=np.int32).reshape(-1)
        edge_w = np.asarray(edge_w, dtype=np.float64).reshape(-1)
        edge_vec_3d = np.asarray(edge_vec_3d, dtype=np.float64).reshape(-1, 3)
        rotations = np.asarray(rotations, dtype=np.float64)
        if edge_i.size == 0:
            return b

        m = int(min(edge_i.size, edge_j.size, edge_w.size, edge_vec_3d.shape[0]))
        ei = edge_i[:m]
        ej = edge_j[:m]
        w = edge_w[:m]
        e3 = edge_vec_3d[:m]
        R_avg = 0.5 * (rotations[ei] + rotations[ej])
        rotated_edge = np.einsum("mij,mj->mi", R_avg, e3)
        contrib = (w[:, None] * rotated_edge).astype(np.float64, copy=False)
        np.add.at(b, ei, -contrib)
        np.add.at(b, ej, contrib)
        return b

    def _compute_distortion(self, mesh: MeshData, uv: np.ndarray) -> np.ndarray:
        return compute_face_distortion(mesh, uv)
