"""
Surface Separator Module
표면 분리 - 기와의 내면/외면 자동 감지 및 분리

법선 방향을 기준으로 메쉬를 내면과 외면으로 분리합니다.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, List
import logging
import numpy as np

from .logging_utils import log_once
from .mesh_loader import MeshData

_LOGGER = logging.getLogger(__name__)


@dataclass
class SeparatedSurfaces:
    """
    분리된 표면 결과
    
    Attributes:
        inner_surface: 내면 메쉬
        outer_surface: 외면 메쉬
        inner_face_indices: 내면에 속하는 원본 면 인덱스
        outer_face_indices: 외면에 속하는 원본 면 인덱스
    """
    inner_surface: Optional[MeshData]
    outer_surface: Optional[MeshData]
    inner_face_indices: np.ndarray
    outer_face_indices: np.ndarray
    # Optional third label for "edge/thickness" faces (기와 미구 등).
    migu_face_indices: np.ndarray | None = None
    # Optional debugging/quality info (method used, fit quality, etc.)
    meta: dict[str, Any] = field(default_factory=dict)
    
    @property
    def has_both(self) -> bool:
        """내면과 외면 모두 존재하는지"""
        return self.inner_surface is not None and self.outer_surface is not None


class SurfaceSeparator:
    """
    표면 분리기
    
    기와처럼 내면/외면이 구분되는 메쉬를 자동으로 분리합니다.
    """
    
    def __init__(self, angle_threshold: float = 90.0):
        """
        Args:
            angle_threshold: 표면 분리 각도 임계값 (도)
        """
        self.angle_threshold = angle_threshold
    
    def auto_detect_surfaces(
        self,
        mesh: MeshData,
        reference_direction: Optional[np.ndarray] = None,
        *,
        method: str = "auto",
        projector_resolution: int = 768,
        view_depth_tol: float | None = None,
        return_submeshes: bool = True,
    ) -> SeparatedSurfaces:
        """
        법선 방향으로 내면/외면 자동 분리
        
        Args:
            mesh: 입력 메쉬
            reference_direction: 기준 방향 (None이면 자동 감지)
            
        Returns:
            SeparatedSurfaces: 분리된 표면들
        """
        m = str(method or "auto").strip().lower()
        if m in {"auto", "best", "smart"}:
            # 1) Cylindrical thickness split (roof tiles): robust and fast.
            try:
                r0 = self._auto_detect_surfaces_by_cylinder_radius(mesh, return_submeshes=bool(return_submeshes))
                if bool(r0.meta.get("cylinder_ok", False)):
                    return r0
            except Exception:
                log_once(
                    _LOGGER,
                    "surface_separator:auto:cylinder_failed",
                    logging.DEBUG,
                    "Auto surface separation: cylinder path failed; falling back",
                    exc_info=True,
                )

            # 2) View-based split along estimated thickness axis (works even if normals are flipped).
            try:
                r1 = self._auto_detect_surfaces_by_views(
                    mesh,
                    resolution=int(projector_resolution),
                    depth_tol=view_depth_tol,
                    return_submeshes=bool(return_submeshes),
                )
                n_faces = int(getattr(mesh, "n_faces", 0) or 0)
                if n_faces <= 0:
                    return r1

                outer_n = int(getattr(r1.outer_face_indices, "size", 0) or 0)
                inner_n = int(getattr(r1.inner_face_indices, "size", 0) or 0)
                if outer_n > max(10, int(0.01 * n_faces)) and inner_n > max(10, int(0.01 * n_faces)):
                    return r1
            except Exception:
                log_once(
                    _LOGGER,
                    "surface_separator:auto:views_failed",
                    logging.DEBUG,
                    "Auto surface separation: view path failed; falling back",
                    exc_info=True,
                )

            # 3) Fallback: global-normal split.
            return self._auto_detect_surfaces_by_normals(
                mesh,
                reference_direction=reference_direction,
                return_submeshes=bool(return_submeshes),
            )

        if m in {"view", "views", "topbottom", "top_bottom", "visible"}:
            return self._auto_detect_surfaces_by_views(
                mesh,
                resolution=int(projector_resolution),
                depth_tol=view_depth_tol,
                return_submeshes=bool(return_submeshes),
            )

        if m in {"cyl", "cylinder", "cylindrical", "tile"}:
            return self._auto_detect_surfaces_by_cylinder_radius(mesh, return_submeshes=bool(return_submeshes))

        return self._auto_detect_surfaces_by_normals(
            mesh,
            reference_direction=reference_direction,
            return_submeshes=bool(return_submeshes),
        )

    def _auto_detect_surfaces_by_normals(
        self,
        mesh: MeshData,
        *,
        reference_direction: Optional[np.ndarray] = None,
        return_submeshes: bool = True,
    ) -> SeparatedSurfaces:
        """법선 기반 자동 분리 (빠르지만, 원통형/곡면에서는 오분류 가능)"""
        mesh.compute_normals(compute_vertex_normals=False)

        if mesh.face_normals is None:
            raise ValueError("Failed to compute face normals")

        if reference_direction is None:
            reference_direction = self._estimate_reference_direction(mesh)

        d = np.asarray(reference_direction, dtype=np.float64).reshape(-1)
        if d.size < 3 or not np.isfinite(d[:3]).all():
            d = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        d = d[:3]
        d = d / (float(np.linalg.norm(d)) + 1e-12)

        fn = np.asarray(mesh.face_normals, dtype=np.float64)
        if fn.ndim != 2 or fn.shape[0] != int(getattr(mesh, "n_faces", 0) or 0) or fn.shape[1] < 3:
            raise ValueError("Invalid face normals")
        dots = np.dot(fn[:, :3], d)

        try:
            theta = float(self.angle_threshold)
        except Exception:
            theta = 90.0
        theta = float(np.clip(theta, 0.0, 90.0))
        cos_thr = float(np.cos(np.deg2rad(theta)))

        outer_strong = dots > cos_thr
        inner_strong = dots < -cos_thr
        ambiguous = ~(outer_strong | inner_strong)

        outer_mask = outer_strong | (ambiguous & (dots > 0.0))
        inner_mask = ~outer_mask

        outer_indices = np.where(outer_mask)[0].astype(np.int32, copy=False)
        inner_indices = np.where(inner_mask)[0].astype(np.int32, copy=False)

        outer_surface = mesh.extract_submesh(outer_indices) if bool(return_submeshes) and outer_indices.size > 0 else None
        inner_surface = mesh.extract_submesh(inner_indices) if bool(return_submeshes) and inner_indices.size > 0 else None

        return SeparatedSurfaces(
            inner_surface=inner_surface,
            outer_surface=outer_surface,
            inner_face_indices=inner_indices,
            outer_face_indices=outer_indices,
            migu_face_indices=None,
            meta={"method": "normals", "reference_direction": d.astype(np.float64, copy=False)},
        )

    def _auto_detect_surfaces_by_cylinder_radius(
        self,
        mesh: MeshData,
        *,
        return_submeshes: bool = True,
    ) -> SeparatedSurfaces:
        """
        원통형(기와) 가정: 원통축 + 반경(r) 분포로 outer/inner/migu를 분리합니다.

        - 반경 기반이므로 face normal winding이 뒤집혀 있어도 비교적 안정적으로 동작합니다.
        - 원통성이 약하거나(혹은 내/외면이 한 장만 있는 경우) 두께 분리가 불명확하면
          meta["cylinder_ok"]=False 로 표시하고 best-effort 결과를 반환합니다.
        """

        vertices = np.asarray(getattr(mesh, "vertices", np.zeros((0, 3))), dtype=np.float64)
        faces = np.asarray(getattr(mesh, "faces", np.zeros((0, 3))), dtype=np.int32)
        if (
            vertices.ndim != 2
            or vertices.shape[0] == 0
            or faces.ndim != 2
            or faces.shape[0] == 0
            or faces.shape[1] < 3
        ):
            empty = np.zeros((0,), dtype=np.int32)
            return SeparatedSurfaces(
                inner_surface=None,
                outer_surface=None,
                inner_face_indices=empty,
                outer_face_indices=empty,
                migu_face_indices=empty,
                meta={"method": "cylinder_radius", "cylinder_ok": False},
            )

        faces = faces[:, :3]
        n_faces = int(faces.shape[0])
        n_vertices = int(vertices.shape[0])

        # ---- helpers (local, minimal deps) ----
        def _unit(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
            a = np.asarray(v, dtype=np.float64).reshape(-1)
            if a.size < 3 or not np.isfinite(a[:3]).all():
                a = np.asarray(fallback, dtype=np.float64).reshape(3)
            a = a[:3]
            n = float(np.linalg.norm(a))
            if not np.isfinite(n) or n < 1e-12:
                a = np.asarray(fallback, dtype=np.float64).reshape(3)
                n = float(np.linalg.norm(a))
            if not np.isfinite(n) or n < 1e-12:
                return np.array([0.0, 1.0, 0.0], dtype=np.float64)
            return (a / n).astype(np.float64, copy=False)

        def _orthonormal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            a = _unit(axis, np.array([0.0, 1.0, 0.0], dtype=np.float64))
            tmp = np.array([0.0, 0.0, 1.0], dtype=np.float64) if abs(float(a[2])) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float64)
            b1 = np.cross(a, tmp)
            n1 = float(np.linalg.norm(b1))
            if not np.isfinite(n1) or n1 < 1e-12:
                tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                b1 = np.cross(a, tmp)
                n1 = float(np.linalg.norm(b1))
            b1 = b1 / (n1 + 1e-12)
            b2 = np.cross(a, b1)
            n2 = float(np.linalg.norm(b2))
            b2 = b2 / (n2 + 1e-12)
            return b1.astype(np.float64, copy=False), b2.astype(np.float64, copy=False)

        def _cylinder_axis_score(v: np.ndarray, axis: np.ndarray) -> float:
            # Lower is better.
            pts = np.asarray(v, dtype=np.float64)
            if pts.ndim != 2 or pts.shape[0] < 16 or pts.shape[1] < 3:
                return float("inf")
            a = _unit(axis, np.array([0.0, 1.0, 0.0], dtype=np.float64))
            t0 = pts[:, :3] @ a.reshape(3,)
            perp = pts[:, :3] - t0[:, None] * a.reshape(1, 3)
            center = perp.mean(axis=0)
            r0 = np.linalg.norm(perp - center.reshape(1, 3), axis=1)
            r0 = r0[np.isfinite(r0)]
            if r0.size < 16:
                return float("inf")
            med = float(np.median(r0))
            if not np.isfinite(med) or med < 1e-9:
                return float("inf")
            sd = float(np.std(r0))
            if not np.isfinite(sd):
                return float("inf")
            return float(sd / med)

        def _robust_circle_fit_2d(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float] | None:
            xx = np.asarray(x, dtype=np.float64).reshape(-1)
            yy = np.asarray(y, dtype=np.float64).reshape(-1)
            m = np.isfinite(xx) & np.isfinite(yy)
            xx = xx[m]
            yy = yy[m]
            if xx.size < 3:
                return None

            def _fit(x1: np.ndarray, y1: np.ndarray) -> tuple[np.ndarray, float] | None:
                try:
                    A = np.column_stack([2.0 * x1, 2.0 * y1, np.ones_like(x1)])
                    b = x1 * x1 + y1 * y1
                    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    a0, b0, c0 = [float(v) for v in sol]
                    r2 = float(c0 + a0 * a0 + b0 * b0)
                    r = float(np.sqrt(max(r2, 0.0)))
                    if not np.isfinite(r) or r <= 1e-12:
                        return None
                    return np.array([a0, b0], dtype=np.float64), r
                except Exception:
                    return None

            fitted = _fit(xx, yy)
            if fitted is None:
                return None
            center_xy, r = fitted

            try:
                res = np.abs(np.hypot(xx - float(center_xy[0]), yy - float(center_xy[1])) - float(r))
                if res.size >= 12 and np.isfinite(res).any():
                    med = float(np.median(res))
                    mad = float(np.median(np.abs(res - med)))
                    thr = med + 3.5 * float(max(mad, 1e-12))
                    keep = res <= thr
                    if int(np.count_nonzero(keep)) >= 3 and int(np.count_nonzero(keep)) < int(res.size):
                        fitted2 = _fit(xx[keep], yy[keep])
                        if fitted2 is not None:
                            center_xy, r = fitted2
            except Exception:
                pass

            return center_xy, float(r)

        # ---- sample vertices for axis/fit (deterministic stride) ----
        try:
            max_fit = int(getattr(mesh, "_cylinder_fit_max_points", 50000) or 50000)
        except Exception:
            max_fit = 50000
        max_fit = max(2000, min(int(max_fit), n_vertices))

        if n_vertices > max_fit:
            step = max(1, int(n_vertices // max_fit))
            idx = np.arange(0, n_vertices, step, dtype=np.int64)
            if idx.size > max_fit:
                idx = idx[:max_fit]
            v_s = vertices[idx, :3]
        else:
            v_s = vertices[:, :3]

        v_s = v_s[np.isfinite(v_s).all(axis=1)]
        if v_s.shape[0] < 32:
            empty = np.zeros((0,), dtype=np.int32)
            return SeparatedSurfaces(
                inner_surface=None,
                outer_surface=None,
                inner_face_indices=empty,
                outer_face_indices=empty,
                migu_face_indices=empty,
                meta={"method": "cylinder_radius", "cylinder_ok": False, "reason": "too_few_points"},
            )

        # PCA candidate axes + world axes (choose by cylinder score)
        try:
            c = np.mean(v_s, axis=0)
            x = v_s - c.reshape(1, 3)
            cov = (x.T @ x) / float(max(1, x.shape[0] - 1))
            w, vecs = np.linalg.eigh(cov)
            order = np.argsort(w)[::-1]
            vecs = vecs[:, order]
        except Exception:
            vecs = np.eye(3, dtype=np.float64)

        candidates = [
            vecs[:, 0],
            vecs[:, 1],
            vecs[:, 2],
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
        ]
        scores = [float(_cylinder_axis_score(v_s, a)) for a in candidates]
        best_i = int(np.nanargmin(np.asarray(scores, dtype=np.float64)))
        axis = _unit(candidates[best_i], np.array([0.0, 1.0, 0.0], dtype=np.float64))
        axis_score = float(scores[best_i]) if best_i < len(scores) else float("inf")

        b1, b2 = _orthonormal_basis(axis)

        # Fit circle center in the plane orthogonal to axis.
        t0 = v_s[:, :3] @ axis.reshape(3,)
        perp0 = v_s[:, :3] - t0[:, None] * axis.reshape(1, 3)
        x0 = perp0 @ b1.reshape(3,)
        y0 = perp0 @ b2.reshape(3,)
        fit = _robust_circle_fit_2d(x0, y0)
        if fit is None:
            cx = float(np.median(x0))
            cy = float(np.median(y0))
            r_fit = float(np.median(np.hypot(x0 - cx, y0 - cy)))
            center_xy = np.array([cx, cy], dtype=np.float64)
        else:
            center_xy, r_fit = fit
        center = (float(center_xy[0]) * b1 + float(center_xy[1]) * b2).astype(np.float64, copy=False)

        try:
            res = np.abs(np.hypot(x0 - float(center_xy[0]), y0 - float(center_xy[1])) - float(r_fit))
            resid_med = float(np.median(res[np.isfinite(res)])) if res.size else float("inf")
        except Exception:
            resid_med = float("inf")
        resid_norm = float(resid_med / float(max(float(r_fit), 1e-12))) if np.isfinite(resid_med) else float("inf")

        # ---- classify faces by centroid radius (chunked) ----
        try:
            chunk = int(getattr(mesh, "_cylinder_classify_chunk_faces", 250000) or 250000)
        except Exception:
            chunk = 250000
        chunk = max(20000, min(int(chunk), 500000))

        radii = np.empty((n_faces,), dtype=np.float32)
        for start in range(0, n_faces, chunk):
            end = min(n_faces, start + chunk)
            f = np.asarray(faces[start:end, :3], dtype=np.int32)
            if f.size == 0:
                continue
            v0 = vertices[f[:, 0]]
            v1 = vertices[f[:, 1]]
            v2 = vertices[f[:, 2]]
            cent = (v0 + v1 + v2) / 3.0
            tt = cent @ axis.reshape(3,)
            perp = cent - tt[:, None] * axis.reshape(1, 3)
            rv = perp - center.reshape(1, 3)
            r = np.linalg.norm(rv, axis=1)
            radii[start:end] = r.astype(np.float32, copy=False)

        r_valid = radii[np.isfinite(radii)].astype(np.float64, copy=False)
        if r_valid.size < 32:
            empty = np.zeros((0,), dtype=np.int32)
            return SeparatedSurfaces(
                inner_surface=None,
                outer_surface=None,
                inner_face_indices=empty,
                outer_face_indices=empty,
                migu_face_indices=empty,
                meta={
                    "method": "cylinder_radius",
                    "cylinder_ok": False,
                    "reason": "too_few_radii",
                    "cylinder_axis": axis,
                    "cylinder_center": center,
                    "cylinder_fit_radius": float(r_fit),
                },
            )

        r_med = float(np.median(r_valid))
        lower = r_valid[r_valid <= r_med]
        upper = r_valid[r_valid >= r_med]
        r_inner = float(np.median(lower)) if lower.size else float(r_med)
        r_outer = float(np.median(upper)) if upper.size else float(r_med)
        thickness = float(r_outer - r_inner)

        try:
            s_in = float(np.median(np.abs(lower - r_inner))) if lower.size else 0.0
            s_out = float(np.median(np.abs(upper - r_outer))) if upper.size else 0.0
        except Exception:
            s_in = 0.0
            s_out = 0.0
        noise = float(max(s_in, s_out, 1e-12))

        # Mid-band width for migu: wide enough to catch thickness side walls.
        # NOTE: centroid radii on side walls tend to skew away from the exact mid radius, so keep this fairly wide.
        margin = float(max(0.30 * abs(thickness), 3.0 * noise))
        if thickness <= 0.0 or not np.isfinite(thickness):
            margin = float(max(margin, 0.0))

        r_mid = 0.5 * (r_inner + r_outer)
        if not np.isfinite(r_mid):
            r_mid = float(r_med)

        finite = np.isfinite(radii)
        thr_in = float(r_mid - margin)
        thr_out = float(r_mid + margin)
        inner_mask = finite & (radii <= np.float32(thr_in))
        outer_mask = finite & (radii >= np.float32(thr_out))
        migu_mask = ~(inner_mask | outer_mask)

        inner_idx = np.where(inner_mask)[0].astype(np.int32, copy=False)
        outer_idx = np.where(outer_mask)[0].astype(np.int32, copy=False)
        migu_idx = np.where(migu_mask)[0].astype(np.int32, copy=False)

        min_part = max(3, int(0.01 * max(1, n_faces)))
        sep_quality = float(abs(thickness) / float(noise)) if noise > 0.0 else float("inf")
        cylinder_ok = bool(
            np.isfinite(axis_score)
            and np.isfinite(resid_norm)
            and resid_norm < 0.08
            and np.isfinite(thickness)
            and thickness > 0.0
            and sep_quality >= 6.0
            and int(inner_idx.size) >= int(min_part)
            and int(outer_idx.size) >= int(min_part)
        )

        outer_surface = mesh.extract_submesh(outer_idx) if bool(return_submeshes) and outer_idx.size > 0 else None
        inner_surface = mesh.extract_submesh(inner_idx) if bool(return_submeshes) and inner_idx.size > 0 else None

        return SeparatedSurfaces(
            inner_surface=inner_surface,
            outer_surface=outer_surface,
            inner_face_indices=inner_idx,
            outer_face_indices=outer_idx,
            migu_face_indices=migu_idx,
            meta={
                "method": "cylinder_radius",
                "cylinder_ok": cylinder_ok,
                "cylinder_axis": axis,
                "cylinder_center": center,
                "cylinder_fit_radius": float(r_fit),
                "cylinder_axis_score": float(axis_score),
                "cylinder_fit_resid_norm": float(resid_norm),
                "cylinder_radius_inner": float(r_inner),
                "cylinder_radius_outer": float(r_outer),
                "cylinder_thickness_est": float(thickness),
                "cylinder_sep_quality": float(sep_quality),
                "cylinder_margin": float(margin),
            },
        )

    def _auto_detect_surfaces_by_views(
        self,
        mesh: MeshData,
        *,
        resolution: int = 768,
        depth_tol: float | None = None,
        return_submeshes: bool = True,
    ) -> SeparatedSurfaces:
        """
        두께축(+/-) 방향에서 '보이는 면'을 기준으로 outer/inner를 분리합니다.
        얇은 쉘(기와/금속판 등)에서 법선 기반 분리가 불안정할 때 사용합니다.

        구현 메모:
        - 정사투영 "카메라"를 사용하는 대신, face centroid를 픽셀 그리드에 binning 하여
          간이 Z-buffer(전면 depth)를 구성합니다. (triangle 래스터화 대비 훨씬 빠르고,
          대용량 메쉬에서도 세부(주름) 면이 downsample 때문에 누락되는 문제를 줄입니다.)
        - +/- 방향에서 둘 다/둘 다 아닌(가려진) 면은 두께축 좌표 + (가능하면) 면 법선 prior로 보정합니다.
        """

        vertices = np.asarray(getattr(mesh, "vertices", np.zeros((0, 3))), dtype=np.float64)
        faces = np.asarray(getattr(mesh, "faces", np.zeros((0, 3))), dtype=np.int32)
        if (
            vertices.ndim != 2
            or vertices.shape[0] == 0
            or faces.ndim != 2
            or faces.shape[0] == 0
            or faces.shape[1] < 3
        ):
            outer_idx = np.zeros((0,), dtype=np.int32)
            inner_idx = np.zeros((0,), dtype=np.int32)
            return SeparatedSurfaces(
                inner_surface=None,
                outer_surface=None,
                inner_face_indices=inner_idx,
                outer_face_indices=outer_idx,
            )

        faces = faces[:, :3]
        n_faces = int(faces.shape[0])

        # Two view directions: +/- thickness axis (orientation-invariant).
        d = np.asarray(self._estimate_reference_direction(mesh), dtype=np.float64).reshape(-1)
        if d.size < 3 or not np.isfinite(d[:3]).all():
            d = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        d = d[:3]
        d = d / (float(np.linalg.norm(d)) + 1e-12)

        # Orthonormal basis for the projection plane (b1, b2) ⟂ d
        tmp = np.array([0.0, 0.0, 1.0], dtype=np.float64) if abs(float(d[2])) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float64)
        b1 = np.cross(d, tmp)
        n1 = float(np.linalg.norm(b1))
        if not np.isfinite(n1) or n1 < 1e-12:
            tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            b1 = np.cross(d, tmp)
            n1 = float(np.linalg.norm(b1))
        b1 = b1 / (n1 + 1e-12)
        b2 = np.cross(d, b1)
        b2 = b2 / (float(np.linalg.norm(b2)) + 1e-12)

        # Projection bounds from vertices (cheap and stable vs. centroid bounds).
        x_all = (vertices[:, :3] @ b1.reshape(3,)).astype(np.float64, copy=False)
        y_all = (vertices[:, :3] @ b2.reshape(3,)).astype(np.float64, copy=False)
        x_min, x_max = float(np.nanmin(x_all)), float(np.nanmax(x_all))
        y_min, y_max = float(np.nanmin(y_all)), float(np.nanmax(y_all))

        width = float(x_max - x_min)
        height = float(y_max - y_min)
        if not np.isfinite(width) or not np.isfinite(height) or (abs(width) < 1e-12 and abs(height) < 1e-12):
            width = 1.0
            height = 1.0

        res = int(max(64, int(resolution)))
        if width > height:
            img_w = int(res)
            img_h = int(res * height / max(1e-12, width))
        else:
            img_h = int(res)
            img_w = int(res * width / max(1e-12, height))

        img_h = max(1, int(img_h))
        img_w = max(1, int(img_w))

        # Total pixels cap (keep it small; we don't need photo-real depth here)
        max_total = 8_000_000
        total = int(img_h) * int(img_w)
        if total > max_total:
            s = float(np.sqrt(float(max_total) / float(max(1, total))))
            img_w = max(1, int(float(img_w) * s))
            img_h = max(1, int(float(img_h) * s))

        dx = float(x_max - x_min)
        dy = float(y_max - y_min)
        if abs(dx) < 1e-12:
            dx = 1e-12
        if abs(dy) < 1e-12:
            dy = 1e-12

        scale = float(max(width / float(img_w), height / float(img_h)))
        tol_val = depth_tol
        if tol_val is None:
            tol_val = float(scale) * 0.75
        tol_f = float(max(0.0, float(tol_val)))

        try:
            chunk = int(getattr(mesh, "_views_classify_chunk_faces", 250000) or 250000)
        except Exception:
            chunk = 250000
        chunk = max(20000, min(int(chunk), 500000))

        depth_plus = np.full((img_h, img_w), -np.inf, dtype=np.float32)
        depth_minus = np.full((img_h, img_w), -np.inf, dtype=np.float32)

        # Pass 1: build depth buffers (front-most depth for +/- directions)
        for start in range(0, n_faces, chunk):
            end = min(n_faces, start + chunk)
            f = np.asarray(faces[start:end, :3], dtype=np.int32)
            if f.size == 0:
                continue
            v0 = vertices[f[:, 0]]
            v1 = vertices[f[:, 1]]
            v2 = vertices[f[:, 2]]
            cent = (v0 + v1 + v2) / 3.0

            x = cent @ b1.reshape(3,)
            y = cent @ b2.reshape(3,)
            dp = cent @ d.reshape(3,)

            sx = (x - x_min) / dx * float(img_w - 1)
            sy = (y - y_min) / dy * float(img_h - 1)
            sy = float(img_h - 1) - sy

            px = np.clip(np.rint(sx), 0, img_w - 1).astype(np.int32, copy=False)
            py = np.clip(np.rint(sy), 0, img_h - 1).astype(np.int32, copy=False)

            try:
                np.maximum.at(depth_plus, (py, px), dp.astype(np.float32, copy=False))
                np.maximum.at(depth_minus, (py, px), (-dp).astype(np.float32, copy=False))
            except Exception:
                pass

        vis_plus = np.zeros((n_faces,), dtype=bool)
        vis_minus = np.zeros((n_faces,), dtype=bool)
        t = np.empty((n_faces,), dtype=np.float32)

        # Pass 2: mark visible faces + store thickness coordinate t
        for start in range(0, n_faces, chunk):
            end = min(n_faces, start + chunk)
            f = np.asarray(faces[start:end, :3], dtype=np.int32)
            if f.size == 0:
                continue
            v0 = vertices[f[:, 0]]
            v1 = vertices[f[:, 1]]
            v2 = vertices[f[:, 2]]
            cent = (v0 + v1 + v2) / 3.0

            x = cent @ b1.reshape(3,)
            y = cent @ b2.reshape(3,)
            dp = cent @ d.reshape(3,)
            t[start:end] = dp.astype(np.float32, copy=False)

            sx = (x - x_min) / dx * float(img_w - 1)
            sy = (y - y_min) / dy * float(img_h - 1)
            sy = float(img_h - 1) - sy

            px = np.clip(np.rint(sx), 0, img_w - 1).astype(np.int32, copy=False)
            py = np.clip(np.rint(sy), 0, img_h - 1).astype(np.int32, copy=False)

            try:
                front_p = depth_plus[py, px].astype(np.float32, copy=False)
                front_m = depth_minus[py, px].astype(np.float32, copy=False)
            except Exception:
                front_p = np.full((end - start,), -np.inf, dtype=np.float32)
                front_m = np.full((end - start,), -np.inf, dtype=np.float32)

            dp32 = dp.astype(np.float32, copy=False)
            dm32 = (-dp).astype(np.float32, copy=False)

            vis_plus[start:end] = np.isfinite(front_p) & np.isfinite(dp32) & (dp32 >= (front_p - float(tol_f)))
            vis_minus[start:end] = np.isfinite(front_m) & np.isfinite(dm32) & (dm32 >= (front_m - float(tol_f)))

        seed_outer = vis_plus & ~vis_minus
        seed_inner = vis_minus & ~vis_plus

        try:
            med = float(np.nanmedian(t.astype(np.float64, copy=False)))
        except Exception:
            med = 0.0

        # Normal prior for occluded folds + migu detection (side walls)
        dotn = None
        try:
            if getattr(mesh, "face_normals", None) is None:
                mesh.compute_normals(compute_vertex_normals=False)
            fn = np.asarray(getattr(mesh, "face_normals", None), dtype=np.float64)
            if fn.ndim == 2 and fn.shape[0] == n_faces and fn.shape[1] >= 3:
                fn = fn[:, :3]
                nrm = np.linalg.norm(fn, axis=1, keepdims=True)
                fn = fn / (nrm + 1e-12)
                dotn = np.asarray(fn @ d.reshape(3,), dtype=np.float64).reshape(-1)
        except Exception:
            dotn = None

        if isinstance(dotn, np.ndarray) and dotn.shape == (n_faces,):
            seed_fix_thr = 0.35  # require a strong normal agreement to override view seeds
            finite_n = np.isfinite(dotn)
            fix_to_outer = seed_inner & finite_n & (dotn >= float(seed_fix_thr))
            fix_to_inner = seed_outer & finite_n & (dotn <= -float(seed_fix_thr))
            if bool(fix_to_outer.any()):
                seed_outer = seed_outer | fix_to_outer
                seed_inner = seed_inner & ~fix_to_outer
            if bool(fix_to_inner.any()):
                seed_inner = seed_inner | fix_to_inner
                seed_outer = seed_outer & ~fix_to_inner

        ambiguous = ~(seed_outer | seed_inner)
        if isinstance(dotn, np.ndarray) and dotn.shape == (n_faces,):
            thr = 0.12  # small but non-zero: ignore near-perpendicular faces (side walls)
            confident = ambiguous & np.isfinite(dotn) & (np.abs(dotn) >= float(thr))
            assign_outer = (ambiguous & ~confident & (t >= float(med))) | (confident & (dotn >= 0.0))
        else:
            assign_outer = ambiguous & (t >= float(med))
        assign_inner = ambiguous & ~assign_outer

        outer_mask = seed_outer | assign_outer
        inner_mask = seed_inner | assign_inner

        migu_mask = None
        if isinstance(dotn, np.ndarray) and dotn.shape == (n_faces,):
            # Side walls are near-perpendicular to the thickness axis.
            migu_absdot_max = 0.35
            mm = np.isfinite(dotn) & (np.abs(dotn) <= float(migu_absdot_max))
            # Guard: if axis guess is bad, this can swallow too much; disable in that case.
            try:
                frac = float(np.mean(mm)) if mm.size else 0.0
            except Exception:
                frac = 0.0
            if frac > 0.40:
                mm = np.zeros((n_faces,), dtype=bool)
            if bool(mm.any()):
                outer_mask = outer_mask & ~mm
                inner_mask = inner_mask & ~mm
                migu_mask = mm

        outer_indices = np.where(outer_mask)[0].astype(np.int32, copy=False)
        inner_indices = np.where(inner_mask)[0].astype(np.int32, copy=False)
        migu_indices = np.where(migu_mask)[0].astype(np.int32, copy=False) if isinstance(migu_mask, np.ndarray) else None

        outer_surface = mesh.extract_submesh(outer_indices) if bool(return_submeshes) and outer_indices.size > 0 else None
        inner_surface = mesh.extract_submesh(inner_indices) if bool(return_submeshes) and inner_indices.size > 0 else None

        return SeparatedSurfaces(
            inner_surface=inner_surface,
            outer_surface=outer_surface,
            inner_face_indices=inner_indices,
            outer_face_indices=outer_indices,
            migu_face_indices=migu_indices,
            meta={
                "method": "views_axis",
                "thickness_axis": d.astype(np.float64, copy=False),
                "resolution": int(res),
                "img_size": (int(img_w), int(img_h)),
                "depth_tol": None if depth_tol is None else float(depth_tol),
                "depth_tol_effective": float(tol_f),
                "migu_absdot_max": 0.35,
            },
        )
    
    def _estimate_reference_direction(self, mesh: MeshData) -> np.ndarray:
        """
        기준 방향 자동 추정

        기와처럼 얇은 쉘/대칭 형상에서는 면적 가중 법선 합이 0에 가까워지기 쉬워서,
        먼저 PCA로 "두께 방향"(최소 분산 축)을 추정하고 실패 시 기존 방식으로 fallback 합니다.
        """
        try:
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            faces = np.asarray(mesh.faces, dtype=np.int32)
            if vertices.ndim != 2 or vertices.shape[0] == 0:
                return np.array([0.0, 0.0, 1.0], dtype=np.float64)
            if faces.ndim != 2 or faces.shape[0] == 0 or faces.shape[1] < 3:
                return np.array([0.0, 0.0, 1.0], dtype=np.float64)

            # 1) PCA: smallest-variance axis ~= sheet/thickness normal (robust for tiles).
            try:
                v = np.asarray(vertices[:, :3], dtype=np.float64)
                v = v[np.isfinite(v).all(axis=1)]
                if v.shape[0] >= 8:
                    c = np.mean(v, axis=0)
                    x = v - c
                    cov = (x.T @ x) / float(max(1, x.shape[0] - 1))
                    w, vecs = np.linalg.eigh(cov)
                    ref = vecs[:, int(np.argsort(w)[0])]
                    ref_norm = float(np.linalg.norm(ref))
                    if ref_norm > 1e-12:
                        ref = (ref / ref_norm).astype(np.float64, copy=False)
                        # Prefer +Z-ish direction to avoid random flips.
                        if float(ref[2]) < 0.0:
                            ref = -ref
                        return ref
            except Exception:
                pass

            # 2) Area-weighted normal sum fallback.
            v0 = vertices[faces[:, 0]]
            v1 = vertices[faces[:, 1]]
            v2 = vertices[faces[:, 2]]

            cross = np.cross(v1 - v0, v2 - v0)
            weighted_normal = np.sum(cross, axis=0)
            norm = float(np.linalg.norm(weighted_normal))
            if norm > 1e-10:
                return (weighted_normal / norm).astype(np.float64, copy=False)
        except Exception:
            log_once(
                _LOGGER,
                "surface_separator:estimate_reference_direction",
                logging.DEBUG,
                "Failed to estimate reference direction; falling back to +Z",
                exc_info=True,
            )

        # 기본값: Z축
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    
    def separate_by_selection(self, mesh: MeshData,
                              selected_face_indices: np.ndarray) -> Tuple[MeshData, MeshData | None]:
        """
        선택된 면으로 수동 분리
        
        Args:
            mesh: 입력 메쉬
            selected_face_indices: 선택된 면 인덱스 (첫 번째 그룹)
            
        Returns:
            (선택된 메쉬, 나머지 메쉬)
        """
        selected = np.asarray(selected_face_indices, dtype=np.int32).reshape(-1)
        n_faces = int(getattr(mesh, "n_faces", 0) or 0)
        if n_faces <= 0:
            return mesh, None

        valid = (selected >= 0) & (selected < n_faces)
        selected_valid = selected[valid]

        mask = np.ones((n_faces,), dtype=bool)
        if selected_valid.size:
            mask[selected_valid] = False
        remaining_indices = np.where(mask)[0].astype(np.int32, copy=False)
        
        selected_mesh = mesh.extract_submesh(selected_valid)
        remaining_mesh = mesh.extract_submesh(remaining_indices) if remaining_indices.size > 0 else None
        
        return selected_mesh, remaining_mesh
    
    def separate_by_curvature(self, mesh: MeshData,
                              curvature_threshold: float = 0.0) -> SeparatedSurfaces:
        """
        곡률 기반 분리 (볼록/오목)
        
        Args:
            mesh: 입력 메쉬
            curvature_threshold: 곡률 임계값
            
        Returns:
            SeparatedSurfaces: 분리된 표면들
        """
        # 정점별 곡률 추정
        curvatures = self._estimate_vertex_curvature(mesh)
        
        # 면별 평균 곡률
        face_curvatures = np.zeros(mesh.n_faces)
        for i, face in enumerate(mesh.faces):
            face_curvatures[i] = curvatures[face].mean()
        
        # 분류: 양의 곡률 (볼록) = 외면, 음의 곡률 (오목) = 내면
        outer_mask = face_curvatures >= curvature_threshold
        inner_mask = ~outer_mask
        
        outer_indices = np.where(outer_mask)[0]
        inner_indices = np.where(inner_mask)[0]
        
        outer_surface = mesh.extract_submesh(outer_indices) if len(outer_indices) > 0 else None
        inner_surface = mesh.extract_submesh(inner_indices) if len(inner_indices) > 0 else None
        
        return SeparatedSurfaces(
            inner_surface=inner_surface,
            outer_surface=outer_surface,
            inner_face_indices=inner_indices,
            outer_face_indices=outer_indices
        )
    
    def _estimate_vertex_curvature(self, mesh: MeshData) -> np.ndarray:
        """정점별 곡률 추정 (간단한 방법)"""
        mesh.compute_normals()
        normals = mesh.normals
        if normals is None:
            raise RuntimeError("Mesh normals are required for curvature estimation")
        
        n = mesh.n_vertices
        curvatures = np.zeros(n)
        
        # 인접 정점 정보
        adjacency = [set() for _ in range(n)]
        for face in mesh.faces:
            for i in range(3):
                adjacency[face[i]].add(face[(i+1) % 3])
                adjacency[face[i]].add(face[(i+2) % 3])
        
        # 각 정점의 곡률 = 이웃 법선과의 평균 각도 차이
        for i in range(n):
            neighbors = list(adjacency[i])
            if len(neighbors) == 0:
                continue
            
            normal_i = normals[i]
            
            # 이웃 법선과의 각도 차이
            angle_diffs = []
            for j in neighbors:
                normal_j = normals[j]
                dot = np.clip(np.dot(normal_i, normal_j), -1, 1)
                angle = np.arccos(dot)
                angle_diffs.append(angle)
            
            curvatures[i] = np.mean(angle_diffs)
        
        return curvatures
    
    def separate_connected_components(self, mesh: MeshData) -> List[MeshData]:
        """
        연결 컴포넌트별로 분리
        
        Args:
            mesh: 입력 메쉬
            
        Returns:
            분리된 메쉬 목록
        """
        m = mesh.n_faces
        
        # 면 인접성 그래프 구성
        face_adjacency = [set() for _ in range(m)]
        
        # 엣지 -> 면 매핑
        edge_to_faces = {}
        for fi, face in enumerate(mesh.faces):
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i+1) % 3]]))
                if edge in edge_to_faces:
                    # 인접 면 연결
                    other_face = edge_to_faces[edge]
                    face_adjacency[fi].add(other_face)
                    face_adjacency[other_face].add(fi)
                else:
                    edge_to_faces[edge] = fi
        
        # BFS로 연결 컴포넌트 찾기
        visited = np.zeros(m, dtype=bool)
        components = []
        
        from collections import deque

        for start_face in range(m):
            if visited[start_face]:
                continue
            
            # BFS
            component = []
            queue = deque([int(start_face)])
            visited[start_face] = True
            
            while queue:
                face = int(queue.popleft())
                component.append(face)
                
                for neighbor in face_adjacency[face]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(int(neighbor))
            
            components.append(np.array(component))
        
        # 각 컴포넌트를 서브메쉬로 추출
        return [mesh.extract_submesh(comp) for comp in components]


# 테스트용
if __name__ == '__main__':
    print("Surface Separator module loaded successfully")
    print("Use: separator = SurfaceSeparator(); result = separator.auto_detect_surfaces(mesh)")
