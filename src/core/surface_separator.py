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
            r1 = None
            # 1) View-first split:
            # treat the shell as two opposite sides and propagate labels over topology.
            # This is more stable on folded/occluded surfaces where simple camera visibility
            # (or a single global cylinder fit) can be misleading.
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
                migu_n = int(getattr(getattr(r1, "migu_face_indices", None), "size", 0) or 0)
                meta1 = getattr(r1, "meta", {}) or {}
                seed_outer_n = int(meta1.get("seed_outer_count", 0) or 0)
                seed_inner_n = int(meta1.get("seed_inner_count", 0) or 0)
                seed_frac = float(seed_outer_n + seed_inner_n) / float(max(1, n_faces))
                migu_frac = float(migu_n) / float(max(1, n_faces))
                views_confident = bool(
                    outer_n > max(10, int(0.01 * n_faces))
                    and inner_n > max(10, int(0.01 * n_faces))
                    and seed_frac >= 0.10
                    and migu_frac <= 0.20
                )
                if views_confident:
                    return r1
            except Exception:
                log_once(
                    _LOGGER,
                    "surface_separator:auto:views_failed",
                    logging.DEBUG,
                    "Auto surface separation: view path failed; falling back",
                    exc_info=True,
                )

            # 2) Fallback for tile-like meshes: cylindrical thickness split.
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

            # If views succeeded but did not pass confidence gates, still prefer it over normals.
            # (Normals-only split can be unstable when winding/curvature is noisy.)
            if r1 is not None:
                return r1

            # 3) Final fallback: global-normal split.
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

    def infer_migu_from_outer_inner(
        self,
        mesh: MeshData,
        *,
        outer_face_indices: set[int] | list[int] | np.ndarray,
        inner_face_indices: set[int] | list[int] | np.ndarray,
        hops: int = 1,
        vertex_dom_ratio: float = 1.20,
        side_absdot_max: float = 0.45,
        max_ratio: float = 0.35,
        reference_direction: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        현재 outer/inner 라벨 경계로부터 migu(측벽/두께 면) 후보를 추론합니다.

        Notes:
        - outer/inner 사이에 "미분류(unknown) 면"이 남아 있는 경우 unknown 우선으로 추론합니다.
        - unknown이 없으면 outer/inner 경계 띠를 얇게 절삭하는 방식으로 후보를 만듭니다.
        """

        faces = np.asarray(getattr(mesh, "faces", np.zeros((0, 3))), dtype=np.int32)
        if faces.ndim != 2 or faces.shape[0] <= 0 or faces.shape[1] < 3:
            return np.zeros((0,), dtype=np.int32), {"method": "boundary_bridge", "reason": "invalid_faces"}
        faces = faces[:, :3]
        n_faces = int(faces.shape[0])
        if n_faces <= 0:
            return np.zeros((0,), dtype=np.int32), {"method": "boundary_bridge", "reason": "empty_faces"}

        # Memory guard for very large meshes.
        if n_faces > 2_000_000:
            return np.zeros((0,), dtype=np.int32), {"method": "boundary_bridge", "reason": "too_large"}

        def _to_face_index_array(ids: set[int] | list[int] | np.ndarray) -> np.ndarray:
            if ids is None:
                return np.zeros((0,), dtype=np.int32)
            try:
                arr = np.asarray(ids, dtype=np.int64).reshape(-1)
            except Exception:
                try:
                    arr = np.asarray(list(ids), dtype=np.int64).reshape(-1)
                except Exception:
                    arr = np.zeros((0,), dtype=np.int64)
            if arr.size <= 0:
                return np.zeros((0,), dtype=np.int32)
            valid = np.isfinite(arr)
            if not bool(np.any(valid)):
                return np.zeros((0,), dtype=np.int32)
            arr = arr[valid]
            arr = arr[(arr >= 0) & (arr < n_faces)]
            if arr.size <= 0:
                return np.zeros((0,), dtype=np.int32)
            return np.unique(arr.astype(np.int32, copy=False))

        outer_idx = _to_face_index_array(outer_face_indices)
        inner_idx = _to_face_index_array(inner_face_indices)
        if outer_idx.size <= 0 or inner_idx.size <= 0:
            return np.zeros((0,), dtype=np.int32), {"method": "boundary_bridge", "reason": "missing_labels"}

        outer_mask = np.zeros((n_faces,), dtype=bool)
        inner_mask = np.zeros((n_faces,), dtype=bool)
        outer_mask[outer_idx] = True
        inner_mask[inner_idx] = True

        # Resolve overlap: outer wins for stability.
        overlap = outer_mask & inner_mask
        if bool(np.any(overlap)):
            inner_mask[overlap] = False
        unknown_mask = ~(outer_mask | inner_mask)
        has_unknown = bool(np.any(unknown_mask))

        # Build adjacent face pairs from shared edges.
        e01 = faces[:, [0, 1]]
        e12 = faces[:, [1, 2]]
        e20 = faces[:, [2, 0]]
        edges = np.vstack([e01, e12, e20]).astype(np.int32, copy=False)
        edges.sort(axis=1)
        # Edge rows are stacked as [all e01, all e12, all e20], so face ids must be tiled (not repeated).
        face_ids = np.tile(np.arange(n_faces, dtype=np.int32), 3)

        order = np.lexsort((edges[:, 1], edges[:, 0]))
        edges_s = edges[order]
        face_s = face_ids[order]

        is_new = np.empty((edges_s.shape[0],), dtype=bool)
        is_new[0] = True
        is_new[1:] = np.any(edges_s[1:] != edges_s[:-1], axis=1)
        starts = np.flatnonzero(is_new).astype(np.int32, copy=False)
        counts = np.diff(np.append(starts, edges_s.shape[0])).astype(np.int32, copy=False)

        pair_mask = counts == 2
        if not bool(np.any(pair_mask)):
            return np.zeros((0,), dtype=np.int32), {"method": "boundary_bridge", "reason": "no_adjacency_pairs"}

        a = face_s[starts[pair_mask]]
        b = face_s[starts[pair_mask] + 1]
        hops_i = int(max(0, min(int(hops), 3)))

        # Direct outer<->inner touching faces (strip carving fallback).
        cross = (outer_mask[a] & inner_mask[b]) | (inner_mask[a] & outer_mask[b])
        seed_cross = np.zeros((n_faces,), dtype=bool)
        if bool(np.any(cross)):
            seed_cross[a[cross]] = True
            seed_cross[b[cross]] = True

        # Bridge faces: adjacent to both outer and inner.
        # This catches true migu faces when they sit between outer and inner.
        adj_outer = np.zeros((n_faces,), dtype=bool)
        adj_inner = np.zeros((n_faces,), dtype=bool)
        adj_outer[a] |= outer_mask[b]
        adj_outer[b] |= outer_mask[a]
        adj_inner[a] |= inner_mask[b]
        adj_inner[b] |= inner_mask[a]
        seed_bridge = adj_outer & adj_inner

        # Unknown-bridge propagation:
        # expand unknown faces connected to outer and inner from both sides, then intersect.
        seed_unknown = np.zeros((n_faces,), dtype=bool)
        touch_outer = np.zeros((n_faces,), dtype=bool)
        touch_inner = np.zeros((n_faces,), dtype=bool)
        ao = unknown_mask[a] & outer_mask[b]
        bo = unknown_mask[b] & outer_mask[a]
        ai = unknown_mask[a] & inner_mask[b]
        bi = unknown_mask[b] & inner_mask[a]
        if bool(np.any(ao)):
            touch_outer[a[ao]] = True
        if bool(np.any(bo)):
            touch_outer[b[bo]] = True
        if bool(np.any(ai)):
            touch_inner[a[ai]] = True
        if bool(np.any(bi)):
            touch_inner[b[bi]] = True

        uu = unknown_mask[a] & unknown_mask[b]
        ua = a[uu]
        ub = b[uu]
        if ua.size > 0:
            reach_outer = touch_outer.copy()
            reach_inner = touch_inner.copy()
            bridge_steps = int(max(1, min(hops_i + 1, 4)))
            for _ in range(bridge_steps):
                near_o = reach_outer[ua] | reach_outer[ub]
                if bool(np.any(near_o)):
                    reach_outer[ua[near_o]] = True
                    reach_outer[ub[near_o]] = True
                near_i = reach_inner[ua] | reach_inner[ub]
                if bool(np.any(near_i)):
                    reach_inner[ua[near_i]] = True
                    reach_inner[ub[near_i]] = True
            seed_unknown = unknown_mask & reach_outer & reach_inner

        if bool(np.any(seed_unknown)):
            seed = seed_unknown | (seed_cross & unknown_mask)
            mode = "unknown_bridge"
        else:
            seed = seed_bridge | seed_cross
            mode = "boundary_strip"

        if not bool(np.any(seed)):
            return np.zeros((0,), dtype=np.int32), {"method": "boundary_bridge", "reason": "no_boundary_seed", "mode": mode}

        belt = seed.copy()
        for _ in range(hops_i):
            near = belt[a] | belt[b]
            if not bool(np.any(near)):
                break
            belt[a[near]] = True
            belt[b[near]] = True

        # Vertex-dominance bridge prior.
        bridge_dom = np.zeros((n_faces,), dtype=bool)
        try:
            n_vertices = int(np.max(faces) + 1) if faces.size > 0 else 0
            if n_vertices > 0:
                v_outer = np.bincount(faces[outer_mask].ravel(), minlength=n_vertices).astype(np.float64, copy=False)
                v_inner = np.bincount(faces[inner_mask].ravel(), minlength=n_vertices).astype(np.float64, copy=False)
                dom_ratio = float(max(1.0, float(vertex_dom_ratio)))
                dom_outer = (v_outer > (dom_ratio * v_inner)) & (v_outer > 0.0)
                dom_inner = (v_inner > (dom_ratio * v_outer)) & (v_inner > 0.0)
                bridge_dom = np.any(dom_outer[faces], axis=1) & np.any(dom_inner[faces], axis=1)
        except Exception:
            bridge_dom = np.zeros((n_faces,), dtype=bool)
        if has_unknown:
            bridge_dom = bridge_dom & unknown_mask
        else:
            # In fully-labeled meshes this prior can over-capture; rely on boundary strip instead.
            bridge_dom = np.zeros((n_faces,), dtype=bool)

        # Side-wall prior: faces near-perpendicular to thickness axis.
        side = np.ones((n_faces,), dtype=bool)
        try:
            d = np.asarray(reference_direction, dtype=np.float64).reshape(-1) if reference_direction is not None else self._estimate_reference_direction(mesh)
            d = np.asarray(d, dtype=np.float64).reshape(-1)
            if d.size < 3 or not np.isfinite(d[:3]).all():
                d = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            d = d[:3]
            d = d / (float(np.linalg.norm(d)) + 1e-12)

            if getattr(mesh, "face_normals", None) is None:
                mesh.compute_normals(compute_vertex_normals=False)
            fn = np.asarray(getattr(mesh, "face_normals", None), dtype=np.float64)
            if fn.ndim == 2 and fn.shape[0] == n_faces and fn.shape[1] >= 3:
                dotn = np.abs(fn[:, :3] @ d.reshape(3,))
                thr = float(np.clip(float(side_absdot_max), 0.0, 1.0))
                side = np.isfinite(dotn) & (dotn <= thr)
        except Exception:
            side = np.ones((n_faces,), dtype=bool)

        if has_unknown:
            keep = ((seed | bridge_dom | (belt & side)) & unknown_mask)
            if not bool(np.any(keep)):
                # Fallback for fully-labeled or noisy cases.
                keep = seed | (belt & side)
        else:
            keep = seed | (belt & side)

        # Prevent over-capture on noisy meshes.
        ratio_max = float(np.clip(float(max_ratio), 0.05, 0.80))
        frac = float(np.mean(keep)) if keep.size else 0.0
        clamped = False
        if frac > ratio_max:
            clamped = True
            keep = seed.copy()
            if has_unknown:
                keep = keep & unknown_mask
                if not bool(np.any(keep)):
                    keep = seed.copy()

        idx = np.where(keep)[0].astype(np.int32, copy=False)
        meta = {
            "method": "boundary_bridge",
            "mode": mode,
            "hops": int(hops_i),
            "seed_count": int(np.count_nonzero(seed)),
            "unknown_count": int(np.count_nonzero(unknown_mask)),
            "ratio": float(frac),
            "max_ratio": float(ratio_max),
            "clamped": bool(clamped),
        }
        return idx, meta

    def assist_outer_inner_from_seeds(
        self,
        mesh: MeshData,
        *,
        outer_face_indices: set[int] | list[int] | np.ndarray,
        inner_face_indices: set[int] | list[int] | np.ndarray,
        migu_face_indices: set[int] | list[int] | np.ndarray | None = None,
        method: str = "views",
        conservative: bool = True,
        min_seed: int = 24,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """
        사용자 수동 라벨(outer/inner)을 씨드로 사용해 미분류 면만 보조 분류합니다.

        원칙:
        - 기존 수동 라벨은 유지(덮어쓰기 없음)
        - migu 라벨은 제외
        - auto 결과는 씨드와 일치하도록 orientation 정렬 후, unknown에만 적용
        - conservative=True면 auto + 두께축(t) 규칙이 동시에 동의하는 면만 채움
        """
        faces = np.asarray(getattr(mesh, "faces", np.zeros((0, 3))), dtype=np.int32)
        vertices = np.asarray(getattr(mesh, "vertices", np.zeros((0, 3))), dtype=np.float64)
        if (
            vertices.ndim != 2
            or vertices.shape[0] <= 0
            or faces.ndim != 2
            or faces.shape[0] <= 0
            or faces.shape[1] < 3
        ):
            empty = np.zeros((0,), dtype=np.int32)
            return empty, empty, {"status": "invalid_mesh", "reason": "empty_or_invalid"}

        faces = faces[:, :3]
        n_faces = int(faces.shape[0])
        min_seed_i = int(max(1, int(min_seed)))

        def _to_face_index_array(ids: set[int] | list[int] | np.ndarray | None) -> np.ndarray:
            if ids is None:
                return np.zeros((0,), dtype=np.int32)
            try:
                arr = np.asarray(ids, dtype=np.int64).reshape(-1)
            except Exception:
                try:
                    arr = np.asarray(list(ids), dtype=np.int64).reshape(-1)
                except Exception:
                    arr = np.zeros((0,), dtype=np.int64)
            if arr.size <= 0:
                return np.zeros((0,), dtype=np.int32)
            valid = np.isfinite(arr)
            if not bool(np.any(valid)):
                return np.zeros((0,), dtype=np.int32)
            arr = arr[valid]
            arr = arr[(arr >= 0) & (arr < n_faces)]
            if arr.size <= 0:
                return np.zeros((0,), dtype=np.int32)
            return np.unique(arr.astype(np.int32, copy=False))

        outer_seed_idx = _to_face_index_array(outer_face_indices)
        inner_seed_idx = _to_face_index_array(inner_face_indices)
        migu_idx = _to_face_index_array(migu_face_indices)

        migu_mask = np.zeros((n_faces,), dtype=bool)
        if migu_idx.size > 0:
            migu_mask[migu_idx] = True

        outer_seed_mask = np.zeros((n_faces,), dtype=bool)
        inner_seed_mask = np.zeros((n_faces,), dtype=bool)
        if outer_seed_idx.size > 0:
            outer_seed_mask[outer_seed_idx] = True
        if inner_seed_idx.size > 0:
            inner_seed_mask[inner_seed_idx] = True

        # Keep labels exclusive and migu-safe (outer wins in overlap).
        outer_seed_mask = outer_seed_mask & ~migu_mask
        inner_seed_mask = inner_seed_mask & ~migu_mask
        overlap_seed = outer_seed_mask & inner_seed_mask
        if bool(np.any(overlap_seed)):
            inner_seed_mask[overlap_seed] = False

        seed_outer_n = int(np.count_nonzero(outer_seed_mask))
        seed_inner_n = int(np.count_nonzero(inner_seed_mask))
        if seed_outer_n < min_seed_i or seed_inner_n < min_seed_i:
            return (
                np.where(outer_seed_mask)[0].astype(np.int32, copy=False),
                np.where(inner_seed_mask)[0].astype(np.int32, copy=False),
                {
                    "status": "missing_seeds",
                    "seed_outer_count": seed_outer_n,
                    "seed_inner_count": seed_inner_n,
                    "min_seed_required": min_seed_i,
                },
            )

        # 1) Auto prediction (views/cylinder/auto) for global shape prior.
        auto_method = str(method or "views").strip().lower()
        if auto_method not in {"views", "view", "visible", "auto", "best", "smart", "cylinder", "cyl", "tile"}:
            auto_method = "views"
        try:
            pred = self.auto_detect_surfaces(mesh, method=auto_method, return_submeshes=False)
        except Exception as e:
            return (
                np.where(outer_seed_mask)[0].astype(np.int32, copy=False),
                np.where(inner_seed_mask)[0].astype(np.int32, copy=False),
                {
                    "status": "auto_failed",
                    "method": auto_method,
                    "error": f"{type(e).__name__}: {e}",
                },
            )

        pred_outer = np.zeros((n_faces,), dtype=bool)
        pred_inner = np.zeros((n_faces,), dtype=bool)
        try:
            po = np.asarray(getattr(pred, "outer_face_indices", np.zeros((0,), dtype=np.int32)), dtype=np.int32).reshape(-1)
        except Exception:
            po = np.zeros((0,), dtype=np.int32)
        try:
            pi = np.asarray(getattr(pred, "inner_face_indices", np.zeros((0,), dtype=np.int32)), dtype=np.int32).reshape(-1)
        except Exception:
            pi = np.zeros((0,), dtype=np.int32)
        if po.size > 0:
            valid = (po >= 0) & (po < n_faces)
            pred_outer[po[valid]] = True
        if pi.size > 0:
            valid = (pi >= 0) & (pi < n_faces)
            pred_inner[pi[valid]] = True

        pred_outer = pred_outer & ~migu_mask
        pred_inner = pred_inner & ~migu_mask

        # 2) Align orientation to user seeds (direct vs swapped).
        direct_hits = int(np.count_nonzero(pred_outer & outer_seed_mask) + np.count_nonzero(pred_inner & inner_seed_mask))
        swapped_hits = int(np.count_nonzero(pred_outer & inner_seed_mask) + np.count_nonzero(pred_inner & outer_seed_mask))
        swapped = bool(swapped_hits > direct_hits)
        mapped_outer = pred_inner if swapped else pred_outer
        mapped_inner = pred_outer if swapped else pred_inner
        mapping = "swapped" if swapped else "direct"

        unknown_mask = ~(outer_seed_mask | inner_seed_mask | migu_mask)

        # 3) Conservative consensus: auto result AND thickness-axis rule.
        rule_outer = np.ones((n_faces,), dtype=bool)
        rule_inner = np.ones((n_faces,), dtype=bool)
        rule_ok = False
        rule_sep_ratio = 0.0
        try:
            if bool(conservative):
                d = np.asarray(self._estimate_reference_direction(mesh), dtype=np.float64).reshape(-1)
                if d.size >= 3 and np.isfinite(d[:3]).all():
                    d = d[:3]
                    d = d / (float(np.linalg.norm(d)) + 1e-12)
                    t = np.empty((n_faces,), dtype=np.float32)
                    try:
                        chunk = int(getattr(mesh, "_assist_axis_chunk_faces", 250000) or 250000)
                    except Exception:
                        chunk = 250000
                    chunk = max(20000, min(chunk, 500000))
                    for start in range(0, n_faces, chunk):
                        end = min(n_faces, start + chunk)
                        f = np.asarray(faces[start:end, :3], dtype=np.int32)
                        if f.size == 0:
                            continue
                        cent = (vertices[f[:, 0]] + vertices[f[:, 1]] + vertices[f[:, 2]]) / 3.0
                        t[start:end] = (cent @ d.reshape(3,)).astype(np.float32, copy=False)

                    t64 = t.astype(np.float64, copy=False)
                    to = t64[outer_seed_mask]
                    ti = t64[inner_seed_mask]
                    if to.size > 0 and ti.size > 0:
                        mo = float(np.nanmedian(to))
                        mi = float(np.nanmedian(ti))
                        mid = 0.5 * (mo + mi)
                        gap = float(abs(mo - mi))
                        mad_o = float(np.nanmedian(np.abs(to - mo))) if to.size > 0 else 0.0
                        mad_i = float(np.nanmedian(np.abs(ti - mi))) if ti.size > 0 else 0.0
                        spread = float(max(mad_o, mad_i, 1e-12))
                        rule_sep_ratio = float(gap / spread) if spread > 0.0 else 0.0
                        try:
                            raw = getattr(mesh, "_assist_rule_min_sep_ratio", None)
                            min_sep_ratio = 1.5 if raw is None else float(raw)
                        except Exception:
                            min_sep_ratio = 1.5
                        if np.isfinite(rule_sep_ratio) and rule_sep_ratio >= float(max(0.0, min_sep_ratio)):
                            dead = float(max(0.0, 0.05 * gap))
                            if mo >= mi:
                                rule_outer = np.isfinite(t64) & (t64 >= (mid + dead))
                                rule_inner = np.isfinite(t64) & (t64 <= (mid - dead))
                            else:
                                rule_outer = np.isfinite(t64) & (t64 <= (mid - dead))
                                rule_inner = np.isfinite(t64) & (t64 >= (mid + dead))
                            rule_ok = True
        except Exception:
            rule_ok = False

        if bool(conservative) and rule_ok:
            # Conservative-but-usable:
            # keep mapped prediction unless thickness rule strongly contradicts it.
            # (strict consensus was too sparse for practical manual assist.)
            veto_outer = unknown_mask & mapped_outer & rule_inner
            veto_inner = unknown_mask & mapped_inner & rule_outer
            add_outer = unknown_mask & mapped_outer & ~veto_outer
            add_inner = unknown_mask & mapped_inner & ~veto_inner
            assist_mode = "mapped_with_veto"
        else:
            add_outer = unknown_mask & mapped_outer
            add_inner = unknown_mask & mapped_inner
            assist_mode = "mapped_only"

        conflict = add_outer & add_inner
        if bool(np.any(conflict)):
            add_outer[conflict] = False
            add_inner[conflict] = False

        outer_final = (outer_seed_mask | add_outer) & ~migu_mask
        inner_final = (inner_seed_mask | add_inner) & ~migu_mask
        # Keep final labels exclusive.
        overlap_final = outer_final & inner_final
        if bool(np.any(overlap_final)):
            inner_final[overlap_final] = False

        unresolved = unknown_mask & ~(add_outer | add_inner)

        outer_idx = np.where(outer_final)[0].astype(np.int32, copy=False)
        inner_idx = np.where(inner_final)[0].astype(np.int32, copy=False)
        unresolved_count = int(np.count_nonzero(unresolved))
        try:
            raw_keep = getattr(mesh, "_assist_unresolved_keep_max", None)
            unresolved_keep_max = 800_000 if raw_keep is None else int(raw_keep)
        except Exception:
            unresolved_keep_max = 800_000
        unresolved_keep_max = max(0, int(unresolved_keep_max))
        unresolved_idx = None
        unresolved_truncated = False
        if unresolved_count > 0 and unresolved_keep_max > 0:
            if unresolved_count <= unresolved_keep_max:
                unresolved_idx = np.where(unresolved)[0].astype(np.int32, copy=False)
            else:
                unresolved_truncated = True

        return outer_idx, inner_idx, {
            "status": "ok",
            "method": "assist_seeded",
            "auto_method": auto_method,
            "auto_mapping": mapping,
            "assist_mode": assist_mode,
            "conservative": bool(conservative),
            "seed_outer_count": seed_outer_n,
            "seed_inner_count": seed_inner_n,
            "added_outer_count": int(np.count_nonzero(add_outer)),
            "added_inner_count": int(np.count_nonzero(add_inner)),
            "unknown_count": int(np.count_nonzero(unknown_mask)),
            "unresolved_count": int(unresolved_count),
            "unresolved_indices": unresolved_idx,
            "unresolved_truncated": bool(unresolved_truncated),
            "migu_count": int(np.count_nonzero(migu_mask)),
            "direct_hits": int(direct_hits),
            "swapped_hits": int(swapped_hits),
            "rule_used": bool(rule_ok),
            "rule_sep_ratio": float(rule_sep_ratio),
        }

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

        # Visibility robustness:
        # sample front depth from a local neighborhood (default: 3x3) to reduce
        # pixel-quantization jitter when model orientation changes.
        try:
            raw_vis_nbhd = getattr(mesh, "_views_visibility_neighborhood", None)
            if raw_vis_nbhd is None:
                vis_nbhd = 2 if n_faces >= 1_000_000 else 1
            else:
                vis_nbhd = int(raw_vis_nbhd)
        except Exception:
            vis_nbhd = 1
        vis_nbhd = max(0, min(vis_nbhd, 2))

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
                if vis_nbhd > 0:
                    h_max = int(img_h - 1)
                    w_max = int(img_w - 1)
                    for oy in range(-vis_nbhd, vis_nbhd + 1):
                        yy = np.clip(py + oy, 0, h_max).astype(np.int32, copy=False)
                        for ox in range(-vis_nbhd, vis_nbhd + 1):
                            if ox == 0 and oy == 0:
                                continue
                            xx = np.clip(px + ox, 0, w_max).astype(np.int32, copy=False)
                            front_p = np.maximum(front_p, depth_plus[yy, xx].astype(np.float32, copy=False))
                            front_m = np.maximum(front_m, depth_minus[yy, xx].astype(np.float32, copy=False))
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

        # Normal prior for migu detection (side walls)
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

        migu_mask = None
        if isinstance(dotn, np.ndarray) and dotn.shape == (n_faces,):
            # Side walls are near-perpendicular to the thickness axis.
            try:
                raw = getattr(mesh, "_views_migu_absdot_max", None)
                migu_absdot_max = 0.35 if raw is None else float(raw)
            except Exception:
                migu_absdot_max = 0.35
            migu_absdot_max = float(np.clip(migu_absdot_max, 0.0, 1.0))
            mm = np.isfinite(dotn) & (np.abs(dotn) <= float(migu_absdot_max))
            # Guard: if axis guess is bad, this can swallow too much; disable in that case.
            try:
                frac = float(np.mean(mm)) if mm.size else 0.0
            except Exception:
                frac = 0.0
            try:
                raw = getattr(mesh, "_views_migu_max_frac", None)
                migu_max_frac = 0.40 if raw is None else float(raw)
            except Exception:
                migu_max_frac = 0.40
            migu_max_frac = float(np.clip(migu_max_frac, 0.05, 0.95))
            if frac > float(migu_max_frac):
                mm = np.zeros((n_faces,), dtype=bool)
            migu_mask = mm

        non_migu = ~migu_mask if isinstance(migu_mask, np.ndarray) else np.ones((n_faces,), dtype=bool)
        outer_seed = seed_outer & non_migu
        inner_seed = seed_inner & non_migu

        outer_mask = None
        inner_mask = None
        topology_used = False
        topology_mode = "fallback"

        # Topology assignment:
        # treat outer/inner as two opposite shell sides, then assign unlabeled faces by
        # connectivity components on the non-migu graph. This avoids mislabeling occluded
        # folds by normal sign alone.
        try:
            use_topology = bool(getattr(mesh, "_views_use_topology_assignment", True))
        except Exception:
            use_topology = True
        try:
            topo_max_faces = int(getattr(mesh, "_views_topology_max_faces", 1_200_000) or 1_200_000)
        except Exception:
            topo_max_faces = 1_200_000

        if (
            use_topology
            and n_faces <= max(10_000, int(topo_max_faces))
            and bool(np.any(outer_seed))
            and bool(np.any(inner_seed))
        ):
            try:
                from scipy import sparse as _sparse
                from scipy.sparse import csgraph as _csgraph

                e01 = faces[:, [0, 1]]
                e12 = faces[:, [1, 2]]
                e20 = faces[:, [2, 0]]
                edges = np.vstack([e01, e12, e20]).astype(np.int32, copy=False)
                edges.sort(axis=1)
                face_ids = np.tile(np.arange(n_faces, dtype=np.int32), 3)

                order = np.lexsort((edges[:, 1], edges[:, 0]))
                edges_s = edges[order]
                face_s = face_ids[order]

                is_new = np.empty((edges_s.shape[0],), dtype=bool)
                is_new[0] = True
                is_new[1:] = np.any(edges_s[1:] != edges_s[:-1], axis=1)
                starts = np.flatnonzero(is_new).astype(np.int32, copy=False)
                counts = np.diff(np.append(starts, edges_s.shape[0])).astype(np.int32, copy=False)
                pair_mask = counts == 2

                a = face_s[starts[pair_mask]]
                b = face_s[starts[pair_mask] + 1]
                keep_pair = non_migu[a] & non_migu[b]
                a = a[keep_pair]
                b = b[keep_pair]

                if a.size > 0:
                    rows = np.concatenate([a, b]).astype(np.int32, copy=False)
                    cols = np.concatenate([b, a]).astype(np.int32, copy=False)
                    data = np.ones((rows.size,), dtype=np.uint8)
                    graph = _sparse.csr_matrix((data, (rows, cols)), shape=(n_faces, n_faces))
                    use_geodesic = False
                    try:
                        topo_geo_max_faces = int(getattr(mesh, "_views_topology_geodesic_max_faces", 400_000) or 400_000)
                    except Exception:
                        topo_geo_max_faces = 400_000

                    # Preferred: shortest-path assignment from outer/inner seed sets.
                    # This preserves local "two-sided shell" structure even when both sides
                    # are connected in one component (e.g. noisy normals or folds).
                    if n_faces <= max(20_000, int(topo_geo_max_faces)):
                        try:
                            src_outer = np.flatnonzero(outer_seed).astype(np.int32, copy=False)
                            src_inner = np.flatnonzero(inner_seed).astype(np.int32, copy=False)

                            # Guard: too many sources can make multi-source dijkstra expensive.
                            # Keep deterministic coverage via stride downsampling.
                            try:
                                max_src = int(getattr(mesh, "_views_topology_geodesic_max_sources", 4096) or 4096)
                            except Exception:
                                max_src = 4096
                            max_src = max(256, int(max_src))
                            if src_outer.size > max_src:
                                step = max(1, int(np.ceil(float(src_outer.size) / float(max_src))))
                                src_outer = src_outer[::step][:max_src]
                            if src_inner.size > max_src:
                                step = max(1, int(np.ceil(float(src_inner.size) / float(max_src))))
                                src_inner = src_inner[::step][:max_src]

                            if src_outer.size > 0 and src_inner.size > 0:
                                d_out = _csgraph.dijkstra(
                                    graph,
                                    directed=False,
                                    indices=src_outer,
                                    unweighted=True,
                                    min_only=True,
                                )
                                d_in = _csgraph.dijkstra(
                                    graph,
                                    directed=False,
                                    indices=src_inner,
                                    unweighted=True,
                                    min_only=True,
                                )

                                d_out = np.asarray(d_out, dtype=np.float64).reshape(-1)
                                d_in = np.asarray(d_in, dtype=np.float64).reshape(-1)
                                f_out = np.isfinite(d_out)
                                f_in = np.isfinite(d_in)

                                face_cls = np.zeros((n_faces,), dtype=np.int8)
                                only_o = non_migu & f_out & ~f_in
                                only_i = non_migu & f_in & ~f_out
                                face_cls[only_o] = 1
                                face_cls[only_i] = -1

                                both = non_migu & f_out & f_in
                                if bool(np.any(both)):
                                    both_idx = np.flatnonzero(both).astype(np.int32, copy=False)
                                    do = d_out[both_idx]
                                    di = d_in[both_idx]
                                    o_win = do < di
                                    i_win = di < do
                                    if bool(np.any(o_win)):
                                        face_cls[both_idx[o_win]] = 1
                                    if bool(np.any(i_win)):
                                        face_cls[both_idx[i_win]] = -1
                                    tie = ~(o_win | i_win)
                                    if bool(np.any(tie)):
                                        tie_idx = both_idx[tie]
                                        face_cls[tie_idx] = np.where(t[tie_idx] >= float(med), 1, -1).astype(np.int8, copy=False)

                                unresolved = non_migu & (face_cls == 0)
                                if bool(np.any(unresolved)):
                                    face_cls[unresolved] = np.where(t[unresolved] >= float(med), 1, -1).astype(np.int8, copy=False)

                                outer_mask = non_migu & (face_cls > 0)
                                inner_mask = non_migu & (face_cls <= 0)
                                topology_used = True
                                topology_mode = "geodesic"
                                use_geodesic = True
                        except Exception:
                            use_geodesic = False

                    if not use_geodesic:
                        comp_n, comp = _csgraph.connected_components(graph, directed=False, return_labels=True)

                        comp_outer = np.bincount(comp[outer_seed], minlength=comp_n).astype(np.int32, copy=False)
                        comp_inner = np.bincount(comp[inner_seed], minlength=comp_n).astype(np.int32, copy=False)

                        t64 = t.astype(np.float64, copy=False)
                        comp_cnt = np.bincount(comp[non_migu], minlength=comp_n).astype(np.int32, copy=False)
                        comp_sum_t = np.bincount(comp[non_migu], weights=t64[non_migu], minlength=comp_n).astype(np.float64, copy=False)
                        comp_mean_t = np.zeros((comp_n,), dtype=np.float64)
                        valid_comp = comp_cnt > 0
                        comp_mean_t[valid_comp] = comp_sum_t[valid_comp] / np.maximum(comp_cnt[valid_comp], 1)

                        comp_cls = np.zeros((comp_n,), dtype=np.int8)  # +1 outer, -1 inner
                        only_outer = (comp_outer > 0) & (comp_inner == 0)
                        only_inner = (comp_inner > 0) & (comp_outer == 0)
                        comp_cls[only_outer] = 1
                        comp_cls[only_inner] = -1

                        both = (comp_outer > 0) & (comp_inner > 0)
                        if bool(np.any(both)):
                            both_idx = np.flatnonzero(both).astype(np.int32, copy=False)
                            oc = comp_outer[both_idx]
                            ic = comp_inner[both_idx]
                            o_win = oc > ic
                            i_win = ic > oc
                            if bool(np.any(o_win)):
                                comp_cls[both_idx[o_win]] = 1
                            if bool(np.any(i_win)):
                                comp_cls[both_idx[i_win]] = -1
                            tie = ~(o_win | i_win)
                            if bool(np.any(tie)):
                                tie_idx = both_idx[tie]
                                comp_cls[tie_idx] = np.where(comp_mean_t[tie_idx] >= float(med), 1, -1).astype(np.int8, copy=False)

                        unassigned = comp_cls == 0
                        if bool(np.any(unassigned)):
                            comp_cls[unassigned] = np.where(comp_mean_t[unassigned] >= float(med), 1, -1).astype(np.int8, copy=False)

                        face_cls = comp_cls[comp]
                        outer_mask = non_migu & (face_cls > 0)
                        inner_mask = non_migu & (face_cls <= 0)
                        topology_used = True
                        topology_mode = "component"
            except Exception:
                topology_used = False
                topology_mode = "fallback"

        if outer_mask is None or inner_mask is None:
            ambiguous = non_migu & ~(outer_seed | inner_seed)
            try:
                fallback_use_normals = bool(getattr(mesh, "_views_fallback_use_normals", False))
            except Exception:
                fallback_use_normals = False
            if fallback_use_normals and isinstance(dotn, np.ndarray) and dotn.shape == (n_faces,):
                try:
                    raw = getattr(mesh, "_views_fallback_normal_conf_thr", None)
                    thr = 0.12 if raw is None else float(raw)
                except Exception:
                    thr = 0.12
                thr = float(np.clip(thr, 0.0, 1.0))
                confident = ambiguous & np.isfinite(dotn) & (np.abs(dotn) >= float(thr))
                assign_outer = (ambiguous & ~confident & (t >= float(med))) | (confident & (dotn >= 0.0))
            else:
                assign_outer = ambiguous & (t >= float(med))
            assign_inner = ambiguous & ~assign_outer
            outer_mask = outer_seed | assign_outer
            inner_mask = inner_seed | assign_inner
            topology_mode = "fallback"

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
                "visibility_neighborhood": int(vis_nbhd),
                "migu_absdot_max": float(migu_absdot_max) if "migu_absdot_max" in locals() else 0.35,
                "topology_assignment": bool(topology_used),
                "topology_mode": str(topology_mode),
                "seed_outer_count": int(np.count_nonzero(outer_seed)),
                "seed_inner_count": int(np.count_nonzero(inner_seed)),
                "fallback_use_normals": bool(locals().get("fallback_use_normals", False)),
            },
        )
    
    def _estimate_reference_direction(self, mesh: MeshData) -> np.ndarray:
        """
        기준 방향 자동 추정

        기와처럼 얇은 쉘/대칭 형상에서는 면적 가중 법선 합이 0에 가까워지기 쉬워서,
        먼저 PCA 축 후보를 만들고, 가능하면 면 법선 정렬 점수로 두께축 후보를 재선정합니다.
        실패 시 기존 방식으로 fallback 합니다.
        """
        try:
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            faces = np.asarray(mesh.faces, dtype=np.int32)
            if vertices.ndim != 2 or vertices.shape[0] == 0:
                return np.array([0.0, 0.0, 1.0], dtype=np.float64)
            if faces.ndim != 2 or faces.shape[0] == 0 or faces.shape[1] < 3:
                return np.array([0.0, 0.0, 1.0], dtype=np.float64)

            # 1) PCA 축 후보 구성
            pca_axes: list[np.ndarray] = []
            pca_ref = None
            try:
                v = np.asarray(vertices[:, :3], dtype=np.float64)
                v = v[np.isfinite(v).all(axis=1)]
                if v.shape[0] >= 8:
                    c = np.mean(v, axis=0)
                    x = v - c
                    cov = (x.T @ x) / float(max(1, x.shape[0] - 1))
                    w, vecs = np.linalg.eigh(cov)
                    order = np.argsort(w)
                    for i in [int(order[0]), int(order[1]), int(order[2])]:
                        a = np.asarray(vecs[:, i], dtype=np.float64).reshape(3)
                        n = float(np.linalg.norm(a))
                        if np.isfinite(n) and n > 1e-12:
                            pca_axes.append((a / n).astype(np.float64, copy=False))
                    if pca_axes:
                        pca_ref = np.asarray(pca_axes[0], dtype=np.float64).reshape(3)
            except Exception:
                pca_axes = []
                pca_ref = None

            # 2) Face normal 정렬 점수로 후보 축 재선정.
            # 얇은 쉘인데 축별 분산이 비슷한 경우(주름/굽힘), 최소분산 축이 두께축이 아닐 수 있음.
            ref = None
            try:
                use_normal_score = bool(getattr(mesh, "_refdir_use_normal_score", True))
            except Exception:
                use_normal_score = True
            if use_normal_score:
                try:
                    if getattr(mesh, "face_normals", None) is None:
                        mesh.compute_normals(compute_vertex_normals=False)
                    fn = np.asarray(getattr(mesh, "face_normals", None), dtype=np.float64)
                    if fn.ndim == 2 and fn.shape[0] == int(faces.shape[0]) and fn.shape[1] >= 3:
                        fn = fn[:, :3]
                        nrm = np.linalg.norm(fn, axis=1, keepdims=True)
                        fn = fn / (nrm + 1e-12)
                        candidates = list(pca_axes) if pca_axes else [
                            np.array([1.0, 0.0, 0.0], dtype=np.float64),
                            np.array([0.0, 1.0, 0.0], dtype=np.float64),
                            np.array([0.0, 0.0, 1.0], dtype=np.float64),
                        ]
                        best_axis = None
                        best_score = -np.inf
                        for cand in candidates:
                            a = np.asarray(cand, dtype=np.float64).reshape(3)
                            n = float(np.linalg.norm(a))
                            if not np.isfinite(n) or n <= 1e-12:
                                continue
                            a = a / n
                            dots = np.abs(fn @ a.reshape(3,))
                            dots = dots[np.isfinite(dots)]
                            if dots.size < 16:
                                continue
                            q75 = float(np.quantile(dots, 0.75))
                            med = float(np.median(dots))
                            score = q75 + 0.15 * med
                            if score > best_score:
                                best_score = score
                                best_axis = a.astype(np.float64, copy=False)
                        if best_axis is not None:
                            ref = best_axis
                except Exception:
                    ref = None

            if ref is None and pca_ref is not None:
                ref = pca_ref
            if isinstance(ref, np.ndarray) and ref.shape == (3,):
                # Prefer +Z-ish direction to avoid random flips.
                if float(ref[2]) < 0.0:
                    ref = -ref
                return ref.astype(np.float64, copy=False)

            # 3) Area-weighted normal sum fallback.
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
