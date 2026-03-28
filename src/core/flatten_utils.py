"""Shared utilities for flatten engines.

Utilities include axis normalization, cutoff helpers, circle fitting, seam handling,
mesh size guards and common UV helpers.
"""

from __future__ import annotations

import logging
import numpy as np

from .logging_utils import log_once
from .mesh_loader import MeshData

_LOGGER = logging.getLogger(__name__)


def _log_ignored_exception(context: str = "Ignored exception") -> None:
    try:
        _LOGGER.debug("%s", context, exc_info=True)
    except Exception:
        pass


def _positive_float(value, default: float, *, min_value: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(out) or out <= float(min_value):
        return float(default)
    return float(out)


def _coerce_explicit_axis_vector(choice) -> np.ndarray | None:
    try:
        arr = np.asarray(choice, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if arr.size < 3 or not np.isfinite(arr[:3]).all():
        return None
    vec = arr[:3].astype(np.float64, copy=True)
    nrm = float(np.linalg.norm(vec))
    if not np.isfinite(nrm) or nrm <= 1e-12:
        return None
    vec = vec / nrm
    pivot = int(np.argmax(np.abs(vec)))
    if float(vec[pivot]) < 0.0:
        vec = -vec
    return vec


def _normalize_cylinder_axis_choice(choice) -> str:
    if _coerce_explicit_axis_vector(choice) is not None:
        return "vector"
    c = str(choice or "").strip().lower()
    if c in {"x", "x축", "x축 기준", "x axis"}:
        return "x"
    if c in {"y", "y축", "y축 기준", "y axis"}:
        return "y"
    if c in {"z", "z축", "z축 기준", "z axis"}:
        return "z"
    if c in {"auto", "자동", "자동 감지", "automatic"}:
        return "auto"
    return "auto"


def _axis_unit_vector(axis) -> np.ndarray:
    explicit = _coerce_explicit_axis_vector(axis)
    if explicit is not None:
        return explicit
    a = _normalize_cylinder_axis_choice(axis)
    if a == "x":
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if a == "y":
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _pca_axes_3d(vertices: np.ndarray) -> np.ndarray:
    """Return 3 PCA axes as unit vectors (columns), ordered by descending variance."""
    v = np.asarray(vertices, dtype=np.float64)
    if v.ndim != 2 or v.shape[0] < 8 or v.shape[1] < 3:
        return np.eye(3, dtype=np.float64)

    v = v[:, :3]
    finite = np.isfinite(v).all(axis=1)
    v = v[finite]
    if v.shape[0] < 8:
        return np.eye(3, dtype=np.float64)

    c = np.mean(v, axis=0)
    x = v - c.reshape(1, 3)
    cov = (x.T @ x) / float(max(1, x.shape[0] - 1))

    try:
        w, vecs = np.linalg.eigh(cov)
        order = np.argsort(w)[::-1]
        vecs = vecs[:, order]
    except Exception:
        return np.eye(3, dtype=np.float64)

    out = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        a = vecs[:, i].astype(np.float64, copy=False).reshape(3)
        n = float(np.linalg.norm(a))
        if not np.isfinite(n) or n < 1e-12:
            a = np.zeros((3,), dtype=np.float64)
            a[i] = 1.0
            n = 1.0
        a = a / n
        pivot = int(np.argmax(np.abs(a)))
        if float(a[pivot]) < 0.0:
            a = -a
        out[:, i] = a
    return out


def _cylinder_axis_score(vertices: np.ndarray, axis: np.ndarray) -> float:
    """Lower is better: how cylindrical the vertex cloud is around axis."""
    v = np.asarray(vertices, dtype=np.float64)
    if v.ndim != 2 or v.shape[0] < 8 or v.shape[1] < 3:
        return float("inf")

    try:
        a = np.asarray(axis, dtype=np.float64).reshape(3)
    except Exception:
        return float("inf")

    n = float(np.linalg.norm(a))
    if not np.isfinite(n) or n < 1e-12:
        return float("inf")
    a = a / n

    t = v[:, :3] @ a.reshape(3,)
    perp = v[:, :3] - t[:, None] * a[None, :]
    center = perp.mean(axis=0)
    r = np.linalg.norm(perp - center.reshape(1, 3), axis=1)
    r = r[np.isfinite(r)]
    if r.size < 8:
        return float("inf")

    med = float(np.median(r))
    if not np.isfinite(med) or med < 1e-9:
        return float("inf")
    sd = float(np.std(r))
    if not np.isfinite(sd):
        return float("inf")
    return float(sd / med)


def _uv_extents_2d(uv: np.ndarray) -> np.ndarray:
    arr = np.asarray(uv, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return np.zeros((2,), dtype=np.float64)
    if arr.shape[1] < 2:
        return np.zeros((2,), dtype=np.float64)
    arr2 = arr[:, :2]
    finite = np.all(np.isfinite(arr2), axis=1)
    if not np.any(finite):
        return np.zeros((2,), dtype=np.float64)
    vv = arr2[finite]
    ext = vv.max(axis=0) - vv.min(axis=0)
    ext = np.asarray(ext, dtype=np.float64).reshape(2)
    ext = np.where(np.isfinite(ext), ext, 0.0)
    ext[ext < 0.0] = 0.0
    return ext


def _mesh_total_area_3d(mesh) -> float:
    vertices = np.asarray(getattr(mesh, "vertices", np.zeros((0, 3))), dtype=np.float64)
    faces = np.asarray(getattr(mesh, "faces", np.zeros((0, 3))), dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[0] == 0 or faces.ndim != 2 or faces.shape[0] == 0:
        return 0.0
    tri = vertices[faces[:, :3]]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    a = np.linalg.norm(np.cross(e1, e2), axis=1) * 0.5
    a = a[np.isfinite(a)]
    return float(a.sum()) if a.size else 0.0


def _mesh_total_area_2d(mesh, uv: np.ndarray) -> float:
    pts = np.asarray(uv, dtype=np.float64)
    faces = np.asarray(getattr(mesh, "faces", np.zeros((0, 3))), dtype=np.int32)
    if pts.ndim != 2 or pts.shape[0] == 0 or faces.ndim != 2 or faces.shape[0] == 0:
        return 0.0
    tri = pts[faces[:, :3], :2]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    a = np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]) * 0.5
    a = a[np.isfinite(a)]
    return float(a.sum()) if a.size else 0.0


def _apply_flatten_size_guard(
    mesh,
    uv: np.ndarray,
    *,
    meta: dict[str, object] | None = None,
) -> tuple[np.ndarray, dict[str, object], bool]:
    """Guard against pathological flatten size explosions.

    Returns (uv, out_meta, applied).
    """
    uv_in = np.asarray(uv, dtype=np.float64)
    uv_out = uv_in.copy()
    out_meta = dict(meta or {})

    try:
        uv_ext = _uv_extents_2d(uv_out)
        flat_w = float(uv_ext[0])
        flat_h = float(uv_ext[1])
        flat_max = float(max(flat_w, flat_h))
        flat_area_bbox = float(flat_w * flat_h)

        mesh_ext = np.asarray(getattr(mesh, "extents", np.zeros((3,), dtype=np.float64)), dtype=np.float64).reshape(-1)
        mesh_ext = mesh_ext[np.isfinite(mesh_ext) & (mesh_ext > 1e-9)]
        ref_span = float(np.max(mesh_ext)) if mesh_ext.size else 0.0

        dim_ratio_before = float(flat_max / ref_span) if ref_span > 1e-12 else 1.0

        area_3d = float(_mesh_total_area_3d(mesh))
        area_2d = float(_mesh_total_area_2d(mesh, uv_out))
        area_ratio_before: float | None
        if area_3d > 1e-12 and np.isfinite(area_2d) and area_2d >= 0.0:
            area_ratio_before = float(area_2d / area_3d)
        else:
            area_ratio_before = None

        warn_dim_ratio = _positive_float(getattr(mesh, "_flatten_size_warn_dim_ratio", 6.0), 6.0, min_value=1.0)
        hard_dim_ratio = _positive_float(getattr(mesh, "_flatten_size_hard_dim_ratio", 30.0), 30.0, min_value=1.0)
        target_dim_ratio = _positive_float(getattr(mesh, "_flatten_size_target_dim_ratio", 12.0), 12.0, min_value=1.0)
        if hard_dim_ratio < warn_dim_ratio:
            hard_dim_ratio = warn_dim_ratio
        if target_dim_ratio > hard_dim_ratio:
            target_dim_ratio = hard_dim_ratio

        warn_area_ratio = _positive_float(getattr(mesh, "_flatten_size_warn_area_ratio", 8.0), 8.0, min_value=1.0)
        hard_area_ratio = _positive_float(getattr(mesh, "_flatten_size_hard_area_ratio", 60.0), 60.0, min_value=1.0)
        target_area_ratio = _positive_float(getattr(mesh, "_flatten_size_target_area_ratio", 24.0), 24.0, min_value=1.0)
        if hard_area_ratio < warn_area_ratio:
            hard_area_ratio = warn_area_ratio
        if target_area_ratio > hard_area_ratio:
            target_area_ratio = hard_area_ratio

        warn_dim = bool(dim_ratio_before > warn_dim_ratio)
        hard_dim = bool(dim_ratio_before > hard_dim_ratio)

        warn_area = False
        hard_area = False
        if area_ratio_before is not None and np.isfinite(area_ratio_before):
            warn_area = bool(area_ratio_before > warn_area_ratio)
            hard_area = bool(area_ratio_before > hard_area_ratio)

        warning = bool(warn_dim or warn_area)
        severe = bool(hard_dim or hard_area)

        applied = False
        guard_scale = 1.0
        dim_ratio_after = dim_ratio_before
        area_ratio_after = area_ratio_before

        if severe:
            scale_candidates: list[float] = [1.0]
            if hard_dim and ref_span > 1e-12 and flat_max > 1e-12:
                scale_candidates.append(float((target_dim_ratio * ref_span) / flat_max))
            if hard_area and area_ratio_before is not None and area_ratio_before > 1e-12:
                scale_candidates.append(float(np.sqrt(target_area_ratio / area_ratio_before)))
            scale_fix = float(min(scale_candidates))

            if np.isfinite(scale_fix) and 1e-9 < scale_fix < 1.0:
                uv_out *= scale_fix
                applied = True
                guard_scale = scale_fix

                uv_ext_after = _uv_extents_2d(uv_out)
                flat_max_after = float(max(float(uv_ext_after[0]), float(uv_ext_after[1])))
                dim_ratio_after = float(flat_max_after / ref_span) if ref_span > 1e-12 else dim_ratio_before

                area_2d_after = float(_mesh_total_area_2d(mesh, uv_out))
                if area_ratio_before is not None and area_3d > 1e-12 and np.isfinite(area_2d_after):
                    area_ratio_after = float(area_2d_after / area_3d)
                warning = True

        out_meta["flatten_size_warning"] = bool(warning)
        out_meta["flatten_size_guard_applied"] = bool(applied)
        out_meta["flatten_size_guard_scale"] = float(guard_scale)
        out_meta["flatten_size_dim_ratio_before"] = float(dim_ratio_before)
        out_meta["flatten_size_dim_ratio_after"] = float(dim_ratio_after)
        out_meta["flatten_size_area_ratio_before"] = None if area_ratio_before is None else float(area_ratio_before)
        out_meta["flatten_size_area_ratio_after"] = None if area_ratio_after is None else float(area_ratio_after)
        out_meta["flatten_size_bbox_area"] = float(flat_area_bbox)
        out_meta["flatten_size_ref_span"] = float(ref_span)
        out_meta["flatten_size_warn_dim_ratio"] = float(warn_dim_ratio)
        out_meta["flatten_size_hard_dim_ratio"] = float(hard_dim_ratio)
        out_meta["flatten_size_warn_area_ratio"] = float(warn_area_ratio)
        out_meta["flatten_size_hard_area_ratio"] = float(hard_area_ratio)
        return uv_out, out_meta, applied
    except Exception:
        _log_ignored_exception("flatten size guard failed")
        out_meta["flatten_size_warning"] = False
        out_meta["flatten_size_guard_applied"] = False
        out_meta["flatten_size_guard_scale"] = 1.0
        return uv_out, out_meta, False


def _similarity_align_2d(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Similarity align source to target (least-squares Procrustes, 2D)."""
    src = np.asarray(source, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    if src.ndim != 2 or tgt.ndim != 2 or src.shape[0] != tgt.shape[0] or src.shape[1] < 2 or tgt.shape[1] < 2:
        return src[:, :2].copy() if src.ndim == 2 and src.shape[1] >= 2 else np.zeros((0, 2), dtype=np.float64)

    src2 = src[:, :2].copy()
    tgt2 = tgt[:, :2].copy()
    finite = np.all(np.isfinite(src2), axis=1) & np.all(np.isfinite(tgt2), axis=1)
    if int(finite.sum()) < 2:
        return src2

    a = src2[finite]
    b = tgt2[finite]
    a_mean = a.mean(axis=0)
    b_mean = b.mean(axis=0)
    a0 = a - a_mean
    b0 = b - b_mean

    denom = float(np.sum(a0 * a0))
    if not np.isfinite(denom) or denom < 1e-12:
        out = src2
        out[finite] = b
        return out

    h = a0.T @ b0
    try:
        u, s, vt = np.linalg.svd(h)
    except Exception:
        return src2

    r = u @ vt
    if float(np.linalg.det(r)) < 0:
        u[:, -1] *= -1.0
        r = u @ vt

    scale = float(np.sum(s)) / denom
    if not np.isfinite(scale) or abs(scale) < 1e-12:
        scale = 1.0

    aligned = (src2 - a_mean) @ r.T
    aligned *= scale
    aligned += b_mean
    return aligned


def _robust_circle_fit_2d(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Robust-ish 2D circle fit (Kasa + one MAD outlier trim)."""
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
            a, b0, c = [float(v) for v in sol]
            r2 = float(c + a * a + b0 * b0)
            r = float(np.sqrt(max(r2, 0.0)))
            if not np.isfinite(r) or r <= 1e-12:
                return None
            return np.array([a, b0], dtype=np.float64), r
        except Exception:
            return None

    fitted = _fit(xx, yy)
    if fitted is None:
        return None
    center, r = fitted

    try:
        res = np.abs(np.hypot(xx - float(center[0]), yy - float(center[1])) - float(r))
        if res.size >= 8 and np.isfinite(res).any():
            med = float(np.median(res))
            mad = float(np.median(np.abs(res - med)))
            thr = med + 3.5 * float(max(mad, 1e-12))
            keep = res <= thr
            if int(np.count_nonzero(keep)) >= 3 and int(np.count_nonzero(keep)) < int(res.size):
                fitted2 = _fit(xx[keep], yy[keep])
                if fitted2 is not None:
                    center, r = fitted2
    except Exception:
        pass

    return center, float(r)


def _angles_to_min_range(
    angles: np.ndarray, *, seam_hint: float | None = None
) -> tuple[np.ndarray, float, float]:
    a = np.asarray(angles, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    if a.size < 2:
        seam = 0.0
        out = np.zeros_like(np.asarray(angles, dtype=np.float64).reshape(-1))
        return out, seam, 0.0

    if seam_hint is not None:
        try:
            seam_val = float(seam_hint)
            if np.isfinite(seam_val):
                raw = np.asarray(angles, dtype=np.float64).reshape(-1) - seam_val
                wrapped = np.mod(raw, 2.0 * np.pi)
                span = float(np.nanmax(wrapped) - np.nanmin(wrapped)) if wrapped.size else 0.0
                return wrapped, float(seam_val), span
        except Exception:
            pass

    a_sorted = np.sort(a)
    diffs = np.diff(a_sorted)
    wrap_gap = (a_sorted[0] + 2.0 * np.pi) - a_sorted[-1]
    gaps = np.concatenate([diffs, [wrap_gap]])
    k = int(np.argmax(gaps))

    if k == a_sorted.size - 1:
        start = float(a_sorted[-1])
        gap = float(wrap_gap)
        seam = start + gap * 0.5
    else:
        start = float(a_sorted[k])
        end = float(a_sorted[k + 1])
        seam = (start + end) * 0.5

    raw = np.asarray(angles, dtype=np.float64).reshape(-1) - seam
    wrapped = np.mod(raw, 2.0 * np.pi)
    span = float(np.nanmax(wrapped) - np.nanmin(wrapped)) if wrapped.size else 0.0
    return wrapped, float(seam), span


def _flatten_cut_lines_points(cut_lines_world: list[list[list[float]]] | None) -> np.ndarray:
    if not cut_lines_world:
        return np.zeros((0, 3), dtype=np.float64)
    pts: list[np.ndarray] = []
    for line in cut_lines_world:
        if line is None:
            continue
        arr = np.asarray(line, dtype=np.float64).reshape(-1, 3)
        if arr.size:
            pts.append(arr)
    if not pts:
        return np.zeros((0, 3), dtype=np.float64)
    out = np.concatenate(pts, axis=0)
    if not np.isfinite(out).all():
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _seam_hint_from_cut_lines(
    cut_lines_world: list[list[list[float]]] | None,
    *,
    axis: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    center: np.ndarray,
) -> float | None:
    pts = _flatten_cut_lines_points(cut_lines_world)
    if pts.size == 0:
        return None
    try:
        axis = np.asarray(axis, dtype=np.float64).reshape(3)
        b1 = np.asarray(b1, dtype=np.float64).reshape(3)
        b2 = np.asarray(b2, dtype=np.float64).reshape(3)
        center = np.asarray(center, dtype=np.float64).reshape(3)
    except Exception:
        return None

    t = pts @ axis
    perp = pts - t[:, None] * axis[None, :]
    r_vec = perp - center[None, :]
    x = r_vec @ b1
    y = r_vec @ b2
    theta = np.arctan2(y, x)
    if theta.size == 0:
        return None
    s = np.sin(theta)
    c = np.cos(theta)
    s_sum = float(np.sum(s))
    c_sum = float(np.sum(c))
    if not np.isfinite(s_sum) or not np.isfinite(c_sum):
        return None
    if abs(s_sum) < 1e-9 and abs(c_sum) < 1e-9:
        return None
    return float(np.arctan2(s_sum, c_sum))


def _smooth_finite_series(values: np.ndarray, *, passes: int = 2) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.float64)

    out = arr.copy()
    finite = np.isfinite(out)
    if not bool(np.any(finite)):
        return np.zeros_like(out)

    idx = np.arange(out.size, dtype=np.float64)
    if int(np.count_nonzero(finite)) == 1:
        out[~finite] = float(out[finite][0])
    else:
        out[~finite] = np.interp(
            idx[~finite],
            idx[finite],
            out[finite],
            left=float(out[finite][0]),
            right=float(out[finite][-1]),
        )

    for _ in range(max(0, int(passes))):
        if out.size < 3:
            break
        tmp = out.copy()
        tmp[1:-1] = 0.25 * out[:-2] + 0.50 * out[1:-1] + 0.25 * out[2:]
        out = tmp
    return out


def _unwrap_angle_series(values: np.ndarray, *, hint: float | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.float64)

    finite = np.isfinite(arr)
    if not bool(np.any(finite)):
        base = 0.0
        if hint is not None:
            try:
                if np.isfinite(float(hint)):
                    base = float(hint)
            except Exception:
                base = 0.0
        return np.full_like(arr, base)

    out = arr.copy()
    first = int(np.flatnonzero(finite)[0])
    prev = float(out[first])
    if hint is not None:
        try:
            hint_f = float(hint)
            if np.isfinite(hint_f):
                prev = hint_f + float(np.angle(np.exp(1j * (prev - hint_f))))
        except Exception:
            pass
    out[first] = prev

    for i in range(first + 1, out.size):
        if not np.isfinite(out[i]):
            out[i] = prev
            continue
        delta = float(np.angle(np.exp(1j * (float(out[i]) - prev))))
        prev = prev + delta
        out[i] = prev

    out[:first] = out[first]
    return out

def _coerce_section_guides(section_guides: object) -> list[dict[str, float | None]]:
    if not isinstance(section_guides, (list, tuple)):
        return []

    parsed: list[dict[str, float | None]] = []
    for item in section_guides:
        source = item if isinstance(item, dict) else None
        if source is None:
            continue
        try:
            station = float(source.get("station", None))
        except Exception:
            continue
        if not np.isfinite(station):
            continue

        try:
            confidence = float(source.get("confidence", 0.0) or 0.0)
        except Exception:
            confidence = 0.0
        confidence = float(np.clip(confidence, 0.0, 1.0))

        try:
            radius_val = source.get("radius_world", None)
            radius = float(radius_val) if radius_val is not None else None
        except Exception:
            radius = None
        if radius is not None and (not np.isfinite(radius) or radius <= 0.0):
            radius = None

        parsed.append(
            {
                "station": float(station),
                "confidence": float(confidence),
                "radius_world": None if radius is None else float(radius),
            }
        )

    if not parsed:
        return []

    parsed.sort(key=lambda item: float(item["station"] or 0.0))
    merged: list[dict[str, float | None]] = []
    for item in parsed:
        if merged and abs(float(merged[-1]["station"] or 0.0) - float(item["station"] or 0.0)) <= 1e-6:
            prev = merged[-1]
            prev_conf = float(prev.get("confidence", 0.0) or 0.0)
            item_conf = float(item.get("confidence", 0.0) or 0.0)
            prev_radius = prev.get("radius_world", None)
            item_radius = item.get("radius_world", None)
            if item_radius is not None and (prev_radius is None or item_conf >= prev_conf):
                merged[-1] = item
            elif item_conf > prev_conf:
                prev["confidence"] = item_conf
        else:
            merged.append(dict(item))
    return merged


def _coerce_finite_3d_array(value) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return arr[:3]


def _choose_best_axis_xyz(vertices: np.ndarray) -> str:
    v = np.asarray(vertices, dtype=np.float64)
    if v.ndim != 2 or v.shape[0] < 4 or v.shape[1] < 3:
        return "z"

    best_axis = "z"
    best_score = float("inf")
    for ax in ("x", "y", "z"):
        axis = _axis_unit_vector(ax)
        t = v @ axis
        perp = v - t[:, None] * axis[None, :]
        center = perp.mean(axis=0)
        r = np.linalg.norm(perp - center[None, :], axis=1)
        r = r[np.isfinite(r)]
        if r.size < 4:
            continue
        med = float(np.median(r))
        if not np.isfinite(med) or med < 1e-9:
            continue
        score = float(np.std(r) / med)
        if score < best_score:
            best_score = score
            best_axis = ax
    return best_axis


def _safe_smoothing_uv(uv: np.ndarray, edge_i: np.ndarray, edge_j: np.ndarray, edge_w: np.ndarray,
                    *, iterations: int, strength: float, fixed_indices=None) -> np.ndarray:
    return _smooth_uv_laplacian(uv, edge_i, edge_j, edge_w, iterations=iterations, strength=strength, fixed_indices=fixed_indices)


def _smooth_uv_laplacian(
    uv: np.ndarray,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    edge_w: np.ndarray,
    *,
    iterations: int = 3,
    strength: float = 0.15,
    fixed_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Lightweight Laplacian smoothing on UV."""
    uv = np.asarray(uv, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[0] == 0:
        return uv
    iters = int(max(0, iterations))
    if iters <= 0:
        return uv
    w = float(strength)
    if not np.isfinite(w) or w <= 0.0:
        return uv
    w = min(0.95, w)

    n = int(uv.shape[0])
    edge_i = np.asarray(edge_i, dtype=np.int32).reshape(-1)
    edge_j = np.asarray(edge_j, dtype=np.int32).reshape(-1)
    edge_w = np.asarray(edge_w, dtype=np.float64).reshape(-1)
    if edge_i.size == 0:
        return uv

    m = int(min(edge_i.size, edge_j.size, edge_w.size))
    ei = edge_i[:m]
    ej = edge_j[:m]
    ew = edge_w[:m]

    fixed_mask = np.zeros((n,), dtype=bool)
    if fixed_indices is not None:
        idx = np.asarray(fixed_indices, dtype=np.int32).reshape(-1)
        valid = (idx >= 0) & (idx < n)
        fixed_mask[idx[valid]] = True

    out = uv.copy()
    for _ in range(iters):
        sum_w = np.zeros((n,), dtype=np.float64)
        sum_uv = np.zeros((n, 2), dtype=np.float64)
        np.add.at(sum_w, ei, ew)
        np.add.at(sum_w, ej, ew)
        np.add.at(sum_uv, ei, ew[:, None] * out[ej])
        np.add.at(sum_uv, ej, ew[:, None] * out[ei])

        avg = np.zeros_like(out)
        mask = sum_w > 1e-12
        if np.any(mask):
            avg[mask] = sum_uv[mask] / sum_w[mask][:, None]
        new_uv = out.copy()
        movable = ~fixed_mask
        new_uv[movable] = (1.0 - w) * out[movable] + w * avg[movable]
        out = new_uv

    return out


def _log_exception(context: str) -> None:
    try:
        log_once(_LOGGER, "flattener:utils_exception", logging.INFO, "%s", context)
    except Exception:
        _log_ignored_exception(context)


def sanitize_mesh(mesh: MeshData) -> MeshData:
    """Remove invalid/degenerate faces and compact used vertices."""
    if mesh is None:
        raise ValueError("mesh is None")

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[0] == 0 or faces.ndim != 2 or faces.shape[0] == 0:
        return mesh

    faces = faces[:, :3].astype(np.int32, copy=False)
    a = faces[:, 0]
    b = faces[:, 1]
    c = faces[:, 2]

    mask = (a != b) & (b != c) & (a != c)
    finite_v = np.all(np.isfinite(vertices), axis=1)
    mask &= finite_v[a] & finite_v[b] & finite_v[c]

    faces = faces[mask]
    if faces.shape[0] == 0:
        return MeshData(
            vertices=np.zeros((0, 3), dtype=np.float64),
            faces=np.zeros((0, 3), dtype=np.int32),
            normals=None,
            face_normals=None,
            uv_coords=None,
            texture=mesh.texture,
            unit=mesh.unit,
            filepath=mesh.filepath,
        )

    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    area2 = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    faces = faces[area2 > 1e-12]
    if faces.shape[0] == 0:
        return MeshData(
            vertices=np.zeros((0, 3), dtype=np.float64),
            faces=np.zeros((0, 3), dtype=np.int32),
            normals=None,
            face_normals=None,
            uv_coords=None,
            texture=mesh.texture,
            unit=mesh.unit,
            filepath=mesh.filepath,
        )

    unique_verts = np.unique(faces.reshape(-1)).astype(np.int32, copy=False)
    new_vertices = vertices[unique_verts]
    new_faces = np.searchsorted(unique_verts, faces).astype(np.int32, copy=False)

    new_uv = None
    if mesh.uv_coords is not None:
        try:
            uv = np.asarray(mesh.uv_coords, dtype=np.float64)
            if uv.ndim == 2 and uv.shape[0] == vertices.shape[0] and uv.shape[1] >= 2:
                new_uv = uv[unique_verts][:, :2].copy()
        except Exception:
            new_uv = None

    return MeshData(
        vertices=new_vertices,
        faces=new_faces,
        normals=None,
        face_normals=None,
        uv_coords=new_uv,
        texture=mesh.texture,
        unit=mesh.unit,
        filepath=mesh.filepath,
    )


def extract_submesh_with_mapping(
    mesh: MeshData, face_indices: np.ndarray
) -> tuple[MeshData, np.ndarray]:
    """Build a compact submesh and return its vertex-to-original mapping."""
    face_indices = np.asarray(face_indices, dtype=np.int32).reshape(-1)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)

    if face_indices.size == 0 or faces.ndim != 2 or faces.shape[0] == 0:
        empty = MeshData(
            vertices=np.zeros((0, 3), dtype=np.float64),
            faces=np.zeros((0, 3), dtype=np.int32),
            normals=None,
            face_normals=None,
            uv_coords=None,
            texture=mesh.texture,
            unit=mesh.unit,
            filepath=mesh.filepath,
        )
        return empty, np.zeros((0,), dtype=np.int32)

    valid = (face_indices >= 0) & (face_indices < int(faces.shape[0]))
    face_indices = face_indices[valid]
    if face_indices.size == 0:
        empty = MeshData(
            vertices=np.zeros((0, 3), dtype=np.float64),
            faces=np.zeros((0, 3), dtype=np.int32),
            normals=None,
            face_normals=None,
            uv_coords=None,
            texture=mesh.texture,
            unit=mesh.unit,
            filepath=mesh.filepath,
        )
        return empty, np.zeros((0,), dtype=np.int32)

    selected_faces = faces[face_indices][:, :3]
    unique_verts = np.unique(selected_faces.reshape(-1)).astype(np.int32, copy=False)
    new_vertices = vertices[unique_verts]
    new_faces = np.searchsorted(unique_verts, selected_faces).astype(np.int32, copy=False)

    new_uv = None
    if mesh.uv_coords is not None:
        try:
            uv = np.asarray(mesh.uv_coords, dtype=np.float64)
            if uv.ndim == 2 and uv.shape[0] == vertices.shape[0] and uv.shape[1] >= 2:
                new_uv = uv[unique_verts][:, :2].copy()
        except Exception:
            new_uv = None

    return (
        MeshData(
            vertices=new_vertices,
            faces=new_faces,
            normals=None,
            face_normals=None,
            uv_coords=new_uv,
            texture=mesh.texture,
            unit=mesh.unit,
            filepath=mesh.filepath,
        ),
        unique_verts,
    )
