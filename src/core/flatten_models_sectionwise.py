"""Section-wise flattening models for elongated tile-like meshes."""

from __future__ import annotations

from typing import Any

import numpy as np

from .flatten_models_cylindrical import cylindrical_parameterization
from .flatten_utils import (
    _angles_to_min_range,
    _axis_unit_vector,
    _coerce_section_guides,
    _normalize_cylinder_axis_choice,
    _pca_axes_3d,
    _robust_circle_fit_2d,
    _seam_hint_from_cut_lines,
    _smooth_finite_series,
    _unwrap_angle_series,
)
from .mesh_loader import MeshData


def _estimate_section_longitudinal_axis(
    vertices: np.ndarray,
    *,
    axis: Any = "auto",
) -> tuple[np.ndarray, str]:
    axis_choice = _normalize_cylinder_axis_choice(axis)
    if axis_choice != "auto":
        return _axis_unit_vector(axis), axis_choice

    try:
        pca_axes = _pca_axes_3d(vertices)
        vec = np.asarray(pca_axes[:, 0], dtype=np.float64).reshape(3)
        nrm = float(np.linalg.norm(vec))
        if np.isfinite(nrm) and nrm > 1e-12:
            return vec / nrm, "pca0"
    except Exception:
        pass

    v = np.asarray(vertices, dtype=np.float64)
    if v.ndim == 2 and v.shape[0] > 0 and v.shape[1] >= 3:
        spans = np.ptp(v[:, :3], axis=0)
        try:
            best = int(np.nanargmax(spans))
        except Exception:
            best = 1
        if best == 0:
            return _axis_unit_vector("x"), "x"
        if best == 1:
            return _axis_unit_vector("y"), "y"
    return _axis_unit_vector("z"), "z"


def sectionwise_quality_gate(
    meta: dict[str, Any] | None,
    *,
    distortion_summary: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Return (needs_fallback, reason) for sectionwise output quality."""
    info = dict(meta or {})
    if bool(info.get("sectionwise_fallback", False)):
        return True, str(info.get("sectionwise_reason", "sectionwise_internal_fallback"))

    fit_valid = int(info.get("section_fit_valid_count", 0) or 0)
    section_count = int(info.get("section_count", 0) or 0)
    mean_span = float(info.get("section_mean_span", 0.0) or 0.0)
    spacing = float(info.get("section_spacing", 0.0) or 0.0)
    centerline = float(info.get("section_centerline_length", 0.0) or 0.0)

    if section_count > 0 and fit_valid < max(4, int(section_count * 0.35)):
        return True, "section_fit_too_sparse"
    if centerline <= 1e-9 or spacing <= 1e-9:
        return True, "section_trace_degenerate"
    if mean_span < 20.0:
        return True, "section_arc_span_too_small"

    dist = dict(distortion_summary or {})
    p95 = float(dist.get("p95", 0.0) or 0.0)
    mean = float(dist.get("mean", 0.0) or 0.0)
    if p95 > 0.82:
        return True, "section_distortion_p95"
    if mean > 0.55:
        return True, "section_distortion_mean"
    return False, ""


def sectionwise_cylindrical_parameterization(
    mesh: MeshData,
    *,
    axis: Any = "auto",
    n_sections: int | None = None,
    cut_lines_world: list[list[list[float]]] | None = None,
    section_guides: list[dict[str, Any]] | None = None,
    record_view: str | None = None,
    return_meta: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Section-wise cylindrical unwrap for roof-tile like shapes."""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[0] == 0 or vertices.shape[1] < 3:
        return np.zeros((0, 2), dtype=np.float64)

    def _fallback(reason: str) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        uv_res = cylindrical_parameterization(
            mesh,
            axis=axis,
            radius=None,
            cut_lines_world=cut_lines_world,
            return_meta=True,
        )
        if isinstance(uv_res, tuple):
            uv0, meta0 = uv_res
        else:
            uv0, meta0 = uv_res, {}
        meta_out = dict(meta0 or {})
        meta_out["sectionwise_fallback"] = True
        meta_out["sectionwise_reason"] = str(reason)
        meta_out["fallback_used_method"] = "cylinder"
        if bool(return_meta):
            return uv0, meta_out
        return uv0

    a, axis_source = _estimate_section_longitudinal_axis(vertices, axis=axis)
    try:
        a = np.asarray(a, dtype=np.float64).reshape(3)
        nrm = float(np.linalg.norm(a))
        if not np.isfinite(nrm) or nrm < 1e-12:
            return _fallback("invalid_axis")
        a = a / nrm
    except Exception:
        return _fallback("axis_exception")

    temp = (
        np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(a[0])) < 0.9
        else np.array([0.0, 1.0, 0.0], dtype=np.float64)
    )
    b1 = np.cross(a, temp)
    b1_n = float(np.linalg.norm(b1))
    if not np.isfinite(b1_n) or b1_n < 1e-12:
        b1 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        b1 = np.cross(a, b1)
        b1_n = float(np.linalg.norm(b1))
    if not np.isfinite(b1_n) or b1_n < 1e-12:
        return _fallback("basis_failed")
    b1 = b1 / b1_n
    b2 = np.cross(a, b1)
    b2_n = float(np.linalg.norm(b2))
    if not np.isfinite(b2_n) or b2_n < 1e-12:
        return _fallback("basis_failed")
    b2 = b2 / b2_n

    s_raw = vertices[:, :3] @ a.reshape(3,)
    x0 = vertices[:, :3] @ b1.reshape(3,)
    y0 = vertices[:, :3] @ b2.reshape(3,)

    finite = np.isfinite(s_raw) & np.isfinite(x0) & np.isfinite(y0)
    if int(np.count_nonzero(finite)) < 16:
        return _fallback("too_few_points")

    s_valid = s_raw[finite]
    x_valid = x0[finite]
    y_valid = y0[finite]

    s_min = float(np.min(s_valid))
    s_max = float(np.max(s_valid))
    span = float(s_max - s_min)
    if not np.isfinite(span) or span < 1e-9:
        return _fallback("degenerate_span")

    guides = _coerce_section_guides(section_guides)
    guide_station_values = np.asarray(
        [float(item["station"]) for item in guides if item.get("station", None) is not None],
        dtype=np.float64,
    ).reshape(-1)
    if guide_station_values.size > 0:
        guide_station_values = np.clip(guide_station_values, s_min, s_max)
        guide_station_values = guide_station_values[np.isfinite(guide_station_values)]
        guide_station_values = np.unique(guide_station_values)

    try:
        n_sections_val = int(n_sections) if n_sections is not None else int(np.sqrt(float(vertices.shape[0])))
    except Exception:
        n_sections_val = 24
    n_sections_val = max(12, min(n_sections_val, 96))

    try:
        auto_sections = np.quantile(s_valid, np.linspace(0.0, 1.0, n_sections_val, dtype=np.float64))
    except Exception:
        auto_sections = np.linspace(s_min, s_max, n_sections_val, dtype=np.float64)

    if guide_station_values.size >= 4:
        s_sections = guide_station_values
        if guide_station_values.size < min(12, n_sections_val):
            s_sections = np.concatenate([s_sections, auto_sections])
    elif guide_station_values.size > 0:
        s_sections = np.concatenate([auto_sections, guide_station_values])
    else:
        s_sections = auto_sections
    s_sections = np.unique(np.asarray(s_sections, dtype=np.float64).reshape(-1))
    if s_sections.size < 4:
        return _fallback("too_few_sections")

    diffs = np.diff(s_sections)
    diffs = diffs[np.isfinite(diffs) & (diffs > 1e-9)]
    spacing = float(np.median(diffs)) if diffs.size else float(span / max(1, s_sections.size - 1))
    if not np.isfinite(spacing) or spacing <= 0.0:
        spacing = float(span / max(1, s_sections.size - 1))
    section_window = float(max(spacing * 1.5, span / max(12.0, float(s_sections.size))))

    min_fit_points = int(max(16, min(96, int(vertices.shape[0] // max(4, s_sections.size)))))
    nearest_k = int(max(min_fit_points, min(192, max(24, int(vertices.shape[0] // max(2, s_sections.size))))))

    cx = np.full((s_sections.size,), np.nan, dtype=np.float64)
    cy = np.full((s_sections.size,), np.nan, dtype=np.float64)
    r_sec = np.full((s_sections.size,), np.nan, dtype=np.float64)
    fit_ok = np.zeros((s_sections.size,), dtype=bool)
    guide_radius_used = np.zeros((s_sections.size,), dtype=bool)

    guide_radius_at_sections = np.full((s_sections.size,), np.nan, dtype=np.float64)
    guide_conf_at_sections = np.zeros((s_sections.size,), dtype=np.float64)
    guided_radius_source_count = 0
    if guides:
        guide_stations = np.asarray([float(item["station"]) for item in guides], dtype=np.float64).reshape(-1)
        guide_conf = np.asarray(
            [float(item.get("confidence", 0.0) or 0.0) for item in guides],
            dtype=np.float64,
        ).reshape(-1)
        if guide_conf.size > 0:
            guide_conf = np.clip(guide_conf, 0.0, 1.0)
            if guide_conf.size == 1:
                guide_conf_at_sections[:] = float(guide_conf[0])
            else:
                guide_conf_at_sections[:] = np.interp(
                    s_sections,
                    guide_stations,
                    guide_conf,
                    left=float(guide_conf[0]),
                    right=float(guide_conf[-1]),
                )

        guide_radii_raw = [
            float(item["radius_world"])
            for item in guides
            if item.get("radius_world", None) is not None and np.isfinite(float(item["radius_world"]))
        ]
        if guide_radii_raw:
            guide_radius_stations = np.asarray(
                [
                    float(item["station"])
                    for item in guides
                    if item.get("radius_world", None) is not None and np.isfinite(float(item["radius_world"]))
                ],
                dtype=np.float64,
            ).reshape(-1)
            guide_radius_values = np.asarray(guide_radii_raw, dtype=np.float64).reshape(-1)
            guided_radius_source_count = int(guide_radius_values.size)
            if guide_radius_values.size == 1:
                guide_radius_at_sections[:] = float(guide_radius_values[0])
            else:
                guide_radius_at_sections[:] = np.interp(
                    s_sections,
                    guide_radius_stations,
                    guide_radius_values,
                    left=float(guide_radius_values[0]),
                    right=float(guide_radius_values[-1]),
                )

    for i, s0 in enumerate(s_sections):
        dist = np.abs(s_valid - float(s0))
        local_idx = np.flatnonzero(dist <= section_window).astype(np.int32, copy=False)
        if local_idx.size < min_fit_points:
            k = int(min(max(min_fit_points, nearest_k), s_valid.size))
            if k <= 0:
                continue
            if k >= s_valid.size:
                local_idx = np.arange(s_valid.size, dtype=np.int32)
            else:
                local_idx = np.argpartition(dist, k - 1)[:k].astype(np.int32, copy=False)
        if local_idx.size < 3:
            continue

        x_sel = np.asarray(x_valid[local_idx], dtype=np.float64)
        y_sel = np.asarray(y_valid[local_idx], dtype=np.float64)

        guide_radius = (
            float(guide_radius_at_sections[i])
            if i < guide_radius_at_sections.size and np.isfinite(guide_radius_at_sections[i]) and guide_radius_at_sections[i] > 1e-9
            else None
        )
        guide_conf = (
            float(guide_conf_at_sections[i])
            if i < guide_conf_at_sections.size and np.isfinite(guide_conf_at_sections[i])
            else 0.0
        )

        fit = _robust_circle_fit_2d(x_sel, y_sel)
        if fit is not None:
            center_xy, radius = fit
            cx[i] = float(center_xy[0])
            cy[i] = float(center_xy[1])
            radius_fit = float(radius)
            if guide_radius is not None:
                guide_blend = float(np.clip(0.35 + (0.40 * guide_conf), 0.35, 0.80))
                r_sec[i] = (1.0 - guide_blend) * radius_fit + guide_blend * float(guide_radius)
                guide_radius_used[i] = True
            else:
                r_sec[i] = radius_fit
            fit_ok[i] = True
            continue

        cx[i] = float(np.median(x_sel))
        cy[i] = float(np.median(y_sel))
        rr = np.hypot(x_sel - float(cx[i]), y_sel - float(cy[i]))
        rr = rr[np.isfinite(rr)]
        radius_guess = float(np.median(rr)) if rr.size else np.nan
        if guide_radius is not None:
            if np.isfinite(radius_guess) and radius_guess > 1e-9:
                guide_blend = float(np.clip(0.45 + (0.35 * guide_conf), 0.45, 0.85))
                r_sec[i] = (1.0 - guide_blend) * radius_guess + guide_blend * float(guide_radius)
            else:
                r_sec[i] = float(guide_radius)
            guide_radius_used[i] = True
        else:
            r_sec[i] = radius_guess

    if int(np.count_nonzero(np.isfinite(cx) & np.isfinite(cy))) < max(4, int(0.25 * s_sections.size)):
        return _fallback("section_fit_failed")

    cx = _smooth_finite_series(cx, passes=2)
    cy = _smooth_finite_series(cy, passes=2)
    r_sec = _smooth_finite_series(r_sec, passes=2)

    mean_center = (float(np.mean(cx)) * b1) + (float(np.mean(cy)) * b2)
    seam_hint = _seam_hint_from_cut_lines(
        cut_lines_world,
        axis=a,
        b1=b1,
        b2=b2,
        center=mean_center,
    )

    seams = np.full((s_sections.size,), np.nan, dtype=np.float64)
    spans = np.full((s_sections.size,), np.nan, dtype=np.float64)
    for i, s0 in enumerate(s_sections):
        dist = np.abs(s_valid - float(s0))
        local_idx = np.flatnonzero(dist <= section_window).astype(np.int32, copy=False)
        if local_idx.size < max(8, min_fit_points // 2):
            k = int(min(max(8, min_fit_points // 2), s_valid.size))
            if k <= 0:
                continue
            if k >= s_valid.size:
                local_idx = np.arange(s_valid.size, dtype=np.int32)
            else:
                local_idx = np.argpartition(dist, k - 1)[:k].astype(np.int32, copy=False)
        if local_idx.size < 2:
            continue
        theta_loc = np.arctan2(y_valid[local_idx] - float(cy[i]), x_valid[local_idx] - float(cx[i]))
        _wrapped, seam_i, span_i = _angles_to_min_range(theta_loc, seam_hint=None)
        seams[i] = float(seam_i)
        spans[i] = float(span_i)

    seams = _unwrap_angle_series(seams, hint=seam_hint)
    seams = _smooth_finite_series(seams, passes=1)

    centerline = (
        s_sections.reshape(-1, 1) * a.reshape(1, 3)
        + cx.reshape(-1, 1) * b1.reshape(1, 3)
        + cy.reshape(-1, 1) * b2.reshape(1, 3)
    )
    centerline_arc = np.zeros((s_sections.size,), dtype=np.float64)
    if centerline.shape[0] >= 2:
        centerline_arc[1:] = np.cumsum(np.linalg.norm(np.diff(centerline, axis=0), axis=1))

    cx_v = np.interp(s_raw, s_sections, cx)
    cy_v = np.interp(s_raw, s_sections, cy)
    seam_v = np.interp(s_raw, s_sections, seams)
    v = np.interp(s_raw, s_sections, centerline_arc)

    u = np.zeros_like(s_raw, dtype=np.float64)
    v_out = np.zeros_like(s_raw, dtype=np.float64)

    x = x0[finite] - cx_v[finite]
    y = y0[finite] - cy_v[finite]
    theta = np.arctan2(y, x)
    theta_wrapped = np.mod(theta - seam_v[finite], 2.0 * np.pi)
    r_local = np.hypot(x, y)
    finite_radius = np.isfinite(r_local) & (r_local > 1e-9)
    if not bool(np.all(finite_radius)):
        fallback_radius = r_sec[np.isfinite(r_sec) & (r_sec > 1e-9)]
        radius_fill = float(np.median(fallback_radius)) if fallback_radius.size else 1.0
        r_local = np.where(finite_radius, r_local, radius_fill)
    u[finite] = theta_wrapped * r_local
    v_out[finite] = v[finite]

    record_view_key = str(record_view or "").strip().lower()
    flip_u = record_view_key == "bottom"
    if flip_u:
        u[finite] = -u[finite]

    uv = np.stack([u, v_out], axis=1)
    if np.any(finite):
        uv_f = uv[finite].copy()
        uv_f[:, 0] -= float(np.min(uv_f[:, 0]))
        uv_f[:, 1] -= float(np.min(uv_f[:, 1]))
        uv[finite] = uv_f
    if not np.isfinite(uv).all():
        uv = np.nan_to_num(uv, nan=0.0, posinf=0.0, neginf=0.0)

    if bool(return_meta):
        meta = {
            "sectionwise": True,
            "section_axis_input": str(axis),
            "section_axis_source": str(axis_source),
            "section_axis": np.asarray(a, dtype=np.float64).reshape(3),
            "section_basis_u": np.asarray(b1, dtype=np.float64).reshape(3),
            "section_basis_v": np.asarray(b2, dtype=np.float64).reshape(3),
            "section_count": int(s_sections.size),
            "section_fit_valid_count": int(np.count_nonzero(fit_ok)),
            "section_guided_count": int(guide_station_values.size),
            "section_guided_radius_count": int(guided_radius_source_count),
            "section_guided_radius_interp_count": int(np.count_nonzero(np.isfinite(guide_radius_at_sections))),
            "section_guided_radius_applied_count": int(np.count_nonzero(guide_radius_used)),
            "section_window": float(section_window),
            "section_spacing": float(spacing),
            "section_centerline_length": float(centerline_arc[-1]) if centerline_arc.size else 0.0,
            "section_mean_radius": float(np.mean(r_sec[np.isfinite(r_sec)])) if np.isfinite(r_sec).any() else 0.0,
            "section_mean_span": float(np.mean(spans[np.isfinite(spans)])) if np.isfinite(spans).any() else 0.0,
            "section_seam_hint": None if seam_hint is None else float(seam_hint),
            "section_record_view": record_view_key,
            "section_u_flipped": bool(flip_u),
        }
        return uv, meta
    return uv
