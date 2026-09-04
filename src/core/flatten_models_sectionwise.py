"""Section-wise flattening models for elongated tile-like meshes."""

from __future__ import annotations

from typing import Any

import numpy as np

from .artifact_cancellation import (
    CancellationProbe,
    poll_cancellation,
    raise_if_cancelled,
)
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


_ROW_SHIFT_SAMPLE_FACE_LIMIT = 100_000
_ROW_SHIFT_GRID_STEPS = 65
_ROW_SHIFT_REFINE_STEPS = 32


def _row_shift_objective(
    delta: float,
    *,
    du: np.ndarray,
    dv: np.ndarray,
    source_length: np.ndarray,
    alpha: np.ndarray,
) -> float:
    target = np.hypot(du + (alpha * float(delta)), dv)
    residual = target - source_length
    value = float(np.mean(residual * residual)) if residual.size else float("inf")
    return value if np.isfinite(value) else float("inf")


def _bounded_row_shift(
    *,
    du: np.ndarray,
    dv: np.ndarray,
    source_length: np.ndarray,
    alpha: np.ndarray,
    cancellation_probe: CancellationProbe | None = None,
) -> float:
    """Find one deterministic per-section U translation increment."""

    usable = (
        np.isfinite(du)
        & np.isfinite(dv)
        & np.isfinite(source_length)
        & np.isfinite(alpha)
        & (source_length > 1e-12)
        & (np.abs(alpha) > 1e-9)
    )
    if int(np.count_nonzero(usable)) < 4:
        return 0.0
    du = np.asarray(du[usable], dtype=np.float64)
    dv = np.asarray(dv[usable], dtype=np.float64)
    source_length = np.asarray(source_length[usable], dtype=np.float64)
    alpha = np.asarray(alpha[usable], dtype=np.float64)

    per_edge_bound = (
        source_length + np.abs(du) + np.abs(dv)
    ) / np.maximum(np.abs(alpha), 1e-9)
    finite_bounds = per_edge_bound[np.isfinite(per_edge_bound)]
    if finite_bounds.size == 0:
        return 0.0
    bound = float(np.quantile(finite_bounds, 0.95))
    if not np.isfinite(bound) or bound <= 1e-9:
        return 0.0
    bound = min(bound, float(np.max(source_length)) * 4.0)

    grid = np.linspace(-bound, bound, _ROW_SHIFT_GRID_STEPS, dtype=np.float64)
    values_list: list[float] = []
    for candidate_index, candidate in enumerate(grid):
        poll_cancellation(cancellation_probe, candidate_index, interval=1)
        values_list.append(
            _row_shift_objective(
                float(candidate),
                du=du,
                dv=dv,
                source_length=source_length,
                alpha=alpha,
            )
        )
    values = np.asarray(values_list, dtype=np.float64)
    best_index = int(np.argmin(values))
    left = float(grid[max(0, best_index - 1)])
    right = float(grid[min(grid.size - 1, best_index + 1)])
    if right <= left:
        return float(grid[best_index])

    golden = (np.sqrt(5.0) - 1.0) * 0.5
    x1 = right - golden * (right - left)
    x2 = left + golden * (right - left)
    f1 = _row_shift_objective(
        x1,
        du=du,
        dv=dv,
        source_length=source_length,
        alpha=alpha,
    )
    f2 = _row_shift_objective(
        x2,
        du=du,
        dv=dv,
        source_length=source_length,
        alpha=alpha,
    )
    for refine_index in range(_ROW_SHIFT_REFINE_STEPS):
        poll_cancellation(cancellation_probe, refine_index, interval=1)
        if f1 <= f2:
            right = x2
            x2 = x1
            f2 = f1
            x1 = right - golden * (right - left)
            f1 = _row_shift_objective(
                x1,
                du=du,
                dv=dv,
                source_length=source_length,
                alpha=alpha,
            )
        else:
            left = x1
            x1 = x2
            f1 = f2
            x2 = left + golden * (right - left)
            f2 = _row_shift_objective(
                x2,
                du=du,
                dv=dv,
                source_length=source_length,
                alpha=alpha,
            )
    return float(x1 if f1 <= f2 else x2)


def _correct_section_row_shift(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    station: np.ndarray,
    section_stations: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    cancellation_probe: CancellationProbe | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Recover longitudinal shear that a rotating/bending tile loses in U."""

    info: dict[str, Any] = {
        "section_row_shift_applied": False,
        "section_row_shift_max_world": 0.0,
        "section_row_shift_station_count": 0,
    }
    verts = np.asarray(vertices, dtype=np.float64)
    tri = np.asarray(faces, dtype=np.int32)
    stations = np.asarray(section_stations, dtype=np.float64).reshape(-1)
    s = np.asarray(station, dtype=np.float64).reshape(-1)
    base_u = np.asarray(u, dtype=np.float64).reshape(-1)
    base_v = np.asarray(v, dtype=np.float64).reshape(-1)
    if (
        tri.ndim != 2
        or tri.shape[0] == 0
        or tri.shape[1] < 3
        or stations.size < 4
        or s.size != verts.shape[0]
        or base_u.size != verts.shape[0]
        or base_v.size != verts.shape[0]
    ):
        return base_u, info

    raise_if_cancelled(cancellation_probe)
    if tri.shape[0] > _ROW_SHIFT_SAMPLE_FACE_LIMIT:
        sample_index = np.linspace(
            0,
            tri.shape[0] - 1,
            _ROW_SHIFT_SAMPLE_FACE_LIMIT,
            dtype=np.int64,
        )
        tri = tri[sample_index]
    edges = np.concatenate(
        (tri[:, (0, 1)], tri[:, (1, 2)], tri[:, (2, 0)]),
        axis=0,
    )
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    raise_if_cancelled(cancellation_probe)
    edge_a = edges[:, 0]
    edge_b = edges[:, 1]
    ds = s[edge_b] - s[edge_a]
    source_length = np.linalg.norm(verts[edge_b, :3] - verts[edge_a, :3], axis=1)
    du = base_u[edge_b] - base_u[edge_a]
    dv = base_v[edge_b] - base_v[edge_a]
    midpoint = 0.5 * (s[edge_a] + s[edge_b])
    finite_edge = (
        np.isfinite(ds)
        & np.isfinite(source_length)
        & np.isfinite(du)
        & np.isfinite(dv)
        & np.isfinite(midpoint)
        & (source_length > 1e-12)
    )
    if int(np.count_nonzero(finite_edge)) < 8:
        return base_u, info
    ds = ds[finite_edge]
    source_length = source_length[finite_edge]
    du = du[finite_edge]
    dv = dv[finite_edge]
    midpoint = midpoint[finite_edge]

    shift = np.zeros(stations.size, dtype=np.float64)
    valid_intervals = 0
    for index in range(stations.size - 1):
        poll_cancellation(cancellation_probe, index, interval=1)
        left = float(stations[index])
        right = float(stations[index + 1])
        spacing = right - left
        if not np.isfinite(spacing) or spacing <= 1e-12:
            shift[index + 1] = shift[index]
            continue
        if index + 2 == stations.size:
            in_interval = (midpoint >= left) & (midpoint <= right)
        else:
            in_interval = (midpoint >= left) & (midpoint < right)
        # Circumferential edges in a scan are never perfectly coplanar.  Treat
        # only edges with a meaningful longitudinal component as constraints;
        # otherwise micron/sub-micron station noise creates a nearly flat
        # objective whose bounded optimum can jump to an arbitrary endpoint.
        in_interval &= (np.abs(ds) >= spacing * 0.05) & (
            np.abs(ds) <= spacing * 2.5
        )
        if int(np.count_nonzero(in_interval)) < 4:
            shift[index + 1] = shift[index]
            continue
        increment = _bounded_row_shift(
            du=du[in_interval],
            dv=dv[in_interval],
            source_length=source_length[in_interval],
            alpha=ds[in_interval] / spacing,
            cancellation_probe=cancellation_probe,
        )
        shift[index + 1] = shift[index] + increment
        valid_intervals += 1

    if valid_intervals < max(3, int(0.25 * (stations.size - 1))):
        return base_u, info
    correction = np.interp(s, stations, shift)
    candidate_u = base_u + correction
    before_length = np.hypot(du, dv)
    candidate_du = candidate_u[edge_b[finite_edge]] - candidate_u[edge_a[finite_edge]]
    after_length = np.hypot(candidate_du, dv)
    before_error = np.abs(before_length - source_length) / source_length
    after_error = np.abs(after_length - source_length) / source_length
    if (
        not np.isfinite(candidate_u).all()
        or float(np.mean(after_error)) >= float(np.mean(before_error))
        or float(np.quantile(after_error, 0.95))
        >= float(np.quantile(before_error, 0.95))
    ):
        return base_u, info

    info.update(
        {
            "section_row_shift_applied": True,
            "section_row_shift_max_world": float(np.max(shift) - np.min(shift)),
            "section_row_shift_station_count": int(stations.size),
        }
    )
    return candidate_u, info


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


#: Per-face distortion a fitted-centre unwrap may not exceed anywhere.
SECTION_DISTORTION_FACE_MAX = 0.25
#: Distortion the 95th-percentile face may not exceed under either centre.
SECTION_DISTORTION_P95_MAX = 0.15
#: Distortion the mean face may not exceed under either centre.
SECTION_DISTORTION_MEAN_MAX = 0.075


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
    mean_span_rad = float(
        info.get("section_mean_span_rad", info.get("section_mean_span", 0.0)) or 0.0
    )
    spacing = float(info.get("section_spacing", 0.0) or 0.0)
    centerline = float(info.get("section_centerline_length", 0.0) or 0.0)

    if section_count > 0 and fit_valid < max(4, int(section_count * 0.35)):
        return True, "section_fit_too_sparse"
    if centerline <= 1e-9 or spacing <= 1e-9:
        return True, "section_trace_degenerate"
    # A narrow arc cannot support a circle fit; when the centre is the
    # measured axis no fit is made, and a 10 mm strip on a pot is exactly the
    # narrow arc this exists to refuse.  Distortion still gates it below.
    if str(info.get("section_center_policy", "fit")) != "axis_origin" and (
        mean_span_rad < float(np.deg2rad(20.0))
    ):
        return True, "section_arc_span_too_small"

    dist = dict(distortion_summary or {})
    p95 = float(dist.get("p95", 0.0) or 0.0)
    mean = float(dist.get("mean", 0.0) or 0.0)
    maximum = float(dist.get("max", 0.0) or 0.0)
    # A fitted centre can put a whole face in the wrong place, and one such
    # face is a failed fit.  With the centre on the measured axis every
    # vertex lands where the axis says, and a face's distortion is only how
    # steeply it stands off the surface of revolution - a temper grain, a
    # scan spike, the wall of an incised line.  That is what a rubbing
    # records, and a finer mesh resolves more of it; the maximum is reported
    # as it is, and the mean and the 95th percentile keep gating the whole.
    if maximum > SECTION_DISTORTION_FACE_MAX and (
        str(info.get("section_center_policy", "fit")) != "axis_origin"
    ):
        return True, "section_distortion_max"
    if p95 > SECTION_DISTORTION_P95_MAX:
        return True, "section_distortion_p95"
    if mean > SECTION_DISTORTION_MEAN_MAX:
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
    seam_angle_microdegrees: int | None = None,
    section_center: str = "fit",
    station: str = "centerline",
    return_meta: bool = False,
    cancellation_probe: CancellationProbe | None = None,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Section-wise cylindrical unwrap for roof-tile like shapes.

    ``section_center`` decides where each section's circle centre comes from.
    ``"fit"`` estimates it from the section's own points, which is right for a
    tile whose axis is only roughly known and whose sections are wide arcs.
    ``"axis_origin"`` fixes it on the axis through the canonical origin, which
    is right for a vessel that has been stood on its measured rotation axis:
    a narrow meridional strip leaves each section only a short arc, and a
    circle fitted to a short arc collapses to a small circle through the
    points, unrolling the strip about a centre that is not the pot's.

    ``station`` decides what the v axis measures.  ``"centerline"`` is the
    length along the sequence of section centres, which for a straight axis is
    the axial height.  ``"meridian"`` is the length along the profile, adding
    the change in section radius to each step - what a paper strip laid on a
    belly actually spans.
    """
    if section_center not in {"fit", "axis_origin"}:
        raise ValueError("section_center must be 'fit' or 'axis_origin'")
    if station not in {"centerline", "meridian"}:
        raise ValueError("station must be 'centerline' or 'meridian'")
    if seam_angle_microdegrees is None:
        fixed_seam_angle_rad: float | None = None
    else:
        if isinstance(seam_angle_microdegrees, bool) or not isinstance(
            seam_angle_microdegrees, (int, np.integer)
        ):
            raise ValueError("seam_angle_microdegrees must be null or an integer")
        seam_angle_value = int(seam_angle_microdegrees)
        if not (-180_000_000 <= seam_angle_value < 180_000_000):
            raise ValueError(
                "seam_angle_microdegrees must be in the half-open range "
                "[-180000000, 180000000)"
            )
        fixed_seam_angle_rad = float(
            np.deg2rad(float(seam_angle_value) / 1_000_000.0)
        )
    raise_if_cancelled(cancellation_probe)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[0] == 0 or vertices.shape[1] < 3:
        return np.zeros((0, 2), dtype=np.float64)

    def _fallback(reason: str) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        raise_if_cancelled(cancellation_probe)
        uv_res = cylindrical_parameterization(
            mesh,
            axis=axis,
            radius=None,
            cut_lines_world=cut_lines_world,
            return_meta=True,
        )
        raise_if_cancelled(cancellation_probe)
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
    guide_station_list: list[float] = []
    for item in guides:
        station_value = item.get("station")
        if station_value is not None:
            guide_station_list.append(float(station_value))
    guide_station_values = np.asarray(
        guide_station_list,
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
    raise_if_cancelled(cancellation_probe)

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
    section_window = float(max(spacing * 0.6, span / max(24.0, float(s_sections.size))))

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
        guide_stations = np.asarray(
            guide_station_list,
            dtype=np.float64,
        ).reshape(-1)
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

        guide_radius_pairs: list[tuple[float, float]] = []
        for item in guides:
            station_value = item.get("station")
            radius_value = item.get("radius_world")
            if station_value is None or radius_value is None:
                continue
            radius_number = float(radius_value)
            if np.isfinite(radius_number):
                guide_radius_pairs.append(
                    (float(station_value), radius_number)
                )
        if guide_radius_pairs:
            guide_radius_stations = np.asarray(
                [item[0] for item in guide_radius_pairs],
                dtype=np.float64,
            ).reshape(-1)
            guide_radius_values = np.asarray(
                [item[1] for item in guide_radius_pairs],
                dtype=np.float64,
            ).reshape(-1)
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
        poll_cancellation(cancellation_probe, i, interval=1)
        dist = np.abs(s_valid - float(s0))
        local_idx = np.flatnonzero(dist <= section_window).astype(np.int32, copy=False)
        window_idx = local_idx
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

        if section_center == "axis_origin":
            # A fit needs many points, so the tile path widens a thin section
            # to its nearest neighbours.  A radius does not: one ring is
            # enough, and neighbours from other heights carry other radii
            # that would bend the profile the meridian is measured along.
            radius_idx = window_idx if window_idx.size > 0 else local_idx
            rr_axis = np.hypot(x_valid[radius_idx], y_valid[radius_idx])
            ss_axis = s_valid[radius_idx]
            keep = np.isfinite(rr_axis) & np.isfinite(ss_axis)
            rr_axis = rr_axis[keep]
            ss_axis = ss_axis[keep]
            cx[i] = 0.0
            cy[i] = 0.0
            # The window usually straddles two rings of different radius; a
            # median would answer with one ring's radius, and the profile the
            # meridian is measured along would step instead of slope.  A line
            # through the window evaluated at the station follows the profile.
            radius_at_station = np.nan
            if rr_axis.size >= 2 and float(np.ptp(ss_axis)) > 1e-9:
                slope, intercept = np.polyfit(ss_axis - float(s0), rr_axis, 1)
                radius_at_station = float(intercept)
            elif rr_axis.size:
                radius_at_station = float(np.median(rr_axis))
            r_sec[i] = radius_at_station
            fit_ok[i] = bool(np.isfinite(radius_at_station))
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
        raise_if_cancelled(cancellation_probe)
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
    if section_center == "axis_origin":
        # The medians are already robust, and smoothing a radius profile
        # flattens the belly it is supposed to measure.
        r_sec = np.where(np.isfinite(r_sec), r_sec, np.nan)
        r_sec = _smooth_finite_series(r_sec, passes=0)
    else:
        r_sec = _smooth_finite_series(r_sec, passes=2)
    raise_if_cancelled(cancellation_probe)

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
        poll_cancellation(cancellation_probe, i, interval=1)
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
        raise_if_cancelled(cancellation_probe)
        seams[i] = (
            float(seam_i)
            if fixed_seam_angle_rad is None
            else fixed_seam_angle_rad
        )
        spans[i] = float(span_i)

    if fixed_seam_angle_rad is None:
        seams = _unwrap_angle_series(seams, hint=seam_hint)
        seams = _smooth_finite_series(seams, passes=1)
    else:
        seams.fill(fixed_seam_angle_rad)

    centerline = (
        s_sections.reshape(-1, 1) * a.reshape(1, 3)
        + cx.reshape(-1, 1) * b1.reshape(1, 3)
        + cy.reshape(-1, 1) * b2.reshape(1, 3)
    )
    centerline_arc = np.zeros((s_sections.size,), dtype=np.float64)
    if centerline.shape[0] >= 2:
        centerline_arc[1:] = np.cumsum(np.linalg.norm(np.diff(centerline, axis=0), axis=1))
    if station == "meridian" and s_sections.size >= 2:
        # Paper follows the profile, not the axis.  Each step along the
        # centreline is lengthened by the change in section radius, so a strip
        # laid on a belly comes out as long as the surface it covered.
        radius_profile = np.asarray(r_sec, dtype=np.float64).reshape(-1)
        meridian_steps = np.hypot(
            np.linalg.norm(np.diff(centerline, axis=0), axis=1),
            np.diff(radius_profile),
        )
        meridian_steps = np.where(np.isfinite(meridian_steps), meridian_steps, 0.0)
        centerline_arc = np.concatenate(([0.0], np.cumsum(meridian_steps)))

    cx_v = np.interp(s_raw, s_sections, cx)
    cy_v = np.interp(s_raw, s_sections, cy)
    seam_v = np.interp(s_raw, s_sections, seams)
    v = np.interp(s_raw, s_sections, centerline_arc)

    u = np.zeros_like(s_raw, dtype=np.float64)
    v_out = np.zeros_like(s_raw, dtype=np.float64)

    x = x0[finite] - cx_v[finite]
    y = y0[finite] - cy_v[finite]
    theta = np.arctan2(y, x)
    if section_center == "axis_origin":
        # A strip is unrolled about its own centre meridian, so that u is zero
        # on that meridian at every height.  Measuring from the seam instead
        # would put an offset of (seam angle) x r(z) on every row, which on a
        # pot is a shear the tile row-shift search then has to undo, and does
        # not undo exactly.  A fixed seam names the cut, which sits opposite.
        if fixed_seam_angle_rad is None:
            theta_ref = float(
                np.arctan2(np.mean(np.sin(theta)), np.mean(np.cos(theta)))
            )
        else:
            theta_ref = float(fixed_seam_angle_rad + np.pi)
        theta_wrapped = np.mod(theta - theta_ref + np.pi, 2.0 * np.pi) - np.pi
        seams.fill(float(np.mod(theta_ref + np.pi, 2.0 * np.pi)))
    else:
        theta_wrapped = np.mod(theta - seam_v[finite], 2.0 * np.pi)
    r_local = np.hypot(x, y)
    finite_radius = np.isfinite(r_local) & (r_local > 1e-9)
    if not bool(np.all(finite_radius)):
        fallback_radius = r_sec[np.isfinite(r_sec) & (r_sec > 1e-9)]
        radius_fill = float(np.median(fallback_radius)) if fallback_radius.size else 1.0
        r_local = np.where(finite_radius, r_local, radius_fill)
    u[finite] = theta_wrapped * r_local
    v_out[finite] = v[finite]
    # The radius about the unrolling centre is the height a rubbing on the
    # developed surface records: what the paper feels, measured the way the
    # strip was unrolled.  It is reported, never used for the coordinates.
    vertex_radius = np.full(s_raw.shape, np.nan, dtype=np.float64)
    vertex_radius[finite] = r_local
    if section_center == "axis_origin":
        # With the centre known there is no lost shear to recover; a search
        # for one can only move rows away from where the axis put them.
        row_shift_meta = {
            "section_row_shift_applied": False,
            "section_row_shift_max_world": 0.0,
            "section_row_shift_station_count": 0,
        }
    else:
        u, row_shift_meta = _correct_section_row_shift(
            vertices=vertices[:, :3],
            faces=np.asarray(mesh.faces, dtype=np.int32),
            station=s_raw,
            section_stations=s_sections,
            u=u,
            v=v_out,
            cancellation_probe=cancellation_probe,
        )
    raise_if_cancelled(cancellation_probe)

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
        mean_span_rad = float(np.mean(spans[np.isfinite(spans)])) if np.isfinite(spans).any() else 0.0
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
            # Keep the legacy key for readers written before units were explicit.
            "section_mean_span": mean_span_rad,
            "section_mean_span_rad": mean_span_rad,
            "section_mean_span_deg": float(np.rad2deg(mean_span_rad)),
            "section_seam_hint": None if seam_hint is None else float(seam_hint),
            "section_fixed_seam_angle_microdegrees": seam_angle_microdegrees,
            "section_center_policy": section_center,
            "section_station_policy": station,
            "section_record_view": record_view_key,
            "section_u_flipped": bool(flip_u),
            "vertex_radius": vertex_radius,
            **row_shift_meta,
        }
        return uv, meta
    return uv
