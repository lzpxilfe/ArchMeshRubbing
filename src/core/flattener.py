"""Flatten orchestration and compatibility API."""

from __future__ import annotations

from typing import Any

import numpy as np

from .flatten_metrics import distortion_summary
from .flatten_models_arap import ARAPFlattener
from .flatten_models_cylindrical import cylindrical_parameterization
from .flatten_models_sectionwise import sectionwise_cylindrical_parameterization, sectionwise_quality_gate
from .flatten_policy import explain_recommendation, fallback_chain_for_context, recommend_flatten_mode
from .flatten_types import FlattenResultMeta, FlattenedMesh
from .flatten_utils import (
    _apply_flatten_size_guard,
    _log_ignored_exception,
    _mesh_total_area_2d,
    _mesh_total_area_3d,
    _similarity_align_2d,
)
from .mesh_loader import MeshData


def _normalize_method(method: str) -> str:
    raw_method = str(method or "arap").strip()
    m_text = raw_method.lower()
    if "기와 추천" in raw_method or ("section" in m_text) or ("tile" in m_text) or ("단면" in raw_method) or ("기와" in raw_method):
        return "section"
    if "저왜곡" in raw_method or "arap" in m_text:
        return "arap"
    if "기록면 기반" in raw_method or ("area" in m_text) or ("면적" in raw_method):
        return "area"
    if "곡면 추적" in raw_method or ("cyl" in m_text) or ("원통" in raw_method):
        return "cylinder"
    if "각도 보존" in raw_method or "lscm" in m_text:
        return "lscm"
    return m_text or "arap"


def _build_result(
    mesh_s: MeshData,
    flattener: ARAPFlattener,
    uv: np.ndarray,
    *,
    faces: np.ndarray | None = None,
    distortion_arr: np.ndarray | None = None,
    scale: float = 1.0,
    meta: dict[str, Any] | None = None,
) -> FlattenedMesh:
    uv_arr = np.asarray(uv, dtype=np.float64)
    face_arr = mesh_s.faces if faces is None else np.asarray(faces, dtype=np.int32)
    out_meta = dict(meta or {})
    uv_guarded, out_meta, guard_applied = _apply_flatten_size_guard(mesh_s, uv_arr, meta=out_meta)

    if distortion_arr is None or bool(guard_applied):
        dist = flattener._compute_distortion(mesh_s, uv_guarded)
    else:
        dist = np.asarray(distortion_arr, dtype=np.float64)

    out_meta["distortion_summary"] = distortion_summary(dist)
    return FlattenedMesh(
        uv=uv_guarded,
        faces=face_arr,
        original_mesh=mesh_s,
        distortion_per_face=dist,
        scale=float(scale),
        meta=out_meta,
    )


def _maybe_smooth_uv(
    flattener: ARAPFlattener,
    mesh_s: MeshData,
    uv_in: np.ndarray,
    *,
    smooth_iters_val: int,
    smooth_strength_val: float,
) -> np.ndarray:
    if smooth_iters_val <= 0 or smooth_strength_val <= 0.0:
        return uv_in
    try:
        edge_i, edge_j, edge_w = flattener._compute_cotangent_edge_weights(mesh_s)
        anchors = flattener._pick_anchor_pair(mesh_s, np.arange(mesh_s.n_vertices, dtype=np.int32))
        return flattener._smooth_uv_laplacian(
            uv_in,
            edge_i,
            edge_j,
            edge_w,
            iterations=smooth_iters_val,
            strength=smooth_strength_val,
            fixed_indices=anchors,
        )
    except Exception:
        _log_ignored_exception()
        return uv_in


def _run_single_method(
    mesh_s: MeshData,
    *,
    method: str,
    iterations: int,
    distortion: float,
    boundary_type: str,
    initial_method: str,
    cylinder_axis: Any,
    cylinder_radius: float | None,
    cut_lines_world: list[list[list[float]]] | None,
    section_guides: list[dict[str, Any]] | None,
    section_record_view: str | None,
    smooth_iters_val: int,
    smooth_strength_val: float,
    pack_compact: bool,
) -> FlattenedMesh:
    m = _normalize_method(method)
    flattener = ARAPFlattener(max_iterations=max(1, iterations) if m == "arap" else 0)

    if m == "arap":
        init_text = str(initial_method or "lscm")
        init_t = init_text.lower().strip()
        initial_uv = None
        init_meta: dict[str, Any] = {}
        if ("cyl" in init_t) or ("원통" in init_text):
            initial_uv_res = cylindrical_parameterization(
                mesh_s,
                axis=cylinder_axis,
                radius=cylinder_radius,
                cut_lines_world=cut_lines_world,
                return_meta=True,
            )
            if isinstance(initial_uv_res, tuple):
                initial_uv, init_meta = initial_uv_res
            else:
                initial_uv = initial_uv_res
        elif ("section" in init_t) or ("tile" in init_t) or ("단면" in init_text) or ("기와" in init_text):
            initial_uv_res = sectionwise_cylindrical_parameterization(
                mesh_s,
                axis=cylinder_axis,
                cut_lines_world=cut_lines_world,
                section_guides=section_guides,
                record_view=section_record_view,
                return_meta=True,
            )
            if isinstance(initial_uv_res, tuple):
                initial_uv, init_meta = initial_uv_res
            else:
                initial_uv = initial_uv_res
        out = flattener.flatten(
            mesh_s,
            boundary_type=str(boundary_type or "free"),
            initial_method=init_text,
            initial_uv=initial_uv,
            smooth_iters=smooth_iters_val,
            smooth_strength=smooth_strength_val,
            pack_compact=bool(pack_compact),
        )
        out_meta = dict(getattr(out, "meta", {}) or {})
        out_meta["flatten_method"] = "arap"
        out_meta["initial_method"] = init_text
        out_meta["iterations"] = int(iterations)
        out_meta["smooth_iters"] = int(smooth_iters_val)
        out_meta["smooth_strength"] = float(smooth_strength_val)
        if init_meta:
            out_meta.update(init_meta)
        return _build_result(
            mesh_s,
            flattener,
            out.uv,
            faces=out.faces,
            distortion_arr=out.distortion_per_face,
            scale=float(getattr(out, "scale", 1.0)),
            meta=out_meta,
        )

    if m == "lscm":
        uv = flattener._safe_initial_parameterization(mesh_s, "lscm")
        uv = np.asarray(uv, dtype=np.float64)
        if uv.ndim != 2 or uv.shape[0] != mesh_s.n_vertices or uv.shape[1] < 2:
            uv = np.zeros((mesh_s.n_vertices, 2), dtype=np.float64)
        else:
            uv = uv[:, :2].copy()
        uv = flattener._orient_uv_pca(uv)
        uv = _maybe_smooth_uv(flattener, mesh_s, uv, smooth_iters_val=smooth_iters_val, smooth_strength_val=smooth_strength_val)
        return _build_result(
            mesh_s,
            flattener,
            uv,
            distortion_arr=flattener._compute_distortion(mesh_s, uv),
            scale=1.0,
            meta={"flatten_method": "lscm"},
        )

    if m in {"area", "area_preserve", "area-preserve", "area_preserving"}:
        w = float(distortion)
        if not np.isfinite(w):
            w = 0.5
        w = float(np.clip(w, 0.0, 1.0))

        uv_area = np.asarray(flattener._safe_initial_parameterization(mesh_s, "tutte"), dtype=np.float64)
        uv_angle = np.asarray(flattener._safe_initial_parameterization(mesh_s, "lscm"), dtype=np.float64)
        if uv_area.ndim != 2 or uv_area.shape[0] != mesh_s.n_vertices or uv_area.shape[1] < 2:
            uv_area = np.zeros((mesh_s.n_vertices, 2), dtype=np.float64)
        else:
            uv_area = uv_area[:, :2].copy()
        if uv_angle.ndim != 2 or uv_angle.shape[0] != mesh_s.n_vertices or uv_angle.shape[1] < 2:
            uv_angle = np.zeros((mesh_s.n_vertices, 2), dtype=np.float64)
        else:
            uv_angle = uv_angle[:, :2].copy()

        uv = (1.0 - w) * _similarity_align_2d(uv_area, uv_angle) + w * uv_angle
        uv = flattener._orient_uv_pca(np.asarray(uv, dtype=np.float64))
        uv = _maybe_smooth_uv(flattener, mesh_s, uv, smooth_iters_val=smooth_iters_val, smooth_strength_val=smooth_strength_val)

        a3 = _mesh_total_area_3d(mesh_s)
        a2 = _mesh_total_area_2d(mesh_s, uv)
        if a3 > 1e-12 and a2 > 1e-12:
            scale = float(np.sqrt(a3 / a2))
            if np.isfinite(scale) and scale > 1e-9:
                uv *= scale

        return _build_result(
            mesh_s,
            flattener,
            uv,
            distortion_arr=flattener._compute_distortion(mesh_s, uv),
            scale=1.0,
            meta={"flatten_method": "area", "distortion_weight": float(w)},
        )

    if m in {"cylinder", "cyl", "cylindrical"}:
        uv_res = cylindrical_parameterization(
            mesh_s,
            axis=cylinder_axis,
            radius=cylinder_radius,
            cut_lines_world=cut_lines_world,
            return_meta=True,
        )
        if isinstance(uv_res, tuple):
            uv, cyl_meta = uv_res
        else:
            uv, cyl_meta = uv_res, {}
        uv = np.asarray(uv, dtype=np.float64)
        if uv.ndim != 2 or uv.shape[0] != mesh_s.n_vertices or uv.shape[1] < 2:
            uv = np.zeros((mesh_s.n_vertices, 2), dtype=np.float64)
        else:
            uv = uv[:, :2].copy()
        uv = flattener._orient_uv_pca(uv)
        uv = _maybe_smooth_uv(flattener, mesh_s, uv, smooth_iters_val=smooth_iters_val, smooth_strength_val=smooth_strength_val)
        meta = {
            "flatten_method": "cylinder",
            "cylinder_axis_input": str(cylinder_axis),
            "cylinder_radius_input": None if cylinder_radius is None else float(cylinder_radius),
            **(cyl_meta or {}),
        }
        return _build_result(
            mesh_s,
            flattener,
            uv,
            distortion_arr=flattener._compute_distortion(mesh_s, uv),
            scale=1.0,
            meta=meta,
        )

    if m in {"section", "tile", "sectionwise", "section-wise", "roof_tile"}:
        uv_res = sectionwise_cylindrical_parameterization(
            mesh_s,
            axis=cylinder_axis,
            cut_lines_world=cut_lines_world,
            section_guides=section_guides,
            record_view=section_record_view,
            return_meta=True,
        )
        if isinstance(uv_res, tuple):
            uv, section_meta = uv_res
        else:
            uv, section_meta = uv_res, {}
        uv = np.asarray(uv, dtype=np.float64)
        if uv.ndim != 2 or uv.shape[0] != mesh_s.n_vertices or uv.shape[1] < 2:
            uv = np.zeros((mesh_s.n_vertices, 2), dtype=np.float64)
        else:
            uv = uv[:, :2].copy()
        uv = _maybe_smooth_uv(flattener, mesh_s, uv, smooth_iters_val=smooth_iters_val, smooth_strength_val=smooth_strength_val)
        meta = {
            "flatten_method": "section",
            "section_axis_input": str(cylinder_axis),
            **(section_meta or {}),
        }
        return _build_result(
            mesh_s,
            flattener,
            uv,
            distortion_arr=flattener._compute_distortion(mesh_s, uv),
            scale=1.0,
            meta=meta,
        )

    raise NotImplementedError(f"Unsupported flatten method: {method}")


def flatten_with_method(
    mesh: MeshData,
    *,
    method: str = "arap",
    iterations: int = 30,
    distortion: float = 0.5,
    boundary_type: str = "free",
    initial_method: str = "lscm",
    cylinder_axis: Any = "auto",
    cylinder_radius: float | None = None,
    cut_lines_world: list[list[list[float]]] | None = None,
    section_guides: list[dict[str, Any]] | None = None,
    section_record_view: str | None = None,
    smooth_iters: int | None = None,
    smooth_strength: float = 0.15,
    pack_compact: bool = True,
) -> FlattenedMesh:
    """Public compatibility API for flattening a mesh with a chosen method."""
    normalized_method = _normalize_method(method)
    iters = max(0, int(iterations))
    flattener = ARAPFlattener(max_iterations=max(1, iters) if normalized_method == "arap" else 0)
    mesh_s = flattener._sanitize_mesh(mesh)
    if mesh_s.n_vertices == 0 or mesh_s.n_faces == 0:
        return FlattenedMesh(
            uv=np.zeros((mesh_s.n_vertices, 2), dtype=np.float64),
            faces=mesh_s.faces,
            original_mesh=mesh_s,
            distortion_per_face=np.zeros((mesh_s.n_faces,), dtype=np.float64),
            scale=1.0,
        )

    if smooth_iters is None:
        smooth_iters_val = 3 if normalized_method == "arap" else 0
    else:
        try:
            smooth_iters_val = int(smooth_iters)
        except Exception:
            smooth_iters_val = 0
        if smooth_iters_val < 0:
            smooth_iters_val = 0

    try:
        smooth_strength_val = float(smooth_strength)
    except Exception:
        smooth_strength_val = 0.0
    if not np.isfinite(smooth_strength_val) or smooth_strength_val < 0.0:
        smooth_strength_val = 0.0

    result = _run_single_method(
        mesh_s,
        method=normalized_method,
        iterations=iters,
        distortion=distortion,
        boundary_type=boundary_type,
        initial_method=initial_method,
        cylinder_axis=cylinder_axis,
        cylinder_radius=cylinder_radius,
        cut_lines_world=cut_lines_world,
        section_guides=section_guides,
        section_record_view=section_record_view,
        smooth_iters_val=smooth_iters_val,
        smooth_strength_val=smooth_strength_val,
        pack_compact=bool(pack_compact),
    )

    if normalized_method == "section":
        meta = dict(getattr(result, "meta", {}) or {})
        dist_summary = distortion_summary(getattr(result, "distortion_per_face", None))
        needs_fallback, fallback_reason = sectionwise_quality_gate(meta, distortion_summary=dist_summary)
        if needs_fallback:
            chain = fallback_chain_for_context("section")[1:]
            for candidate in chain:
                candidate_result = _run_single_method(
                    mesh_s,
                    method=candidate,
                    iterations=iters,
                    distortion=distortion,
                    boundary_type=boundary_type,
                    initial_method=initial_method,
                    cylinder_axis=cylinder_axis,
                    cylinder_radius=cylinder_radius,
                    cut_lines_world=cut_lines_world,
                    section_guides=section_guides,
                    section_record_view=section_record_view,
                    smooth_iters_val=smooth_iters_val,
                    smooth_strength_val=smooth_strength_val,
                    pack_compact=bool(pack_compact),
                )
                candidate_meta = dict(getattr(candidate_result, "meta", {}) or {})
                candidate_meta["fallback_from"] = "section"
                candidate_meta["fallback_reason"] = str(fallback_reason or meta.get("sectionwise_reason", "section_quality_gate"))
                candidate_meta["fallback_chain"] = list(fallback_chain_for_context("section"))
                candidate_meta["fallback_used_method"] = str(candidate)
                candidate_meta["requested_flatten_method"] = "section"
                candidate_result.meta = candidate_meta
                return candidate_result

    return result


def flatten_with_recommendation(
    mesh: MeshData,
    *,
    interpretation_state: Any = None,
    user_pref: Any = None,
    **kwargs,
) -> FlattenedMesh:
    """Flatten using the policy recommendation unless the caller already chose a method."""
    explicit_method = kwargs.get("method", None)
    recommendation = recommend_flatten_mode(mesh, interpretation_state, explicit_method or user_pref)
    method = explicit_method or recommendation.method
    out = flatten_with_method(mesh, method=method, **{k: v for k, v in kwargs.items() if k != "method"})
    meta = dict(getattr(out, "meta", {}) or {})
    meta["recommended_method"] = recommendation.method
    meta["recommended_label"] = recommendation.ui_label
    meta["recommendation_reason"] = recommendation.reason or explain_recommendation(mesh, interpretation_state, explicit_method or user_pref)
    meta["recommendation_confidence"] = float(recommendation.confidence)
    meta["recommendation_fallback_chain"] = list(recommendation.fallback_chain)
    meta["recommendation_alternatives"] = [
        {
            "method": item.method,
            "label": item.label,
            "reason": item.reason,
        }
        for item in recommendation.alternatives
    ]
    if recommendation.badge:
        meta["recommendation_badge"] = recommendation.badge
    out.meta = meta
    return out


__all__ = [
    "ARAPFlattener",
    "FlattenResultMeta",
    "FlattenedMesh",
    "cylindrical_parameterization",
    "flatten_with_method",
    "flatten_with_recommendation",
    "sectionwise_cylindrical_parameterization",
]


if __name__ == "__main__":
    print("Flatten orchestration module loaded successfully")
