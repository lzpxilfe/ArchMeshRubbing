"""Policy layer that maps mesh interpretation to flatten recommendations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .mesh_loader import MeshData
from .tile_form_model import TileInterpretationState
from .tile_profile_fitting import fit_circle_2d

METHOD_UI_LABELS = {
    "section": "기와 추천 펼침",
    "arap": "저왜곡 펼침",
    "area": "기록면 기반 펼침",
    "cylinder": "곡면 추적 펼침",
    "lscm": "각도 보존 펼침",
}

METHOD_UI_REASONS = {
    "section": "기와형/장축 반복 단면 패턴이 높아 기와 추천 펼침을 기본으로 권장",
    "arap": "형상 왜곡을 상대적으로 억제하는 일반 펼침",
    "area": "면적 안정성과 기록면 해석을 우선하는 펼침",
    "cylinder": "축을 따라 곡면을 추적하는 펼침",
    "lscm": "각도 보존을 우선하는 기본 펼침",
}

RECOMMENDATION_BADGE = "기와 추천"


def normalize_flatten_method_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    if "기와 추천" in str(value or "") or "section" in text or "tile" in text or "단면" in str(value or "") or "기와" in str(value or ""):
        return "section"
    if "저왜곡" in str(value or "") or "arap" in text:
        return "arap"
    if "기록면 기반" in str(value or "") or "area" in text or "면적" in str(value or ""):
        return "area"
    if "곡면 추적" in str(value or "") or "cyl" in text or "원통" in str(value or ""):
        return "cylinder"
    if "각도 보존" in str(value or "") or "lscm" in text:
        return "lscm"
    return ""


@dataclass(slots=True)
class FlattenAlternative:
    method: str
    label: str
    reason: str


@dataclass(slots=True)
class FlattenRecommendation:
    enabled: bool
    method: str
    ui_label: str
    reason: str
    confidence: float
    badge: str = ""
    tile_confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    alternatives: list[FlattenAlternative] = field(default_factory=list)
    fallback_chain: list[str] = field(default_factory=list)
    applied_default_method: str = "arap"

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "method": self.ui_label,
            "method_key": self.method,
            "reason": str(self.reason or ""),
            "confidence": float(self.confidence),
            "tile_confidence": float(self.tile_confidence),
            "badge": str(self.badge or ""),
            "alternatives": [
                {"method": item.method, "label": item.label, "reason": item.reason}
                for item in self.alternatives
            ],
            "fallback_chain": list(self.fallback_chain),
            "applied_default_method": self.applied_default_method,
        }


def fallback_chain_for_context(context: str | dict[str, Any] | None = None) -> list[str]:
    if isinstance(context, dict):
        primary = normalize_flatten_method_name(context.get("method"))
    else:
        primary = normalize_flatten_method_name(context)
    if primary == "section":
        return ["section", "area", "cylinder", "arap"]
    if primary == "cylinder":
        return ["cylinder", "area", "arap"]
    if primary == "area":
        return ["area", "cylinder", "arap"]
    if primary == "lscm":
        return ["lscm", "arap"]
    return ["arap", "area", "cylinder"]


def build_alternatives(
    base_recommendation: FlattenRecommendation,
    mesh_quality: dict[str, Any] | None = None,
) -> list[FlattenAlternative]:
    del mesh_quality
    ordered: list[str]
    if base_recommendation.method == "section":
        ordered = ["area", "cylinder", "arap"]
    elif base_recommendation.method == "area":
        ordered = ["section", "cylinder", "arap"]
    elif base_recommendation.method == "cylinder":
        ordered = ["section", "area", "arap"]
    else:
        ordered = ["section", "area", "cylinder"]

    return [
        FlattenAlternative(
            method=method,
            label=METHOD_UI_LABELS[method],
            reason=METHOD_UI_REASONS[method],
        )
        for method in ordered
    ]


def _tile_state_signature(state: TileInterpretationState | None) -> tuple[Any, ...]:
    if state is None:
        return ()
    axis_hint = getattr(state, "axis_hint", None)
    axis_vec = tuple(np.round(np.asarray(getattr(axis_hint, "vector_world", ()) or (), dtype=np.float64), 4).tolist())
    section_count = sum(
        1 for item in list(getattr(state, "section_observations", []) or []) if bool(getattr(item, "accepted", False))
    )
    analyzed_sections = sum(
        1 for item in list(getattr(state, "section_observations", []) or []) if int(getattr(item, "profile_point_count", 0) or 0) > 0
    )
    mandrel = getattr(state, "mandrel_fit", None)
    radius = getattr(mandrel, "radius_world", None) if mandrel is not None else None
    return (
        str(getattr(state, "tile_class", "") or ""),
        axis_vec,
        int(section_count),
        int(analyzed_sections),
        None if radius is None else float(np.round(float(radius), 4)),
    )


def _empty_recommendation(user_pref: Any = None) -> FlattenRecommendation:
    preferred = normalize_flatten_method_name(user_pref) or "arap"
    return FlattenRecommendation(
        enabled=False,
        method=preferred,
        ui_label=METHOD_UI_LABELS.get(preferred, METHOD_UI_LABELS["arap"]),
        reason="",
        confidence=0.0,
        tile_confidence=0.0,
        fallback_chain=fallback_chain_for_context(preferred),
        applied_default_method=preferred,
    )


def recommend_flatten_mode(
    mesh: MeshData | None,
    interpretation_state: TileInterpretationState | None = None,
    user_pref: Any = None,
) -> FlattenRecommendation:
    if mesh is None:
        return _empty_recommendation(user_pref)

    vertices = np.asarray(getattr(mesh, "vertices", None), dtype=np.float64).reshape(-1, 3)
    finite = np.isfinite(vertices).all(axis=1)
    vertices = vertices[finite]
    if vertices.shape[0] < 180:
        return _empty_recommendation(user_pref)

    centered = vertices - np.mean(vertices, axis=0)
    cov = centered.T @ centered
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
    except Exception:
        return _empty_recommendation(user_pref)

    order = np.argsort(eigvals)[::-1]
    eigvals = np.asarray(eigvals[order], dtype=np.float64)
    eigvecs = np.asarray(eigvecs[:, order], dtype=np.float64)
    if eigvals.shape[0] < 3:
        return _empty_recommendation(user_pref)

    scale = np.maximum(np.sqrt(eigvals), 1e-12)
    length_ratio = float(scale[0] / max(scale[1], 1e-12))
    shape_ratio = float(scale[1] / max(scale[2], 1e-12))

    axis_vec = np.asarray(eigvecs[:, 0], dtype=np.float64).reshape(3)
    state = interpretation_state
    if getattr(state, "axis_hint", None) is not None:
        axis_hint = getattr(state, "axis_hint", None)
        try:
            axis_hint_vec = np.asarray(axis_hint.vector_world, dtype=np.float64).reshape(3)
        except Exception:
            axis_hint_vec = None
        if axis_hint_vec is not None and np.isfinite(axis_hint_vec).all() and np.linalg.norm(axis_hint_vec) > 1e-12:
            axis_vec = axis_hint_vec / float(np.linalg.norm(axis_hint_vec))

    nrm = float(np.linalg.norm(axis_vec))
    if not np.isfinite(nrm) or nrm < 1e-12:
        axis_vec = np.asarray(eigvecs[:, 0], dtype=np.float64).reshape(3)
        nrm = float(np.linalg.norm(axis_vec))
    axis_vec = axis_vec / max(nrm, 1e-12)

    proj = centered @ axis_vec
    finite_proj = proj[np.isfinite(proj)]
    if finite_proj.size < 20:
        return _empty_recommendation(user_pref)

    s_min, s_max = float(np.quantile(finite_proj, 0.02)), float(np.quantile(finite_proj, 0.98))
    span = max(s_max - s_min, 1e-12)
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(axis_vec[0])) > 0.95:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    basis1 = np.cross(axis_vec, ref)
    basis_norm = float(np.linalg.norm(basis1))
    if not np.isfinite(basis_norm) or basis_norm < 1e-12:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        basis1 = np.cross(axis_vec, ref)
        basis_norm = float(np.linalg.norm(basis1))
    if not np.isfinite(basis_norm) or basis_norm < 1e-12:
        return _empty_recommendation(user_pref)
    basis1 = basis1 / basis_norm
    basis2 = np.cross(axis_vec, basis1)

    station_count = min(12, max(7, vertices.shape[0] // 240))
    stations = np.quantile(finite_proj, np.linspace(0.1, 0.9, station_count))
    half_window = max(0.012 * span, span / max(1.5 * station_count, 1.0))

    fit_conf: list[float] = []
    fit_span: list[float] = []
    fit_radius: list[float] = []
    fit_rmse_rel: list[float] = []

    for station in stations:
        mask = np.abs(finite_proj - float(station)) <= half_window
        mask_idx = np.nonzero(mask)[0]
        if mask_idx.size < 12:
            continue
        ring = centered[mask_idx]
        q = ring - np.outer(np.asarray(proj[mask_idx], dtype=np.float64), axis_vec)
        x = q @ basis1
        y = q @ basis2
        xy = np.column_stack([x, y])
        fit = fit_circle_2d(xy, min_points=12)
        if not bool(getattr(fit, "is_defined", lambda: False)()):
            continue
        radius = float(getattr(fit, "radius", 0.0))
        if not np.isfinite(radius) or radius <= 1e-9:
            continue
        rmse = float(getattr(fit, "rmse", 0.0))
        span_deg = float(getattr(fit, "arc_span_deg", 0.0))
        fit_conf.append(float(np.clip(float(getattr(fit, "confidence", 0.0)), 0.0, 1.0)))
        fit_span.append(float(span_deg))
        fit_radius.append(radius)
        fit_rmse_rel.append(float(rmse / max(radius, 1e-12)))

    valid_sections = len(fit_conf)
    if valid_sections < max(4, station_count // 2):
        return _empty_recommendation(user_pref)

    conf_med = float(np.median(fit_conf))
    span_med = float(np.median(fit_span))
    radius_mean = float(np.mean(fit_radius))
    radius_std = float(np.std(fit_radius))
    rmse_med = float(np.median(fit_rmse_rel))
    radius_cv = float(radius_std / max(radius_mean, 1e-12))
    section_density = valid_sections / max(float(station_count), 1.0)

    score = 0.0
    evidence: list[str] = []
    if length_ratio >= 3.0:
        score += 0.32
        evidence.append("장축 대 횡축 길이비가 큼")
    elif length_ratio >= 2.0:
        score += 0.20
        evidence.append("장축이 비교적 뚜렷함")
    else:
        score += 0.05

    if shape_ratio >= 1.4:
        score += 0.08
        evidence.append("곡면 단면 두께비가 안정적임")

    if conf_med >= 0.45:
        score += 0.18
        evidence.append("반복 단면의 원호 적합 신뢰도가 높음")
    elif conf_med >= 0.30:
        score += 0.10

    if span_med >= 45.0:
        score += 0.10
        evidence.append(f"단면 곡면 스팬 중간값 {span_med:.0f}°")

    if rmse_med <= 0.35:
        score += 0.14

    if section_density >= 0.7:
        score += 0.10
        evidence.append("길이축을 따라 단면 반복성이 높음")

    if radius_cv <= 0.35:
        score += 0.08
    elif radius_cv <= 0.55:
        score += 0.05

    if state is not None:
        section_count = len(
            [item for item in list(getattr(state, "section_observations", []) or []) if bool(getattr(item, "accepted", False))]
        )
        if section_count >= 3:
            score += 0.12
            evidence.append(f"대표 단면 {section_count}개가 이미 제안됨")
        elif section_count >= 1:
            score += 0.06
            evidence.append("대표 단면 후보 존재")
        if bool(getattr(state, "mandrel_fit", None)) and bool(getattr(state.mandrel_fit, "is_defined", lambda: False)()):
            score += 0.10
            evidence.append("와통 피팅 정보 존재")
        if bool(getattr(state, "axis_hint", None) and getattr(state.axis_hint, "is_defined", lambda: False)()):
            score += 0.06

    tile_confidence = float(np.clip(score, 0.0, 1.0))
    recommended_method = "section" if tile_confidence >= 0.65 else "arap"
    reason = (
        METHOD_UI_REASONS["section"] if recommended_method == "section" else METHOD_UI_REASONS["arap"]
    )
    if recommended_method == "section" and evidence:
        reason = f"{METHOD_UI_REASONS['section']} ({' / '.join(evidence[:4])})"

    recommendation = FlattenRecommendation(
        enabled=bool(recommended_method == "section"),
        method=recommended_method,
        ui_label=METHOD_UI_LABELS[recommended_method],
        reason=reason,
        confidence=tile_confidence,
        badge=RECOMMENDATION_BADGE if recommended_method == "section" else "",
        tile_confidence=tile_confidence,
        evidence=evidence[:6],
        fallback_chain=fallback_chain_for_context(recommended_method),
        applied_default_method=recommended_method,
    )
    recommendation.alternatives = build_alternatives(recommendation)
    return recommendation


def get_recommended_method(
    mesh: MeshData | None,
    interpretation_state: TileInterpretationState | None = None,
    user_pref: Any = None,
) -> str:
    return recommend_flatten_mode(mesh, interpretation_state, user_pref).method


def explain_recommendation(
    mesh: MeshData | None,
    interpretation_state: TileInterpretationState | None = None,
    user_pref: Any = None,
) -> str:
    recommendation = recommend_flatten_mode(mesh, interpretation_state, user_pref)
    return recommendation.reason
