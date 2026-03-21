"""
Synthetic tile generator and evaluation helpers.

These utilities provide:
- synthetic roof-tile meshes with known fabrication parameters
- ground-truth interpretation state
- evaluation metrics for estimated tile interpretation states
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import math
from pathlib import Path
from typing import Any

import numpy as np

from .mesh_loader import MeshData
from .tile_form_model import (
    AxisHint,
    AxisSource,
    MandrelFitResult,
    SectionObservation,
    SplitScheme,
    TileClass,
    TileInterpretationState,
)


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if not math.isfinite(out):
        return float(default)
    return float(out)


def _safe_optional_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return float(out)


def _theta_span_degrees(tile_class: TileClass, split_scheme: SplitScheme) -> float:
    if split_scheme == SplitScheme.HALF:
        return 176.0 if tile_class == TileClass.SUGKIWA else 184.0
    if split_scheme == SplitScheme.QUARTER:
        return 88.0 if tile_class == TileClass.SUGKIWA else 96.0
    return 136.0


def _ground_truth_sections(
    *,
    stations: np.ndarray,
    radii: np.ndarray,
    theta_span_rad: float,
    axis_vector_world: tuple[float, float, float],
) -> list[SectionObservation]:
    sections: list[SectionObservation] = []
    for station, radius in zip(stations.tolist(), radii.tolist(), strict=False):
        width = 2.0 * float(radius) * float(math.sin(theta_span_rad * 0.5))
        depth = float(radius) * float(1.0 - math.cos(theta_span_rad * 0.5))
        sections.append(
            SectionObservation(
                station=float(station),
                origin_world=(0.0, float(station), 0.0),
                normal_world=axis_vector_world,
                confidence=1.0,
                accepted=True,
                profile_contour_count=1,
                profile_point_count=96,
                profile_width_world=max(0.0, width),
                profile_depth_world=max(0.0, depth),
                profile_radius_median_world=float(radius),
                profile_radius_iqr_world=0.0,
                note="ground_truth",
            )
        )
    return sections


@dataclass(slots=True)
class SyntheticTileSpec:
    tile_class: TileClass = TileClass.SUGKIWA
    split_scheme: SplitScheme = SplitScheme.QUARTER
    length_world: float = 180.0
    radius_base_world: float = 68.0
    radius_amplitude_world: float = 7.5
    theta_span_deg: float = 0.0
    axial_samples: int = 28
    angular_samples: int = 36
    twist_deg: float = 4.0
    bend_world: float = 6.0
    axial_slope_world: float = 2.0
    noise_std_world: float = 0.0
    thickness_world: float = 0.0
    seed: int = 0
    unit: str = "mm"
    record_view: str = "top"
    name: str = ""

    def resolved_theta_span_deg(self) -> float:
        span = float(self.theta_span_deg or 0.0)
        if span > 0.0:
            return span
        return _theta_span_degrees(self.tile_class, self.split_scheme)

    def resolved_name(self) -> str:
        if str(self.name or "").strip():
            return str(self.name).strip()
        return f"synthetic_{self.tile_class.value}_{self.split_scheme.value}_seed{int(self.seed)}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "tile_class": str(self.tile_class.value),
            "split_scheme": str(self.split_scheme.value),
            "length_world": float(self.length_world),
            "radius_base_world": float(self.radius_base_world),
            "radius_amplitude_world": float(self.radius_amplitude_world),
            "theta_span_deg": float(self.theta_span_deg),
            "axial_samples": int(self.axial_samples),
            "angular_samples": int(self.angular_samples),
            "twist_deg": float(self.twist_deg),
            "bend_world": float(self.bend_world),
            "axial_slope_world": float(self.axial_slope_world),
            "noise_std_world": float(self.noise_std_world),
            "thickness_world": float(self.thickness_world),
            "seed": int(self.seed),
            "unit": str(self.unit or "mm"),
            "record_view": str(self.record_view or ""),
            "name": str(self.name or ""),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "SyntheticTileSpec":
        if not isinstance(data, dict):
            return cls()
        return cls(
            tile_class=TileClass.from_value(data.get("tile_class")),
            split_scheme=SplitScheme.from_value(data.get("split_scheme")),
            length_world=_safe_float(data.get("length_world", 180.0), 180.0),
            radius_base_world=_safe_float(data.get("radius_base_world", 68.0), 68.0),
            radius_amplitude_world=_safe_float(data.get("radius_amplitude_world", 7.5), 7.5),
            theta_span_deg=_safe_float(data.get("theta_span_deg", 0.0), 0.0),
            axial_samples=max(4, int(_safe_float(data.get("axial_samples", 28), 28))),
            angular_samples=max(8, int(_safe_float(data.get("angular_samples", 36), 36))),
            twist_deg=_safe_float(data.get("twist_deg", 4.0), 4.0),
            bend_world=_safe_float(data.get("bend_world", 6.0), 6.0),
            axial_slope_world=_safe_float(data.get("axial_slope_world", 2.0), 2.0),
            noise_std_world=max(0.0, _safe_float(data.get("noise_std_world", 0.0), 0.0)),
            thickness_world=max(0.0, _safe_float(data.get("thickness_world", 0.0), 0.0)),
            seed=int(_safe_float(data.get("seed", 0), 0)),
            unit=str(data.get("unit", "mm") or "mm"),
            record_view=str(data.get("record_view", "top") or "top"),
            name=str(data.get("name", "") or ""),
        )


def default_synthetic_tile_spec(
    *,
    tile_class: TileClass,
    split_scheme: SplitScheme,
    seed: int = 0,
) -> SyntheticTileSpec:
    radius_base = 72.0 if tile_class == TileClass.AMKIWA else 64.0
    radius_amp = 8.0 if split_scheme == SplitScheme.HALF else 6.0
    bend_world = 8.0 if tile_class == TileClass.SUGKIWA else 5.0
    return SyntheticTileSpec(
        tile_class=tile_class,
        split_scheme=split_scheme,
        length_world=190.0 if split_scheme == SplitScheme.HALF else 165.0,
        radius_base_world=radius_base,
        radius_amplitude_world=radius_amp,
        axial_samples=30,
        angular_samples=40,
        twist_deg=4.5 if tile_class == TileClass.SUGKIWA else 3.0,
        bend_world=bend_world,
        axial_slope_world=2.5,
        seed=int(seed),
        record_view="top",
    )


def synthetic_tile_spec_from_preset(preset: object, *, seed: int = 0) -> SyntheticTileSpec:
    key = str(preset or "").strip().lower()
    if key == "sugkiwa_half":
        return default_synthetic_tile_spec(
            tile_class=TileClass.SUGKIWA,
            split_scheme=SplitScheme.HALF,
            seed=int(seed),
        )
    if key == "amkiwa_quarter":
        return default_synthetic_tile_spec(
            tile_class=TileClass.AMKIWA,
            split_scheme=SplitScheme.QUARTER,
            seed=int(seed),
        )
    if key == "amkiwa_half":
        return default_synthetic_tile_spec(
            tile_class=TileClass.AMKIWA,
            split_scheme=SplitScheme.HALF,
            seed=int(seed),
        )
    return default_synthetic_tile_spec(
        tile_class=TileClass.SUGKIWA,
        split_scheme=SplitScheme.QUARTER,
        seed=int(seed),
    )


@dataclass(slots=True)
class SyntheticTileGroundTruth:
    spec: SyntheticTileSpec = field(default_factory=SyntheticTileSpec)
    ground_truth_state: TileInterpretationState = field(default_factory=TileInterpretationState)
    axis_vector_world: tuple[float, float, float] = (0.0, 1.0, 0.0)
    axis_origin_world: tuple[float, float, float] = (0.0, 0.0, 0.0)
    section_stations: list[float] = field(default_factory=list)
    section_radii_world: list[float] = field(default_factory=list)
    selected_faces: list[int] = field(default_factory=list)
    mesh_name: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "ground_truth_state": self.ground_truth_state.to_dict(),
            "axis_vector_world": list(self.axis_vector_world),
            "axis_origin_world": list(self.axis_origin_world),
            "section_stations": [float(x) for x in self.section_stations],
            "section_radii_world": [float(x) for x in self.section_radii_world],
            "selected_faces": [int(x) for x in self.selected_faces],
            "mesh_name": str(self.mesh_name or ""),
            "note": str(self.note or ""),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "SyntheticTileGroundTruth":
        if not isinstance(data, dict):
            return cls()
        axis_vec = data.get("axis_vector_world", [0.0, 1.0, 0.0])
        axis_org = data.get("axis_origin_world", [0.0, 0.0, 0.0])
        return cls(
            spec=SyntheticTileSpec.from_dict(data.get("spec")),
            ground_truth_state=TileInterpretationState.from_dict(data.get("ground_truth_state")),
            axis_vector_world=(
                _safe_float(axis_vec[0] if len(axis_vec) > 0 else 0.0),
                _safe_float(axis_vec[1] if len(axis_vec) > 1 else 1.0, 1.0),
                _safe_float(axis_vec[2] if len(axis_vec) > 2 else 0.0),
            ),
            axis_origin_world=(
                _safe_float(axis_org[0] if len(axis_org) > 0 else 0.0),
                _safe_float(axis_org[1] if len(axis_org) > 1 else 0.0),
                _safe_float(axis_org[2] if len(axis_org) > 2 else 0.0),
            ),
            section_stations=[_safe_float(x) for x in list(data.get("section_stations", []) or [])],
            section_radii_world=[_safe_float(x) for x in list(data.get("section_radii_world", []) or [])],
            selected_faces=[int(_safe_float(x, 0.0)) for x in list(data.get("selected_faces", []) or [])],
            mesh_name=str(data.get("mesh_name", "") or ""),
            note=str(data.get("note", "") or ""),
        )

    def summary_lines(self) -> list[str]:
        return [
            f"합성 정답: {self.spec.tile_class.label_ko} / {self.spec.split_scheme.label_ko}",
            f"길이 {self.spec.length_world:.1f} {self.spec.unit} | 반경 {self.spec.radius_base_world:.1f} ± {self.spec.radius_amplitude_world:.1f}",
            f"단면 정답 {len(self.section_stations)}개 | 선택 면 {len(self.selected_faces)}개",
        ]


@dataclass(slots=True)
class SyntheticTileArtifact:
    mesh: MeshData
    truth: SyntheticTileGroundTruth
    name: str


@dataclass(slots=True)
class TileEvaluationReport:
    tile_class_match: bool = False
    split_scheme_match: bool = False
    record_view_match: bool = False
    axis_angle_error_deg: float | None = None
    axis_origin_offset_world: float | None = None
    section_station_mae_world: float | None = None
    section_radius_mae_world: float | None = None
    mandrel_radius_abs_error_world: float | None = None
    completeness: float = 0.0
    overall_score: float = 0.0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tile_class_match": bool(self.tile_class_match),
            "split_scheme_match": bool(self.split_scheme_match),
            "record_view_match": bool(self.record_view_match),
            "axis_angle_error_deg": self.axis_angle_error_deg,
            "axis_origin_offset_world": self.axis_origin_offset_world,
            "section_station_mae_world": self.section_station_mae_world,
            "section_radius_mae_world": self.section_radius_mae_world,
            "mandrel_radius_abs_error_world": self.mandrel_radius_abs_error_world,
            "completeness": float(self.completeness),
            "overall_score": float(self.overall_score),
            "note": str(self.note or ""),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "TileEvaluationReport":
        if not isinstance(data, dict):
            return cls()
        return cls(
            tile_class_match=bool(data.get("tile_class_match", False)),
            split_scheme_match=bool(data.get("split_scheme_match", False)),
            record_view_match=bool(data.get("record_view_match", False)),
            axis_angle_error_deg=_safe_optional_float(data.get("axis_angle_error_deg")),
            axis_origin_offset_world=_safe_optional_float(data.get("axis_origin_offset_world")),
            section_station_mae_world=_safe_optional_float(data.get("section_station_mae_world")),
            section_radius_mae_world=_safe_optional_float(data.get("section_radius_mae_world")),
            mandrel_radius_abs_error_world=_safe_optional_float(data.get("mandrel_radius_abs_error_world")),
            completeness=_clip01(_safe_float(data.get("completeness", 0.0), 0.0)),
            overall_score=_clip01(_safe_float(data.get("overall_score", 0.0), 0.0)),
            note=str(data.get("note", "") or ""),
        )

    def summary_lines(self, unit: str = "mm") -> list[str]:
        lines = [
            f"평가 점수: {self.overall_score * 100.0:.0f} / 100 | 완성도 {self.completeness * 100.0:.0f}%",
            (
                f"유형 {'정답' if self.tile_class_match else '오답'} | "
                f"분할 {'정답' if self.split_scheme_match else '오답'} | "
                f"기록면 {'정답' if self.record_view_match else '오답'}"
            ),
        ]
        if self.axis_angle_error_deg is not None:
            lines.append(f"축 각도 오차: {self.axis_angle_error_deg:.2f} deg")
        if self.mandrel_radius_abs_error_world is not None:
            lines.append(f"와통 반경 오차: {self.mandrel_radius_abs_error_world:.3f} {unit}")
        if self.section_radius_mae_world is not None:
            lines.append(f"단면 반경 MAE: {self.section_radius_mae_world:.3f} {unit}")
        return lines


@dataclass(slots=True)
class SyntheticBenchmarkCaseResult:
    preset: str = ""
    seed: int = 0
    mesh_name: str = ""
    bundle_path: str = ""
    review_path: str = ""
    overall_score: float = 0.0
    tile_class_match: bool = False
    split_scheme_match: bool = False
    record_view_match: bool = False
    mandrel_radius_abs_error_world: float | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "preset": str(self.preset or ""),
            "seed": int(self.seed),
            "mesh_name": str(self.mesh_name or ""),
            "bundle_path": str(self.bundle_path or ""),
            "review_path": str(self.review_path or ""),
            "overall_score": float(self.overall_score),
            "tile_class_match": bool(self.tile_class_match),
            "split_scheme_match": bool(self.split_scheme_match),
            "record_view_match": bool(self.record_view_match),
            "mandrel_radius_abs_error_world": self.mandrel_radius_abs_error_world,
            "note": str(self.note or ""),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "SyntheticBenchmarkCaseResult":
        if not isinstance(data, dict):
            return cls()
        return cls(
            preset=str(data.get("preset", "") or ""),
            seed=int(_safe_float(data.get("seed", 0), 0)),
            mesh_name=str(data.get("mesh_name", "") or ""),
            bundle_path=str(data.get("bundle_path", "") or ""),
            review_path=str(data.get("review_path", "") or ""),
            overall_score=_clip01(_safe_float(data.get("overall_score", 0.0), 0.0)),
            tile_class_match=bool(data.get("tile_class_match", False)),
            split_scheme_match=bool(data.get("split_scheme_match", False)),
            record_view_match=bool(data.get("record_view_match", False)),
            mandrel_radius_abs_error_world=_safe_optional_float(data.get("mandrel_radius_abs_error_world")),
            note=str(data.get("note", "") or ""),
        )


@dataclass(slots=True)
class SyntheticBenchmarkSuiteReport:
    created_at_iso: str = ""
    output_dir: str = ""
    case_count: int = 0
    average_score: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0
    pass_threshold: float = 0.9
    pass_count: int = 0
    fail_count: int = 0
    presets: list[str] = field(default_factory=list)
    seeds: list[int] = field(default_factory=list)
    cases: list[SyntheticBenchmarkCaseResult] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at_iso": str(self.created_at_iso or ""),
            "output_dir": str(self.output_dir or ""),
            "case_count": int(self.case_count),
            "average_score": float(self.average_score),
            "min_score": float(self.min_score),
            "max_score": float(self.max_score),
            "pass_threshold": float(self.pass_threshold),
            "pass_count": int(self.pass_count),
            "fail_count": int(self.fail_count),
            "presets": [str(x) for x in list(self.presets or [])],
            "seeds": [int(_safe_float(x, 0)) for x in list(self.seeds or [])],
            "cases": [item.to_dict() for item in list(self.cases or [])],
            "note": str(self.note or ""),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "SyntheticBenchmarkSuiteReport":
        if not isinstance(data, dict):
            return cls()
        raw_cases = data.get("cases", [])
        return cls(
            created_at_iso=str(data.get("created_at_iso", "") or ""),
            output_dir=str(data.get("output_dir", "") or ""),
            case_count=max(0, int(_safe_float(data.get("case_count", 0), 0))),
            average_score=_clip01(_safe_float(data.get("average_score", 0.0), 0.0)),
            min_score=_clip01(_safe_float(data.get("min_score", 0.0), 0.0)),
            max_score=_clip01(_safe_float(data.get("max_score", 0.0), 0.0)),
            pass_threshold=_clip01(_safe_float(data.get("pass_threshold", 0.9), 0.9)),
            pass_count=max(0, int(_safe_float(data.get("pass_count", 0), 0))),
            fail_count=max(0, int(_safe_float(data.get("fail_count", 0), 0))),
            presets=[str(x) for x in list(data.get("presets", []) or [])],
            seeds=[int(_safe_float(x, 0)) for x in list(data.get("seeds", []) or [])],
            cases=[
                SyntheticBenchmarkCaseResult.from_dict(item)
                for item in list(raw_cases or [])
                if isinstance(item, dict)
            ],
            note=str(data.get("note", "") or ""),
        )

    def summary_lines(self) -> list[str]:
        return [
            f"합성 벤치마크 {self.case_count}건",
            f"평균 {self.average_score * 100.0:.1f} / 100 | 최소 {self.min_score * 100.0:.1f} | 최대 {self.max_score * 100.0:.1f}",
            f"기준점수 {self.pass_threshold * 100.0:.0f} / 100 | 통과 {self.pass_count}건 | 실패 {self.fail_count}건",
            f"preset {len(self.presets)}개 | seed {len(self.seeds)}개",
        ]

    def failing_case_lines(self, *, limit: int = 3) -> list[str]:
        threshold = float(self.pass_threshold)
        failing = [
            item for item in list(self.cases or [])
            if float(item.overall_score) + 1e-12 < threshold
        ]
        if not failing:
            return ["모든 synthetic benchmark 케이스가 기준점수를 통과했습니다."]
        lines: list[str] = []
        for item in failing[: max(1, int(limit))]:
            lines.append(
                f"FAIL {item.preset} seed {int(item.seed)} | {item.overall_score * 100.0:.1f} / 100"
            )
        if len(failing) > max(1, int(limit)):
            lines.append(f"... 외 {len(failing) - max(1, int(limit))}건")
        return lines


def _section_guides_from_state(state: TileInterpretationState | None) -> list[dict[str, Any]]:
    guides: list[dict[str, Any]] = []
    if state is None:
        return guides

    observations = [
        SectionObservation.from_dict(item.to_dict()) if isinstance(item, SectionObservation) else SectionObservation.from_dict(item)
        for item in list(getattr(state, "section_observations", []) or [])
    ]
    accepted = [item for item in observations if bool(getattr(item, "accepted", True))]
    source = accepted if accepted else observations
    for item in source:
        station = _safe_optional_float(getattr(item, "station", None))
        if station is None:
            continue
        guide: dict[str, Any] = {
            "station": float(station),
            "confidence": float(max(0.0, min(1.0, float(getattr(item, "confidence", 0.0) or 0.0)))),
        }
        radius_world = _safe_optional_float(getattr(item, "profile_radius_median_world", None))
        if radius_world is not None and radius_world > 0.0:
            guide["radius_world"] = float(radius_world)
        guides.append(guide)
    return guides


def _synthetic_record_label(record_view: str) -> str:
    view_key = str(record_view or "").strip().lower()
    if view_key == "top":
        return "상면 기록면"
    if view_key == "bottom":
        return "하면 기록면"
    return "기록면"


def render_synthetic_tile_review_sheet(
    artifact: SyntheticTileArtifact,
    *,
    interpretation_state: TileInterpretationState | None = None,
    evaluation_report: TileEvaluationReport | None = None,
    dpi: int = 300,
):
    from .flattener import flatten_with_method
    from .recording_surface_review import (
        RecordingSurfaceReviewOptions,
        build_recording_surface_summary_lines,
        render_recording_surface_review,
    )

    state = TileInterpretationState.from_dict(
        (interpretation_state or artifact.truth.ground_truth_state).to_dict()
    )
    truth_state = TileInterpretationState.from_dict(artifact.truth.ground_truth_state.to_dict())
    axis_vector = (
        state.mandrel_fit.axis_vector_world
        or state.axis_hint.vector_world
        or truth_state.mandrel_fit.axis_vector_world
        or truth_state.axis_hint.vector_world
        or artifact.truth.axis_vector_world
    )
    mandrel_radius = _safe_optional_float(
        getattr(getattr(state, "mandrel_fit", None), "radius_world", None)
    )
    if mandrel_radius is None or mandrel_radius <= 0.0:
        mandrel_radius = _safe_optional_float(
            getattr(getattr(truth_state, "mandrel_fit", None), "radius_world", None)
        )
    record_view = str(getattr(state, "record_view", "") or artifact.truth.spec.record_view or "top").strip().lower()
    section_guides = _section_guides_from_state(state)
    if not section_guides:
        section_guides = _section_guides_from_state(truth_state)

    flattened = flatten_with_method(
        artifact.mesh,
        method="section",
        cylinder_axis=axis_vector if axis_vector is not None else "auto",
        cylinder_radius=mandrel_radius,
        section_guides=section_guides,
        section_record_view=record_view,
    )

    extra_lines: list[str] = []
    extra_lines.extend(artifact.truth.summary_lines())
    if evaluation_report is not None:
        extra_lines.extend(evaluation_report.summary_lines(unit=str(getattr(artifact.mesh, "unit", "") or "mm")))

    summary_lines = build_recording_surface_summary_lines(
        flattened,
        record_label=_synthetic_record_label(record_view),
        target_label="합성 기와",
        strategy_suffix=" · synthetic benchmark",
        mode_label="합성 기준면 검토",
        tile_class_label=state.tile_class.label_ko,
        split_scheme_label=state.split_scheme.label_ko,
        record_strategy_label=str(getattr(state, "record_strategy", "") or "synthetic_ground_truth"),
        guide_count=len(section_guides),
        mandrel_radius_world=mandrel_radius,
        extra_lines=extra_lines,
    )
    review = render_recording_surface_review(
        flattened,
        options=RecordingSurfaceReviewOptions(
            dpi=int(max(72, int(dpi))),
            title=f"합성 기록면 검토 시트 - {str(artifact.name or artifact.truth.mesh_name or 'synthetic_tile')}",
            summary_lines=summary_lines,
        ),
    )
    return review


def save_synthetic_tile_bundle(
    artifact: SyntheticTileArtifact,
    mesh_output_path: str | Path,
    *,
    interpretation_state: TileInterpretationState | None = None,
    evaluation_report: TileEvaluationReport | None = None,
    include_review_sheet: bool = True,
    review_dpi: int = 300,
) -> dict[str, str]:
    from .mesh_loader import MeshProcessor

    mesh_path = Path(mesh_output_path).expanduser()
    mesh_path.parent.mkdir(parents=True, exist_ok=True)

    MeshProcessor().save_mesh(artifact.mesh, str(mesh_path))

    truth_path = mesh_path.with_suffix(".truth.json")
    truth_path.write_text(
        json_dumps(artifact.truth.to_dict()),
        encoding="utf-8",
    )

    paths: dict[str, str] = {
        "mesh": str(mesh_path),
        "truth": str(truth_path),
    }

    if interpretation_state is not None:
        interpretation_path = mesh_path.with_suffix(".interpretation.json")
        interpretation_path.write_text(
            json_dumps(interpretation_state.to_dict()),
            encoding="utf-8",
        )
        paths["interpretation"] = str(interpretation_path)

    if evaluation_report is not None:
        evaluation_path = mesh_path.with_suffix(".evaluation.json")
        evaluation_path.write_text(
            json_dumps(evaluation_report.to_dict()),
            encoding="utf-8",
        )
        paths["evaluation"] = str(evaluation_path)

    if bool(include_review_sheet):
        review = render_synthetic_tile_review_sheet(
            artifact,
            interpretation_state=interpretation_state,
            evaluation_report=evaluation_report,
            dpi=int(max(72, int(review_dpi))),
        )
        review_path = mesh_path.with_suffix(".review.png")
        review.combined_image.save(review_path)
        paths["review"] = str(review_path)

    manifest_path = mesh_path.with_suffix(".bundle.json")
    manifest_path.write_text(
        json_dumps(
            {
                "created_at_iso": datetime.now().isoformat(timespec="seconds"),
                "name": str(artifact.name or artifact.truth.mesh_name or mesh_path.stem),
                "mesh": mesh_path.name,
                "truth": truth_path.name,
                "interpretation": Path(paths["interpretation"]).name if "interpretation" in paths else None,
                "evaluation": Path(paths["evaluation"]).name if "evaluation" in paths else None,
                "review": Path(paths["review"]).name if "review" in paths else None,
            }
        ),
        encoding="utf-8",
    )
    paths["bundle"] = str(manifest_path)
    return paths


def save_synthetic_benchmark_suite(
    output_dir: str | Path,
    *,
    presets: list[str] | tuple[str, ...] = ("sugkiwa_quarter", "sugkiwa_half", "amkiwa_quarter", "amkiwa_half"),
    seeds: list[int] | tuple[int, ...] = (1,),
    include_review_sheets: bool = True,
    review_dpi: int = 300,
    pass_threshold: float = 0.9,
) -> SyntheticBenchmarkSuiteReport:
    root = Path(output_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    case_results: list[SyntheticBenchmarkCaseResult] = []
    normalized_presets = [str(item or "").strip().lower() for item in list(presets or []) if str(item or "").strip()]
    normalized_seeds = [int(_safe_float(item, 0)) for item in list(seeds or [])]
    normalized_seeds = list(dict.fromkeys(normalized_seeds))
    if not normalized_presets:
        normalized_presets = ["sugkiwa_quarter"]
    if not normalized_seeds:
        normalized_seeds = [1]
    threshold = _clip01(_safe_float(pass_threshold, 0.9))

    for preset in normalized_presets:
        preset_dir = root / preset
        preset_dir.mkdir(parents=True, exist_ok=True)
        for seed in normalized_seeds:
            spec = synthetic_tile_spec_from_preset(preset, seed=int(seed))
            artifact = generate_synthetic_tile(spec)
            report = evaluate_tile_interpretation(artifact.truth.ground_truth_state, artifact.truth)
            mesh_path = preset_dir / f"{artifact.name}.obj"
            bundle_paths = save_synthetic_tile_bundle(
                artifact,
                mesh_path,
                interpretation_state=artifact.truth.ground_truth_state,
                evaluation_report=report,
                include_review_sheet=include_review_sheets,
                review_dpi=review_dpi,
            )
            case_results.append(
                SyntheticBenchmarkCaseResult(
                    preset=preset,
                    seed=int(seed),
                    mesh_name=str(artifact.name or artifact.truth.mesh_name or ""),
                    bundle_path=str(bundle_paths.get("bundle", "")),
                    review_path=str(bundle_paths.get("review", "")),
                    overall_score=float(report.overall_score),
                    tile_class_match=bool(report.tile_class_match),
                    split_scheme_match=bool(report.split_scheme_match),
                    record_view_match=bool(report.record_view_match),
                    mandrel_radius_abs_error_world=report.mandrel_radius_abs_error_world,
                    note="synthetic_benchmark_baseline",
                )
            )

    scores = np.asarray([float(item.overall_score) for item in case_results], dtype=np.float64)
    suite = SyntheticBenchmarkSuiteReport(
        created_at_iso=datetime.now().isoformat(timespec="seconds"),
        output_dir=str(root),
        case_count=len(case_results),
        average_score=float(np.mean(scores)) if scores.size > 0 else 0.0,
        min_score=float(np.min(scores)) if scores.size > 0 else 0.0,
        max_score=float(np.max(scores)) if scores.size > 0 else 0.0,
        pass_threshold=float(threshold),
        pass_count=sum(1 for item in case_results if float(item.overall_score) + 1e-12 >= threshold),
        fail_count=sum(1 for item in case_results if float(item.overall_score) + 1e-12 < threshold),
        presets=list(normalized_presets),
        seeds=list(normalized_seeds),
        cases=case_results,
        note="synthetic_benchmark_suite",
    )

    summary_json = root / "synthetic_benchmark_summary.json"
    summary_json.write_text(json_dumps(suite.to_dict()), encoding="utf-8")

    summary_csv = root / "synthetic_benchmark_summary.csv"
    csv_lines = [
        "preset,seed,mesh_name,overall_score,pass_threshold,passed,tile_class_match,split_scheme_match,record_view_match,mandrel_radius_abs_error_world,bundle_path,review_path"
    ]
    for item in case_results:
        passed = float(item.overall_score) + 1e-12 >= threshold
        csv_lines.append(
            ",".join(
                [
                    str(item.preset),
                    str(int(item.seed)),
                    str(item.mesh_name),
                    f"{float(item.overall_score):.6f}",
                    f"{float(threshold):.6f}",
                    "1" if passed else "0",
                    "1" if item.tile_class_match else "0",
                    "1" if item.split_scheme_match else "0",
                    "1" if item.record_view_match else "0",
                    "" if item.mandrel_radius_abs_error_world is None else f"{float(item.mandrel_radius_abs_error_world):.6f}",
                    str(item.bundle_path),
                    str(item.review_path),
                ]
            )
        )
    summary_csv.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    return suite


def json_dumps(data: Any) -> str:
    import json

    return json.dumps(data, ensure_ascii=False, indent=2)


def generate_synthetic_tile(spec: SyntheticTileSpec) -> SyntheticTileArtifact:
    spec = SyntheticTileSpec.from_dict(spec.to_dict())
    rng = np.random.default_rng(int(spec.seed))

    n_len = max(4, int(spec.axial_samples))
    n_theta = max(8, int(spec.angular_samples))
    theta_span_deg = float(spec.resolved_theta_span_deg())
    theta_span_rad = math.radians(theta_span_deg)
    theta0 = -0.5 * theta_span_rad
    theta1 = 0.5 * theta_span_rad

    ys = np.linspace(-0.5 * float(spec.length_world), 0.5 * float(spec.length_world), n_len + 1, dtype=np.float64)
    thetas = np.linspace(theta0, theta1, n_theta + 1, dtype=np.float64)

    orientation_sign = -1.0 if spec.tile_class == TileClass.AMKIWA else 1.0
    length_safe = max(abs(float(spec.length_world)), 1e-6)

    vertices: list[list[float]] = []
    radius_rows: list[float] = []
    for y in ys:
        t = (float(y) / length_safe) + 0.5
        radius = float(spec.radius_base_world) + float(spec.radius_amplitude_world) * float(math.sin(math.pi * t))
        twist = math.radians(float(spec.twist_deg)) * float(math.sin(2.0 * math.pi * t))
        x_offset = float(spec.bend_world) * float(math.sin(math.pi * t))
        z_offset = float(spec.axial_slope_world) * float(math.sin(2.0 * math.pi * t))
        radius_rows.append(radius)
        for theta in thetas:
            angle = float(theta + twist)
            x = x_offset + (radius * float(math.cos(angle)))
            z = z_offset + (orientation_sign * radius * float(math.sin(angle)))
            if float(spec.noise_std_world) > 0.0:
                noise = rng.normal(0.0, float(spec.noise_std_world), size=3)
                x += float(noise[0])
                y_local = float(y) + float(noise[1])
                z += float(noise[2])
            else:
                y_local = float(y)
            vertices.append([x, y_local, z])

    def idx(row: int, col: int) -> int:
        return int(row) * (n_theta + 1) + int(col)

    faces: list[list[int]] = []
    for row in range(n_len):
        for col in range(n_theta):
            a = idx(row, col)
            b = idx(row, col + 1)
            c = idx(row + 1, col + 1)
            d = idx(row + 1, col)
            faces.append([a, b, c])
            faces.append([a, c, d])

    mesh = MeshData(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int32),
        unit=str(spec.unit or "mm"),
    )

    section_quantiles = np.asarray([0.15, 0.325, 0.5, 0.675, 0.85], dtype=np.float64)
    section_stations = np.quantile(ys, section_quantiles)
    row_station_lookup = np.interp(section_stations, ys, np.asarray(radius_rows, dtype=np.float64))
    axis_vector = (0.0, 1.0, 0.0)
    truth_sections = _ground_truth_sections(
        stations=np.asarray(section_stations, dtype=np.float64),
        radii=np.asarray(row_station_lookup, dtype=np.float64),
        theta_span_rad=float(theta_span_rad),
        axis_vector_world=axis_vector,
    )

    radius_array = np.asarray(row_station_lookup, dtype=np.float64)
    radius_world = float(np.median(radius_array))
    radius_spread_world = float(np.quantile(radius_array, 0.75) - np.quantile(radius_array, 0.25))

    truth_state = TileInterpretationState(
        tile_class=spec.tile_class,
        split_scheme=spec.split_scheme,
        axis_hint=AxisHint(
            source=AxisSource.USER_GUIDED,
            vector_world=axis_vector,
            origin_world=(0.0, 0.0, 0.0),
            confidence=1.0,
            face_count=int(mesh.n_faces),
            note="synthetic_ground_truth",
        ),
        section_observations=truth_sections,
        mandrel_fit=MandrelFitResult(
            radius_world=radius_world,
            radius_spread_world=radius_spread_world,
            axis_origin_world=(0.0, 0.0, 0.0),
            axis_vector_world=axis_vector,
            confidence=1.0,
            used_sections=len(truth_sections),
            used_points=len(truth_sections) * 96,
            scope="synthetic_ground_truth",
            note="synthetic_ground_truth",
        ),
        record_view=str(spec.record_view or "top"),
        record_strategy="synthetic_ground_truth",
        workflow_stage="record_surface",
        note="synthetic_ground_truth",
    )
    truth_state.touch()

    truth = SyntheticTileGroundTruth(
        spec=spec,
        ground_truth_state=truth_state,
        axis_vector_world=axis_vector,
        axis_origin_world=(0.0, 0.0, 0.0),
        section_stations=[float(x) for x in section_stations.tolist()],
        section_radii_world=[float(x) for x in radius_array.tolist()],
        selected_faces=list(range(int(mesh.n_faces))),
        mesh_name=spec.resolved_name(),
        note="synthetic_ground_truth",
    )
    return SyntheticTileArtifact(mesh=mesh, truth=truth, name=spec.resolved_name())


def _axis_angle_error_deg(estimated: AxisHint, truth: SyntheticTileGroundTruth) -> float | None:
    if not estimated.is_defined():
        return None
    est = np.asarray(estimated.vector_world or (0.0, 0.0, 0.0), dtype=np.float64).reshape(3)
    tru = np.asarray(truth.axis_vector_world, dtype=np.float64).reshape(3)
    est_n = float(np.linalg.norm(est))
    tru_n = float(np.linalg.norm(tru))
    if est_n <= 1e-12 or tru_n <= 1e-12:
        return None
    est /= est_n
    tru /= tru_n
    dot = float(np.clip(abs(np.dot(est, tru)), 0.0, 1.0))
    return float(math.degrees(math.acos(dot)))


def _axis_origin_error(estimated: AxisHint, truth: SyntheticTileGroundTruth) -> float | None:
    if estimated.origin_world is None:
        return None
    est = np.asarray(estimated.origin_world, dtype=np.float64).reshape(3)
    tru = np.asarray(truth.axis_origin_world, dtype=np.float64).reshape(3)
    if est.size < 3 or tru.size < 3:
        return None
    return float(np.linalg.norm(est - tru))


def _section_errors(
    state: TileInterpretationState,
    truth: SyntheticTileGroundTruth,
) -> tuple[float | None, float | None]:
    estimated = [
        item
        for item in list(getattr(state, "section_observations", []) or [])
        if getattr(item, "station", None) is not None
    ]
    if not estimated:
        return None, None

    truth_pairs = list(zip(truth.section_stations, truth.section_radii_world, strict=False))
    if not truth_pairs:
        return None, None

    station_errors: list[float] = []
    radius_errors: list[float] = []
    for station, radius in truth_pairs:
        best = min(
            estimated,
            key=lambda item: abs(float(getattr(item, "station", 0.0) or 0.0) - float(station)),
        )
        station_errors.append(abs(float(getattr(best, "station", 0.0) or 0.0) - float(station)))
        est_radius = _safe_optional_float(getattr(best, "profile_radius_median_world", None))
        if est_radius is not None:
            radius_errors.append(abs(est_radius - float(radius)))

    station_mae = float(np.mean(np.asarray(station_errors, dtype=np.float64))) if station_errors else None
    radius_mae = float(np.mean(np.asarray(radius_errors, dtype=np.float64))) if radius_errors else None
    return station_mae, radius_mae


def evaluate_tile_interpretation(
    state: TileInterpretationState,
    truth: SyntheticTileGroundTruth,
) -> TileEvaluationReport:
    truth_state = truth.ground_truth_state
    tile_class_match = TileClass.from_value(state.tile_class) == TileClass.from_value(truth_state.tile_class)
    split_scheme_match = SplitScheme.from_value(state.split_scheme) == SplitScheme.from_value(truth_state.split_scheme)
    record_view_match = str(state.record_view or "").strip().lower() == str(truth_state.record_view or "").strip().lower()

    axis_angle_error_deg = _axis_angle_error_deg(state.axis_hint, truth)
    axis_origin_offset_world = _axis_origin_error(state.axis_hint, truth)
    section_station_mae_world, section_radius_mae_world = _section_errors(state, truth)

    est_radius = _safe_optional_float(getattr(getattr(state, "mandrel_fit", None), "radius_world", None))
    truth_radius = _safe_optional_float(getattr(getattr(truth_state, "mandrel_fit", None), "radius_world", None))
    mandrel_radius_abs_error_world = None
    if est_radius is not None and truth_radius is not None:
        mandrel_radius_abs_error_world = abs(est_radius - truth_radius)

    completeness_parts = [
        1.0 if tile_class_match or state.tile_class != TileClass.UNKNOWN else 0.0,
        1.0 if split_scheme_match or state.split_scheme != SplitScheme.UNKNOWN else 0.0,
        1.0 if state.axis_hint.is_defined() else 0.0,
        1.0 if len(list(state.section_observations or [])) > 0 else 0.0,
        1.0 if bool(getattr(state.mandrel_fit, "is_defined", lambda: False)()) else 0.0,
        1.0 if str(state.record_view or "").strip().lower() in {"top", "bottom"} else 0.0,
    ]
    completeness = float(np.mean(np.asarray(completeness_parts, dtype=np.float64)))

    axis_score = 0.0
    if axis_angle_error_deg is not None:
        axis_score = max(axis_score, 1.0 - (axis_angle_error_deg / 20.0))
    if axis_origin_offset_world is not None and truth_radius is not None:
        axis_score = max(
            0.0,
            min(1.0, (axis_score * 0.7) + ((1.0 - (axis_origin_offset_world / max(truth_radius * 0.35, 1e-6))) * 0.3)),
        )

    section_score = 0.0
    if section_station_mae_world is not None:
        station_norm = max(abs(float(truth.spec.length_world)) * 0.08, 1e-6)
        section_score += 0.5 * max(0.0, 1.0 - (section_station_mae_world / station_norm))
    if section_radius_mae_world is not None and truth_radius is not None:
        radius_norm = max(abs(float(truth_radius)) * 0.12, 1e-6)
        section_score += 0.5 * max(0.0, 1.0 - (section_radius_mae_world / radius_norm))

    mandrel_score = 0.0
    if mandrel_radius_abs_error_world is not None and truth_radius is not None:
        mandrel_norm = max(abs(float(truth_radius)) * 0.10, 1e-6)
        mandrel_score = max(0.0, 1.0 - (mandrel_radius_abs_error_world / mandrel_norm))

    overall_score = (
        (0.14 if tile_class_match else 0.0)
        + (0.10 if split_scheme_match else 0.0)
        + (0.24 * _clip01(axis_score))
        + (0.22 * _clip01(section_score))
        + (0.20 * _clip01(mandrel_score))
        + (0.10 if record_view_match else 0.0)
    )

    return TileEvaluationReport(
        tile_class_match=bool(tile_class_match),
        split_scheme_match=bool(split_scheme_match),
        record_view_match=bool(record_view_match),
        axis_angle_error_deg=axis_angle_error_deg,
        axis_origin_offset_world=axis_origin_offset_world,
        section_station_mae_world=section_station_mae_world,
        section_radius_mae_world=section_radius_mae_world,
        mandrel_radius_abs_error_world=mandrel_radius_abs_error_world,
        completeness=_clip01(completeness),
        overall_score=_clip01(overall_score),
        note="synthetic_evaluation",
    )
