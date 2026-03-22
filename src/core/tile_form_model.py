"""
Tile interpretation state models for fabrication-aware archaeology workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import math
from typing import Any


def _coerce_xyz(value: Any) -> tuple[float, float, float] | None:
    try:
        arr = list(value)
    except Exception:
        return None
    if len(arr) < 3:
        return None
    try:
        x = float(arr[0])
        y = float(arr[1])
        z = float(arr[2])
    except Exception:
        return None
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
        return None
    return (x, y, z)


def _normalize_xyz(value: Any) -> tuple[float, float, float] | None:
    xyz = _coerce_xyz(value)
    if xyz is None:
        return None
    x, y, z = xyz
    norm = math.sqrt((x * x) + (y * y) + (z * z))
    if norm <= 1e-12 or not math.isfinite(norm):
        return None
    vec = [x / norm, y / norm, z / norm]
    pivot = max(range(3), key=lambda i: abs(vec[i]))
    if float(vec[pivot]) < 0.0:
        vec = [-vec[0], -vec[1], -vec[2]]
    return (float(vec[0]), float(vec[1]), float(vec[2]))


class TileClass(str, Enum):
    UNKNOWN = "unknown"
    SUGKIWA = "sugkiwa"
    AMKIWA = "amkiwa"

    @classmethod
    def from_value(cls, value: Any) -> "TileClass":
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().lower()
        if text in {"sugkiwa", "수키와", "수"}:
            return cls.SUGKIWA
        if text in {"amkiwa", "암키와", "암"}:
            return cls.AMKIWA
        return cls.UNKNOWN

    @property
    def label_ko(self) -> str:
        return {
            self.UNKNOWN: "미상",
            self.SUGKIWA: "수키와",
            self.AMKIWA: "암키와",
        }.get(self, "미상")


class SplitScheme(str, Enum):
    UNKNOWN = "unknown"
    HALF = "half"
    QUARTER = "quarter"

    @classmethod
    def from_value(cls, value: Any) -> "SplitScheme":
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().lower()
        if text in {"half", "2", "2-way", "2분할", "둘"}:
            return cls.HALF
        if text in {"quarter", "4", "4-way", "4분할", "넷"}:
            return cls.QUARTER
        return cls.UNKNOWN

    @property
    def label_ko(self) -> str:
        return {
            self.UNKNOWN: "미상",
            self.HALF: "2분할",
            self.QUARTER: "4분할",
        }.get(self, "미상")


class AxisSource(str, Enum):
    UNKNOWN = "unknown"
    FULL_MESH_PCA = "full_mesh_pca"
    SELECTED_PATCH_PCA = "selected_patch_pca"
    USER_GUIDED = "user_guided"

    @classmethod
    def from_value(cls, value: Any) -> "AxisSource":
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().lower()
        for item in cls:
            if text == item.value:
                return item
        return cls.UNKNOWN

    @property
    def label_ko(self) -> str:
        return {
            self.UNKNOWN: "미정",
            self.FULL_MESH_PCA: "전체 메쉬 장축 추정",
            self.SELECTED_PATCH_PCA: "현재 선택 장축 추정",
            self.USER_GUIDED: "사용자 지정",
        }.get(self, "미정")


@dataclass(slots=True)
class AxisHint:
    source: AxisSource = AxisSource.UNKNOWN
    vector_world: tuple[float, float, float] | None = None
    origin_world: tuple[float, float, float] | None = None
    confidence: float = 0.0
    face_count: int = 0
    note: str = ""

    def is_defined(self) -> bool:
        return self.vector_world is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": str(self.source.value),
            "vector_world": list(self.vector_world) if self.vector_world is not None else None,
            "origin_world": list(self.origin_world) if self.origin_world is not None else None,
            "confidence": float(max(0.0, min(1.0, self.confidence))),
            "face_count": int(max(0, int(self.face_count))),
            "note": str(self.note or ""),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "AxisHint":
        if not isinstance(data, dict):
            return cls()
        try:
            confidence = float(data.get("confidence", 0.0) or 0.0)
        except Exception:
            confidence = 0.0
        try:
            face_count = int(data.get("face_count", 0) or 0)
        except Exception:
            face_count = 0
        return cls(
            source=AxisSource.from_value(data.get("source")),
            vector_world=_normalize_xyz(data.get("vector_world")),
            origin_world=_coerce_xyz(data.get("origin_world")),
            confidence=max(0.0, min(1.0, confidence)),
            face_count=max(0, face_count),
            note=str(data.get("note", "") or ""),
        )


@dataclass(slots=True)
class SectionObservation:
    station: float | None = None
    origin_world: tuple[float, float, float] | None = None
    normal_world: tuple[float, float, float] | None = None
    confidence: float = 0.0
    accepted: bool = True
    profile_contour_count: int = 0
    profile_point_count: int = 0
    profile_width_world: float = 0.0
    profile_depth_world: float = 0.0
    profile_center_world: tuple[float, float, float] | None = None
    profile_radius_median_world: float | None = None
    profile_radius_iqr_world: float = 0.0
    profile_fit_rmse_world: float = 0.0
    profile_arc_span_deg: float = 0.0
    profile_fit_confidence: float = 0.0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "station": float(self.station) if self.station is not None else None,
            "origin_world": list(self.origin_world) if self.origin_world is not None else None,
            "normal_world": list(self.normal_world) if self.normal_world is not None else None,
            "confidence": float(max(0.0, min(1.0, self.confidence))),
            "accepted": bool(self.accepted),
            "profile_contour_count": int(max(0, int(self.profile_contour_count))),
            "profile_point_count": int(max(0, int(self.profile_point_count))),
            "profile_width_world": float(max(0.0, self.profile_width_world)),
            "profile_depth_world": float(max(0.0, self.profile_depth_world)),
            "profile_center_world": list(self.profile_center_world) if self.profile_center_world is not None else None,
            "profile_radius_median_world": (
                float(self.profile_radius_median_world) if self.profile_radius_median_world is not None else None
            ),
            "profile_radius_iqr_world": float(max(0.0, self.profile_radius_iqr_world)),
            "profile_fit_rmse_world": float(max(0.0, self.profile_fit_rmse_world)),
            "profile_arc_span_deg": float(max(0.0, self.profile_arc_span_deg)),
            "profile_fit_confidence": float(max(0.0, min(1.0, self.profile_fit_confidence))),
            "note": str(self.note or ""),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "SectionObservation":
        if not isinstance(data, dict):
            return cls()
        try:
            station = data.get("station", None)
            station_val = float(station) if station is not None else None
        except Exception:
            station_val = None
        try:
            confidence = float(data.get("confidence", 0.0) or 0.0)
        except Exception:
            confidence = 0.0
        try:
            contour_count = int(data.get("profile_contour_count", 0) or 0)
        except Exception:
            contour_count = 0
        try:
            point_count = int(data.get("profile_point_count", 0) or 0)
        except Exception:
            point_count = 0
        try:
            profile_width = float(data.get("profile_width_world", 0.0) or 0.0)
        except Exception:
            profile_width = 0.0
        try:
            profile_depth = float(data.get("profile_depth_world", 0.0) or 0.0)
        except Exception:
            profile_depth = 0.0
        try:
            profile_radius = data.get("profile_radius_median_world", None)
            profile_radius_val = float(profile_radius) if profile_radius is not None else None
        except Exception:
            profile_radius_val = None
        if profile_radius_val is not None and not math.isfinite(profile_radius_val):
            profile_radius_val = None
        try:
            profile_radius_iqr = float(data.get("profile_radius_iqr_world", 0.0) or 0.0)
        except Exception:
            profile_radius_iqr = 0.0
        try:
            profile_fit_rmse = float(data.get("profile_fit_rmse_world", 0.0) or 0.0)
        except Exception:
            profile_fit_rmse = 0.0
        try:
            profile_arc_span = float(data.get("profile_arc_span_deg", 0.0) or 0.0)
        except Exception:
            profile_arc_span = 0.0
        try:
            profile_fit_confidence = float(data.get("profile_fit_confidence", 0.0) or 0.0)
        except Exception:
            profile_fit_confidence = 0.0
        return cls(
            station=station_val,
            origin_world=_coerce_xyz(data.get("origin_world")),
            normal_world=_normalize_xyz(data.get("normal_world")),
            confidence=max(0.0, min(1.0, confidence)),
            accepted=bool(data.get("accepted", True)),
            profile_contour_count=max(0, contour_count),
            profile_point_count=max(0, point_count),
            profile_width_world=max(0.0, profile_width if math.isfinite(profile_width) else 0.0),
            profile_depth_world=max(0.0, profile_depth if math.isfinite(profile_depth) else 0.0),
            profile_center_world=_coerce_xyz(data.get("profile_center_world")),
            profile_radius_median_world=profile_radius_val,
            profile_radius_iqr_world=max(0.0, profile_radius_iqr if math.isfinite(profile_radius_iqr) else 0.0),
            profile_fit_rmse_world=max(0.0, profile_fit_rmse if math.isfinite(profile_fit_rmse) else 0.0),
            profile_arc_span_deg=max(0.0, profile_arc_span if math.isfinite(profile_arc_span) else 0.0),
            profile_fit_confidence=max(0.0, min(1.0, profile_fit_confidence if math.isfinite(profile_fit_confidence) else 0.0)),
            note=str(data.get("note", "") or ""),
        )


@dataclass(slots=True)
class MandrelFitResult:
    radius_world: float | None = None
    radius_spread_world: float = 0.0
    axis_origin_world: tuple[float, float, float] | None = None
    axis_vector_world: tuple[float, float, float] | None = None
    confidence: float = 0.0
    used_sections: int = 0
    used_points: int = 0
    scope: str = ""
    note: str = ""

    def is_defined(self) -> bool:
        return self.radius_world is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "radius_world": float(self.radius_world) if self.radius_world is not None else None,
            "radius_spread_world": float(max(0.0, self.radius_spread_world)),
            "axis_origin_world": list(self.axis_origin_world) if self.axis_origin_world is not None else None,
            "axis_vector_world": list(self.axis_vector_world) if self.axis_vector_world is not None else None,
            "confidence": float(max(0.0, min(1.0, self.confidence))),
            "used_sections": int(max(0, int(self.used_sections))),
            "used_points": int(max(0, int(self.used_points))),
            "scope": str(self.scope or ""),
            "note": str(self.note or ""),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "MandrelFitResult":
        if not isinstance(data, dict):
            return cls()
        try:
            radius_world = data.get("radius_world", None)
            radius_val = float(radius_world) if radius_world is not None else None
        except Exception:
            radius_val = None
        if radius_val is not None and not math.isfinite(radius_val):
            radius_val = None
        try:
            spread = float(data.get("radius_spread_world", 0.0) or 0.0)
        except Exception:
            spread = 0.0
        try:
            confidence = float(data.get("confidence", 0.0) or 0.0)
        except Exception:
            confidence = 0.0
        try:
            used_sections = int(data.get("used_sections", 0) or 0)
        except Exception:
            used_sections = 0
        try:
            used_points = int(data.get("used_points", 0) or 0)
        except Exception:
            used_points = 0
        return cls(
            radius_world=radius_val,
            radius_spread_world=max(0.0, spread if math.isfinite(spread) else 0.0),
            axis_origin_world=_coerce_xyz(data.get("axis_origin_world")),
            axis_vector_world=_normalize_xyz(data.get("axis_vector_world")),
            confidence=max(0.0, min(1.0, confidence)),
            used_sections=max(0, used_sections),
            used_points=max(0, used_points),
            scope=str(data.get("scope", "") or ""),
            note=str(data.get("note", "") or ""),
        )


@dataclass(slots=True)
class TileInterpretationSlot:
    slot_key: str = ""
    label: str = ""
    selected_faces: list[int] = field(default_factory=list)
    tile_class: TileClass = TileClass.UNKNOWN
    split_scheme: SplitScheme = SplitScheme.UNKNOWN
    axis_hint: AxisHint = field(default_factory=AxisHint)
    section_observations: list[SectionObservation] = field(default_factory=list)
    mandrel_fit: MandrelFitResult = field(default_factory=MandrelFitResult)
    record_view: str = ""
    record_strategy: str = ""
    workflow_stage: str = "hypothesis"
    note: str = ""
    updated_at_iso: str = ""

    @staticmethod
    def _normalize_face_ids(values: Any) -> list[int]:
        out: list[int] = []
        seen: set[int] = set()
        try:
            iterable = list(values or [])
        except Exception:
            iterable = []
        for item in iterable:
            try:
                face_id = int(item)
            except Exception:
                continue
            if face_id < 0 or face_id in seen:
                continue
            seen.add(face_id)
            out.append(face_id)
        out.sort()
        return out

    @classmethod
    def from_state(
        cls,
        state: "TileInterpretationState",
        *,
        slot_key: str,
        label: str,
        selected_faces: Any = (),
    ) -> "TileInterpretationSlot":
        return cls(
            slot_key=str(slot_key or ""),
            label=str(label or ""),
            selected_faces=cls._normalize_face_ids(selected_faces),
            tile_class=TileClass.from_value(getattr(state, "tile_class", TileClass.UNKNOWN)),
            split_scheme=SplitScheme.from_value(getattr(state, "split_scheme", SplitScheme.UNKNOWN)),
            axis_hint=AxisHint.from_dict(getattr(state, "axis_hint", AxisHint()).to_dict()),
            section_observations=[
                SectionObservation.from_dict(item.to_dict())
                for item in list(getattr(state, "section_observations", []) or [])
            ],
            mandrel_fit=MandrelFitResult.from_dict(getattr(state, "mandrel_fit", MandrelFitResult()).to_dict()),
            record_view=str(getattr(state, "record_view", "") or ""),
            record_strategy=str(getattr(state, "record_strategy", "") or ""),
            workflow_stage=str(getattr(state, "workflow_stage", "hypothesis") or "hypothesis"),
            note=str(getattr(state, "note", "") or ""),
            updated_at_iso=datetime.now().isoformat(timespec="seconds"),
        )

    def to_state(self) -> "TileInterpretationState":
        return TileInterpretationState(
            tile_class=TileClass.from_value(self.tile_class),
            split_scheme=SplitScheme.from_value(self.split_scheme),
            axis_hint=AxisHint.from_dict(self.axis_hint.to_dict()),
            section_observations=[SectionObservation.from_dict(item.to_dict()) for item in self.section_observations],
            mandrel_fit=MandrelFitResult.from_dict(self.mandrel_fit.to_dict()),
            record_view=str(self.record_view or ""),
            record_strategy=str(self.record_strategy or ""),
            workflow_stage=str(self.workflow_stage or "hypothesis"),
            note=str(self.note or ""),
            updated_at_iso=str(self.updated_at_iso or ""),
        )

    def summary_label(self) -> str:
        parts: list[str] = []
        if str(self.label or "").strip():
            parts.append(str(self.label).strip())
        if self.record_view in {"top", "bottom"}:
            parts.append("상면" if str(self.record_view) == "top" else "하면")
        if self.selected_faces:
            parts.append(f"선택 {len(self.selected_faces)}면")
        if self.mandrel_fit.is_defined():
            parts.append("와통")
        return " | ".join(parts) if parts else "빈 슬롯"

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_key": str(self.slot_key or ""),
            "label": str(self.label or ""),
            "selected_faces": list(self._normalize_face_ids(self.selected_faces)),
            "tile_class": str(self.tile_class.value),
            "split_scheme": str(self.split_scheme.value),
            "axis_hint": self.axis_hint.to_dict(),
            "section_observations": [item.to_dict() for item in self.section_observations],
            "mandrel_fit": self.mandrel_fit.to_dict(),
            "record_view": str(self.record_view or ""),
            "record_strategy": str(self.record_strategy or ""),
            "workflow_stage": str(self.workflow_stage or "hypothesis"),
            "note": str(self.note or ""),
            "updated_at_iso": str(self.updated_at_iso or ""),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "TileInterpretationSlot":
        if not isinstance(data, dict):
            return cls()
        raw_sections = data.get("section_observations", [])
        sections: list[SectionObservation] = []
        if isinstance(raw_sections, list):
            sections = [SectionObservation.from_dict(item) for item in raw_sections if isinstance(item, dict)]
        return cls(
            slot_key=str(data.get("slot_key", "") or ""),
            label=str(data.get("label", "") or ""),
            selected_faces=cls._normalize_face_ids(data.get("selected_faces", [])),
            tile_class=TileClass.from_value(data.get("tile_class")),
            split_scheme=SplitScheme.from_value(data.get("split_scheme")),
            axis_hint=AxisHint.from_dict(data.get("axis_hint")),
            section_observations=sections,
            mandrel_fit=MandrelFitResult.from_dict(data.get("mandrel_fit")),
            record_view=str(data.get("record_view", "") or ""),
            record_strategy=str(data.get("record_strategy", "") or ""),
            workflow_stage=str(data.get("workflow_stage", "hypothesis") or "hypothesis"),
            note=str(data.get("note", "") or ""),
            updated_at_iso=str(data.get("updated_at_iso", "") or ""),
        )


@dataclass(slots=True)
class TileInterpretationState:
    tile_class: TileClass = TileClass.UNKNOWN
    split_scheme: SplitScheme = SplitScheme.UNKNOWN
    axis_hint: AxisHint = field(default_factory=AxisHint)
    section_observations: list[SectionObservation] = field(default_factory=list)
    mandrel_fit: MandrelFitResult = field(default_factory=MandrelFitResult)
    saved_slots: list[TileInterpretationSlot] = field(default_factory=list)
    record_view: str = ""
    record_strategy: str = ""
    workflow_stage: str = "hypothesis"
    note: str = ""
    updated_at_iso: str = ""

    def touch(self) -> None:
        self.updated_at_iso = datetime.now().isoformat(timespec="seconds")

    def summary_lines(self) -> list[str]:
        lines = [
            f"유형: {self.tile_class.label_ko}",
            f"분할 가설: {self.split_scheme.label_ko}",
        ]
        if self.axis_hint.is_defined():
            vec = self.axis_hint.vector_world or (0.0, 0.0, 0.0)
            lines.append(
                "길이축 힌트: "
                f"{self.axis_hint.source.label_ko} "
                f"(x={vec[0]:+.3f}, y={vec[1]:+.3f}, z={vec[2]:+.3f}, "
                f"신뢰도 {self.axis_hint.confidence * 100.0:.0f}%)"
            )
        else:
            lines.append("길이축 힌트: 아직 없음")
        lines.append(f"대표 단면 후보: {len(self.section_observations)}개")
        if self.mandrel_fit.is_defined():
            lines.append(
                f"와통 초벌 피팅: R={float(self.mandrel_fit.radius_world):.3f}, "
                f"후보 {int(self.mandrel_fit.used_sections)}개"
            )
        else:
            lines.append("와통 초벌 피팅: 아직 없음")
        if str(self.record_view or "").lower() in {"top", "bottom"}:
            label = "상면" if str(self.record_view).lower() == "top" else "하면"
            lines.append(f"기록면 준비: {label} ({self.record_strategy or 'auto'})")
        else:
            lines.append("기록면 준비: 아직 없음")
        lines.append(f"저장 슬롯: {len(self.saved_slots)}개")
        return lines

    @staticmethod
    def _clone_slots(slots: list[TileInterpretationSlot]) -> list[TileInterpretationSlot]:
        return [TileInterpretationSlot.from_dict(item.to_dict()) for item in list(slots or [])]

    def get_saved_slot(self, slot_key: Any) -> TileInterpretationSlot | None:
        key = str(slot_key or "").strip()
        if not key:
            return None
        for item in list(self.saved_slots or []):
            if str(getattr(item, "slot_key", "") or "") == key:
                return item
        return None

    def save_slot(self, *, slot_key: str, label: str, selected_faces: Any = ()) -> TileInterpretationSlot:
        slot = TileInterpretationSlot.from_state(
            self,
            slot_key=str(slot_key or ""),
            label=str(label or ""),
            selected_faces=selected_faces,
        )
        slots = [item for item in list(self.saved_slots or []) if str(getattr(item, "slot_key", "") or "") != slot.slot_key]
        slots.append(slot)
        slots.sort(key=lambda item: str(getattr(item, "slot_key", "") or ""))
        self.saved_slots = slots
        return slot

    def clear_slots(self) -> None:
        self.saved_slots = []

    def restore_slot(self, slot_key: Any) -> tuple["TileInterpretationState", list[int]]:
        slot = self.get_saved_slot(slot_key)
        if slot is None:
            raise KeyError(str(slot_key or ""))
        restored = slot.to_state()
        restored.saved_slots = self._clone_slots(self.saved_slots)
        return restored, list(slot.selected_faces)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tile_class": str(self.tile_class.value),
            "split_scheme": str(self.split_scheme.value),
            "axis_hint": self.axis_hint.to_dict(),
            "section_observations": [item.to_dict() for item in self.section_observations],
            "mandrel_fit": self.mandrel_fit.to_dict(),
            "saved_slots": [item.to_dict() for item in self.saved_slots],
            "record_view": str(self.record_view or ""),
            "record_strategy": str(self.record_strategy or ""),
            "workflow_stage": str(self.workflow_stage or "hypothesis"),
            "note": str(self.note or ""),
            "updated_at_iso": str(self.updated_at_iso or ""),
        }

    @classmethod
    def from_dict(cls, data: Any) -> "TileInterpretationState":
        if not isinstance(data, dict):
            return cls()
        raw_sections = data.get("section_observations", [])
        sections: list[SectionObservation] = []
        if isinstance(raw_sections, list):
            sections = [SectionObservation.from_dict(item) for item in raw_sections if isinstance(item, dict)]
        raw_slots = data.get("saved_slots", [])
        slots: list[TileInterpretationSlot] = []
        if isinstance(raw_slots, list):
            slots = [TileInterpretationSlot.from_dict(item) for item in raw_slots if isinstance(item, dict)]
        return cls(
            tile_class=TileClass.from_value(data.get("tile_class")),
            split_scheme=SplitScheme.from_value(data.get("split_scheme")),
            axis_hint=AxisHint.from_dict(data.get("axis_hint")),
            section_observations=sections,
            mandrel_fit=MandrelFitResult.from_dict(data.get("mandrel_fit")),
            saved_slots=slots,
            record_view=str(data.get("record_view", "") or ""),
            record_strategy=str(data.get("record_strategy", "") or ""),
            workflow_stage=str(data.get("workflow_stage", "hypothesis") or "hypothesis"),
            note=str(data.get("note", "") or ""),
            updated_at_iso=str(data.get("updated_at_iso", "") or ""),
        )
