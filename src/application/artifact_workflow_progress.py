"""Record-derived progress for the native archaeological workflow.

The GUI is only a projection of this state.  Progress is rebuilt from the
immutable ArtifactDocument owned by a validated ArtifactSession, so reopening
a project, changing Align, and reactivating an older Align all produce the
same answer without widget-local flags.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from src.core.artifact_document import (
    RecordFreshness,
    RecordLifecycleStatus,
)
from src.core.artifact_rubbing_record import RUBBING_RECORD_TYPE
from src.core.artifact_session import ArtifactSession
from src.core.artifact_vector_record import (
    ArtifactVectorRecordError,
    PlanarFrame,
    VectorRecordKind,
)


REQUIRED_CUTLINE_VIEWS = ("top", "front", "right")
REQUIRED_SIX_VIEWS = ("top", "bottom", "front", "back", "right", "left")

_CUTLINE_RECORD_TYPE = VectorRecordKind.CUTLINE.record_type
_OUTLINE_RECORD_TYPE = VectorRecordKind.OUTLINE.record_type
_CUTLINE_AXES = {
    "top": (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
    "front": (
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, -1.0, 0.0),
    ),
    "right": (
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
    ),
}


class ArtifactWorkflowStep(str, Enum):
    CUTLINE = "cutline"
    OUTLINE = "outline"
    DIGITAL_RUBBING = "digital_rubbing"


@dataclass(frozen=True, slots=True)
class ArtifactWorkflowStepProgress:
    """One deterministic workflow step and the evidence that satisfies it."""

    step: ArtifactWorkflowStep
    required_views: tuple[str, ...]
    completed_views: tuple[str, ...]
    enabled: bool

    def __post_init__(self) -> None:
        step = ArtifactWorkflowStep(self.step)
        required = tuple(self.required_views)
        completed = tuple(self.completed_views)
        if not required or len(set(required)) != len(required):
            raise ValueError("required_views must be non-empty and unique")
        if any(view not in required for view in completed):
            raise ValueError("completed_views must be a subset of required_views")
        if completed != tuple(view for view in required if view in completed):
            raise ValueError("completed_views must follow required_views order")
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a bool")
        object.__setattr__(self, "step", step)
        object.__setattr__(self, "required_views", required)
        object.__setattr__(self, "completed_views", completed)

    @property
    def completed_count(self) -> int:
        return len(self.completed_views)

    @property
    def required_count(self) -> int:
        return len(self.required_views)

    @property
    def missing_views(self) -> tuple[str, ...]:
        completed = frozenset(self.completed_views)
        return tuple(view for view in self.required_views if view not in completed)

    @property
    def complete(self) -> bool:
        return self.completed_count == self.required_count


@dataclass(frozen=True, slots=True)
class ArtifactWorkflowProgress:
    """Sequential Open/Align/Cutline/Outline/Rubbing workflow projection."""

    align_ready: bool
    cutline: ArtifactWorkflowStepProgress
    outline: ArtifactWorkflowStepProgress
    rubbing: ArtifactWorkflowStepProgress

    def __post_init__(self) -> None:
        if not isinstance(self.align_ready, bool):
            raise TypeError("align_ready must be a bool")
        expected = (
            (self.cutline, ArtifactWorkflowStep.CUTLINE),
            (self.outline, ArtifactWorkflowStep.OUTLINE),
            (self.rubbing, ArtifactWorkflowStep.DIGITAL_RUBBING),
        )
        if any(
            not isinstance(progress, ArtifactWorkflowStepProgress)
            or progress.step is not step
            for progress, step in expected
        ):
            raise TypeError("workflow progress contains a mismatched step")
        if (
            self.cutline.required_views != REQUIRED_CUTLINE_VIEWS
            or self.outline.required_views != REQUIRED_SIX_VIEWS
            or self.rubbing.required_views != REQUIRED_SIX_VIEWS
        ):
            raise ValueError("workflow progress uses unsupported required views")
        expected_enabled = (
            self.align_ready,
            self.align_ready and self.cutline.complete,
            self.align_ready and self.cutline.complete and self.outline.complete,
        )
        observed_enabled = (
            self.cutline.enabled,
            self.outline.enabled,
            self.rubbing.enabled,
        )
        if observed_enabled != expected_enabled:
            raise ValueError("workflow progress gates are inconsistent")

    @classmethod
    def empty(cls, *, align_ready: bool = False) -> "ArtifactWorkflowProgress":
        return _progress_from_completed_views(
            align_ready=align_ready,
            completed_cutline=frozenset(),
            completed_outline=frozenset(),
            completed_rubbing=frozenset(),
        )

    def for_step(self, step: ArtifactWorkflowStep) -> ArtifactWorkflowStepProgress:
        resolved = ArtifactWorkflowStep(step)
        if resolved is ArtifactWorkflowStep.CUTLINE:
            return self.cutline
        if resolved is ArtifactWorkflowStep.OUTLINE:
            return self.outline
        return self.rubbing


def _cutline_view(recipe: Mapping[str, object]) -> str | None:
    if recipe.get("kind") != VectorRecordKind.CUTLINE.value:
        return None
    raw_frame = recipe.get("frame")
    if not isinstance(raw_frame, Mapping):
        return None
    try:
        frame = PlanarFrame.from_dict(raw_frame)
    except (ArtifactVectorRecordError, TypeError, ValueError):
        return None
    axes = (
        frame.u_axis_world,
        frame.v_axis_world,
        frame.normal_world,
    )
    for view, expected in _CUTLINE_AXES.items():
        if axes == expected:
            return view
    return None


def _declared_view(
    recipe: Mapping[str, object],
    *,
    expected_kind: str,
    required_views: tuple[str, ...],
) -> str | None:
    if recipe.get("kind") != expected_kind:
        return None
    view = recipe.get("view")
    if isinstance(view, str) and view in required_views:
        return view
    return None


def _ordered_completed_views(
    required_views: tuple[str, ...],
    completed: frozenset[str],
) -> tuple[str, ...]:
    return tuple(view for view in required_views if view in completed)


def _progress_from_completed_views(
    *,
    align_ready: bool,
    completed_cutline: frozenset[str],
    completed_outline: frozenset[str],
    completed_rubbing: frozenset[str],
) -> ArtifactWorkflowProgress:
    cutline_views = _ordered_completed_views(
        REQUIRED_CUTLINE_VIEWS,
        completed_cutline,
    )
    outline_views = _ordered_completed_views(
        REQUIRED_SIX_VIEWS,
        completed_outline,
    )
    rubbing_views = _ordered_completed_views(
        REQUIRED_SIX_VIEWS,
        completed_rubbing,
    )
    cutline_complete = len(cutline_views) == len(REQUIRED_CUTLINE_VIEWS)
    outline_complete = len(outline_views) == len(REQUIRED_SIX_VIEWS)
    return ArtifactWorkflowProgress(
        align_ready=align_ready,
        cutline=ArtifactWorkflowStepProgress(
            step=ArtifactWorkflowStep.CUTLINE,
            required_views=REQUIRED_CUTLINE_VIEWS,
            completed_views=cutline_views,
            enabled=align_ready,
        ),
        outline=ArtifactWorkflowStepProgress(
            step=ArtifactWorkflowStep.OUTLINE,
            required_views=REQUIRED_SIX_VIEWS,
            completed_views=outline_views,
            enabled=align_ready and cutline_complete,
        ),
        rubbing=ArtifactWorkflowStepProgress(
            step=ArtifactWorkflowStep.DIGITAL_RUBBING,
            required_views=REQUIRED_SIX_VIEWS,
            completed_views=rubbing_views,
            enabled=align_ready and cutline_complete and outline_complete,
        ),
    )


def derive_artifact_workflow_progress(
    session: ArtifactSession,
    *,
    align_ready: bool,
) -> ArtifactWorkflowProgress:
    """Derive sequential gates from one validated session's record graph."""

    if not isinstance(session, ArtifactSession):
        raise TypeError("session must be an ArtifactSession")
    if not isinstance(align_ready, bool):
        raise TypeError("align_ready must be a bool")

    document = session.document
    freshnesses = document.record_freshnesses()
    completed_cutline: set[str] = set()
    completed_outline: set[str] = set()
    completed_rubbing: set[str] = set()
    for record in document.records:
        if record.lifecycle_status is not RecordLifecycleStatus.READY:
            continue
        if freshnesses.get(record.id) is not RecordFreshness.FRESH:
            continue
        if record.type == _CUTLINE_RECORD_TYPE:
            view = _cutline_view(record.recipe)
            if view is not None:
                completed_cutline.add(view)
        elif record.type == _OUTLINE_RECORD_TYPE:
            view = _declared_view(
                record.recipe,
                expected_kind=VectorRecordKind.OUTLINE.value,
                required_views=REQUIRED_SIX_VIEWS,
            )
            if view is not None:
                completed_outline.add(view)
        elif record.type == RUBBING_RECORD_TYPE:
            view = _declared_view(
                record.recipe,
                expected_kind="digital_rubbing",
                required_views=REQUIRED_SIX_VIEWS,
            )
            if view is not None:
                completed_rubbing.add(view)

    return _progress_from_completed_views(
        align_ready=align_ready,
        completed_cutline=frozenset(completed_cutline),
        completed_outline=frozenset(completed_outline),
        completed_rubbing=frozenset(completed_rubbing),
    )


__all__ = [
    "ArtifactWorkflowProgress",
    "ArtifactWorkflowStep",
    "ArtifactWorkflowStepProgress",
    "REQUIRED_CUTLINE_VIEWS",
    "REQUIRED_SIX_VIEWS",
    "derive_artifact_workflow_progress",
]
