"""The readings a drawing needs, made where the artifact is open.

The plate can draw a ridge, a groove, a corner, a far silhouette and the
shade of a relief.  Until now nothing in the application could produce any
of them: the records existed, the composer drew them, and the only way to
get one was to write Python.  That is the gap this closes.

Each reading is one call: compute from the session as the active Align
stands it, then commit as a record with the archaeologist's name and the
moment.  The rules are the core's - a groove is read on a vessel stood on
its rotation axis, a corner belongs to one surface, a far silhouette is
the half behind a view - and this layer adds none of its own.  What it
adds is one shape for all five, so the window and the tests do not each
learn five different calls, and so a reading refused says which reading
and why.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.core.artifact_cancellation import CancellationProbe
from src.core.artifact_crease_record import (
    ArtifactCreaseRecordError,
    commit_crease_reading,
    compute_crease_reading,
)
from src.core.artifact_far_silhouette import (
    ArtifactFarSilhouetteError,
    commit_far_silhouette,
    compute_artifact_far_silhouette,
)
from src.core.artifact_profile_break import (
    ArtifactProfileBreakError,
    commit_profile_breaks,
    compute_artifact_profile_breaks,
)
from src.core.artifact_profile_groove import (
    ArtifactProfileGrooveError,
    commit_profile_grooves,
    compute_artifact_profile_grooves,
)
from src.core.artifact_relief_shade import (
    ArtifactReliefShadeError,
    commit_relief_shade,
    compute_relief_shade,
)
from src.core.artifact_session import ArtifactSession


class ArtifactReadingError(RuntimeError):
    """A reading could not be taken, or could not be recorded."""


#: The readings this layer can take, in the order a drawing usually wants
#: them: the shape of the profile first, then what is on the wall.
CREASE = "crease"
PROFILE_BREAK = "profile_break"
PROFILE_GROOVE = "profile_groove"
FAR_SILHOUETTE = "far_silhouette"
RELIEF_SHADE = "relief_shade"
READING_KINDS: tuple[str, ...] = (
    PROFILE_BREAK,
    PROFILE_GROOVE,
    CREASE,
    FAR_SILHOUETTE,
    RELIEF_SHADE,
)

#: What each reading is called on the page, and what it is for.
READING_LABELS: dict[str, tuple[str, str]] = {
    PROFILE_BREAK: ("단면 꺾임", "굽 경계와 돌출 모서리를 입면의 수평 내선으로"),
    PROFILE_GROOVE: ("홈 (침선)", "벽을 한 바퀴 도는 홈을 골과 두 능선으로"),
    CREASE: ("능선", "석기의 격지면 사이 볼록 주름을 내선으로"),
    FAR_SILHOUETTE: ("뒷면 실루엣", "뷰 평면 뒤 반쪽의 윤곽 — 단면 쪽 뒷선에 씀"),
    RELIEF_SHADE: ("양각 음영", "양각 문양의 음영 — 도판에서 점묘로 찍음"),
}


@dataclass(frozen=True, slots=True)
class ReadingOutcome:
    """A committed reading: the session that now holds it, and what it says."""

    session: ArtifactSession
    record_id: str
    kind: str
    qc: Mapping[str, Any]
    raster: Any = None
    """The shade's pixels, which its record stores a receipt of, not itself."""

    @property
    def summary(self) -> str:
        """One line for a status bar: what was read, in the reading's terms."""

        counts = {
            PROFILE_BREAK: ("break_count", "꺾임"),
            PROFILE_GROOVE: ("groove_count", "홈"),
            CREASE: ("chain_count", "능선"),
        }
        if self.kind in counts:
            key, word = counts[self.kind]
            return f"{READING_LABELS[self.kind][0]}: {word} {self.qc.get(key, 0)}개"
        if self.kind == FAR_SILHOUETTE:
            return f"뒷면 실루엣: 면 {self.qc.get('far_face_count', 0)}개"
        return f"양각 음영: 먹이 닿은 픽셀 {self.qc.get('trusted_pixel_count', 0)}개"


def _computer(kind: str) -> Callable[..., Any]:
    return {
        CREASE: compute_crease_reading,
        PROFILE_BREAK: compute_artifact_profile_breaks,
        PROFILE_GROOVE: compute_artifact_profile_grooves,
        FAR_SILHOUETTE: compute_artifact_far_silhouette,
        RELIEF_SHADE: compute_relief_shade,
    }[kind]


def _committer(kind: str) -> Callable[..., ArtifactSession]:
    return {
        CREASE: commit_crease_reading,
        PROFILE_BREAK: commit_profile_breaks,
        PROFILE_GROOVE: commit_profile_grooves,
        FAR_SILHOUETTE: commit_far_silhouette,
        RELIEF_SHADE: commit_relief_shade,
    }[kind]


_ERRORS = (
    ArtifactCreaseRecordError,
    ArtifactFarSilhouetteError,
    ArtifactProfileBreakError,
    ArtifactProfileGrooveError,
    ArtifactReliefShadeError,
)


def take_reading(
    session: ArtifactSession,
    kind: str,
    *,
    record_id: str,
    created_at: str,
    operator: str,
    options: Mapping[str, Any] | None = None,
    depends_on_record_ids: Sequence[str] = (),
    cancellation_probe: CancellationProbe | None = None,
) -> ReadingOutcome:
    """Take one reading of the artifact and record it.

    ``options`` are the reading's own - a corner's least angle, a groove's
    least depth, a silhouette's grid, a shade's window - and are passed to
    the core untouched, so this layer can never quietly change what a
    reading means.  A refusal names the reading.
    """

    if not isinstance(session, ArtifactSession):
        raise ArtifactReadingError("열린 ArtifactDocument 세션이 없습니다")
    if kind not in READING_KINDS:
        raise ArtifactReadingError(f"모르는 판독입니다: {kind!r}")
    if not str(record_id).strip():
        raise ArtifactReadingError("판독에는 record id가 필요합니다")
    if record_id in session.document.record_index:
        raise ArtifactReadingError(f"{record_id!r}는 이미 이 문서에 있는 기록입니다")
    label = READING_LABELS[kind][0]
    arguments = dict(options or {})
    try:
        if kind == FAR_SILHOUETTE:
            view = arguments.pop("view", "front")
            computation = _computer(kind)(
                session, view, cancellation_probe=cancellation_probe, **arguments
            )
        else:
            computation = _computer(kind)(
                session, cancellation_probe=cancellation_probe, **arguments
            )
    except _ERRORS as exc:
        raise ArtifactReadingError(f"{label}을 읽지 못했습니다: {exc}") from exc
    except (TypeError, ValueError) as exc:
        raise ArtifactReadingError(f"{label}의 설정이 맞지 않습니다: {exc}") from exc
    try:
        committed = _committer(kind)(
            session,
            computation,
            record_id=record_id,
            created_at=created_at,
            operator=operator,
            depends_on_record_ids=tuple(depends_on_record_ids),
        )
    except _ERRORS as exc:
        raise ArtifactReadingError(f"{label}을 기록하지 못했습니다: {exc}") from exc
    return ReadingOutcome(
        session=committed,
        record_id=str(record_id),
        kind=kind,
        qc=dict(getattr(computation, "qc", {}) or {}),
        raster=getattr(computation, "raster", None),
    )


__all__ = [
    "CREASE",
    "FAR_SILHOUETTE",
    "PROFILE_BREAK",
    "PROFILE_GROOVE",
    "READING_KINDS",
    "READING_LABELS",
    "RELIEF_SHADE",
    "ArtifactReadingError",
    "ReadingOutcome",
    "take_reading",
]
