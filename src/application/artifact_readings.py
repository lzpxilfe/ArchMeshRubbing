"""The readings a drawing needs, made where the artifact is open.

The plate can draw a ridge, a groove, a corner, a far silhouette, the
shade of a relief, the lines of a pattern and the paint cut out of the
colour.  Until now nothing in the application could produce any of them:
the records existed, the composer drew them, and the only way to get one
was to write Python.  That is the gap this closes.

Each reading is one call: compute from the session as the active Align
stands it, then commit as a record with the archaeologist's name and the
moment.  The rules are the core's - a groove is read on a vessel stood on
its rotation axis, a corner belongs to one surface, a far silhouette is
the half behind a view - and this layer adds none of its own.  What it
adds is one shape for all seven, so the window and the tests do not each
learn seven different calls, and so a reading refused says which reading
and why.

Two of the seven read the scanner's images rather than the mesh alone -
the pattern is in the texture and nowhere else in the file - so this layer
opens the OBJ's texture coordinates and the map beside them.  That is the
only file-reading it does, and it is exactly what the core asks for.
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
from src.core.artifact_paint_cutout import (
    ArtifactPaintCutoutError,
    commit_paint_cutout,
    compute_paint_cutout,
)
from src.core.artifact_session import ArtifactSession
from src.core.artifact_texture_lines import (
    ArtifactTextureLinesError,
    commit_texture_lines,
    compute_texture_lines,
)
from src.core.artifact_texture_paint import ArtifactTexturePaintError, read_colour_map
from src.core.artifact_texture_relief import (
    ArtifactTextureReliefError,
    read_normal_map,
    read_obj_texture_atlas,
)


class ArtifactReadingError(RuntimeError):
    """A reading could not be taken, or could not be recorded."""


#: The readings this layer can take, in the order a drawing usually wants
#: them: the shape of the profile first, then what is on the wall.
CREASE = "crease"
PROFILE_BREAK = "profile_break"
PROFILE_GROOVE = "profile_groove"
FAR_SILHOUETTE = "far_silhouette"
RELIEF_SHADE = "relief_shade"
TEXTURE_LINES = "texture_lines"
PAINT_CUTOUT = "paint_cutout"
READING_KINDS: tuple[str, ...] = (
    PROFILE_BREAK,
    PROFILE_GROOVE,
    CREASE,
    FAR_SILHOUETTE,
    RELIEF_SHADE,
    TEXTURE_LINES,
    PAINT_CUTOUT,
)

#: The readings that read the scanner's own images rather than the mesh
#: alone.  They need the OBJ's texture coordinates and one map beside it,
#: which is the only thing about them this layer has to arrange.
TEXTURE_READINGS: frozenset[str] = frozenset({TEXTURE_LINES, PAINT_CUTOUT})

#: What each reading is called on the page, and what it is for.
READING_LABELS: dict[str, tuple[str, str]] = {
    PROFILE_BREAK: ("단면 꺾임", "굽 경계와 돌출 모서리를 입면의 수평 내선으로"),
    PROFILE_GROOVE: ("홈 (침선)", "벽을 한 바퀴 도는 홈을 골과 두 능선으로"),
    CREASE: ("능선", "석기의 격지면 사이 볼록 주름을 내선으로"),
    FAR_SILHOUETTE: ("뒷면 실루엣", "뷰 평면 뒤 반쪽의 윤곽 — 단면 쪽 뒷선에 씀"),
    RELIEF_SHADE: ("양각 음영", "양각 문양의 음영 — 도판에서 점묘로 찍음"),
    TEXTURE_LINES: (
        "문양 내선",
        "법선 지도의 시문선(음각·양각), 또는 색 지도의 채색선 — 입면의 내선으로",
    ),
    PAINT_CUTOUT: (
        "채색 따내기",
        "색 지도에서 채색된 자리만 따서 도형 위 제자리에 붙임",
    ),
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
            TEXTURE_LINES: ("line_count", "선"),
        }
        if self.kind in counts:
            key, word = counts[self.kind]
            return f"{READING_LABELS[self.kind][0]}: {word} {self.qc.get(key, 0)}개"
        if self.kind == FAR_SILHOUETTE:
            return f"뒷면 실루엣: 면 {self.qc.get('far_face_count', 0)}개"
        if self.kind == PAINT_CUTOUT:
            painted = self.qc.get("texture_paint_painted_pixel_count", 0)
            return f"채색 따내기: 채색 픽셀 {painted}개"
        return f"양각 음영: 먹이 닿은 픽셀 {self.qc.get('trusted_pixel_count', 0)}개"


def _computer(kind: str) -> Callable[..., Any]:
    return {
        CREASE: compute_crease_reading,
        PROFILE_BREAK: compute_artifact_profile_breaks,
        PROFILE_GROOVE: compute_artifact_profile_grooves,
        FAR_SILHOUETTE: compute_artifact_far_silhouette,
        RELIEF_SHADE: compute_relief_shade,
        TEXTURE_LINES: compute_texture_lines,
        PAINT_CUTOUT: compute_paint_cutout,
    }[kind]


def _committer(kind: str) -> Callable[..., ArtifactSession]:
    return {
        CREASE: commit_crease_reading,
        PROFILE_BREAK: commit_profile_breaks,
        PROFILE_GROOVE: commit_profile_grooves,
        FAR_SILHOUETTE: commit_far_silhouette,
        RELIEF_SHADE: commit_relief_shade,
        TEXTURE_LINES: commit_texture_lines,
        PAINT_CUTOUT: commit_paint_cutout,
    }[kind]


def _texture_inputs(kind: str, arguments: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    """Read the OBJ's texture coordinates and the map beside them.

    This is the one thing these two readings need that the other five do
    not: the pattern is in the scanner's images, not in the mesh, so the
    files have to be opened before the core can be asked anything.  Which
    map it is decides what is read - a normal map gives the incised or
    raised lines, a colour map the painted ones - and the paint cutout can
    only be read from colour.
    """

    atlas_path = str(arguments.pop("atlas_path", "") or "").strip()
    normal_map_path = str(arguments.pop("normal_map_path", "") or "").strip()
    colour_map_path = str(arguments.pop("colour_map_path", "") or "").strip()
    if not atlas_path:
        raise ArtifactReadingError(
            "텍스처 좌표가 있는 OBJ 파일이 필요합니다 (atlas_path)"
        )
    if kind == PAINT_CUTOUT and not colour_map_path:
        raise ArtifactReadingError("채색 따내기에는 색 지도가 필요합니다 (colour_map_path)")
    if kind == TEXTURE_LINES and not (normal_map_path or colour_map_path):
        raise ArtifactReadingError(
            "문양 내선에는 법선 지도나 색 지도 가운데 하나가 필요합니다"
        )
    try:
        atlas = read_obj_texture_atlas(atlas_path)
    except (ArtifactTextureReliefError, OSError) as exc:
        raise ArtifactReadingError(f"OBJ의 텍스처 좌표를 읽지 못했습니다: {exc}") from exc
    try:
        if not colour_map_path:
            return atlas, read_normal_map(normal_map_path), {}
        colour_map = read_colour_map(colour_map_path)
    except (ArtifactTextureReliefError, ArtifactTexturePaintError, OSError) as exc:
        raise ArtifactReadingError(f"지도 이미지를 읽지 못했습니다: {exc}") from exc
    if kind == PAINT_CUTOUT:
        # The cutout is read from colour and nothing else, so the colour map
        # is its second argument.
        return atlas, colour_map, {}
    # Painted lines: the tracer still takes a normal map in that position,
    # and reads colour instead when it is given one by name.
    normal_map = None
    if normal_map_path:
        try:
            normal_map = read_normal_map(normal_map_path)
        except (ArtifactTextureReliefError, OSError) as exc:
            raise ArtifactReadingError(f"법선 지도를 읽지 못했습니다: {exc}") from exc
    return atlas, normal_map, {"colour_map": colour_map}


_ERRORS = (
    ArtifactCreaseRecordError,
    ArtifactFarSilhouetteError,
    ArtifactPaintCutoutError,
    ArtifactProfileBreakError,
    ArtifactProfileGrooveError,
    ArtifactReliefShadeError,
    ArtifactTextureLinesError,
    ArtifactTexturePaintError,
    ArtifactTextureReliefError,
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
        elif kind in TEXTURE_READINGS:
            atlas, texture_map, extra = _texture_inputs(kind, arguments)
            computation = _computer(kind)(
                session,
                atlas,
                texture_map,
                cancellation_probe=cancellation_probe,
                **extra,
                **arguments,
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
    "PAINT_CUTOUT",
    "PROFILE_BREAK",
    "PROFILE_GROOVE",
    "READING_KINDS",
    "READING_LABELS",
    "RELIEF_SHADE",
    "TEXTURE_LINES",
    "TEXTURE_READINGS",
    "ArtifactReadingError",
    "ReadingOutcome",
    "take_reading",
]
