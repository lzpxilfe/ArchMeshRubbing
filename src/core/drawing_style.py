"""Report drawing conventions as a closed, hash-locked presentation contract.

An archaeological measured drawing does not read as one uniform stroke.  A cut
face, a visible outline, an internal opening, a restored area and a centre axis
are different line kinds, and a reader identifies them by weight and dash before
reading a single label.  This module is where those kinds are named, and where a preset
binds each kind to the paper-millimetre weight and dash that draws it.

Two boundaries hold this apart from the measured geometry:

* Nothing here can move a coordinate.  A preset chooses stroke weight, dash and
  fill, all of which are presentation.  The path data is the record's, byte for
  byte, and a preset is applied on top of it.
* Nothing here is authoritative about a convention.  Every real drawing
  standard belongs to a published source, and a preset that claims one must
  carry its identifier.  A preset with `source_id=None` is provisional: it is a
  working default that lets the pipeline be built and tested, and it says so in
  the drawing's own provenance rather than passing itself off as a standard.

See `docs/DRAWING_CONVENTIONS.md` for how a sourced preset is added.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .canonical_json import canonical_json_sha256


DRAWING_STYLE_SCHEMA_VERSION = "1.0.0"


class DrawingStyleError(ValueError):
    """A drawing style vocabulary or preset is not usable as declared."""


# The closed line-kind vocabulary.  The order is the drawing order: later kinds
# paint over earlier ones, and it is also the order layers appear in the SVG, so
# that two renders of the same record always produce the same bytes.
#
# A kind is added here only when something can actually produce it.  Naming
# kinds the pipeline cannot yet draw would put empty layers in every drawing and
# invite a preset to describe conventions nothing honours.
SECTION_CUT = "section_cut"
OUTLINE_VISIBLE = "outline_visible"
OUTLINE_HOLE = "outline_hole"
CONDITION_MISSING = "condition_missing"
CONDITION_RESTORED = "condition_restored"
CONDITION_WORN = "condition_worn"
CONDITION_CRACK = "condition_crack"
CENTER_AXIS = "center_axis"

LINE_KINDS: tuple[str, ...] = (
    SECTION_CUT,
    OUTLINE_VISIBLE,
    OUTLINE_HOLE,
    # Condition sits between the outline and the centre axis: it is drawn over
    # the shape it describes, and under the axis, which is a construction line
    # and has to stay readable across everything.  Within the group the three
    # area conditions come before the crack, because a crack is a line on the
    # object and a line drawn under an area is a line a reader loses.
    CONDITION_MISSING,
    CONDITION_RESTORED,
    CONDITION_WORN,
    CONDITION_CRACK,
    CENTER_AXIS,
)

# How a condition record's kind is drawn.  The keys are the closed vocabulary of
# `artifact_condition_annotation.CONDITION_KINDS`, repeated as literals rather
# than imported: presentation must not depend on the record layer.  A test
# asserts the two agree, so the duplication cannot drift silently.
CONDITION_LINE_KINDS: Mapping[str, str] = {
    "crack": CONDITION_CRACK,
    "missing": CONDITION_MISSING,
    "restored": CONDITION_RESTORED,
    "worn": CONDITION_WORN,
}


def line_kind_for_condition(kind: str) -> str:
    """Return the line kind one condition record is drawn as.

    A condition boundary is stored as an outline, so its paths carry the
    `exterior` and `hole` roles.  Drawing it by role would print damage as the
    artifact's own outline; the record's kind is what decides the line.
    """

    line_kind = CONDITION_LINE_KINDS.get(str(kind))
    if line_kind is None:
        known = ", ".join(sorted(CONDITION_LINE_KINDS))
        raise DrawingStyleError(
            f"condition kind {kind!r} has no drawing style; known kinds are {known}"
        )
    return line_kind


# How the roles carried by an existing vector record map onto line kinds.
#
# Records keep their own role names, which are part of a payload hash that must
# never move.  The mapping is presentation, so it lives here rather than in the
# record layer.
RECORD_ROLE_LINE_KINDS: Mapping[str, str] = {
    "section": SECTION_CUT,
    "exterior": OUTLINE_VISIBLE,
    "hole": OUTLINE_HOLE,
}


def line_kind_for_record_role(role: str) -> str:
    """Return the line kind a record role is drawn as."""

    kind = RECORD_ROLE_LINE_KINDS.get(str(role))
    if kind is None:
        known = ", ".join(sorted(RECORD_ROLE_LINE_KINDS))
        raise DrawingStyleError(
            f"vector path role {role!r} has no drawing style; known roles are {known}"
        )
    return kind


def layer_id(kind: str) -> str:
    """Return the deterministic SVG group id for one line kind."""

    if kind not in LINE_KINDS:
        raise DrawingStyleError(f"unknown line kind: {kind!r}")
    return f"layer-{kind.replace('_', '-')}"


def _paper_mm(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DrawingStyleError(f"{field_name} must be a number of millimetres")
    number = float(value)
    if not (0.0 < number <= 10.0):
        raise DrawingStyleError(
            f"{field_name} must be greater than 0 and at most 10 millimetres"
        )
    return number


def _dash_pattern(value: object, *, field_name: str) -> tuple[float, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise DrawingStyleError(f"{field_name} must be a sequence of millimetres")
    pattern = tuple(_paper_mm(item, field_name=f"{field_name}[]") for item in value)
    if len(pattern) % 2 != 0:
        raise DrawingStyleError(
            f"{field_name} must alternate dash and gap lengths, so its length is even"
        )
    if not pattern:
        raise DrawingStyleError(f"{field_name} must not be an empty pattern; use None")
    return pattern


@dataclass(frozen=True, slots=True)
class LineStyle:
    """How one line kind is drawn, in paper millimetres at 1:1."""

    stroke_width_mm: float
    dash_pattern_mm: tuple[float, ...] = ()
    hatch: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stroke_width_mm",
            _paper_mm(self.stroke_width_mm, field_name="stroke_width_mm"),
        )
        object.__setattr__(
            self,
            "dash_pattern_mm",
            _dash_pattern(self.dash_pattern_mm or None, field_name="dash_pattern_mm")
            if self.dash_pattern_mm
            else (),
        )
        if not isinstance(self.hatch, bool):
            raise DrawingStyleError("hatch must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dash_pattern_mm": list(self.dash_pattern_mm),
            "hatch": self.hatch,
            "stroke_width_mm": self.stroke_width_mm,
        }


@dataclass(frozen=True, slots=True)
class HatchStyle:
    """The parallel-line fill drawn inside a cut face."""

    spacing_mm: float
    stroke_width_mm: float
    angle_deg: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "spacing_mm", _paper_mm(self.spacing_mm, field_name="spacing_mm")
        )
        object.__setattr__(
            self,
            "stroke_width_mm",
            _paper_mm(self.stroke_width_mm, field_name="stroke_width_mm"),
        )
        if isinstance(self.angle_deg, bool) or not isinstance(
            self.angle_deg, (int, float)
        ):
            raise DrawingStyleError("angle_deg must be a number of degrees")
        angle = float(self.angle_deg)
        if not (0.0 <= angle < 180.0):
            raise DrawingStyleError("angle_deg must be at least 0 and less than 180")
        object.__setattr__(self, "angle_deg", angle)
        if self.stroke_width_mm >= self.spacing_mm:
            raise DrawingStyleError(
                "hatch stroke_width_mm must be smaller than spacing_mm, "
                "or the fill prints as a solid block"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "angle_deg": self.angle_deg,
            "spacing_mm": self.spacing_mm,
            "stroke_width_mm": self.stroke_width_mm,
        }


@dataclass(frozen=True, slots=True)
class DrawingStylePreset:
    """One named set of line conventions, identified by its canonical hash."""

    preset_id: str
    lines: Mapping[str, LineStyle]
    hatch: HatchStyle
    source_id: str | None = None

    def __post_init__(self) -> None:
        preset_id = str(self.preset_id).strip()
        if not preset_id:
            raise DrawingStyleError("preset_id must be a non-empty string")
        object.__setattr__(self, "preset_id", preset_id)
        missing = sorted(set(LINE_KINDS) - set(self.lines))
        if missing:
            raise DrawingStyleError(
                f"preset {preset_id!r} does not style every line kind: "
                f"{', '.join(missing)}"
            )
        unknown = sorted(set(self.lines) - set(LINE_KINDS))
        if unknown:
            raise DrawingStyleError(
                f"preset {preset_id!r} styles unknown line kinds: {', '.join(unknown)}"
            )
        for kind, style in self.lines.items():
            if not isinstance(style, LineStyle):
                raise DrawingStyleError(f"style for {kind!r} must be a LineStyle")
        if self.source_id is not None and not str(self.source_id).strip():
            raise DrawingStyleError("source_id must be None or a non-empty string")
        object.__setattr__(self, "lines", dict(self.lines))

    @property
    def provisional(self) -> bool:
        """Whether this preset states a convention no published source backs."""

        return self.source_id is None

    def style(self, kind: str) -> LineStyle:
        try:
            return self.lines[kind]
        except KeyError as exc:
            raise DrawingStyleError(f"unknown line kind: {kind!r}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "hatch": self.hatch.to_dict(),
            "lines": {kind: self.lines[kind].to_dict() for kind in sorted(self.lines)},
            "preset_id": self.preset_id,
            "provisional": self.provisional,
            "schema_version": DRAWING_STYLE_SCHEMA_VERSION,
            "source_id": self.source_id,
        }

    def sha256(self) -> str:
        """Return the canonical digest a drawing records to prove its styling."""

        return canonical_json_sha256(self.to_dict())


# The one preset that ships today.
#
# Every number here is provisional.  They are chosen to be legible at 1:1 on
# paper and to keep the kinds distinguishable from one another, not because a
# standard specifies them.  Replacing them with sourced values, and recording
# the source, is the work described in docs/DRAWING_CONVENTIONS.md; until then
# `provisional` is true and every drawing made with this preset says so.
PROVISIONAL_PRESET_ID = "provisional/v1"

_PRESETS: dict[str, DrawingStylePreset] = {
    PROVISIONAL_PRESET_ID: DrawingStylePreset(
        preset_id=PROVISIONAL_PRESET_ID,
        lines={
            SECTION_CUT: LineStyle(stroke_width_mm=0.35, hatch=True),
            OUTLINE_VISIBLE: LineStyle(stroke_width_mm=0.25),
            OUTLINE_HOLE: LineStyle(stroke_width_mm=0.25),
            # Four dashes chosen only to stay apart from one another and from
            # the outline at 1:1.  No published convention is claimed; hatching
            # is not used because this renderer draws one hatch geometry for
            # every hatched kind, so two hatched kinds would print alike.
            CONDITION_MISSING: LineStyle(
                stroke_width_mm=0.25,
                dash_pattern_mm=(1.5, 1.5),
            ),
            CONDITION_RESTORED: LineStyle(
                stroke_width_mm=0.25,
                dash_pattern_mm=(3.0, 1.0, 0.5, 1.0),
            ),
            CONDITION_WORN: LineStyle(
                stroke_width_mm=0.18,
                dash_pattern_mm=(0.5, 0.5),
            ),
            CONDITION_CRACK: LineStyle(stroke_width_mm=0.3),
            CENTER_AXIS: LineStyle(
                stroke_width_mm=0.13,
                dash_pattern_mm=(4.0, 1.0, 1.0, 1.0),
            ),
        },
        hatch=HatchStyle(spacing_mm=1.0, stroke_width_mm=0.13, angle_deg=45.0),
        source_id=None,
    ),
}


def available_presets() -> tuple[str, ...]:
    """Return every preset id, in a stable order."""

    return tuple(sorted(_PRESETS))


def get_preset(preset_id: str) -> DrawingStylePreset:
    """Return one preset by id."""

    preset = _PRESETS.get(str(preset_id))
    if preset is None:
        known = ", ".join(available_presets())
        raise DrawingStyleError(
            f"unknown drawing style preset: {preset_id!r}; available presets are {known}"
        )
    return preset


__all__ = [
    "CENTER_AXIS",
    "CONDITION_CRACK",
    "CONDITION_LINE_KINDS",
    "CONDITION_MISSING",
    "CONDITION_RESTORED",
    "CONDITION_WORN",
    "DRAWING_STYLE_SCHEMA_VERSION",
    "DrawingStyleError",
    "DrawingStylePreset",
    "HatchStyle",
    "LINE_KINDS",
    "LineStyle",
    "OUTLINE_HOLE",
    "OUTLINE_VISIBLE",
    "PROVISIONAL_PRESET_ID",
    "RECORD_ROLE_LINE_KINDS",
    "SECTION_CUT",
    "available_presets",
    "get_preset",
    "layer_id",
    "line_kind_for_condition",
    "line_kind_for_record_role",
]
