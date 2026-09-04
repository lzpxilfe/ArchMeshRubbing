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

#: The source id of a preset whose weights the drafter typed in.  It is not a
#: published convention any more than the shipped provisional one is, so it
#: is provisional too; what distinguishes it is that the drawing carries its
#: full definition, since no registry holds it.
USER_PRESET_SOURCE_ID = "user"
USER_PRESET_ID_PREFIX = "user/"
#: One PostScript point in millimetres.  Illustrator and most report styles
#: state weights in points; the contract is paper millimetres.
POINT_MM = 25.4 / 72.0


class DrawingStyleError(ValueError):
    """A drawing style vocabulary or preset is not usable as declared."""


def pt_to_mm(points: float) -> float:
    return float(points) * POINT_MM


def mm_to_pt(millimetres: float) -> float:
    return float(millimetres) / POINT_MM


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
TECHNIQUE_GROOVE_EDGE = "technique_groove_edge"
TECHNIQUE_GROOVE_TROUGH = "technique_groove_trough"
# The marks a potter's tools and hands left, recorded as painted regions
# (annotation.technique.v1) and drawn by their conventions.
TECHNIQUE_COIL_JOINT = "technique_coil_joint"
TECHNIQUE_FINGER_MARK = "technique_finger_mark"
TECHNIQUE_PADDLING = "technique_paddling"
TECHNIQUE_WATER_SMOOTHING = "technique_water_smoothing"
TECHNIQUE_WOOD_GRAIN = "technique_wood_grain"
CENTER_AXIS = "center_axis"

LINE_KINDS: tuple[str, ...] = (
    SECTION_CUT,
    OUTLINE_VISIBLE,
    OUTLINE_HOLE,
    # Technique sits on the surface, so it is drawn over the outline; and it is
    # drawn under condition, because condition says which part of the drawing
    # can be read at all.  A technique line lost under a restored patch is the
    # right outcome: nobody should read a maker's tooling off a modern repair.
    TECHNIQUE_GROOVE_EDGE,
    TECHNIQUE_GROOVE_TROUGH,
    TECHNIQUE_COIL_JOINT,
    TECHNIQUE_FINGER_MARK,
    TECHNIQUE_PADDLING,
    TECHNIQUE_WATER_SMOOTHING,
    TECHNIQUE_WOOD_GRAIN,
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

# What the drafter calls each kind.  The ids above are the contract; these are
# the words on the panel and in the docs, in the vocabulary of Korean report
# drawing: 외선 the outer contour, 내선 an inner contour, 단면선 the cut, 간선 a
# line broken a few times, 직선 a continuous technique line, 중심선 the axis.
LINE_KIND_LABELS_KO: Mapping[str, str] = {
    SECTION_CUT: "단면선",
    OUTLINE_VISIBLE: "외선 (외곽선)",
    OUTLINE_HOLE: "내선 (구멍·안쪽 윤곽)",
    TECHNIQUE_GROOVE_EDGE: "직선 (홈 가장자리)",
    TECHNIQUE_GROOVE_TROUGH: "간선 (홈 바닥)",
    TECHNIQUE_COIL_JOINT: "테쌓기흔",
    TECHNIQUE_FINGER_MARK: "지두흔 (U자)",
    TECHNIQUE_PADDLING: "타날흔",
    TECHNIQUE_WATER_SMOOTHING: "물손질흔",
    TECHNIQUE_WOOD_GRAIN: "목리조정흔",
    CONDITION_MISSING: "결실",
    CONDITION_RESTORED: "복원",
    CONDITION_WORN: "마모",
    CONDITION_CRACK: "균열",
    CENTER_AXIS: "중심선",
}

# 간선: the recessed line at the bottom of a groove is drawn as a straight line
# broken a few times, and how many times is the drafter's judgement - "two or
# three" is the usual answer.  A dash pattern cannot express that, because a
# pattern fixes the frequency of the breaks and not their number; so the line
# is emitted as separate collinear segments and the count lives here.
#
# Both numbers are provisional, like everything else in this module.
GROOVE_TROUGH_BREAK_COUNT = 2
GROOVE_TROUGH_BREAK_MM = 1.6

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


# How a technique record's kind is drawn.  Keys are the closed vocabulary of
# `artifact_technique_annotation.TECHNIQUE_KINDS`, repeated as literals for the
# same reason as the conditions; a test asserts the two agree.
TECHNIQUE_LINE_KINDS: Mapping[str, str] = {
    "coil_joint": TECHNIQUE_COIL_JOINT,
    "finger_mark": TECHNIQUE_FINGER_MARK,
    "paddling": TECHNIQUE_PADDLING,
    "water_smoothing": TECHNIQUE_WATER_SMOOTHING,
    "wood_grain_smoothing": TECHNIQUE_WOOD_GRAIN,
}


def line_kind_for_technique(kind: str) -> str:
    """Return the line kind one technique record is drawn as."""

    line_kind = TECHNIQUE_LINE_KINDS.get(str(kind))
    if line_kind is None:
        known = ", ".join(sorted(TECHNIQUE_LINE_KINDS))
        raise DrawingStyleError(
            f"technique kind {kind!r} has no drawing style; known kinds are {known}"
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

    @classmethod
    def from_dict(cls, value: object, *, field_name: str = "line style") -> "LineStyle":
        mapping = _closed_mapping(
            value, {"dash_pattern_mm", "hatch", "stroke_width_mm"}, field_name=field_name
        )
        dash = mapping["dash_pattern_mm"]
        if not isinstance(dash, Sequence) or isinstance(dash, (str, bytes)):
            raise DrawingStyleError(f"{field_name}.dash_pattern_mm must be a list")
        return cls(
            stroke_width_mm=_paper_mm(
                mapping["stroke_width_mm"], field_name=f"{field_name}.stroke_width_mm"
            ),
            dash_pattern_mm=tuple(
                _paper_mm(item, field_name=f"{field_name}.dash_pattern_mm[]")
                for item in dash
            ),
            hatch=mapping["hatch"],
        )


def _closed_mapping(
    value: object, keys: set[str], *, field_name: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DrawingStyleError(f"{field_name} must be an object")
    if set(value) != keys:
        raise DrawingStyleError(
            f"{field_name} must have exactly the keys {', '.join(sorted(keys))}"
        )
    return value


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

    @classmethod
    def from_dict(cls, value: object, *, field_name: str = "hatch") -> "HatchStyle":
        mapping = _closed_mapping(
            value, {"angle_deg", "spacing_mm", "stroke_width_mm"}, field_name=field_name
        )
        return cls(
            spacing_mm=mapping["spacing_mm"],
            stroke_width_mm=mapping["stroke_width_mm"],
            angle_deg=mapping["angle_deg"],
        )


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
        if not isinstance(self.hatch, HatchStyle):
            raise DrawingStyleError("hatch must be a HatchStyle")
        # A user preset is named by its content, so the same weights are always
        # the same preset and a drawing's claim about it can be checked.
        is_user = self.source_id == USER_PRESET_SOURCE_ID
        if is_user != preset_id.startswith(USER_PRESET_ID_PREFIX):
            raise DrawingStyleError(
                f"preset {preset_id!r}: only a user preset carries the "
                f"{USER_PRESET_ID_PREFIX!r} prefix, and every user preset does"
            )
        if is_user and preset_id != user_preset_id(self.lines, self.hatch):
            raise DrawingStyleError(
                f"user preset {preset_id!r} is not named by its own content"
            )

    @property
    def provisional(self) -> bool:
        """Whether this preset states a convention no published source backs.

        A preset the drafter typed in is provisional too: it is their choice,
        recorded with the drawing, not a convention a publication stands for.
        """

        return self.source_id is None or self.source_id == USER_PRESET_SOURCE_ID

    @property
    def is_user(self) -> bool:
        return self.source_id == USER_PRESET_SOURCE_ID

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

    @classmethod
    def from_dict(cls, value: object) -> "DrawingStylePreset":
        """Rebuild a preset from its own `to_dict`, refusing anything looser."""

        mapping = _closed_mapping(
            value,
            {"hatch", "lines", "preset_id", "provisional", "schema_version", "source_id"},
            field_name="drawing style preset",
        )
        if mapping["schema_version"] != DRAWING_STYLE_SCHEMA_VERSION:
            raise DrawingStyleError(
                f"drawing style preset schema must be {DRAWING_STYLE_SCHEMA_VERSION}"
            )
        lines_value = mapping["lines"]
        if not isinstance(lines_value, Mapping):
            raise DrawingStyleError("drawing style preset lines must be an object")
        lines = {
            str(kind): LineStyle.from_dict(style, field_name=f"lines[{kind!r}]")
            for kind, style in lines_value.items()
        }
        source_id = mapping["source_id"]
        if source_id is not None and not isinstance(source_id, str):
            raise DrawingStyleError("drawing style preset source_id must be a string or null")
        preset = cls(
            preset_id=str(mapping["preset_id"]),
            lines=lines,
            hatch=HatchStyle.from_dict(mapping["hatch"]),
            source_id=source_id,
        )
        if mapping["provisional"] is not preset.provisional:
            raise DrawingStyleError(
                f"drawing style preset {preset.preset_id!r} disagrees about being provisional"
            )
        return preset


def user_preset_id(lines: Mapping[str, LineStyle], hatch: HatchStyle) -> str:
    """The content-addressed id of a user preset with these lines and hatch."""

    digest = canonical_json_sha256(
        {
            "hatch": hatch.to_dict(),
            "lines": {kind: lines[kind].to_dict() for kind in sorted(lines)},
            "schema_version": DRAWING_STYLE_SCHEMA_VERSION,
            "source_id": USER_PRESET_SOURCE_ID,
        }
    )
    return f"{USER_PRESET_ID_PREFIX}{digest[:12]}"


def user_preset(
    stroke_widths_mm: Mapping[str, float],
    *,
    base_preset_id: str = "provisional/v1",
    hatch_cut_faces: bool | None = None,
) -> DrawingStylePreset:
    """A preset with the drafter's own weights on the base preset's dashes.

    ``stroke_widths_mm`` names any subset of the line kinds; kinds left out
    keep the base weight.  Dash patterns and the hatch geometry are the base
    preset's: a weight is the number a report style states, the rest is how
    this renderer tells the kinds apart.  The result is named by its content,
    so the same weights always give the same preset id.

    ``hatch_cut_faces`` says whether the cut face inside a section is shaded.
    Reports do it both ways - some hatch the cut, some leave it blank and let
    the section line carry it - and neither is a measurement, so it is the
    drafter's to choose.  ``None``, the default, keeps the base preset's
    choice, and a preset built that way is exactly the preset it was before
    this argument existed.  Only the cut face answers to it: the boundary of
    the section is drawn either way, and no other kind is hatched.
    """

    base = get_preset(base_preset_id)
    unknown = sorted(set(stroke_widths_mm) - set(LINE_KINDS))
    if unknown:
        raise DrawingStyleError(f"unknown line kinds: {', '.join(unknown)}")
    if hatch_cut_faces is not None and not isinstance(hatch_cut_faces, bool):
        raise DrawingStyleError("hatch_cut_faces must be a boolean or None")
    lines = {
        kind: (
            LineStyle(
                stroke_width_mm=_paper_mm(
                    stroke_widths_mm[kind], field_name=f"stroke_widths_mm[{kind!r}]"
                ),
                dash_pattern_mm=style.dash_pattern_mm,
                hatch=style.hatch,
            )
            if kind in stroke_widths_mm
            else style
        )
        for kind, style in base.lines.items()
    }
    if hatch_cut_faces is not None and SECTION_CUT in lines:
        cut = lines[SECTION_CUT]
        lines[SECTION_CUT] = LineStyle(
            stroke_width_mm=cut.stroke_width_mm,
            dash_pattern_mm=cut.dash_pattern_mm,
            hatch=hatch_cut_faces,
        )
    return DrawingStylePreset(
        preset_id=user_preset_id(lines, base.hatch),
        lines=lines,
        hatch=base.hatch,
        source_id=USER_PRESET_SOURCE_ID,
    )


def resolve_preset(value: object) -> DrawingStylePreset:
    """Turn a preset id, a preset, or a preset's dict into the preset.

    A registered id resolves through the registry; a dict is rebuilt and, if it
    names a registered preset, must match it byte for byte.
    """

    if isinstance(value, DrawingStylePreset):
        return value
    if isinstance(value, str):
        return get_preset(value)
    if isinstance(value, Mapping):
        preset = DrawingStylePreset.from_dict(value)
        registered = _PRESETS.get(preset.preset_id)
        if registered is not None and registered.sha256() != preset.sha256():
            raise DrawingStyleError(
                f"preset {preset.preset_id!r} is registered with different values"
            )
        return preset
    raise DrawingStyleError("style_preset must be a preset id, a preset, or its dict")


def preset_claim(preset: DrawingStylePreset) -> dict[str, Any]:
    """What a drawing records about the preset it was styled with.

    A registered preset is named and digested; a user preset also carries its
    full definition, because no registry anywhere else holds it.
    """

    claim: dict[str, Any] = {
        "preset_id": preset.preset_id,
        "provisional": preset.provisional,
        "sha256": preset.sha256(),
        "source_id": preset.source_id,
    }
    if preset.is_user:
        claim["definition"] = preset.to_dict()
    return claim


def preset_from_claim(claim: object) -> DrawingStylePreset:
    """Re-prove a drawing's preset claim and return the preset it names."""

    if not isinstance(claim, Mapping):
        raise DrawingStyleError("style preset claim must be an object")
    keys = set(claim)
    definition = claim.get("definition")
    expected = {"preset_id", "provisional", "sha256", "source_id"}
    if keys != expected and keys != expected | {"definition"}:
        raise DrawingStyleError(
            "style preset claim must have exactly the keys "
            + ", ".join(sorted(expected))
            + " and, for a user preset, definition"
        )
    preset_id = str(claim.get("preset_id") or "").strip()
    if not preset_id:
        raise DrawingStyleError("style preset claim names no preset")
    if "definition" in keys:
        preset = DrawingStylePreset.from_dict(definition)
        if not preset.is_user:
            raise DrawingStyleError(
                "only a user preset carries its definition in a drawing's claim"
            )
        if preset.preset_id != preset_id:
            raise DrawingStyleError(
                f"style preset claim names {preset_id!r} but defines {preset.preset_id!r}"
            )
    else:
        if preset_id.startswith(USER_PRESET_ID_PREFIX):
            raise DrawingStyleError(
                f"user preset {preset_id!r} is claimed without its definition"
            )
        preset = get_preset(preset_id)
    if claim.get("sha256") != preset.sha256():
        raise DrawingStyleError(
            f"drawing style preset {preset_id!r} no longer matches the digest "
            "recorded with this drawing"
        )
    if claim.get("provisional") is not preset.provisional:
        raise DrawingStyleError(
            f"drawing style preset {preset_id!r} disagrees about being provisional"
        )
    if claim.get("source_id") != preset.source_id:
        raise DrawingStyleError(
            f"drawing style preset {preset_id!r} disagrees about its source"
        )
    return preset


# The one preset that ships today.
#
# Every number here is provisional.  They are chosen to be legible at 1:1 on
# paper and to keep the kinds distinguishable from one another, not because a
# standard specifies them.  Replacing them with sourced values, and recording
# the source, is the work described in docs/DRAWING_CONVENTIONS.md; until then
# `provisional` is true and every drawing made with this preset says so.
PROVISIONAL_PRESET_ID = "provisional/v1"
#: Pen widths from 그림 27 of the 2013 한국문화유산협회 (then
#: 한국문화재조사연구기관협회) measured-drawing course text, docs/REFERENCES.md
#: [K1].  Named for what it follows, as docs/DRAWING_CONVENTIONS.md asks.
KCHA_2013_PEN_PRESET_ID = "kcha-2013-pen/v1"
KCHA_2013_SOURCE_ID = "K1"

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
            # A groove is one line that goes in and two that stand out.  Both
            # are drawn lighter than the outline, because they describe the
            # surface rather than the artifact's edge; the recessed one carries
            # no dash pattern because its breaks are cut into the geometry.
            TECHNIQUE_GROOVE_EDGE: LineStyle(stroke_width_mm=0.18),
            TECHNIQUE_GROOVE_TROUGH: LineStyle(stroke_width_mm=0.18),
            # Five technique marks, all with one fine pen: a mark is told
            # from the others by the strokes drawn for it (drawing_marks),
            # never by a dash pattern - a report drawing draws the marks,
            # not a coded boundary around them.  Provisional, like the rest.
            TECHNIQUE_COIL_JOINT: LineStyle(stroke_width_mm=0.13),
            TECHNIQUE_FINGER_MARK: LineStyle(stroke_width_mm=0.13),
            TECHNIQUE_PADDLING: LineStyle(stroke_width_mm=0.13),
            TECHNIQUE_WATER_SMOOTHING: LineStyle(stroke_width_mm=0.13),
            TECHNIQUE_WOOD_GRAIN: LineStyle(stroke_width_mm=0.13),
            CENTER_AXIS: LineStyle(
                stroke_width_mm=0.13,
                dash_pattern_mm=(4.0, 1.0, 1.0, 1.0),
            ),
        },
        hatch=HatchStyle(spacing_mm=1.0, stroke_width_mm=0.13, angle_deg=45.0),
        source_id=None,
    ),
    KCHA_2013_PEN_PRESET_ID: DrawingStylePreset(
        preset_id=KCHA_2013_PEN_PRESET_ID,
        lines={
            # 그림 27 (유물제도 펜 굵기), [K1] p.25: 단면 0.6, 평면·입면 0.4,
            # 결실부 0.1; the figure marks 0.3 on the emphasised lines
            # (돌대·실선) and 0.1 on the fine ones (허선, 내부 세부).
            SECTION_CUT: LineStyle(stroke_width_mm=0.6, hatch=True),
            OUTLINE_VISIBLE: LineStyle(stroke_width_mm=0.4),
            OUTLINE_HOLE: LineStyle(stroke_width_mm=0.4),
            # 결실부 0.1 ([K1] p.25).  A restored form is drawn dashed
            # ([K2] 도면 2 rim; [K1] p.12); the dash lengths are not stated
            # in the source and are the provisional preset's.
            CONDITION_MISSING: LineStyle(stroke_width_mm=0.1),
            CONDITION_RESTORED: LineStyle(
                stroke_width_mm=0.1, dash_pattern_mm=(3.0, 1.0, 0.5, 1.0)
            ),
            CONDITION_WORN: LineStyle(stroke_width_mm=0.1, dash_pattern_mm=(0.5, 0.5)),
            CONDITION_CRACK: LineStyle(stroke_width_mm=0.1),
            # 실선 (a raised 돌대·침선) 0.3 and 허선 0.1, [K1] p.19 and 그림 27.
            TECHNIQUE_GROOVE_EDGE: LineStyle(stroke_width_mm=0.3),
            TECHNIQUE_GROOVE_TROUGH: LineStyle(stroke_width_mm=0.1),
            # Technique marks are drawn with the fine pen ([K2] 도면 2, 3, 6).
            TECHNIQUE_COIL_JOINT: LineStyle(stroke_width_mm=0.1),
            TECHNIQUE_FINGER_MARK: LineStyle(stroke_width_mm=0.1),
            TECHNIQUE_PADDLING: LineStyle(stroke_width_mm=0.1),
            TECHNIQUE_WATER_SMOOTHING: LineStyle(stroke_width_mm=0.1),
            TECHNIQUE_WOOD_GRAIN: LineStyle(stroke_width_mm=0.1),
            # The centre line is the fine pen; a dotted centre line marks a
            # drawing reconstructed from a fragment ([K3] p.49).
            CENTER_AXIS: LineStyle(
                stroke_width_mm=0.1, dash_pattern_mm=(4.0, 1.0, 1.0, 1.0)
            ),
        },
        # The source does not give the hatch; these are the provisional
        # preset's, with the pen thinned to the source's fine line.
        hatch=HatchStyle(spacing_mm=1.0, stroke_width_mm=0.1, angle_deg=45.0),
        source_id=KCHA_2013_SOURCE_ID,
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
    "GROOVE_TROUGH_BREAK_COUNT",
    "GROOVE_TROUGH_BREAK_MM",
    "DrawingStyleError",
    "DrawingStylePreset",
    "HatchStyle",
    "LINE_KINDS",
    "LINE_KIND_LABELS_KO",
    "LineStyle",
    "OUTLINE_HOLE",
    "OUTLINE_VISIBLE",
    "POINT_MM",
    "PROVISIONAL_PRESET_ID",
    "RECORD_ROLE_LINE_KINDS",
    "SECTION_CUT",
    "KCHA_2013_PEN_PRESET_ID",
    "KCHA_2013_SOURCE_ID",
    "TECHNIQUE_COIL_JOINT",
    "TECHNIQUE_FINGER_MARK",
    "TECHNIQUE_LINE_KINDS",
    "TECHNIQUE_PADDLING",
    "TECHNIQUE_WATER_SMOOTHING",
    "TECHNIQUE_WOOD_GRAIN",
    "USER_PRESET_ID_PREFIX",
    "USER_PRESET_SOURCE_ID",
    "available_presets",
    "get_preset",
    "layer_id",
    "line_kind_for_condition",
    "line_kind_for_record_role",
    "mm_to_pt",
    "preset_claim",
    "preset_from_claim",
    "pt_to_mm",
    "resolve_preset",
    "user_preset",
    "user_preset_id",
]
