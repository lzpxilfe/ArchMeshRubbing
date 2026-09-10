"""A plate's specification: everything the composer was told, as a document.

A measured-drawing plate is the product of an artifact document and a set
of decisions - which records, at what scale, the fold and its steps round
a motif, the presumed lines, the cutouts pasted, the lines the archaeologist
struck out.  Those decisions live in `DrawingSheetOptions`; until now they
lived only in the code that built the options, so a plate could be verified
but not remade, and the archaeologist could not change one number and have
the plate again.

The specification is that set of decisions as canonical JSON: a closed
mapping, every option present, numbers as numbers, an infinite reach as the
word ``wall``.  It round-trips exactly (`plate_spec(*plate_spec_options(s))
== s` for any spec the composer wrote), it travels inside the plate's
sidecar, and the command line rebuilds a plate from a project and a spec.
Nothing in it can move a measurement: it names records and says how they
are laid out, and every value goes through the options' own validation.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .canonical_json import canonical_json_bytes, canonical_json_sha256
from .drawing_sheet import (
    DrawingSheetError,
    DrawingSheetOptions,
    Interpretation,
    SheetPage,
    TitleBlock,
)
from .drawing_style import (
    DrawingStyleError,
    DrawingStylePreset,
    get_preset,
    resolve_preset,
)

PLATE_SPEC_FORMAT = "archmeshrubbing.plate-spec/v1"
PLATE_SPEC_SCHEMA_VERSION = "1.0.0"
#: The word a spec writes for a fold step that goes to the silhouette.
REACH_TO_WALL = "wall"
MAX_PLATE_SPEC_BYTES = 1_000_000


class PlateSpecError(DrawingSheetError):
    """A plate specification is malformed or names what it cannot mean."""


# --- writing --------------------------------------------------------------


def _preset_value(preset: str | DrawingStylePreset) -> str | dict[str, Any]:
    """A registered preset by id; a user preset in full, as it is nowhere else."""

    resolved = resolve_preset(preset)
    return resolved.to_dict() if resolved.is_user else resolved.preset_id


def _reach_value(reach: float) -> float | str:
    return REACH_TO_WALL if math.isinf(reach) else float(reach)


def _decisions(entries: Iterable[Sequence[Any]]) -> list[list[Any]]:
    """A per-item decision map, written in a fixed order.

    Choices the composer looks up by key - a corner's style, a mark's angle,
    a pattern struck out - are a set, not a sequence: the same choices in
    another order are the same plate, and must be the same bytes.  Options
    whose order the drawing itself follows (the records, the cutouts pasted,
    the presumed lines and their numbering) keep the order they were given.
    """

    return sorted(([*entry] for entry in entries), key=lambda entry: [str(item) for item in entry])


def plate_spec(record_ids: Sequence[str], options: DrawingSheetOptions) -> dict[str, Any]:
    """The specification of a plate composed from ``record_ids`` with ``options``.

    Every option is written, defaults included, so a reader sees the whole
    decision and a spec is the same bytes whether an option was set to its
    default or left alone.
    """

    if not isinstance(options, DrawingSheetOptions):
        raise PlateSpecError("options must be DrawingSheetOptions")
    ids = [str(record_id) for record_id in record_ids]
    if not ids:
        raise PlateSpecError("a plate spec names at least one record")
    return {
        "break_reach": options.break_reach,
        "break_records": list(options.break_records),
        "break_solid_min_deg": int(options.break_solid_min_deg),
        "section_marks": _decisions(options.section_marks),
        "sherd_breaks": _decisions(options.sherd_breaks),
        "break_styles": _decisions(
            [record_id, int(index), style] for record_id, index, style in options.break_styles
        ),
        "center_axis_style": options.center_axis_style,
        "condition_records": list(options.condition_records),
        "crease_records": list(options.crease_records),
        "far_silhouettes": [list(pair) for pair in options.far_silhouettes],
        "format": PLATE_SPEC_FORMAT,
        "groove_records": list(options.groove_records),
        "gutter_mm": float(options.gutter_mm),
        "interpretation": options.interpretation.to_dict(),
        "line_cap": options.line_cap,
        "mirror_elevation_side": options.mirror_elevation_side,
        "mirror_jogs": [
            [record_id, float(along_from), float(along_to), _reach_value(reach)]
            for record_id, along_from, along_to, reach in options.mirror_jogs
        ],
        "mirror_sections": [list(pair) for pair in options.mirror_sections],
        "outline_reach": options.outline_reach,
        "page": {
            "margin_mm": float(options.page.margin_mm),
            "orientation": options.page.orientation,
            "size": options.page.size,
        },
        "paint_cutout_ink_percent": int(options.paint_cutout_ink_percent),
        "paint_cutouts": [list(entry) for entry in options.paint_cutouts],
        "plan_over_elevation": None if options.plan_over_elevation is None else list(options.plan_over_elevation),
        "plan_with_sections": None if options.plan_with_sections is None else list(options.plan_with_sections),
        "presumed_lines": [
            [record_id, kind, float(height), float(length)]
            for record_id, kind, height, length in options.presumed_lines
        ],
        "presumed_section": [
            [record_id, [[float(u), float(v)] for u, v in points]]
            for record_id, points in options.presumed_section
        ],
        "records": ids,
        "relief_developments_on_axis": [
            list(pair) for pair in options.relief_developments_on_axis
        ],
        "relief_stipples": [list(pair) for pair in options.relief_stipples],
        "rubbing_notes": _decisions(options.rubbing_notes),
        "rubbing_on_axis_fit": options.rubbing_on_axis_fit,
        "rubbing_on_axis_trim": options.rubbing_on_axis_trim,
        "rubbings_on_axis": [list(pair) for pair in options.rubbings_on_axis],
        "scale_denominator": float(options.scale_denominator),
        "schema_version": PLATE_SPEC_SCHEMA_VERSION,
        "show_center_axis": bool(options.show_center_axis),
        "stipple_dot_mm": float(options.stipple_dot_mm),
        "stipple_pitch_mm": float(options.stipple_pitch_mm),
        "stroke_color": options.stroke_color,
        "style_preset": _preset_value(options.style_preset),
        "technique_angles_deg": _decisions(
            [record_id, float(angle)] for record_id, angle in options.technique_angles_deg
        ),
        "technique_records": list(options.technique_records),
        "technique_representations": _decisions(options.technique_representations),
        "texture_line_hidden_patterns": _decisions(
            [record_id, int(index)] for record_id, index in options.texture_line_hidden_patterns
        ),
        "texture_line_records": list(options.texture_line_records),
        "title": options.title,
        "title_block": {
            "artifact_label": options.title_block.artifact_label,
            "rows": [list(row) for row in options.title_block.rows],
        },
    }


def plate_spec_bytes(spec: Mapping[str, Any]) -> bytes:
    """The spec as canonical JSON, the form it is hashed and written in."""

    return canonical_json_bytes(dict(spec))


def plate_spec_sha256(spec: Mapping[str, Any]) -> str:
    return canonical_json_sha256(dict(spec))


def write_plate_spec(path: str | Path, spec: Mapping[str, Any]) -> Path:
    """Write a spec to disk, indented for a person, keys in canonical order."""

    plate_spec_options(spec)  # never write what could not be read back
    out_path = Path(path)
    text = json.dumps(json.loads(plate_spec_bytes(spec).decode("utf-8")), ensure_ascii=False, indent=2)
    out_path.write_text(text + "\n", encoding="utf-8")
    return out_path


# --- reading --------------------------------------------------------------


_KNOWN_KEYS = frozenset(
    {
        "break_reach", "break_records", "break_solid_min_deg", "break_styles", "center_axis_style",
        "condition_records", "crease_records", "far_silhouettes", "format", "groove_records", "gutter_mm",
        "interpretation", "line_cap", "mirror_elevation_side", "mirror_jogs", "mirror_sections",
        "outline_reach", "page", "paint_cutout_ink_percent", "paint_cutouts", "plan_over_elevation",
        "plan_with_sections", "presumed_lines", "presumed_section", "records", "relief_developments_on_axis", "relief_stipples", "rubbing_notes",
        "rubbing_on_axis_fit", "rubbing_on_axis_trim", "rubbings_on_axis", "scale_denominator",
        "schema_version", "show_center_axis", "stipple_dot_mm", "stipple_pitch_mm", "stroke_color",
        "section_marks", "sherd_breaks", "style_preset", "technique_angles_deg", "technique_records",
        "technique_representations",
        "texture_line_hidden_patterns", "texture_line_records", "title", "title_block",
    }
)
_REQUIRED_KEYS = frozenset({"format", "schema_version", "records", "title_block"})
_INTERPRETATION_KEYS = frozenset(
    {"groove_edge_emphasis", "line_smoothing_mm", "note", "straight_far_edges", "stroke_straightening_deg"}
)


def _number(value: object, *, field_name: str, allow_wall: bool = False) -> float:
    if allow_wall and value == REACH_TO_WALL:
        return math.inf
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        wall = f" or {REACH_TO_WALL!r}" if allow_wall else ""
        raise PlateSpecError(f"plate spec {field_name} must be a finite number{wall}")
    return float(value)


def _integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlateSpecError(f"plate spec {field_name} must be an integer")
    return int(value)


def _text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise PlateSpecError(f"plate spec {field_name} must be text")
    return value


def _boolean(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise PlateSpecError(f"plate spec {field_name} must be true or false")
    return value


def _rows(value: object, *, field_name: str, width: int) -> list[list[Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise PlateSpecError(f"plate spec {field_name} must be a list")
    rows: list[list[Any]] = []
    for index, row in enumerate(value):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != width:
            raise PlateSpecError(f"plate spec {field_name}[{index}] must be a list of {width}")
        rows.append(list(row))
    return rows


def _texts(value: object, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise PlateSpecError(f"plate spec {field_name} must be a list of record ids")
    return tuple(_text(item, field_name=f"{field_name}[{index}]") for index, item in enumerate(value))


def _text_pairs(value: object, *, field_name: str) -> tuple[tuple[str, str], ...]:
    return tuple(
        (_text(a, field_name=f"{field_name}[{index}][0]"), _text(b, field_name=f"{field_name}[{index}][1]"))
        for index, (a, b) in enumerate(_rows(value, field_name=field_name, width=2))
    )


def _preset(value: object) -> str | DrawingStylePreset:
    try:
        if isinstance(value, str):
            return get_preset(value).preset_id
        if isinstance(value, Mapping):
            return resolve_preset(value)
    except DrawingStyleError as exc:
        raise PlateSpecError(str(exc)) from exc
    raise PlateSpecError("plate spec style_preset must be a registered preset id or a user preset's definition")


def plate_spec_options(spec: object) -> tuple[list[str], DrawingSheetOptions]:
    """The record ids and options a spec asks for, refusing anything looser.

    Unknown keys are refused - a misspelt option would otherwise be silently
    a default - and a key left out is that option's default.  Every value
    then goes through `DrawingSheetOptions` itself, so a spec can ask for
    nothing the composer would not accept.
    """

    if not isinstance(spec, Mapping):
        raise PlateSpecError("a plate spec must be an object")
    unknown = sorted(set(spec) - _KNOWN_KEYS)
    if unknown:
        raise PlateSpecError(f"plate spec has unknown keys: {', '.join(unknown)}")
    missing = sorted(_REQUIRED_KEYS - set(spec))
    if missing:
        raise PlateSpecError(f"plate spec lacks: {', '.join(missing)}")
    if spec["format"] != PLATE_SPEC_FORMAT:
        raise PlateSpecError(f"plate spec format must be {PLATE_SPEC_FORMAT!r}")
    if spec["schema_version"] != PLATE_SPEC_SCHEMA_VERSION:
        raise PlateSpecError(f"plate spec schema_version must be {PLATE_SPEC_SCHEMA_VERSION!r}")
    records = list(_texts(spec["records"], field_name="records"))
    if not records:
        raise PlateSpecError("plate spec records must name at least one record")

    title_block_value = spec["title_block"]
    if not isinstance(title_block_value, Mapping) or set(title_block_value) - {"artifact_label", "rows"}:
        raise PlateSpecError("plate spec title_block must be an object with artifact_label and rows")
    if "artifact_label" not in title_block_value:
        raise PlateSpecError("plate spec title_block lacks artifact_label")
    title_rows = tuple(
        (_text(label, field_name="title_block.rows"), _text(value, field_name="title_block.rows"))
        for label, value in _rows(title_block_value.get("rows", []), field_name="title_block.rows", width=2)
    )
    def given(key: str) -> bool:
        return key in spec

    kwargs: dict[str, Any] = {}
    try:
        kwargs["title_block"] = TitleBlock(
            artifact_label=_text(title_block_value["artifact_label"], field_name="title_block.artifact_label"),
            rows=title_rows,
        )
        page_value = spec.get("page")
        if page_value is not None:
            if not isinstance(page_value, Mapping) or set(page_value) - {"margin_mm", "orientation", "size"}:
                raise PlateSpecError("plate spec page must be an object of size, orientation and margin_mm")
            page_kwargs: dict[str, Any] = {}
            if "size" in page_value:
                page_kwargs["size"] = _text(page_value["size"], field_name="page.size")
            if "orientation" in page_value:
                page_kwargs["orientation"] = _text(page_value["orientation"], field_name="page.orientation")
            if "margin_mm" in page_value:
                page_kwargs["margin_mm"] = _number(page_value["margin_mm"], field_name="page.margin_mm")
            kwargs["page"] = SheetPage(**page_kwargs)
        interpretation_value = spec.get("interpretation")
        if interpretation_value is not None:
            if not isinstance(interpretation_value, Mapping) or set(interpretation_value) - _INTERPRETATION_KEYS:
                raise PlateSpecError(
                    "plate spec interpretation must be an object of "
                    f"{', '.join(sorted(_INTERPRETATION_KEYS))}"
                )
            interp_kwargs: dict[str, Any] = {}
            for key in ("groove_edge_emphasis", "line_smoothing_mm", "stroke_straightening_deg"):
                if key in interpretation_value:
                    interp_kwargs[key] = _number(interpretation_value[key], field_name=f"interpretation.{key}")
            if "straight_far_edges" in interpretation_value:
                interp_kwargs["straight_far_edges"] = _boolean(
                    interpretation_value["straight_far_edges"], field_name="interpretation.straight_far_edges"
                )
            if "note" in interpretation_value:
                interp_kwargs["note"] = _text(interpretation_value["note"], field_name="interpretation.note")
            kwargs["interpretation"] = Interpretation(**interp_kwargs)
    except PlateSpecError:
        raise
    except DrawingSheetError as exc:
        raise PlateSpecError(f"plate spec is not a plate the composer accepts: {exc}") from exc

    for key in (
        "break_reach", "center_axis_style", "line_cap", "mirror_elevation_side", "outline_reach",
        "rubbing_on_axis_fit", "rubbing_on_axis_trim", "stroke_color", "title",
    ):
        if given(key):
            kwargs[key] = _text(spec[key], field_name=key)
    for key in ("scale_denominator", "gutter_mm", "stipple_dot_mm", "stipple_pitch_mm"):
        if given(key):
            kwargs[key] = _number(spec[key], field_name=key)
    for key in ("break_solid_min_deg", "paint_cutout_ink_percent"):
        if given(key):
            kwargs[key] = _integer(spec[key], field_name=key)
    if given("show_center_axis"):
        kwargs["show_center_axis"] = _boolean(spec["show_center_axis"], field_name="show_center_axis")
    if given("style_preset"):
        kwargs["style_preset"] = _preset(spec["style_preset"])
    for key in (
        "break_records", "condition_records", "crease_records", "groove_records",
        "technique_records", "texture_line_records",
    ):
        if given(key):
            kwargs[key] = _texts(spec[key], field_name=key)
    for key in (
        "far_silhouettes", "mirror_sections", "relief_developments_on_axis", "relief_stipples",
        "rubbing_notes", "rubbings_on_axis", "technique_representations",
    ):
        if given(key):
            kwargs[key] = _text_pairs(spec[key], field_name=key)
    if given("mirror_jogs"):
        kwargs["mirror_jogs"] = tuple(
            (
                _text(record_id, field_name="mirror_jogs"),
                _number(along_from, field_name="mirror_jogs along_from_mm"),
                _number(along_to, field_name="mirror_jogs along_to_mm"),
                _number(reach, field_name="mirror_jogs reach_mm", allow_wall=True),
            )
            for record_id, along_from, along_to, reach in _rows(spec["mirror_jogs"], field_name="mirror_jogs", width=4)
        )
    if given("paint_cutouts"):
        kwargs["paint_cutouts"] = tuple(
            tuple(_text(item, field_name="paint_cutouts") for item in row)
            for row in _rows(spec["paint_cutouts"], field_name="paint_cutouts", width=3)
        )
    if given("presumed_lines"):
        kwargs["presumed_lines"] = tuple(
            (
                _text(record_id, field_name="presumed_lines"),
                _text(kind, field_name="presumed_lines kind"),
                _number(height, field_name="presumed_lines height_mm"),
                _number(length, field_name="presumed_lines length_mm"),
            )
            for record_id, kind, height, length in _rows(spec["presumed_lines"], field_name="presumed_lines", width=4)
        )
    if given("presumed_section"):
        kwargs["presumed_section"] = tuple(
            (
                _text(record_id, field_name="presumed_section"),
                tuple(
                    (
                        _number(point[0], field_name="presumed_section u_mm"),
                        _number(point[1], field_name="presumed_section v_mm"),
                    )
                    for point in _rows(
                        points, field_name="presumed_section polyline", width=2
                    )
                ),
            )
            for record_id, points in _rows(
                spec["presumed_section"], field_name="presumed_section", width=2
            )
        )
    if given("break_styles"):
        kwargs["break_styles"] = tuple(
            (
                _text(record_id, field_name="break_styles"),
                _integer(index, field_name="break_styles corner index"),
                _text(style, field_name="break_styles style"),
            )
            for record_id, index, style in _rows(spec["break_styles"], field_name="break_styles", width=3)
        )
    if given("section_marks"):
        kwargs["section_marks"] = tuple(
            (
                _text(section_id, field_name="section_marks"),
                _text(figure_id, field_name="section_marks figure"),
            )
            for section_id, figure_id in _rows(spec["section_marks"], field_name="section_marks", width=2)
        )
    if given("sherd_breaks"):
        kwargs["sherd_breaks"] = tuple(
            (
                _text(record_id, field_name="sherd_breaks"),
                _text(side, field_name="sherd_breaks side"),
            )
            for record_id, side in _rows(spec["sherd_breaks"], field_name="sherd_breaks", width=2)
        )
    if given("technique_angles_deg"):
        kwargs["technique_angles_deg"] = tuple(
            (_text(record_id, field_name="technique_angles_deg"), _number(angle, field_name="technique_angles_deg"))
            for record_id, angle in _rows(spec["technique_angles_deg"], field_name="technique_angles_deg", width=2)
        )
    if given("texture_line_hidden_patterns"):
        kwargs["texture_line_hidden_patterns"] = tuple(
            (
                _text(record_id, field_name="texture_line_hidden_patterns"),
                _integer(index, field_name="texture_line_hidden_patterns index"),
            )
            for record_id, index in _rows(spec["texture_line_hidden_patterns"], field_name="texture_line_hidden_patterns", width=2)
        )
    for key, width in (("plan_over_elevation", 2), ("plan_with_sections", 3)):
        if given(key) and spec[key] is not None:
            value = spec[key]
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != width:
                raise PlateSpecError(f"plate spec {key} must be null or a list of {width} record ids")
            kwargs[key] = tuple(_text(item, field_name=key) for item in value)
    try:
        options = DrawingSheetOptions(**kwargs)
    except DrawingSheetError as exc:
        raise PlateSpecError(f"plate spec is not a plate the composer accepts: {exc}") from exc
    return records, options


def read_plate_spec(path: str | Path) -> dict[str, Any]:
    """Read a spec file, refusing anything that is not one."""

    in_path = Path(path)
    size = in_path.stat().st_size
    if size > MAX_PLATE_SPEC_BYTES:
        raise PlateSpecError(f"plate spec file is larger than {MAX_PLATE_SPEC_BYTES} bytes")
    try:
        spec = json.loads(in_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise PlateSpecError(f"plate spec file is not valid JSON: {exc}") from exc
    plate_spec_options(spec)
    return spec


__all__ = [
    "MAX_PLATE_SPEC_BYTES",
    "PLATE_SPEC_FORMAT",
    "PLATE_SPEC_SCHEMA_VERSION",
    "REACH_TO_WALL",
    "PlateSpecError",
    "plate_spec",
    "plate_spec_bytes",
    "plate_spec_options",
    "plate_spec_sha256",
    "read_plate_spec",
    "write_plate_spec",
]
