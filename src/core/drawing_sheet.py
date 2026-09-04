"""Compose verified vector records into one printable measured-drawing sheet.

A 1:1 SVG export is the measurement.  A sheet is the page a reader receives: an
elevation and a section beside each other, reduced to a stated scale, with a
scale bar and a title block that says what they are looking at.  Those are the
parts a report figure cannot omit, and the parts a single-record export cannot
supply.

Three properties hold the sheet honest:

* **The scale is printed, always.**  A reduced drawing whose reduction is not
  stated cannot be measured off the page, and a caller cannot suppress the row
  that states it.
* **Weights are paper millimetres at every scale.**  Coordinates are divided by
  the scale denominator; stroke widths, dash lengths and hatch spacing are not.
  A 0.35 mm cut line is 0.35 mm on paper whether the sheet is 1:1 or 1:4.
* **It never silently shrinks to fit.**  Content that does not fit the page at
  the requested scale is an error naming what would have to change, because the
  alternative is a page that says 1:2 and measures as something else.

The sheet is presentation.  Each figure records the digest of the payload it
drew, so a sheet can be checked against the records it claims, but the sheet
itself is not a measurement and never becomes one.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field, replace
import hashlib
import math
from typing import Any, Mapping, Sequence

from .artifact_axis_alignment import AXIS_ALIGN_RECIPE_KIND
from .artifact_developed_rubbing import (
    ArtifactDevelopedRubbingError,
    DEVELOPED_RUBBING_RECORD_TYPE,
    DevelopedRubbingRaster,
    developed_rubbing_receipt_from_record,
)
from .artifact_rubbing_extractor import DigitalRubbingRaster
from .artifact_rubbing_record import (
    ArtifactRubbingRecordError,
    RUBBING_RECORD_TYPE,
    rubbing_receipt_from_record,
)
from .canonical_png import CanonicalPNGError, encode_canonical_ga8_png
from .artifact_condition_annotation import (
    ArtifactConditionAnnotationError,
    CONDITION_RECORD_TYPE,
    ConditionAnnotationPayload,
    condition_payload_from_record,
)
from .artifact_document import (
    ArtifactDocument,
    ArtifactDocumentError,
    DerivedRecord,
    RecordFreshness,
    RecordLifecycleStatus,
)
from .artifact_profile_groove import (
    ArtifactProfileGrooveError,
    PROFILE_GROOVE_RECORD_TYPE,
    ProfileGroovePayload,
    profile_groove_payload_from_record,
)
from .artifact_vector_export import (
    ArtifactVectorExportError,
    _payload_bounds,
    _require_exportable_record,
    center_axis_vector_path,
    profile_groove_vector_paths,
)
from .artifact_vector_record import (
    VectorGeometryPayload,
    VectorPath,
    VectorRecordKind,
)
from .canonical_json import canonical_json_bytes
from .drawing_style import (
    CENTER_AXIS,
    DrawingStyleError,
    DrawingStylePreset,
    get_preset as get_drawing_style_preset,
    line_kind_for_condition,
    line_kind_for_record_role,
)
from .drawing_svg import (
    Placement,
    SVG_NAMESPACE,
    SVGRenderError,
    center_axis_line,
    center_axis_segment,
    clip_closed_ring,
    clip_open_path,
    finite_number,
    hatch_pattern_elements,
    hatched_kinds,
    layer_elements,
    number_token,
    split_ring_off_line,
    xml_attribute,
)


DRAWING_SHEET_SCHEMA_VERSION = "1.0.0"
DRAWING_SHEET_FORMAT = "archmeshrubbing.drawing-sheet.svg/v1"
DRAWING_SHEET_SVG_NAME = "sheet.svg"
DRAWING_SHEET_SIDECAR_NAME = "sheet.provenance.json"

MAX_DRAWING_SHEET_SVG_BYTES = 64 * 1024 * 1024
MAX_DRAWING_SHEET_FIGURES = 32
MAX_DRAWING_SHEET_CONDITION_RECORDS = 64
MAX_DRAWING_SHEET_RASTER_BYTES = 24 * 1024 * 1024
RUBBING_RECORD_TYPES = frozenset({RUBBING_RECORD_TYPE, DEVELOPED_RUBBING_RECORD_TYPE})
RUBBING_ON_AXIS_FIT_HEIGHT = "axis_height"
RUBBING_ON_AXIS_FIT_PAPER = "paper"
RUBBING_ON_AXIS_FITS = (RUBBING_ON_AXIS_FIT_HEIGHT, RUBBING_ON_AXIS_FIT_PAPER)
DRAWING_SHEET_PNG_METADATA_FORMAT = "archmeshrubbing_drawing_sheet_png_metadata"

# ISO 216 sizes as portrait width x height in millimetres.
PAGE_SIZES_MM: Mapping[str, tuple[float, float]] = {
    "A5": (148.0, 210.0),
    "A4": (210.0, 297.0),
    "A3": (297.0, 420.0),
    "A2": (420.0, 594.0),
    "A1": (594.0, 841.0),
}
ORIENTATIONS = ("portrait", "landscape")

_TITLE_BLOCK_WIDTH_MM = 78.0
_TITLE_BLOCK_ROW_MM = 5.0
_TITLE_BLOCK_FONT_MM = 2.6
_TITLE_BLOCK_PADDING_MM = 1.6
_SCALE_BAR_HEIGHT_MM = 2.4
_SCALE_BAR_LABEL_MM = 2.6
_SCALE_BAR_SEGMENTS = 4
_SCALE_BAR_MIN_PAPER_MM = 25.0
_SCALE_BAR_MAX_PAPER_MM = 90.0
_HAIRLINE_MM = 0.13
_FONT_STACK = "'Noto Sans KR', 'Malgun Gothic', sans-serif"

# The scale bar's band: the bar itself plus the row of labels beneath it.
_SCALE_BAR_BAND_MM = _SCALE_BAR_HEIGHT_MM + _SCALE_BAR_LABEL_MM
# Clear space between the last figure and the footer band, so a drawing never
# appears to touch the sheet's own annotations.
_FOOTER_GAP_MM = 4.0


class DrawingSheetError(ValueError):
    """A sheet cannot be composed as requested."""


@dataclass(frozen=True, slots=True)
class SheetPage:
    """The physical page a sheet is drawn on."""

    size: str = "A4"
    orientation: str = "portrait"
    margin_mm: float = 12.0

    def __post_init__(self) -> None:
        size = str(self.size).strip().upper()
        if size not in PAGE_SIZES_MM:
            known = ", ".join(sorted(PAGE_SIZES_MM))
            raise DrawingSheetError(f"unknown page size: {self.size!r}; known sizes are {known}")
        object.__setattr__(self, "size", size)
        orientation = str(self.orientation).strip().lower()
        if orientation not in ORIENTATIONS:
            raise DrawingSheetError(
                f"orientation must be one of: {', '.join(ORIENTATIONS)}"
            )
        object.__setattr__(self, "orientation", orientation)
        try:
            margin = finite_number(
                self.margin_mm, field_name="margin_mm", minimum=0.0
            )
        except SVGRenderError as exc:
            raise DrawingSheetError(str(exc)) from exc
        if margin > 100.0:
            raise DrawingSheetError("margin_mm must be at most 100")
        object.__setattr__(self, "margin_mm", margin)
        if self.content_width_mm <= 0.0:
            raise DrawingSheetError(
                "margin_mm leaves no room to draw on this page size"
            )

    @property
    def width_mm(self) -> float:
        portrait_width, portrait_height = PAGE_SIZES_MM[self.size]
        return portrait_width if self.orientation == "portrait" else portrait_height

    @property
    def height_mm(self) -> float:
        portrait_width, portrait_height = PAGE_SIZES_MM[self.size]
        return portrait_height if self.orientation == "portrait" else portrait_width

    @property
    def content_width_mm(self) -> float:
        return self.width_mm - 2.0 * self.margin_mm

    def to_dict(self) -> dict[str, Any]:
        return {
            "height_mm": self.height_mm,
            "margin_mm": self.margin_mm,
            "orientation": self.orientation,
            "size": self.size,
            "width_mm": self.width_mm,
        }


@dataclass(frozen=True, slots=True)
class TitleBlock:
    """What the sheet says about itself.

    The scale row is added by the composer and cannot be supplied here: a
    measured drawing that does not print its own reduction cannot be measured.
    """

    artifact_label: str
    rows: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        label = str(self.artifact_label).strip()
        if not label:
            raise DrawingSheetError("artifact_label must be a non-empty string")
        if len(label) > 120:
            raise DrawingSheetError("artifact_label must not exceed 120 characters")
        object.__setattr__(self, "artifact_label", label)
        rows: list[tuple[str, str]] = []
        for entry in self.rows:
            if not isinstance(entry, Sequence) or len(entry) != 2:
                raise DrawingSheetError("each title block row must be (label, value)")
            row_label = str(entry[0]).strip()
            row_value = str(entry[1]).strip()
            if not row_label:
                raise DrawingSheetError("title block row labels must not be empty")
            if len(row_label) > 24 or len(row_value) > 96:
                raise DrawingSheetError("title block row is too long for the block")
            rows.append((row_label, row_value))
        if len(rows) > 6:
            raise DrawingSheetError("a title block holds at most six extra rows")
        object.__setattr__(self, "rows", tuple(rows))


@dataclass(frozen=True, slots=True)
class DrawingSheetOptions:
    """Everything that decides what the composed page looks like."""

    title_block: TitleBlock
    scale_denominator: float = 1.0
    page: SheetPage = field(default_factory=SheetPage)
    style_preset: str = "provisional/v1"
    show_center_axis: bool = False
    mirror_sections: tuple[tuple[str, str], ...] = ()
    """(elevation record id, section record id) pairs drawn as one figure.

    The pottery convention: the left half of the figure is the elevation and
    the right half is the section through the same plane, joined at the
    rotation axis.  The elevation record keeps its place in `record_ids`; the
    section record is not a figure of its own and must not be listed there.
    """
    condition_records: tuple[str, ...] = ()
    """Condition annotations to draw over the figures, by record id.

    Empty by default, and an empty tuple changes nothing: a sheet composed
    without it is byte for byte the sheet it was before condition records
    existed.  A named record is drawn only onto figures that share its view;
    the same damage seen from another direction has its own boundary in the
    same record, and the sheet uses whichever one matches.
    """
    rubbings_on_axis: tuple[tuple[str, str], ...] = ()
    """(developed rubbing record id, elevation record id) pairs to paste flush.

    The pottery convention: the strip rubbing is pasted with one edge exactly
    on the centre line, so the rubbing and the elevation's own lines run into
    each other.  The rubbing goes on the elevation side of the axis, at the
    heights it was taken from.  It is drawn inside that figure and must not
    also be listed as a figure of its own.
    """
    rubbing_on_axis_fit: str = "paper"
    """How a pasted rubbing meets the elevation's heights.

    ``paper`` pastes the sheet whole, at its own length, from the height its
    bottom row was taken at - the way a real sheet is pasted.  A rubbing is a
    rubbing and a measured drawing is a measured drawing; on a belly the paper
    is a little longer than the wall is tall, and that is allowed to show.
    ``axis_height`` instead pastes it in bands, each at the height it was
    taken from, so a groove in the rubbing sits level with the same groove's
    line in the elevation, at the cost of shortening the bands on a belly.
    """
    groove_records: tuple[str, ...] = ()
    """Groove readings to draw on the figures, by record id.

    Empty by default, and an empty tuple changes nothing.  A groove that runs
    right round the artifact is drawn only on a figure whose plane contains the
    rotation axis: seen from above it is a circle, not a line.
    """
    gutter_mm: float = 8.0
    stroke_color: str = "#111111"
    title: str = "ArchMeshRubbing measured drawing sheet"

    def __post_init__(self) -> None:
        if not isinstance(self.title_block, TitleBlock):
            raise DrawingSheetError("title_block must be a TitleBlock")
        if not isinstance(self.page, SheetPage):
            raise DrawingSheetError("page must be a SheetPage")
        if not isinstance(self.show_center_axis, bool):
            raise DrawingSheetError("show_center_axis must be a boolean")
        mirror_sections: list[tuple[str, str]] = []
        for pair in self.mirror_sections:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise DrawingSheetError(
                    "mirror_sections entries must be "
                    "(elevation record id, section record id) pairs"
                )
            elevation_id, section_id = (str(item).strip() for item in pair)
            if not elevation_id or not section_id:
                raise DrawingSheetError("mirror_sections entries must be record ids")
            if elevation_id == section_id:
                raise DrawingSheetError(
                    "a record cannot be both halves of one mirrored figure"
                )
            mirror_sections.append((elevation_id, section_id))
        halves = [item for pair in mirror_sections for item in pair]
        if len(set(halves)) != len(halves):
            raise DrawingSheetError(
                "a record can be one half of at most one mirrored figure"
            )
        object.__setattr__(self, "mirror_sections", tuple(mirror_sections))
        condition_records = tuple(self.condition_records)
        if any(
            not isinstance(record_id, str) or not record_id.strip()
            for record_id in condition_records
        ):
            raise DrawingSheetError("condition_records must be record ids")
        if len(set(condition_records)) != len(condition_records):
            raise DrawingSheetError(
                "the same condition record cannot be drawn twice on one sheet"
            )
        if len(condition_records) > MAX_DRAWING_SHEET_CONDITION_RECORDS:
            raise DrawingSheetError(
                f"a sheet draws at most {MAX_DRAWING_SHEET_CONDITION_RECORDS} "
                "condition records"
            )
        object.__setattr__(self, "condition_records", condition_records)
        groove_records = tuple(self.groove_records)
        if any(
            not isinstance(record_id, str) or not record_id.strip()
            for record_id in groove_records
        ):
            raise DrawingSheetError("groove_records must be record ids")
        if len(set(groove_records)) != len(groove_records):
            raise DrawingSheetError(
                "the same groove record cannot be drawn twice on one sheet"
            )
        if len(groove_records) > MAX_DRAWING_SHEET_CONDITION_RECORDS:
            raise DrawingSheetError(
                f"a sheet draws at most {MAX_DRAWING_SHEET_CONDITION_RECORDS} "
                "groove records"
            )
        object.__setattr__(self, "groove_records", groove_records)
        on_axis: list[tuple[str, str]] = []
        for pair in self.rubbings_on_axis:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise DrawingSheetError(
                    "rubbings_on_axis entries must be "
                    "(rubbing record id, elevation record id) pairs"
                )
            rubbing_id, elevation_id = (str(item).strip() for item in pair)
            if not rubbing_id or not elevation_id:
                raise DrawingSheetError("rubbings_on_axis entries must be record ids")
            if rubbing_id == elevation_id:
                raise DrawingSheetError(
                    "a record cannot be both the rubbing and the elevation it is "
                    "pasted on"
                )
            on_axis.append((rubbing_id, elevation_id))
        if len({rubbing for rubbing, _ in on_axis}) != len(on_axis):
            raise DrawingSheetError(
                "a rubbing can be pasted on at most one figure"
            )
        if len({elevation for _, elevation in on_axis}) != len(on_axis):
            raise DrawingSheetError(
                "a figure takes at most one rubbing on its axis"
            )
        object.__setattr__(self, "rubbings_on_axis", tuple(on_axis))
        if self.rubbing_on_axis_fit not in RUBBING_ON_AXIS_FITS:
            raise DrawingSheetError(
                "rubbing_on_axis_fit must be one of "
                f"{', '.join(RUBBING_ON_AXIS_FITS)}"
            )
        try:
            denominator = finite_number(
                self.scale_denominator,
                field_name="scale_denominator",
                strictly_positive=True,
            )
            gutter = finite_number(self.gutter_mm, field_name="gutter_mm", minimum=0.0)
        except SVGRenderError as exc:
            raise DrawingSheetError(str(exc)) from exc
        if denominator < 1.0:
            raise DrawingSheetError(
                "scale_denominator must be at least 1; a sheet reduces, it does not "
                "enlarge a measured drawing"
            )
        if denominator > 1000.0:
            raise DrawingSheetError("scale_denominator must be at most 1000")
        object.__setattr__(self, "scale_denominator", denominator)
        object.__setattr__(self, "gutter_mm", gutter)
        try:
            get_drawing_style_preset(self.style_preset)
        except DrawingStyleError as exc:
            raise DrawingSheetError(str(exc)) from exc
        color = str(self.stroke_color).strip().lower()
        if len(color) != 7 or not color.startswith("#"):
            raise DrawingSheetError("stroke_color must be a six-digit hexadecimal color")
        try:
            int(color[1:], 16)
        except ValueError as exc:
            raise DrawingSheetError(
                "stroke_color must be a six-digit hexadecimal color"
            ) from exc
        object.__setattr__(self, "stroke_color", color)
        title = str(self.title).strip()
        if not title or len(title) > 512:
            raise DrawingSheetError("title must be between 1 and 512 characters")
        object.__setattr__(self, "title", title)
        if self.content_height_mm <= 0.0:
            raise DrawingSheetError(
                f"{self.page.size} {self.page.orientation} with a "
                f"{self.page.margin_mm} mm margin leaves no room to draw above the "
                f"title block; use a larger page, a smaller margin, or fewer "
                "title block rows"
            )

    @property
    def physical_scale(self) -> str:
        return f"1:{_scale_token(self.scale_denominator)}"

    @property
    def title_block_rows(self) -> int:
        """Artifact label, the mandatory scale row, the caller's rows, document."""

        return 2 + len(self.title_block.rows) + 1

    @property
    def title_block_height_mm(self) -> float:
        return _TITLE_BLOCK_ROW_MM * self.title_block_rows

    @property
    def footer_height_mm(self) -> float:
        """Height of the band the sheet reserves for its own annotations.

        The title block and the scale bar sit side by side, but the band is
        reserved across the full width rather than only under each one.  A
        figure that reached into the gap between them would still print as a
        drawing crowding the caption, and a rule that depends on how wide the
        title block happens to be is a rule that breaks when a label gets
        longer.
        """

        return max(self.title_block_height_mm, _SCALE_BAR_BAND_MM)

    @property
    def content_height_mm(self) -> float:
        """Drawable height, with the footer band and its clearance removed."""

        return (
            self.page.height_mm
            - 2.0 * self.page.margin_mm
            - self.footer_height_mm
            - _FOOTER_GAP_MM
        )


@dataclass(frozen=True, slots=True)
class DrawingSheetBundle:
    svg_bytes: bytes
    sidecar_bytes: bytes
    svg_sha256: str
    sidecar_sha256: str


def _scale_token(denominator: float) -> str:
    """Return `4` rather than `4.0`, so a sheet reads `1:4`."""

    if float(denominator).is_integer():
        return str(int(denominator))
    return number_token(denominator, field_name="scale_denominator")


def scale_bar_length_mm(scale_denominator: float) -> float:
    """Return the artifact length a scale bar should span, in millimetres.

    A scale bar is only useful if a reader can name the number under it, so the
    candidates are 1, 2 and 5 times a power of ten.  Among those, take the
    longest bar that still fits the paper band; if even the shortest candidate
    overflows it, the longest bar that fits is still better than none.
    """

    denominator = float(scale_denominator)
    candidates: list[float] = []
    exponent = -1
    while exponent <= 6:
        for step in (1.0, 2.0, 5.0):
            candidates.append(step * (10.0**exponent))
        exponent += 1
    fitting = [
        length
        for length in candidates
        if _SCALE_BAR_MIN_PAPER_MM <= length / denominator <= _SCALE_BAR_MAX_PAPER_MM
    ]
    if fitting:
        return max(fitting)
    under = [
        length for length in candidates if length / denominator <= _SCALE_BAR_MAX_PAPER_MM
    ]
    if under:
        return max(under)
    return min(candidates)


def scale_bar_label(length_mm: float) -> str:
    """Return the bar's end label in the unit a reader would say out loud.

    Nobody labels a bar "100 cm" when they mean a metre, and nobody labels a
    5 cm bar "50 mm" on a pottery drawing.  The unit follows the magnitude.
    """

    if length_mm >= 1000.0 and float(length_mm / 1000.0).is_integer():
        return f"{int(length_mm / 1000.0)} m"
    if length_mm >= 10.0 and float(length_mm / 10.0).is_integer():
        return f"{int(length_mm / 10.0)} cm"
    if float(length_mm).is_integer():
        return f"{int(length_mm)} mm"
    return f"{number_token(length_mm, field_name='scale_bar.length_mm')} mm"


@dataclass(frozen=True, slots=True)
class _RasterImage:
    """A 1:1 rubbing raster ready to be placed, and what proves it."""

    data_uri: str
    raster_sha256: str
    pixels_per_meter: int
    width_pixels: int
    height_pixels: int


@dataclass(frozen=True, slots=True)
class _AttachedRaster:
    """A rubbing pasted inside another figure, flush against the axis."""

    record_id: str
    recipe_hash: str
    image: _RasterImage
    rectangle_mm: tuple[float, float, float, float]
    base_height_um: int
    top_height_um: int
    fit: str
    band_heights_mm: tuple[float, ...]
    """Record-mm v of each band boundary, bottom to top, in the figure's frame."""


@dataclass(frozen=True, slots=True)
class _Prepared:
    """One figure's content, before it knows where on the page it goes."""

    record_id: str
    record_type: str
    recipe_hash: str
    payload_sha256: str
    bounds: tuple[float, float, float, float]
    paths_by_kind: Mapping[str, list[Any]]
    mirror_section_record_id: str | None = None
    fill_only_ids: frozenset[str] = frozenset()
    raster: _RasterImage | None = None
    attached: _AttachedRaster | None = None


@dataclass(frozen=True, slots=True)
class _Figure:
    record_id: str
    record_type: str
    recipe_hash: str
    payload_sha256: str
    placement: Placement
    paths_by_kind: Mapping[str, list[Any]]
    mirror_section_record_id: str | None = None
    fill_only_ids: frozenset[str] = frozenset()
    raster: _RasterImage | None = None
    attached: _AttachedRaster | None = None


def _lay_out(
    figures: Sequence[_Prepared],
    *,
    options: DrawingSheetOptions,
) -> list[_Figure]:
    """Place figures left to right, wrapping into rows, and refuse to overflow.

    Rows keep the order the caller gave, because a reader of a report figure
    expects the elevation and the section in the order the caption names them.
    """

    page = options.page
    denominator = options.scale_denominator
    available_width = page.content_width_mm
    available_height = options.content_height_mm

    placed: list[_Figure] = []
    cursor_x = page.margin_mm
    cursor_y = page.margin_mm
    row_height = 0.0
    row_count = 0

    for prepared in figures:
        record_id = prepared.record_id
        bounds = prepared.bounds
        probe = Placement(content_bounds_mm=bounds, scale_denominator=denominator)
        width, height = probe.width_mm, probe.height_mm
        if width > available_width or height > available_height:
            overflow = max(width / available_width, height / available_height)
            # Round the suggestion up, never to nearest: a suggested scale that
            # still does not fit is worse than no suggestion at all.
            suggestion = math.ceil(denominator * overflow)
            raise DrawingSheetError(
                f"record {record_id!r} does not fit {page.size} "
                f"{page.orientation} at {options.physical_scale}: it needs "
                f"{width:.1f} x {height:.1f} mm of the available "
                f"{available_width:.1f} x {available_height:.1f} mm. Use a scale "
                f"denominator of {suggestion} or more, or a larger page."
            )
        gutter = options.gutter_mm if row_count else 0.0
        if cursor_x + gutter + width > page.margin_mm + available_width + 1e-9:
            cursor_y += row_height + options.gutter_mm
            cursor_x = page.margin_mm
            row_height = 0.0
            row_count = 0
            gutter = 0.0
        if cursor_y + height > page.margin_mm + available_height + 1e-9:
            raise DrawingSheetError(
                f"the figures do not fit {page.size} {page.orientation} at "
                f"{options.physical_scale}; reduce the scale, use a larger page, "
                "or put fewer records on one sheet"
            )
        cursor_x += gutter
        placed.append(
            _Figure(
                record_id=record_id,
                record_type=prepared.record_type,
                recipe_hash=prepared.recipe_hash,
                payload_sha256=prepared.payload_sha256,
                placement=Placement(
                    content_bounds_mm=bounds,
                    origin_mm=(cursor_x, cursor_y),
                    scale_denominator=denominator,
                ),
                paths_by_kind=prepared.paths_by_kind,
                mirror_section_record_id=prepared.mirror_section_record_id,
                fill_only_ids=prepared.fill_only_ids,
                raster=prepared.raster,
                attached=prepared.attached,
            )
        )
        cursor_x += width
        row_height = max(row_height, height)
        row_count += 1
    return placed


def _text_element(
    text: str,
    *,
    x_mm: float,
    y_mm: float,
    size_mm: float,
    color: str,
    anchor: str = "start",
    weight: str | None = None,
) -> str:
    weight_attribute = "" if weight is None else f' font-weight="{weight}"'
    return (
        f'<text x="{number_token(x_mm, field_name="text.x")}" '
        f'y="{number_token(y_mm, field_name="text.y")}" '
        f'font-family="{_FONT_STACK}" '
        f'font-size="{number_token(size_mm, field_name="text.size")}" '
        f'fill="{color}" text-anchor="{anchor}"{weight_attribute}>'
        f"{xml_attribute(text)}</text>"
    )


def _scale_bar_elements(options: DrawingSheetOptions) -> tuple[list[str], dict[str, Any]]:
    """Return the scale bar, and what the sidecar records about it."""

    page = options.page
    length_mm = scale_bar_length_mm(options.scale_denominator)
    paper_mm = length_mm / options.scale_denominator
    segment = paper_mm / _SCALE_BAR_SEGMENTS
    # Bottom-aligned with the title block, so the two read as one footer row.
    top = page.height_mm - page.margin_mm - _SCALE_BAR_BAND_MM
    left = page.margin_mm

    lines = [f'  <g id="scale-bar" stroke="{options.stroke_color}" '
             f'stroke-width="{number_token(_HAIRLINE_MM, field_name="hairline")}">']
    for index in range(_SCALE_BAR_SEGMENTS):
        x = left + index * segment
        # Alternating solid and empty cells are what makes a bar readable at a
        # glance; the outline alone gives the reader nothing to count.
        fill = options.stroke_color if index % 2 == 0 else "none"
        lines.append(
            f'    <rect x="{number_token(x, field_name="scale_bar.x")}" '
            f'y="{number_token(top, field_name="scale_bar.y")}" '
            f'width="{number_token(segment, field_name="scale_bar.width")}" '
            f'height="{number_token(_SCALE_BAR_HEIGHT_MM, field_name="scale_bar.height")}" '
            f'fill="{fill}"/>'
        )
    label_y = top + _SCALE_BAR_HEIGHT_MM + _SCALE_BAR_LABEL_MM
    lines.append(
        "    "
        + _text_element(
            "0",
            x_mm=left,
            y_mm=label_y,
            size_mm=_SCALE_BAR_LABEL_MM,
            color=options.stroke_color,
        )
    )
    label = scale_bar_label(length_mm)
    lines.append(
        "    "
        + _text_element(
            label,
            x_mm=left + paper_mm,
            y_mm=label_y,
            size_mm=_SCALE_BAR_LABEL_MM,
            color=options.stroke_color,
            anchor="middle",
        )
    )
    lines.append("  </g>")
    return lines, {
        "artifact_length_mm": length_mm,
        "label": label,
        "paper_length_mm": paper_mm,
        "segments": _SCALE_BAR_SEGMENTS,
    }


def _title_block_elements(
    options: DrawingSheetOptions,
    *,
    document_manifest_sha256: str,
) -> tuple[list[str], list[dict[str, str]]]:
    """Return the title block, and the rows it prints."""

    block = options.title_block
    rows: list[tuple[str, str]] = [("유물", block.artifact_label)]
    # The scale is derived and mandatory.  A reduced drawing that does not say
    # what it was reduced by cannot be measured off the page.
    rows.append(("축척", options.physical_scale))
    rows.extend(block.rows)
    rows.append(("문서", document_manifest_sha256[:12]))

    page = options.page
    assert len(rows) == options.title_block_rows
    height = options.title_block_height_mm
    left = page.width_mm - page.margin_mm - _TITLE_BLOCK_WIDTH_MM
    top = page.height_mm - page.margin_mm - height

    lines = [
        f'  <g id="title-block" stroke="{options.stroke_color}" '
        f'stroke-width="{number_token(_HAIRLINE_MM, field_name="hairline")}" fill="none">',
        f'    <rect x="{number_token(left, field_name="title_block.x")}" '
        f'y="{number_token(top, field_name="title_block.y")}" '
        f'width="{number_token(_TITLE_BLOCK_WIDTH_MM, field_name="title_block.width")}" '
        f'height="{number_token(height, field_name="title_block.height")}"/>',
    ]
    for index, (label, value) in enumerate(rows):
        baseline = top + _TITLE_BLOCK_ROW_MM * index + _TITLE_BLOCK_ROW_MM - 1.6
        if index:
            divider_y = top + _TITLE_BLOCK_ROW_MM * index
            lines.append(
                f'    <path d="M {number_token(left, field_name="title_block.x")} '
                f'{number_token(divider_y, field_name="title_block.y")} '
                f'L {number_token(left + _TITLE_BLOCK_WIDTH_MM, field_name="title_block.x")} '
                f'{number_token(divider_y, field_name="title_block.y")}"/>'
            )
        lines.append(
            "    "
            + _text_element(
                label,
                x_mm=left + _TITLE_BLOCK_PADDING_MM,
                y_mm=baseline,
                size_mm=_TITLE_BLOCK_FONT_MM,
                color=options.stroke_color,
                weight="bold",
            )
        )
        lines.append(
            "    "
            + _text_element(
                value,
                x_mm=left + _TITLE_BLOCK_WIDTH_MM - _TITLE_BLOCK_PADDING_MM,
                y_mm=baseline,
                size_mm=_TITLE_BLOCK_FONT_MM,
                color=options.stroke_color,
                anchor="end",
            )
        )
    lines.append("  </g>")
    return lines, [{"label": label, "value": value} for label, value in rows]


def _require_drawable_condition_record(
    document: ArtifactDocument,
    record_id: str,
) -> tuple[DerivedRecord, ConditionAnnotationPayload]:
    """Resolve one condition record under the same rules a figure answers to."""

    record = document.record_index.get(record_id)
    if record is None:
        raise DrawingSheetError(f"condition record {record_id!r} does not exist")
    if record.type != CONDITION_RECORD_TYPE:
        raise DrawingSheetError(
            f"record {record_id!r} is not a condition annotation"
        )
    if record.lifecycle_status is not RecordLifecycleStatus.READY:
        raise DrawingSheetError("only READY condition records may be drawn")
    try:
        freshness = document.record_freshness(record.id)
    except ArtifactDocumentError as exc:
        raise DrawingSheetError(str(exc)) from exc
    if freshness is not RecordFreshness.FRESH:
        raise DrawingSheetError(
            "only FRESH condition records may be drawn "
            f"(got {freshness.value}); a condition drawn under a superseded "
            "alignment would sit somewhere the artifact no longer is"
        )
    try:
        payload = condition_payload_from_record(record)
    except ArtifactConditionAnnotationError as exc:
        raise DrawingSheetError(str(exc)) from exc
    return record, payload


def _require_drawable_groove_record(
    document: ArtifactDocument,
    record_id: str,
) -> tuple[DerivedRecord, ProfileGroovePayload]:
    """Resolve one groove reading under the same rules a figure answers to."""

    record = document.record_index.get(record_id)
    if record is None:
        raise DrawingSheetError(f"groove record {record_id!r} does not exist")
    if record.type != PROFILE_GROOVE_RECORD_TYPE:
        raise DrawingSheetError(f"record {record_id!r} is not a groove reading")
    if record.lifecycle_status is not RecordLifecycleStatus.READY:
        raise DrawingSheetError("only READY groove records may be drawn")
    try:
        freshness = document.record_freshness(record.id)
    except ArtifactDocumentError as exc:
        raise DrawingSheetError(str(exc)) from exc
    if freshness is not RecordFreshness.FRESH:
        raise DrawingSheetError(
            "only FRESH groove records may be drawn "
            f"(got {freshness.value}); a groove read under a superseded "
            "alignment names a height on an artifact standing somewhere else"
        )
    try:
        payload = profile_groove_payload_from_record(record)
    except ArtifactProfileGrooveError as exc:
        raise DrawingSheetError(str(exc)) from exc
    return record, payload


def _groove_paths_for_figure(
    figure_payload: Any,
    grooves: Sequence[tuple[DerivedRecord, ProfileGroovePayload]],
) -> tuple[dict[str, list[Any]], list[dict[str, str]]]:
    """Return the groove layers that belong on one figure, and what they are.

    A groove is a fact about the artifact's own axis, so unlike a condition
    boundary it is not tied to one view: any figure whose plane contains that
    axis shows it, elevation and section alike.  A plan view shows it as a
    circle instead, and gets nothing.
    """

    by_kind: dict[str, list[Any]] = {}
    drawn: list[dict[str, str]] = []
    for record, payload in grooves:
        try:
            paths = profile_groove_vector_paths(
                figure_payload, payload.grooves, record_id=record.id
            )
        except ArtifactVectorExportError as exc:
            raise DrawingSheetError(str(exc)) from exc
        if not paths:
            continue
        for kind, groove_paths in paths.items():
            by_kind.setdefault(kind, []).extend(groove_paths)
        drawn.append(
            {
                "groove_count": str(len(payload.grooves)),
                "record_id": record.id,
            }
        )
    return by_kind, drawn


def _proven_raster_image(
    document: ArtifactDocument,
    record: DerivedRecord,
    raster: Any,
) -> tuple[_RasterImage, Mapping[str, Any]]:
    """Encode a rubbing raster once its record's receipt has vouched for it."""

    developed = record.type == DEVELOPED_RUBBING_RECORD_TYPE
    expected_type = (
        DevelopedRubbingRaster if developed else DigitalRubbingRaster
    )
    if not isinstance(raster, expected_type):
        raise DrawingSheetError(
            f"record {record.id!r} needs a {expected_type.__name__} to be drawn"
        )
    try:
        receipt = (
            developed_rubbing_receipt_from_record(record)
            if developed
            else rubbing_receipt_from_record(record)
        )
    except (ArtifactDevelopedRubbingError, ArtifactRubbingRecordError) as exc:
        raise DrawingSheetError(str(exc)) from exc
    if raster.receipt() != receipt:
        raise DrawingSheetError(
            f"the raster given for record {record.id!r} is not the one its "
            "receipt describes"
        )
    pixels_per_meter = int(receipt["pixels_per_meter"])
    metadata = {
        "document_id": document.document_id,
        "format": DRAWING_SHEET_PNG_METADATA_FORMAT,
        "raster_sha256": receipt["raster_sha256"],
        "recipe_hash": record.recipe_hash,
        "record_id": record.id,
        "record_type": record.type,
        "schema_version": DRAWING_SHEET_SCHEMA_VERSION,
    }
    try:
        png_bytes = encode_canonical_ga8_png(
            raster.pixels,
            pixels_per_meter=pixels_per_meter,
            metadata=metadata,
        )
    except CanonicalPNGError as exc:
        raise DrawingSheetError(str(exc)) from exc
    encoded = base64.b64encode(png_bytes).decode("ascii")
    if len(encoded) > MAX_DRAWING_SHEET_RASTER_BYTES:
        raise DrawingSheetError(
            f"record {record.id!r} embeds {len(encoded)} bytes of raster, above "
            f"the {MAX_DRAWING_SHEET_RASTER_BYTES}-byte sheet limit; compute the "
            "rubbing at a lower physical resolution"
        )
    return (
        _RasterImage(
            data_uri=f"data:image/png;base64,{encoded}",
            raster_sha256=str(receipt["raster_sha256"]),
            pixels_per_meter=pixels_per_meter,
            width_pixels=int(receipt["width_pixels"]),
            height_pixels=int(receipt["height_pixels"]),
        ),
        receipt,
    )


def _prepare_raster_figure(
    document: ArtifactDocument,
    record: DerivedRecord,
    raster: Any,
) -> _Prepared:
    """Turn a proven rubbing raster into a figure of its own physical size.

    A rubbing record stores a receipt, not pixels, so the caller recomputes
    the raster and hands it in; the receipt is what decides whether those
    pixels are the record's.  The strip then goes on the page at the sheet's
    own scale, the way a rubber tapes the paper beside the drawing.
    """

    image, receipt = _proven_raster_image(document, record, raster)
    pixels_per_meter = int(receipt["pixels_per_meter"])
    width_mm = float(receipt["width_pixels"]) * 1000.0 / float(pixels_per_meter)
    height_mm = float(receipt["height_pixels"]) * 1000.0 / float(pixels_per_meter)
    return _Prepared(
        record_id=record.id,
        record_type=record.type,
        recipe_hash=record.recipe_hash,
        payload_sha256=str(receipt["raster_sha256"]),
        bounds=(0.0, 0.0, width_mm, height_mm),
        paths_by_kind={},
        raster=image,
    )


def _attach_rubbing_on_axis(
    document: ArtifactDocument,
    *,
    rubbing_id: str,
    raster: Any,
    elevation: DerivedRecord,
    elevation_payload: VectorGeometryPayload,
    fit: str,
) -> _AttachedRaster:
    """Paste a strip rubbing flush against the elevation's centre line.

    The strip was taken along the meridian that faces the viewer, so it goes
    where that meridian appears: on the elevation side of the axis, one edge
    exactly on the line.  Vertically it sits at the height its bottom row was
    taken from, and it keeps its own paper size - the meridian arc - because a
    rubbing is paper and paper does not shrink to the axial height.  On a
    belly the strip is therefore a little taller than the elevation between
    the same two heights; the sidecar states both.
    """

    record = document.record_index.get(rubbing_id)
    if record is None:
        raise DrawingSheetError(f"rubbing record {rubbing_id!r} does not exist")
    if record.type != DEVELOPED_RUBBING_RECORD_TYPE:
        raise DrawingSheetError(
            f"record {rubbing_id!r} is not a rubbing on a developed surface, so "
            "it has no meridian to paste along the axis"
        )
    if record.lifecycle_status is not RecordLifecycleStatus.READY:
        raise DrawingSheetError(f"only READY records may be drawn (record {rubbing_id!r})")
    if document.record_freshness(rubbing_id) is not RecordFreshness.FRESH:
        raise DrawingSheetError(f"only FRESH records may be drawn (record {rubbing_id!r})")
    if raster is None:
        raise DrawingSheetError(
            f"record {rubbing_id!r} is a rubbing, so its recomputed raster must "
            "be given to the sheet; a rubbing record stores a receipt, not pixels"
        )
    if elevation.type != VectorRecordKind.OUTLINE.record_type:
        raise DrawingSheetError(
            f"record {elevation.id!r} is not an outline, so a rubbing cannot be "
            "pasted on it as an elevation"
        )
    base_height = record.qc.get("artboard_base_height_um")
    top_height = record.qc.get("artboard_top_height_um")
    profile = record.qc.get("artboard_height_profile_um")
    if (
        type(base_height) is not int
        or type(top_height) is not int
        or not isinstance(profile, Sequence)
        or len(profile) < 2
        or any(type(value) is not int for value in profile)
        or profile[0] != base_height
        or profile[-1] != top_height
        or any(later < earlier for earlier, later in zip(profile, profile[1:]))
    ):
        raise DrawingSheetError(
            f"rubbing record {rubbing_id!r} does not say what heights its "
            "artboard was taken from; it was computed before that was recorded, "
            "so compute the rubbing again"
        )
    if top_height <= base_height:
        raise DrawingSheetError(
            f"rubbing record {rubbing_id!r} spans no height on the artifact, so "
            "there is nowhere on the elevation to paste it"
        )
    image, receipt = _proven_raster_image(document, record, raster)
    try:
        line = center_axis_line(elevation_payload.frame.to_dict())
    except SVGRenderError as exc:
        raise DrawingSheetError(str(exc)) from exc
    if line is None or abs(line[1][0]) > 1e-9 or abs(line[1][1] - 1.0) > 1e-9:
        raise DrawingSheetError(
            f"the rotation axis is not the vertical of {elevation.id!r}, so a "
            "rubbing cannot be pasted flush against it there"
        )
    base, _direction = line
    pixels_per_meter = int(receipt["pixels_per_meter"])
    width_mm = float(receipt["width_pixels"]) * 1000.0 / float(pixels_per_meter)
    height_mm = float(receipt["height_pixels"]) * 1000.0 / float(pixels_per_meter)
    bottom = base[1] + float(base_height) / 1000.0
    if fit == RUBBING_ON_AXIS_FIT_PAPER:
        band_heights = (bottom, bottom + height_mm)
    else:
        band_heights = tuple(base[1] + float(value) / 1000.0 for value in profile)
    rectangle = (base[0] - width_mm, band_heights[0], base[0], band_heights[-1])
    return _AttachedRaster(
        record_id=record.id,
        recipe_hash=record.recipe_hash,
        image=image,
        rectangle_mm=rectangle,
        base_height_um=int(base_height),
        top_height_um=int(top_height),
        fit=fit,
        band_heights_mm=band_heights,
    )


def _condition_paths_for_figure(
    figure_record_type: str,
    figure_payload_frame: Any,
    conditions: Sequence[tuple[DerivedRecord, ConditionAnnotationPayload]],
) -> tuple[dict[str, list[Any]], list[dict[str, str]]]:
    """Return the condition layers that belong on one figure, and what they are.

    Two things have to agree before a boundary is drawn.  The figure must be a
    projection, because a condition boundary is the silhouette of a region seen
    from one direction and a section drawing shows what a plane cuts, not what
    is behind it - a section can share a plane with a view and still be the
    wrong page for it.  And the plane must be the same one: the match is the
    frame itself rather than a declared view name, so a region cannot end up
    laid over a drawing it does not describe.
    """

    by_kind: dict[str, list[Any]] = {}
    drawn: list[dict[str, str]] = []
    if figure_record_type != VectorRecordKind.OUTLINE.record_type:
        return by_kind, drawn
    for record, payload in conditions:
        for boundary in payload.views:
            if boundary.outline.frame != figure_payload_frame:
                continue
            try:
                kind = line_kind_for_condition(payload.condition)
            except DrawingStyleError as exc:
                raise DrawingSheetError(str(exc)) from exc
            for path in boundary.outline.paths:
                by_kind.setdefault(kind, []).append(
                    replace(path, id=f"condition:{record.id}:{boundary.view}:{path.id}")
                )
            drawn.append(
                {
                    "condition_kind": payload.condition,
                    "line_kind": kind,
                    "record_id": record.id,
                    "view": boundary.view,
                }
            )
            break
    return by_kind, drawn


def _bounds_with_attachment(
    bounds: tuple[float, float, float, float],
    attached: _AttachedRaster | None,
) -> tuple[float, float, float, float]:
    """Grow a figure's extent to hold the rubbing pasted inside it."""

    if attached is None:
        return bounds
    left, bottom, right, top = attached.rectangle_mm
    return (
        min(bounds[0], left),
        min(bounds[1], bottom),
        max(bounds[2], right),
        max(bounds[3], top),
    )


def _paths_bounds(
    paths_by_kind: Mapping[str, Sequence[Any]],
) -> tuple[float, float, float, float]:
    """Return the extent of already-built drawing paths, in record millimetres."""

    points = [
        point
        for paths in paths_by_kind.values()
        for path in paths
        for point in path.points_mm
    ]
    if not points:
        raise DrawingSheetError("a figure with no drawable path has no extent")
    us = [float(point[0]) for point in points]
    vs = [float(point[1]) for point in points]
    return (min(us), min(vs), max(us), max(vs))


def _clipped_half(
    paths_by_kind: Mapping[str, Sequence[Any]],
    *,
    preset: DrawingStylePreset,
    base: Sequence[float],
    direction: Sequence[float],
    keep_negative: bool,
    id_prefix: str,
    half_name: str,
) -> tuple[dict[str, list[Any]], set[str]]:
    """Return one half of a figure's paths, cut at the rotation axis.

    A ring the cut passes through comes back as the open chains that were
    actually measured; the chord closing it lies on the axis and is where the
    drawing was folded, not an edge of the artifact.  Where that ring carried a
    hatch, the closed shape is kept too, but fill-only and unstroked, so the
    cut face is still shaded without printing a boundary along the axis.
    """

    halved: dict[str, list[Any]] = {}
    fill_only: set[str] = set()
    for kind, paths in paths_by_kind.items():
        hatched = preset.style(kind).hatch
        for path in paths:
            try:
                if path.closed:
                    ring = clip_closed_ring(
                        path.points_mm,
                        base=base,
                        direction=direction,
                        keep_negative=keep_negative,
                        label=f"{half_name} half, path {path.id!r}",
                    )
                    if ring is None:
                        continue
                    chains = split_ring_off_line(
                        ring, base=base, direction=direction
                    )
                    if chains is None:
                        pieces: list[tuple[list[Any], bool]] = [(list(ring), True)]
                    else:
                        pieces = [(chain, False) for chain in chains]
                        if hatched:
                            halved.setdefault(kind, []).append(
                                replace(
                                    path,
                                    id=f"{id_prefix}{path.id}:fill",
                                    points_mm=tuple(ring),
                                )
                            )
                            fill_only.add(f"{id_prefix}{path.id}:fill")
                else:
                    pieces = [
                        (piece, False)
                        for piece in clip_open_path(
                            path.points_mm,
                            base=base,
                            direction=direction,
                            keep_negative=keep_negative,
                        )
                    ]
            except SVGRenderError as exc:
                raise DrawingSheetError(str(exc)) from exc
            for index, (piece, closed) in enumerate(pieces):
                suffix = "" if len(pieces) == 1 else f":{index:04d}"
                halved.setdefault(kind, []).append(
                    replace(
                        path,
                        id=f"{id_prefix}{path.id}{suffix}",
                        closed=closed,
                        points_mm=tuple(piece),
                    )
                )
    return halved, fill_only


def _mirrored_figure(
    document: ArtifactDocument,
    *,
    elevation: DerivedRecord,
    elevation_payload: VectorGeometryPayload,
    elevation_by_kind: Mapping[str, Sequence[Any]],
    section_record_id: str,
    axis_ready: bool,
    preset: DrawingStylePreset,
) -> tuple[
    DerivedRecord,
    dict[str, list[Any]],
    tuple[float, float, float, float],
    set[str],
]:
    """Join an elevation's left half and a section's right half into one figure.

    This is the convention a wheel-thrown vessel is drawn in: one figure whose
    left side is the outside of the pot and whose right side is the wall cut
    through the axis, so a reader sees profile and thickness at once without
    matching two drawings to each other.

    Everything it needs is checkable, so everything it needs is checked.  Half
    a pot is only meaningful about the axis the pot turns on, the two records
    have to be in one plane before they can be two halves of one figure, and a
    half that comes back empty means the drawing would be a lie of omission.
    """

    if not axis_ready:
        raise DrawingSheetError(
            "a half-elevation and half-section figure needs an artifact "
            "positioned on its rotation axis; the active Align was not made "
            "from one, so there is no axis to fold the drawing about"
        )
    try:
        section, section_payload, _qc = _require_exportable_record(
            document,
            section_record_id,
        )
    except ArtifactVectorExportError as exc:
        raise DrawingSheetError(str(exc)) from exc
    if elevation.type != VectorRecordKind.OUTLINE.record_type:
        raise DrawingSheetError(
            f"record {elevation.id!r} is not an outline, so it cannot be the "
            "elevation half of a mirrored figure"
        )
    if section.type != VectorRecordKind.CUTLINE.record_type:
        raise DrawingSheetError(
            f"record {section.id!r} is not a cutline, so it cannot be the "
            "section half of a mirrored figure"
        )
    if section_payload.frame != elevation_payload.frame:
        raise DrawingSheetError(
            f"records {elevation.id!r} and {section.id!r} are not in the same "
            "plane, so they cannot be two halves of one figure"
        )
    try:
        line = center_axis_line(elevation_payload.frame.to_dict())
    except SVGRenderError as exc:
        raise DrawingSheetError(str(exc)) from exc
    if line is None:
        raise DrawingSheetError(
            f"the rotation axis is perpendicular to the plane of {elevation.id!r}, "
            "so it projects to a point and there is no line to fold about"
        )
    base, direction = line

    section_by_kind: dict[str, list[Any]] = {}
    for path in section_payload.paths:
        try:
            kind = line_kind_for_record_role(path.role)
        except DrawingStyleError as exc:
            raise DrawingSheetError(str(exc)) from exc
        section_by_kind.setdefault(kind, []).append(path)

    left, left_fill_only = _clipped_half(
        elevation_by_kind,
        preset=preset,
        base=base,
        direction=direction,
        keep_negative=True,
        id_prefix="mirror:left:",
        half_name="elevation",
    )
    right, right_fill_only = _clipped_half(
        section_by_kind,
        preset=preset,
        base=base,
        direction=direction,
        keep_negative=False,
        id_prefix="mirror:right:",
        half_name="section",
    )
    if not left:
        raise DrawingSheetError(
            f"the elevation {elevation.id!r} has nothing left of the rotation "
            "axis, so the mirrored figure would be half empty"
        )
    if not right:
        raise DrawingSheetError(
            f"the section {section.id!r} has nothing right of the rotation "
            "axis, so the mirrored figure would be half empty"
        )

    combined: dict[str, list[Any]] = {}
    for half in (left, right):
        for kind, paths in half.items():
            combined.setdefault(kind, []).extend(paths)
    bounds = _paths_bounds(combined)
    # The axis is the seam of this convention, not an optional annotation: the
    # two halves meet on it, and without it a reader cannot tell a joined
    # figure from one drawing of an asymmetric object.
    try:
        segment = center_axis_segment(elevation_payload.frame.to_dict(), bounds)
    except SVGRenderError as exc:
        raise DrawingSheetError(str(exc)) from exc
    if segment is not None:
        combined.setdefault(CENTER_AXIS, []).append(
            VectorPath(
                id="mirror:center-axis",
                role=CENTER_AXIS,
                closed=False,
                points_mm=segment,
            )
        )
    return section, combined, bounds, left_fill_only | right_fill_only


def _sheet_provenance(
    document: ArtifactDocument,
    placed: Sequence[_Figure],
    *,
    options: DrawingSheetOptions,
    scale_bar: Mapping[str, Any],
    title_rows: Sequence[Mapping[str, str]],
    center_axis: Mapping[str, Any],
    condition: Mapping[str, Any] | None,
    groove: Mapping[str, Any] | None,
    mirrored: Sequence[Mapping[str, str]],
    rubbings_on_axis: Sequence[Mapping[str, str]] = (),
) -> dict[str, Any]:
    preset = get_drawing_style_preset(options.style_preset)
    provenance: dict[str, Any] = {
        "center_axis": dict(center_axis),
        "document_id": document.document_id,
        "document_manifest_sha256": document.canonical_sha256,
        "figures": [
            {
                "height_mm": figure.placement.height_mm,
                "origin_mm": list(figure.placement.origin_mm),
                "record_id": figure.record_id,
                "record_type": figure.record_type,
                "recipe_hash": figure.recipe_hash,
                "width_mm": figure.placement.width_mm,
                **(
                    {"vector_payload_sha256": figure.payload_sha256}
                    if figure.raster is None
                    else {
                        "raster_height_pixels": figure.raster.height_pixels,
                        "raster_pixels_per_meter": figure.raster.pixels_per_meter,
                        "raster_sha256": figure.raster.raster_sha256,
                        "raster_width_pixels": figure.raster.width_pixels,
                    }
                ),
                **(
                    {}
                    if figure.attached is None
                    else {
                        "rubbing_on_axis": {
                            "artboard_base_height_um": figure.attached.base_height_um,
                            "artboard_top_height_um": figure.attached.top_height_um,
                            "paper_height_mm": (
                                figure.attached.rectangle_mm[3]
                                - figure.attached.rectangle_mm[1]
                            ),
                            "raster_height_pixels": figure.attached.image.height_pixels,
                            "raster_pixels_per_meter": (
                                figure.attached.image.pixels_per_meter
                            ),
                            "raster_sha256": figure.attached.image.raster_sha256,
                            "raster_width_pixels": figure.attached.image.width_pixels,
                            "recipe_hash": figure.attached.recipe_hash,
                            "record_id": figure.attached.record_id,
                            "rectangle_mm": list(figure.attached.rectangle_mm),
                            "fit": figure.attached.fit,
                            "band_heights_mm": list(figure.attached.band_heights_mm),
                            "side": "elevation",
                        }
                    }
                ),
            }
            for figure in placed
        ],
        "format": DRAWING_SHEET_FORMAT,
        "page": options.page.to_dict(),
        "physical_scale": options.physical_scale,
        "scale_bar": dict(scale_bar),
        "scale_denominator": options.scale_denominator,
        "schema_version": DRAWING_SHEET_SCHEMA_VERSION,
        "style_preset": {
            "preset_id": preset.preset_id,
            "provisional": preset.provisional,
            "sha256": preset.sha256(),
            "source_id": preset.source_id,
        },
        "title": options.title,
        "title_block": [dict(row) for row in title_rows],
        "unit": "mm",
    }
    if condition is not None:
        # Added only when the caller asked for condition records, so a sheet
        # composed without them keeps the exact bytes it had before.
        provenance["condition"] = dict(condition)
    if groove is not None:
        provenance["groove"] = dict(groove)
    if mirrored:
        provenance["mirrored_figures"] = [dict(entry) for entry in mirrored]
    if rubbings_on_axis:
        provenance["rubbings_on_axis"] = [dict(entry) for entry in rubbings_on_axis]
    return provenance


def _attached_raster_elements(
    attached: _AttachedRaster,
    *,
    placement: Placement,
    index: int,
) -> list[str]:
    """Paste the rubbing as one image, or as bands each at its own height.

    A band is the same image seen through a nested viewport whose viewBox
    selects its rows, stretched to the band's height on the page.  The pixels
    are never resampled or re-encoded: the raster on the sheet is still the
    one the record's receipt proves, band by band.
    """

    left, _bottom, right, _top = attached.rectangle_mm
    denominator = placement.scale_denominator
    width_paper = (right - left) / denominator
    image = attached.image
    boundaries = attached.band_heights_mm
    band_count = len(boundaries) - 1
    rows_per_band = image.height_pixels / band_count
    prefix = f"rubbing-on-axis-{index:04d}"
    # The pixels are embedded once and every band refers to them, so sixteen
    # bands cost sixteen viewports, not sixteen copies of the PNG.
    elements: list[str] = [
        "      <defs>",
        (
            f'        <image id="{prefix}-pixels" width="{image.width_pixels}" '
            f'height="{image.height_pixels}" '
            'preserveAspectRatio="none" image-rendering="pixelated" '
            f'xlink:href="{xml_attribute(image.data_uri)}"/>'
        ),
        "      </defs>",
    ]
    for band in range(band_count):
        lower = boundaries[band]
        upper = boundaries[band + 1]
        if upper <= lower:
            continue
        paper_x, paper_y = placement.paper_xy((left, upper))
        band_paper_height = (upper - lower) / denominator
        # Rows count from the top of the image; band 0 is the bottom of the
        # artboard, which is the last rows.
        row_top = image.height_pixels - rows_per_band * (band + 1)
        elements.append(
            f'      <svg id="{prefix}-band-{band:02d}" '
            f'data-record-id="{xml_attribute(attached.record_id)}" '
            f'x="{number_token(paper_x, field_name="rubbing.x")}" '
            f'y="{number_token(paper_y, field_name="rubbing.y")}" '
            f'width="{number_token(width_paper, field_name="rubbing.width")}" '
            f'height="{number_token(band_paper_height, field_name="rubbing.height")}" '
            f'viewBox="0 {number_token(row_top, field_name="rubbing.row")} '
            f'{image.width_pixels} {number_token(rows_per_band, field_name="rubbing.rows")}" '
            'preserveAspectRatio="none">'
        )
        elements.append(f'        <use xlink:href="#{prefix}-pixels"/>')
        elements.append("      </svg>")
    return elements


def _render_sheet(
    placed: Sequence[_Figure],
    *,
    options: DrawingSheetOptions,
    provenance: Mapping[str, Any],
    scale_bar_lines: Sequence[str],
    title_block_lines: Sequence[str],
) -> bytes:
    page = options.page
    preset = get_drawing_style_preset(options.style_preset)
    width_token = number_token(page.width_mm, field_name="page.width_mm")
    height_token = number_token(page.height_mm, field_name="page.height_mm")
    metadata_text = canonical_json_bytes(provenance).decode("utf-8").rstrip("\n")
    # SVG 1.1 addresses embedded images through xlink, so the namespace is
    # declared only on a sheet that carries one; a sheet of pure line work is
    # byte for byte the sheet it was before rubbings could be placed.
    xlink_declaration = (
        ' xmlns:xlink="http://www.w3.org/1999/xlink"'
        if any(
            figure.raster is not None or figure.attached is not None
            for figure in placed
        )
        else ""
    )

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="{SVG_NAMESPACE}"{xlink_declaration} version="1.1" '
            f'width="{width_token}mm" height="{height_token}mm" '
            f'viewBox="0 0 {width_token} {height_token}">'
        ),
        f"  <title>{xml_attribute(options.title)}</title>",
        (
            '  <metadata id="archmeshrubbing-provenance">'
            f"{xml_attribute(metadata_text)}</metadata>"
        ),
    ]

    sheet_hatched = sorted(
        {
            kind
            for figure in placed
            for kind in hatched_kinds(figure.paths_by_kind, preset=preset)
        }
    )
    if sheet_hatched:
        lines.append("  <defs>")
        lines.extend(
            hatch_pattern_elements(
                sheet_hatched,
                preset=preset,
                color=options.stroke_color,
                indent="    ",
            )
        )
        lines.append("  </defs>")

    lines.append(
        '  <g id="sheet-figures" fill="none" '
        f'stroke="{options.stroke_color}" '
        'stroke-linecap="round" stroke-linejoin="round">'
    )
    for index, figure in enumerate(placed):
        mirror_attribute = (
            ""
            if figure.mirror_section_record_id is None
            else (
                " data-mirror-section-record-id="
                f'"{xml_attribute(figure.mirror_section_record_id)}"'
            )
        )
        lines.append(
            f'    <g id="figure-{index:04d}" '
            f'data-record-id="{xml_attribute(figure.record_id)}" '
            f'data-record-type="{xml_attribute(figure.record_type)}"'
            f"{mirror_attribute}>"
        )
        if figure.raster is not None:
            placement = figure.placement
            origin_x, origin_y = placement.origin_mm
            lines.append(
                '      <image id="'
                f'rubbing-{index:04d}" '
                f'x="{number_token(origin_x, field_name="figure.x")}" '
                f'y="{number_token(origin_y, field_name="figure.y")}" '
                f'width="{number_token(placement.width_mm, field_name="figure.width")}" '
                f'height="{number_token(placement.height_mm, field_name="figure.height")}" '
                'preserveAspectRatio="none" image-rendering="pixelated" '
                f'xlink:href="{xml_attribute(figure.raster.data_uri)}"/>'
            )
        elif figure.attached is None:
            lines.extend(
                layer_elements(
                    figure.paths_by_kind,
                    preset=preset,
                    placement=figure.placement,
                    hatched=hatched_kinds(figure.paths_by_kind, preset=preset),
                    indent="      ",
                    fill_only_ids=figure.fill_only_ids,
                )
            )
        else:
            # The rubbing is paper pasted onto the drawing: it covers the lines
            # beneath it, and the centre line - the seam the paper is pasted to
            # - is drawn back over it, the one construction line that has to
            # stay readable across everything.
            hatched = hatched_kinds(figure.paths_by_kind, preset=preset)
            under = {
                kind: paths
                for kind, paths in figure.paths_by_kind.items()
                if kind != CENTER_AXIS
            }
            over = {
                kind: paths
                for kind, paths in figure.paths_by_kind.items()
                if kind == CENTER_AXIS
            }
            lines.extend(
                layer_elements(
                    under,
                    preset=preset,
                    placement=figure.placement,
                    hatched=hatched,
                    indent="      ",
                    fill_only_ids=figure.fill_only_ids,
                )
            )
            lines.extend(
                _attached_raster_elements(
                    figure.attached,
                    placement=figure.placement,
                    index=index,
                )
            )
            lines.extend(
                layer_elements(
                    over,
                    preset=preset,
                    placement=figure.placement,
                    hatched=hatched,
                    indent="      ",
                    fill_only_ids=figure.fill_only_ids,
                )
            )
        lines.append("    </g>")
    lines.append("  </g>")

    lines.extend(scale_bar_lines)
    lines.extend(title_block_lines)
    lines.append("</svg>")

    svg_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    if len(svg_bytes) > MAX_DRAWING_SHEET_SVG_BYTES:
        raise DrawingSheetError("sheet SVG exceeds the export safety limit")
    return svg_bytes


def compose_drawing_sheet(
    document: ArtifactDocument,
    record_ids: Sequence[str],
    *,
    options: DrawingSheetOptions,
    rasters: Mapping[str, Any] | None = None,
) -> DrawingSheetBundle:
    """Compose READY and FRESH records into one printable sheet.

    ``record_ids`` may name vector records and rubbing records alike, in the
    order they should appear.  A rubbing record stores a receipt rather than
    pixels, so its recomputed raster is passed in ``rasters`` under the same
    id and is drawn only if it matches that receipt.
    """

    if not isinstance(options, DrawingSheetOptions):
        raise DrawingSheetError("options must be DrawingSheetOptions")
    rasters = dict(rasters or {})
    ids = [str(record_id) for record_id in record_ids]
    if not ids:
        raise DrawingSheetError("a sheet needs at least one record")
    attached_by_elevation = {
        elevation_id: rubbing_id
        for rubbing_id, elevation_id in options.rubbings_on_axis
    }
    unplaced_rasters = sorted(
        set(rasters) - set(ids) - set(attached_by_elevation.values())
    )
    if unplaced_rasters:
        raise DrawingSheetError(
            "a raster was given for a record the sheet does not draw: "
            f"{', '.join(unplaced_rasters)}"
        )
    if len(ids) > MAX_DRAWING_SHEET_FIGURES:
        raise DrawingSheetError(
            f"a sheet holds at most {MAX_DRAWING_SHEET_FIGURES} figures"
        )
    if len(set(ids)) != len(ids):
        raise DrawingSheetError("the same record cannot appear twice on one sheet")

    # The axis is where the active Align put the artifact, so it is drawable
    # only when that Align established one.  Asking for it under a manual drag
    # is not an error the user can act on; the honest answer is to draw the
    # sheet without a line nothing backs, and to say so in the sidecar.
    align_id = document.active_align_revision_id
    align = (
        document.align_revision_index.get(align_id)
        if isinstance(align_id, str)
        else None
    )
    align_recipe_kind = str(getattr(align, "recipe", {}).get("kind", "") or "")
    draw_center_axis = (
        options.show_center_axis and align_recipe_kind == AXIS_ALIGN_RECIPE_KIND
    )

    mirror_by_elevation = dict(options.mirror_sections)
    unplaced = sorted(set(mirror_by_elevation) - set(ids))
    if unplaced:
        raise DrawingSheetError(
            "the elevation half of a mirrored figure must be one of the sheet's "
            f"records: {', '.join(unplaced)}"
        )
    listed_sections = sorted(set(mirror_by_elevation.values()) & set(ids))
    if listed_sections:
        raise DrawingSheetError(
            "the section half of a mirrored figure is drawn inside that figure, "
            f"so it must not also be a figure of its own: {', '.join(listed_sections)}"
        )

    unplaced = sorted(set(attached_by_elevation) - set(ids))
    if unplaced:
        raise DrawingSheetError(
            "a rubbing is pasted on one of the sheet's figures, so its elevation "
            f"must be one of them: {', '.join(unplaced)}"
        )
    listed_rubbings = sorted(set(attached_by_elevation.values()) & set(ids))
    if listed_rubbings:
        raise DrawingSheetError(
            "a rubbing pasted on the axis is drawn inside that figure, so it "
            f"must not also be a figure of its own: {', '.join(listed_rubbings)}"
        )

    conditions = [
        _require_drawable_condition_record(document, record_id)
        for record_id in options.condition_records
    ]
    grooves = [
        _require_drawable_groove_record(document, record_id)
        for record_id in options.groove_records
    ]
    condition_drawn: list[dict[str, str]] = []
    groove_drawn: list[dict[str, str]] = []
    attached_drawn: list[dict[str, str]] = []
    mirrored: list[dict[str, str]] = []

    prepared: list[_Prepared] = []
    for record_id in ids:
        rubbing_record = document.record_index.get(record_id)
        if (
            rubbing_record is not None
            and rubbing_record.type in RUBBING_RECORD_TYPES
        ):
            if rubbing_record.lifecycle_status is not RecordLifecycleStatus.READY:
                raise DrawingSheetError(
                    f"only READY records may be drawn (record {record_id!r})"
                )
            if document.record_freshness(record_id) is not RecordFreshness.FRESH:
                raise DrawingSheetError(
                    f"only FRESH records may be drawn (record {record_id!r})"
                )
            if record_id not in rasters:
                raise DrawingSheetError(
                    f"record {record_id!r} is a rubbing, so its recomputed raster "
                    "must be given to the sheet; a rubbing record stores a "
                    "receipt, not pixels"
                )
            prepared.append(
                _prepare_raster_figure(
                    document,
                    rubbing_record,
                    rasters[record_id],
                )
            )
            continue
        try:
            record, payload, _record_qc = _require_exportable_record(document, record_id)
        except ArtifactVectorExportError as exc:
            raise DrawingSheetError(str(exc)) from exc
        by_kind: dict[str, list[Any]] = {}
        for path in payload.paths:
            try:
                kind = line_kind_for_record_role(path.role)
            except DrawingStyleError as exc:
                raise DrawingSheetError(str(exc)) from exc
            by_kind.setdefault(kind, []).append(path)
        condition_by_kind, drawn = _condition_paths_for_figure(
            record.type, payload.frame, conditions
        )
        for kind, condition_paths in condition_by_kind.items():
            by_kind.setdefault(kind, []).extend(condition_paths)
        condition_drawn.extend(
            {"figure_record_id": record.id, **entry} for entry in drawn
        )
        groove_by_kind, grooves_drawn = _groove_paths_for_figure(payload, grooves)
        for kind, groove_paths in groove_by_kind.items():
            by_kind.setdefault(kind, []).extend(groove_paths)
        groove_drawn.extend(
            {"figure_record_id": record.id, **entry} for entry in grooves_drawn
        )
        section_record_id = mirror_by_elevation.get(record.id)
        attached: _AttachedRaster | None = None
        attached_rubbing_id = attached_by_elevation.get(record.id)
        if attached_rubbing_id is not None:
            attached = _attach_rubbing_on_axis(
                document,
                rubbing_id=attached_rubbing_id,
                raster=rasters.get(attached_rubbing_id),
                elevation=record,
                elevation_payload=payload,
                fit=options.rubbing_on_axis_fit,
            )
            attached_drawn.append(
                {
                    "figure_record_id": record.id,
                    "rubbing_record_id": attached.record_id,
                }
            )
        if section_record_id is None:
            if draw_center_axis:
                try:
                    axis_path = center_axis_vector_path(payload)
                except ArtifactVectorExportError as exc:
                    raise DrawingSheetError(str(exc)) from exc
                if axis_path is not None:
                    by_kind.setdefault(CENTER_AXIS, []).append(axis_path)
            prepared.append(
                _Prepared(
                    record_id=record.id,
                    record_type=record.type,
                    recipe_hash=record.recipe_hash,
                    payload_sha256=payload.sha256,
                    bounds=_bounds_with_attachment(_payload_bounds(payload), attached),
                    paths_by_kind=by_kind,
                    attached=attached,
                )
            )
            continue
        # A mirrored figure draws its own axis, so the caller's centre-axis
        # switch is not consulted: the two halves meet on that line.
        section, combined, bounds, fill_only_ids = _mirrored_figure(
            document,
            elevation=record,
            elevation_payload=payload,
            elevation_by_kind=by_kind,
            section_record_id=section_record_id,
            axis_ready=align_recipe_kind == AXIS_ALIGN_RECIPE_KIND,
            preset=get_drawing_style_preset(options.style_preset),
        )
        mirrored.append(
            {
                "elevation_record_id": record.id,
                "elevation_side": "left",
                "section_record_id": section.id,
                "section_recipe_hash": section.recipe_hash,
                "section_side": "right",
            }
        )
        prepared.append(
            _Prepared(
                record_id=record.id,
                record_type=record.type,
                recipe_hash=record.recipe_hash,
                payload_sha256=payload.sha256,
                bounds=_bounds_with_attachment(bounds, attached),
                paths_by_kind=combined,
                mirror_section_record_id=section.id,
                fill_only_ids=frozenset(fill_only_ids),
                attached=attached,
            )
        )

    try:
        placed = _lay_out(prepared, options=options)
        scale_bar_lines, scale_bar = _scale_bar_elements(options)
        title_block_lines, title_rows = _title_block_elements(
            options,
            document_manifest_sha256=document.canonical_sha256,
        )
        provenance = _sheet_provenance(
            document,
            placed,
            options=options,
            scale_bar=scale_bar,
            title_rows=title_rows,
            center_axis={
                "align_recipe_kind": align_recipe_kind,
                "align_revision_id": str(align_id or ""),
                "drawn": draw_center_axis,
                "requested": options.show_center_axis,
            },
            condition=(
                {
                    "drawn": sorted(
                        condition_drawn,
                        key=lambda entry: (
                            entry["figure_record_id"],
                            entry["record_id"],
                        ),
                    ),
                    "records": [
                        {
                            "condition_kind": payload.condition,
                            "face_count": payload.face_count,
                            "payload_sha256": payload.sha256,
                            "recipe_hash": record.recipe_hash,
                            "record_id": record.id,
                            "selection_sha256": payload.selection_sha256,
                        }
                        for record, payload in sorted(
                            conditions, key=lambda item: item[0].id
                        )
                    ],
                }
                if conditions
                else None
            ),
            groove=(
                {
                    "drawn": sorted(
                        groove_drawn,
                        key=lambda entry: (
                            entry["figure_record_id"],
                            entry["record_id"],
                        ),
                    ),
                    "records": [
                        {
                            "groove_count": len(payload.grooves),
                            "payload_sha256": payload.sha256,
                            "recipe_hash": record.recipe_hash,
                            "record_id": record.id,
                            "trough_heights_um": [
                                groove.trough_height_um
                                for groove in payload.grooves
                            ],
                        }
                        for record, payload in sorted(
                            grooves, key=lambda item: item[0].id
                        )
                    ],
                }
                if grooves
                else None
            ),
            mirrored=sorted(
                mirrored, key=lambda entry: entry["elevation_record_id"]
            ),
            rubbings_on_axis=sorted(
                attached_drawn, key=lambda entry: entry["figure_record_id"]
            ),
        )
        svg_bytes = _render_sheet(
            placed,
            options=options,
            provenance=provenance,
            scale_bar_lines=scale_bar_lines,
            title_block_lines=title_block_lines,
        )
    except SVGRenderError as exc:
        raise DrawingSheetError(str(exc)) from exc

    sidecar = dict(provenance)
    sidecar["artifact"] = {
        "file": DRAWING_SHEET_SVG_NAME,
        "media_type": "image/svg+xml",
        "sha256": hashlib.sha256(svg_bytes).hexdigest(),
        "size_bytes": len(svg_bytes),
    }
    sidecar_bytes = canonical_json_bytes(sidecar)
    return DrawingSheetBundle(
        svg_bytes=svg_bytes,
        sidecar_bytes=sidecar_bytes,
        svg_sha256=hashlib.sha256(svg_bytes).hexdigest(),
        sidecar_sha256=hashlib.sha256(sidecar_bytes).hexdigest(),
    )


def validate_drawing_sheet_bytes(svg_bytes: bytes, sidecar_bytes: bytes) -> None:
    """Check a sheet against its own sidecar, without the document.

    This is the offline half: it proves the SVG is the one the sidecar
    describes and that the sidecar is internally consistent.  Proving the
    figures are the records they name additionally needs the document, which
    `compose_drawing_sheet` re-derives when it builds the sheet.
    """

    import json  # noqa: PLC0415

    if not isinstance(svg_bytes, (bytes, bytearray)):
        raise DrawingSheetError("svg_bytes must be bytes")
    try:
        sidecar = json.loads(bytes(sidecar_bytes).decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise DrawingSheetError(f"sheet sidecar is not valid JSON: {exc}") from exc
    if not isinstance(sidecar, Mapping):
        raise DrawingSheetError("sheet sidecar must be an object")
    if sidecar.get("format") != DRAWING_SHEET_FORMAT:
        raise DrawingSheetError("sheet sidecar declares an unsupported format")
    if sidecar.get("schema_version") != DRAWING_SHEET_SCHEMA_VERSION:
        raise DrawingSheetError("sheet sidecar declares an unsupported schema version")
    artifact = sidecar.get("artifact")
    if not isinstance(artifact, Mapping):
        raise DrawingSheetError("sheet sidecar has no artifact block")
    if artifact.get("sha256") != hashlib.sha256(bytes(svg_bytes)).hexdigest():
        raise DrawingSheetError("sheet SVG does not match the digest in its sidecar")
    if artifact.get("size_bytes") != len(bytes(svg_bytes)):
        raise DrawingSheetError("sheet SVG does not match the size in its sidecar")
    if canonical_json_bytes(sidecar) != bytes(sidecar_bytes):
        raise DrawingSheetError("sheet sidecar is not in canonical JSON form")

    preset_claim = sidecar.get("style_preset")
    if not isinstance(preset_claim, Mapping):
        raise DrawingSheetError("sheet sidecar has no style preset block")
    try:
        preset = get_drawing_style_preset(str(preset_claim.get("preset_id")))
    except DrawingStyleError as exc:
        raise DrawingSheetError(str(exc)) from exc
    if preset_claim.get("sha256") != preset.sha256():
        raise DrawingSheetError(
            "drawing style preset no longer matches the digest recorded with this sheet"
        )

    # The scale must be on the page, not only in the metadata.
    rows = sidecar.get("title_block")
    if not isinstance(rows, Sequence) or not any(
        isinstance(row, Mapping) and row.get("value") == sidecar.get("physical_scale")
        for row in rows
    ):
        raise DrawingSheetError("sheet title block does not print its own scale")


__all__ = [
    "DRAWING_SHEET_FORMAT",
    "DRAWING_SHEET_SCHEMA_VERSION",
    "DRAWING_SHEET_SIDECAR_NAME",
    "DRAWING_SHEET_SVG_NAME",
    "DrawingSheetBundle",
    "DrawingSheetError",
    "DrawingSheetOptions",
    "MAX_DRAWING_SHEET_CONDITION_RECORDS",
    "MAX_DRAWING_SHEET_FIGURES",
    "ORIENTATIONS",
    "PAGE_SIZES_MM",
    "SheetPage",
    "TitleBlock",
    "compose_drawing_sheet",
    "scale_bar_label",
    "scale_bar_length_mm",
    "validate_drawing_sheet_bytes",
]
