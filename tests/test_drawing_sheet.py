from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from src.core.artifact_condition_annotation import (
    commit_condition_annotation,
    compute_condition_annotation,
)
from src.core.artifact_document import RecordFreshness
from src.core.artifact_outline_extractor import compute_artifact_outline
from src.core.artifact_rubbing_extractor import (
    DigitalRubbingRaster,
    commit_artifact_rubbing,
    compute_artifact_rubbing,
)
from src.core.artifact_session import ArtifactSession
from src.core.artifact_vector_extractor import (
    commit_vector_computation,
    compute_artifact_cutline,
)
from src.core.artifact_vector_record import PlanarFrame
from src.core.canonical_json import canonical_json_bytes
from src.core.drawing_sheet import (
    COMPUTED_RUBBING_NOTE,
    DrawingSheetError,
    DrawingSheetOptions,
    SheetPage,
    TitleBlock,
    compose_drawing_sheet,
    computed_rubbing_caption,
    scale_bar_label,
    scale_bar_length_mm,
    validate_drawing_sheet_bytes,
)
from src.core.drawing_style import PROVISIONAL_PRESET_ID, get_preset
from src.core.mesh_import_recipe import current_mesh_import_recipe
from src.core.mesh_loader import MeshData
from src.core.source_identity import SourceFingerprint


SVG_NS = "{http://www.w3.org/2000/svg}"

CUTLINE_ID = "record:sheet-cutline"
OUTLINE_ID = "record:sheet-outline"


def _session(half_extent_mm: float = 1.0) -> ArtifactSession:
    """A tetrahedron whose canonical extent is 2 x half_extent_mm millimetres."""

    h = float(half_extent_mm)
    mesh = MeshData(
        vertices=np.array(
            [[h, h, h], [-h, -h, h], [-h, h, -h], [h, -h, -h]],
            dtype=np.float64,
        ),
        faces=np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32),
        unit="mm",
        source_identity=SourceFingerprint(
            sha256="c" * 64,
            size_bytes=64,
            mtime_ns=1,
            original_name="sheet.ply",
            format="ply",
        ),
        source_format="ply",
        source_import_recipe=current_mesh_import_recipe("ply"),
    )
    session = ArtifactSession.create_from_source(
        mesh,
        resolved_source_path="/source/sheet.ply",
        unit="mm",
        axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
        handedness="right",
        software_version="sheet-test",
        operator="tester",
        created_at="2026-09-03T00:00:00Z",
        document_id="artifact:sheet-test",
        metadata_revision_id="metadata:sheet-test",
        align_revision_id="align:sheet-test",
    )
    cutline = compute_artifact_cutline(
        session,
        PlanarFrame(
            origin_world_mm=(0.0, 0.0, 0.0),
            u_axis_world=(1.0, 0.0, 0.0),
            v_axis_world=(0.0, 1.0, 0.0),
            normal_world=(0.0, 0.0, 1.0),
        ),
    )
    session = commit_vector_computation(
        session,
        cutline,
        record_id=CUTLINE_ID,
        created_at="2026-09-03T00:00:01Z",
        operator="tester",
    )
    outline = compute_artifact_outline(session, "top", precision_grid_mm=0.01)
    return commit_vector_computation(
        session,
        outline,
        record_id=OUTLINE_ID,
        created_at="2026-09-03T00:00:02Z",
        operator="tester",
    )


def _options(**overrides) -> DrawingSheetOptions:
    settings = {
        "title_block": TitleBlock(
            artifact_label="시험 유물 001",
            rows=(("작성", "tester"), ("일자", "2026-09-03")),
        ),
        "scale_denominator": 1.0,
    }
    settings.update(overrides)
    return DrawingSheetOptions(**settings)


def test_a_sheet_carries_every_requested_figure_in_the_given_order() -> None:
    document = _session().document
    bundle = compose_drawing_sheet(
        document, [OUTLINE_ID, CUTLINE_ID], options=_options()
    )

    root = ET.fromstring(bundle.svg_bytes)
    figures = root.find(f"{SVG_NS}g[@id='sheet-figures']")
    assert figures is not None
    assert [child.attrib["data-record-id"] for child in figures] == [
        OUTLINE_ID,
        CUTLINE_ID,
    ]


def test_the_page_is_the_declared_physical_size() -> None:
    document = _session().document
    for size, orientation, width, height in (
        ("A4", "portrait", "210", "297"),
        ("A4", "landscape", "297", "210"),
        ("A3", "portrait", "297", "420"),
    ):
        bundle = compose_drawing_sheet(
            document,
            [OUTLINE_ID],
            options=_options(page=SheetPage(size=size, orientation=orientation)),
        )
        root = ET.fromstring(bundle.svg_bytes)
        assert root.attrib["width"] == f"{width}mm"
        assert root.attrib["height"] == f"{height}mm"
        assert root.attrib["viewBox"] == f"0 0 {width} {height}"


def test_reducing_the_scale_shrinks_coordinates_but_never_line_weights() -> None:
    """A 0.35 mm cut line is 0.35 mm on paper at 1:1 and at 1:4 alike.

    This is the whole reason weights are paper millimetres: a drawing reduced
    to fit a report page must still print lines a reader can tell apart.
    """

    document = _session().document
    preset = get_preset(PROVISIONAL_PRESET_ID)
    expected = str(preset.style("outline_visible").stroke_width_mm)

    widths: list[float] = []
    for denominator in (1.0, 2.0, 4.0):
        bundle = compose_drawing_sheet(
            document,
            [OUTLINE_ID],
            options=_options(scale_denominator=denominator),
        )
        root = ET.fromstring(bundle.svg_bytes)
        layer = root.find(
            f"{SVG_NS}g[@id='sheet-figures']/{SVG_NS}g/{SVG_NS}g[@id='layer-outline-visible']"
        )
        assert layer is not None
        assert layer.attrib["stroke-width"] == expected

        sidecar = json.loads(bundle.sidecar_bytes)
        widths.append(sidecar["figures"][0]["width_mm"])

    assert widths[0] == pytest.approx(widths[1] * 2.0)
    assert widths[0] == pytest.approx(widths[2] * 4.0)


def test_the_sheet_prints_its_own_scale() -> None:
    """A reduced drawing that does not state its reduction cannot be measured."""

    document = _session().document
    bundle = compose_drawing_sheet(
        document, [OUTLINE_ID], options=_options(scale_denominator=3.0)
    )

    root = ET.fromstring(bundle.svg_bytes)
    texts = [element.text for element in root.iter(f"{SVG_NS}text")]
    assert "1:3" in texts
    assert "축척" in texts

    sidecar = json.loads(bundle.sidecar_bytes)
    assert sidecar["physical_scale"] == "1:3"
    assert {"label": "축척", "value": "1:3"} in sidecar["title_block"]


def test_a_caller_cannot_supply_a_competing_scale_row() -> None:
    """The scale row is derived, so it cannot disagree with the geometry."""

    options = _options(
        scale_denominator=2.0,
        title_block=TitleBlock(artifact_label="A", rows=(("축척", "1:99"),)),
    )
    document = _session().document
    bundle = compose_drawing_sheet(document, [OUTLINE_ID], options=options)

    sidecar = json.loads(bundle.sidecar_bytes)
    derived = [row for row in sidecar["title_block"] if row["label"] == "축척"]
    assert {"label": "축척", "value": "1:2"} in derived


def test_figures_never_reach_into_the_footer_band() -> None:
    """A drawing that crowds the title block is a drawing that lost its caption."""

    options = _options()
    page = options.page
    title_block_top = page.height_mm - page.margin_mm - options.title_block_height_mm
    last_drawable_y = page.margin_mm + options.content_height_mm

    assert last_drawable_y < title_block_top

    document = _session().document
    bundle = compose_drawing_sheet(
        document, [OUTLINE_ID, CUTLINE_ID], options=options
    )
    sidecar = json.loads(bundle.sidecar_bytes)
    for figure in sidecar["figures"]:
        bottom = figure["origin_mm"][1] + figure["height_mm"]
        assert bottom <= last_drawable_y


def test_content_that_does_not_fit_is_refused_with_a_usable_scale() -> None:
    """Silently shrinking would print a page that says 1:1 and measures 1:3."""

    document = _session(half_extent_mm=200.0).document

    with pytest.raises(DrawingSheetError, match="does not fit") as caught:
        compose_drawing_sheet(
            document,
            [OUTLINE_ID],
            options=_options(page=SheetPage(size="A5"), scale_denominator=1.0),
        )
    message = str(caught.value)
    assert "scale denominator of" in message

    # The suggested scale is one that actually works.
    suggested = float(message.split("scale denominator of")[1].split()[0])
    compose_drawing_sheet(
        document,
        [OUTLINE_ID],
        options=_options(page=SheetPage(size="A5"), scale_denominator=suggested),
    )


def test_a_sheet_may_not_enlarge_a_measured_drawing() -> None:
    with pytest.raises(DrawingSheetError, match="at least 1"):
        _options(scale_denominator=0.5)


def test_a_scale_bar_spans_a_length_a_reader_can_name() -> None:
    for denominator in (1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 50.0):
        length = scale_bar_length_mm(denominator)
        mantissa = length
        while mantissa >= 10.0:
            mantissa /= 10.0
        while mantissa < 1.0:
            mantissa *= 10.0
        assert round(mantissa, 9) in (1.0, 2.0, 5.0)
        assert length / denominator <= 90.0


def test_scale_bar_labels_use_the_unit_a_reader_would_say() -> None:
    assert scale_bar_label(50.0) == "5 cm"
    assert scale_bar_label(5.0) == "5 mm"
    assert scale_bar_label(1000.0) == "1 m"
    assert scale_bar_label(2000.0) == "2 m"


def test_the_scale_bar_is_drawn_at_the_reduced_length() -> None:
    document = _session().document
    bundle = compose_drawing_sheet(
        document, [OUTLINE_ID], options=_options(scale_denominator=4.0)
    )

    sidecar = json.loads(bundle.sidecar_bytes)
    bar = sidecar["scale_bar"]
    assert bar["paper_length_mm"] == pytest.approx(bar["artifact_length_mm"] / 4.0)

    root = ET.fromstring(bundle.svg_bytes)
    cells = root.findall(f"{SVG_NS}g[@id='scale-bar']/{SVG_NS}rect")
    assert len(cells) == bar["segments"]
    drawn = sum(float(cell.attrib["width"]) for cell in cells)
    assert drawn == pytest.approx(bar["paper_length_mm"])


def test_a_sheet_is_deterministic_and_checks_against_its_own_sidecar() -> None:
    document = _session().document
    first = compose_drawing_sheet(
        document, [OUTLINE_ID, CUTLINE_ID], options=_options()
    )
    second = compose_drawing_sheet(
        document, [OUTLINE_ID, CUTLINE_ID], options=_options()
    )

    assert first.svg_bytes == second.svg_bytes
    assert first.sidecar_bytes == second.sidecar_bytes
    validate_drawing_sheet_bytes(first.svg_bytes, first.sidecar_bytes)

    with pytest.raises(DrawingSheetError, match="does not match the digest"):
        validate_drawing_sheet_bytes(
            first.svg_bytes + b"<!-- -->", first.sidecar_bytes
        )


def test_a_sheet_records_the_payload_digest_of_every_figure() -> None:
    """A sheet must be checkable against the records it claims to show."""

    session = _session()
    document = session.document
    bundle = compose_drawing_sheet(
        document, [OUTLINE_ID, CUTLINE_ID], options=_options()
    )

    sidecar = json.loads(bundle.sidecar_bytes)
    assert sidecar["document_manifest_sha256"] == document.canonical_sha256
    for figure in sidecar["figures"]:
        record = document.record_index[figure["record_id"]]
        assert figure["recipe_hash"] == record.recipe_hash
        assert figure["record_type"] == record.type
        assert len(figure["vector_payload_sha256"]) == 64


def test_a_stale_or_missing_record_never_reaches_a_sheet() -> None:
    document = _session().document

    with pytest.raises(DrawingSheetError, match="does not exist"):
        compose_drawing_sheet(document, ["record:nope"], options=_options())

    with pytest.raises(DrawingSheetError, match="at least one"):
        compose_drawing_sheet(document, [], options=_options())

    with pytest.raises(DrawingSheetError, match="cannot appear twice"):
        compose_drawing_sheet(
            document, [OUTLINE_ID, OUTLINE_ID], options=_options()
        )


def test_an_edited_preset_invalidates_a_sheet_that_used_it() -> None:
    document = _session().document
    bundle = compose_drawing_sheet(document, [OUTLINE_ID], options=_options())
    sidecar = json.loads(bundle.sidecar_bytes)

    sidecar["style_preset"]["sha256"] = "0" * 64
    forged = canonical_json_bytes(sidecar)
    with pytest.raises(DrawingSheetError, match="no longer matches the digest"):
        validate_drawing_sheet_bytes(bundle.svg_bytes, forged)


def test_the_title_block_names_who_recorded_the_artifact() -> None:
    """The GUI fills the 작성 row from the Align revision, not from a text box.

    A drawing should name whoever measured the artifact, not whoever happened
    to print it, so this reproduces the slot's own construction headlessly.
    """

    session = _session()
    document = session.document
    align = document.align_revision_index[document.active_align_revision_id]

    bundle = compose_drawing_sheet(
        document,
        [OUTLINE_ID],
        options=DrawingSheetOptions(
            title_block=TitleBlock(
                artifact_label="시험 유물 001",
                rows=(("작성", align.operator),),
            ),
        ),
    )

    rows = json.loads(bundle.sidecar_bytes)["title_block"]
    assert {"label": "작성", "value": align.operator} in rows
    assert rows[0] == {"label": "유물", "value": "시험 유물 001"}
    assert rows[-1]["label"] == "문서"


def test_a_sheet_with_no_operator_row_is_still_well_formed() -> None:
    """An Align revision without an operator must not produce an empty row."""

    document = _session().document
    bundle = compose_drawing_sheet(
        document,
        [OUTLINE_ID],
        options=DrawingSheetOptions(title_block=TitleBlock(artifact_label="A")),
    )

    rows = json.loads(bundle.sidecar_bytes)["title_block"]
    assert [row["label"] for row in rows] == ["유물", "축척", "문서"]
    validate_drawing_sheet_bytes(bundle.svg_bytes, bundle.sidecar_bytes)


def test_an_unknown_page_size_or_orientation_is_named_in_the_error() -> None:
    with pytest.raises(DrawingSheetError, match="known sizes are"):
        SheetPage(size="B4")
    with pytest.raises(DrawingSheetError, match="orientation must be"):
        SheetPage(orientation="sideways")


# --- condition annotations ----------------------------------------------------

CONDITION_ID = "record:sheet-condition"


def _condition_session() -> ArtifactSession:
    """The sheet session with one restored region covering a single face."""

    session = _session()
    computation = compute_condition_annotation(
        session,
        condition="restored",
        face_indices=[0],
        precision_grid_mm=0.01,
    )
    return commit_condition_annotation(
        session,
        computation,
        record_id=CONDITION_ID,
        created_at="2026-09-03T00:00:03Z",
        operator="tester",
    )


def test_a_sheet_without_condition_records_is_the_sheet_it_always_was() -> None:
    """The overlay is opt-in, down to the byte."""

    document = _condition_session().document
    plain = compose_drawing_sheet(document, [OUTLINE_ID], options=_options())
    explicit = compose_drawing_sheet(
        document, [OUTLINE_ID], options=_options(condition_records=())
    )

    assert plain.svg_bytes == explicit.svg_bytes
    assert b"condition" not in plain.svg_bytes
    assert "condition" not in json.loads(plain.sidecar_bytes.decode("utf-8"))


def test_a_condition_region_is_drawn_on_the_figure_that_shares_its_view() -> None:
    document = _condition_session().document
    bundle = compose_drawing_sheet(
        document,
        [OUTLINE_ID, CUTLINE_ID],
        options=_options(condition_records=(CONDITION_ID,)),
    )

    root = ET.fromstring(bundle.svg_bytes)
    figures = root.find(f"{SVG_NS}g[@id='sheet-figures']")
    assert figures is not None
    by_record = {child.attrib["data-record-id"]: child for child in figures}

    # The outline is a top view, and the condition record holds a top boundary.
    outline_layers = [
        layer.attrib["id"] for layer in by_record[OUTLINE_ID].findall(f"{SVG_NS}g")
    ]
    assert "layer-condition-restored" in outline_layers
    # The layer sits after the outline it annotates and before any axis.
    assert outline_layers.index("layer-outline-visible") < outline_layers.index(
        "layer-condition-restored"
    )

    # The cutline is a section on an arbitrary plane; no boundary matches it.
    cutline_layers = [
        layer.attrib["id"] for layer in by_record[CUTLINE_ID].findall(f"{SVG_NS}g")
    ]
    assert not [layer for layer in cutline_layers if "condition" in layer]

    condition_layer = by_record[OUTLINE_ID].find(
        f"{SVG_NS}g[@id='layer-condition-restored']"
    )
    assert condition_layer is not None
    assert condition_layer.attrib["stroke-dasharray"] == "3,1,0.5,1"
    for path in condition_layer:
        assert path.attrib["id"].startswith(f"condition:{CONDITION_ID}:top:")


def test_the_sidecar_says_which_region_was_drawn_where() -> None:
    document = _condition_session().document
    bundle = compose_drawing_sheet(
        document,
        [OUTLINE_ID, CUTLINE_ID],
        options=_options(condition_records=(CONDITION_ID,)),
    )

    sidecar = json.loads(bundle.sidecar_bytes.decode("utf-8"))
    condition = sidecar["condition"]
    assert [entry["record_id"] for entry in condition["records"]] == [CONDITION_ID]
    assert condition["records"][0]["condition_kind"] == "restored"
    assert condition["records"][0]["face_count"] == 1
    assert condition["drawn"] == [
        {
            "condition_kind": "restored",
            "figure_record_id": OUTLINE_ID,
            "line_kind": "condition_restored",
            "record_id": CONDITION_ID,
            "view": "top",
        }
    ]
    validate_drawing_sheet_bytes(bundle.svg_bytes, bundle.sidecar_bytes)


# --- technique annotations ----------------------------------------------------

TECHNIQUE_ID = "record:sheet-technique"


def _technique_session(technique: str = "coil_joint") -> ArtifactSession:
    """The sheet session with one technique mark covering a single face."""

    from src.core.artifact_technique_annotation import (  # noqa: PLC0415
        commit_technique_annotation,
        compute_technique_annotation,
    )

    session = _session()
    computation = compute_technique_annotation(
        session,
        technique=technique,
        face_indices=[0],
        precision_grid_mm=0.01,
    )
    return commit_technique_annotation(
        session,
        computation,
        record_id=TECHNIQUE_ID,
        created_at="2026-09-04T00:00:03Z",
        operator="tester",
    )


def test_a_sheet_without_technique_records_is_the_sheet_it_always_was() -> None:
    document = _technique_session().document
    plain = compose_drawing_sheet(document, [OUTLINE_ID], options=_options())
    explicit = compose_drawing_sheet(
        document, [OUTLINE_ID], options=_options(technique_records=())
    )
    assert explicit.svg_bytes == plain.svg_bytes
    assert explicit.sidecar_bytes == plain.sidecar_bytes
    assert b"technique" not in plain.svg_bytes
    assert "technique" not in json.loads(plain.sidecar_bytes.decode("utf-8"))


def test_a_technique_mark_is_drawn_as_strokes_on_the_figure_that_shares_its_view() -> None:
    document = _technique_session("water_smoothing").document
    bundle = compose_drawing_sheet(
        document,
        [OUTLINE_ID, CUTLINE_ID],
        options=_options(technique_records=(TECHNIQUE_ID,)),
    )

    root = ET.fromstring(bundle.svg_bytes)
    figures = root.find(f"{SVG_NS}g[@id='sheet-figures']")
    assert figures is not None
    by_record = {child.attrib["data-record-id"]: child for child in figures}
    outline_layers = [
        child.attrib["id"]
        for child in by_record[OUTLINE_ID]
        if child.tag == f"{SVG_NS}g"
    ]
    assert "layer-technique-water-smoothing" in outline_layers
    # Technique sits over the outline and under condition and the axis.
    assert outline_layers.index("layer-outline-visible") < outline_layers.index(
        "layer-technique-water-smoothing"
    )
    cutline_layers = [
        child.attrib["id"]
        for child in by_record[CUTLINE_ID]
        if child.tag == f"{SVG_NS}g"
    ]
    assert not [layer for layer in cutline_layers if "technique" in layer]
    layer = by_record[OUTLINE_ID].find(
        f"{SVG_NS}g[@id='layer-technique-water-smoothing']"
    )
    assert layer is not None
    # One fine solid pen: the mark is told by its strokes, not a dash code.
    assert "stroke-dasharray" not in layer.attrib
    paths = list(layer)
    # Wet-hand lines are open lines laid across the region, never the
    # region's boundary.
    assert paths
    assert paths[0].attrib["id"] == f"technique:{TECHNIQUE_ID}:top:0"
    assert not any(path.attrib["d"].rstrip().endswith("Z") for path in paths)

    sidecar = json.loads(bundle.sidecar_bytes.decode("utf-8"))
    technique = sidecar["technique"]
    assert [entry["record_id"] for entry in technique["records"]] == [TECHNIQUE_ID]
    assert technique["records"][0]["technique_kind"] == "water_smoothing"
    assert technique["records"][0]["face_count"] == 1
    (entry,) = technique["drawn"]
    assert entry["figure_record_id"] == OUTLINE_ID
    assert entry["line_kind"] == "technique_water_smoothing"
    assert entry["representation"] == "parallel_lines"
    assert entry["stroke_count"] == len(paths)
    assert entry["angle_deg"] == 0.0
    # Nothing decided this mark's wall but the faces the drafter painted.
    assert entry["side_decided_by"] == "surface_side"
    assert entry["view"] == "top"
    assert entry["seed"].startswith(f"{TECHNIQUE_ID}:top:")
    assert (
        technique["styles"]["technique_water_smoothing"]["representation"]
        == "parallel_lines"
    )
    validate_drawing_sheet_bytes(bundle.svg_bytes, bundle.sidecar_bytes)

    # The same sheet twice is the same bytes: the strokes are seeded.
    again = compose_drawing_sheet(
        document,
        [OUTLINE_ID, CUTLINE_ID],
        options=_options(technique_records=(TECHNIQUE_ID,)),
    )
    assert again.svg_bytes == bundle.svg_bytes


def test_a_finger_mark_is_drawn_as_an_oval_inside_the_region_not_by_its_boundary() -> None:
    from shapely.geometry import Polygon  # noqa: PLC0415

    from src.core.artifact_technique_annotation import (  # noqa: PLC0415
        technique_payload_from_record,
    )

    document = _technique_session("finger_mark").document
    bundle = compose_drawing_sheet(
        document,
        [OUTLINE_ID],
        options=_options(technique_records=(TECHNIQUE_ID,)),
    )

    root = ET.fromstring(bundle.svg_bytes)
    figures = root.find(f"{SVG_NS}g[@id='sheet-figures']")
    assert figures is not None
    figure = next(iter(figures))
    layer = figure.find(f"{SVG_NS}g[@id='layer-technique-finger-mark']")
    assert layer is not None
    paths = list(layer)
    # One press on one face: one closed oval, not the face's triangle.
    assert len(paths) == 1
    assert paths[0].attrib["id"] == f"technique:{TECHNIQUE_ID}:top:0"
    assert paths[0].attrib["d"].rstrip().endswith("Z")
    assert paths[0].attrib.get("fill", "none") == "none"

    payload = technique_payload_from_record(document.record_index[TECHNIQUE_ID])
    top = next(view for view in payload.views if view.view == "top")
    region = Polygon(top.outline.paths[0].points_mm)
    # The oval sits on the region and is about its size.
    numbers = [float(token) for token in paths[0].attrib["d"].replace("M", " ").replace("L", " ").replace("Z", " ").split()]
    points = list(zip(numbers[0::2], numbers[1::2]))
    assert len(points) >= 12

    sidecar = json.loads(bundle.sidecar_bytes.decode("utf-8"))
    (entry,) = sidecar["technique"]["drawn"]
    assert entry["representation"] == "press_ovals"
    assert entry["stroke_count"] == 1
    assert region.area > 0.0
    validate_drawing_sheet_bytes(bundle.svg_bytes, bundle.sidecar_bytes)


def test_a_press_can_be_asked_for_as_an_inverted_u() -> None:
    """The drafter says which of the two readings of a press is on the paper."""

    document = _technique_session("finger_mark").document
    closed = compose_drawing_sheet(
        document, [OUTLINE_ID], options=_options(technique_records=(TECHNIQUE_ID,))
    )
    opened = compose_drawing_sheet(
        document,
        [OUTLINE_ID],
        options=_options(
            technique_records=(TECHNIQUE_ID,),
            technique_representations=((TECHNIQUE_ID, "press_arcs"),),
        ),
    )
    assert opened.svg_bytes != closed.svg_bytes

    figure = next(
        iter(ET.fromstring(opened.svg_bytes).find(f"{SVG_NS}g[@id='sheet-figures']"))
    )
    layer = figure.find(f"{SVG_NS}g[@id='layer-technique-finger-mark']")
    assert layer is not None
    (path,) = list(layer)
    # An arc has no interior, so it neither closes nor is filled.
    assert not path.attrib["d"].rstrip().endswith("Z")
    assert path.attrib.get("fill", "none") == "none"

    entry = json.loads(opened.sidecar_bytes)["technique"]["drawn"][0]
    assert entry["representation"] == "press_arcs"
    assert entry["stroke_count"] == 1
    validate_drawing_sheet_bytes(opened.svg_bytes, opened.sidecar_bytes)

    # A kind with one drawing does not take a preference.
    seam = _technique_session("coil_joint").document
    with pytest.raises(DrawingSheetError, match="not drawn as"):
        compose_drawing_sheet(
            seam,
            [OUTLINE_ID],
            options=_options(
                technique_records=(TECHNIQUE_ID,),
                technique_representations=((TECHNIQUE_ID, "press_arcs"),),
                mirror_sections=(),
            ),
        )
    with pytest.raises(DrawingSheetError, match="not in technique_records"):
        _options(technique_representations=((TECHNIQUE_ID, "press_arcs"),))
    with pytest.raises(DrawingSheetError, match="must be one of"):
        _options(
            technique_records=(TECHNIQUE_ID,),
            technique_representations=((TECHNIQUE_ID, "scribble"),),
        )


def test_a_coil_seam_without_a_section_half_says_so_rather_than_moving_outside() -> None:
    """테쌓기흔 is read on the inner wall, so a plain figure cannot carry it."""

    document = _technique_session("coil_joint").document
    bundle = compose_drawing_sheet(
        document, [OUTLINE_ID], options=_options(technique_records=(TECHNIQUE_ID,))
    )
    figure = next(
        iter(ET.fromstring(bundle.svg_bytes).find(f"{SVG_NS}g[@id='sheet-figures']"))
    )
    assert figure.find(f"{SVG_NS}g[@id='layer-technique-coil-joint']") is None

    technique = json.loads(bundle.sidecar_bytes)["technique"]
    assert technique["drawn"] == []
    assert technique["not_drawn"] == [
        {
            "figure_record_id": OUTLINE_ID,
            "reason": "interior_needs_section_half",
            "record_id": TECHNIQUE_ID,
        }
    ]
    # The faces themselves are on the outside; the convention still rules.
    assert technique["records"][0]["surface_side"] == "exterior"


def test_a_direction_given_for_a_record_turns_its_strokes() -> None:
    document = _technique_session("wood_grain_smoothing").document
    default = compose_drawing_sheet(
        document, [OUTLINE_ID], options=_options(technique_records=(TECHNIQUE_ID,))
    )
    turned = compose_drawing_sheet(
        document,
        [OUTLINE_ID],
        options=_options(
            technique_records=(TECHNIQUE_ID,),
            technique_angles_deg=((TECHNIQUE_ID, 30.0),),
        ),
    )
    assert turned.svg_bytes != default.svg_bytes
    default_entry = json.loads(default.sidecar_bytes)["technique"]["drawn"][0]
    turned_entry = json.loads(turned.sidecar_bytes)["technique"]["drawn"][0]
    assert default_entry["angle_deg"] == 90.0
    assert turned_entry["angle_deg"] == 30.0
    assert json.loads(turned.sidecar_bytes)["technique"]["styles"][
        "technique_wood_grain"
    ]["angle_deg"] == 30.0

    with pytest.raises(DrawingSheetError, match="not in technique_records"):
        _options(technique_angles_deg=((TECHNIQUE_ID, 30.0),))
    with pytest.raises(DrawingSheetError, match="finite"):
        _options(
            technique_records=(TECHNIQUE_ID,),
            technique_angles_deg=((TECHNIQUE_ID, float("nan")),),
        )


def test_a_sheet_refuses_a_record_that_is_not_a_technique_annotation() -> None:
    document = _technique_session().document

    with pytest.raises(DrawingSheetError, match="not a technique annotation"):
        compose_drawing_sheet(
            document,
            [OUTLINE_ID],
            options=_options(technique_records=(OUTLINE_ID,)),
        )
    with pytest.raises(DrawingSheetError, match="not a condition annotation"):
        compose_drawing_sheet(
            document,
            [OUTLINE_ID],
            options=_options(condition_records=(TECHNIQUE_ID,)),
        )
    with pytest.raises(DrawingSheetError, match="does not exist"):
        compose_drawing_sheet(
            document,
            [OUTLINE_ID],
            options=_options(technique_records=("record:nowhere",)),
        )
    with pytest.raises(DrawingSheetError, match="cannot be drawn twice"):
        _options(technique_records=(TECHNIQUE_ID, TECHNIQUE_ID))


def test_a_sheet_refuses_a_record_that_is_not_a_condition_annotation() -> None:
    document = _condition_session().document

    with pytest.raises(DrawingSheetError, match="not a condition annotation"):
        compose_drawing_sheet(
            document,
            [OUTLINE_ID],
            options=_options(condition_records=(OUTLINE_ID,)),
        )
    with pytest.raises(DrawingSheetError, match="does not exist"):
        compose_drawing_sheet(
            document,
            [OUTLINE_ID],
            options=_options(condition_records=("record:absent",)),
        )
    with pytest.raises(DrawingSheetError, match="cannot be drawn twice"):
        _options(condition_records=(CONDITION_ID, CONDITION_ID))


def test_a_condition_under_a_superseded_alignment_is_refused() -> None:
    """A boundary is where the artifact was; a new Align moves the artifact."""

    session = _condition_session()
    moved = session.commit_preview(
        translation_mm=(5.0, 0.0, 0.0),
        rotation_deg=(0.0, 0.0, 0.0),
        scale=1.0,
        operator="tester",
        created_at="2026-09-03T00:00:04Z",
        revision_id="align:moved",
    )

    assert moved.document.record_freshness(CONDITION_ID) is not RecordFreshness.FRESH
    with pytest.raises(DrawingSheetError, match="only FRESH condition records"):
        compose_drawing_sheet(
            moved.document,
            [OUTLINE_ID],
            options=_options(condition_records=(CONDITION_ID,)),
        )


RUBBING_ID = "record:sheet-rubbing"


def _session_with_rubbing(
    *, pixels_per_mm: int = 10, view: str = "top"
) -> tuple[ArtifactSession, DigitalRubbingRaster]:
    """The tetrahedron plus a six-view rubbing of it, ready to be placed."""

    session = _session()
    computation = compute_artifact_rubbing(
        session,
        view,
        pixels_per_mm=pixels_per_mm,
        margin_um=0,
        reference_radius_um=500,
        depth_quantization_um=10,
        black_point_um=100,
        ink_strength_percent=100,
        relief_polarity="bidirectional",
    )
    committed = commit_artifact_rubbing(
        session,
        computation,
        record_id=RUBBING_ID,
        created_at="2026-09-03T00:00:03Z",
        operator="tester",
    )
    return committed, computation.raster


def _images(svg_bytes: bytes) -> list[dict[str, str]]:
    root = ET.fromstring(svg_bytes.decode("utf-8"))
    return [
        dict(element.attrib)
        for element in root.iter(f"{SVG_NS}image")
    ]


def test_a_rubbing_is_placed_beside_the_drawing_at_its_own_size() -> None:
    session, raster = _session_with_rubbing()

    bundle = compose_drawing_sheet(
        session.document,
        [OUTLINE_ID, RUBBING_ID],
        options=_options(),
        rasters={RUBBING_ID: raster},
    )
    images = _images(bundle.svg_bytes)
    assert len(images) == 1
    image = images[0]
    expected_mm = raster.width_pixels * 1000.0 / raster.pixels_per_meter
    assert float(image["width"]) == pytest.approx(expected_mm, abs=1e-6)
    assert float(image["height"]) == pytest.approx(
        raster.height_pixels * 1000.0 / raster.pixels_per_meter, abs=1e-6
    )
    assert image["preserveAspectRatio"] == "none"
    assert image[
        "{http://www.w3.org/1999/xlink}href"
    ].startswith("data:image/png;base64,")

    sidecar = json.loads(bundle.sidecar_bytes.decode("utf-8"))
    figures = {figure["record_id"]: figure for figure in sidecar["figures"]}
    assert set(figures) == {OUTLINE_ID, RUBBING_ID}
    rubbing_figure = figures[RUBBING_ID]
    assert rubbing_figure["raster_sha256"] == raster.raster_sha256
    assert rubbing_figure["raster_pixels_per_meter"] == raster.pixels_per_meter
    assert rubbing_figure["raster_width_pixels"] == raster.width_pixels
    # A rubbing has no vector payload, and the sidecar does not pretend it does.
    assert "vector_payload_sha256" not in rubbing_figure
    assert "vector_payload_sha256" in figures[OUTLINE_ID]


def test_a_sheet_with_a_rubbing_says_the_rubbing_was_computed() -> None:
    """A rubbing computed from a mesh can pass for paper and ink, and a reader
    who takes it for a paper rubbing has been misled about what was measured.
    So the sheet says so where the reader looks, and nothing can leave it out:
    a mandatory title block row, and under the rubbing a caption naming the
    model and the numbers that turned depth into ink."""

    session, raster = _session_with_rubbing()
    bundle = compose_drawing_sheet(
        session.document,
        [OUTLINE_ID, RUBBING_ID],
        options=_options(),
        rasters={RUBBING_ID: raster},
    )
    validate_drawing_sheet_bytes(bundle.svg_bytes, bundle.sidecar_bytes)
    svg = bundle.svg_bytes.decode("utf-8")
    sidecar = json.loads(bundle.sidecar_bytes.decode("utf-8"))

    rows = sidecar["title_block"]
    labels = [row["label"] for row in rows]
    # Right after the scale, before the caller's own rows.
    assert labels[:3] == ["유물", "축척", "탁본"] and labels[-1] == "문서"
    assert rows[2]["value"] == COMPUTED_RUBBING_NOTE
    assert COMPUTED_RUBBING_NOTE in svg
    assert sidecar["computed_rubbing_note"] == COMPUTED_RUBBING_NOTE

    figures = {figure["record_id"]: figure for figure in sidecar["figures"]}
    caption = figures[RUBBING_ID]["caption"]
    # Read off the recipe the raster was made with, not typed anywhere.
    # A six-view raster inks the wall as seen from one direction; paper would
    # have followed the curvature.  The caption calls it what it is.
    assert caption == "정사영 요철 · 전개 아님 · 높이 모델 · 창 0.5 mm · 검정 0.1 mm · 먹 100%"
    assert caption == computed_rubbing_caption(
        session.document.record_index[RUBBING_ID].recipe, developed=False
    )
    assert "caption" not in figures[OUTLINE_ID]
    assert 'id="rubbing-caption-0001"' in svg
    assert caption in svg
    # The caption sits under the paper, inside the figure's own extent, so
    # the paper is still drawn at its own physical size.
    image = _images(bundle.svg_bytes)[0]
    assert float(image["height"]) == pytest.approx(
        raster.height_pixels * 1000.0 / raster.pixels_per_meter, abs=1e-6
    )
    assert figures[RUBBING_ID]["height_mm"] > float(image["height"])

    # A sheet of line work says nothing about rubbings and keeps its bytes.
    plain = compose_drawing_sheet(session.document, [OUTLINE_ID], options=_options())
    plain_sidecar = json.loads(plain.sidecar_bytes.decode("utf-8"))
    assert "computed_rubbing_note" not in plain_sidecar
    assert "탁본" not in [row["label"] for row in plain_sidecar["title_block"]]
    assert COMPUTED_RUBBING_NOTE not in plain.svg_bytes.decode("utf-8")

    # The validator holds the line: a sidecar whose title block lost the note,
    # or a figure that lost its caption, is refused.
    without_note = json.loads(bundle.sidecar_bytes.decode("utf-8"))
    without_note["title_block"] = [
        row for row in without_note["title_block"] if row["label"] != "탁본"
    ]
    with pytest.raises(DrawingSheetError, match="computed from the mesh"):
        validate_drawing_sheet_bytes(
            bundle.svg_bytes, _resigned_sidecar(without_note, bundle.svg_bytes)
        )
    without_caption = json.loads(bundle.sidecar_bytes.decode("utf-8"))
    for figure in without_caption["figures"]:
        figure.pop("caption", None)
    with pytest.raises(DrawingSheetError, match="carries no caption"):
        validate_drawing_sheet_bytes(
            bundle.svg_bytes, _resigned_sidecar(without_caption, bundle.svg_bytes)
        )


def _resigned_sidecar(sidecar: dict, svg_bytes: bytes) -> bytes:
    """Canonical bytes of an edited sidecar, its SVG digest left as it was."""

    return canonical_json_bytes(sidecar)


def test_the_sheet_scale_reduces_the_rubbing_with_everything_else() -> None:
    session, raster = _session_with_rubbing()
    full_size = raster.width_pixels * 1000.0 / raster.pixels_per_meter

    for denominator in (1.0, 2.0, 4.0):
        bundle = compose_drawing_sheet(
            session.document,
            [RUBBING_ID],
            options=_options(scale_denominator=denominator),
            rasters={RUBBING_ID: raster},
        )
        image = _images(bundle.svg_bytes)[0]
        assert float(image["width"]) == pytest.approx(
            full_size / denominator, abs=1e-6
        )


def test_a_sheet_of_line_work_alone_declares_no_image_namespace() -> None:
    document = _session().document
    bundle = compose_drawing_sheet(document, [OUTLINE_ID], options=_options())

    assert b"xmlns:xlink" not in bundle.svg_bytes
    assert b"<image" not in bundle.svg_bytes


def test_the_raster_must_be_the_one_the_record_receipted() -> None:
    session, _raster = _session_with_rubbing()
    _other_session, other_raster = _session_with_rubbing(pixels_per_mm=5)

    with pytest.raises(DrawingSheetError, match="not the one its receipt"):
        compose_drawing_sheet(
            session.document,
            [RUBBING_ID],
            options=_options(),
            rasters={RUBBING_ID: other_raster},
        )


def test_a_rubbing_without_its_pixels_is_refused_not_skipped() -> None:
    session, raster = _session_with_rubbing()

    with pytest.raises(DrawingSheetError, match="receipt, not pixels"):
        compose_drawing_sheet(session.document, [RUBBING_ID], options=_options())
    with pytest.raises(DrawingSheetError, match="does not draw"):
        compose_drawing_sheet(
            session.document,
            [OUTLINE_ID],
            options=_options(),
            rasters={RUBBING_ID: raster},
        )


def test_a_rubbing_left_stale_by_a_new_align_is_not_drawn() -> None:
    session, raster = _session_with_rubbing()
    moved = session.commit_preview(
        translation_mm=(1.0, 0.0, 0.0),
        rotation_deg=(0.0, 0.0, 0.0),
        scale=1.0,
        pivot_mm=(0.0, 0.0, 0.0),
        operator="tester",
        created_at="2026-09-03T00:00:04Z",
        revision_id="align:sheet-moved",
    )
    assert moved.document.record_freshness(RUBBING_ID) is not RecordFreshness.FRESH

    with pytest.raises(DrawingSheetError, match="only FRESH"):
        compose_drawing_sheet(
            moved.document,
            [RUBBING_ID],
            options=_options(),
            rasters={RUBBING_ID: raster},
        )
