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
from src.core.artifact_session import ArtifactSession
from src.core.artifact_vector_extractor import (
    commit_vector_computation,
    compute_artifact_cutline,
)
from src.core.artifact_vector_record import PlanarFrame
from src.core.canonical_json import canonical_json_bytes
from src.core.drawing_sheet import (
    DrawingSheetError,
    DrawingSheetOptions,
    SheetPage,
    TitleBlock,
    compose_drawing_sheet,
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
        kind="restored",
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
