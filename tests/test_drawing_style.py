from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from src.core.drawing_svg import SVGRenderError, number_token

from src.core.artifact_outline_extractor import compute_artifact_outline
from src.core.artifact_session import ArtifactSession
from src.core.artifact_vector_export import (
    ArtifactVectorExportError,
    VectorSVGOptions,
    build_vector_export,
    validate_vector_export_bytes,
)
from src.core.artifact_vector_extractor import (
    commit_vector_computation,
    compute_artifact_cutline,
)
from src.core.artifact_vector_record import PlanarFrame
from src.core.drawing_style import (
    CENTER_AXIS,
    CONDITION_CRACK,
    CONDITION_LINE_KINDS,
    LINE_KINDS,
    OUTLINE_HOLE,
    OUTLINE_VISIBLE,
    PROVISIONAL_PRESET_ID,
    SECTION_CUT,
    TECHNIQUE_GROOVE_EDGE,
    TECHNIQUE_GROOVE_TROUGH,
    TECHNIQUE_LINE_KINDS,
    DrawingStyleError,
    HatchStyle,
    LineStyle,
    available_presets,
    get_preset,
    layer_id,
    line_kind_for_condition,
    line_kind_for_record_role,
)
from src.core.mesh_import_recipe import current_mesh_import_recipe
from src.core.mesh_loader import MeshData
from src.core.source_identity import SourceFingerprint


SVG_NS = "{http://www.w3.org/2000/svg}"


def _session() -> ArtifactSession:
    mesh = MeshData(
        vertices=np.array(
            [
                [1.0, 1.0, 1.0],
                [-1.0, -1.0, 1.0],
                [-1.0, 1.0, -1.0],
                [1.0, -1.0, -1.0],
            ],
            dtype=np.float64,
        ),
        faces=np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32),
        unit="mm",
        source_identity=SourceFingerprint(
            sha256="d" * 64,
            size_bytes=64,
            mtime_ns=1,
            original_name="style.ply",
            format="ply",
        ),
        source_format="ply",
        source_import_recipe=current_mesh_import_recipe("ply"),
    )
    session = ArtifactSession.create_from_source(
        mesh,
        resolved_source_path="/source/style.ply",
        unit="mm",
        axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
        handedness="right",
        software_version="style-test",
        operator="tester",
        created_at="2026-09-03T00:00:00Z",
        document_id="artifact:style-test",
        metadata_revision_id="metadata:style-test",
        align_revision_id="align:style-test",
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
        record_id="record:style-cutline",
        created_at="2026-09-03T00:00:01Z",
        operator="tester",
    )
    outline = compute_artifact_outline(session, "top", precision_grid_mm=0.01)
    return commit_vector_computation(
        session,
        outline,
        record_id="record:style-outline",
        created_at="2026-09-03T00:00:02Z",
        operator="tester",
    )


def test_the_shipped_preset_says_it_is_provisional() -> None:
    """No preset may pass off invented weights as a published convention."""

    preset = get_preset(PROVISIONAL_PRESET_ID)

    assert preset.provisional is True
    assert preset.source_id is None
    assert set(preset.lines) == set(LINE_KINDS)


def test_the_sourced_preset_carries_the_textbook_pen_widths() -> None:
    """kcha-2013-pen/v1 follows 그림 27 of [K1]: 단면 0.6, 입면 0.4, 결실부 0.1."""

    from src.core.drawing_style import (
        CONDITION_MISSING,
        KCHA_2013_PEN_PRESET_ID,
        KCHA_2013_SOURCE_ID,
        TECHNIQUE_WOOD_GRAIN,
    )

    preset = get_preset(KCHA_2013_PEN_PRESET_ID)
    assert preset.provisional is False
    assert preset.source_id == KCHA_2013_SOURCE_ID == "K1"
    assert set(preset.lines) == set(LINE_KINDS)
    assert preset.style(SECTION_CUT).stroke_width_mm == 0.6
    assert preset.style(SECTION_CUT).hatch is True
    assert preset.style(OUTLINE_VISIBLE).stroke_width_mm == 0.4
    assert preset.style(OUTLINE_HOLE).stroke_width_mm == 0.4
    assert preset.style(CONDITION_MISSING).stroke_width_mm == 0.1
    assert preset.style(TECHNIQUE_GROOVE_EDGE).stroke_width_mm == 0.3
    assert preset.style(TECHNIQUE_GROOVE_TROUGH).stroke_width_mm == 0.1
    assert preset.style(TECHNIQUE_WOOD_GRAIN).stroke_width_mm == 0.1
    assert available_presets() == (KCHA_2013_PEN_PRESET_ID, PROVISIONAL_PRESET_ID)
    # The source is on the register, so a reader can check the numbers.
    references = (Path(__file__).resolve().parents[1] / "docs" / "REFERENCES.md").read_text(
        encoding="utf-8"
    )
    assert "`[K1]`" in references
    assert "유물 실측의 이해" in references


def test_a_preset_must_style_every_line_kind() -> None:
    with pytest.raises(DrawingStyleError, match="does not style every line kind"):
        get_preset(PROVISIONAL_PRESET_ID).__class__(
            preset_id="partial",
            lines={LINE_KINDS[0]: LineStyle(stroke_width_mm=0.3)},
            hatch=HatchStyle(spacing_mm=1.0, stroke_width_mm=0.1, angle_deg=45.0),
        )


def test_hatch_lines_must_be_thinner_than_their_spacing() -> None:
    """A hatch as thick as its pitch prints as a solid block, not a fill."""

    with pytest.raises(DrawingStyleError, match="solid block"):
        HatchStyle(spacing_mm=0.5, stroke_width_mm=0.5, angle_deg=45.0)


def test_a_preset_digest_moves_when_any_weight_moves() -> None:
    preset = get_preset(PROVISIONAL_PRESET_ID)
    altered = preset.__class__(
        preset_id=preset.preset_id,
        lines={
            **preset.lines,
            LINE_KINDS[0]: LineStyle(
                stroke_width_mm=preset.style(LINE_KINDS[0]).stroke_width_mm + 0.01,
                hatch=preset.style(LINE_KINDS[0]).hatch,
            ),
        },
        hatch=preset.hatch,
        source_id=preset.source_id,
    )

    assert altered.sha256() != preset.sha256()


def test_an_unknown_preset_or_role_is_named_in_the_error() -> None:
    with pytest.raises(DrawingStyleError, match="available presets are"):
        get_preset("no-such-preset")
    with pytest.raises(DrawingStyleError, match="known roles are"):
        line_kind_for_record_role("annotation")


def test_an_unstyled_drawing_renders_the_exact_bytes_it_always_did() -> None:
    """Presets must not rewrite drawings that were exported without one.

    Every package already written re-renders its own SVG at verification time,
    so a change to the default rendering would retire all of them at once.
    """

    document = _session().document
    default = build_vector_export(document, "record:style-cutline")
    explicit = build_vector_export(
        document,
        "record:style-cutline",
        options=VectorSVGOptions(style_preset=None),
    )

    assert default.svg_bytes == explicit.svg_bytes
    assert b"layer-" not in default.svg_bytes
    assert b"<defs>" not in default.svg_bytes
    assert "style_preset" not in json.loads(default.sidecar_bytes)["presentation"]


def test_a_styled_drawing_separates_line_kinds_into_layers() -> None:
    document = _session().document
    bundle = build_vector_export(
        document,
        "record:style-outline",
        options=VectorSVGOptions(style_preset=PROVISIONAL_PRESET_ID),
    )

    root = ET.fromstring(bundle.svg_bytes)
    body = root.find(f"{SVG_NS}g")
    assert body is not None
    assert body.attrib["id"] == "measured-vectors"

    layers = [child for child in body if child.tag == f"{SVG_NS}g"]
    assert layers, "a styled drawing must place its paths inside line-kind layers"
    preset = get_preset(PROVISIONAL_PRESET_ID)
    for layer in layers:
        kind = next(k for k in LINE_KINDS if layer_id(k) == layer.attrib["id"])
        assert layer.attrib["stroke-width"] == str(preset.style(kind).stroke_width_mm)
        assert list(layer), "an empty layer must not be emitted"

    # Layers follow the vocabulary order, so two renders cannot disagree.
    order = [
        next(k for k in LINE_KINDS if layer_id(k) == layer.attrib["id"])
        for layer in layers
    ]
    assert order == [kind for kind in LINE_KINDS if kind in order]


def test_a_cut_face_is_hatched_and_an_open_path_is_not_filled() -> None:
    document = _session().document
    bundle = build_vector_export(
        document,
        "record:style-cutline",
        options=VectorSVGOptions(style_preset=PROVISIONAL_PRESET_ID),
    )

    root = ET.fromstring(bundle.svg_bytes)
    patterns = root.findall(f"{SVG_NS}defs/{SVG_NS}pattern")
    assert [pattern.attrib["id"] for pattern in patterns] == ["hatch-section-cut"]

    filled = 0
    for path in root.iter(f"{SVG_NS}path"):
        if path.get("fill", "").startswith("url("):
            assert path.attrib["d"].endswith("Z"), "only a closed path has an interior"
            filled += 1
    assert filled, "the cut face must carry the hatch fill"


def test_a_styled_drawing_is_deterministic_and_verifies_offline() -> None:
    document = _session().document
    options = VectorSVGOptions(style_preset=PROVISIONAL_PRESET_ID)

    first = build_vector_export(document, "record:style-outline", options=options)
    second = build_vector_export(document, "record:style-outline", options=options)

    assert first.svg_bytes == second.svg_bytes
    assert first.sidecar_bytes == second.sidecar_bytes
    validate_vector_export_bytes(first.svg_bytes, first.sidecar_bytes)


def test_a_drawing_records_the_preset_digest_it_was_drawn_with() -> None:
    """A preset edited after the fact must not silently restyle the drawing."""

    document = _session().document
    bundle = build_vector_export(
        document,
        "record:style-outline",
        options=VectorSVGOptions(style_preset=PROVISIONAL_PRESET_ID),
    )
    sidecar = json.loads(bundle.sidecar_bytes)

    claim = sidecar["presentation"]["style_preset"]
    assert claim["sha256"] == get_preset(PROVISIONAL_PRESET_ID).sha256()
    assert claim["provisional"] is True

    sidecar["presentation"]["style_preset"]["sha256"] = "0" * 64
    forged = (
        json.dumps(
            sidecar,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    with pytest.raises(ArtifactVectorExportError, match="no longer matches the digest"):
        validate_vector_export_bytes(bundle.svg_bytes, forged)


def test_a_styled_sidecar_validates_against_the_shipped_schema() -> None:
    """The Python contract and the JSON Schema must agree about a styled drawing.

    An offline verifier may consult either, so a package that the exporter
    accepts and the schema rejects would be unverifiable outside this process.
    """

    import importlib

    jsonschema = importlib.import_module("jsonschema")
    referencing = importlib.import_module("referencing")

    root = Path(__file__).resolve().parents[1]
    schemas = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "schemas").glob("*.schema.json"))
    ]
    registry = referencing.Registry().with_resources(
        [
            (schema["$id"], referencing.Resource.from_contents(schema))
            for schema in schemas
            if "$id" in schema
        ]
    )
    export_schema = json.loads(
        (root / "schemas/vector_export-1.6.0.schema.json").read_text(encoding="utf-8")
    )
    validator = jsonschema.Draft202012Validator(export_schema, registry=registry)

    document = _session().document
    for preset in (None, PROVISIONAL_PRESET_ID):
        bundle = build_vector_export(
            document,
            "record:style-outline",
            options=VectorSVGOptions(style_preset=preset),
        )
        sidecar = json.loads(bundle.sidecar_bytes)
        assert ("style_preset" in sidecar["presentation"]) is (preset is not None)
        validator.validate(sidecar)


def test_a_margin_too_small_for_the_widest_line_is_refused() -> None:
    """A layer's own weight replaces the single stroke, so the margin must clear it."""

    preset = get_preset(PROVISIONAL_PRESET_ID)
    widest = max(style.stroke_width_mm for style in preset.lines.values())
    default_stroke = VectorSVGOptions().stroke_width_mm
    # Wide enough for the unstyled drawing, too narrow for the preset's heaviest
    # line: without a preset this margin is fine, with one it would clip.
    margin = (default_stroke + widest) / 4.0
    assert default_stroke / 2.0 <= margin < widest / 2.0

    VectorSVGOptions(margin_mm=margin)

    with pytest.raises(ArtifactVectorExportError, match="widest stroke"):
        VectorSVGOptions(margin_mm=margin, style_preset=PROVISIONAL_PRESET_ID)


def test_every_preset_and_layer_id_is_stable_and_distinct() -> None:
    assert available_presets() == tuple(sorted(available_presets()))
    ids = [layer_id(kind) for kind in LINE_KINDS]
    assert len(set(ids)) == len(ids)
    assert all(identifier.startswith("layer-") for identifier in ids)


def test_the_drawing_conventions_document_ships_with_the_vocabulary() -> None:
    """Every line kind must be documented where a sourced value can be filled in."""

    text = (
        Path(__file__).resolve().parents[1] / "docs/DRAWING_CONVENTIONS.md"
    ).read_text(encoding="utf-8")

    for kind in LINE_KINDS:
        assert f"`{kind}`" in text


def test_the_condition_vocabulary_matches_the_record_layer_exactly() -> None:
    """`CONDITION_LINE_KINDS` repeats the record vocabulary as literals.

    That duplication is deliberate - presentation must not import the record
    layer - so this is the check that keeps the two from drifting apart.
    """

    from src.core.artifact_condition_annotation import CONDITION_KINDS

    assert tuple(sorted(CONDITION_LINE_KINDS)) == CONDITION_KINDS
    for kind in CONDITION_KINDS:
        assert line_kind_for_condition(kind) in LINE_KINDS

    with pytest.raises(DrawingStyleError, match="has no drawing style"):
        line_kind_for_condition("chipped")


def test_the_technique_vocabulary_matches_the_record_layer_exactly() -> None:
    """`TECHNIQUE_LINE_KINDS` repeats the record vocabulary as literals."""

    from src.core.artifact_technique_annotation import TECHNIQUE_KINDS
    from src.core.drawing_style import (
        LINE_KINDS,
        TECHNIQUE_LINE_KINDS,
        line_kind_for_technique,
    )

    assert tuple(sorted(TECHNIQUE_LINE_KINDS)) == TECHNIQUE_KINDS
    for kind in TECHNIQUE_KINDS:
        assert line_kind_for_technique(kind) in LINE_KINDS
        assert line_kind_for_technique(kind).startswith("technique_")
    with pytest.raises(DrawingStyleError, match="slipping"):
        line_kind_for_technique("slipping")


def test_technique_marks_are_drawn_over_the_outline_and_under_condition() -> None:
    from src.core.drawing_style import (
        CONDITION_MISSING,
        LINE_KINDS,
        OUTLINE_VISIBLE,
        TECHNIQUE_LINE_KINDS,
        TECHNIQUE_GROOVE_TROUGH,
    )

    for line_kind in list(TECHNIQUE_LINE_KINDS.values()) + [TECHNIQUE_GROOVE_TROUGH]:
        assert LINE_KINDS.index(OUTLINE_VISIBLE) < LINE_KINDS.index(line_kind)
        assert LINE_KINDS.index(line_kind) < LINE_KINDS.index(CONDITION_MISSING)


def test_condition_layers_are_drawn_over_the_shape_and_under_the_axis() -> None:
    condition_kinds = [kind for kind in LINE_KINDS if kind.startswith("condition_")]

    assert LINE_KINDS.index(OUTLINE_HOLE) < LINE_KINDS.index(condition_kinds[0])
    assert LINE_KINDS.index(condition_kinds[-1]) < LINE_KINDS.index(CENTER_AXIS)
    # A crack is a line on the object, so nothing in the group covers it.
    assert condition_kinds[-1] == CONDITION_CRACK


def test_every_condition_kind_is_visually_distinguishable_in_the_preset() -> None:
    preset = get_preset(PROVISIONAL_PRESET_ID)
    signatures = {
        kind: (
            preset.style(kind).stroke_width_mm,
            preset.style(kind).dash_pattern_mm,
        )
        for kind in LINE_KINDS
    }

    # Some kinds deliberately share a pen and are told apart on paper by
    # something other than it.
    deliberate = (
        # An outline and the hole inside it are the same line; a reader tells
        # them apart by where they are, not by weight.
        (OUTLINE_VISIBLE, OUTLINE_HOLE),
        # A groove's bottom and its two edges are drawn with one pen, the way
        # a drafter draws them.  The bottom is a 간선: what distinguishes it is
        # that its geometry is broken, which no style signature can carry.
        (TECHNIQUE_GROOVE_EDGE, TECHNIQUE_GROOVE_TROUGH),
    )
    for first, second in deliberate:
        assert signatures[first] == signatures[second]
    # The five technique marks all take one fine solid pen: each is drawn
    # as its own strokes (ovals, a seam, clusters, lines), never as a coded
    # boundary, so the pen carries nothing and must not pretend to.
    mark_kinds = [kind for kind in LINE_KINDS if kind in TECHNIQUE_LINE_KINDS.values()]
    assert len({signatures[kind] for kind in mark_kinds}) == 1
    assert all(preset.style(kind).dash_pattern_mm == () for kind in mark_kinds)
    others = {kind: signatures[kind] for kind in LINE_KINDS if kind not in mark_kinds}
    assert len(set(others.values())) == len(others) - len(deliberate), others
    # And the marks' pen is not another kind's pen.
    assert signatures[mark_kinds[0]] not in others.values()


def test_a_measured_coordinate_is_not_refused_for_its_thirteenth_decimal() -> None:
    """A cut line through a measured pot produced this y in millimetres.

    Rounded to the twelve decimals every ArchMeshRubbing SVG uses, it differs
    from the double by 5.009e-13 mm - a hair over half the last decimal, and
    so a tolerance set at exactly that boundary rejected it and refused the
    whole sheet.  Half a picometre is not information; the token must carry it.
    """

    value = 17.4219396566915
    token = number_token(value, field_name="path.y")

    assert token == "17.421939656691"
    assert abs(float(token) - value) < 1e-12
    # The guard still exists: it separates a rounded value from an unwritable
    # one, so a value it cannot express at all is still refused.
    with pytest.raises(SVGRenderError, match="finite number"):
        number_token(float("inf"), field_name="path.y")
