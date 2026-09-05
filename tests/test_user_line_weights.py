"""The drafter's own line weights, in points or millimetres, on the record.

A report style states weights - 0.5 pt for the outline, 0.3 pt for the
section - and the program's job is to draw them, not to argue.  A user
preset takes those numbers, keeps the shipped preset's dashes and hatch so
the kinds stay distinguishable, is named by its content, and travels with
every drawing made under it in full: no registry anywhere else holds it.
"""

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path

import pytest

from src.core.artifact_vector_export import (
    ArtifactVectorExportError,
    VectorSVGOptions,
    build_vector_export,
    validate_vector_export_bytes,
)
from src.core.canonical_json import canonical_json_bytes
from src.core.drawing_sheet import (
    DrawingSheetError,
    DrawingSheetOptions,
    TitleBlock,
    compose_drawing_sheet,
    validate_drawing_sheet_bytes,
)
from src.core.drawing_style import (
    LINE_KINDS,
    LINE_KIND_LABELS_KO,
    OUTLINE_VISIBLE,
    POINT_MM,
    PROVISIONAL_PRESET_ID,
    SECTION_CUT,
    USER_PRESET_ID_PREFIX,
    DrawingStyleError,
    DrawingStylePreset,
    get_preset,
    mm_to_pt,
    preset_claim,
    preset_from_claim,
    pt_to_mm,
    resolve_preset,
    user_preset,
)
from test_drawing_sheet import OUTLINE_ID, _options, _session

ROOT = Path(__file__).resolve().parents[1]


def test_points_and_millimetres_convert_both_ways() -> None:
    assert pt_to_mm(1.0) == pytest.approx(0.352778, abs=1e-6)
    assert mm_to_pt(pt_to_mm(0.5)) == pytest.approx(0.5)
    assert POINT_MM == pytest.approx(25.4 / 72.0)


def test_every_line_kind_has_a_name_the_drafter_uses() -> None:
    assert set(LINE_KIND_LABELS_KO) == set(LINE_KINDS)
    assert LINE_KIND_LABELS_KO[OUTLINE_VISIBLE].startswith("외선")
    assert LINE_KIND_LABELS_KO[SECTION_CUT] == "단면선"


def test_a_user_preset_is_named_by_its_content_and_keeps_the_base_dashes() -> None:
    base = get_preset(PROVISIONAL_PRESET_ID)
    weights = {OUTLINE_VISIBLE: pt_to_mm(0.5), SECTION_CUT: pt_to_mm(0.8)}
    first = user_preset(weights)
    again = user_preset(dict(weights))

    assert first.preset_id.startswith(USER_PRESET_ID_PREFIX)
    assert first.preset_id == again.preset_id
    assert first.sha256() == again.sha256()
    assert first.is_user and first.provisional
    assert first.style(OUTLINE_VISIBLE).stroke_width_mm == pytest.approx(pt_to_mm(0.5))
    # Kinds left out keep the base weight; every kind keeps the base dashes.
    for kind in LINE_KINDS:
        assert first.style(kind).dash_pattern_mm == base.style(kind).dash_pattern_mm
        assert first.style(kind).hatch == base.style(kind).hatch
        if kind not in weights:
            assert first.style(kind).stroke_width_mm == base.style(kind).stroke_width_mm
    assert first.hatch == base.hatch
    # A different weight is a different preset.
    other = user_preset({OUTLINE_VISIBLE: pt_to_mm(0.6)})
    assert other.preset_id != first.preset_id


def test_a_user_preset_round_trips_through_its_dict_and_claim() -> None:
    preset = user_preset({OUTLINE_VISIBLE: 0.2})
    rebuilt = DrawingStylePreset.from_dict(preset.to_dict())
    assert rebuilt == preset
    assert resolve_preset(preset.to_dict()) == preset
    assert resolve_preset(preset) is preset
    assert resolve_preset(PROVISIONAL_PRESET_ID) == get_preset(PROVISIONAL_PRESET_ID)

    claim = preset_claim(preset)
    assert claim["definition"] == preset.to_dict()
    assert preset_from_claim(claim) == preset
    # A registered preset is claimed without a definition, and must be.
    registered_claim = preset_claim(get_preset(PROVISIONAL_PRESET_ID))
    assert "definition" not in registered_claim
    assert preset_from_claim(registered_claim) == get_preset(PROVISIONAL_PRESET_ID)


def test_a_user_preset_cannot_lie_about_itself() -> None:
    preset = user_preset({OUTLINE_VISIBLE: 0.2})
    with pytest.raises(DrawingStyleError, match="named by its own content"):
        DrawingStylePreset(
            preset_id="user/000000000000",
            lines=preset.lines,
            hatch=preset.hatch,
            source_id="user",
        )
    with pytest.raises(DrawingStyleError, match="prefix"):
        DrawingStylePreset(
            preset_id="mine", lines=preset.lines, hatch=preset.hatch, source_id="user"
        )
    with pytest.raises(DrawingStyleError, match="unknown line kinds"):
        user_preset({"decoration": 0.2})
    with pytest.raises(DrawingStyleError, match="at most 10"):
        user_preset({OUTLINE_VISIBLE: 12.0})
    # A claim naming a user preset without its definition cannot be checked.
    claim = preset_claim(preset)
    claim.pop("definition")
    with pytest.raises(DrawingStyleError, match="without its definition"):
        preset_from_claim(claim)
    # A definition edited after the fact no longer matches the digest.
    edited = preset_claim(preset)
    edited["definition"]["lines"][OUTLINE_VISIBLE]["stroke_width_mm"] = 0.3
    with pytest.raises(DrawingStyleError):
        preset_from_claim(edited)
    # A registered id claimed with a mismatching definition is refused.
    with pytest.raises(DrawingStyleError, match="registered with different values"):
        registered = get_preset(PROVISIONAL_PRESET_ID).to_dict()
        registered["lines"][OUTLINE_VISIBLE]["stroke_width_mm"] = 0.9
        resolve_preset(registered)


def test_a_sheet_drawn_with_user_weights_carries_them_and_verifies() -> None:
    session = _session()
    preset = user_preset({OUTLINE_VISIBLE: pt_to_mm(0.5)})
    bundle = compose_drawing_sheet(
        session.document, [OUTLINE_ID], options=_options(style_preset=preset)
    )
    validate_drawing_sheet_bytes(bundle.svg_bytes, bundle.sidecar_bytes)
    sidecar = json.loads(bundle.sidecar_bytes.decode("utf-8"))
    claim = sidecar["style_preset"]
    assert claim["preset_id"] == preset.preset_id
    assert claim["provisional"] is True
    assert claim["source_id"] == "user"
    assert claim["definition"]["lines"][OUTLINE_VISIBLE]["stroke_width_mm"] == (
        pytest.approx(pt_to_mm(0.5))
    )
    import xml.etree.ElementTree as ET

    root = ET.fromstring(bundle.svg_bytes)
    layer = next(
        element
        for element in root.iter("{http://www.w3.org/2000/svg}g")
        if element.attrib.get("id") == "layer-outline-visible"
    )
    assert float(layer.attrib["stroke-width"]) == pytest.approx(pt_to_mm(0.5), abs=1e-6)

    # The same sheet with the shipped preset keeps the exact bytes it had.
    shipped = compose_drawing_sheet(session.document, [OUTLINE_ID], options=_options())
    assert "definition" not in json.loads(shipped.sidecar_bytes)["style_preset"]

    tampered = copy.deepcopy(sidecar)
    tampered["style_preset"]["definition"]["lines"][OUTLINE_VISIBLE]["stroke_width_mm"] = 0.4
    with pytest.raises(DrawingSheetError):
        validate_drawing_sheet_bytes(bundle.svg_bytes, canonical_json_bytes(tampered))
    with pytest.raises(DrawingSheetError):
        DrawingSheetOptions(title_block=TitleBlock(artifact_label="A"), style_preset="user/abc")


def test_a_vector_export_with_user_weights_needs_the_1_3_sidecar_and_validates() -> None:
    jsonschema = importlib.import_module("jsonschema")
    referencing = importlib.import_module("referencing")
    document = _session().document
    preset = user_preset({OUTLINE_VISIBLE: pt_to_mm(0.5), SECTION_CUT: pt_to_mm(0.8)})
    options = VectorSVGOptions(style_preset=preset)
    assert options.style_preset == preset

    bundle = build_vector_export(document, OUTLINE_ID, options=options)
    sidecar = json.loads(bundle.sidecar_bytes)
    assert sidecar["schema_version"] == "1.5.0"
    claim = sidecar["presentation"]["style_preset"]
    assert claim["definition"]["preset_id"] == preset.preset_id
    validate_vector_export_bytes(bundle.svg_bytes, bundle.sidecar_bytes)

    def load(name: str) -> dict:
        return json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8"))

    registry = referencing.Registry()
    for name in (
        "vector_payload-1.0.0.schema.json",
        "vector_export-1.0.0.schema.json",
        "mesh_admission_receipt-1.0.0.schema.json",
        "mesh_import_recipe-1.0.0.schema.json",
        "mesh_import_recipe-2.0.0.schema.json",
    ):
        schema = load(name)
        registry = registry.with_resource(
            schema["$id"], referencing.Resource.from_contents(schema)
        )
    validator = jsonschema.Draft202012Validator(
        load("vector_export-1.5.0.schema.json"), registry=registry
    )
    assert list(validator.iter_errors(sidecar)) == []
    # 1.2.0 has no room for the definition, in the schema or in the runtime.
    previous = jsonschema.Draft202012Validator(
        load("vector_export-1.2.0.schema.json"), registry=registry
    )
    shaped = copy.deepcopy(sidecar)
    shaped["schema_version"] = "1.2.0"
    assert list(previous.iter_errors(shaped))
    with pytest.raises(ArtifactVectorExportError, match="user drawing style preset"):
        from src.core.artifact_vector_export import _validated_style_preset

        _validated_style_preset(claim, schema_version="1.2.0")

    # The drawing is what the weights say, layer by layer.
    import xml.etree.ElementTree as ET

    root = ET.fromstring(bundle.svg_bytes)
    layer = next(
        element
        for element in root.iter("{http://www.w3.org/2000/svg}g")
        if element.attrib.get("id") == "layer-outline-visible"
    )
    assert float(layer.attrib["stroke-width"]) == pytest.approx(
        preset.style(OUTLINE_VISIBLE).stroke_width_mm, abs=1e-6
    )
