"""채색 문양을 이미지로: a painted mark cut out of the colour map and pasted
on the drawing at its own place, or beneath it as a detail, in the
drawing's own tone."""

from __future__ import annotations

import base64
import json
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from PIL import Image

from src.core.artifact_outline_extractor import compute_artifact_outline
from src.core.artifact_paint_cutout import (
    PAINT_CUTOUT_PIXEL_FORMAT_COLOUR,
    PAINT_CUTOUT_RECORD_TYPE,
    PAINT_CUTOUT_TONE_COLOUR,
    ArtifactPaintCutoutError,
    PaintCutoutRaster,
    commit_paint_cutout,
    compute_paint_cutout,
    extract_paint_cutout,
    paint_cutout_receipt_from_record,
    paint_cutout_tone,
    require_paint_cutout_raster,
    validate_paint_cutout_recipe,
)
from src.core.artifact_record_validation import validate_known_records
from src.core.artifact_texture_paint import read_colour_map
from src.core.artifact_texture_relief import read_obj_texture_atlas
from src.core.artifact_vector_extractor import commit_vector_computation, compute_artifact_cutline
from src.core.artifact_vector_record import PlanarFrame
from src.core.drawing_sheet import (
    DrawingSheetError,
    DrawingSheetOptions,
    SheetPage,
    TitleBlock,
    compose_drawing_sheet,
    validate_drawing_sheet_bytes,
)
from src.core.project_file import load_artifact_project, save_artifact_project
from synthetic_vessel import FLOOR_MM, HEIGHT_MM, positioned_vessel_session
from test_texture_relief import MAP_SIDE, _write_textured_obj

STAMP = "2026-09-06T00:00:00Z"
BAND_LOW, BAND_HIGH = 40.0, 44.0
SVG_NS = "{http://www.w3.org/2000/svg}"


@pytest.fixture(scope="module")
def painted():
    """The synthetic vessel with a gold band painted round it, positioned,
    with its front outline and its section, and the colour map at hand."""

    session, vertices, faces = positioned_vessel_session(segments=96, rings=40, document_id="artifact:cutout")
    directory = tempfile.mkdtemp()
    obj_path = Path(directory) / "vessel.obj"
    _write_textured_obj(obj_path, vertices, np.asarray(faces, dtype=np.int64))
    atlas = read_obj_texture_atlas(obj_path)
    rgb = np.full((MAP_SIDE, MAP_SIDE, 3), 245, dtype=np.uint8)
    z = HEIGHT_MM * (1.0 - (np.arange(MAP_SIDE) + 0.5) / MAP_SIDE)
    rgb[(z >= BAND_LOW) & (z <= BAND_HIGH)] = np.array([200, 160, 60], dtype=np.uint8)
    map_path = Path(directory) / "vessel_bc.png"
    Image.fromarray(rgb, mode="RGB").save(map_path)
    colour_map = read_colour_map(map_path)
    session = commit_vector_computation(
        session, compute_artifact_outline(session, "front", precision_grid_mm=0.5),
        record_id="record:front", created_at="2026-09-06T00:01:00Z", operator="tester",
    )
    session = commit_vector_computation(
        session,
        compute_artifact_cutline(session, PlanarFrame(origin_world_mm=(0.0, 0.0, 0.0), u_axis_world=(1.0, 0.0, 0.0), v_axis_world=(0.0, 0.0, 1.0), normal_world=(0.0, -1.0, 0.0))),
        record_id="record:section", created_at="2026-09-06T00:02:00Z", operator="tester",
    )
    return session, atlas, colour_map


def _options(**overrides):
    settings = dict(title_block=TitleBlock(artifact_label="금띠 시험 호"), page=SheetPage(size="A4", orientation="portrait"), scale_denominator=2.0)
    settings.update(overrides)
    return DrawingSheetOptions(**settings)


def test_the_painted_band_is_cut_out_where_it_is_and_the_record_reopens(painted) -> None:
    """Seen from the front the gold band is a strip across the wall at its
    height: the cutout's paper holds it with its margin, its ink is the
    paint's coverage, a window cuts out one part of it only, and the record
    keeps a receipt the pixels must match."""

    session, atlas, colour_map = painted
    computation = compute_paint_cutout(session, atlas, colour_map, view="front", pixels_per_mm=5)
    raster = computation.raster
    left, bottom, right, top = raster.rectangle_mm
    assert bottom < BAND_LOW - FLOOR_MM < BAND_HIGH - FLOOR_MM < top
    assert top - bottom < BAND_HIGH - BAND_LOW + 3.0
    assert right - left > 60.0
    assert raster.pixels[..., 0].max() == 0 and raster.pixels[..., 1].max() == 255
    assert computation.qc["inked_pixel_count"] > 1000 and computation.qc["view"] == "front"
    assert validate_paint_cutout_recipe(computation.recipe) == computation.recipe
    assert computation.recipe["window"] is None
    # A window keeps the part of the band left of the axis only.
    windowed = compute_paint_cutout(session, atlas, colour_map, view="front", pixels_per_mm=5, window_mm=(-50.0, 0.0, -5.0, 80.0))
    wl, _wb, wr, _wt = windowed.raster.rectangle_mm
    assert wr < 0.0 and wl < -30.0
    assert windowed.recipe["window"] == {"bottom_um": 0, "left_um": -50_000, "right_um": -5_000, "top_um": 80_000}
    with pytest.raises(ArtifactPaintCutoutError, match="nothing is painted"):
        compute_paint_cutout(session, atlas, colour_map, view="front", window_mm=(-50.0, 0.0, -5.0, 10.0))
    with pytest.raises(ArtifactPaintCutoutError, match="full_thousandths"):
        compute_paint_cutout(session, atlas, colour_map, view="front", threshold_thousandths=300, full_thousandths=200)
    forged = json.loads(json.dumps(computation.recipe))
    forged["texture_paint"]["band_um"] = 800
    with pytest.raises(ArtifactPaintCutoutError, match="band_um must be 0"):
        validate_paint_cutout_recipe(forged)
    # Committed, saved, reopened: the receipt is the raster's, and a raster
    # that is not the one the receipt proves is refused.
    session = commit_paint_cutout(session, computation, record_id="record:cutout", created_at=STAMP, operator="tester")
    record = session.document.record_index["record:cutout"]
    assert record.type == PAINT_CUTOUT_RECORD_TYPE
    receipt = paint_cutout_receipt_from_record(record)
    assert receipt["raster_sha256"] == raster.raster_sha256 and receipt["view"] == "front"
    assert require_paint_cutout_raster(record, raster) is raster
    other = PaintCutoutRaster(pixels=raster.pixels, pixels_per_meter=raster.pixels_per_meter, left_um=raster.left_um + 1, bottom_um=raster.bottom_um, view="front")
    with pytest.raises(ArtifactPaintCutoutError, match="not the one record"):
        require_paint_cutout_raster(record, other)
    path = Path(tempfile.mkdtemp()) / "cutout.amr"
    save_artifact_project(path, session.document)
    reopened = load_artifact_project(path)
    validate_known_records(reopened)
    assert paint_cutout_receipt_from_record(reopened.record_index["record:cutout"]) == receipt
    vertices = np.asarray(session.materialize().mesh.vertices)
    faces = np.asarray(session.materialize().mesh.faces)
    again, _qc = extract_paint_cutout(vertices, faces, atlas, colour_map, record.recipe)
    assert again.raster_sha256 == raster.raster_sha256


def test_a_cutout_is_pasted_in_place_or_below_and_never_across_the_fold(painted) -> None:
    """In place the image sits at the band's own rectangle on the front
    elevation, under the line work, in the sheet's ink tone; below it hangs
    centred under the figure; on a mirrored figure a cutout that crosses the
    fold is refused unless the fold steps round it, and the sidecar says
    what was pasted where."""

    session, atlas, colour_map = painted
    left_part = compute_paint_cutout(session, atlas, colour_map, view="front", pixels_per_mm=5, window_mm=(-50.0, 0.0, -5.0, 80.0))
    session = commit_paint_cutout(session, left_part, record_id="record:cutout:left", created_at=STAMP, operator="tester")
    whole = compute_paint_cutout(session, atlas, colour_map, view="front", pixels_per_mm=5)
    session = commit_paint_cutout(session, whole, record_id="record:cutout:whole", created_at="2026-09-06T00:00:01Z", operator="tester")
    middle = compute_paint_cutout(session, atlas, colour_map, view="front", pixels_per_mm=5, window_mm=(-20.0, 0.0, 20.0, 80.0))
    session = commit_paint_cutout(session, middle, record_id="record:cutout:middle", created_at="2026-09-06T00:00:02Z", operator="tester")
    rasters = {"record:cutout:left": left_part.raster, "record:cutout:whole": whole.raster}

    bundle = compose_drawing_sheet(
        session.document, ["record:front"],
        options=_options(paint_cutouts=(("record:cutout:left", "record:front", "in_place"), ("record:cutout:whole", "record:front", "below")), paint_cutout_ink_percent=60),
        rasters=rasters,
    )
    validate_drawing_sheet_bytes(bundle.svg_bytes, bundle.sidecar_bytes)
    root = ET.fromstring(bundle.svg_bytes)
    images = [el for el in root.iter(f"{SVG_NS}image") if el.attrib.get("id", "").startswith("paint-cutout-")]
    assert [el.attrib["data-placement"] for el in images] == ["in_place", "below"]
    assert all(el.attrib["opacity"] == "0.6" for el in images)
    assert all(el.attrib["xlink:href" if "xlink:href" in el.attrib else "{http://www.w3.org/1999/xlink}href"].startswith("data:image/png;base64,") for el in images)
    # The image comes before the line layers in its figure, so lines draw over it.
    figure = next(el for el in root.iter(f"{SVG_NS}g") if el.attrib.get("id", "").startswith("figure-"))
    children = list(figure)
    assert children[0].tag == f"{SVG_NS}image"
    # Below hangs under the figure: its paper y is beneath the outline's.
    below_y = float(images[1].attrib["y"])
    in_place_y = float(images[0].attrib["y"])
    assert below_y > in_place_y
    sidecar = json.loads(bundle.sidecar_bytes.decode("utf-8"))
    drawn = sidecar["paint_cutouts"]["drawn"]
    assert [(entry["record_id"], entry["placement"], entry["ink_percent"]) for entry in drawn] == [
        ("record:cutout:left", "in_place", "60"), ("record:cutout:whole", "below", "60"),
    ]
    assert {entry["record_id"] for entry in sidecar["paint_cutouts"]["records"]} == {"record:cutout:left", "record:cutout:whole"}
    assert "paint_cutouts" not in json.loads(compose_drawing_sheet(session.document, ["record:front"], options=_options()).sidecar_bytes.decode("utf-8"))

    mirrored = _options(mirror_sections=(("record:front", "record:section"),))
    # The left part lies on the elevation's side: pasted.
    compose_drawing_sheet(session.document, ["record:front"], options=_options(mirror_sections=(("record:front", "record:section"),), paint_cutouts=(("record:cutout:left", "record:front", "in_place"),)), rasters={"record:cutout:left": left_part.raster})
    # The whole band crosses the fold: refused, unless the fold steps round it.
    with pytest.raises(DrawingSheetError, match="crosses the fold"):
        compose_drawing_sheet(session.document, ["record:front"], options=_options(mirror_sections=(("record:front", "record:section"),), paint_cutouts=(("record:cutout:whole", "record:front", "in_place"),)), rasters={"record:cutout:whole": whole.raster})
    # A cutout across the axis but clear of the wall's cut: the fold steps round it.
    wl, wb, wr, wt = middle.raster.rectangle_mm
    stepped = compose_drawing_sheet(
        session.document, ["record:front"],
        options=_options(mirror_sections=(("record:front", "record:section"),), mirror_jogs=(("record:front", wb - 1.0, wt + 1.0, wr + 1.0),), paint_cutouts=(("record:cutout:middle", "record:front", "in_place"),)),
        rasters={"record:cutout:middle": middle.raster},
    )
    validate_drawing_sheet_bytes(stepped.svg_bytes, stepped.sidecar_bytes)
    assert mirrored is not None
    # The raster must be the record's, be given, and the view must match the figure's plane.
    with pytest.raises(DrawingSheetError, match="needs its raster"):
        compose_drawing_sheet(session.document, ["record:front"], options=_options(paint_cutouts=(("record:cutout:left", "record:front", "in_place"),)))
    with pytest.raises(DrawingSheetError, match="not the one record"):
        compose_drawing_sheet(session.document, ["record:front"], options=_options(paint_cutouts=(("record:cutout:left", "record:front", "in_place"),)), rasters={"record:cutout:left": whole.raster})
    with pytest.raises(DrawingSheetError, match="placement must be one of"):
        _options(paint_cutouts=(("record:cutout:left", "record:front", "beside"),))
    with pytest.raises(DrawingSheetError, match="does not draw for"):
        compose_drawing_sheet(session.document, ["record:front"], options=_options(paint_cutouts=(("record:cutout:left", "record:absent", "in_place"),)), rasters={"record:cutout:left": left_part.raster})


def test_a_cutout_can_keep_the_paint_s_own_colour(painted) -> None:
    """Asked for the paint's colour rather than ink, the cutout is the
    band's own RGB with the coverage as its alpha - the wand's selection
    lifted whole - pasted at full strength whatever the sheet's ink
    percent, as an RGBA image; the receipt names the format, and ink
    pixels are not the colour record's."""

    session, atlas, colour_map = painted
    colour = compute_paint_cutout(session, atlas, colour_map, view="front", pixels_per_mm=5, tone=PAINT_CUTOUT_TONE_COLOUR)
    raster = colour.raster
    assert raster.channels == 4 and raster.pixel_format == PAINT_CUTOUT_PIXEL_FORMAT_COLOUR
    solid = raster.pixels[..., 3] == 255
    assert solid.any()
    assert (raster.pixels[solid][:, :3] == np.array([200, 160, 60], dtype=np.uint8)).all(), "the band's colour as the map has it"
    assert (raster.pixels[raster.pixels[..., 3] == 0][:, :3] == 0).all(), "clear pixels carry no colour"
    ink = compute_paint_cutout(session, atlas, colour_map, view="front", pixels_per_mm=5)
    assert ink.raster.channels == 2 and ink.raster.rectangle_mm == raster.rectangle_mm
    assert np.array_equal(ink.raster.alpha, raster.alpha), "the same coverage, whichever tone"
    assert colour.recipe["ink_policy"]["tone"] == PAINT_CUTOUT_TONE_COLOUR and paint_cutout_tone(colour.recipe) == PAINT_CUTOUT_TONE_COLOUR
    with pytest.raises(ArtifactPaintCutoutError, match="tone must be one of"):
        compute_paint_cutout(session, atlas, colour_map, view="front", tone="sepia/v1")

    session = commit_paint_cutout(session, colour, record_id="record:cutout:colour", created_at=STAMP, operator="tester")
    record = session.document.record_index["record:cutout:colour"]
    receipt = paint_cutout_receipt_from_record(record)
    assert receipt["pixel_format"] == PAINT_CUTOUT_PIXEL_FORMAT_COLOUR
    assert receipt["raw_pixel_byte_length"] == 4 * raster.width_pixels * raster.height_pixels
    with pytest.raises(ArtifactPaintCutoutError, match="not the one record"):
        require_paint_cutout_raster(record, ink.raster)
    with pytest.raises(ArtifactPaintCutoutError, match="HxWx2 .* or HxWx4"):
        PaintCutoutRaster(pixels=raster.pixels[..., :3].copy(), pixels_per_meter=raster.pixels_per_meter, left_um=0, bottom_um=0, view="front")

    bundle = compose_drawing_sheet(
        session.document, ["record:front"],
        options=_options(paint_cutouts=(("record:cutout:colour", "record:front", "in_place"),), paint_cutout_ink_percent=60),
        rasters={"record:cutout:colour": raster},
    )
    validate_drawing_sheet_bytes(bundle.svg_bytes, bundle.sidecar_bytes)
    root = ET.fromstring(bundle.svg_bytes)
    image = next(el for el in root.iter(f"{SVG_NS}image") if el.attrib.get("id", "").startswith("paint-cutout-"))
    assert float(image.attrib["opacity"]) == 1.0, "the paint's own colour is pasted whole"
    href = image.attrib["xlink:href" if "xlink:href" in image.attrib else "{http://www.w3.org/1999/xlink}href"]
    png = base64.b64decode(href.split(",", 1)[1])
    assert png[25] == 6, "an RGBA PNG (colour type 6)"
    drawn = json.loads(bundle.sidecar_bytes.decode("utf-8"))["paint_cutouts"]["drawn"]
    assert [(entry["record_id"], entry["tone"], entry["ink_percent"]) for entry in drawn] == [
        ("record:cutout:colour", PAINT_CUTOUT_TONE_COLOUR, "100"),
    ]
