"""양각 음영: a motif carved in relief is not drawn in lines.  Its shade is
read off the wall as a drafter sees it from the side, lit from the upper
left, and the sheet shows its bulk by stippling - dots denser where the
surface turns from the light."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from src.core.artifact_outline_extractor import compute_artifact_outline
from src.core.artifact_record_validation import validate_known_records
from src.core.artifact_relief_shade import (
    RELIEF_SHADE_PAYLOAD_EXTENSION_KEY,
    RELIEF_SHADE_RECORD_TYPE,
    ArtifactReliefShadeError,
    ReliefShadeRaster,
    commit_relief_shade,
    compute_relief_shade,
    relief_shade_receipt_from_record,
    relief_shade_recipe,
    require_relief_shade_raster,
    validate_relief_shade_recipe,
)
from src.core.artifact_vector_extractor import commit_vector_computation, compute_artifact_cutline
from src.core.artifact_vector_record import PlanarFrame
from src.core.drawing_sheet import (
    DrawingSheetError,
    DrawingSheetOptions,
    SheetPage,
    TitleBlock,
    _cell_uniforms,
    compose_drawing_sheet,
    validate_drawing_sheet_bytes,
)
from synthetic_vessel import positioned_vessel_session

STAMP = "2026-09-07T00:00:00Z"
#: One petal in relief on the front of the wall: 1.5 mm proud at its
#: middle, falling smoothly to the wall over 20 degrees either way and 15 mm
#: up and down.  The front view looks along -y, so the front is at -90
#: degrees.
PETAL_ANGLE = -math.pi / 2.0
PETAL_Z = 50.0


def _petal(angle_rad: float, z_mm: float) -> float:
    across = math.remainder(angle_rad - PETAL_ANGLE, 2.0 * math.pi) / math.radians(20.0)
    up = (z_mm - PETAL_Z) / 15.0
    if abs(across) >= 1.0 or abs(up) >= 1.0:
        return 0.0
    return 1.5 * math.cos(0.5 * math.pi * across) ** 2 * math.cos(0.5 * math.pi * up) ** 2


@pytest.fixture(scope="module")
def petalled():
    session, _vertices, _faces = positioned_vessel_session(segments=96, rings=45, relief=_petal, document_id="artifact:petalled")
    session = commit_vector_computation(
        session, compute_artifact_outline(session, "front", precision_grid_mm=0.5),
        record_id="record:front", created_at="2026-09-07T00:01:00Z", operator="tester",
    )
    session = commit_vector_computation(
        session, compute_artifact_outline(session, "top", precision_grid_mm=0.5),
        record_id="record:top", created_at="2026-09-07T00:01:30Z", operator="tester",
    )
    return commit_vector_computation(
        session,
        compute_artifact_cutline(session, PlanarFrame(origin_world_mm=(0.0, 0.0, 0.0), u_axis_world=(1.0, 0.0, 0.0), v_axis_world=(0.0, 0.0, 1.0), normal_world=(0.0, -1.0, 0.0))),
        record_id="record:section", created_at="2026-09-07T00:02:00Z", operator="tester",
    )


def test_the_shade_falls_on_the_side_of_the_petal_that_turns_from_the_light(petalled) -> None:
    """Lit from the upper left, the petal's right flank and its lower slope
    are in shade and its left flank is not; the smooth wall around it is
    not shaded at all.  The recipe rebuilds to its own bytes, a plan view
    has no base to read the relief against, and a window that sees no
    relief is refused rather than shading nothing."""

    computation = compute_relief_shade(petalled, view="front")
    raster = computation.raster
    assert validate_relief_shade_recipe(computation.recipe) == computation.recipe
    left, bottom, right, top = raster.rectangle_mm
    # The shade lies about the petal, not the whole wall: a margin round it.
    assert -30.0 < left < -8.0 and 8.0 < right < 30.0, raster.rectangle_mm
    # The floor circle stands at canonical height zero, so the petal's middle is at 40.
    assert 20.0 < bottom < 35.0 and 48.0 < top < 62.0, raster.rectangle_mm
    darkness = raster.darkness.astype(np.float64)
    columns = left + (np.arange(raster.width_pixels) + 0.5) * 1000.0 / raster.pixels_per_meter
    right_flank = darkness[:, columns > 2.0].mean()
    left_flank = darkness[:, columns < -2.0].mean()
    assert right_flank > 4.0 * left_flank, (right_flank, left_flank)
    # The petal stands a millimetre and more proud; the wall about it is flat.
    assert computation.qc["relief_max_um"] > 800 and abs(computation.qc["relief_p50_um"]) < 100, computation.qc
    with pytest.raises(ArtifactReliefShadeError, match="view must be one of"):
        compute_relief_shade(petalled, view="top")
    with pytest.raises(ArtifactReliefShadeError, match="flat to the light"):
        compute_relief_shade(petalled, view="front", window_mm=(-60.0, 12.0, 60.0, 25.0))
    with pytest.raises(ArtifactReliefShadeError, match="no relief lies"):
        compute_relief_shade(petalled, view="front", window_mm=(-60.0, 200.0, 60.0, 300.0))
    with pytest.raises(ArtifactReliefShadeError, match="slow_um must be wider"):
        relief_shade_recipe(view="front", source_vertex_count=10, source_face_count=10, slow_um=300, grain_um=400)


def test_the_record_keeps_the_receipt_and_the_pixels_travel_beside_it(petalled) -> None:
    computation = compute_relief_shade(petalled, view="front")
    session = commit_relief_shade(petalled, computation, record_id="record:shade", created_at=STAMP, operator="tester")
    record = session.document.record_index["record:shade"]
    assert record.type == RELIEF_SHADE_RECORD_TYPE
    receipt = relief_shade_receipt_from_record(record)
    assert receipt["raster_sha256"] == computation.raster.raster_sha256
    assert record.extensions[RELIEF_SHADE_PAYLOAD_EXTENSION_KEY]["view"] == "front"
    validate_known_records(session.document)
    assert require_relief_shade_raster(record, computation.raster) is computation.raster
    other = ReliefShadeRaster(
        pixels=np.zeros((2, 2, 2), dtype=np.uint8), pixels_per_meter=10_000, left_um=0, bottom_um=0, view="front"
    )
    with pytest.raises(ArtifactReliefShadeError, match="is not the one record"):
        require_relief_shade_raster(record, other)


def test_the_sheet_stipples_the_shade_on_the_elevation_half_only(petalled) -> None:
    """Dots are laid on a paper grid by a hash of the cell, so the same
    shade at the same scale gives the same sheet; on a mirrored figure the
    dots that fall on the section's side are left off and counted; without
    the option the sheet keeps its bytes."""

    computation = compute_relief_shade(petalled, view="front")
    session = commit_relief_shade(petalled, computation, record_id="record:shade", created_at=STAMP, operator="tester")
    rasters = {"record:shade": computation.raster}
    title = TitleBlock(artifact_label="양각 시험 호")
    page = SheetPage(size="A4", orientation="portrait")

    def sheet(**kwargs):
        return compose_drawing_sheet(
            session.document, ["record:front"],
            options=DrawingSheetOptions(
                title_block=title, page=page, scale_denominator=2.0,
                mirror_sections=(("record:front", "record:section"),), **kwargs,
            ),
            rasters=rasters if "relief_stipples" in kwargs else None,
        )

    bundle = sheet(relief_stipples=(("record:shade", "record:front"),))
    validate_drawing_sheet_bytes(bundle.svg_bytes, bundle.sidecar_bytes)
    root = ET.fromstring(bundle.svg_bytes)
    groups = [el for el in root.iter() if el.attrib.get("id", "").startswith("relief-stipple-")]
    assert len(groups) == 1 and groups[0].attrib["data-record-id"] == "record:shade"
    circles = [el for el in groups[0] if el.tag.endswith("circle")]
    axis = next(el for el in root.iter() if "mirror:center-axis" in el.attrib.get("id", ""))
    axis_x = float(axis.attrib["d"].replace("M", " ").replace("L", " ").split()[0])
    assert len(circles) > 50
    assert all(float(c.attrib["cx"]) < axis_x for c in circles), "dots fall on the elevation's side only"
    assert all(float(c.attrib["r"]) == 0.06 for c in circles)
    sidecar = json.loads(bundle.sidecar_bytes.decode("utf-8"))
    block = sidecar["relief_stipples"]
    assert block["pitch_paper_mm"] == 0.15 and block["dot_paper_mm"] == 0.12 and block["hash"] == "splitmix64_cell/v1"
    (drawn,) = block["drawn"]
    assert drawn["record_id"] == "record:shade" and drawn["half"] == "elevation"
    assert int(drawn["dot_count"]) == len(circles) and int(drawn["dropped_section_side_count"]) > 50
    assert block["records"][0]["raster_sha256"] == computation.raster.raster_sha256
    # The same shade, the same sheet; no option, the old sheet.
    assert sheet(relief_stipples=(("record:shade", "record:front"),)).svg_bytes == bundle.svg_bytes
    plain = sheet()
    assert b"relief-stipple" not in plain.svg_bytes
    assert "relief_stipples" not in json.loads(plain.sidecar_bytes.decode("utf-8"))
    # A coarser grid and a bigger dot are the drafter's choice.
    coarse = sheet(relief_stipples=(("record:shade", "record:front"),), stipple_pitch_mm=0.3, stipple_dot_mm=0.2)
    coarse_root = ET.fromstring(coarse.svg_bytes)
    coarse_circles = [el for el in coarse_root.iter() if el.tag.endswith("circle")]
    assert 0 < len(coarse_circles) < len(circles) and all(float(c.attrib["r"]) == 0.1 for c in coarse_circles)
    with pytest.raises(DrawingSheetError, match="needs its raster"):
        compose_drawing_sheet(
            session.document, ["record:front"],
            options=DrawingSheetOptions(
                title_block=title, page=page, scale_denominator=2.0, relief_stipples=(("record:shade", "record:front"),),
            ),
        )
    with pytest.raises(DrawingSheetError, match="not the plane of"):
        compose_drawing_sheet(
            session.document, ["record:top"],
            options=DrawingSheetOptions(
                title_block=title, page=page, scale_denominator=2.0, relief_stipples=(("record:shade", "record:top"),),
            ),
            rasters=rasters,
        )
    with pytest.raises(DrawingSheetError, match="is not a relief shade"):
        compose_drawing_sheet(
            session.document, ["record:front"],
            options=DrawingSheetOptions(
                title_block=title, page=page, scale_denominator=2.0, relief_stipples=(("record:front", "record:front"),),
            ),
            rasters={"record:front": computation.raster},
        )
    with pytest.raises(DrawingSheetError, match="stipple_pitch_mm must be"):
        DrawingSheetOptions(title_block=title, page=page, scale_denominator=2.0, stipple_pitch_mm=0.0)
    with pytest.raises(DrawingSheetError, match="cannot be stippled twice"):
        DrawingSheetOptions(
            title_block=title, page=page, scale_denominator=2.0,
            relief_stipples=(("record:shade", "record:front"), ("record:shade", "record:front")),
        )


def test_the_cell_hash_is_uniform_and_the_same_every_time() -> None:
    ix, iy = np.meshgrid(np.arange(-50, 50), np.arange(-50, 50))
    first = _cell_uniforms(ix.ravel(), iy.ravel())
    assert first.shape == (10_000, 3) and (first >= 0.0).all() and (first < 1.0).all()
    assert np.array_equal(first, _cell_uniforms(ix.ravel(), iy.ravel()))
    assert abs(first.mean() - 0.5) < 0.02
    assert not np.array_equal(first[:, 0], first[:, 1])


def test_the_hollows_between_motifs_can_be_shaded_darker(petalled) -> None:
    """A hand stipples the ground between raised motifs darker the deeper
    it lies; the cavity term adds that to the slope's shade and is off by
    default, so the default shade keeps its bytes."""

    plain = compute_relief_shade(petalled, view="front")
    hollowed = compute_relief_shade(petalled, view="front", cavity_um=300, cavity_gain_thousandths=800)
    assert validate_relief_shade_recipe(hollowed.recipe)["shade_policy"]["cavity_um"] == 300
    assert hollowed.qc["shaded_pixel_count"] > plain.qc["shaded_pixel_count"]
    assert compute_relief_shade(petalled, view="front").raster.raster_sha256 == plain.raster.raster_sha256
