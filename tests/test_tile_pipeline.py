"""기와: both walls develop, and the rubbing shows what the mesh carries.

The tile path is 전개 then 탁본 - unroll the recorded surface, then ink it -
and a tile has two of them: the 등면 the paddle struck, carrying its 타날문,
and the 내면 the clay took from the 와통, carrying the 포목흔.  Both halves of
the path are asked here of synthetic tiles: that a development does not fold
or tear, that a selection folding through the thickness is refused rather
than flattened, that the ink is the paddle's cord because the cord is in the
mesh, that the two walls develop to different widths because they lie at
different radii, that a whole tapered tile develops to a trapezoid rather
than being squared off, and that the 내면's own radius gives back the mould
it was formed on.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.core.artifact_developed_rubbing import (
    ARTBOARD_DEVELOPMENT_BOUNDS,
    commit_developed_rubbing,
    compute_developed_rubbing,
)
from src.core.artifact_rubbing_extractor import (
    RELIEF_MODEL_CONTACT,
    RUBBING_TONE_MEDIUM,
    rubbing_tone_settings,
)
from src.core.artifact_tile_unwrap_extractor import (
    MAX_TILE_UNWRAP_QC_FACES,
    SECTION_CENTER_CANONICAL_AXIS,
    STATION_CENTERLINE_ARC,
    ArtifactTileUnwrapError,
    commit_artifact_tile_unwrap,
    compute_artifact_tile_unwrap,
)
from synthetic_tile import AMKIWA, TileShape, tile_session

STAMP = "2026-09-05T00:00:00Z"
STEP_MM = 2.0
#: A fragment rather than a whole tile: 기와 are usually found broken, and a
#: whole one sampled finely enough to carry a 3 mm cord is past the
#: recording surface's 250,000-face limit anyway.
FRAGMENT = TileShape(
    kind=AMKIWA,
    length_mm=120.0,
    inner_radius_mm=210.0,
    thickness_mm=20.0,
    span_deg=40.0,
)


def _tile(*, relief: bool = True):
    return tile_session(
        FRAGMENT,
        axial_step_mm=STEP_MM,
        angular_step_mm=STEP_MM,
        relief=relief,
        on_canonical_axis=True,
    )


def _back_faces(vertices: np.ndarray, faces: np.ndarray, *, whole: bool = True) -> list[int]:
    """The struck back: the wall further from the tile's own axis.

    Standing on its axis, the tile's cylinder axis is +Z through the origin.
    ``whole`` asks that every corner of a face be on that wall.  A face with
    one corner on the inner wall belongs to a side or an end - it folds
    through the tile's thickness, and no development can flatten it.
    """

    radius = np.hypot(vertices[:, 0], vertices[:, 1])
    threshold = FRAGMENT.inner_radius_mm + FRAGMENT.thickness_mm / 2.0
    if whole:
        on_back = (radius > threshold)[faces].all(axis=1)
    else:
        on_back = np.hypot(
            vertices[faces].mean(axis=1)[:, 0], vertices[faces].mean(axis=1)[:, 1]
        ) > threshold
    inside = np.abs(vertices[faces].mean(axis=1)[:, 2]) < (FRAGMENT.length_mm / 2.0 - 5.0)
    return [int(index) for index in np.nonzero(on_back & inside)[0]]


def _develop(session, faces_selected):
    return compute_artifact_tile_unwrap(
        session,
        longitudinal_axis="z",
        record_view="top",
        selected_face_indices=faces_selected,
        n_sections=12,
        section_center_policy=SECTION_CENTER_CANONICAL_AXIS,
        station_policy=STATION_CENTERLINE_ARC,
    )


def test_a_tiles_back_develops_without_folding_or_tearing() -> None:
    session, vertices, faces = _tile()
    selected = _back_faces(vertices, faces)
    assert 2000 < len(selected) < 250_000

    unwrap = _develop(session, selected)
    qc = unwrap.qc
    assert qc["foldover_face_count"] == 0
    assert qc["degenerate_uv_face_count"] == 0
    assert qc["negative_orientation_face_count"] == 0
    assert qc["boundary_loop_count"] == 1
    assert qc["connected_component_count"] == 1
    assert qc["face_count"] == len(selected)
    # The development is as long as the piece of tile it came from.
    assert qc["height_um"] == pytest.approx(
        (FRAGMENT.length_mm - 10.0) * 1000.0, rel=0.02
    )


def test_a_plain_tile_develops_exactly() -> None:
    """A segment of a cylinder is developable, so nothing should distort.

    What is left over on a corded tile is the cord, not the unrolling.
    """

    session, vertices, faces = _tile(relief=False)
    unwrap = _develop(session, _back_faces(vertices, faces))
    # A thousandth, which is the arithmetic's own noise on a 40 degree arc.
    assert unwrap.qc["distortion_max_millionths"] < 1_000
    assert unwrap.qc["distortion_mean_millionths"] < 200

    corded, corded_vertices, corded_faces = _tile()
    with_cord = _develop(corded, _back_faces(corded_vertices, corded_faces))
    assert with_cord.qc["distortion_max_millionths"] > 0
    # And it is the relief's own steepness, not a failure: still small.
    assert with_cord.qc["distortion_mean_millionths"] < 100_000


def test_a_selection_that_folds_through_the_thickness_is_refused() -> None:
    """Selecting by a face's centroid lets the tile's sides in.

    A side strip runs from the outer wall to the inner one, so flattening it
    stretches it by several times, and the sides of a fragment are not even
    joined to each other.  Which rule catches it depends on what came in -
    the distortion gate, or the one component the recording surface must be -
    but it is caught, and that is what a drafter needs: the development is
    refused rather than published with a wall folded into it.
    """

    session, vertices, faces = _tile(relief=False)
    whole = _back_faces(vertices, faces)
    by_centroid = _back_faces(vertices, faces, whole=False)
    assert len(by_centroid) > len(whole)
    assert not pytest.approx(0) == len(by_centroid) - len(whole)

    with pytest.raises(ArtifactTileUnwrapError) as refusal:
        _develop(session, by_centroid)
    assert any(
        word in str(refusal.value)
        for word in ("distortion", "component", "boundary", "foldover")
    ), str(refusal.value)


def _rubbing_of(relief: bool) -> np.ndarray:
    session, vertices, faces = _tile(relief=relief)
    unwrap = _develop(session, _back_faces(vertices, faces))
    session = commit_artifact_tile_unwrap(
        session, unwrap, record_id="record:unwrap:back", created_at=STAMP, operator="tester"
    )
    tone = rubbing_tone_settings(RUBBING_TONE_MEDIUM, relief_model=RELIEF_MODEL_CONTACT)
    computation = compute_developed_rubbing(
        session,
        "record:unwrap:back",
        pixels_per_mm=4,
        margin_um=0,
        reference_radius_um=700,
        depth_quantization_um=5,
        black_point_um=120,
        ink_strength_percent=100,
        relief_polarity="raised",
        relief_model=RELIEF_MODEL_CONTACT,
        contact_ink_percent=tone["contact_ink_percent"],
        artboard_policy=ARTBOARD_DEVELOPMENT_BOUNDS,
    )
    commit_developed_rubbing(
        session, computation, record_id="record:rubbing:back", created_at=STAMP, operator="tester"
    )
    pixels = np.asarray(computation.raster.pixels)
    covered = pixels[:, :, 1] > 0
    assert covered.mean() > 0.5
    return pixels[:, :, 0][covered].astype(np.float64)


def test_the_rubbing_shows_the_cord_because_the_cord_is_in_the_mesh() -> None:
    """The ink is the mesh's relief: take the relief away and it goes."""

    corded = _rubbing_of(True)
    plain = _rubbing_of(False)

    # A tile with nothing on its back inks flat: the paper lies on a cylinder
    # and touches it everywhere, so every pixel takes the same contact tone.
    assert float(plain.std()) < 1.0
    assert float(plain.max()) - float(plain.min()) < 4.0
    # A corded one does not.  The ridges take the full contact tone and the
    # paper misses between them, so the ink runs from that tone up toward the
    # paper - which is the whole point of taking a rubbing of a tile.
    assert float(corded.std()) > 10.0
    assert float(corded.min()) == pytest.approx(float(plain.min()), abs=2.0)
    assert float(corded.max()) > float(plain.max()) + 40.0


#: A whole 암키와, not a fragment: 34 cm long, a 76 degree arc.
WHOLE = TileShape(
    kind=AMKIWA,
    length_mm=340.0,
    inner_radius_mm=210.0,
    thickness_mm=20.0,
    span_deg=76.0,
    taper=0.10,
)
#: What a 3 mm 승문 actually needs.  Measured on this tile, the ink range
#: across the development is 142 at a 0.4 mm mesh and 150 at 1.2 mm - flat -
#: and collapses to 105 only at 1.6 mm, which is past the cord's own Nyquist.
#: So 1.2 mm is the coarse end of the useful range, and it is also what makes
#: a whole tile fit the 250,000-face recording surface.
DRAWING_STEP_MM = 1.2


def test_a_whole_tile_fits_the_recording_surface_at_the_step_the_cord_needs() -> None:
    """The 250,000-face cap does not stand between a 기와 and its 탁본.

    A whole tile sampled three times finer than the cord needs does exceed
    the cap - 555,000 faces of back at 0.6 mm - but that mesh draws no better
    than a 1.2 mm one, and the cap is published in the receipt schema and
    hashed into every payload header, so it is not a number to raise on a
    demo's say-so.  What the drafter needs is the step, and this holds it.
    """

    session, vertices, faces = tile_session(
        WHOLE,
        axial_step_mm=DRAWING_STEP_MM,
        angular_step_mm=DRAWING_STEP_MM,
        on_canonical_axis=True,
    )
    radius = np.hypot(vertices[:, 0], vertices[:, 1])
    threshold = WHOLE.inner_radius_mm + WHOLE.thickness_mm / 2.0
    centres = vertices[faces].mean(axis=1)
    selected = [
        int(index)
        for index in np.nonzero(
            (radius > threshold)[faces].all(axis=1)
            & (np.abs(centres[:, 2]) < WHOLE.length_mm / 2.0 - 6.0)
        )[0]
    ]
    assert 100_000 < len(selected) < MAX_TILE_UNWRAP_QC_FACES

    unwrap = compute_artifact_tile_unwrap(
        session,
        longitudinal_axis="z",
        record_view="top",
        selected_face_indices=selected,
        n_sections=16,
        section_center_policy=SECTION_CENTER_CANONICAL_AXIS,
        station_policy=STATION_CENTERLINE_ARC,
    )
    assert unwrap.qc["foldover_face_count"] == 0
    assert unwrap.qc["connected_component_count"] == 1
    # The whole tile, end margins aside, on one development.
    assert unwrap.qc["height_um"] == pytest.approx(
        (WHOLE.length_mm - 12.0) * 1000.0, rel=0.02
    )

    session = commit_artifact_tile_unwrap(
        session, unwrap, record_id="record:whole", created_at=STAMP, operator="tester"
    )
    tone = rubbing_tone_settings(RUBBING_TONE_MEDIUM, relief_model=RELIEF_MODEL_CONTACT)
    rubbing = compute_developed_rubbing(
        session,
        "record:whole",
        pixels_per_mm=4,
        margin_um=0,
        reference_radius_um=700,
        depth_quantization_um=5,
        black_point_um=120,
        ink_strength_percent=100,
        relief_polarity="raised",
        relief_model=RELIEF_MODEL_CONTACT,
        contact_ink_percent=tone["contact_ink_percent"],
        artboard_policy=ARTBOARD_DEVELOPMENT_BOUNDS,
    )
    pixels = np.asarray(rubbing.raster.pixels)
    covered = pixels[:, :, 1] > 0
    ink = pixels[:, :, 0][covered].astype(np.float64)
    low, high = np.percentile(ink, [2.0, 98.0])
    # And the cord is on the paper at full strength, not a smudge.
    assert float(high - low) > 100.0


#: A tile whose arc is wide enough that the difference is unmistakable.
ARC = TileShape(
    kind=AMKIWA,
    length_mm=110.0,
    inner_radius_mm=80.0,
    thickness_mm=16.0,
    span_deg=120.0,
)


def test_the_rubbing_is_wider_than_the_outline_and_by_exactly_the_arc() -> None:
    """기와의 탁본은 외선보다 클 수밖에 없다.

    Unrolling a curved surface measures the arc; looking down at it measures
    the arc's chord.  So a development of a tile is wider than the tile's own
    outline, by theta / (2 sin(theta/2)) and by nothing else - 7.7% on a 76
    degree 암키와, 55% on a half-round 수키와.  Both figures can sit on one
    sheet at one scale, so this is a thing a reader has to be told rather
    than left to discover with a ruler.
    """

    from src.core.artifact_outline_extractor import compute_artifact_outline
    from src.core.artifact_vector_extractor import commit_vector_computation
    from src.core.artifact_vector_record import vector_payload_from_record
    from src.core.drawing_sheet import (
        COMPUTED_RUBBING_CAPTION_PREFIX,
        DEVELOPED_WIDTH_NOTE,
        computed_rubbing_caption,
    )

    session, vertices, faces = tile_session(
        ARC, axial_step_mm=STEP_MM, angular_step_mm=STEP_MM, on_canonical_axis=True
    )
    radius = np.hypot(vertices[:, 0], vertices[:, 1])
    threshold = ARC.inner_radius_mm + ARC.thickness_mm / 2.0
    centres = vertices[faces].mean(axis=1)
    selected = [
        int(index)
        for index in np.nonzero(
            (radius > threshold)[faces].all(axis=1)
            & (np.abs(centres[:, 2]) < ARC.length_mm / 2.0 - 5.0)
        )[0]
    ]

    # The shadow: what the tile covers seen from the side, in millimetres.
    outline = compute_artifact_outline(session, "front", precision_grid_mm=0.2)
    session = commit_vector_computation(
        session, outline, record_id="record:tile:side", created_at=STAMP, operator="tester"
    )
    points = np.vstack(
        [
            np.asarray(path.points_mm, dtype=np.float64)
            for path in vector_payload_from_record(
                session.document.record_index["record:tile:side"]
            ).paths
        ]
    )
    shadow_mm = float(points[:, 0].max() - points[:, 0].min())

    # The development: the same surface unrolled.
    unwrap = compute_artifact_tile_unwrap(
        session,
        longitudinal_axis="z",
        record_view="top",
        selected_face_indices=selected,
        n_sections=12,
        section_center_policy=SECTION_CENTER_CANONICAL_AXIS,
        station_policy=STATION_CENTERLINE_ARC,
    )
    developed_mm = float(unwrap.unwrap.uv_um[:, 0].max() - unwrap.unwrap.uv_um[:, 0].min()) / 1000.0

    theta = math.radians(ARC.span_deg)
    arc_over_chord = theta / (2.0 * math.sin(theta / 2.0))
    assert arc_over_chord > 1.2
    assert developed_mm > shadow_mm
    assert developed_mm / shadow_mm == pytest.approx(arc_over_chord, rel=0.02)

    # And the sheet says so on the rubbing itself, in every rubbing's caption.
    caption = computed_rubbing_caption(
        {
            "relief_policy": {
                "model": "contact_envelope/v1",
                "reference_radius_requested_um": 700,
                "black_point_requested_um": 120,
                "contact_ink_percent": 70,
            }
        },
        developed=True,
    )
    assert caption.startswith(COMPUTED_RUBBING_CAPTION_PREFIX)
    assert DEVELOPED_WIDTH_NOTE in caption
    # A relief read off a projection is not a development and does not claim
    # an arc.
    flat = computed_rubbing_caption(
        {
            "relief_policy": {
                "model": "contact_envelope/v1",
                "reference_radius_requested_um": 700,
                "black_point_requested_um": 120,
                "contact_ink_percent": 70,
            }
        },
        developed=False,
    )
    assert DEVELOPED_WIDTH_NOTE not in flat


#: A whole 암키와, tapered so courses lap: 34 cm long, a 76 degree arc at the
#: wide end and a tenth less at the narrow one.
TAPERED = TileShape(
    kind=AMKIWA,
    length_mm=340.0,
    inner_radius_mm=210.0,
    thickness_mm=20.0,
    span_deg=76.0,
    taper=0.10,
)


def _wall_faces(shape, vertices, faces, *, outer: bool, margin_mm: float = 6.0):
    """The faces of one wall, whole - no face straddling the thickness."""

    radius = np.hypot(vertices[:, 0], vertices[:, 1])
    middle = shape.inner_radius_mm + shape.thickness_mm / 2.0
    on_wall = (radius > middle) if outer else (radius < middle)
    centres = vertices[faces].mean(axis=1)
    inside = np.abs(centres[:, 2]) < shape.length_mm / 2.0 - margin_mm
    return [int(index) for index in np.nonzero(on_wall[faces].all(axis=1) & inside)[0]]


def _develop_wall(session, selected, *, sections: int = 12):
    return compute_artifact_tile_unwrap(
        session,
        longitudinal_axis="z",
        record_view="top",
        selected_face_indices=selected,
        n_sections=sections,
        section_center_policy=SECTION_CENTER_CANONICAL_AXIS,
        station_policy=STATION_CENTERLINE_ARC,
    )


def test_a_tile_gives_a_rubbing_of_each_wall_and_they_are_not_the_same_size() -> None:
    """기와는 내면·외면 탁본이 다 들어간다.

    Both walls develop from the same tile: the 등면 the paddle struck, and the
    내면 the clay took from the 와통.  They are not the same development - the
    outer wall is a wall thickness further from the axis, so unrolling it
    measures a longer arc, by exactly (R + t) / R.  A sheet carrying both must
    therefore not be read as two views of one surface.
    """

    shape = TileShape(
        kind=AMKIWA,
        length_mm=140.0,
        inner_radius_mm=210.0,
        thickness_mm=20.0,
        span_deg=50.0,
    )
    session, vertices, faces = tile_session(
        shape, axial_step_mm=1.0, angular_step_mm=1.0, on_canonical_axis=True
    )
    widths = {}
    radii = {}
    for outer in (True, False):
        selected = _wall_faces(shape, vertices, faces, outer=outer)
        assert len(selected) > 10_000
        unwrap = _develop_wall(session, selected)
        assert unwrap.qc["foldover_face_count"] == 0
        assert unwrap.qc["connected_component_count"] == 1
        widths[outer] = unwrap.qc["width_um"] / 1000.0
        radii[outer] = unwrap.qc["section_mean_radius_um"] / 1000.0

    # Each wall is found at its own radius, to a tenth of a millimetre.
    assert radii[False] == pytest.approx(shape.inner_radius_mm, abs=0.5)
    assert radii[True] == pytest.approx(
        shape.inner_radius_mm + shape.thickness_mm, abs=0.5
    )
    # And the developed widths are in that same ratio, not equal.
    expected = (shape.inner_radius_mm + shape.thickness_mm) / shape.inner_radius_mm
    assert widths[True] / widths[False] == pytest.approx(expected, rel=0.01)


def test_the_inner_rubbing_shows_the_cloth_when_the_black_point_suits_it() -> None:
    """포목흔 is a tenth the cord's depth, so it needs its own 검정 기준.

    The 승문 on the 등면 stands 0.35 mm proud and inks at a 0.12 mm black
    point.  The 포목 weave on the 내면 is 0.09 mm all told, so that same black
    point is deeper than the whole texture and crushes it.  Measured on this
    tile, the weave's own contrast - what is left after a 4 mm high-pass, so
    the 26 mm 모골 facets do not count - runs 29 at 0.12 mm and 53 at 0.04 mm.
    The rule that follows is worth saying out loud: set the black point at
    about half the relief's depth.
    """

    from scipy.ndimage import uniform_filter

    shape = TileShape(
        kind=AMKIWA,
        length_mm=100.0,
        inner_radius_mm=210.0,
        thickness_mm=20.0,
        span_deg=40.0,
    )
    pixels_per_mm = 12
    session, vertices, faces = tile_session(
        shape, axial_step_mm=0.6, angular_step_mm=0.6, on_canonical_axis=True
    )
    unwrap = _develop_wall(
        session, _wall_faces(shape, vertices, faces, outer=False, margin_mm=5.0)
    )
    session = commit_artifact_tile_unwrap(
        session, unwrap, record_id="record:inner", created_at=STAMP, operator="tester"
    )
    tone = rubbing_tone_settings(RUBBING_TONE_MEDIUM, relief_model=RELIEF_MODEL_CONTACT)

    def weave_contrast(black_point_um: int) -> float:
        computation = compute_developed_rubbing(
            session,
            "record:inner",
            pixels_per_mm=pixels_per_mm,
            margin_um=0,
            reference_radius_um=700,
            depth_quantization_um=2,
            black_point_um=black_point_um,
            ink_strength_percent=100,
            relief_polarity="raised",
            relief_model=RELIEF_MODEL_CONTACT,
            contact_ink_percent=tone["contact_ink_percent"],
            artboard_policy=ARTBOARD_DEVELOPMENT_BOUNDS,
        )
        pixels = np.asarray(computation.raster.pixels)
        covered = pixels[:, :, 1] > 0
        grey = pixels[:, :, 0].astype(np.float64)
        rows, columns = np.nonzero(covered)
        window = int(round(4.0 * pixels_per_mm))
        patch = grey[
            rows.min() + 3 * window : rows.max() - 3 * window,
            columns.min() + 3 * window : columns.max() - 3 * window,
        ]
        return float((patch - uniform_filter(patch, size=window, mode="nearest")).std())

    for_the_cord = weave_contrast(120)
    for_the_cloth = weave_contrast(40)
    assert for_the_cloth > 40.0
    assert for_the_cloth > 1.5 * for_the_cord


def test_a_whole_tile_develops_to_a_trapezoid_because_its_ends_differ() -> None:
    """완형 기와의 위아래 호 길이는 다르다.

    A 암키와 is tapered so that one course laps the next, so the arc at one
    end is not the arc at the other and the development is a trapezoid, not a
    rectangle.  The program must not square it off: cropping the development
    to a rectangle would take a strip of the wide end away, and pasting it on
    the sheet as a rectangle would make a reader measure the wrong width.
    """

    session, vertices, faces = tile_session(
        TAPERED, axial_step_mm=1.2, angular_step_mm=1.2, on_canonical_axis=True
    )
    selected = _wall_faces(TAPERED, vertices, faces, outer=True)
    unwrap = _develop_wall(session, selected, sections=16)
    uv_mm = np.asarray(unwrap.unwrap.uv_um, dtype=np.float64) / 1000.0

    def width_at(station_mm: float) -> float:
        here = np.abs(uv_mm[:, 1] - station_mm) < 1.5
        return float(uv_mm[here, 0].max() - uv_mm[here, 0].min())

    low, high = float(uv_mm[:, 1].min()), float(uv_mm[:, 1].max())
    wide, narrow = width_at(low + 2.0), width_at(high - 2.0)
    if wide < narrow:
        wide, narrow = narrow, wide
    # The ends are read 6 mm in from the cut ends, so the taper they see is
    # the taper over that shorter run.
    kept = 1.0 - 2.0 * 6.0 / TAPERED.length_mm
    expected = (1.0 - TAPERED.taper * (1.0 - kept) / 2.0) / (
        1.0 - TAPERED.taper * (1.0 + kept) / 2.0
    )
    assert wide / narrow == pytest.approx(expected, rel=0.01)
    assert wide - narrow > 25.0
    # The record's own width is the widest station, not an average.
    assert unwrap.qc["width_um"] / 1000.0 == pytest.approx(wide, abs=1.0)


def test_the_inner_development_measures_the_mould_the_tile_was_formed_on() -> None:
    """추정 와통 지름: the 내면 still carries the mould's radius.

    Unrolling fits a circle to every section, so the radius is measured
    already and only has to be read back.  Read off the 내면 it is the 와통;
    read off the 등면 it is the outer form, one wall thickness larger, and
    the difference is the tile's own thickness.
    """

    from src.core.artifact_tile_unwrap_record import developed_cylinder_from_record

    session, vertices, faces = tile_session(
        TAPERED, axial_step_mm=1.2, angular_step_mm=1.2, on_canonical_axis=True
    )
    measured = {}
    for outer in (False, True):
        unwrap = _develop_wall(
            session, _wall_faces(TAPERED, vertices, faces, outer=outer), sections=16
        )
        committed = commit_artifact_tile_unwrap(
            session,
            unwrap,
            record_id="record:wall",
            created_at=STAMP,
            operator="tester",
        )
        cylinder = developed_cylinder_from_record(
            committed.document.record_index["record:wall"]
        )
        measured[outer] = cylinder
        assert cylinder.diameter_um == 2 * cylinder.radius_um
        assert cylinder.section_count == 16
        assert cylinder.section_fit_valid_count == 16

    # The 와통 this tile was formed on, to within a millimetre of the truth.
    assert measured[False].diameter_mm == pytest.approx(
        2.0 * TAPERED.inner_radius_mm, abs=1.0
    )
    # And the two readings differ by the wall, both ways.
    assert measured[True].radius_mm - measured[False].radius_mm == pytest.approx(
        TAPERED.thickness_mm, abs=0.5
    )
