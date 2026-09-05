"""기와: the back develops, and the rubbing shows what the mesh carries.

The tile path is 전개 then 탁본 - unroll the struck surface, then ink it -
and both halves are asked here of a synthetic tile fragment: that the
development does not fold or tear, that a selection which folds through the
tile's thickness is refused rather than flattened, and that the ink on the
paper is the paddle's cord because the cord is in the mesh.
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
