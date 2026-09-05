"""The relief a scan keeps in its normal map, read onto a developed strip.

The museum's 빗살무늬토기 carries no incision in its mesh and every comb
stroke in an 8192 x 8192 object-space normal map (docs/REAL_DATA_TRIAL.md).
These tests build that situation: a generated vessel whose mesh is smooth,
an OBJ that gives every corner a texture coordinate, and a normal map drawn
from a relief the mesh does not have - grooves at known heights - and ask
the developed rubbing to find the grooves where they are.
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.core.artifact_developed_rubbing import (
    ARTBOARD_LARGEST_COVERED_RECTANGLE,
    DEVELOPED_RUBBING_ALGORITHM,
    DEVELOPED_TEXTURE_RUBBING_ALGORITHM,
    ArtifactDevelopedRubbingError,
    TextureReliefSource,
    commit_developed_rubbing,
    compute_developed_rubbing,
    compute_developed_rubbing_from_recipe,
    developed_rubbing_recipe,
    validate_developed_rubbing_recipe,
)
from src.core.project_file import load_artifact_project, save_artifact_project
from src.core.artifact_record_validation import validate_known_records
from src.core.artifact_rubbing_export import build_rubbing_export, validate_rubbing_export_bytes
from src.core.artifact_rubbing_extractor import RELIEF_MODEL_CONTACT
from src.core.artifact_surface_strip import select_positioned_surface_strip, strip_parameters
from src.core.artifact_texture_relief import (
    NORMAL_MAP_ENCODING,
    NORMAL_MAP_ENCODING_X_FLIPPED,
    NORMAL_MAP_ENCODINGS,
    TEXTURE_RELIEF_DEPTH_MEASURE,
    ArtifactTextureReliefError,
    rank_normal_map_encodings,
    read_normal_map,
    read_obj_texture_atlas,
    require_atlas_matches,
    rigid_rotation_between,
    texture_relief_block,
    validate_texture_relief_block,
)
from src.core.artifact_tile_unwrap_extractor import extract_tile_unwrap_development
from src.core.artifact_tile_unwrap_extractor import (
    SECTION_CENTER_CANONICAL_AXIS,
    STATION_MERIDIAN_ARC,
    commit_artifact_tile_unwrap,
    compute_artifact_tile_unwrap,
)
from src.core.canonical_json import canonical_json_sha256
from synthetic_vessel import HEIGHT_MM, outer_radius, positioned_vessel_session

STAMP = "2026-09-05T00:00:00Z"
#: Where the grooves are cut into the normal map, in millimetres up the wall.
GROOVE_HEIGHTS_MM = (30.0, 45.0, 60.0)
GROOVE_HALF_WIDTH_MM = 0.6
GROOVE_DEPTH_MM = 0.3
MAP_SIDE = 1024


def _write_textured_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """The vessel as an OBJ whose every corner carries a texture coordinate:
    u round the axis, v up the wall, the seam at +X where no strip is cut."""

    lines = ["# generated vessel with cylindrical texture coordinates"]
    for x, y, z in vertices.tolist():
        lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
    for x, y, z in vertices.tolist():
        u = (math.atan2(y, x) % (2.0 * math.pi)) / (2.0 * math.pi)
        v = z / HEIGHT_MM
        lines.append(f"vt {u:.6f} {v:.6f}")
    for a, b, c in faces.tolist():
        lines.append(f"f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _groove_relief(z_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Depth of the grooves at each height, and its derivative in z."""

    depth = np.zeros_like(z_mm)
    slope = np.zeros_like(z_mm)
    for centre in GROOVE_HEIGHTS_MM:
        t = (z_mm - centre) / GROOVE_HALF_WIDTH_MM
        inside = np.abs(t) < 1.0
        bump = (1.0 - t**2) ** 2
        depth = depth - GROOVE_DEPTH_MM * np.where(inside, bump, 0.0)
        slope = slope - GROOVE_DEPTH_MM * np.where(
            inside, 2.0 * (1.0 - t**2) * (-2.0 * t) / GROOVE_HALF_WIDTH_MM, 0.0
        )
    return depth, slope


def _write_normal_map(path: Path) -> None:
    """An object-space normal map of the vessel's outer wall with the grooves
    cut in: the normal of r(z) = R(z) + d(z) about the axis."""

    from PIL import Image

    u = (np.arange(MAP_SIDE) + 0.5) / MAP_SIDE
    v = 1.0 - (np.arange(MAP_SIDE) + 0.5) / MAP_SIDE  # row 0 is the top of the image
    theta = 2.0 * math.pi * u[None, :]
    z = HEIGHT_MM * v[:, None]
    radius = np.array([outer_radius(float(h)) for h in z[:, 0]])[:, None]
    step = 0.05
    d_radius = (
        np.array([outer_radius(float(h) + step) - outer_radius(float(h) - step) for h in z[:, 0]])
        / (2.0 * step)
    )[:, None]
    _depth, d_depth = _groove_relief(z)
    slope_z = d_radius + d_depth
    normal = np.stack(
        [np.cos(theta) * np.ones_like(z), np.sin(theta) * np.ones_like(z), -slope_z * np.ones_like(theta)],
        axis=-1,
    )
    _ = radius
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    rgb = np.clip(np.rint((normal + 1.0) * 127.5), 0, 255).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)


@pytest.fixture(scope="module")
def textured():
    session, vertices, faces = positioned_vessel_session(
        segments=96, rings=40, document_id="artifact:textured"
    )
    directory = tempfile.mkdtemp()
    obj_path = Path(directory) / "vessel.obj"
    map_path = Path(directory) / "vessel_nor.png"
    _write_textured_obj(obj_path, vertices, np.asarray(faces, dtype=np.int64))
    _write_normal_map(map_path)
    atlas = read_obj_texture_atlas(obj_path)
    normal_map = read_normal_map(map_path)
    strip = select_positioned_surface_strip(
        session,
        strip_parameters(
            reference_angle_microdegrees=-90_000_000,
            width_um=20_000,
            minimum_height_um=10_000,
            maximum_height_um=75_000,
        ),
    )
    unwrap = compute_artifact_tile_unwrap(
        session,
        longitudinal_axis="z",
        record_view="top",
        selected_face_indices=strip.face_indices,
        n_sections=12,
        section_center_policy=SECTION_CENTER_CANONICAL_AXIS,
        station_policy=STATION_MERIDIAN_ARC,
    )
    session = commit_artifact_tile_unwrap(
        session, unwrap, record_id="record:unwrap:strip", created_at=STAMP, operator="tester"
    )
    return session, vertices, np.asarray(faces, dtype=np.int64), atlas, normal_map, obj_path, map_path


def _rubbing(session, source, **overrides):
    settings = dict(
        pixels_per_mm=10,
        margin_um=0,
        reference_radius_um=2_000,
        depth_quantization_um=10,
        black_point_um=100,
        ink_strength_percent=100,
        relief_polarity="raised",
        relief_model=RELIEF_MODEL_CONTACT,
        contact_ink_percent=70,
        artboard_policy=ARTBOARD_LARGEST_COVERED_RECTANGLE,
    )
    settings.update(overrides)
    return compute_developed_rubbing(session, "record:unwrap:strip", texture_relief=source, **settings)


def test_the_atlas_is_the_mesh_welded_and_says_so_when_it_is_not(textured) -> None:
    _session, vertices, faces, atlas, normal_map, obj_path, _map_path = textured
    assert atlas.triangle_count == faces.shape[0]
    assert np.array_equal(atlas.triangles, faces)
    assert np.abs(atlas.vertices - vertices).max() < 1e-5
    assert atlas.sha256 == hashlib.sha256(obj_path.read_bytes()).hexdigest()
    require_atlas_matches(atlas, vertices, faces)
    with pytest.raises(ArtifactTextureReliefError, match="triangles"):
        require_atlas_matches(atlas, vertices, faces[:-1])
    with pytest.raises(ArtifactTextureReliefError, match="stray"):
        require_atlas_matches(atlas, vertices + 0.01, faces)
    # A rigid motion is recovered exactly; a stretch is refused.
    angle = math.radians(37.0)
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    moved = vertices @ rotation.T + np.array([3.0, -2.0, 5.0])
    assert np.allclose(rigid_rotation_between(atlas.vertices, moved), rotation, atol=1e-9)
    with pytest.raises(ArtifactTextureReliefError, match="rigid"):
        rigid_rotation_between(atlas.vertices, vertices * 1.01)
    assert normal_map.width == MAP_SIDE and normal_map.height == MAP_SIDE
    block = texture_relief_block(atlas, normal_map, smoothing_um=1_000)
    assert validate_texture_relief_block(block) == block
    with pytest.raises(ArtifactTextureReliefError, match="smoothing_um"):
        texture_relief_block(atlas, normal_map, smoothing_um=10)
    with pytest.raises(ArtifactTextureReliefError, match="exactly"):
        validate_texture_relief_block({**block, "extra": 1})


def test_the_grooves_the_mesh_lacks_are_inked_where_the_map_puts_them(textured) -> None:
    """The mesh is smooth, so a mesh-depth rubbing of the strip is blank
    where the map's grooves are; the texture-relief rubbing inks them at
    the heights they were drawn at, and its recipe says where it came from."""

    session, _vertices, _faces, atlas, normal_map, _obj_path, _map_path = textured
    source = TextureReliefSource(atlas=atlas, normal_map=normal_map)
    textured_rubbing = _rubbing(session, source)
    plain_rubbing = _rubbing(session, None)

    recipe = textured_rubbing.recipe_dict()
    assert recipe["algorithm"] == DEVELOPED_TEXTURE_RUBBING_ALGORITHM
    assert recipe["depth_policy"]["measure"] == TEXTURE_RELIEF_DEPTH_MEASURE
    assert recipe["texture_relief"]["atlas"]["sha256"] == atlas.sha256
    assert recipe["texture_relief"]["normal_map"]["sha256"] == normal_map.sha256
    assert validate_developed_rubbing_recipe(recipe) == recipe
    plain = plain_rubbing.recipe_dict()
    assert plain["algorithm"] == DEVELOPED_RUBBING_ALGORITHM
    assert "texture_relief" not in plain
    qc = textured_rubbing.qc_dict()
    assert qc["depth_measure"] == TEXTURE_RELIEF_DEPTH_MEASURE
    assert qc["texture_relief_unmatched_corner_count"] == 0
    assert qc["texture_relief_unreadable_pixel_count"] == 0
    # The grooves are 0.3 mm deep; smoothed over 1 mm and sampled at the
    # map's texel, the integration recovers most of that depth.
    assert 150 <= -qc["texture_relief_height_min_um_rounded"] <= 600

    def ink_by_height(computation) -> tuple[np.ndarray, np.ndarray]:
        # Row 0 of the packed raster is the artboard's bottom row; the QC's
        # height profile says what height each band of rows sits at.
        raster = computation.raster
        pixels = raster.pixels
        profile = pixels[:, :, 0].astype(np.float64).mean(axis=1)
        bands = np.asarray(computation.qc_dict()["artboard_height_profile_um"], dtype=np.float64) / 1000.0
        rows = np.arange(pixels.shape[0]) + 0.5
        stations = np.linspace(0.0, float(pixels.shape[0]), bands.shape[0])
        return np.interp(rows, stations, bands), profile

    heights, ink = ink_by_height(textured_rubbing)
    _plain_heights, plain_ink = ink_by_height(plain_rubbing)
    on_groove = np.zeros(heights.shape, dtype=bool)
    for centre in GROOVE_HEIGHTS_MM:
        on_groove |= np.abs(heights - centre) < 0.4
    off_groove = np.ones(heights.shape, dtype=bool)
    for centre in GROOVE_HEIGHTS_MM:
        off_groove &= np.abs(heights - centre) > 2.5
    assert on_groove.sum() >= 3 and off_groove.sum() > 100
    # Raised polarity under the contact model: the wall takes the ink and a
    # groove stays paper-white, so the groove rows are the light ones.
    assert float(ink[on_groove].mean()) > float(ink[off_groove].mean()) + 40.0
    # The mesh-depth rubbing sees no groove: what tone it has there is the
    # facet noise it has everywhere, a fraction of the texture's contrast.
    texture_contrast = float(ink[on_groove].mean()) - float(ink[off_groove].mean())
    plain_contrast = abs(float(plain_ink[on_groove].mean()) - float(plain_ink[off_groove].mean()))
    assert plain_contrast < 25.0 and plain_contrast < texture_contrast / 3.0


def test_a_texture_rubbing_reopens_and_exports_and_needs_its_files_to_recompute(textured) -> None:
    session, _vertices, _faces, atlas, normal_map, obj_path, map_path = textured
    source = TextureReliefSource(atlas=atlas, normal_map=normal_map)
    computation = _rubbing(session, source)
    committed = commit_developed_rubbing(
        session, computation, record_id="record:rubbing:texture", created_at=STAMP, operator="tester"
    )
    validate_known_records(committed.document)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "textured.amr"
        save_artifact_project(path, committed.document)
        reopened = load_artifact_project(path)
    record = reopened.record_index["record:rubbing:texture"]
    assert record.recipe["texture_relief"]["normal_map"]["sha256"] == normal_map.sha256

    # Recomputing the same recipe with the same files gives the same raster;
    # without the files it is refused, and with other files it is refused.
    again = compute_developed_rubbing_from_recipe(committed, record.recipe, texture_relief=source)
    assert again.raster.raster_sha256 == computation.raster.raster_sha256
    with pytest.raises(ArtifactDevelopedRubbingError, match="needs its texture atlas"):
        compute_developed_rubbing_from_recipe(committed, record.recipe)
    other_map = Path(tempfile.mkdtemp()) / "other_nor.png"
    from PIL import Image

    Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8), mode="RGB").save(other_map)
    with pytest.raises(ArtifactDevelopedRubbingError, match="not the one the recipe names"):
        compute_developed_rubbing_from_recipe(
            committed,
            record.recipe,
            texture_relief=TextureReliefSource(atlas=atlas, normal_map=read_normal_map(other_map)),
        )

    # The export carries the recipe under the 1.4.0 contract and round-trips.
    bundle = build_rubbing_export(committed.document, "record:rubbing:texture", computation.raster)
    sidecar = json.loads(bundle.sidecar_bytes.decode("utf-8"))
    assert sidecar["schema_version"] == "1.4.0"
    assert sidecar["recipe"]["texture_relief"]["atlas"]["sha256"] == atlas.sha256
    loaded = validate_rubbing_export_bytes(bundle.png_bytes, bundle.sidecar_bytes, document=committed.document)
    assert loaded.raster_sha256 == computation.raster.raster_sha256
    assert obj_path.exists() and map_path.exists()


def test_the_map_is_read_under_the_encoding_the_numbers_favour(textured) -> None:
    """Nothing in a normal map file says which axes its baker mirrored, and
    the museum's map has x reversed against its own OBJ.  Read with the
    wrong convention the slopes do not integrate; the misfit of each
    candidate is a number the drafter chooses by, the choice goes into the
    recipe, and a file read under another convention is refused against it."""

    session, _vertices, _faces, atlas, normal_map, _obj_path, _map_path = textured
    record = session.document.record_index["record:unwrap:strip"]
    mesh = session.materialize().mesh
    unwrap, _qc, _radius = extract_tile_unwrap_development(mesh, record.recipe)
    canonical = np.asarray(mesh.vertices, dtype=np.float64)
    misfits = rank_normal_map_encodings(
        developed_uv_mm=np.asarray(unwrap.uv_um, dtype=np.float64) / 1000.0,
        developed_faces=np.asarray(unwrap.faces, dtype=np.int64),
        developed_points_mm=canonical[np.asarray(unwrap.source_vertex_indices, dtype=np.int64)],
        source_face_indices=np.asarray(unwrap.source_face_indices, dtype=np.int64),
        source_vertex_indices=np.asarray(unwrap.source_vertex_indices, dtype=np.int64),
        atlas=atlas,
        normal_map=normal_map,
        source_to_canonical_rotation=rigid_rotation_between(atlas.vertices, canonical),
        pixels_per_mm=10,
        smoothing_um=1_000,
    )
    assert set(misfits) == set(NORMAL_MAP_ENCODINGS)
    # The generated map follows the convention as written: it integrates
    # best read as is.
    assert min(misfits, key=misfits.__getitem__) == NORMAL_MAP_ENCODING
    assert misfits[NORMAL_MAP_ENCODING] < 50_000

    flipped = normal_map.with_encoding(NORMAL_MAP_ENCODING_X_FLIPPED)
    computation = _rubbing(session, TextureReliefSource(atlas=atlas, normal_map=flipped))
    recipe = computation.recipe_dict()
    assert recipe["texture_relief"]["normal_map"]["encoding"] == NORMAL_MAP_ENCODING_X_FLIPPED
    assert recipe["texture_relief"]["normal_map"]["sha256"] == normal_map.sha256
    with pytest.raises(ArtifactDevelopedRubbingError, match="the recipe says"):
        compute_developed_rubbing_from_recipe(
            session, recipe, texture_relief=TextureReliefSource(atlas=atlas, normal_map=normal_map)
        )
    with pytest.raises(ArtifactTextureReliefError, match="encoding"):
        normal_map.with_encoding("object_space_rgb8_upside_down/v1")


def test_the_mesh_rubbing_recipe_keeps_its_bytes() -> None:
    """A recipe drawn from the mesh must not have changed by a byte: every
    developed rubbing record ever written rebuilds through it."""

    recipe = developed_rubbing_recipe(
        development_record_id="record:unwrap:x",
        development_sha256="a" * 64,
        development_recipe_hash="b" * 64,
        pixels_per_mm=10,
        margin_um=0,
        reference_radius_um=2_000,
        depth_quantization_um=10,
        black_point_um=200,
        ink_strength_percent=100,
        relief_polarity="raised",
    )
    assert "texture_relief" not in recipe
    assert canonical_json_sha256(recipe) == (
        "9728d185c3be92ca916b6eb844eaecc467be0068f84b8a3458003b1d74c7aa4f"
    )


def test_the_sheet_says_the_ink_came_from_a_map(textured) -> None:
    from src.core.artifact_outline_extractor import compute_artifact_outline
    from src.core.artifact_vector_extractor import commit_vector_computation, compute_artifact_cutline
    from src.core.artifact_vector_record import PlanarFrame
    from src.core.drawing_sheet import (
        TEXTURE_RELIEF_CAPTION_TOKEN,
        TEXTURE_RUBBING_NOTE,
        DrawingSheetOptions,
        SheetPage,
        TitleBlock,
        compose_drawing_sheet,
        validate_drawing_sheet_bytes,
    )

    session, _vertices, _faces, atlas, normal_map, _obj_path, _map_path = textured
    computation = _rubbing(session, TextureReliefSource(atlas=atlas, normal_map=normal_map))
    session = commit_developed_rubbing(
        session, computation, record_id="record:rubbing:sheet", created_at=STAMP, operator="tester"
    )
    outline = compute_artifact_outline(session, "front", precision_grid_mm=0.05)
    session = commit_vector_computation(
        session, outline, record_id="record:elevation", created_at=STAMP, operator="tester"
    )
    cut = compute_artifact_cutline(
        session,
        PlanarFrame(
            origin_world_mm=(0.0, 0.0, 0.0),
            u_axis_world=(1.0, 0.0, 0.0),
            v_axis_world=(0.0, 0.0, 1.0),
            normal_world=(0.0, -1.0, 0.0),
        ),
    )
    session = commit_vector_computation(
        session, cut, record_id="record:section", created_at=STAMP, operator="tester"
    )
    bundle = compose_drawing_sheet(
        session.document,
        ["record:elevation"],
        options=DrawingSheetOptions(
            title_block=TitleBlock(artifact_label="합성 토기", rows=(("작성", "tester"),)),
            page=SheetPage(size="A4", orientation="portrait"),
            scale_denominator=1.0,
            mirror_sections=(("record:elevation", "record:section"),),
            rubbings_on_axis=(("record:rubbing:sheet", "record:elevation"),),
        ),
        rasters={"record:rubbing:sheet": computation.raster},
    )
    validate_drawing_sheet_bytes(bundle.svg_bytes, bundle.sidecar_bytes)
    sidecar = json.loads(bundle.sidecar_bytes)
    assert sidecar["computed_rubbing_note"] == TEXTURE_RUBBING_NOTE
    assert any(row["value"] == TEXTURE_RUBBING_NOTE for row in sidecar["title_block"])
    captions = [figure["caption"] for figure in sidecar["figures"] if "caption" in figure]
    assert captions and all(TEXTURE_RELIEF_CAPTION_TOKEN in caption for caption in captions)
