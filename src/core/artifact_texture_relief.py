"""Relief a scan keeps in its texture rather than in its mesh.

A published scan is often a coarse mesh with the fine surface baked into an
image: the museum's 빗살무늬토기 has faces 1.3 mm across and no incision in
them, and an 8192 x 8192 object-space normal map that carries every comb
stroke.  A rubbing drawn from mesh depth shows nothing there.  This module
reads that relief back onto a developed strip so the developed rubbing can
draw it: the OBJ's texture coordinates say which texel each point of the
mesh wears, the normal map says how the surface really faces there, and the
difference from how the surface faces at large is a slope, which integrated
over the developed strip is a height.

Everything here is derived from files the recipe names by SHA-256 - the OBJ
whose corners carry the texture coordinates and the normal map image - and
from the development itself.  The mesh the session holds must be the OBJ's
geometry, welded by position; ``require_atlas_matches`` proves it before
anything is read.  What the normal map was baked from is not in the file,
so a sheet that carries this rubbing must say where it came from.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .artifact_cancellation import CancellationProbe, poll_cancellation, raise_if_cancelled


TEXTURE_ATLAS_KIND = "wavefront_obj_corner_uv/v1"
#: How a texel's RGB encodes the normal.  All are object-space, (n + 1) / 2
#: in 8 bits; they differ in which axes the baker mirrored.  A bake made in a
#: left-handed tool comes out with x reversed against the OBJ's frame, and
#: nothing in the file says so - the museum's map is one such.  The choice is
#: the recipe's, and ``rank_normal_map_encodings`` measures each candidate
#: by how well its slopes integrate, so the drafter chooses with numbers.
NORMAL_MAP_ENCODING = "object_space_rgb8/v1"
NORMAL_MAP_ENCODING_X_FLIPPED = "object_space_rgb8_x_flipped/v1"
NORMAL_MAP_ENCODING_Y_FLIPPED = "object_space_rgb8_y_flipped/v1"
NORMAL_MAP_ENCODING_XY_FLIPPED = "object_space_rgb8_xy_flipped/v1"
NORMAL_MAP_ENCODINGS: tuple[str, ...] = (
    NORMAL_MAP_ENCODING,
    NORMAL_MAP_ENCODING_X_FLIPPED,
    NORMAL_MAP_ENCODING_Y_FLIPPED,
    NORMAL_MAP_ENCODING_XY_FLIPPED,
)
_ENCODING_SIGNS: dict[str, tuple[float, float, float]] = {
    NORMAL_MAP_ENCODING: (1.0, 1.0, 1.0),
    NORMAL_MAP_ENCODING_X_FLIPPED: (-1.0, 1.0, 1.0),
    NORMAL_MAP_ENCODING_Y_FLIPPED: (1.0, -1.0, 1.0),
    NORMAL_MAP_ENCODING_XY_FLIPPED: (-1.0, -1.0, 1.0),
}
TEXTURE_RELIEF_INTEGRATION = "frankot_chellappa_on_developed_raster/v1"
TEXTURE_RELIEF_BASE = "sampled_normal_gaussian_smoothed/v1"
TEXTURE_RELIEF_DEPTH_MEASURE = "texture_normal_map_height/v1"

#: Vertices closer than this are one vertex when the OBJ is welded, and a
#: mesh vertex must lie within it of the atlas vertex it stands for.
ATLAS_WELD_MM = 1e-3
MAX_ATLAS_BYTES = 2 * 1024 * 1024 * 1024
MAX_ATLAS_TRIANGLES = 2_000_000
MAX_NORMAL_MAP_SIDE = 16_384
MAX_NORMAL_MAP_BYTES = 1024 * 1024 * 1024
MIN_TEXTURE_RELIEF_SMOOTHING_UM = 100
MAX_TEXTURE_RELIEF_SMOOTHING_UM = 50_000
DEFAULT_TEXTURE_RELIEF_SMOOTHING_UM = 1_000
#: The row block the rasteriser works in, as the depth rasteriser does.
_ROW_BLOCK = 128


class ArtifactTextureReliefError(ValueError):
    """Texture relief cannot be read safely from these files and this mesh."""


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class TextureAtlas:
    """An OBJ's geometry welded by position, with the texture coordinate at
    every triangle corner.

    ``vertices`` are the welded positions in the file's own frame and units;
    ``triangles`` index them; ``corner_uv`` holds, for each triangle, the
    (u, v) of its three corners in the same order.  A vertex on a texture
    seam keeps one position and different corner coordinates, which is why
    the coordinates live on corners and not on vertices.
    """

    vertices: np.ndarray
    triangles: np.ndarray
    corner_uv: np.ndarray
    sha256: str
    byte_length: int

    def __post_init__(self) -> None:
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ArtifactTextureReliefError("atlas vertices must be (n, 3)")
        if self.triangles.ndim != 2 or self.triangles.shape[1] != 3:
            raise ArtifactTextureReliefError("atlas triangles must be (m, 3)")
        if self.corner_uv.shape != (self.triangles.shape[0], 3, 2):
            raise ArtifactTextureReliefError("atlas corner_uv must be (m, 3, 2)")
        if not (isinstance(self.sha256, str) and len(self.sha256) == 64):
            raise ArtifactTextureReliefError("atlas sha256 must be 64 hex characters")

    @property
    def triangle_count(self) -> int:
        return int(self.triangles.shape[0])

    def recipe_block(self) -> dict[str, Any]:
        return {
            "byte_length": int(self.byte_length),
            "kind": TEXTURE_ATLAS_KIND,
            "sha256": self.sha256,
            "triangle_count": self.triangle_count,
            "vertex_count": int(self.vertices.shape[0]),
        }


def read_obj_texture_atlas(path: str | Path) -> TextureAtlas:
    """Read an OBJ's positions and texture coordinates, welded by position.

    Polygons are fanned into triangles from their first corner, in file
    order, so the triangle order is a function of the file alone.  Every
    corner must carry a texture coordinate; a corner without one is refused,
    since a relief cannot be read where the surface wears no texel.
    Normals, materials and groups are ignored.
    """

    source = Path(path)
    if not source.is_file():
        raise ArtifactTextureReliefError(f"texture atlas is not a file: {source}")
    byte_length = source.stat().st_size
    if byte_length > MAX_ATLAS_BYTES:
        raise ArtifactTextureReliefError("texture atlas exceeds the size limit")
    positions: list[list[float]] = []
    texcoords: list[list[float]] = []
    corners: list[tuple[int, int, int, int, int, int]] = []
    with source.open("r", encoding="utf-8", errors="strict") as handle:
        for number, line in enumerate(handle, start=1):
            if line.startswith("v "):
                parts = line.split()
                if len(parts) < 4:
                    raise ArtifactTextureReliefError(f"OBJ line {number}: vertex needs x y z")
                positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("vt "):
                parts = line.split()
                if len(parts) < 3:
                    raise ArtifactTextureReliefError(f"OBJ line {number}: vt needs u v")
                texcoords.append([float(parts[1]), float(parts[2])])
            elif line.startswith("f "):
                refs: list[tuple[int, int]] = []
                for token in line.split()[1:]:
                    fields = token.split("/")
                    if len(fields) < 2 or not fields[0] or not fields[1]:
                        raise ArtifactTextureReliefError(
                            f"OBJ line {number}: a face corner has no texture coordinate"
                        )
                    refs.append((int(fields[0]), int(fields[1])))
                if len(refs) < 3:
                    raise ArtifactTextureReliefError(f"OBJ line {number}: a face needs three corners")
                for k in range(1, len(refs) - 1):
                    corners.append((*refs[0], *refs[k], *refs[k + 1]))
                if len(corners) > MAX_ATLAS_TRIANGLES:
                    raise ArtifactTextureReliefError("texture atlas exceeds the triangle limit")
    if not positions or not corners:
        raise ArtifactTextureReliefError("texture atlas has no textured faces")
    vertices = np.asarray(positions, dtype=np.float64)
    uv = np.asarray(texcoords, dtype=np.float64)
    refs_array = np.asarray(corners, dtype=np.int64).reshape(-1, 3, 2)

    def resolve(index: np.ndarray, count: int, name: str) -> np.ndarray:
        # OBJ indices are 1-based; negative ones count from the end.
        resolved = np.where(index > 0, index - 1, count + index)
        if (resolved < 0).any() or (resolved >= count).any():
            raise ArtifactTextureReliefError(f"OBJ face refers to a {name} that does not exist")
        return resolved

    vertex_index = resolve(refs_array[:, :, 0], vertices.shape[0], "vertex")
    uv_index = resolve(refs_array[:, :, 1], uv.shape[0], "texture coordinate")
    if not np.isfinite(vertices).all() or not np.isfinite(uv).all():
        raise ArtifactTextureReliefError("texture atlas contains non-finite coordinates")
    keys = np.round(vertices / ATLAS_WELD_MM).astype(np.int64)
    _unique, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    welded = vertices[np.sort(first)]
    # Renumber so welded vertex order follows first appearance in the file.
    order = np.argsort(first, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(order.shape[0])
    triangles = rank[inverse[vertex_index]]
    degenerate = (
        (triangles[:, 0] == triangles[:, 1])
        | (triangles[:, 1] == triangles[:, 2])
        | (triangles[:, 0] == triangles[:, 2])
    )
    triangles = triangles[~degenerate]
    corner_uv = uv[uv_index][~degenerate]
    return TextureAtlas(
        vertices=welded,
        triangles=triangles,
        corner_uv=corner_uv,
        sha256=_sha256_of_file(source),
        byte_length=int(byte_length),
    )


def write_atlas_geometry(atlas: TextureAtlas, path: str | Path) -> Path:
    """Write the atlas's welded geometry as a binary triangle PLY the loader
    opens: the mesh a session must hold for this atlas to be read onto it.

    Positions are written as float32, so they stray from the atlas by less
    than the weld distance; triangles keep the atlas's order, which is how a
    developed face finds its texture corners.
    """

    import trimesh  # noqa: PLC0415

    target = Path(path)
    mesh = trimesh.Trimesh(atlas.vertices, atlas.triangles, process=False)
    mesh.export(target, file_type="ply", encoding="binary")
    return target


def require_atlas_matches(
    atlas: TextureAtlas,
    source_vertices: object,
    source_faces: object,
) -> None:
    """Prove the session's source mesh is the atlas's geometry: the same
    triangles, in the same order, each corner within the weld distance."""

    vertices = np.asarray(source_vertices, dtype=np.float64)
    faces = np.asarray(source_faces, dtype=np.int64)
    if faces.shape != atlas.triangles.shape:
        raise ArtifactTextureReliefError(
            f"texture atlas has {atlas.triangle_count} triangles but the mesh has "
            f"{int(faces.shape[0])}; the mesh must be the atlas's geometry welded by position"
        )
    if vertices.shape[0] != atlas.vertices.shape[0]:
        raise ArtifactTextureReliefError(
            f"texture atlas has {int(atlas.vertices.shape[0])} welded vertices but the mesh "
            f"has {int(vertices.shape[0])}"
        )
    gap = np.abs(vertices[faces] - atlas.vertices[atlas.triangles]).max()
    if not np.isfinite(gap) or gap > ATLAS_WELD_MM:
        raise ArtifactTextureReliefError(
            f"mesh corners stray {gap:.4f} mm from the texture atlas; the mesh is not this OBJ"
        )


@dataclass(frozen=True, slots=True)
class NormalMap:
    """An object-space normal map: each texel's RGB is the surface normal,
    (n + 1) / 2 in 8 bits, in the atlas's own frame, under ``encoding``."""

    rgb: np.ndarray
    sha256: str
    byte_length: int
    encoding: str = NORMAL_MAP_ENCODING

    def __post_init__(self) -> None:
        if self.rgb.ndim != 3 or self.rgb.shape[2] != 3 or self.rgb.dtype != np.uint8:
            raise ArtifactTextureReliefError("normal map must be an (h, w, 3) uint8 image")
        if not (isinstance(self.sha256, str) and len(self.sha256) == 64):
            raise ArtifactTextureReliefError("normal map sha256 must be 64 hex characters")
        if self.encoding not in NORMAL_MAP_ENCODINGS:
            raise ArtifactTextureReliefError(
                f"normal map encoding must be one of {', '.join(NORMAL_MAP_ENCODINGS)}"
            )

    def with_encoding(self, encoding: str) -> "NormalMap":
        """The same image read under another convention."""

        return NormalMap(
            rgb=self.rgb, sha256=self.sha256, byte_length=self.byte_length, encoding=encoding
        )

    @property
    def height(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def width(self) -> int:
        return int(self.rgb.shape[1])

    def recipe_block(self) -> dict[str, Any]:
        return {
            "byte_length": int(self.byte_length),
            "encoding": self.encoding,
            "height": self.height,
            "sha256": self.sha256,
            "width": self.width,
        }


def read_normal_map(path: str | Path, *, encoding: str = NORMAL_MAP_ENCODING) -> NormalMap:
    """Decode a normal map image to RGB bytes, with the file's own hash."""

    from PIL import Image  # noqa: PLC0415

    source = Path(path)
    if not source.is_file():
        raise ArtifactTextureReliefError(f"normal map is not a file: {source}")
    byte_length = source.stat().st_size
    if byte_length > MAX_NORMAL_MAP_BYTES:
        raise ArtifactTextureReliefError("normal map exceeds the size limit")
    with Image.open(source) as image:
        if image.width > MAX_NORMAL_MAP_SIDE or image.height > MAX_NORMAL_MAP_SIDE:
            raise ArtifactTextureReliefError("normal map exceeds the side limit")
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return NormalMap(
        rgb=np.ascontiguousarray(rgb),
        sha256=_sha256_of_file(source),
        byte_length=int(byte_length),
        encoding=encoding,
    )


def texture_relief_block(
    atlas: TextureAtlas,
    normal_map: NormalMap,
    *,
    smoothing_um: int = DEFAULT_TEXTURE_RELIEF_SMOOTHING_UM,
) -> dict[str, Any]:
    """The recipe block that names the two files and the one number."""

    if isinstance(smoothing_um, bool) or not isinstance(smoothing_um, (int, np.integer)):
        raise ArtifactTextureReliefError("smoothing_um must be an integer")
    smoothing = int(smoothing_um)
    if not MIN_TEXTURE_RELIEF_SMOOTHING_UM <= smoothing <= MAX_TEXTURE_RELIEF_SMOOTHING_UM:
        raise ArtifactTextureReliefError(
            f"smoothing_um must be in the inclusive range "
            f"{MIN_TEXTURE_RELIEF_SMOOTHING_UM}..{MAX_TEXTURE_RELIEF_SMOOTHING_UM}"
        )
    return {
        "atlas": atlas.recipe_block(),
        "base_normal": TEXTURE_RELIEF_BASE,
        "integration": TEXTURE_RELIEF_INTEGRATION,
        "normal_map": normal_map.recipe_block(),
        "sampling": "nearest_texel/v1",
        "smoothing_um": smoothing,
    }


def validate_texture_relief_block(value: object) -> dict[str, Any]:
    """Check the block's shape and ranges without the files at hand."""

    if not isinstance(value, Mapping):
        raise ArtifactTextureReliefError("texture_relief must be an object")
    expected_keys = {"atlas", "base_normal", "integration", "normal_map", "sampling", "smoothing_um"}
    if set(value) != expected_keys:
        raise ArtifactTextureReliefError(
            f"texture_relief must carry exactly {', '.join(sorted(expected_keys))}"
        )
    if value["base_normal"] != TEXTURE_RELIEF_BASE or value["integration"] != TEXTURE_RELIEF_INTEGRATION:
        raise ArtifactTextureReliefError("texture_relief names a method this release does not have")
    if value["sampling"] != "nearest_texel/v1":
        raise ArtifactTextureReliefError("texture_relief names a sampling this release does not have")
    smoothing = value["smoothing_um"]
    if isinstance(smoothing, bool) or not isinstance(smoothing, int):
        raise ArtifactTextureReliefError("texture_relief smoothing_um must be an integer")
    if not MIN_TEXTURE_RELIEF_SMOOTHING_UM <= smoothing <= MAX_TEXTURE_RELIEF_SMOOTHING_UM:
        raise ArtifactTextureReliefError("texture_relief smoothing_um is out of range")
    atlas = value["atlas"]
    if not isinstance(atlas, Mapping) or set(atlas) != {
        "byte_length",
        "kind",
        "sha256",
        "triangle_count",
        "vertex_count",
    }:
        raise ArtifactTextureReliefError("texture_relief atlas block is malformed")
    if atlas["kind"] != TEXTURE_ATLAS_KIND:
        raise ArtifactTextureReliefError("texture_relief names an atlas kind this release does not have")
    normal_map = value["normal_map"]
    if not isinstance(normal_map, Mapping) or set(normal_map) != {
        "byte_length",
        "encoding",
        "height",
        "sha256",
        "width",
    }:
        raise ArtifactTextureReliefError("texture_relief normal_map block is malformed")
    if normal_map["encoding"] not in NORMAL_MAP_ENCODINGS:
        raise ArtifactTextureReliefError(
            "texture_relief names a normal map encoding this release does not have"
        )
    for block in (atlas, normal_map):
        sha = block["sha256"]
        if not (isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)):
            raise ArtifactTextureReliefError("texture_relief sha256 must be 64 lowercase hex characters")
        for key in block:
            if key in ("kind", "encoding", "sha256"):
                continue
            number = block[key]
            if isinstance(number, bool) or not isinstance(number, int) or number < 0:
                raise ArtifactTextureReliefError(f"texture_relief {key} must be a non-negative integer")
    return {
        "atlas": dict(atlas),
        "base_normal": TEXTURE_RELIEF_BASE,
        "integration": TEXTURE_RELIEF_INTEGRATION,
        "normal_map": dict(normal_map),
        "sampling": "nearest_texel/v1",
        "smoothing_um": int(smoothing),
    }


def require_texture_relief_sources(
    block: Mapping[str, Any],
    atlas: TextureAtlas,
    normal_map: NormalMap,
) -> None:
    """The files at hand must be the files the recipe names."""

    if atlas.sha256 != block["atlas"]["sha256"] or atlas.byte_length != int(block["atlas"]["byte_length"]):
        raise ArtifactTextureReliefError(
            "the texture atlas at hand is not the one the recipe names "
            f"(recipe sha256 {block['atlas']['sha256'][:12]}..., file {atlas.sha256[:12]}...)"
        )
    if atlas.triangle_count != int(block["atlas"]["triangle_count"]) or int(
        atlas.vertices.shape[0]
    ) != int(block["atlas"]["vertex_count"]):
        raise ArtifactTextureReliefError("the texture atlas at hand welds to different counts")
    if normal_map.sha256 != block["normal_map"]["sha256"] or normal_map.byte_length != int(
        block["normal_map"]["byte_length"]
    ):
        raise ArtifactTextureReliefError(
            "the normal map at hand is not the one the recipe names "
            f"(recipe sha256 {block['normal_map']['sha256'][:12]}..., file {normal_map.sha256[:12]}...)"
        )
    if normal_map.width != int(block["normal_map"]["width"]) or normal_map.height != int(
        block["normal_map"]["height"]
    ):
        raise ArtifactTextureReliefError("the normal map at hand has different dimensions")
    if normal_map.encoding != block["normal_map"]["encoding"]:
        raise ArtifactTextureReliefError(
            f"the normal map at hand is read as {normal_map.encoding}; the recipe says "
            f"{block['normal_map']['encoding']}"
        )


def rigid_rotation_between(source_points: np.ndarray, canonical_points: np.ndarray) -> np.ndarray:
    """The rotation that carries source directions into canonical ones, fitted
    to matching points and required to be rigid within the weld distance."""

    if source_points.shape != canonical_points.shape or source_points.shape[0] < 4:
        raise ArtifactTextureReliefError("rotation needs at least four matching points")
    source_centre = source_points.mean(axis=0)
    canonical_centre = canonical_points.mean(axis=0)
    covariance = (canonical_points - canonical_centre).T @ (source_points - source_centre)
    u, _s, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    fitted = (source_points - source_centre) @ rotation.T + canonical_centre
    gap = float(np.abs(fitted - canonical_points).max())
    if gap > 0.05:
        raise ArtifactTextureReliefError(
            f"the mesh is not a rigid motion of the atlas: points stray {gap:.3f} mm"
        )
    return rotation


def texture_relief_depth_field(
    *,
    developed_uv_mm: np.ndarray,
    developed_faces: np.ndarray,
    developed_points_mm: np.ndarray,
    source_face_indices: np.ndarray,
    source_vertex_indices: np.ndarray,
    atlas: TextureAtlas,
    normal_map: NormalMap,
    source_to_canonical_rotation: np.ndarray,
    pixels_per_mm: int,
    margin_pixels: int,
    smoothing_um: int,
    cancellation_probe: CancellationProbe | None = None,
) -> tuple[np.ndarray, int, int, dict[str, Any]]:
    """The height the normal map implies at every pixel centre of the
    developed lattice, in millimetres, as a depth field the relief renderer
    takes: finite where the development covers the pixel, -inf elsewhere.

    The lattice is the one the mesh-depth rubbing uses - same origin, same
    margin, same pixel-centre rule - so the two draw on the same paper.  At
    each pixel the developed triangle's barycentric weights give the texel;
    the texel's object-space normal, turned into the canonical frame, minus
    the same normal smoothed over ``smoothing_um`` across the raster, is a
    small tilt; its components along the developed u and v axes (each
    triangle's own world directions of those axes) are the slopes of the
    height, and the slopes are integrated over the whole raster by the
    Fourier method of Frankot and Chellappa.  The smoothing takes out what
    the coarse mesh and the map disagree about at large; what is left is
    the relief the mesh does not have.
    """

    from scipy.ndimage import gaussian_filter  # noqa: PLC0415

    raise_if_cancelled(cancellation_probe)
    uv = np.asarray(developed_uv_mm, dtype=np.float64)
    faces = np.asarray(developed_faces, dtype=np.int64)
    points = np.asarray(developed_points_mm, dtype=np.float64)
    source_faces = np.asarray(source_face_indices, dtype=np.int64)
    source_vertices = np.asarray(source_vertex_indices, dtype=np.int64)
    rotation = np.asarray(source_to_canonical_rotation, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[1] != 2 or points.shape != (uv.shape[0], 3):
        raise ArtifactTextureReliefError("developed coordinates and points must match")
    if faces.ndim != 2 or faces.shape[1] != 3 or source_faces.shape != (faces.shape[0],):
        raise ArtifactTextureReliefError("developed faces need one source face each")
    if source_vertices.shape != (uv.shape[0],):
        raise ArtifactTextureReliefError("developed vertices need one source vertex each")
    if rotation.shape != (3, 3):
        raise ArtifactTextureReliefError("rotation must be 3 x 3")
    if int(pixels_per_mm) <= 0:
        raise ArtifactTextureReliefError("pixels_per_mm must be positive")
    if (source_faces < 0).any() or (source_faces >= atlas.triangle_count).any():
        raise ArtifactTextureReliefError("a developed face refers to a triangle the atlas lacks")

    scaled = uv * float(pixels_per_mm)
    if not np.isfinite(scaled).all():
        raise ArtifactTextureReliefError("developed coordinates are not finite")
    minimum_u = math.floor(float(scaled[:, 0].min())) - int(margin_pixels)
    maximum_u = math.ceil(float(scaled[:, 0].max())) + int(margin_pixels)
    minimum_v = math.floor(float(scaled[:, 1].min())) - int(margin_pixels)
    maximum_v = math.ceil(float(scaled[:, 1].max())) + int(margin_pixels)
    width = maximum_u - minimum_u
    height = maximum_v - minimum_v
    if width <= 0 or height <= 0:
        raise ArtifactTextureReliefError("developed raster has zero extent")
    if width * height > 8_000_000:
        raise ArtifactTextureReliefError("developed raster exceeds the pixel limit")
    local = scaled - np.array([minimum_u, minimum_v], dtype=np.float64)

    map_h, map_w = normal_map.height, normal_map.width
    signs = np.asarray(_ENCODING_SIGNS[normal_map.encoding], dtype=np.float64)
    sampled = np.zeros((height, width, 3), dtype=np.float64)
    axis_u = np.zeros((height, width, 3), dtype=np.float64)
    axis_v = np.zeros((height, width, 3), dtype=np.float64)
    covered = np.zeros((height, width), dtype=bool)
    unmatched_corners = 0
    epsilon = 1e-12
    for face_index in range(faces.shape[0]):
        poll_cancellation(cancellation_probe, face_index)
        corners = faces[face_index]
        triangle = local[corners]
        world = points[corners]
        # The atlas triangle this developed face came from, its corners matched
        # to ours by the welded vertex they share.
        atlas_face = int(source_faces[face_index])
        atlas_corners = atlas.triangles[atlas_face]
        corner_uv = np.zeros((3, 2), dtype=np.float64)
        for slot in range(3):
            hit = np.flatnonzero(atlas_corners == source_vertices[corners[slot]])
            if hit.size == 0:
                unmatched_corners += 1
                corner_uv[slot] = atlas.corner_uv[atlas_face, slot]
            else:
                corner_uv[slot] = atlas.corner_uv[atlas_face, int(hit[0])]
        ax, ay = float(triangle[0, 0]), float(triangle[0, 1])
        bx, by = float(triangle[1, 0]), float(triangle[1, 1])
        cx, cy = float(triangle[2, 0]), float(triangle[2, 1])
        denominator = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if denominator == 0.0:
            continue
        # World directions of the developed axes over this triangle.
        basis = np.vstack([triangle[1] - triangle[0], triangle[2] - triangle[0]]) / float(pixels_per_mm)
        jacobian, _res, rank, _sv = np.linalg.lstsq(basis, np.vstack([world[1] - world[0], world[2] - world[0]]), rcond=None)
        if rank < 2:
            continue
        e_u = jacobian[0] / max(float(np.linalg.norm(jacobian[0])), 1e-12)
        e_v = jacobian[1] / max(float(np.linalg.norm(jacobian[1])), 1e-12)
        minimum_x = max(0, int(math.floor(min(ax, bx, cx) - 0.5)))
        maximum_x = min(width - 1, int(math.ceil(max(ax, bx, cx) - 0.5)))
        minimum_y = max(0, int(math.floor(min(ay, by, cy) - 0.5)))
        maximum_y = min(height - 1, int(math.ceil(max(ay, by, cy) - 0.5)))
        if minimum_x > maximum_x or minimum_y > maximum_y:
            continue
        xs = np.arange(minimum_x, maximum_x + 1, dtype=np.float64) + 0.5
        for y_start in range(minimum_y, maximum_y + 1, _ROW_BLOCK):
            y_stop = min(maximum_y + 1, y_start + _ROW_BLOCK)
            ys = np.arange(y_start, y_stop, dtype=np.float64)[:, None] + 0.5
            w0 = ((by - cy) * (xs[None, :] - cx) + (cx - bx) * (ys - cy)) / denominator
            w1 = ((cy - ay) * (xs[None, :] - cx) + (ax - cx) * (ys - cy)) / denominator
            w2 = 1.0 - w0 - w1
            inside = (w0 >= -epsilon) & (w1 >= -epsilon) & (w2 >= -epsilon)
            if not bool(np.any(inside)):
                continue
            tex_u = w0 * corner_uv[0, 0] + w1 * corner_uv[1, 0] + w2 * corner_uv[2, 0]
            tex_v = w0 * corner_uv[0, 1] + w1 * corner_uv[1, 1] + w2 * corner_uv[2, 1]
            # Texture v runs up the image; row 0 of the array is the top.
            col = np.clip(np.floor(tex_u * map_w).astype(np.int64), 0, map_w - 1)
            row = np.clip(np.floor((1.0 - tex_v) * map_h).astype(np.int64), 0, map_h - 1)
            normals = (normal_map.rgb[row, col].astype(np.float64) / 127.5 - 1.0) * signs
            normals = normals @ rotation.T
            block = sampled[y_start:y_stop, minimum_x : maximum_x + 1]
            block[inside] = normals[inside]
            axis_u[y_start:y_stop, minimum_x : maximum_x + 1][inside] = e_u
            axis_v[y_start:y_stop, minimum_x : maximum_x + 1][inside] = e_v
            covered[y_start:y_stop, minimum_x : maximum_x + 1] |= inside
    raise_if_cancelled(cancellation_probe)
    covered_count = int(np.count_nonzero(covered))
    if covered_count == 0:
        raise ArtifactTextureReliefError("the development covers no pixel centre")
    lengths = np.linalg.norm(sampled, axis=-1)
    valid = covered & (lengths > 0.5)
    sampled[valid] /= lengths[valid][:, None]
    sampled[~valid] = 0.0

    sigma_px = float(smoothing_um) / 1000.0 * float(pixels_per_mm)
    weight = gaussian_filter(valid.astype(np.float64), sigma_px)
    base = np.empty_like(sampled)
    for channel in range(3):
        base[..., channel] = gaussian_filter(sampled[..., channel], sigma_px) / np.maximum(weight, 1e-9)
    base_length = np.linalg.norm(base, axis=-1)
    good = valid & (base_length > 0.5)
    base[good] /= base_length[good][:, None]
    tilt = np.where(good[..., None], sampled - base, 0.0)
    # Small-slope relation: the surface normal is the base normal less the
    # gradient of the height, so the height's slope along an axis is minus
    # the tilt's component along it.
    slope_u = -np.einsum("ijk,ijk->ij", tilt, axis_u)
    slope_v = -np.einsum("ijk,ijk->ij", tilt, axis_v)
    raise_if_cancelled(cancellation_probe)
    # Integrate on the lattice in pixel steps.  Row index grows with v here
    # (row 0 is the lowest v, as in the mesh-depth rasteriser; the raster
    # is turned top-to-bottom only when it is packed), so the row derivative
    # is the v slope itself.
    p = slope_u
    q = slope_v
    wx = np.fft.fftfreq(width) * 2.0 * np.pi
    wy = np.fft.fftfreq(height) * 2.0 * np.pi
    grid_x, grid_y = np.meshgrid(wx, wy)
    denominator_k = grid_x**2 + grid_y**2
    denominator_k[0, 0] = 1.0
    spectrum = (-1j * grid_x * np.fft.fft2(p) - 1j * grid_y * np.fft.fft2(q)) / denominator_k
    spectrum[0, 0] = 0.0
    height_px = np.real(np.fft.ifft2(spectrum))
    height_mm = height_px / float(pixels_per_mm)
    raise_if_cancelled(cancellation_probe)
    if good.any():
        height_mm -= float(np.median(height_mm[good]))
    residual_energy = float(np.sum(p[good] ** 2 + q[good] ** 2))
    gy, gx = np.gradient(height_px)
    misfit = float(np.sum((gx - p)[good] ** 2 + (gy - q)[good] ** 2))
    depth = np.full((height, width), -np.inf, dtype=np.float64)
    depth[good] = height_mm[good]
    qc: dict[str, Any] = {
        "texture_relief_covered_pixel_count": covered_count,
        "texture_relief_height_max_um_rounded": int(round(float(height_mm[good].max()) * 1000.0)) if good.any() else 0,
        "texture_relief_height_min_um_rounded": int(round(float(height_mm[good].min()) * 1000.0)) if good.any() else 0,
        "texture_relief_integration_misfit_millionths": int(
            round(1_000_000.0 * misfit / residual_energy)
        )
        if residual_energy > 0.0
        else 0,
        "texture_relief_unmatched_corner_count": unmatched_corners,
        "texture_relief_unreadable_pixel_count": int(np.count_nonzero(covered & ~good)),
    }
    return depth, minimum_u, minimum_v, qc


def rank_normal_map_encodings(
    *,
    developed_uv_mm: np.ndarray,
    developed_faces: np.ndarray,
    developed_points_mm: np.ndarray,
    source_face_indices: np.ndarray,
    source_vertex_indices: np.ndarray,
    atlas: TextureAtlas,
    normal_map: NormalMap,
    source_to_canonical_rotation: np.ndarray,
    pixels_per_mm: int,
    smoothing_um: int,
    cancellation_probe: CancellationProbe | None = None,
) -> dict[str, int]:
    """The integration misfit, in millionths, of each encoding the map could
    be read under, on this development.

    A slope field that came from a real height integrates to one: the
    misfit is small.  Read with an axis mirrored, the same field does not,
    and the misfit says so.  The numbers are for the drafter to choose by
    and to record; nothing here chooses.
    """

    misfits: dict[str, int] = {}
    for encoding in NORMAL_MAP_ENCODINGS:
        raise_if_cancelled(cancellation_probe)
        _depth, _u, _v, qc = texture_relief_depth_field(
            developed_uv_mm=developed_uv_mm,
            developed_faces=developed_faces,
            developed_points_mm=developed_points_mm,
            source_face_indices=source_face_indices,
            source_vertex_indices=source_vertex_indices,
            atlas=atlas,
            normal_map=normal_map.with_encoding(encoding),
            source_to_canonical_rotation=source_to_canonical_rotation,
            pixels_per_mm=pixels_per_mm,
            margin_pixels=0,
            smoothing_um=smoothing_um,
            cancellation_probe=cancellation_probe,
        )
        misfits[encoding] = int(qc["texture_relief_integration_misfit_millionths"])
    return misfits


__all__ = [
    "ATLAS_WELD_MM",
    "DEFAULT_TEXTURE_RELIEF_SMOOTHING_UM",
    "MAX_TEXTURE_RELIEF_SMOOTHING_UM",
    "MIN_TEXTURE_RELIEF_SMOOTHING_UM",
    "NORMAL_MAP_ENCODING",
    "NORMAL_MAP_ENCODINGS",
    "NORMAL_MAP_ENCODING_XY_FLIPPED",
    "NORMAL_MAP_ENCODING_X_FLIPPED",
    "NORMAL_MAP_ENCODING_Y_FLIPPED",
    "TEXTURE_ATLAS_KIND",
    "TEXTURE_RELIEF_BASE",
    "TEXTURE_RELIEF_DEPTH_MEASURE",
    "TEXTURE_RELIEF_INTEGRATION",
    "ArtifactTextureReliefError",
    "NormalMap",
    "TextureAtlas",
    "read_normal_map",
    "rank_normal_map_encodings",
    "read_obj_texture_atlas",
    "require_atlas_matches",
    "require_texture_relief_sources",
    "rigid_rotation_between",
    "texture_relief_block",
    "texture_relief_depth_field",
    "validate_texture_relief_block",
    "write_atlas_geometry",
]
