"""뗀석기: a flaked stone tool as a closed solid, for tests that need one.

Everything the program was built on so far turns on an axis - a pot, a
tile's cylinder.  A stone tool does not.  The guidelines put it flat on the
table in the direction it was used ([K2] 2014, 석기 실측 방법: an arrowhead
tang down, a hoe blade down), draw the plan, and cut a long and a cross
section - or, better, draw all six projections in third-angle ([K2]), and
a dagger always with its side view.  On the plan the flake scars are drawn
as inner lines (내선, [K1] 2013 p. 48): the ridges where one scar meets the
next are what tell a reader how the tool was made.

This tool is a leaf-shaped flake worked on one face.  Its dorsal face is
the lower envelope of a handful of planes, so the scars are flat and the
ridges between them are sharp, and its ventral face is one smooth surface
with the bulb of percussion swelling near the platform end.  The two meet
at the margin.  Sizes are those of a hand-sized biface, not a copy of any
particular tool, and nothing here is random.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from src.core.artifact_session import ArtifactSession
from src.core.mesh_import_recipe import current_mesh_import_recipe
from src.core.mesh_loader import MeshData
from src.core.source_identity import SourceFingerprint


@dataclass(frozen=True, slots=True)
class Facet:
    """One flake scar on the dorsal face: a plane, in millimetres.

    ``height_mm`` is the plane's height over the centre of the plan and
    ``slope`` how steeply it falls, per millimetre, in the direction
    ``direction_deg`` measured in the plan.
    """

    height_mm: float
    slope: float
    direction_deg: float


@dataclass(frozen=True, slots=True)
class LithicShape:
    #: Half the length, along +X, and the plan's fullness: 0 is an ellipse
    #: and more makes the platform end (-X) broader than the tip (+X).
    half_length_mm: float = 42.0
    half_width_mm: float = 28.0
    taper: float = 0.28
    #: The ventral face's greatest depth below the margin, and the bulb's.
    ventral_depth_mm: float = 5.0
    bulb_mm: float = 2.5
    #: How far the ridges are rounded over.  0 keeps them sharp, as planes
    #: meeting at an edge; a real scan's ridge is rounded over a millimetre
    #: or two by the stone and by the scanner, and this is that.
    rounding_mm: float = 0.0
    #: The dorsal scars.  Their lower envelope, tapered to nothing at the
    #: margin, is the dorsal face.
    #: Steep enough that the ridges between them bend by 45-60 degrees, as
    #: the ridges of a worked biface do; a scar's plane and its neighbour's
    #: meet at twice the arctangent of their slopes.
    #: The two lowest planes fall away to either side, so they meet along
    #: the tool's length in a central ridge, as a biface's do.
    facets: tuple[Facet, ...] = (
        Facet(height_mm=15.5, slope=0.55, direction_deg=90.0),
        Facet(height_mm=15.5, slope=0.55, direction_deg=-90.0),
        Facet(height_mm=22.0, slope=0.38, direction_deg=0.0),
        Facet(height_mm=21.0, slope=0.35, direction_deg=180.0),
        Facet(height_mm=21.0, slope=0.60, direction_deg=45.0),
        Facet(height_mm=23.0, slope=0.60, direction_deg=-135.0),
    )

    def __post_init__(self) -> None:
        for name in ("half_length_mm", "half_width_mm", "ventral_depth_mm"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not (0.0 <= self.taper < 0.6):
            raise ValueError("taper must be between 0 and 0.6")
        if not self.facets:
            raise ValueError("a flaked tool has at least one scar")
        if self.rounding_mm < 0.0:
            raise ValueError("rounding_mm cannot be negative")
        # The scars must stay above the margin all the way to it: a plane
        # that dips under it would fold the dorsal face into a valley.
        for index in range(360):
            angle = math.radians(index)
            r = plan_radius(self, angle)
            if dorsal_height(self, r * math.cos(angle), r * math.sin(angle)) <= 0.0:
                raise ValueError("a scar falls below the margin before reaching it")



def plan_radius(shape: LithicShape, angle_rad: float) -> float:
    """The margin's distance from the plan's centre in one direction."""

    c, s = math.cos(angle_rad), math.sin(angle_rad)
    # An ellipse, pinched toward the tip: the platform end stays broad.
    ellipse = 1.0 / math.sqrt(
        (c / shape.half_length_mm) ** 2 + (s / shape.half_width_mm) ** 2
    )
    return ellipse * (1.0 - shape.taper * 0.5 * (1.0 + c) * abs(s))


def dorsal_height(shape: LithicShape, x_mm: float, y_mm: float) -> float:
    """The lower envelope of the scars over one point of the plan.

    Each scar is a plane falling away in its own direction; the stone is
    under all of them, so the face is the lowest plane at each point - a
    dome, whose creases stand out.
    """

    heights = []
    for facet in shape.facets:
        along = x_mm * math.cos(math.radians(facet.direction_deg)) + y_mm * math.sin(
            math.radians(facet.direction_deg)
        )
        heights.append(facet.height_mm - facet.slope * along)
    lowest = min(heights)
    if shape.rounding_mm <= 0.0:
        return lowest
    # A soft minimum: the planes blend over about rounding_mm either side of
    # where they meet, and the crest stays on the line where they would have.
    k = shape.rounding_mm * 0.5
    return lowest - k * math.log(sum(math.exp(-(h - lowest) / k) for h in heights))


def ventral_depth(shape: LithicShape, x_mm: float, y_mm: float) -> float:
    """How far below the margin the ventral face lies at one point."""

    bulb_centre = (-shape.half_length_mm * 0.55, 0.0)
    spread = shape.half_width_mm * 0.45
    reach = ((x_mm - bulb_centre[0]) ** 2 + (y_mm - bulb_centre[1]) ** 2) / (
        2.0 * spread * spread
    )
    return shape.ventral_depth_mm + shape.bulb_mm * math.exp(-reach)


#: Defined after the surface functions its own check calls.
BIFACE_SHAPE = LithicShape()


Cell = tuple[list[tuple[float, float]], list[bool]]


def _dorsal_cells(shape: LithicShape, *, segments: int) -> list[Cell]:
    """The plan cut into one region per scar.

    A scar's region is where its plane is the lowest: the margin polygon
    clipped by one half-plane per other scar.  Each cell is its vertices in
    order and, for the edge leaving each vertex, whether that edge is a piece
    of the margin (True) or a crease shared with the neighbouring scar.
    """

    margin: list[tuple[float, float]] = []
    for segment in range(segments):
        angle = 2.0 * math.pi * segment / segments
        r = plan_radius(shape, angle)
        margin.append((r * math.cos(angle), r * math.sin(angle)))

    def clip(cell: Cell, a: tuple[float, float], c: float) -> Cell:
        # Keep the side a . p <= c (Sutherland-Hodgman, one half-plane).  A
        # cut edge keeps its tag on the piece that survives; the edge that
        # closes the cut runs along the clip line and is a crease.
        polygon, tags = cell
        out_vertices: list[tuple[float, float]] = []
        out_tags: list[bool] = []
        first_in_tag: bool | None = None

        def emit(point: tuple[float, float], incoming: bool) -> None:
            nonlocal first_in_tag
            if out_vertices:
                out_tags.append(incoming)
            else:
                first_in_tag = incoming
            out_vertices.append(point)

        count = len(polygon)
        for index in range(count):
            previous, current = polygon[index - 1], polygon[index]
            tag = tags[index - 1]
            dp = a[0] * previous[0] + a[1] * previous[1] - c
            dc = a[0] * current[0] + a[1] * current[1] - c
            inside_previous, inside_current = dp <= 0.0, dc <= 0.0
            if inside_previous and inside_current:
                emit(current, tag)
            elif inside_previous and not inside_current:
                t = dp / (dp - dc)
                emit((previous[0] + t * (current[0] - previous[0]), previous[1] + t * (current[1] - previous[1])), tag)
            elif not inside_previous and inside_current:
                t = dp / (dp - dc)
                emit((previous[0] + t * (current[0] - previous[0]), previous[1] + t * (current[1] - previous[1])), False)
                emit(current, tag)
        if out_vertices:
            out_tags.append(bool(first_in_tag))
        return out_vertices, out_tags

    cells: list[Cell] = []
    for i, facet in enumerate(shape.facets):
        cell: Cell = (list(margin), [True] * len(margin))
        di = (math.cos(math.radians(facet.direction_deg)), math.sin(math.radians(facet.direction_deg)))
        for j, other in enumerate(shape.facets):
            if j == i:
                continue
            dj = (math.cos(math.radians(other.direction_deg)), math.sin(math.radians(other.direction_deg)))
            # P_i <= P_j  <=>  (s_j d_j - s_i d_i) . p <= h_j - h_i
            a = (other.slope * dj[0] - facet.slope * di[0], other.slope * dj[1] - facet.slope * di[1])
            cell = clip(cell, a, other.height_mm - facet.height_mm)
            if len(cell[0]) < 3:
                break
        # Drop repeated points; a zero-length edge is no edge.
        polygon, tags = cell
        cleaned: list[tuple[float, float]] = []
        cleaned_tags: list[bool] = []
        for point, tag in zip(polygon, tags):
            if cleaned and math.hypot(point[0] - cleaned[-1][0], point[1] - cleaned[-1][1]) < 1e-6:
                cleaned_tags[-1] = tag
                continue
            cleaned.append(point)
            cleaned_tags.append(tag)
        if len(cleaned) > 1 and math.hypot(cleaned[0][0] - cleaned[-1][0], cleaned[0][1] - cleaned[-1][1]) < 1e-6:
            cleaned.pop()
            cleaned_tags.pop()
        if len(cleaned) < 3:
            continue
        # A crease edge can run the tool's length; the margin's edges are
        # about two millimetres.  Cut the creases to the same pitch, or the
        # strips inside the cell become needles whose diagonals read as
        # creases of their own on the tapered surface.
        pitch = 2.0 * math.pi * max(shape.half_length_mm, shape.half_width_mm) / segments
        divided: list[tuple[float, float]] = []
        divided_tags: list[bool] = []
        for index, point in enumerate(cleaned):
            following = cleaned[(index + 1) % len(cleaned)]
            tag = cleaned_tags[index]
            pieces = 1
            if not tag:
                pieces = max(1, math.ceil(math.hypot(following[0] - point[0], following[1] - point[1]) / pitch))
            for k in range(pieces):
                t = k / pieces
                divided.append((point[0] + t * (following[0] - point[0]), point[1] + t * (following[1] - point[1])))
                divided_tags.append(tag)
        cells.append((divided, divided_tags))
    return cells


def _on_margin(cell: Cell, index: int) -> bool:
    """Whether a cell vertex lies on the margin: an edge either side is."""

    _polygon, tags = cell
    return bool(tags[index] or tags[index - 1])


def dorsal_creases(
    shape: LithicShape = BIFACE_SHAPE, *, segments: int = 120
) -> list[tuple[np.ndarray, np.ndarray]]:
    """The scars' ridges as 3D segments, exactly as the mesh carries them.

    Every cell edge that is not a piece of the margin is a crease shared by
    two scars; each is returned once, as a pair of endpoints.
    """

    seen: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    creases: list[tuple[np.ndarray, np.ndarray]] = []
    for cell in _dorsal_cells(shape, segments=segments):
        polygon, tags = cell
        for index, current in enumerate(polygon):
            if tags[index]:
                continue
            following = polygon[(index + 1) % len(polygon)]
            key = tuple(sorted(((round(current[0], 6), round(current[1], 6)), (round(following[0], 6), round(following[1], 6)))))
            if key in seen:
                continue
            seen.add(key)
            creases.append(
                (
                    np.asarray([current[0], current[1], _dorsal_z(shape, (current[0], current[1], _on_margin(cell, index)))]),
                    np.asarray([following[0], following[1], _dorsal_z(shape, (following[0], following[1], _on_margin(cell, (index + 1) % len(polygon))))]),
                )
            )
    return creases


def _ring_fraction(ring: int, rings: int) -> float:
    """Rings packed toward the margin, where both faces bevel to the edge.

    The bevel is where the surface bends fastest, and rings spaced evenly
    would sample it so coarsely that the strips between them read as ridges.
    """

    u = ring / rings
    return 1.0 - (1.0 - u) ** 1.5


def _taper(shape: LithicShape, x_mm: float, y_mm: float) -> float:
    # 1 at the centre, 0 at the margin, flat at the centre.
    r = math.hypot(x_mm, y_mm)
    if r < 1e-12:
        return 1.0
    s = min(r / plan_radius(shape, math.atan2(y_mm, x_mm)), 1.0)
    return 1.0 - s ** 4


def _dorsal_z(shape: LithicShape, point: tuple[float, float, bool]) -> float:
    if point[2]:
        return 0.0
    return dorsal_height(shape, point[0], point[1]) * _taper(shape, point[0], point[1])


def _ventral_z(shape: LithicShape, point: tuple[float, float, bool]) -> float:
    if point[2]:
        return 0.0
    return -ventral_depth(shape, point[0], point[1]) * _taper(shape, point[0], point[1])


def flaked_tool(
    shape: LithicShape = BIFACE_SHAPE,
    *,
    rings: int = 20,
    segments: int = 120,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (vertices, faces) for one closed flaked tool.

    The dorsal face is meshed scar by scar, each scar's region shrunk toward
    its own centre in rings, so the creases between scars are mesh edges -
    which is what a scan of a sharp ridge carries too.  The ventral face is
    one surface in rings from the margin to the centre.  The two share the
    margin, so the solid is watertight, and both are drawn to nothing at
    the margin, so the edge is an edge.
    """

    if rings < 2 or segments < 8:
        raise ValueError("a tool needs at least 2 rings and 8 segments")
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    index_of: dict[tuple[float, float, float], int] = {}

    def vertex(x: float, y: float, z: float) -> int:
        key = (round(x, 6), round(y, 6), round(z, 6))
        found = index_of.get(key)
        if found is None:
            found = len(vertices)
            index_of[key] = found
            vertices.append([float(x), float(y), float(z)])
        return found

    def sew(inner: list[int], outer: list[int], *, upward: bool) -> None:
        count = len(outer)
        for i in range(count):
            a, b = inner[i], inner[(i + 1) % count]
            d, c = outer[i], outer[(i + 1) % count]
            for triangle in (([a, d, c], [a, c, b]) if upward else ([a, c, d], [a, b, c])):
                if len(set(triangle)) == 3:
                    faces.append(triangle)

    def fan(centre: int, ring: list[int], *, upward: bool) -> None:
        count = len(ring)
        for i in range(count):
            a, b = ring[i], ring[(i + 1) % count]
            triangle = [centre, a, b] if upward else [centre, b, a]
            if len(set(triangle)) == 3:
                faces.append(triangle)

    cells = _dorsal_cells(shape, segments=segments)
    cell_rings = rings
    margin_points: dict[tuple[float, float], tuple[float, float, bool]] = {}
    for cell in cells:
        polygon, _tags = cell
        cx = sum(p[0] for p in polygon) / len(polygon)
        cy = sum(p[1] for p in polygon) / len(polygon)
        previous_ring: list[int] | None = None
        centre_index = vertex(cx, cy, _dorsal_z(shape, (cx, cy, False)))
        for ring in range(1, cell_rings + 1):
            t = _ring_fraction(ring, cell_rings)
            current_ring: list[int] = []
            for index, point in enumerate(polygon):
                x = cx + t * (point[0] - cx)
                y = cy + t * (point[1] - cy)
                on_margin = ring == cell_rings and _on_margin(cell, index)
                if on_margin:
                    margin_points[(round(x, 6), round(y, 6))] = (x, y, True)
                current_ring.append(vertex(x, y, _dorsal_z(shape, (x, y, on_margin))))
            if previous_ring is None:
                fan(centre_index, current_ring, upward=True)
            else:
                sew(previous_ring, current_ring, upward=True)
            previous_ring = current_ring

    # The ventral face, in rings from the shared margin to its own centre.
    margin = sorted(margin_points.values(), key=lambda p: math.atan2(p[1], p[0]))
    ventral_centre = vertex(0.0, 0.0, _ventral_z(shape, (0.0, 0.0, False)))
    previous_ring = None
    for ring in range(1, rings + 1):
        t = _ring_fraction(ring, rings)
        current_ring = []
        for point in margin:
            x, y = t * point[0], t * point[1]
            on_margin = ring == rings
            current_ring.append(vertex(x, y, _ventral_z(shape, (x, y, on_margin))))
        if previous_ring is None:
            fan(ventral_centre, current_ring, upward=False)
        else:
            sew(previous_ring, current_ring, upward=False)
        previous_ring = current_ring
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def dorsal_sheet(
    shape: LithicShape = BIFACE_SHAPE, *, pitch_mm: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """The dorsal face alone, meshed the way a scan meshes it: a regular
    grid of small triangles, ``pitch_mm`` apart, with no regard for where
    the ridges are.

    ``flaked_tool`` puts a vertex on every ridge and long needle triangles
    between them, which is the mesh a generator makes and no scanner does.
    A reading meant for scans is tested on this sheet, against the same
    ``dorsal_creases``.  Open at the margin; wound upward (+z outward).
    """

    if pitch_mm <= 0.0:
        raise ValueError("pitch_mm must be positive")
    half_x = shape.half_length_mm + pitch_mm
    half_y = shape.half_width_mm + pitch_mm
    xs = np.arange(-half_x, half_x + pitch_mm * 0.5, pitch_mm)
    ys = np.arange(-half_y, half_y + pitch_mm * 0.5, pitch_mm)
    columns, rows = xs.shape[0], ys.shape[0]
    index = -np.ones((rows, columns), dtype=np.int64)
    vertices: list[tuple[float, float, float]] = []
    for row, y in enumerate(ys.tolist()):
        for column, x in enumerate(xs.tolist()):
            r = math.hypot(x, y)
            if r > plan_radius(shape, math.atan2(y, x)) - 1e-9:
                continue
            index[row, column] = len(vertices)
            vertices.append((x, y, _dorsal_z(shape, (x, y, False))))
    faces: list[tuple[int, int, int]] = []
    for row in range(rows - 1):
        for column in range(columns - 1):
            a, b = index[row, column], index[row, column + 1]
            c, d = index[row + 1, column], index[row + 1, column + 1]
            if min(a, b, c, d) < 0:
                continue
            # Counter-clockwise seen from +z.
            faces.append((int(a), int(b), int(d)))
            faces.append((int(a), int(d), int(c)))
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def plan_area_mm2(shape: LithicShape, *, steps: int = 20000) -> float:
    """The plan's area, by quadrature, to compare an outline against."""

    total = 0.0
    for index in range(steps):
        angle = 2.0 * math.pi * (index + 0.5) / steps
        total += 0.5 * plan_radius(shape, angle) ** 2
    return total * (2.0 * math.pi / steps)


def lithic_session(
    shape: LithicShape = BIFACE_SHAPE,
    *,
    rings: int = 20,
    segments: int = 120,
    document_id: str = "artifact:biface",
) -> tuple[ArtifactSession, np.ndarray, np.ndarray]:
    """One tool in a session, lying as it would on the table: no Align.

    A stone tool is positioned by its use direction, by hand, and this one
    already lies that way - the tip on +X, the platform on -X, the dorsal
    face up - so the canonical frame is the source frame.
    """

    vertices, faces = flaked_tool(shape, rings=rings, segments=segments)
    mesh = MeshData(
        vertices=vertices,
        faces=faces,
        unit="mm",
        filepath=Path("/source/biface.ply"),
        source_identity=SourceFingerprint(
            sha256="b" * 64,
            size_bytes=int(vertices.size),
            mtime_ns=1,
            original_name="biface.ply",
            format="ply",
        ),
        source_format="ply",
        source_import_recipe=current_mesh_import_recipe("ply"),
    )
    session = ArtifactSession.create_from_source(
        mesh,
        resolved_source_path="/source/biface.ply",
        unit="mm",
        axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
        handedness="right",
        software_version="lithic-test",
        operator="tester",
        created_at="2026-09-05T00:00:00Z",
        document_id=document_id,
        metadata_revision_id=f"metadata:{document_id}",
        align_revision_id=f"align:{document_id}",
    )
    return session, vertices, faces


__all__ = [
    "BIFACE_SHAPE",
    "Facet",
    "LithicShape",
    "dorsal_creases",
    "dorsal_sheet",
    "dorsal_height",
    "flaked_tool",
    "lithic_session",
    "plan_area_mm2",
    "plan_radius",
    "ventral_depth",
]
