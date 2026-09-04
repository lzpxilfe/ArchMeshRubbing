"""암키와 · 수키와: roof tiles as closed solids, for tests that need a tile.

`src/core/tile_synthetic.py` already makes a tile, but it makes the surface a
tile's unwrap has to guess an axis and a split scheme from: one open shell,
no thickness, no marks.  A drawing or a rubbing needs the other thing - a
tile with two walls, cut ends, and the surface a fired tile actually carries -
so this builds that.

Both tiles are a segment of a cylinder, hollow, closed:

    암키와  the wide shallow one.  Laid concave side up, so its convex back -
            the face the paddle struck - is underneath.  Trapezoidal in plan:
            one end is narrower, which is what lets courses overlap.
    수키와  the half-round one that covers the joint between two 암키와.  Laid
            convex side up.  A 유단식 tile ends in a 미구, a length of smaller
            radius that slides under the next tile, and the step where the
            radius drops is the 언강.

And the two surfaces are not the same surface:

    등면 (convex, outer)  타날문 - the paddle's cord, a family of ridges a few
            millimetres apart, laid obliquely and struck in overlapping
            patches, which is what a rubbing of a tile shows.
    내면 (concave, inner)  포목흔 - the cloth the clay was laid on, a woven
            grid about a millimetre across and very shallow, over 모골흔, the
            longitudinal facets of the mould the cylinder was formed on.

Everything is deterministic: no random numbers anywhere, so the same
arguments give the same mesh, byte for byte.
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


AMKIWA = "amkiwa"
SUGKIWA = "sugkiwa"
TILE_KINDS = (AMKIWA, SUGKIWA)


@dataclass(frozen=True, slots=True)
class TileShape:
    """The measurements a tile is described by, in millimetres.

    Sizes are in the range Korean roof tiles are usually published in; they
    are a plausible tile, not a copy of any particular one.
    """

    kind: str
    length_mm: float
    #: Inner radius of the cylinder the tile is a segment of.
    inner_radius_mm: float
    thickness_mm: float
    #: How much of the cylinder the tile spans, at its wider end.
    span_deg: float
    #: How much narrower the far end is, as a share of the span.  0 keeps the
    #: tile rectangular in plan.
    taper: float = 0.0
    #: 수키와 only: the length of the 미구 and how far the radius drops for it.
    tongue_mm: float = 0.0
    tongue_drop_mm: float = 0.0
    #: How long the step at the 언강 takes to fall.
    tongue_step_mm: float = 8.0

    def __post_init__(self) -> None:
        if self.kind not in TILE_KINDS:
            raise ValueError(f"tile kind must be one of {', '.join(TILE_KINDS)}")
        for name in ("length_mm", "inner_radius_mm", "thickness_mm", "span_deg"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not (0.0 <= self.taper < 0.5):
            raise ValueError("taper must be between 0 and 0.5")
        if self.kind == AMKIWA and self.tongue_mm:
            raise ValueError("a 미구 belongs to a 수키와, not to a 암키와")
        if self.tongue_mm and self.tongue_mm >= self.length_mm / 2.0:
            raise ValueError("the 미구 cannot be half the tile")

    @property
    def outer_radius_mm(self) -> float:
        return self.inner_radius_mm + self.thickness_mm


#: A 암키와 of about 34 x 28 cm: wide, shallow, and tapered so courses lap.
AMKIWA_SHAPE = TileShape(
    kind=AMKIWA,
    length_mm=340.0,
    inner_radius_mm=210.0,
    thickness_mm=20.0,
    span_deg=76.0,
    taper=0.10,
)
#: A 유단식 수키와 of about 33 cm with a 55 mm 미구.
SUGKIWA_SHAPE = TileShape(
    kind=SUGKIWA,
    length_mm=330.0,
    inner_radius_mm=60.0,
    thickness_mm=18.0,
    span_deg=178.0,
    tongue_mm=55.0,
    tongue_drop_mm=11.0,
    tongue_step_mm=9.0,
)


def _smooth_step(x: float) -> float:
    """0 below 0, 1 above 1, with no corner at either end."""

    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return 0.5 * (1.0 - math.cos(math.pi * x))


#: The paddle: how far apart its cords are, how far they stand proud, and
#: how big a patch one strike leaves.  A tile is beaten all over in strokes
#: that overlap, each landing at its own angle, and the overlaps are what a
#: 타날문 rubbing actually shows.
_CORD_PITCH_MM = 3.0
_CORD_HEIGHT_MM = 0.35
_STRIKE_ALONG_MM = 58.0
_STRIKE_ACROSS_MM = 44.0


def _strike_hash(i: int, j: int, salt: int) -> float:
    """A settled number in [0, 1) for one strike - no random state anywhere."""

    h = (i * 73856093) ^ (j * 19349663) ^ (salt * 83492791)
    h &= 0xFFFFFFFF
    h ^= h >> 13
    h = (h * 0x5BD1E995) & 0xFFFFFFFF
    h ^= h >> 15
    return h / 4294967296.0


def cord_relief(along_mm: float, across_mm: float) -> float:
    """타날문: the cord of a paddle, in overlapping strikes.

    Each strike lays ridges 3 mm apart, about 0.35 mm proud, at its own
    angle a little off the tile's length.  Where two strikes overlap the
    later one is the one that shows, so the strikes are combined by taking
    the stronger - which is why a corded tile reads as patches of cord at
    slightly different angles rather than as one endless comb.
    """

    i0 = math.floor(along_mm / _STRIKE_ALONG_MM)
    j0 = math.floor(across_mm / _STRIKE_ACROSS_MM)
    best = 0.0
    for i in (i0 - 1, i0, i0 + 1):
        for j in (j0 - 1, j0, j0 + 1):
            # Where this strike landed, how hard, and at what angle.
            centre_along = (i + 0.5 + 0.25 * (_strike_hash(i, j, 1) - 0.5)) * _STRIKE_ALONG_MM
            centre_across = (j + 0.5 + 0.3 * (_strike_hash(i, j, 2) - 0.5)) * _STRIKE_ACROSS_MM
            du = (along_mm - centre_along) / (0.72 * _STRIKE_ALONG_MM)
            dv = (across_mm - centre_across) / (0.72 * _STRIKE_ACROSS_MM)
            reach = du * du + dv * dv
            if reach >= 1.0:
                continue
            # A strike is firmest under the middle of the paddle and fades
            # toward its edge, so overlaps are gradual.
            firmness = (0.70 + 0.30 * _strike_hash(i, j, 3)) * (1.0 - reach) ** 0.6
            angle = math.radians(8.0 + 14.0 * _strike_hash(i, j, 4))
            phase = _strike_hash(i, j, 5) * _CORD_PITCH_MM
            u = (along_mm * math.cos(angle) + across_mm * math.sin(angle)) + phase
            ridge = 0.5 * (1.0 - math.cos(2.0 * math.pi * u / _CORD_PITCH_MM))
            best = max(best, _CORD_HEIGHT_MM * firmness * ridge)
    return best


def cloth_relief(along_mm: float, across_mm: float) -> float:
    """포목흔 over 모골흔: the cloth's weave, and the mould's facets.

    The weave is about 1.2 mm and very shallow - a rubbing shows it as a
    tone, not as lines - and it lies over the longitudinal facets the tile
    took from the mould it was formed on.
    """

    warp = 0.5 * (1.0 - math.cos(2.0 * math.pi * along_mm / 1.2))
    weft = 0.5 * (1.0 - math.cos(2.0 * math.pi * across_mm / 1.35))
    weave = -0.09 * (0.6 * warp + 0.4 * weft)
    facet = -0.12 * abs(((across_mm / 26.0) % 1.0) - 0.5)
    return weave + facet


def _tile_grid(
    shape: TileShape,
    *,
    axial_step_mm: float,
    angular_step_mm: float,
    relief: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the outer and inner surface grids, each (n_s+1, n_theta+1, 3).

    The cylinder's axis is +Y.  A 수키와 arcs above its axis so its convex
    face is up; a 암키와 hangs below its own, so its concave face is up, and
    it is then dropped so that its lowest point sits on z = 0.
    """

    span = math.radians(shape.span_deg)
    outer = shape.outer_radius_mm
    n_s = max(4, int(round(shape.length_mm / axial_step_mm)))
    n_theta = max(4, int(round(outer * span / angular_step_mm)))
    # A 암키와 is the same arc turned over, and turning it over is a half turn
    # about the tile's own length - a rotation, not a mirror.  Mirroring it
    # in z alone would leave every face wound the wrong way round, and a
    # solid whose faces face inward has a negative volume.
    sign = 1.0 if shape.kind == SUGKIWA else -1.0
    centre_z = 0.0 if shape.kind == SUGKIWA else outer

    outer_grid = np.empty((n_s + 1, n_theta + 1, 3), dtype=np.float64)
    inner_grid = np.empty((n_s + 1, n_theta + 1, 3), dtype=np.float64)
    for i in range(n_s + 1):
        s = i / n_s
        y = shape.length_mm * (s - 0.5)
        # The 미구 is turned down over the last of the tile's length.
        drop = 0.0
        if shape.tongue_mm:
            start = shape.length_mm - shape.tongue_mm
            drop = shape.tongue_drop_mm * _smooth_step(
                (shape.length_mm * s - start) / shape.tongue_step_mm
            )
        # A 암키와 narrows toward one end; a 수키와 keeps its span.
        span_here = span * (1.0 - shape.taper * s)
        for j in range(n_theta + 1):
            theta = span_here * (j / n_theta - 0.5)
            r_inner = shape.inner_radius_mm - drop
            r_outer = r_inner + shape.thickness_mm
            if relief:
                arc_outer = theta * r_outer
                arc_inner = theta * r_inner
                r_outer += cord_relief(y + shape.length_mm / 2.0, arc_outer)
                r_inner -= cloth_relief(y + shape.length_mm / 2.0, arc_inner)
            for grid, radius in ((outer_grid, r_outer), (inner_grid, r_inner)):
                grid[i, j] = (
                    sign * radius * math.sin(theta),
                    y,
                    centre_z + sign * radius * math.cos(theta),
                )
    if shape.kind == AMKIWA:
        lowest = float(min(outer_grid[:, :, 2].min(), inner_grid[:, :, 2].min()))
        outer_grid[:, :, 2] -= lowest
        inner_grid[:, :, 2] -= lowest
    return outer_grid, inner_grid


def _stood_on_the_canonical_axis(grid: np.ndarray, axis_z_mm: float) -> np.ndarray:
    """Turn a tile so its own cylinder axis is +Z through the origin.

    A quarter turn about +X and a slide down z - a rotation, so the winding
    is untouched.  This is what an Align does for a real tile once the
    drafter has established its axis; the fixture has no Align, so a caller
    who needs the canonical pose asks for it here and says so.
    """

    x = grid[..., 0]
    y = grid[..., 1]
    z = grid[..., 2] - axis_z_mm
    return np.stack([x, -z, y], axis=-1)


def hollow_tile(
    shape: TileShape = AMKIWA_SHAPE,
    *,
    axial_step_mm: float = 2.0,
    angular_step_mm: float = 2.0,
    relief: bool = True,
    on_canonical_axis: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (vertices, faces) for one closed tile.

    The two surfaces are stitched by four strips - the two cut ends and the
    two sides - so the mesh is watertight: every edge belongs to exactly two
    faces, which is what the volume metric and the outline's topology check
    both need.  Faces wind counter-clockwise seen from outside.

    By default the tile lies as it would on a roof, its cylinder axis along
    +Y.  ``on_canonical_axis`` stands it on that axis instead, +Z through the
    origin, which is where a development that centres its sections on the
    canonical axis needs it.
    """

    outer_grid, inner_grid = _tile_grid(
        shape,
        axial_step_mm=axial_step_mm,
        angular_step_mm=angular_step_mm,
        relief=relief,
    )
    if on_canonical_axis:
        axis_z = 0.0 if shape.kind == SUGKIWA else float(
            shape.outer_radius_mm - min(outer_grid[..., 2].min(), inner_grid[..., 2].min())
        )
        outer_grid = _stood_on_the_canonical_axis(outer_grid, axis_z)
        inner_grid = _stood_on_the_canonical_axis(inner_grid, axis_z)
    rows, columns = outer_grid.shape[0], outer_grid.shape[1]
    vertices = np.vstack(
        [outer_grid.reshape(-1, 3), inner_grid.reshape(-1, 3)]
    )
    outer_count = rows * columns

    def out(i: int, j: int) -> int:
        return i * columns + j

    def inn(i: int, j: int) -> int:
        return outer_count + i * columns + j

    faces: list[tuple[int, int, int]] = []
    for i in range(rows - 1):
        for j in range(columns - 1):
            # Outer: seen from outside the tile.
            faces.append((out(i, j), out(i, j + 1), out(i + 1, j + 1)))
            faces.append((out(i, j), out(i + 1, j + 1), out(i + 1, j)))
            # Inner: the same quad, wound the other way.
            faces.append((inn(i, j), inn(i + 1, j + 1), inn(i, j + 1)))
            faces.append((inn(i, j), inn(i + 1, j), inn(i + 1, j + 1)))
    # The two cut ends.
    for j in range(columns - 1):
        faces.append((out(0, j), inn(0, j), inn(0, j + 1)))
        faces.append((out(0, j), inn(0, j + 1), out(0, j + 1)))
        last = rows - 1
        faces.append((out(last, j), out(last, j + 1), inn(last, j + 1)))
        faces.append((out(last, j), inn(last, j + 1), inn(last, j)))
    # The two sides.
    for i in range(rows - 1):
        faces.append((out(i, 0), out(i + 1, 0), inn(i + 1, 0)))
        faces.append((out(i, 0), inn(i + 1, 0), inn(i, 0)))
        edge = columns - 1
        faces.append((out(i, edge), inn(i, edge), inn(i + 1, edge)))
        faces.append((out(i, edge), inn(i + 1, edge), out(i + 1, edge)))
    return vertices, np.asarray(faces, dtype=np.int32)


def tile_session(
    shape: TileShape = AMKIWA_SHAPE,
    *,
    axial_step_mm: float = 2.0,
    angular_step_mm: float = 2.0,
    relief: bool = True,
    on_canonical_axis: bool = False,
    document_id: str | None = None,
) -> tuple[ArtifactSession, np.ndarray, np.ndarray]:
    """One tile in a session, with its arrays.

    No Align is committed, so the canonical frame is the source frame.  A
    development that centres its sections on the canonical axis therefore
    needs the tile to arrive already standing on it: pass
    ``on_canonical_axis``, which is the fixture standing in for the Align a
    real tile would be given.
    """

    vertices, faces = hollow_tile(
        shape,
        axial_step_mm=axial_step_mm,
        angular_step_mm=angular_step_mm,
        relief=relief,
        on_canonical_axis=on_canonical_axis,
    )
    name = document_id or f"artifact:{shape.kind}"
    mesh = MeshData(
        vertices=vertices,
        faces=faces,
        unit="mm",
        filepath=Path(f"/source/{shape.kind}.ply"),
        source_identity=SourceFingerprint(
            sha256=("9" if shape.kind == AMKIWA else "5") * 64,
            size_bytes=int(vertices.size),
            mtime_ns=1,
            original_name=f"{shape.kind}.ply",
            format="ply",
        ),
        source_format="ply",
        source_import_recipe=current_mesh_import_recipe("ply"),
    )
    session = ArtifactSession.create_from_source(
        mesh,
        resolved_source_path=f"/source/{shape.kind}.ply",
        unit="mm",
        axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
        handedness="right",
        software_version="tile-test",
        operator="tester",
        created_at="2026-09-05T00:00:00Z",
        document_id=name,
        metadata_revision_id=f"metadata:{name}",
        align_revision_id=f"align:{name}",
    )
    return session, vertices, faces


__all__ = [
    "AMKIWA",
    "AMKIWA_SHAPE",
    "SUGKIWA",
    "SUGKIWA_SHAPE",
    "TILE_KINDS",
    "TileShape",
    "cloth_relief",
    "cord_relief",
    "hollow_tile",
    "tile_session",
]
