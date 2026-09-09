"""이음: sewing a piece back onto the wall it stands on, as a declared step.

`artifact_mesh_bodies` says a scan came in pieces and lists what each open
edge lies against; it deliberately stops there.  This is the other half: the
archaeologist has looked at the artifact, decided that a piece is of the pot
rather than resting in it, and the decision has to become geometry that can
be reproduced from the same source rather than a mesh someone edited once.

Two joints, because a scanner leaves two.

**A piece standing on a wall.**  A boss, a lug or a handle is a volume, so
its skin meets the wall along *two* edges - one where the upper surface runs
in, one where the under surface does - and the band of wall between them is
buried inside the join, not visible surface.  So that band is deleted and
each of the piece's edges sewn to the edge the deletion made.  Sewing the
piece's two edges to *each other* instead - the obvious move, and the one
that closes every boundary - seals the piece into a lens and leaves it
standing loose.

**A thin edge seen from both sides.**  A foot's rim, a sherd's break: the
head saw either face but could not get over the edge between them, so the
file carries two open rings a wall's thickness apart.  Both faces are real
surface and nothing is buried, so nothing is cut - the two rings are sewn
straight together.  Which of the two a joint is, is the archaeologist's
call; telling them apart is what the diagnosis's candidate list is for.

Two rules hold throughout.

* **No vertex is added or moved.**  Every triangle written here runs between
  vertices the scan already had.  A repair that moved points would be a
  reconstruction, and the texture atlas a rubbing is read from would no
  longer match its own mesh.
* **Nothing is decided here.**  Which rings, and onto which body, comes in
  as an argument; each is named by a hash of what it is rather than by an
  index, so replaying a repair against a mesh that has moved on fails loudly
  instead of sewing the wrong edge.

Where the band to delete is found without an axis: a vertex of the wall is
on one side of the joint or the other according to the sign of `d1 - d2`,
its distance to the first ring less its distance to the second, and the band
is the star of the faces that zero runs through.  The wall's *other* face is
the same distance away in space, so the search is kept to faces that turn
towards the piece and are reachable across shared edges from where the rings
actually land - the outer skin of a pot is a long walk from its inner skin
even where the two are two millimetres apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .artifact_condition_annotation import (
    MAX_CONDITION_TOTAL_FACES,
    ArtifactConditionAnnotationError,
    face_indices_from_ranges,
    validate_face_ranges,
)
from .artifact_mesh_bodies import (
    boundary_rings,
    face_bodies,
    ring_as_triangles,
    surface_distances,
)
from .canonical_json import canonical_json_sha256


MESH_REPAIR_SCHEMA_VERSION = "1.0.0"
MESH_REPAIR_EXTENSION_KEY = "org.archmeshrubbing:mesh-repair-v1"

#: A piece's two edges onto the body it stands on: the wall between them is
#: cut away first, because it is buried in the joint.
SEW_RING_PAIR_TO_BODY = "sew_ring_pair_to_body/v1"

#: The other joint a scan leaves: a thin edge the head could see either side
#: of but not over - a foot's rim, a sherd's broken face - which arrives as
#: two open rings facing each other across the clay's own thickness.  Nothing
#: is cut for this one; the two edges are sewn straight together.
SEW_RING_TO_RING = "sew_ring_to_ring/v1"

#: Not a join at all: taking out a patch that is not surface of the artifact.
#: A modelling tool asked to close a scan caps whatever it could not see with
#: a fan of large flat triangles, and on 24ET0021 two such fans - one over
#: each skin of the foot - are what made the 대각 read as solid.  Removing it is a decision about the pot, so it is
#: declared and its evidence - how much coarser the patch is than the surface
#: around it - goes in the receipt beside it.
DROP_FACES = "drop_faces/v1"
REPAIR_KINDS: tuple[str, ...] = (
    SEW_RING_PAIR_TO_BODY,
    SEW_RING_TO_RING,
    DROP_FACES,
)

#: How far from the rings the wall is searched.  Wide enough to cross the
#: shadow band and a few rows of faces either side of it, narrow enough that
#: the search does not wander off round a small pot.
DEFAULT_REACH_UM = 8_000
MIN_REACH_UM = 100
MAX_REACH_UM = 200_000

MAX_REPAIR_STEPS = 64


class ArtifactMeshRepairError(ValueError):
    """The repair cannot be carried out as declared."""


def _positive_int(value: object, *, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactMeshRepairError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ArtifactMeshRepairError(
            f"{name} must be between {minimum} and {maximum}, not {value}"
        )
    return value


def ring_sha256(ring: Sequence[int] | np.ndarray) -> str:
    """Name one open ring by what it is, not by where its walk began.

    The walk starts at whichever vertex the edge table happened to hand over
    first, so the ring is rotated to start at its lowest vertex before it is
    hashed.  Direction is kept: a ring read the other way round bounds the
    other side and is not the same edge.
    """

    indices = np.asarray(ring, dtype=np.int64).reshape(-1)
    if indices.size < 3:
        raise ArtifactMeshRepairError("a ring needs at least three vertices")
    rolled = np.roll(indices, -int(np.argmin(indices)))
    return canonical_json_sha256(
        {
            "kind": "open_ring",
            "schema_version": MESH_REPAIR_SCHEMA_VERSION,
            "vertices": [int(value) for value in rolled],
        }
    )


def body_sha256(faces: np.ndarray, body_faces: Sequence[int] | np.ndarray) -> str:
    """Name one body by the set of vertices its faces use."""

    selected = np.asarray(body_faces, dtype=np.int64).reshape(-1)
    if selected.size == 0:
        raise ArtifactMeshRepairError("a body needs at least one face")
    used = np.unique(np.asarray(faces, dtype=np.int64)[selected])
    return canonical_json_sha256(
        {
            "kind": "body",
            "schema_version": MESH_REPAIR_SCHEMA_VERSION,
            "vertices": [int(value) for value in used],
        }
    )


@dataclass(frozen=True)
class RingPairJoin:
    """The decision: these two open edges belong on that body."""

    first_ring_sha256: str
    second_ring_sha256: str
    body_sha256: str
    reach_um: int = DEFAULT_REACH_UM
    kind: str = SEW_RING_PAIR_TO_BODY

    def __post_init__(self) -> None:
        for name in ("first_ring_sha256", "second_ring_sha256", "body_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise ArtifactMeshRepairError(f"{name} must be a sha256 digest")
        if self.first_ring_sha256 == self.second_ring_sha256:
            raise ArtifactMeshRepairError(
                "a join needs two different rings; the piece meets the wall along "
                "two edges, one for each of its skins"
            )
        if self.kind != SEW_RING_PAIR_TO_BODY:
            raise ArtifactMeshRepairError(f"unsupported repair kind: {self.kind!r}")
        object.__setattr__(
            self,
            "reach_um",
            _positive_int(
                self.reach_um, name="reach_um", minimum=MIN_REACH_UM, maximum=MAX_REACH_UM
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "body_sha256": self.body_sha256,
            "first_ring_sha256": self.first_ring_sha256,
            "kind": self.kind,
            "reach_um": self.reach_um,
            "second_ring_sha256": self.second_ring_sha256,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "RingPairJoin":
        expected = {
            "body_sha256",
            "first_ring_sha256",
            "kind",
            "reach_um",
            "second_ring_sha256",
        }
        if set(data) != expected:
            raise ArtifactMeshRepairError(
                f"a join is written with exactly {sorted(expected)}"
            )
        return cls(
            first_ring_sha256=str(data["first_ring_sha256"]),
            second_ring_sha256=str(data["second_ring_sha256"]),
            body_sha256=str(data["body_sha256"]),
            reach_um=data["reach_um"],  # type: ignore[arg-type]
            kind=str(data["kind"]),
        )


@dataclass(frozen=True)
class RingToRingJoin:
    """The decision: these two open edges are the two sides of one edge."""

    first_ring_sha256: str
    second_ring_sha256: str
    reach_um: int = DEFAULT_REACH_UM
    kind: str = SEW_RING_TO_RING

    def __post_init__(self) -> None:
        for name in ("first_ring_sha256", "second_ring_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise ArtifactMeshRepairError(f"{name} must be a sha256 digest")
        if self.first_ring_sha256 == self.second_ring_sha256:
            raise ArtifactMeshRepairError(
                "a join needs two different rings; one edge cannot be sewn to itself"
            )
        if self.kind != SEW_RING_TO_RING:
            raise ArtifactMeshRepairError(f"unsupported repair kind: {self.kind!r}")
        object.__setattr__(
            self,
            "reach_um",
            _positive_int(
                self.reach_um, name="reach_um", minimum=MIN_REACH_UM, maximum=MAX_REACH_UM
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "first_ring_sha256": self.first_ring_sha256,
            "kind": self.kind,
            "reach_um": self.reach_um,
            "second_ring_sha256": self.second_ring_sha256,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "RingToRingJoin":
        expected = {"first_ring_sha256", "kind", "reach_um", "second_ring_sha256"}
        if set(data) != expected:
            raise ArtifactMeshRepairError(
                f"a join is written with exactly {sorted(expected)}"
            )
        return cls(
            first_ring_sha256=str(data["first_ring_sha256"]),
            second_ring_sha256=str(data["second_ring_sha256"]),
            reach_um=data["reach_um"],  # type: ignore[arg-type]
            kind=str(data["kind"]),
        )


def face_set_sha256(faces: np.ndarray, indices: Sequence[int] | np.ndarray) -> str:
    """Name a patch by the triangles it holds, not by where they sit.

    Face indices shift as soon as a step removes anything, so a patch chosen
    on one state of the mesh is only the same patch if the triangles at those
    indices are still the same triangles.  This is what says so.
    """

    selected = np.asarray(indices, dtype=np.int64).reshape(-1)
    if selected.size == 0:
        raise ArtifactMeshRepairError("a patch needs at least one face")
    triangles = np.asarray(faces, dtype=np.int64)[np.sort(selected)]
    return canonical_json_sha256(
        {
            "faces": [[int(a), int(b), int(c)] for a, b, c in triangles],
            "kind": "face_set",
            "schema_version": MESH_REPAIR_SCHEMA_VERSION,
        }
    )


@dataclass(frozen=True)
class DropFaces:
    """The decision: this patch is not surface of the artifact."""

    face_ranges: tuple[tuple[int, int], ...]
    total_face_count: int
    selection_sha256: str
    kind: str = DROP_FACES

    def __post_init__(self) -> None:
        if not isinstance(self.selection_sha256, str) or len(self.selection_sha256) != 64:
            raise ArtifactMeshRepairError("selection_sha256 must be a sha256 digest")
        if self.kind != DROP_FACES:
            raise ArtifactMeshRepairError(f"unsupported repair kind: {self.kind!r}")
        total = _positive_int(
            self.total_face_count,
            name="total_face_count",
            minimum=1,
            maximum=MAX_CONDITION_TOTAL_FACES,
        )
        object.__setattr__(self, "total_face_count", total)
        try:
            canonical = validate_face_ranges(
                [tuple(pair) for pair in self.face_ranges], total_face_count=total
            )
        except ArtifactConditionAnnotationError as exc:
            raise ArtifactMeshRepairError(f"the patch is not a face set: {exc}") from exc
        object.__setattr__(self, "face_ranges", canonical)

    def indices(self) -> np.ndarray:
        return face_indices_from_ranges(
            [list(pair) for pair in self.face_ranges],
            total_face_count=self.total_face_count,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "face_ranges": [[start, end] for start, end in self.face_ranges],
            "kind": self.kind,
            "selection_sha256": self.selection_sha256,
            "total_face_count": self.total_face_count,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "DropFaces":
        expected = {"face_ranges", "kind", "selection_sha256", "total_face_count"}
        if set(data) != expected:
            raise ArtifactMeshRepairError(
                f"a patch removal is written with exactly {sorted(expected)}"
            )
        ranges = data["face_ranges"]
        if not isinstance(ranges, Sequence) or isinstance(ranges, (str, bytes)):
            raise ArtifactMeshRepairError("face_ranges must be a list of pairs")
        return cls(
            face_ranges=tuple(tuple(int(v) for v in pair) for pair in ranges),  # type: ignore[misc]
            total_face_count=data["total_face_count"],  # type: ignore[arg-type]
            selection_sha256=str(data["selection_sha256"]),
            kind=str(data["kind"]),
        )


def drop_faces(
    vertices_mm: np.ndarray, faces: np.ndarray, indices: Sequence[int] | np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Take one patch out, and record how unlike the surface round it it was.

    The evidence goes in the receipt rather than into a rule.  A fill is
    coarse and unnaturally even - 24ET0021's two are 192 triangles apiece in
    three rings of 54, 26 and 25 mm², each ring's triangles within a percent
    of one another, against 3.5 mm² for the scanned 대각 beside them - but a
    rule that dropped anything coarse would eat a flat scanned floor, and one
    that demanded coarseness would refuse a legitimate removal.  So the ratio
    recorded here is the plain one a reader can check: the patch's median
    triangle against the median of the faces bordering it.  It is a milder
    number than the comparison with real surface further off, because the
    border is where a fill grades into the scan; both are worth having and
    only this one is local enough to compute without judgement.
    """

    vertices = np.asarray(vertices_mm, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    selected = np.unique(np.asarray(indices, dtype=np.int64).reshape(-1))
    if selected.size == 0:
        raise ArtifactMeshRepairError("a patch needs at least one face")
    if selected.min() < 0 or selected.max() >= triangles.shape[0]:
        raise ArtifactMeshRepairError("the patch names a face the mesh does not have")
    if selected.size >= triangles.shape[0]:
        raise ArtifactMeshRepairError("a repair that removes every face leaves nothing")
    if int(face_bodies(triangles[selected]).max()) + 1 != 1:
        raise ArtifactMeshRepairError(
            "the patch is in more than one piece; a fill is one patch, and removing "
            "several at once hides which was which"
        )

    def _areas(chosen: np.ndarray) -> np.ndarray:
        corners = vertices[triangles[chosen]]
        return 0.5 * np.linalg.norm(
            np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
            axis=1,
        )

    inside = _areas(selected)
    keep = np.ones(triangles.shape[0], dtype=bool)
    keep[selected] = False
    border = np.flatnonzero(
        keep & np.isin(triangles, np.unique(triangles[selected])).any(axis=1)
    )
    outside = _areas(border) if border.size else np.zeros(0)
    ratio = (
        float(np.median(inside) / np.median(outside))
        if outside.size and float(np.median(outside)) > 0.0
        else 0.0
    )
    return triangles[keep], {
        "border_face_count": int(border.size),
        "kind": DROP_FACES,
        "median_area_ratio_millionths": int(round(ratio * 1_000_000)),
        "removed_area_um2": int(round(float(inside.sum()) * 1_000_000)),
        "removed_face_count": int(selected.size),
        "added_face_count": 0,
    }


@dataclass(frozen=True)
class MeshRepairReceipt:
    """What the declared steps did, in numbers a reader can check."""

    schema_version: str
    steps: tuple[Mapping[str, Any], ...]
    removed_face_count: int
    added_face_count: int
    body_count_before: int
    body_count_after: int
    open_ring_count_before: int
    open_ring_count_after: int
    non_manifold_edge_count_after: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "added_face_count": self.added_face_count,
            "body_count_after": self.body_count_after,
            "body_count_before": self.body_count_before,
            "non_manifold_edge_count_after": self.non_manifold_edge_count_after,
            "open_ring_count_after": self.open_ring_count_after,
            "open_ring_count_before": self.open_ring_count_before,
            "removed_face_count": self.removed_face_count,
            "schema_version": self.schema_version,
            "steps": [dict(step) for step in self.steps],
        }


def _arc_parameter(vertices: np.ndarray, loop: Sequence[int]) -> np.ndarray:
    """Where each point of a closed loop sits along it, from 0 to 1."""

    points = vertices[np.asarray(loop, dtype=np.int64)]
    step = np.linalg.norm(np.diff(np.concatenate([points, points[:1]]), axis=0), axis=1)
    total = float(step.sum())
    if total <= 0.0:
        raise ArtifactMeshRepairError("a boundary loop of zero length cannot be sewn")
    return np.concatenate([[0.0], np.cumsum(step)]) / total


def _sew(
    vertices: np.ndarray, loop_a: Sequence[int], loop_b: Sequence[int]
) -> list[tuple[int, int, int]]:
    """Sew two boundary loops into one band of triangles.

    The loops carry different numbers of points, so they are matched on how
    far along each one a point sits rather than one for one, and at each step
    the loop whose next point comes first takes the triangle.  Which way
    round `loop_b` runs, and where it starts, are chosen by fit - the pairing
    is a fact about the two curves, not about any axis.  Every triangle then
    uses its loop's boundary edge backwards, which is what makes the new
    faces wind the same way as the ones they are sewn to.
    """

    a = list(loop_a)
    pa = _arc_parameter(vertices, a)
    points_a = vertices[np.asarray(a, dtype=np.int64)]

    best: tuple[float, list[int], bool, np.ndarray] | None = None
    for reversed_b in (False, True):
        walk = list(loop_b)[::-1] if reversed_b else list(loop_b)
        start = int(
            np.argmin(np.linalg.norm(vertices[np.asarray(walk)] - points_a[0], axis=1))
        )
        walk = walk[start:] + walk[:start]
        pb = _arc_parameter(vertices, walk)
        at = np.clip(np.searchsorted(pb, pa[:-1], side="right") - 1, 0, len(walk) - 1)
        cost = float(
            np.linalg.norm(vertices[np.asarray(walk)][at] - points_a, axis=1).sum()
        )
        if best is None or cost < best[0]:
            best = (cost, walk, reversed_b, pb)
    assert best is not None
    _, b, reversed_b, pb = best

    def triangle(
        ring: list[int], index: int, backwards: bool, other: int
    ) -> tuple[int, int, int]:
        head = ring[index % len(ring)]
        tail = ring[(index + 1) % len(ring)]
        return (head, tail, other) if backwards else (tail, head, other)

    out: list[tuple[int, int, int]] = []
    ia, ib = 0, 0
    while ia < len(a) or ib < len(b):
        take_a = ib >= len(b) or (ia < len(a) and pa[ia + 1] <= pb[ib + 1])
        if take_a:
            out.append(triangle(a, ia, False, b[ib % len(b)]))
            ia += 1
        else:
            out.append(triangle(b, ib, reversed_b, a[ia % len(a)]))
            ib += 1
    return out


def _face_neighbourhood(
    vertices: np.ndarray,
    faces: np.ndarray,
    target: np.ndarray,
    rings: tuple[np.ndarray, np.ndarray],
    piece_vertices: np.ndarray,
    reach_mm: float,
) -> np.ndarray:
    """The wall around the joint - the skin the piece stands on, only.

    A pot's two skins pass within a couple of millimetres of each other, and
    the joint's edges can be nearer the far one than the near one where the
    wall is thin, so distance alone takes a band out of both.  Two things
    narrow it, neither of which needs the artifact to have an axis:

    * a face of the wall belongs to the joint only if it **faces the piece** -
      its outward normal points towards the piece's surface rather than away
      from it, which is true of the skin the piece stands on and false of the
      skin behind it;
    * and only if it is joined across shared edges to where the piece's edges
      actually land, so a patch that merely passes nearby is left alone.
    """

    centroids = vertices[faces[target]].mean(axis=1)
    near = np.minimum(
        surface_distances(centroids, vertices, ring_as_triangles(rings[0])),
        surface_distances(centroids, vertices, ring_as_triangles(rings[1])),
    ) <= reach_mm
    if not near.any():
        raise ArtifactMeshRepairError(
            f"no face of the body lies within {reach_mm:g} mm of either edge; "
            "the rings do not rest on this body"
        )

    if piece_vertices.size:
        corners = vertices[faces[target]]
        normals = np.cross(
            corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
        )
        lengths = np.linalg.norm(normals, axis=1)
        _, nearest = cKDTree(vertices[piece_vertices]).query(centroids)
        towards = vertices[piece_vertices][nearest] - centroids
        facing = np.einsum("ij,ij->i", normals, towards) > 0.0
        near &= facing & (lengths > 0.0)
        if not near.any():
            raise ArtifactMeshRepairError(
                "no face of the body near the edges turns towards the piece; the "
                "piece stands behind this body rather than on it"
            )

    seed_points = vertices[np.concatenate(rings)]
    candidate_vertices = np.unique(faces[target[near]])
    _, landed = cKDTree(vertices[candidate_vertices]).query(seed_points)
    landing = np.isin(
        faces[target], candidate_vertices[np.unique(landed)]
    ).any(axis=1)

    selected = np.flatnonzero(near)
    labels = face_bodies(faces[target[selected]])
    reached = np.unique(labels[landing[selected]])
    if reached.size == 0:
        raise ArtifactMeshRepairError(
            "the edges land on no face of the body that turns towards the piece"
        )
    return target[selected[np.isin(labels, reached)]]


def _straddling_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    neighbourhood: np.ndarray,
    target: np.ndarray,
    rings: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """The faces buried in the joint: the mid-curve's row, and its neighbours.

    The mid-curve is where a wall vertex stops being nearer the first edge
    and starts being nearer the second, and the faces it runs through are the
    obvious thing to take out.  Taking only those leaves a ragged cut that
    can pinch shut at a vertex the curve passes exactly through, which is not
    a boundary that can be sewn.  So every face of the body touching one of
    those vertices goes - the whole star, not only the part inside the search
    - because a vertex left holding faces on both sides of the cut is exactly
    the pinch that cannot be walked.
    """

    used = np.unique(faces[neighbourhood])
    signed = surface_distances(
        vertices[used], vertices, ring_as_triangles(rings[0])
    ) - surface_distances(vertices[used], vertices, ring_as_triangles(rings[1]))
    lookup = np.full(vertices.shape[0], np.nan)
    lookup[used] = signed
    corner = lookup[faces[neighbourhood]]
    # Touching the curve counts, not only straddling it.  A wall whose rows
    # happen to fall symmetrically between the two edges puts a whole row of
    # vertices *on* the mid-curve, and under a strict test no face crosses it
    # at all: the faces above are all on one side, the faces below all on the
    # other, and the cut comes out empty on a mesh that plainly has a joint.
    crossed = neighbourhood[(corner.min(axis=1) <= 0.0) & (corner.max(axis=1) >= 0.0)]
    if crossed.size == 0:
        return crossed
    touched = np.unique(faces[crossed])
    return target[np.isin(faces[target], touched).any(axis=1)]


def _new_rings(before: list[list[int]], after: list[list[int]]) -> list[list[int]]:
    known = {ring_sha256(ring) for ring in before}
    return [ring for ring in after if ring_sha256(ring) not in known]


def sew_ring_pair_to_body(
    vertices_mm: np.ndarray,
    faces: np.ndarray,
    *,
    first_ring: Sequence[int],
    second_ring: Sequence[int],
    body_faces: Sequence[int] | np.ndarray,
    reach_um: int = DEFAULT_REACH_UM,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sew a piece's two open edges onto the body they rest against."""

    vertices = np.asarray(vertices_mm, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    reach_mm = (
        _positive_int(
            reach_um, name="reach_um", minimum=MIN_REACH_UM, maximum=MAX_REACH_UM
        )
        / 1000.0
    )
    rings = (
        np.asarray(first_ring, dtype=np.int64),
        np.asarray(second_ring, dtype=np.int64),
    )
    target = np.asarray(body_faces, dtype=np.int64).reshape(-1)

    # The piece is whatever bodies the two edges belong to, minus the body
    # they are being sewn onto: that is what the wall has to turn towards.
    labels = face_bodies(triangles)
    on_ring = np.isin(triangles, np.unique(np.concatenate(rings))).any(axis=1)
    piece_bodies = np.setdiff1d(np.unique(labels[on_ring]), np.unique(labels[target]))
    piece_faces = np.flatnonzero(np.isin(labels, piece_bodies))
    # Without the edge vertices themselves.  Those sit on the wall, so the
    # direction from a wall face to the nearest of them is nearly in that
    # face's own plane and its sign is noise - which is exactly at the joint,
    # the one place the answer has to be right.
    piece_vertices = (
        np.setdiff1d(np.unique(triangles[piece_faces]), np.unique(np.concatenate(rings)))
        if piece_faces.size
        else np.zeros(0, np.int64)
    )

    neighbourhood = _face_neighbourhood(
        vertices, triangles, target, rings, piece_vertices, reach_mm
    )
    buried = _straddling_faces(vertices, triangles, neighbourhood, target, rings)
    if buried.size == 0:
        raise ArtifactMeshRepairError(
            "no face of the body lies between the two edges, so there is nothing "
            "buried in the joint; the two edges are probably on the same side of it"
        )

    before = boundary_rings(triangles)
    keep = np.ones(triangles.shape[0], dtype=bool)
    keep[buried] = False
    opened = triangles[keep]
    fresh = _new_rings(before, boundary_rings(opened))
    if len(fresh) != 2:
        raise ArtifactMeshRepairError(
            f"cutting the body along the joint left {len(fresh)} new edges, not two; "
            "the mid-curve of the joint does not go once round the piece"
        )

    def nearer(loop: list[int]) -> int:
        points = vertices[np.asarray(loop, dtype=np.int64)]
        return int(
            np.argmin(
                [
                    float(
                        surface_distances(
                            points, vertices, ring_as_triangles(ring)
                        ).mean()
                    )
                    for ring in rings
                ]
            )
        )

    sides = [nearer(loop) for loop in fresh]
    if sorted(sides) != [0, 1]:
        raise ArtifactMeshRepairError(
            "both new edges of the body came out nearer the same ring; the joint "
            "cannot be told apart at this reach"
        )
    paired = {side: loop for side, loop in zip(sides, fresh)}

    sewn: list[tuple[int, int, int]] = []
    for index, ring in enumerate(rings):
        sewn.extend(_sew(vertices, paired[index], list(ring)))
    after = np.concatenate([opened, np.asarray(sewn, dtype=np.int64)])
    return after, {
        "added_face_count": len(sewn),
        "kind": SEW_RING_PAIR_TO_BODY,
        "reach_um": int(round(reach_mm * 1000.0)),
        "removed_face_count": int(buried.size),
        "searched_face_count": int(neighbourhood.size),
    }


def sew_ring_to_ring(
    vertices_mm: np.ndarray,
    faces: np.ndarray,
    *,
    first_ring: Sequence[int],
    second_ring: Sequence[int],
    reach_um: int = DEFAULT_REACH_UM,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sew two open edges that face each other straight together.

    Nothing is cut: where a scanner could see either side of a thin edge but
    not over it, both sides are real surface and the only thing missing is
    the strip between them.  The two rings are required to face each other
    all the way round before anything is written, so that this cannot be
    used to pull two unrelated edges together across a pot.
    """

    vertices = np.asarray(vertices_mm, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    reach_mm = (
        _positive_int(
            reach_um, name="reach_um", minimum=MIN_REACH_UM, maximum=MAX_REACH_UM
        )
        / 1000.0
    )
    first = np.asarray(first_ring, dtype=np.int64)
    second = np.asarray(second_ring, dtype=np.int64)

    gaps = np.concatenate(
        [
            surface_distances(vertices[first], vertices, ring_as_triangles(second)),
            surface_distances(vertices[second], vertices, ring_as_triangles(first)),
        ]
    )
    if float(gaps.max()) > reach_mm:
        raise ArtifactMeshRepairError(
            f"the two edges are up to {float(gaps.max()):.2f} mm apart, beyond the "
            f"{reach_mm:g} mm reach; they do not face each other all the way round"
        )

    sewn = _sew(vertices, list(first), list(second))
    after = np.concatenate([triangles, np.asarray(sewn, dtype=np.int64)])
    return after, {
        "added_face_count": len(sewn),
        "kind": SEW_RING_TO_RING,
        "maximum_gap_um": int(round(float(gaps.max()) * 1000.0)),
        "median_gap_um": int(round(float(np.median(gaps)) * 1000.0)),
        "reach_um": int(round(reach_mm * 1000.0)),
        "removed_face_count": 0,
    }


def _edge_health(faces: np.ndarray) -> tuple[int, int]:
    """Edges held by more than two faces, and edges the two faces disagree on.

    The second is what catches a seam sewn with a twist in it: a band whose
    triangles are wound against the faces they were sewn to is still edge
    manifold, and only the direction each of the two faces walks the shared
    edge says that one of them is inside out.
    """

    directed = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]
    )
    sign = np.where(directed[:, 0] < directed[:, 1], 1, -1).astype(np.int64)
    key = np.sort(directed, axis=1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    key, sign = key[order], sign[order]
    starts = np.concatenate(
        [[0], np.flatnonzero(np.any(key[1:] != key[:-1], axis=1)) + 1]
    )
    ends = np.concatenate([starts[1:], [key.shape[0]]])
    counts = ends - starts
    sums = np.add.reduceat(sign, starts)
    return (
        int(np.count_nonzero(counts > 2)),
        int(np.count_nonzero((counts == 2) & (sums != 0))),
    )


def apply_mesh_repair(
    vertices_mm: np.ndarray,
    faces: np.ndarray,
    joins: Sequence[RingPairJoin | RingToRingJoin | DropFaces],
) -> tuple[np.ndarray, MeshRepairReceipt]:
    """Carry out declared joins against this mesh, or refuse and say why.

    Each join names its rings and its body by hash, so a document opened
    against a mesh that is not the one the decision was taken on stops here
    rather than sewing whatever happens to be at that index.
    """

    vertices = np.asarray(vertices_mm, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if not joins:
        raise ArtifactMeshRepairError("a repair with no joins changes nothing")
    if len(joins) > MAX_REPAIR_STEPS:
        raise ArtifactMeshRepairError(
            f"{len(joins)} joins is beyond the {MAX_REPAIR_STEPS} one repair carries"
        )

    bodies_before = int(face_bodies(triangles).max()) + 1
    rings_before = len(boundary_rings(triangles))
    _, mismatched_before = _edge_health(triangles)
    removed = added = 0
    steps: list[Mapping[str, Any]] = []
    for join in joins:
        if join.kind not in REPAIR_KINDS:
            raise ArtifactMeshRepairError(f"unsupported repair kind: {join.kind!r}")
        if isinstance(join, DropFaces):
            selected = join.indices()
            if join.total_face_count != triangles.shape[0]:
                raise ArtifactMeshRepairError(
                    f"the patch was chosen on a mesh of {join.total_face_count} faces "
                    f"and this one has {triangles.shape[0]}; the repair was decided on "
                    "a different mesh"
                )
            if face_set_sha256(triangles, selected) != join.selection_sha256:
                raise ArtifactMeshRepairError(
                    "the faces at the declared indices are not the declared patch; "
                    "the repair was decided on a different mesh"
                )
            triangles, stats = drop_faces(vertices, triangles, selected)
            removed += int(stats["removed_face_count"])
            added += int(stats["added_face_count"])
            steps.append({**join.to_dict(), **stats})
            continue

        rings = {ring_sha256(ring): ring for ring in boundary_rings(triangles)}
        missing = [
            name
            for name, digest in (
                ("first ring", join.first_ring_sha256),
                ("second ring", join.second_ring_sha256),
            )
            if digest not in rings
        ]
        if missing:
            raise ArtifactMeshRepairError(
                f"this mesh has no {' and no '.join(missing)} of the declared join; "
                "the repair was decided on a different mesh"
            )
        if isinstance(join, RingToRingJoin):
            triangles, stats = sew_ring_to_ring(
                vertices,
                triangles,
                first_ring=rings[join.first_ring_sha256],
                second_ring=rings[join.second_ring_sha256],
                reach_um=join.reach_um,
            )
            removed += int(stats["removed_face_count"])
            added += int(stats["added_face_count"])
            steps.append({**join.to_dict(), **stats})
            continue
        labels = face_bodies(triangles)
        bodies = {
            body_sha256(triangles, np.flatnonzero(labels == index)): np.flatnonzero(
                labels == index
            )
            for index in range(int(labels.max()) + 1)
        }
        if join.body_sha256 not in bodies:
            raise ArtifactMeshRepairError(
                "this mesh has no body of the declared join; the repair was decided "
                "on a different mesh"
            )
        triangles, stats = sew_ring_pair_to_body(
            vertices,
            triangles,
            first_ring=rings[join.first_ring_sha256],
            second_ring=rings[join.second_ring_sha256],
            body_faces=bodies[join.body_sha256],
            reach_um=join.reach_um,
        )
        removed += int(stats["removed_face_count"])
        added += int(stats["added_face_count"])
        steps.append({**join.to_dict(), **stats})

    non_manifold, mismatched = _edge_health(triangles)
    if non_manifold:
        raise ArtifactMeshRepairError(
            f"the sewn mesh has {non_manifold} edges held by more than two faces; "
            "the repair is discarded rather than recorded"
        )
    if mismatched > mismatched_before:
        raise ArtifactMeshRepairError(
            f"the seam left {mismatched - mismatched_before} edges whose two faces "
            "wind against each other; the band was sewn with a twist in it and the "
            "repair is discarded rather than recorded"
        )
    receipt = MeshRepairReceipt(
        schema_version=MESH_REPAIR_SCHEMA_VERSION,
        steps=tuple(steps),
        removed_face_count=removed,
        added_face_count=added,
        body_count_before=bodies_before,
        body_count_after=int(face_bodies(triangles).max()) + 1,
        open_ring_count_before=rings_before,
        open_ring_count_after=len(boundary_rings(triangles)),
        non_manifold_edge_count_after=non_manifold,
    )
    return triangles, receipt


__all__ = [
    "ArtifactMeshRepairError",
    "DEFAULT_REACH_UM",
    "MESH_REPAIR_EXTENSION_KEY",
    "MESH_REPAIR_SCHEMA_VERSION",
    "MeshRepairReceipt",
    "REPAIR_KINDS",
    "DROP_FACES",
    "DropFaces",
    "RingPairJoin",
    "RingToRingJoin",
    "SEW_RING_PAIR_TO_BODY",
    "SEW_RING_TO_RING",
    "apply_mesh_repair",
    "body_sha256",
    "ring_sha256",
    "drop_faces",
    "face_set_sha256",
    "sew_ring_pair_to_body",
    "sew_ring_to_ring",
]