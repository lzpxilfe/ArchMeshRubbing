"""갈라져 들어온 스캔: how many bodies arrived, and what each open edge faces.

A scanner's head cannot reach into an inner corner.  Where a boss, a handle
or a lug meets the wall it stands on, the reconstruction is left with a band
of shadow a millimetre or two wide running all the way round the joint, and
what comes out of the scanner is not one artifact but several closed or
half-closed sheets that merely stand inside one another.

Nothing downstream notices.  A pot that arrived in three pieces takes a
rotation axis, a section and a rubbing exactly as a whole one does, and the
two numbers a mesh is usually judged by - no boundary edges, no non-manifold
edges - are *both perfect* for two closed bodies sitting apart.  The section
comes out as two or three closed loops instead of one, which is the only
place it shows, and a loop count is easy to read as a formality.

So this module counts the bodies and names what each open ring lies against,
and it is careful to do one thing it would be easy not to do: **it does not
choose.**  On 24ET0021 the boss's upper skin ended 0.2-2.3 mm from the wall
and about 2.4 mm from the boss's own lower skin.  Both are inside any
sensible tolerance, and sewing the two skins to each other closes every
boundary, passes every check, and leaves the boss standing loose in the bowl
with the drawing quietly wrong.  A rule that picks the nearest candidate
picks that one.  The candidates are therefore all reported, ranked, and left
to the archaeologist, who is the one who can look at the pot.

The report is a finding about the file, not a measurement of the artifact,
so it is not a record.  What becomes a record is the decision taken about
it, and the repair that decision authorises.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .artifact_cancellation import CancellationProbe, raise_if_cancelled


MESH_BODIES_SCHEMA_VERSION = "1.0.0"

#: An open ring can be sewn to the *surface* of another body - the boss
#: standing on a wall, where the wall's skin runs on past the joint - or to
#: another open ring, which is the shape a rim has when the scan stopped on
#: both sides of a thin edge.  The two need different repairs, so they are
#: named apart rather than pooled as "something nearby".
JOIN_TO_BODY_SURFACE = "body_surface"
JOIN_TO_OPEN_RING = "open_ring"
JOIN_TARGET_KINDS: tuple[str, ...] = (JOIN_TO_BODY_SURFACE, JOIN_TO_OPEN_RING)

#: What the ring's candidate list amounts to.  "several" is not a failure of
#: the diagnosis; it is the diagnosis.
RING_UNATTACHED = "unattached"
RING_ONE_CANDIDATE = "one_candidate"
RING_SEVERAL_CANDIDATES = "several_candidates"
RING_VERDICTS: tuple[str, ...] = (
    RING_UNATTACHED,
    RING_ONE_CANDIDATE,
    RING_SEVERAL_CANDIDATES,
)

#: A joint's shadow is as wide as the scanner's head could not reach, which
#: on a hand-held scanner is a millimetre or two.  Three leaves room for a
#: coarse scan without sweeping in the far wall of a small pot.
DEFAULT_SEAM_TOLERANCE_UM = 3_000
MIN_SEAM_TOLERANCE_UM = 1
MAX_SEAM_TOLERANCE_UM = 1_000_000

#: A ring lying against a surface is close to it *all the way round*, not on
#: average: one arc of it resting on the wall while the rest wanders off is a
#: broken edge, not a joint.  The worst point is allowed this multiple of the
#: tolerance before the candidate is dropped.
SEAM_MAXIMUM_MULTIPLE = 2.0

#: Ranked candidates kept per ring.  Beyond a handful the list stops being a
#: thing a person reads and becomes a thing they skim.
MAX_RING_CANDIDATES = 4

MAX_DIAGNOSED_FACES = 4_000_000
MAX_DIAGNOSED_BODIES = 4096
MAX_DIAGNOSED_RINGS = 4096

#: Every point of every open ring is dropped onto every other body and every
#: other ring, so the work is the product of the two and a mesh shredded into
#: thousands of scraps would run for hours.  A real scan is nowhere near
#: this; a file that is comes back with a refusal rather than a hang.
MAX_GAP_POINT_TESTS = 20_000_000


class ArtifactMeshBodiesError(ValueError):
    """The mesh cannot be diagnosed as it stands."""


def _validated(
    vertices_mm: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices_mm, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ArtifactMeshBodiesError("vertices must be an (n, 3) array of millimetres")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ArtifactMeshBodiesError("faces must be an (m, 3) array of triangles")
    if triangles.shape[0] == 0:
        raise ArtifactMeshBodiesError("a mesh with no faces has nothing to diagnose")
    if triangles.shape[0] > MAX_DIAGNOSED_FACES:
        raise ArtifactMeshBodiesError(
            f"{triangles.shape[0]} faces is beyond the {MAX_DIAGNOSED_FACES} this diagnosis reads"
        )
    if not np.isfinite(vertices).all():
        raise ArtifactMeshBodiesError("vertices must all be finite")
    if triangles.min(initial=0) < 0 or triangles.max(initial=-1) >= vertices.shape[0]:
        raise ArtifactMeshBodiesError("a face names a vertex the mesh does not have")
    return vertices, triangles


def _seam_tolerance(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactMeshBodiesError("seam_tolerance_um must be an integer")
    if not MIN_SEAM_TOLERANCE_UM <= value <= MAX_SEAM_TOLERANCE_UM:
        raise ArtifactMeshBodiesError(
            f"seam_tolerance_um must be between {MIN_SEAM_TOLERANCE_UM} and "
            f"{MAX_SEAM_TOLERANCE_UM}, not {value}"
        )
    return value


def _edge_groups(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Undirected edges sorted into groups, with the face each copy came from."""

    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    owner = np.tile(np.arange(faces.shape[0], dtype=np.int64), 3)
    key = np.sort(directed, axis=1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    return key[order], owner[order], directed[order]


def face_bodies(faces: np.ndarray) -> np.ndarray:
    """A body label per face, joined across shared edges.

    Edges, not vertices: two sheets that touch at a single point are two
    bodies, and calling them one would hide exactly the case this module
    exists for.
    """

    triangles = np.asarray(faces, dtype=np.int64)
    key, owner, _ = _edge_groups(triangles)
    if key.shape[0] < 2:
        return np.zeros(triangles.shape[0], dtype=np.int64)
    same = np.all(key[1:] == key[:-1], axis=1)
    rows = owner[1:][same]
    cols = owner[:-1][same]
    count = int(triangles.shape[0])
    graph = coo_matrix(
        (np.ones(rows.shape[0], dtype=np.int8), (rows, cols)), shape=(count, count)
    )
    _, labels = connected_components(graph, directed=False)
    return np.asarray(labels, dtype=np.int64)


def _open_edges(faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The directed edges only one face holds, and the face that holds each."""

    key, owner, directed = _edge_groups(faces)
    if key.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.int64)
    starts = np.concatenate(
        [[0], np.flatnonzero(np.any(key[1:] != key[:-1], axis=1)) + 1]
    )
    ends = np.concatenate([starts[1:], [key.shape[0]]])
    single = starts[(ends - starts) == 1]
    return directed[single], owner[single]


def _rings_from_open_edges(
    open_edges: np.ndarray, owners: np.ndarray
) -> tuple[list[list[int]], list[int]]:
    """Walk the open edges into loops, keeping the face each loop started on."""

    if open_edges.shape[0] == 0:
        return [], []
    nxt: dict[int, int] = {}
    owner_of: dict[int, int] = {}
    for (tail, head), face in zip(open_edges, owners):
        tail_index = int(tail)
        if tail_index in nxt:
            raise ArtifactMeshBodiesError(
                f"vertex {tail_index} starts two boundary edges; the open edge pinches "
                "there and how to split it is a judgement about the artifact"
            )
        nxt[tail_index] = int(head)
        owner_of[tail_index] = int(face)

    rings: list[list[int]] = []
    ring_owners: list[int] = []
    seen: set[int] = set()
    for start in nxt:
        if start in seen:
            continue
        ring: list[int] = []
        cursor: int | None = start
        while cursor is not None and cursor not in seen:
            seen.add(cursor)
            ring.append(cursor)
            cursor = nxt.get(cursor)
        if len(ring) > 2:
            rings.append(ring)
            ring_owners.append(owner_of[start])
        if len(rings) > MAX_DIAGNOSED_RINGS:
            raise ArtifactMeshBodiesError(
                f"more than {MAX_DIAGNOSED_RINGS} open rings; this is a mesh to repair "
                "in a mesh tool before it is measured"
            )
    return rings, ring_owners


def boundary_rings(faces: np.ndarray) -> list[list[int]]:
    """Ordered vertex loops of the edges that only one face holds.

    Each loop follows the winding of the faces it borders, which is what a
    later repair needs in order to sew new triangles the right way round.  A
    boundary that passes through one vertex twice is refused rather than cut
    arbitrarily: which way it should be split is a judgement about the
    artifact, and guessing it here would be the same mistake as guessing a
    joint.
    """

    triangles = np.asarray(faces, dtype=np.int64)
    return _rings_from_open_edges(*_open_edges(triangles))[0]


@dataclass(frozen=True)
class MeshBody:
    """One edge-connected sheet of the file."""

    index: int
    face_count: int
    vertex_count: int
    open_ring_count: int
    closed: bool
    minimum_um: tuple[int, int, int]
    maximum_um: tuple[int, int, int]

    def qc_dict(self) -> dict[str, Any]:
        return {
            "closed": self.closed,
            "face_count": self.face_count,
            "index": self.index,
            "maximum_um": list(self.maximum_um),
            "minimum_um": list(self.minimum_um),
            "open_ring_count": self.open_ring_count,
            "vertex_count": self.vertex_count,
        }


@dataclass(frozen=True)
class RingCandidate:
    """Something an open ring could be sewn to, and how far away it is."""

    kind: str
    body_index: int
    ring_index: int | None
    median_gap_um: int
    maximum_gap_um: int

    def qc_dict(self) -> dict[str, Any]:
        return {
            "body_index": self.body_index,
            "kind": self.kind,
            "maximum_gap_um": self.maximum_gap_um,
            "median_gap_um": self.median_gap_um,
            "ring_index": self.ring_index,
        }


@dataclass(frozen=True)
class OpenRing:
    """One closed loop of open edge, and what it might belong to."""

    index: int
    body_index: int
    point_count: int
    verdict: str
    candidates: tuple[RingCandidate, ...]
    minimum_um: tuple[int, int, int]
    maximum_um: tuple[int, int, int]

    def qc_dict(self) -> dict[str, Any]:
        return {
            "body_index": self.body_index,
            "candidates": [candidate.qc_dict() for candidate in self.candidates],
            "index": self.index,
            "maximum_um": list(self.maximum_um),
            "minimum_um": list(self.minimum_um),
            "point_count": self.point_count,
            "verdict": self.verdict,
        }


@dataclass(frozen=True)
class MeshBodyReport:
    """What the file turned out to be."""

    schema_version: str
    seam_tolerance_um: int
    bodies: tuple[MeshBody, ...]
    open_rings: tuple[OpenRing, ...]

    @property
    def one_body(self) -> bool:
        return len(self.bodies) == 1

    @property
    def needs_decision(self) -> bool:
        """Whether a person has to look before this mesh is measured.

        More than one body always does.  So does a single body carrying an
        open ring that lies against something: a pot cut open by the scanner
        and a pot with a joint the scanner could not see look the same from
        here, and only the artifact settles it.
        """

        if not self.one_body:
            return True
        return any(ring.candidates for ring in self.open_rings)

    def qc_dict(self) -> dict[str, Any]:
        return {
            "body_count": len(self.bodies),
            "bodies": [body.qc_dict() for body in self.bodies],
            "needs_decision": self.needs_decision,
            "open_ring_count": len(self.open_rings),
            "open_rings": [ring.qc_dict() for ring in self.open_rings],
            "schema_version": self.schema_version,
            "seam_tolerance_um": self.seam_tolerance_um,
        }


def _micrometres(value: np.ndarray) -> tuple[int, int, int]:
    scaled = np.rint(np.asarray(value, dtype=np.float64) * 1000.0)
    return (int(scaled[0]), int(scaled[1]), int(scaled[2]))


def _point_triangle_distance(
    point: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> np.ndarray:
    """Distance from one point to each of many triangles.

    The seven regions of a triangle's plane - three vertices, three edges,
    the face - each have their own closest point, and the barycentric test
    that picks between them is Ericson's.  Written out because the distance
    that matters here is to the *surface*: a wall drawn with two rows of
    vertices twenty millimetres apart is still one millimetre away from a rim
    resting against its middle, and measuring to the nearest vertex would
    call that joint a gap of twenty.
    """

    ab = b - a
    ac = c - a
    ap = point - a
    d1 = np.einsum("ij,ij->i", ab, ap)
    d2 = np.einsum("ij,ij->i", ac, ap)
    bp = point - b
    d3 = np.einsum("ij,ij->i", ab, bp)
    d4 = np.einsum("ij,ij->i", ac, bp)
    cp = point - c
    d5 = np.einsum("ij,ij->i", ab, cp)
    d6 = np.einsum("ij,ij->i", ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2

    def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        safe = np.where(np.abs(denominator) < 1e-30, 1.0, denominator)
        return np.clip(numerator / safe, 0.0, 1.0)

    closest = np.empty_like(a)
    taken = np.zeros(a.shape[0], dtype=bool)

    def _claim(mask: np.ndarray, value: np.ndarray) -> None:
        fresh = mask & ~taken
        closest[fresh] = value[fresh]
        taken[fresh] = True

    _claim((d1 <= 0.0) & (d2 <= 0.0), a)
    _claim((d3 >= 0.0) & (d4 <= d3), b)
    _claim(
        (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0),
        a + ab * _ratio(d1, d1 - d3)[:, None],
    )
    _claim((d6 >= 0.0) & (d5 <= d6), c)
    _claim(
        (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0),
        a + ac * _ratio(d2, d2 - d6)[:, None],
    )
    _claim(
        (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0),
        b + (c - b) * _ratio(d4 - d3, (d4 - d3) + (d5 - d6))[:, None],
    )
    total = va + vb + vc
    safe_total = np.where(np.abs(total) < 1e-30, 1.0, total)
    _claim(
        np.ones(a.shape[0], dtype=bool),
        a + ab * (vb / safe_total)[:, None] + ac * (vc / safe_total)[:, None],
    )
    return np.linalg.norm(point - closest, axis=1)


@dataclass(frozen=True)
class _Surface:
    """A body's triangles, indexed so a point can be dropped onto them."""

    corners: tuple[np.ndarray, np.ndarray, np.ndarray]
    tree: cKDTree
    vertex_ids: np.ndarray
    incident_starts: np.ndarray
    incident_faces: np.ndarray


def _surface(vertices: np.ndarray, triangles: np.ndarray) -> _Surface:
    used = np.unique(triangles)
    local = np.searchsorted(used, triangles)
    owner = np.repeat(np.arange(triangles.shape[0], dtype=np.int64), 3)
    order = np.argsort(local.reshape(-1), kind="stable")
    counts = np.bincount(local.reshape(-1), minlength=used.shape[0])
    return _Surface(
        corners=(
            vertices[triangles[:, 0]],
            vertices[triangles[:, 1]],
            vertices[triangles[:, 2]],
        ),
        tree=cKDTree(vertices[used]),
        vertex_ids=used,
        incident_starts=np.concatenate([[0], np.cumsum(counts)]),
        incident_faces=owner[order],
    )


#: Nearest vertices whose triangles are tried when dropping a point onto a
#: surface.  The closest triangle is all but always one of these; a mesh
#: where it is not mixes triangles so unlike in size that the diagnosis would
#: be reading noise anyway.
GAP_NEIGHBOUR_VERTICES = 12


def ring_as_triangles(ring: np.ndarray) -> np.ndarray:
    """A closed ring of vertices written as triangles, for distance work.

    Each segment becomes a triangle with two corners the same, which the
    point-to-triangle routine reduces to the point-to-segment case, so a ring
    and a surface are measured against by one piece of code.
    """

    indices = np.asarray(ring, dtype=np.int64).reshape(-1)
    rolled = np.roll(indices, -1)
    return np.column_stack([indices, rolled, rolled])


def _distances(points: np.ndarray, surface: _Surface) -> np.ndarray:
    """Distance from each point to the nearest point on the surface."""

    neighbours = min(GAP_NEIGHBOUR_VERTICES, surface.vertex_ids.shape[0])
    _, found = surface.tree.query(points, k=neighbours)
    # query drops the k axis when k is 1 and the point axis when there is one
    # point; both shapes have to come back as (points, neighbours).
    found = np.asarray(found).reshape(points.shape[0], -1)
    distances = np.empty(points.shape[0], dtype=np.float64)
    for index in range(points.shape[0]):
        faces: list[np.ndarray] = []
        for vertex in np.atleast_1d(found[index]):
            start = surface.incident_starts[vertex]
            end = surface.incident_starts[vertex + 1]
            faces.append(surface.incident_faces[start:end])
        candidates = np.unique(np.concatenate(faces))
        a, b, c = surface.corners
        distances[index] = float(
            _point_triangle_distance(
                points[index], a[candidates], b[candidates], c[candidates]
            ).min()
        )
    return distances


def surface_distances(
    points: np.ndarray, vertices: np.ndarray, triangles: np.ndarray
) -> np.ndarray:
    """Distance from each point to the nearest point on those triangles.

    Public because a repair needs the same measurement the diagnosis makes:
    which faces of a wall lie between two edges resting against it is a
    question about distance to the edges, not about any axis the artifact may
    or may not have.
    """

    return _distances(
        np.asarray(points, dtype=np.float64).reshape(-1, 3),
        _surface(
            np.asarray(vertices, dtype=np.float64),
            np.asarray(triangles, dtype=np.int64).reshape(-1, 3),
        ),
    )


def _gap(points: np.ndarray, surface: _Surface) -> tuple[int, int]:
    """Median and worst distance from a ring's points to a body's surface."""

    distances = _distances(points, surface)
    return (
        int(round(float(np.median(distances)) * 1000.0)),
        int(round(float(np.max(distances)) * 1000.0)),
    )


def diagnose_mesh_bodies(
    vertices_mm: np.ndarray,
    faces: np.ndarray,
    *,
    seam_tolerance_um: int = DEFAULT_SEAM_TOLERANCE_UM,
    cancellation_probe: CancellationProbe | None = None,
) -> MeshBodyReport:
    """Count the bodies in a mesh and say what each open ring faces."""

    vertices, triangles = _validated(vertices_mm, faces)
    tolerance_um = _seam_tolerance(seam_tolerance_um)
    worst_um = int(round(tolerance_um * SEAM_MAXIMUM_MULTIPLE))
    raise_if_cancelled(cancellation_probe)

    labels = face_bodies(triangles)
    body_count = int(labels.max()) + 1 if labels.size else 0
    if body_count > MAX_DIAGNOSED_BODIES:
        raise ArtifactMeshBodiesError(
            f"{body_count} separate bodies is beyond the {MAX_DIAGNOSED_BODIES} this "
            "diagnosis reads; the file is probably a scene rather than an artifact"
        )
    raise_if_cancelled(cancellation_probe)

    body_faces = [np.flatnonzero(labels == index) for index in range(body_count)]
    body_vertices = [np.unique(triangles[selected]) for selected in body_faces]
    body_surfaces = [_surface(vertices, triangles[selected]) for selected in body_faces]

    # A ring belongs to the body of the face that holds its edges, not to
    # whichever body happens to own its vertices: two sheets meeting at a
    # single vertex share that vertex while remaining two bodies.
    rings, ring_owner_faces = _rings_from_open_edges(*_open_edges(triangles))
    raise_if_cancelled(cancellation_probe)
    ring_points = [vertices[np.asarray(ring, dtype=np.int64)] for ring in rings]
    ring_body = [int(labels[face]) for face in ring_owner_faces]
    # A ring is measured against as a chain of segments, written as triangles
    # with two corners the same so that one distance routine serves both.
    ring_surfaces = [_surface(vertices, ring_as_triangles(ring)) for ring in rings]

    targets = max(body_count - 1, 0) + max(len(ring_points) - 1, 0)
    tests = sum(points.shape[0] for points in ring_points) * targets
    if tests > MAX_GAP_POINT_TESTS:
        raise ArtifactMeshBodiesError(
            f"{len(ring_points)} open rings against {body_count} bodies would take "
            f"{tests} measurements, beyond the {MAX_GAP_POINT_TESTS} this diagnosis "
            "makes; the file is shredded rather than merely split"
        )

    open_rings: list[OpenRing] = []
    for index, points in enumerate(ring_points):
        raise_if_cancelled(cancellation_probe)
        candidates: list[RingCandidate] = []
        for other in range(body_count):
            if other == ring_body[index]:
                continue
            median_um, maximum_um = _gap(points, body_surfaces[other])
            if median_um <= tolerance_um and maximum_um <= worst_um:
                candidates.append(
                    RingCandidate(
                        kind=JOIN_TO_BODY_SURFACE,
                        body_index=other,
                        ring_index=None,
                        median_gap_um=median_um,
                        maximum_gap_um=maximum_um,
                    )
                )
        for other in range(len(ring_points)):
            if other == index:
                continue
            median_um, maximum_um = _gap(points, ring_surfaces[other])
            if median_um <= tolerance_um and maximum_um <= worst_um:
                candidates.append(
                    RingCandidate(
                        kind=JOIN_TO_OPEN_RING,
                        body_index=ring_body[other],
                        ring_index=other,
                        median_gap_um=median_um,
                        maximum_gap_um=maximum_um,
                    )
                )
        candidates.sort(key=lambda item: (item.median_gap_um, item.maximum_gap_um))
        kept = tuple(candidates[:MAX_RING_CANDIDATES])
        if not kept:
            verdict = RING_UNATTACHED
        elif len(kept) == 1:
            verdict = RING_ONE_CANDIDATE
        else:
            verdict = RING_SEVERAL_CANDIDATES
        open_rings.append(
            OpenRing(
                index=index,
                body_index=ring_body[index],
                point_count=len(rings[index]),
                verdict=verdict,
                candidates=kept,
                minimum_um=_micrometres(points.min(axis=0)),
                maximum_um=_micrometres(points.max(axis=0)),
            )
        )

    ring_counts = np.zeros(body_count, dtype=np.int64)
    for ring in open_rings:
        ring_counts[ring.body_index] += 1
    bodies = tuple(
        MeshBody(
            index=index,
            face_count=int(body_faces[index].shape[0]),
            vertex_count=int(body_vertices[index].shape[0]),
            open_ring_count=int(ring_counts[index]),
            closed=bool(ring_counts[index] == 0),
            minimum_um=_micrometres(vertices[body_vertices[index]].min(axis=0)),
            maximum_um=_micrometres(vertices[body_vertices[index]].max(axis=0)),
        )
        for index in range(body_count)
    )
    return MeshBodyReport(
        schema_version=MESH_BODIES_SCHEMA_VERSION,
        seam_tolerance_um=tolerance_um,
        bodies=bodies,
        open_rings=tuple(open_rings),
    )


def describe_mesh_bodies(report: MeshBodyReport) -> tuple[str, ...]:
    """The report as lines to put in front of the person opening the file.

    Korean, because these lines are read at the moment a scan is opened and
    the parts of a pot are named here the way the drawing names them.
    """

    def _mm(value: int) -> str:
        return f"{value / 1000.0:.2f}"

    lines: list[str] = []
    if report.one_body:
        lines.append("이 파일은 몸 하나입니다.")
    else:
        lines.append(
            f"이 파일은 유물 하나가 아니라 서로 붙어 있지 않은 몸 {len(report.bodies)}개입니다."
        )
    for body in report.bodies:
        state = "닫힘" if body.closed else f"열린 고리 {body.open_ring_count}개"
        lines.append(
            f"  · 몸 {body.index + 1}: 면 {body.face_count:,}개 · "
            f"z {body.minimum_um[2] / 1000.0:.1f}-{body.maximum_um[2] / 1000.0:.1f} mm · {state}"
        )

    joined = [ring for ring in report.open_rings if ring.candidates]
    if joined:
        lines.append("")
        lines.append(
            f"열린 자리 {len(joined)}곳이 무언가에 닿아 있습니다. "
            "어디에 이을지는 실측자가 정합니다."
        )
        for ring in joined:
            lines.append(f"  · 고리 {ring.index + 1} (몸 {ring.body_index + 1} · 점 {ring.point_count}개)")
            for candidate in ring.candidates:
                target = (
                    f"몸 {candidate.body_index + 1}의 면"
                    if candidate.ring_index is None
                    else f"고리 {candidate.ring_index + 1}"
                )
                lines.append(
                    f"      {target}에 {_mm(candidate.median_gap_um)} mm "
                    f"(가장 먼 점 {_mm(candidate.maximum_gap_um)} mm)"
                )
    if not report.one_body:
        lines.append("")
        lines.append(
            "경계 모서리와 비다양체 수가 0이어도 몸이 여럿이면 도면은 갈라진 채로 그려집니다. "
            "단면이 닫힌 고리 여러 개로 나오는 것이 그 증상입니다."
        )
    return tuple(lines)
