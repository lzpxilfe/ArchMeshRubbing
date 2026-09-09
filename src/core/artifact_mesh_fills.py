"""메우기: the flat lids someone put over the holes before we got the file.

A scanner leaves a hole wherever its head could not see - the underside of a
pedestal, the top of a knob, the inside of a joint.  Modelling software will
close such a hole on request, and most files that reach a drafter have been
through that step, because a watertight mesh is what a print or a viewer
wants.  The result is surface that looks exactly like measured surface and
is not: it is the software's guess at what the artifact does where nobody
looked.

Drawn without comment, that guess becomes a fact.  On 24ET0021 the two flat
lids over the top of the knob put a solid 2.6 mm slab across the section,
and the drawing then says the knob is closed - which is a claim about the
pot that no one measured.

The fills are recognisable because of how they are made.  Hole-closing runs
a fan: one new point in the middle of the hole, one triangle to every edge
of its rim.  That gives a patch where

  * every face touches one shared point,
  * the faces number exactly as many as the rim's points, and
  * the faces are all *the same size*, because they were laid out from one
    centre rather than measured one at a time.

The third is the one that cannot be faked by a scan.  A scanner's triangles
vary in area by tens of percent from one to the next even on a flat wall; a
fan whose areas agree to within a fraction of a percent was computed.  This
module looks for that signature, reports what it finds with the numbers it
found it by, and stops there.  Removing a fill is a repair, and repairs are
the archaeologist's to authorise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .artifact_cancellation import CancellationProbe, poll_cancellation, raise_if_cancelled


MESH_FILLS_SCHEMA_VERSION = "1.0.0"

#: A fan needs enough triangles for "they are all the same size" to mean
#: anything.  Below about a dozen, ordinary scan triangles round a small
#: crater can agree by chance.
DEFAULT_MINIMUM_FAN_FACES = 12
MIN_MINIMUM_FAN_FACES = 6
MAX_MINIMUM_FAN_FACES = 4096

#: How far the faces of a fan may disagree in area, as a fraction of their
#: mean, in millionths.  This is the whole discriminator, so it is set from
#: what the two kinds of surface actually measure: the four fills in
#: 24ET0021 come out at 1-3 millionths, and the flattest patch of scanned
#: wall in the same file at 23,000.  A thousand sits four hundred times
#: above the fills and twenty times below the wall.
DEFAULT_AREA_SPREAD_MILLIONTHS = 1_000
MIN_AREA_SPREAD_MILLIONTHS = 1
MAX_AREA_SPREAD_MILLIONTHS = 1_000_000

#: A fill spans in one triangle what the scan spends many on, because the
#: hole it covers is exactly where no detail was recorded.  Patches no
#: coarser than the mesh around them are left alone: they are more likely a
#: flat facet of the artifact than an invention.
DEFAULT_COARSENESS_MILLIONTHS = 2_000_000
MIN_COARSENESS_MILLIONTHS = 1_000_000
MAX_COARSENESS_MILLIONTHS = 1_000_000_000

MAX_SEARCHED_FACES = 8_000_000
MAX_REPORTED_FILLS = 64


class ArtifactMeshFillsError(ValueError):
    """The mesh cannot be searched for fills as it stands."""


def _validated(
    vertices_mm: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices_mm, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ArtifactMeshFillsError("vertices must be an (n, 3) array of millimetres")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ArtifactMeshFillsError("faces must be an (m, 3) array of triangles")
    if triangles.shape[0] == 0:
        raise ArtifactMeshFillsError("a mesh with no faces has nothing to search")
    if triangles.shape[0] > MAX_SEARCHED_FACES:
        raise ArtifactMeshFillsError(
            f"{triangles.shape[0]} faces is beyond the {MAX_SEARCHED_FACES} this search reads"
        )
    if not np.isfinite(vertices).all():
        raise ArtifactMeshFillsError("vertices must all be finite")
    if triangles.min(initial=0) < 0 or triangles.max(initial=-1) >= vertices.shape[0]:
        raise ArtifactMeshFillsError("a face names a vertex the mesh does not have")
    return vertices, triangles


def _bounded_int(value: object, *, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactMeshFillsError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ArtifactMeshFillsError(
            f"{name} must be between {minimum} and {maximum}, not {value}"
        )
    return value


def face_areas_mm2(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """The area of every triangle, in square millimetres."""

    corners = vertices[faces]
    return 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
        axis=1,
    )


def _faces_by_vertex(
    faces: np.ndarray, vertex_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each vertex, the slice of a face list holding the faces that use it."""

    order = np.argsort(faces.ravel(), kind="stable")
    used = faces.ravel()[order]
    owners = (order // 3).astype(np.int64)
    starts = np.searchsorted(used, np.arange(vertex_count, dtype=np.int64))
    ends = np.searchsorted(used, np.arange(vertex_count, dtype=np.int64), side="right")
    return owners, starts, ends


@dataclass(frozen=True)
class FilledHole:
    """One fan of triangles that closes a hole rather than measuring a surface."""

    apex_vertex: int
    face_count: int
    rim_point_count: int
    apex_um: tuple[int, int, int]
    mean_area_um2: int
    mesh_median_area_um2: int
    coarseness_millionths: int
    area_spread_millionths: int
    flatness_um: int
    span_um: int

    def qc_dict(self) -> dict[str, Any]:
        return {
            "apex_um": list(self.apex_um),
            "apex_vertex": self.apex_vertex,
            "area_spread_millionths": self.area_spread_millionths,
            "coarseness_millionths": self.coarseness_millionths,
            "face_count": self.face_count,
            "flatness_um": self.flatness_um,
            "mean_area_um2": self.mean_area_um2,
            "mesh_median_area_um2": self.mesh_median_area_um2,
            "rim_point_count": self.rim_point_count,
            "span_um": self.span_um,
        }


@dataclass(frozen=True)
class MeshFillReport:
    """What the file turned out to have had closed for it."""

    schema_version: str
    minimum_fan_faces: int
    area_spread_millionths: int
    coarseness_millionths: int
    mesh_median_area_um2: int
    searched_face_count: int
    fills: tuple[FilledHole, ...]

    @property
    def filled_face_count(self) -> int:
        return sum(fill.face_count for fill in self.fills)

    @property
    def needs_decision(self) -> bool:
        return bool(self.fills)

    def qc_dict(self) -> dict[str, Any]:
        return {
            "area_spread_millionths": self.area_spread_millionths,
            "coarseness_millionths": self.coarseness_millionths,
            "fill_count": len(self.fills),
            "filled_face_count": self.filled_face_count,
            "fills": [fill.qc_dict() for fill in self.fills],
            "mesh_median_area_um2": self.mesh_median_area_um2,
            "minimum_fan_faces": self.minimum_fan_faces,
            "schema_version": self.schema_version,
            "searched_face_count": self.searched_face_count,
        }


def _micrometres(value: np.ndarray) -> tuple[int, int, int]:
    scaled = np.rint(np.asarray(value, dtype=np.float64) * 1000.0)
    return (int(scaled[0]), int(scaled[1]), int(scaled[2]))


def _is_single_fan(faces: np.ndarray, apex: int) -> bool:
    """Whether the faces round one point form a closed fan, each used once.

    A hole-closing fan walks the rim once: every face contributes one rim
    edge, and those edges chain into a single loop.  A point in the middle of
    ordinary surface does the same, which is why the fan alone is not the
    finding - but a patch that is *not* a closed fan is certainly not one of
    these fills, and dropping it here keeps the report honest.
    """

    rim: list[tuple[int, int]] = []
    for face in faces:
        first, second, third = (int(index) for index in face)
        if first == apex:
            rim.append((second, third))
        elif second == apex:
            rim.append((third, first))
        elif third == apex:
            rim.append((first, second))
        else:
            return False
    forward = dict(rim)
    if len(forward) != len(rim):
        return False
    start = rim[0][0]
    cursor = start
    for _ in range(len(rim)):
        following = forward.get(cursor)
        if following is None:
            return False
        cursor = following
    return cursor == start


def _flatness_mm(points: np.ndarray) -> float:
    """How far the fan's rim strays from the best plane through it."""

    centred = points - points.mean(axis=0)
    if centred.shape[0] < 3:
        return 0.0
    _, _, right = np.linalg.svd(centred, full_matrices=False)
    normal = right[-1]
    return float(np.abs(centred @ normal).max())


def find_filled_holes(
    vertices_mm: np.ndarray,
    faces: np.ndarray,
    *,
    minimum_fan_faces: int = DEFAULT_MINIMUM_FAN_FACES,
    area_spread_millionths: int = DEFAULT_AREA_SPREAD_MILLIONTHS,
    coarseness_millionths: int = DEFAULT_COARSENESS_MILLIONTHS,
    cancellation_probe: CancellationProbe | None = None,
) -> MeshFillReport:
    """Find the fans that close holes, and report them with their evidence."""

    raise_if_cancelled(cancellation_probe)
    vertices, triangles = _validated(vertices_mm, faces)
    fan_faces = _bounded_int(
        minimum_fan_faces,
        name="minimum_fan_faces",
        minimum=MIN_MINIMUM_FAN_FACES,
        maximum=MAX_MINIMUM_FAN_FACES,
    )
    spread_limit = _bounded_int(
        area_spread_millionths,
        name="area_spread_millionths",
        minimum=MIN_AREA_SPREAD_MILLIONTHS,
        maximum=MAX_AREA_SPREAD_MILLIONTHS,
    )
    coarse_limit = _bounded_int(
        coarseness_millionths,
        name="coarseness_millionths",
        minimum=MIN_COARSENESS_MILLIONTHS,
        maximum=MAX_COARSENESS_MILLIONTHS,
    )

    areas = face_areas_mm2(vertices, triangles)
    raise_if_cancelled(cancellation_probe)
    mesh_median = float(np.median(areas))
    if mesh_median <= 0.0:
        raise ArtifactMeshFillsError(
            "half the faces of this mesh have no area; it cannot be searched for fills"
        )

    owners, starts, ends = _faces_by_vertex(triangles, vertices.shape[0])
    raise_if_cancelled(cancellation_probe)
    counts = ends - starts
    candidates = np.flatnonzero(counts >= fan_faces)
    raise_if_cancelled(cancellation_probe)

    found: list[FilledHole] = []
    for step, apex in enumerate(candidates):
        poll_cancellation(cancellation_probe, step)
        fan = owners[starts[apex] : ends[apex]]
        fan_areas = areas[fan]
        mean_area = float(fan_areas.mean())
        if mean_area <= 0.0:
            continue
        spread = float(fan_areas.std() / mean_area)
        if spread * 1_000_000.0 > spread_limit:
            continue
        if mean_area * 1_000_000.0 < coarse_limit * mesh_median:
            continue
        fan_triangles = triangles[fan]
        rim = np.setdiff1d(np.unique(fan_triangles), np.asarray([apex]))
        if rim.size != fan.size:
            continue
        if not _is_single_fan(fan_triangles, int(apex)):
            continue
        rim_points = vertices[rim]
        found.append(
            FilledHole(
                apex_vertex=int(apex),
                face_count=int(fan.size),
                rim_point_count=int(rim.size),
                apex_um=_micrometres(vertices[apex]),
                mean_area_um2=int(round(mean_area * 1_000_000.0)),
                mesh_median_area_um2=int(round(mesh_median * 1_000_000.0)),
                coarseness_millionths=int(round(mean_area / mesh_median * 1_000_000.0)),
                area_spread_millionths=int(round(spread * 1_000_000.0)),
                flatness_um=int(round(_flatness_mm(rim_points) * 1000.0)),
                span_um=int(
                    round(
                        float(
                            np.linalg.norm(
                                rim_points.max(axis=0) - rim_points.min(axis=0)
                            )
                        )
                        * 1000.0
                    )
                ),
            )
        )
        if len(found) > MAX_REPORTED_FILLS:
            raise ArtifactMeshFillsError(
                f"more than {MAX_REPORTED_FILLS} filled holes; this is a mesh to look "
                "at in a mesh tool before it is measured"
            )

    found.sort(key=lambda fill: (-fill.face_count, fill.apex_vertex))
    raise_if_cancelled(cancellation_probe)
    return MeshFillReport(
        schema_version=MESH_FILLS_SCHEMA_VERSION,
        minimum_fan_faces=fan_faces,
        area_spread_millionths=spread_limit,
        coarseness_millionths=coarse_limit,
        mesh_median_area_um2=int(round(mesh_median * 1_000_000.0)),
        searched_face_count=int(triangles.shape[0]),
        fills=tuple(found),
    )


def fill_face_indices(faces: np.ndarray, fill: FilledHole) -> np.ndarray:
    """The faces of one reported fill, as indices into the mesh it was found in.

    Handed to `drop_faces` this is the whole of the repair: the fan comes
    away and the hole it covered is open again, which is the state the
    scanner actually left the artifact in.
    """

    triangles = np.asarray(faces, dtype=np.int64)
    if not isinstance(fill, FilledHole):
        raise ArtifactMeshFillsError("fill must be a FilledHole")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ArtifactMeshFillsError("faces must be an (m, 3) array of triangles")
    found = np.flatnonzero((triangles == fill.apex_vertex).any(axis=1))
    if found.size != fill.face_count:
        raise ArtifactMeshFillsError(
            f"the fill was found with {fill.face_count} faces but this mesh has "
            f"{found.size} at that point; it is not the mesh the fill was found in"
        )
    return found


def describe_filled_holes(report: MeshFillReport) -> tuple[str, ...]:
    """The report as lines to put in front of the person opening the file."""

    if not report.fills:
        return ("메우기로 보이는 면은 없습니다.",)

    lines = [
        f"스캐너가 잰 것이 아니라 소프트웨어가 채워 넣은 것으로 보이는 자리가 "
        f"{len(report.fills)}곳 있습니다 (면 {report.filled_face_count:,}개).",
    ]
    for index, fill in enumerate(report.fills):
        x, y, z = (value / 1000.0 for value in fill.apex_um)
        lines.append(
            f"  · 메우기 {index + 1}: 면 {fill.face_count}개가 한 점에 모여 있고 "
            f"넓이가 모두 같습니다 (차이 {fill.area_spread_millionths / 10_000.0:.2f}%)"
        )
        lines.append(
            f"      한 면 {fill.mean_area_um2 / 1_000_000.0:.2f} mm² · "
            f"이 메쉬 중앙값의 {fill.coarseness_millionths / 1_000_000.0:.1f}배 · "
            f"지름 약 {fill.span_um / 1000.0:.1f} mm · "
            f"가운데 점 (x {x:.1f}, y {y:.1f}, z {z:.1f}) mm"
        )
    lines.append("")
    lines.append(
        "스캐너의 삼각형은 옆면끼리도 넓이가 수십 %씩 다릅니다. "
        "한 점에서 부챗살로 뻗은 면들의 넓이가 소수점 아래까지 같다면 그것은 잰 면이 아니라 계산된 면입니다."
    )
    lines.append(
        "이대로 두면 도면은 그 자리를 실제 유물의 면으로 그립니다. "
        "떼어낼지, 둔 채 추정선(점선)으로 그릴지는 실물을 보고 정하세요."
    )
    return tuple(lines)
