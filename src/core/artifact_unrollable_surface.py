"""펼 수 있는 기록면: the part of a recording surface a development can carry.

A paper rubbing of a weathered 기와편 comes off blank where the surface
undercuts - a spall with a lip, a chip that curls back, a lump whose sides lean
over - because the paper bridges those places instead of following them.  The
authoritative tile unwrap refuses a development holding a single folded face or
a single overlap.  That is right for the development and wrong for a recorder
who painted a whole sheet: on 204 암키와편 the 외면 folds at 781 faces in 18
places, 0.23% of its area, and the whole sheet was refused for them.

This module answers "which of these faces can the development carry?" by
asking the development.  Each round builds the recipe the development would
build, runs the same sectionwise parameterization on the same submesh,
quantizes to the same micrometre grid, and takes out what the development's
own gates would refuse:

* faces the development turns inside out - the minority orientation, the
  walls of an undercut;
* faces the micrometre grid collapses;
* of two faces that land on one place of the paper, the one the paper cannot
  reach: behind the other as seen from where the paper lies.  On the outer
  face of the wall that is the one nearer the axis, on the inner face the one
  farther from it.  "Land on one place" is decided by the gate's own exact
  integer separating-axis test.

Then the patch is settled back into what the topology audit accepts - one
edge-connected piece, no repeated face or non-manifold edge, boundary loops
that do not branch - and the round repeats until the development passes.
There is no threshold anywhere: every face that goes is one the development's
gate named, or one the audit would not let a development hold.

Only a development about a measured axis (``canonical_axis_origin``) is
served.  Where the paper lies is a statement about the drum; about centres
fitted section by section there is no one side for it to be on.

The result is a face selection, like a painted one or a 회전축 기준 외면 띠,
and it is kept: `surface.unrollable_selection.v1` records the request, the
development it was tested for, every face it left out, how many of them each
gate named and where the places are, with QC on the share of area.  A
development of the kept faces depends on that record, and document
validation refuses a development that claims it without developing exactly
those faces the way it was tested - so a rubbing never leaves a place blank
without the record saying so.  The record is rechecked by running the
request again (`verify_unrollable_selection_record_against_mesh`).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, DecimalException
import json
import math
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .artifact_cancellation import CancellationProbe, poll_cancellation, raise_if_cancelled
from .artifact_document import (
    ArtifactDocument,
    ArtifactDocumentError,
    DerivedRecord,
    OperationContext,
    RecordFreshness,
    RecordLifecycleStatus,
    canonical_recipe_hash,
)
from .artifact_mandrel import FACING_AWAY_FROM_AXIS, FACING_TOWARD_AXIS, FACINGS
from .artifact_scene_adapter import ArtifactProjectionSnapshot
from .artifact_session import ArtifactSession, ArtifactSessionError
from .artifact_tile_unwrap_extractor import (
    MAX_TILE_UNWRAP_COORDINATE_UM,
    MAX_TILE_UNWRAP_FACES,
    MAX_TILE_UNWRAP_GRID_ASSIGNMENTS,
    MAX_TILE_UNWRAP_OVERLAP_CANDIDATES,
    SECTION_CENTER_CANONICAL_AXIS,
    STATION_CENTERLINE_ARC,
    STATION_MERIDIAN_ARC,
    TILE_UNWRAP_COORDINATE_QUANTUM_UM,
    ArtifactTileUnwrapError,
    _positive_area_triangle_overlap,
    _submesh_for_selection,
    selection_face_indices,
    tile_unwrap_recipe,
    tile_unwrap_selection,
    validate_tile_unwrap_recipe,
    validate_tile_unwrap_selection,
)
from .canonical_json import canonical_json_bytes, canonical_json_sha256
from .flatten_models_sectionwise import sectionwise_cylindrical_parameterization
from .mesh_loader import MeshData


#: Each round re-parameterizes a smaller patch, which can move a fold it had
#: not reached before; real sheets settle in two or three.
MAX_UNROLLABLE_ROUNDS = 12
#: A settled patch smaller than this is not a recording surface any more.
MIN_UNROLLABLE_FACES = 16
_SETTLE_ROUNDS = 64


_MEASURED_AXIS_ONLY = (
    "which faces the paper can reach is only defined about a measured axis; "
    "stand the tile on its 와통 and develop about the canonical axis"
)


class ArtifactUnrollableSurfaceError(ValueError):
    """No part of the requested surface can be carried by a development."""


@dataclass(frozen=True, slots=True)
class UnrollablePlace:
    """One connected place the development could not carry."""

    centre_mm: tuple[float, float, float]
    face_count: int
    area_mm2: float


@dataclass(frozen=True, slots=True)
class UnrollableSurface:
    """The faces a development can carry, and what was left out to get there."""

    face_indices: np.ndarray
    excluded_face_indices: np.ndarray
    requested_face_count: int
    requested_area_mm2: float
    excluded_area_mm2: float
    folded_face_count: int
    overlapped_face_count: int
    collapsed_face_count: int
    settled_face_count: int
    places: tuple[UnrollablePlace, ...]
    rounds: int
    surface_faces_axis: bool

    def __post_init__(self) -> None:
        for name in ("face_indices", "excluded_face_indices"):
            values = np.asarray(getattr(self, name), dtype=np.int64).reshape(-1)
            if values.size and np.any(np.diff(values) <= 0):
                raise ArtifactUnrollableSurfaceError(f"{name} must be sorted and unique")
            values.setflags(write=False)
            object.__setattr__(self, name, values)

    @property
    def face_count(self) -> int:
        return int(self.face_indices.shape[0])

    @property
    def excluded_area_share(self) -> float:
        if self.requested_area_mm2 <= 0.0:
            return 0.0
        return self.excluded_area_mm2 / self.requested_area_mm2

    def summary(self) -> dict[str, object]:
        return {
            "collapsed_face_count": self.collapsed_face_count,
            "excluded_area_mm2": round(self.excluded_area_mm2, 3),
            "excluded_area_share": round(self.excluded_area_share, 6),
            "excluded_face_count": int(self.excluded_face_indices.shape[0]),
            "folded_face_count": self.folded_face_count,
            "kept_face_count": self.face_count,
            "overlapped_face_count": self.overlapped_face_count,
            "place_count": len(self.places),
            "requested_area_mm2": round(self.requested_area_mm2, 3),
            "requested_face_count": self.requested_face_count,
            "rounds": self.rounds,
            "settled_face_count": self.settled_face_count,
        }


def _face_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    corners = vertices[faces]
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    return 0.5 * np.linalg.norm(cross, axis=1)


def _edge_components(faces: np.ndarray) -> np.ndarray:
    """Edge-connected component label of every face (vertex-only contact is not a join)."""
    count = int(faces.shape[0])
    if count == 0:
        return np.zeros((0,), dtype=np.int64)
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    owners = np.tile(np.arange(count, dtype=np.int64), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges = edges[order]
    owners = owners[order]
    same = np.all(edges[1:] == edges[:-1], axis=1)
    rows = owners[:-1][same]
    columns = owners[1:][same]
    graph = coo_matrix(
        (np.ones(rows.size, dtype=np.int8), (rows, columns)), shape=(count, count)
    )
    _count, labels = connected_components(graph, directed=False)
    return np.asarray(labels, dtype=np.int64)


def _settle(
    faces: np.ndarray,
    keep: np.ndarray,
    *,
    cancellation_probe: CancellationProbe | None,
) -> np.ndarray:
    """Reduce a face set to what the development's topology audit accepts."""

    keep = np.unique(np.asarray(keep, dtype=np.int64))
    for round_index in range(_SETTLE_ROUNDS):
        poll_cancellation(cancellation_probe, round_index)
        if keep.size < MIN_UNROLLABLE_FACES:
            raise ArtifactUnrollableSurfaceError(
                "settling the recording surface left too few faces to develop"
            )
        patch = faces[keep]

        canonical = np.sort(patch, axis=1)
        _unique, first, counts = np.unique(canonical, axis=0, return_index=True, return_counts=True)
        if np.any(counts > 1):
            keep = keep[np.sort(first)]
            continue

        edges = np.sort(np.concatenate([patch[:, [0, 1]], patch[:, [1, 2]], patch[:, [2, 0]]]), axis=1)
        unique_edges, inverse, edge_counts = np.unique(
            edges, axis=0, return_inverse=True, return_counts=True
        )
        owner = np.tile(np.arange(patch.shape[0], dtype=np.int64), 3)
        if np.any(edge_counts > 2):
            drop = np.unique(owner[(edge_counts > 2)[inverse.reshape(-1)]])
            keep = np.delete(keep, drop)
            continue

        labels = _edge_components(patch)
        names, sizes = np.unique(labels, return_counts=True)
        if names.size > 1:
            # The largest piece; on a tie the one holding the lowest face.
            best = sorted(
                ((-int(size), int(np.flatnonzero(labels == name).min()), int(name))
                 for name, size in zip(names, sizes, strict=True))
            )[0][2]
            keep = keep[labels == best]
            continue

        boundary = unique_edges[edge_counts == 1]
        if boundary.size == 0:
            raise ArtifactUnrollableSurfaceError(
                "the recording surface is a closed shell and has no boundary to open it at"
            )
        boundary_vertices, degree = np.unique(boundary.reshape(-1), return_counts=True)
        branching = boundary_vertices[degree != 2]
        if branching.size:
            touching = np.isin(patch, branching).any(axis=1)
            keep = keep[~touching]
            continue
        return keep
    raise ArtifactUnrollableSurfaceError("the recording surface would not settle")


def _overlapping_pairs(
    uv_um: np.ndarray,
    faces: np.ndarray,
    candidates: np.ndarray,
    *,
    cancellation_probe: CancellationProbe | None,
) -> list[tuple[int, int]]:
    """Every pair of candidate faces that overlaps with positive area.

    The grid is built the way the development's gate builds its own, and the
    pair test is the gate's own exact integer predicate, so a pair named here
    is one the gate would name.
    """

    index = np.asarray(candidates, dtype=np.int64)
    if index.size < 2:
        return []
    triangles = uv_um[faces[index]]
    minimum = triangles.min(axis=1)
    maximum = triangles.max(axis=1)
    global_minimum = minimum.min(axis=0)
    global_maximum = maximum.max(axis=0)
    span_x = int(global_maximum[0] - global_minimum[0]) + 1
    span_y = int(global_maximum[1] - global_minimum[1]) + 1
    target_cells = max(1, math.ceil(index.size / 8))
    cells_x = max(1, math.ceil(math.sqrt(target_cells * float(span_x) / float(span_y))))
    cells_y = max(1, math.ceil(target_cells / cells_x))
    cell_width = max(1, math.ceil(span_x / cells_x))
    cell_height = max(1, math.ceil(span_y / cells_y))
    first_x = (minimum[:, 0] - global_minimum[0]) // cell_width
    last_x = (maximum[:, 0] - global_minimum[0]) // cell_width
    first_y = (minimum[:, 1] - global_minimum[1]) // cell_height
    last_y = (maximum[:, 1] - global_minimum[1]) // cell_height
    assignments = int(np.sum((last_x - first_x + 1) * (last_y - first_y + 1)))
    if assignments > MAX_TILE_UNWRAP_GRID_ASSIGNMENTS:
        raise ArtifactUnrollableSurfaceError(
            "the development's overlap grid exceeds its bounded assignment budget"
        )

    cells: dict[tuple[int, int], list[int]] = {}
    for position in range(index.size):
        poll_cancellation(cancellation_probe, position)
        for cell_x in range(int(first_x[position]), int(last_x[position]) + 1):
            for cell_y in range(int(first_y[position]), int(last_y[position]) + 1):
                cells.setdefault((cell_x, cell_y), []).append(position)

    pairs: list[tuple[int, int]] = []
    examined = 0
    for cell_number, cell in enumerate(sorted(cells)):
        poll_cancellation(cancellation_probe, cell_number)
        members = cells[cell]
        for offset, left in enumerate(members):
            for right in members[offset + 1:]:
                # A pair shares several cells; test it only in the one it owns.
                if (
                    max(int(first_x[left]), int(first_x[right])),
                    max(int(first_y[left]), int(first_y[right])),
                ) != cell:
                    continue
                if (
                    min(int(maximum[left, 0]), int(maximum[right, 0]))
                    <= max(int(minimum[left, 0]), int(minimum[right, 0]))
                    or min(int(maximum[left, 1]), int(maximum[right, 1]))
                    <= max(int(minimum[left, 1]), int(minimum[right, 1]))
                ):
                    continue
                examined += 1
                if examined > MAX_TILE_UNWRAP_OVERLAP_CANDIDATES:
                    raise ArtifactUnrollableSurfaceError(
                        "the development's overlap test exceeds its examined-pair budget"
                    )
                if _positive_area_triangle_overlap(triangles[left], triangles[right]):
                    pairs.append((int(index[left]), int(index[right])))
    return pairs


def _places(
    vertices: np.ndarray,
    faces: np.ndarray,
    excluded: np.ndarray,
    areas: np.ndarray,
) -> tuple[UnrollablePlace, ...]:
    if excluded.size == 0:
        return ()
    labels = _edge_components(faces[excluded])
    places: list[UnrollablePlace] = []
    for name in np.unique(labels):
        members = excluded[labels == name]
        weights = areas[members]
        centres = vertices[faces[members]].mean(axis=1)
        total = float(weights.sum())
        centre = (
            (centres * weights[:, None]).sum(axis=0) / total
            if total > 0.0
            else centres.mean(axis=0)
        )
        places.append(
            UnrollablePlace(
                centre_mm=(float(centre[0]), float(centre[1]), float(centre[2])),
                face_count=int(members.size),
                area_mm2=total,
            )
        )
    places.sort(key=lambda place: (-place.face_count, place.centre_mm))
    return tuple(places)


def unrollable_recording_surface(
    mesh: MeshData,
    selected_face_indices: Sequence[int] | np.ndarray,
    *,
    longitudinal_axis: str,
    record_view: str,
    n_sections: int = 32,
    seam_angle_microdegrees: int | None = None,
    section_center_policy: str = SECTION_CENTER_CANONICAL_AXIS,
    station_policy: str = STATION_CENTERLINE_ARC,
    cancellation_probe: CancellationProbe | None = None,
) -> UnrollableSurface:
    """Return the part of the selected surface a development about the axis can carry.

    ``mesh`` is the canonical projection - the tile standing on its measured
    axis - and the development parameters are the ones the recorder will pass
    to the development itself, so the answer is about that development.
    """

    raise_if_cancelled(cancellation_probe)
    if section_center_policy != SECTION_CENTER_CANONICAL_AXIS:
        raise ArtifactUnrollableSurfaceError(_MEASURED_AXIS_ONLY)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    total = int(faces.shape[0])
    requested = np.unique(np.asarray(selected_face_indices, dtype=np.int64).reshape(-1))
    if requested.size == 0:
        raise ArtifactUnrollableSurfaceError("no recording-surface face was selected")
    if int(requested[0]) < 0 or int(requested[-1]) >= total:
        raise ArtifactUnrollableSurfaceError("recording-surface face index is out of range")
    try:
        # Validates the request the way the development will: axis, view,
        # section count and the recording-surface face limit.
        validate_tile_unwrap_recipe(
            tile_unwrap_recipe(
                longitudinal_axis=longitudinal_axis,
                record_view=record_view,
                total_face_count=total,
                selected_face_indices=requested,
                n_sections=n_sections,
                seam_angle_microdegrees=seam_angle_microdegrees,
                section_center_policy=section_center_policy,
                station_policy=station_policy,
            )
        )
    except ArtifactTileUnwrapError as exc:
        raise ArtifactUnrollableSurfaceError(str(exc)) from exc

    areas = _face_areas(vertices, faces)
    axis_index = "xyz".index(str(longitudinal_axis))
    across = [index for index in range(3) if index != axis_index]
    radius = np.hypot(vertices[:, across[0]], vertices[:, across[1]])

    keep = _settle(faces, requested, cancellation_probe=cancellation_probe)
    settled = int(requested.size - keep.size)
    folded_total = 0
    overlapped_total = 0
    collapsed_total = 0
    faces_axis = False
    for round_number in range(1, MAX_UNROLLABLE_ROUNDS + 1):
        raise_if_cancelled(cancellation_probe)
        submesh, _source_vertices, source_faces = _submesh_for_selection(mesh, keep)
        result = sectionwise_cylindrical_parameterization(
            submesh,
            axis=str(longitudinal_axis),
            n_sections=int(n_sections),
            record_view=str(record_view),
            seam_angle_microdegrees=seam_angle_microdegrees,
            section_center="axis_origin",
            station="meridian" if station_policy == STATION_MERIDIAN_ARC else "centerline",
            return_meta=True,
            cancellation_probe=cancellation_probe,
        )
        if not isinstance(result, tuple):  # pragma: no cover - return_meta contract
            raise ArtifactUnrollableSurfaceError("the parameterization returned no metadata")
        uv, meta = result
        if bool(meta.get("sectionwise_fallback", False)):
            raise ArtifactUnrollableSurfaceError(
                "the development would fall back to another algorithm: "
                f"{meta.get('sectionwise_reason', 'sectionwise_internal_fallback')}"
            )
        achieved = int(meta.get("section_count", 0))
        if achieved != int(n_sections):
            raise ArtifactUnrollableSurfaceError(
                f"the development requested {int(n_sections)} sections but this "
                f"recording surface supports {achieved}; set n_sections to {achieved}"
            )
        uv_mm = np.asarray(uv, dtype=np.float64)
        uv_mm = uv_mm - np.min(uv_mm, axis=0, keepdims=True)
        scaled = uv_mm * (1000.0 / TILE_UNWRAP_COORDINATE_QUANTUM_UM)
        if not np.isfinite(scaled).all() or np.any(scaled > MAX_TILE_UNWRAP_COORDINATE_UM):
            raise ArtifactUnrollableSurfaceError("the development exceeds the exact coordinate grid")
        uv_um = np.rint(scaled).astype(np.int64)
        local = np.asarray(submesh.faces, dtype=np.int64)

        first_edge = uv_um[local[:, 1]] - uv_um[local[:, 0]]
        second_edge = uv_um[local[:, 2]] - uv_um[local[:, 0]]
        signed = first_edge[:, 0] * second_edge[:, 1] - first_edge[:, 1] * second_edge[:, 0]
        positive = int(np.count_nonzero(signed > 0))
        negative = int(np.count_nonzero(signed < 0))
        collapsed = signed == 0
        folded = (signed < 0) if negative <= positive else (signed > 0)

        # Where the paper lies: the side the sheet's own faces look to.
        local_vertices = np.asarray(submesh.vertices, dtype=np.float64)
        corners = local_vertices[local]
        normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        centres = corners.mean(axis=1)
        radial = centres[:, across]
        radial_length = np.linalg.norm(radial, axis=1)
        radial_length[radial_length == 0.0] = 1.0
        facing = float(
            np.sum((normals[:, across] * radial).sum(axis=1) / radial_length)
        )
        faces_axis = facing < 0.0
        depth = radius[_source_vertices][local].mean(axis=1)

        candidates = np.flatnonzero(~folded & ~collapsed)
        behind: set[int] = set()
        for left, right in _overlapping_pairs(
            uv_um, local, candidates, cancellation_probe=cancellation_probe
        ):
            # The paper lies outside an outer face and inside an inner one.
            if faces_axis:
                hidden = left if depth[left] > depth[right] else right
            else:
                hidden = left if depth[left] < depth[right] else right
            behind.add(int(hidden))

        drop_local = np.flatnonzero(folded | collapsed)
        if behind:
            drop_local = np.union1d(drop_local, np.fromiter(behind, dtype=np.int64))
        if drop_local.size == 0:
            excluded = np.setdiff1d(requested, keep)
            return UnrollableSurface(
                face_indices=keep,
                excluded_face_indices=excluded,
                requested_face_count=int(requested.size),
                requested_area_mm2=float(areas[requested].sum()),
                excluded_area_mm2=float(areas[excluded].sum()),
                folded_face_count=folded_total,
                overlapped_face_count=overlapped_total,
                collapsed_face_count=collapsed_total,
                settled_face_count=settled,
                places=_places(vertices, faces, excluded, areas),
                rounds=round_number,
                surface_faces_axis=faces_axis,
            )
        folded_total += int(np.count_nonzero(folded))
        collapsed_total += int(np.count_nonzero(collapsed))
        overlapped_total += int(len(behind))
        before = int(keep.size)
        keep = np.setdiff1d(keep, np.asarray(source_faces, dtype=np.int64)[drop_local])
        keep = _settle(faces, keep, cancellation_probe=cancellation_probe)
        settled += before - int(drop_local.size) - int(keep.size)
    raise ArtifactUnrollableSurfaceError(
        f"the recording surface did not become developable in {MAX_UNROLLABLE_ROUNDS} rounds"
    )


# ---------------------------------------------------------------------------
# The record: what was asked for, what was left out, and a development's
# dependency on it
# ---------------------------------------------------------------------------

UNROLLABLE_RECORD_TYPE = "surface.unrollable_selection.v1"
#: The measurement operation this recipe belongs to; every measurement recipe
#: names its kind so a work item cannot run the wrong one.
UNROLLABLE_RECIPE_KIND = "unrollable_selection"
UNROLLABLE_ALGORITHM = "archmeshrubbing.unrollable_recording_surface"
UNROLLABLE_ALGORITHM_VERSION = "1.0.0"
UNROLLABLE_RECIPE_SCHEMA_VERSION = "1.0.0"
UNROLLABLE_RECEIPT_SCHEMA_VERSION = "1.0.0"
UNROLLABLE_COORDINATE_SPACE = "canonical_aligned_mm/v1"
#: What is left out, in the order each round applies it: the faces the
#: development folds, the faces its micrometre grid collapses, of two faces
#: landing on one place of the paper the one behind, and what the topology
#: audit will not let a development hold.
UNROLLABLE_EXCLUSION_POLICY = (
    "development_gate_folds+collapses+hidden_overlaps+topology_settle/v1"
)
UNROLLABLE_EXTENSION_KEY = "org.archmeshrubbing:unrollable-selection-v1"
UNROLLABLE_MEDIA_TYPE = "application/vnd.archmeshrubbing.unrollable-selection-receipt+json"
UNROLLABLE_REF_PREFIX = "urn:archmeshrubbing:unrollable-selection:sha256:"
#: The largest places are listed by where they are; every place is counted.
MAX_RECORDED_UNROLLABLE_PLACES = 64
MAX_UNROLLABLE_RECEIPT_BYTES = 16 * 1024 * 1024
#: The development a request is tested for, named as the development's own
#: recipe names it.
_DEVELOPMENT_KEYS = (
    "longitudinal_axis",
    "n_sections",
    "record_view",
    "seam_angle_microdegrees",
    "seam_policy",
    "section_center_policy",
    "station_policy",
)
_MM_QUANTUM = Decimal("0.001")
_SHARE_QUANTUM = Decimal("0.000001")
_MM_RE = re.compile(r"^-?(0|[1-9][0-9]*)\.[0-9]{3}$")
_SHARE_RE = re.compile(r"^(0|1)\.[0-9]{6}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
#: A rerun recomputes areas and centres in floating point; another build may
#: move them in the last digit, never by this much.
_RECHECK_TOLERANCE_MM = 0.002


def _fixed(value: float, quantum: Decimal) -> str:
    number = Decimal(str(float(value)))
    if not number.is_finite():
        raise ArtifactUnrollableSurfaceError("an exclusion figure must be finite")
    try:
        quantized = number.quantize(quantum, rounding=ROUND_HALF_EVEN)
    except DecimalException as exc:
        raise ArtifactUnrollableSurfaceError(
            "an exclusion figure cannot be written as a fixed decimal"
        ) from exc
    if quantized == 0:
        quantized = Decimal(0).quantize(quantum)
    return format(quantized, "f")


def _decimal(value: object, pattern: re.Pattern[str], *, name: str) -> Decimal:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ArtifactUnrollableSurfaceError(f"{name} is not a canonical fixed decimal")
    number = Decimal(value)
    if value.startswith("-") and number == 0:
        raise ArtifactUnrollableSurfaceError(f"{name} writes zero with a sign")
    return number


def _count(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactUnrollableSurfaceError(f"{name} must be an integer")
    if value < minimum or value > MAX_TILE_UNWRAP_FACES:
        raise ArtifactUnrollableSurfaceError(
            f"{name} must be within [{minimum}, {MAX_TILE_UNWRAP_FACES}]"
        )
    return int(value)


def _exact(value: object, keys: frozenset[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactUnrollableSurfaceError(f"{name} must be an object")
    missing = sorted(keys - set(value))
    unexpected = sorted(set(value) - keys)
    if missing:
        raise ArtifactUnrollableSurfaceError(f"{name} is missing fields: {', '.join(missing)}")
    if unexpected:
        raise ArtifactUnrollableSurfaceError(
            f"{name} has unsupported fields: {', '.join(unexpected)}"
        )
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(value[key]) for key in sorted(value)})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


def _frozen_mapping(value: Mapping[str, Any], *, name: str) -> Mapping[str, Any]:
    try:
        decoded = json.loads(canonical_json_bytes(_thaw(value)))
    except (TypeError, ValueError) as exc:
        raise ArtifactUnrollableSurfaceError(f"{name} is not strict JSON") from exc
    if not isinstance(decoded, dict):
        raise ArtifactUnrollableSurfaceError(f"{name} must be an object")
    frozen = _freeze(decoded)
    assert isinstance(frozen, Mapping)
    return frozen


def _selection_sha(total_face_count: int, indices: np.ndarray) -> str:
    return str(
        tile_unwrap_selection(
            total_face_count=total_face_count, selected_face_indices=indices
        )["selection_sha256"]
    )


_RECIPE_KEYS = frozenset(
    {
        "algorithm",
        "algorithm_version",
        "coordinate_space",
        "development",
        "exclusion_policy",
        "kind",
        "schema_version",
        "selection",
    }
)


def unrollable_selection_recipe(
    *,
    total_face_count: int,
    selected_face_indices: Sequence[int] | np.ndarray,
    longitudinal_axis: str,
    record_view: str,
    n_sections: int = 32,
    seam_angle_microdegrees: int | None = None,
    section_center_policy: str = SECTION_CENTER_CANONICAL_AXIS,
    station_policy: str = STATION_CENTERLINE_ARC,
) -> dict[str, Any]:
    """The complete recipe: the painted request and the development it is for.

    The request is written in the encoding a development uses and the
    parameters are the development's own, validated by building its recipe,
    so the record and the development it serves can be compared field for
    field.
    """

    if section_center_policy != SECTION_CENTER_CANONICAL_AXIS:
        raise ArtifactUnrollableSurfaceError(_MEASURED_AXIS_ONLY)
    try:
        development = tile_unwrap_recipe(
            longitudinal_axis=longitudinal_axis,
            record_view=record_view,
            total_face_count=total_face_count,
            selected_face_indices=selected_face_indices,
            n_sections=n_sections,
            seam_angle_microdegrees=seam_angle_microdegrees,
            section_center_policy=section_center_policy,
            station_policy=station_policy,
        )
    except ArtifactTileUnwrapError as exc:
        raise ArtifactUnrollableSurfaceError(str(exc)) from exc
    return {
        "algorithm": UNROLLABLE_ALGORITHM,
        "algorithm_version": UNROLLABLE_ALGORITHM_VERSION,
        "coordinate_space": UNROLLABLE_COORDINATE_SPACE,
        "development": {key: development[key] for key in _DEVELOPMENT_KEYS},
        "exclusion_policy": UNROLLABLE_EXCLUSION_POLICY,
        "kind": UNROLLABLE_RECIPE_KIND,
        "schema_version": UNROLLABLE_RECIPE_SCHEMA_VERSION,
        "selection": development["selection"],
    }


def validate_unrollable_selection_recipe(value: object) -> dict[str, Any]:
    recipe = _exact(value, _RECIPE_KEYS, name="unrollable selection recipe")
    development = _exact(
        recipe["development"],
        frozenset(_DEVELOPMENT_KEYS),
        name="unrollable selection development",
    )
    try:
        selection = validate_tile_unwrap_selection(_thaw(recipe["selection"]))
    except ArtifactTileUnwrapError as exc:
        raise ArtifactUnrollableSurfaceError(
            f"unrollable selection recipe selection: {exc}"
        ) from exc
    rebuilt = unrollable_selection_recipe(
        total_face_count=int(selection["total_face_count"]),
        selected_face_indices=selection_face_indices(selection),
        longitudinal_axis=development["longitudinal_axis"],
        record_view=development["record_view"],
        n_sections=development["n_sections"],
        seam_angle_microdegrees=development["seam_angle_microdegrees"],
        section_center_policy=development["section_center_policy"],
        station_policy=development["station_policy"],
    )
    if canonical_json_bytes(rebuilt) != canonical_json_bytes(_thaw(recipe)):
        raise ArtifactUnrollableSurfaceError(
            "unrollable selection recipe is not the canonical recipe for its request"
        )
    return rebuilt


def unrollable_selection_hash(recipe: Mapping[str, Any]) -> str:
    return str(validate_unrollable_selection_recipe(recipe)["selection"]["selection_sha256"])


_RECEIPT_KEYS = frozenset(
    {
        "algorithm",
        "algorithm_version",
        "areas",
        "counts",
        "excluded_selection",
        "kept_selection_sha256",
        "place_count",
        "places",
        "requested_selection_sha256",
        "rounds",
        "schema_version",
        "surface_facing",
    }
)
_COUNT_KEYS = frozenset(
    {
        "collapsed_face_count",
        "excluded_face_count",
        "folded_face_count",
        "kept_face_count",
        "overlapped_face_count",
        "requested_face_count",
        "settled_face_count",
    }
)
_AREA_KEYS = frozenset(
    {
        "excluded_area_mm2_decimal",
        "excluded_area_share_decimal",
        "requested_area_mm2_decimal",
    }
)
_PLACE_KEYS = frozenset({"area_mm2_decimal", "centre_mm_decimal", "face_count"})


def _receipt_for(
    surface: UnrollableSurface,
    recipe: Mapping[str, Any],
    vertices: np.ndarray,
    faces: np.ndarray,
) -> dict[str, Any]:
    selection = recipe["selection"]
    total = int(selection["total_face_count"])
    requested = selection_face_indices(selection)
    areas = _face_areas(vertices, faces)
    # Exactly rounded sums, so a rerun on another build writes the same digits.
    requested_area = math.fsum(areas[requested].tolist())
    excluded_area = math.fsum(areas[surface.excluded_face_indices].tolist())
    share = excluded_area / requested_area if requested_area > 0.0 else 0.0
    excluded = surface.excluded_face_indices
    return {
        "algorithm": UNROLLABLE_ALGORITHM,
        "algorithm_version": UNROLLABLE_ALGORITHM_VERSION,
        "areas": {
            "excluded_area_mm2_decimal": _fixed(excluded_area, _MM_QUANTUM),
            "excluded_area_share_decimal": _fixed(share, _SHARE_QUANTUM),
            "requested_area_mm2_decimal": _fixed(requested_area, _MM_QUANTUM),
        },
        "counts": {
            "collapsed_face_count": int(surface.collapsed_face_count),
            "excluded_face_count": int(excluded.shape[0]),
            "folded_face_count": int(surface.folded_face_count),
            "kept_face_count": surface.face_count,
            "overlapped_face_count": int(surface.overlapped_face_count),
            "requested_face_count": int(surface.requested_face_count),
            "settled_face_count": int(surface.settled_face_count),
        },
        "excluded_selection": (
            None
            if excluded.size == 0
            else tile_unwrap_selection(total_face_count=total, selected_face_indices=excluded)
        ),
        "kept_selection_sha256": _selection_sha(total, surface.face_indices),
        "place_count": len(surface.places),
        "places": [
            {
                "area_mm2_decimal": _fixed(place.area_mm2, _MM_QUANTUM),
                "centre_mm_decimal": [_fixed(value, _MM_QUANTUM) for value in place.centre_mm],
                "face_count": int(place.face_count),
            }
            for place in surface.places[:MAX_RECORDED_UNROLLABLE_PLACES]
        ],
        "requested_selection_sha256": str(selection["selection_sha256"]),
        "rounds": int(surface.rounds),
        "schema_version": UNROLLABLE_RECEIPT_SCHEMA_VERSION,
        "surface_facing": (
            FACING_TOWARD_AXIS if surface.surface_faces_axis else FACING_AWAY_FROM_AXIS
        ),
    }


def validate_unrollable_selection_receipt(value: object) -> dict[str, Any]:
    receipt = _exact(value, _RECEIPT_KEYS, name="unrollable selection receipt")
    for key, expected in (
        ("algorithm", UNROLLABLE_ALGORITHM),
        ("algorithm_version", UNROLLABLE_ALGORITHM_VERSION),
        ("schema_version", UNROLLABLE_RECEIPT_SCHEMA_VERSION),
    ):
        if receipt[key] != expected:
            raise ArtifactUnrollableSurfaceError(f"unrollable selection receipt {key} is unsupported")
    counts_block = _exact(receipt["counts"], _COUNT_KEYS, name="unrollable selection counts")
    counts = {key: _count(counts_block[key], name=f"counts.{key}") for key in _COUNT_KEYS}
    if counts["kept_face_count"] < MIN_UNROLLABLE_FACES:
        raise ArtifactUnrollableSurfaceError("unrollable selection keeps too few faces to develop")
    if counts["kept_face_count"] + counts["excluded_face_count"] != counts["requested_face_count"]:
        raise ArtifactUnrollableSurfaceError(
            "unrollable selection counts do not add up to the request"
        )
    areas = _exact(receipt["areas"], _AREA_KEYS, name="unrollable selection areas")
    requested_area = _decimal(
        areas["requested_area_mm2_decimal"], _MM_RE, name="requested_area_mm2_decimal"
    )
    excluded_area = _decimal(
        areas["excluded_area_mm2_decimal"], _MM_RE, name="excluded_area_mm2_decimal"
    )
    share = _decimal(
        areas["excluded_area_share_decimal"], _SHARE_RE, name="excluded_area_share_decimal"
    )
    if requested_area <= 0 or excluded_area < 0 or excluded_area > requested_area or share > 1:
        raise ArtifactUnrollableSurfaceError("unrollable selection areas are inconsistent")
    excluded_selection = receipt["excluded_selection"]
    if excluded_selection is None:
        if counts["excluded_face_count"] != 0 or excluded_area != 0 or share != 0:
            raise ArtifactUnrollableSurfaceError(
                "unrollable selection leaves faces out without naming them"
            )
    else:
        try:
            excluded_value = validate_tile_unwrap_selection(_thaw(excluded_selection))
        except ArtifactTileUnwrapError as exc:
            raise ArtifactUnrollableSurfaceError(
                f"unrollable selection excluded faces: {exc}"
            ) from exc
        if int(excluded_value["selected_face_count"]) != counts["excluded_face_count"]:
            raise ArtifactUnrollableSurfaceError(
                "unrollable selection names a different number of faces than it counts"
            )
    for key in ("kept_selection_sha256", "requested_selection_sha256"):
        if not isinstance(receipt[key], str) or _SHA256_RE.fullmatch(receipt[key]) is None:
            raise ArtifactUnrollableSurfaceError(f"unrollable selection {key} is invalid")
    place_count = _count(receipt["place_count"], name="place_count")
    places = receipt["places"]
    if not isinstance(places, (list, tuple)) or len(places) != min(
        place_count, MAX_RECORDED_UNROLLABLE_PLACES
    ):
        raise ArtifactUnrollableSurfaceError("unrollable selection places do not match their count")
    if (place_count == 0) != (counts["excluded_face_count"] == 0):
        raise ArtifactUnrollableSurfaceError(
            "unrollable selection places and excluded faces disagree"
        )
    listed = 0
    for index, entry in enumerate(places):
        place = _exact(entry, _PLACE_KEYS, name=f"places[{index}]")
        listed += _count(place["face_count"], name=f"places[{index}].face_count", minimum=1)
        if _decimal(place["area_mm2_decimal"], _MM_RE, name=f"places[{index}].area") < 0:
            raise ArtifactUnrollableSurfaceError(f"places[{index}] has a negative area")
        centre = place["centre_mm_decimal"]
        if not isinstance(centre, (list, tuple)) or len(centre) != 3:
            raise ArtifactUnrollableSurfaceError(f"places[{index}] centre must be three decimals")
        for axis, item in enumerate(centre):
            _decimal(item, _MM_RE, name=f"places[{index}].centre[{axis}]")
    if listed > counts["excluded_face_count"]:
        raise ArtifactUnrollableSurfaceError("unrollable selection places hold more faces than it left out")
    _count(receipt["rounds"], name="rounds", minimum=1)
    if int(receipt["rounds"]) > MAX_UNROLLABLE_ROUNDS:
        raise ArtifactUnrollableSurfaceError("unrollable selection took more rounds than allowed")
    if receipt["surface_facing"] not in FACINGS:
        raise ArtifactUnrollableSurfaceError("unrollable selection surface_facing is invalid")
    output = _thaw(receipt)
    assert isinstance(output, dict)
    return output


def _validate_recipe_against_receipt(
    recipe: Mapping[str, Any], receipt: Mapping[str, Any]
) -> None:
    selection = recipe["selection"]
    if (
        receipt["requested_selection_sha256"] != selection["selection_sha256"]
        or int(receipt["counts"]["requested_face_count"]) != int(selection["selected_face_count"])
    ):
        raise ArtifactUnrollableSurfaceError(
            "unrollable selection recipe and receipt name different requests"
        )
    total = int(selection["total_face_count"])
    requested = selection_face_indices(selection)
    excluded_selection = receipt["excluded_selection"]
    if excluded_selection is None:
        excluded = np.zeros((0,), dtype=np.int64)
    else:
        if int(excluded_selection["total_face_count"]) != total:
            raise ArtifactUnrollableSurfaceError(
                "unrollable selection excluded faces name a different mesh"
            )
        excluded = selection_face_indices(excluded_selection)
    if excluded.size and not bool(np.isin(excluded, requested, assume_unique=True).all()):
        raise ArtifactUnrollableSurfaceError(
            "unrollable selection leaves out a face that was not asked for"
        )
    kept = np.setdiff1d(requested, excluded, assume_unique=True)
    if _selection_sha(total, kept) != receipt["kept_selection_sha256"]:
        raise ArtifactUnrollableSurfaceError(
            "unrollable selection kept faces are not the request less what it left out"
        )


def _qc_from_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    counts = receipt["counts"]
    areas = receipt["areas"]
    return {
        "excluded_area_mm2": float(Decimal(areas["excluded_area_mm2_decimal"])),
        "excluded_area_share": float(Decimal(areas["excluded_area_share_decimal"])),
        "excluded_face_count": int(counts["excluded_face_count"]),
        "kept_face_count": int(counts["kept_face_count"]),
        "place_count": int(receipt["place_count"]),
        "requested_face_count": int(counts["requested_face_count"]),
        "rounds": int(receipt["rounds"]),
        "surface_facing": str(receipt["surface_facing"]),
    }


def _validate_qc_against_receipt(qc: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
    if canonical_json_bytes(_thaw(qc)) != canonical_json_bytes(_qc_from_receipt(receipt)):
        raise ArtifactUnrollableSurfaceError("unrollable selection QC does not match its receipt")


def extract_unrollable_selection(
    mesh: MeshData,
    recipe: Mapping[str, Any],
    *,
    cancellation_probe: CancellationProbe | None = None,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    """Run the request and write down what it kept and what it left out.

    ``mesh`` is the canonical projection the request is made in.  Returns the
    receipt, its QC and the kept faces, which are what the development of
    this request is to be made from.
    """

    validated = validate_unrollable_selection_recipe(recipe)
    selection = validated["selection"]
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if int(selection["total_face_count"]) != int(faces.shape[0]):
        raise ArtifactUnrollableSurfaceError("the unrollable selection names a different mesh")
    development = validated["development"]
    surface = unrollable_recording_surface(
        mesh,
        selection_face_indices(selection),
        longitudinal_axis=str(development["longitudinal_axis"]),
        record_view=str(development["record_view"]),
        n_sections=int(development["n_sections"]),
        seam_angle_microdegrees=development["seam_angle_microdegrees"],
        section_center_policy=str(development["section_center_policy"]),
        station_policy=str(development["station_policy"]),
        cancellation_probe=cancellation_probe,
    )
    receipt = validate_unrollable_selection_receipt(
        _receipt_for(surface, validated, np.asarray(mesh.vertices, dtype=np.float64), faces)
    )
    _validate_recipe_against_receipt(validated, receipt)
    return receipt, _qc_from_receipt(receipt), np.asarray(surface.face_indices, dtype=np.int64)


@dataclass(frozen=True, slots=True)
class UnrollableSelectionComputation:
    context: OperationContext
    projection_snapshot: ArtifactProjectionSnapshot
    receipt: Mapping[str, Any]
    recipe: Mapping[str, Any]
    qc: Mapping[str, Any]
    kept_face_indices: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.context, OperationContext):
            raise ArtifactUnrollableSurfaceError("context must be OperationContext")
        if not isinstance(self.projection_snapshot, ArtifactProjectionSnapshot):
            raise ArtifactUnrollableSurfaceError(
                "projection_snapshot must be ArtifactProjectionSnapshot"
            )
        recipe = validate_unrollable_selection_recipe(self.recipe)
        receipt = validate_unrollable_selection_receipt(self.receipt)
        _validate_recipe_against_receipt(recipe, receipt)
        if canonical_recipe_hash(recipe) != self.context.recipe_hash:
            raise ArtifactUnrollableSurfaceError(
                "unrollable selection recipe does not match its OperationContext"
            )
        if self.context.selection_hash != recipe["selection"]["selection_sha256"]:
            raise ArtifactUnrollableSurfaceError(
                "unrollable selection hash does not match its request"
            )
        snapshot = self.projection_snapshot
        if (
            tuple(self.context.source_asset_ids) != (snapshot.source_asset_id,)
            or self.context.geometry_revision_id != snapshot.geometry_revision_id
            or self.context.source_metadata_revision_id != snapshot.source_metadata_revision_id
            or self.context.align_revision_id != snapshot.align_revision_id
        ):
            raise ArtifactUnrollableSurfaceError(
                "projection snapshot does not match the unrollable selection context"
            )
        qc = _frozen_mapping(self.qc, name="unrollable selection qc")
        _validate_qc_against_receipt(qc, receipt)
        kept = np.array(self.kept_face_indices, dtype=np.int64).reshape(-1)
        total = int(recipe["selection"]["total_face_count"])
        if _selection_sha(total, kept) != receipt["kept_selection_sha256"]:
            raise ArtifactUnrollableSurfaceError(
                "kept faces do not match the unrollable selection receipt"
            )
        kept.setflags(write=False)
        object.__setattr__(self, "recipe", _frozen_mapping(recipe, name="recipe"))
        object.__setattr__(self, "receipt", _frozen_mapping(receipt, name="receipt"))
        object.__setattr__(self, "qc", qc)
        object.__setattr__(self, "kept_face_indices", kept)

    @property
    def record_type(self) -> str:
        return UNROLLABLE_RECORD_TYPE

    @property
    def geometry_ref(self) -> str:
        return UNROLLABLE_REF_PREFIX + canonical_json_sha256(self.receipt_dict())

    def recipe_dict(self) -> dict[str, Any]:
        value = _thaw(self.recipe)
        assert isinstance(value, dict)
        return value

    def receipt_dict(self) -> dict[str, Any]:
        value = _thaw(self.receipt)
        assert isinstance(value, dict)
        return value

    def qc_dict(self) -> dict[str, Any]:
        value = _thaw(self.qc)
        assert isinstance(value, dict)
        return value


def unrollable_selection_computation_matches_active_projection(
    session: ArtifactSession,
    computation: UnrollableSelectionComputation,
) -> bool:
    if not isinstance(session, ArtifactSession) or not isinstance(
        computation, UnrollableSelectionComputation
    ):
        return False
    try:
        current = session.projection_snapshot()
    except ArtifactSessionError:
        return False
    return current.render_key == computation.projection_snapshot.render_key


def commit_unrollable_selection(
    session: ArtifactSession,
    computation: UnrollableSelectionComputation,
    *,
    record_id: str,
    created_at: str,
    operator: str,
    depends_on_record_ids: Sequence[str] = (),
) -> ArtifactSession:
    if not unrollable_selection_computation_matches_active_projection(session, computation):
        raise ArtifactUnrollableSurfaceError(
            "unrollable selection is stale for the active projection"
        )
    receipt = computation.receipt_dict()
    qc = computation.qc_dict()
    _validate_qc_against_receipt(qc, receipt)
    receipt_bytes = canonical_json_bytes(receipt)
    if len(receipt_bytes) > MAX_UNROLLABLE_RECEIPT_BYTES:
        raise ArtifactUnrollableSurfaceError("unrollable selection receipt exceeds its limit")
    receipt_sha256 = canonical_json_sha256(receipt)
    extensions = {
        UNROLLABLE_EXTENSION_KEY: {
            "media_type": UNROLLABLE_MEDIA_TYPE,
            "receipt": receipt,
            "receipt_byte_length": len(receipt_bytes),
            "receipt_sha256": receipt_sha256,
            "schema_version": UNROLLABLE_RECEIPT_SCHEMA_VERSION,
        }
    }
    try:
        document = session.document.append_record_from_context(
            context=computation.context,
            id=record_id,
            type=UNROLLABLE_RECORD_TYPE,
            geometry_ref=UNROLLABLE_REF_PREFIX + receipt_sha256,
            recipe=computation.recipe_dict(),
            qc=qc,
            lifecycle_status=RecordLifecycleStatus.READY,
            created_at=created_at,
            operator=operator,
            depends_on_record_ids=depends_on_record_ids,
            extensions=extensions,
        )
    except ArtifactDocumentError as exc:
        raise ArtifactUnrollableSurfaceError(str(exc)) from exc
    return session.with_document(document)


def unrollable_selection_receipt_from_record(record: DerivedRecord) -> dict[str, Any]:
    if not isinstance(record, DerivedRecord):
        raise ArtifactUnrollableSurfaceError("record must be a DerivedRecord")
    if record.type != UNROLLABLE_RECORD_TYPE:
        raise ArtifactUnrollableSurfaceError("record is not an unrollable selection")
    descriptor = _exact(
        record.extensions.get(UNROLLABLE_EXTENSION_KEY),
        frozenset(
            {"media_type", "receipt", "receipt_byte_length", "receipt_sha256", "schema_version"}
        ),
        name="unrollable selection descriptor",
    )
    if (
        descriptor["media_type"] != UNROLLABLE_MEDIA_TYPE
        or descriptor["schema_version"] != UNROLLABLE_RECEIPT_SCHEMA_VERSION
    ):
        raise ArtifactUnrollableSurfaceError("unrollable selection descriptor is invalid")
    declared_length = _count(
        descriptor["receipt_byte_length"], name="receipt_byte_length", minimum=2
    )
    receipt = validate_unrollable_selection_receipt(_thaw(descriptor["receipt"]))
    receipt_bytes = canonical_json_bytes(receipt)
    if len(receipt_bytes) != declared_length or len(receipt_bytes) > MAX_UNROLLABLE_RECEIPT_BYTES:
        raise ArtifactUnrollableSurfaceError("unrollable selection receipt length is invalid")
    receipt_sha256 = canonical_json_sha256(receipt)
    if descriptor["receipt_sha256"] != receipt_sha256:
        raise ArtifactUnrollableSurfaceError("unrollable selection receipt hash is invalid")
    if record.geometry_ref != UNROLLABLE_REF_PREFIX + receipt_sha256:
        raise ArtifactUnrollableSurfaceError("unrollable selection geometry_ref is invalid")
    recipe = validate_unrollable_selection_recipe(_thaw(record.recipe))
    _validate_recipe_against_receipt(recipe, receipt)
    if record.selection_hash != recipe["selection"]["selection_sha256"]:
        raise ArtifactUnrollableSurfaceError("unrollable selection hash is invalid")
    _validate_qc_against_receipt(record.qc, receipt)
    return receipt


def _development_fields(recipe: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _thaw(recipe.get(key)) for key in _DEVELOPMENT_KEYS}


def validate_unrollable_selection_records(document: ArtifactDocument) -> None:
    """Every exclusion record is sound, and every development that claims one
    develops exactly the faces it kept, the way it was tested, about the same
    axis."""

    if not isinstance(document, ArtifactDocument):
        raise ArtifactUnrollableSurfaceError("document must be an ArtifactDocument")
    exclusions: dict[str, tuple[DerivedRecord, dict[str, Any], dict[str, Any]]] = {}
    for record in document.records:
        if record.type == UNROLLABLE_RECORD_TYPE:
            receipt = unrollable_selection_receipt_from_record(record)
            recipe = validate_unrollable_selection_recipe(_thaw(record.recipe))
            exclusions[record.id] = (record, recipe, receipt)
    if not exclusions:
        return
    from .artifact_tile_unwrap_record import TILE_UNWRAP_RECORD_TYPE  # noqa: PLC0415

    for record in document.records:
        if record.type != TILE_UNWRAP_RECORD_TYPE:
            continue
        for dependency in record.depends_on_record_ids:
            if dependency not in exclusions:
                continue
            exclusion, recipe, receipt = exclusions[dependency]
            development = _thaw(record.recipe)
            selection = development.get("selection") if isinstance(development, dict) else None
            if (
                not isinstance(selection, dict)
                or selection.get("selection_sha256") != receipt["kept_selection_sha256"]
            ):
                raise ArtifactUnrollableSurfaceError(
                    f"development {record.id} depends on {dependency} but does not "
                    "develop the faces it kept"
                )
            if _development_fields(development) != recipe["development"]:
                raise ArtifactUnrollableSurfaceError(
                    f"development {record.id} is not the development {dependency} was tested for"
                )
            if record.align_revision_id != exclusion.align_revision_id:
                raise ArtifactUnrollableSurfaceError(
                    f"development {record.id} is about another axis than {dependency}"
                )


def unrollable_selection_for_development(
    document: ArtifactDocument,
    development_recipe: Mapping[str, Any],
) -> str | None:
    """The newest FRESH exclusion whose kept faces and development are these.

    A development made from what an exclusion kept depends on it, so the
    record of what was left out travels with the development and with every
    rubbing drawn on it.
    """

    selection = development_recipe.get("selection")
    if not isinstance(selection, Mapping):
        return None
    wanted = _development_fields(development_recipe)
    for record in reversed(document.records):
        if record.type != UNROLLABLE_RECORD_TYPE:
            continue
        if document.record_freshness(record.id) is not RecordFreshness.FRESH:
            continue
        try:
            receipt = unrollable_selection_receipt_from_record(record)
        except ArtifactUnrollableSurfaceError:
            continue
        development = record.recipe.get("development")
        if (
            receipt["kept_selection_sha256"] == selection.get("selection_sha256")
            and isinstance(development, Mapping)
            and _development_fields(development) == wanted
        ):
            return record.id
    return None


def unrollable_selection_for_request(
    document: ArtifactDocument,
    *,
    total_face_count: int,
    selected_face_indices: Sequence[int] | np.ndarray,
    longitudinal_axis: str,
    record_view: str,
    n_sections: int = 32,
    seam_angle_microdegrees: int | None = None,
    section_center_policy: str = SECTION_CENTER_CANONICAL_AXIS,
    station_policy: str = STATION_CENTERLINE_ARC,
) -> str | None:
    """The exclusion a development asked for with these options would depend on.

    Takes the development's own options, so the caller about to begin a
    development can ask before it does.  A request that is not a valid
    development about a measured axis has none.
    """

    if section_center_policy != SECTION_CENTER_CANONICAL_AXIS:
        return None
    try:
        development = tile_unwrap_recipe(
            longitudinal_axis=longitudinal_axis,
            record_view=record_view,
            total_face_count=total_face_count,
            selected_face_indices=selected_face_indices,
            n_sections=n_sections,
            seam_angle_microdegrees=seam_angle_microdegrees,
            section_center_policy=section_center_policy,
            station_policy=station_policy,
        )
    except ArtifactTileUnwrapError:
        return None
    return unrollable_selection_for_development(document, development)


def verify_unrollable_selection_record_against_mesh(
    record: DerivedRecord,
    mesh: MeshData,
    *,
    cancellation_probe: CancellationProbe | None = None,
) -> dict[str, Any]:
    """Recheck the claim by running the request again.

    ``mesh`` must be the canonical projection under the record's own Align
    revision.  What was left out, face for face, and every count must come
    out the same; areas and centres, recomputed in floating point, within
    two micrometres (or square millimetres).
    """

    receipt = unrollable_selection_receipt_from_record(record)
    rerun, _qc, _kept = extract_unrollable_selection(
        mesh, _thaw(record.recipe), cancellation_probe=cancellation_probe
    )
    for key in (
        "counts",
        "excluded_selection",
        "kept_selection_sha256",
        "place_count",
        "requested_selection_sha256",
        "rounds",
        "surface_facing",
    ):
        if canonical_json_bytes(rerun[key]) != canonical_json_bytes(receipt[key]):
            raise ArtifactUnrollableSurfaceError(
                f"run again on this mesh the request leaves out another surface ({key})"
            )

    def close(left: str, right: str) -> bool:
        return abs(float(left) - float(right)) <= _RECHECK_TOLERANCE_MM

    for key in ("excluded_area_mm2_decimal", "requested_area_mm2_decimal"):
        if not close(rerun["areas"][key], receipt["areas"][key]):
            raise ArtifactUnrollableSurfaceError(
                f"run again on this mesh the request measures another {key}"
            )
    for stored, again in zip(receipt["places"], rerun["places"], strict=True):
        if stored["face_count"] != again["face_count"] or not (
            close(stored["area_mm2_decimal"], again["area_mm2_decimal"])
            and all(
                close(a, b)
                for a, b in zip(stored["centre_mm_decimal"], again["centre_mm_decimal"], strict=True)
            )
        ):
            raise ArtifactUnrollableSurfaceError(
                "run again on this mesh the request leaves out another place"
            )
    return receipt


__all__ = [
    "ArtifactUnrollableSurfaceError",
    "MAX_RECORDED_UNROLLABLE_PLACES",
    "MAX_UNROLLABLE_ROUNDS",
    "UNROLLABLE_ALGORITHM",
    "UNROLLABLE_EXCLUSION_POLICY",
    "UNROLLABLE_EXTENSION_KEY",
    "UNROLLABLE_RECIPE_KIND",
    "UNROLLABLE_RECORD_TYPE",
    "UnrollablePlace",
    "UnrollableSelectionComputation",
    "UnrollableSurface",
    "commit_unrollable_selection",
    "extract_unrollable_selection",
    "unrollable_recording_surface",
    "unrollable_selection_computation_matches_active_projection",
    "unrollable_selection_for_development",
    "unrollable_selection_for_request",
    "unrollable_selection_hash",
    "unrollable_selection_receipt_from_record",
    "unrollable_selection_recipe",
    "validate_unrollable_selection_receipt",
    "validate_unrollable_selection_recipe",
    "validate_unrollable_selection_records",
    "verify_unrollable_selection_record_against_mesh",
]
