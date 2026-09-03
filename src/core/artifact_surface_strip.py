"""회전축 기준 외면 띠: cutting the strip a rubber would tape onto the pot.

A rubbing of a pot's body is normally taken as a strip of constant width run
down one meridian, and the person taking it decides where by looking at the
pot: this meridian, about this wide, from the rim down to here.  Painting
that strip onto a mesh face by face is slow and the width comes out uneven,
so this module cuts it from those three numbers instead, about the axis the
artifact was measured on.

The hard part is not the strip, it is which surface it is on.  A vessel wall
is two sheets, and a rubbing is of the outer one.  Two independent signals
say which is which, and both must agree:

* **The face normal.**  On the outer sheet it points away from the axis, on
  the inner sheet towards it.  This is only meaningful if the patch is
  consistently wound, so an inconsistently wound patch is refused rather
  than guessed at.
* **The radius.**  The outer sheet is the one farther from the axis.  When
  both sheets are present and the outward-facing set is the *nearer* one,
  the mesh is inside-out and the whole cut is refused: silently returning
  the inner wall would put an unrecognisable rubbing on a drawing.

Nothing here is a record.  The result is a face selection, exactly like a
painted one, and the tile-unwrap record that consumes it stores the faces it
was given.  The parameters that produced them are not part of that record
yet; see docs/POTTERY_STRIP_UNWRAP.md.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .artifact_cancellation import CancellationProbe, raise_if_cancelled


STRIP_SELECTION_SCHEMA_VERSION = "1.0.0"
STRIP_SELECTION_KIND = "meridional_outer_strip/v1"

# A wall whose profile slope is 3.7 still faces outwards by this much, while a
# rim annulus or a floor disc (normal along the axis) does not.
DEFAULT_STRIP_NORMAL_ANGLE_MICRODEGREES = 75_000_000
MIN_STRIP_NORMAL_ANGLE_MICRODEGREES = 1_000_000
MAX_STRIP_NORMAL_ANGLE_MICRODEGREES = 89_000_000

MIN_STRIP_WIDTH_UM = 100
MAX_STRIP_WIDTH_UM = 10_000_000
MAX_STRIP_HEIGHT_UM = 100_000_000
MIN_STRIP_ANGLE_MICRODEGREES = -180_000_000
MAX_STRIP_ANGLE_MICRODEGREES_EXCLUSIVE = 180_000_000

# The tile unwrap refuses a recording surface above this, so a strip that
# exceeds it could never be unrolled; say so here instead of later.
MAX_STRIP_FACES = 250_000
MAX_STRIP_SOURCE_FACES = 2_000_000
MAX_STRIP_SOURCE_VERTICES = 5_000_000

_MINIMUM_RADIUS_MM = 1e-3
_AXIS_BASIS = {"x": (1, 2), "y": (2, 0), "z": (0, 1)}


class ArtifactSurfaceStripError(ValueError):
    """A strip of the outer surface cannot be cut safely from this mesh."""


def _strict_int(value: object, *, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ArtifactSurfaceStripError(f"{name} must be an integer")
    number = int(value)
    if number < minimum or number > maximum:
        raise ArtifactSurfaceStripError(
            f"{name} must be in the inclusive range {minimum}..{maximum}"
        )
    return number


def _axis(value: object) -> str:
    if not isinstance(value, str) or value not in _AXIS_BASIS:
        raise ArtifactSurfaceStripError("longitudinal_axis must be x, y, or z")
    return value


def strip_parameters(
    *,
    longitudinal_axis: str = "z",
    reference_angle_microdegrees: int = 0,
    width_um: int | None = None,
    minimum_height_um: int | None = None,
    maximum_height_um: int | None = None,
    maximum_normal_angle_microdegrees: int = DEFAULT_STRIP_NORMAL_ANGLE_MICRODEGREES,
) -> dict[str, Any]:
    """Resolve the three numbers a rubber would decide, in exact integers.

    ``width_um`` is the width of the paper, measured along the surface, so the
    angular half-width narrows as the body swells.  ``None`` means the whole
    revolution.  The height range is measured along the axis; ``None`` on
    either end means the artifact's own extent there.
    """

    axis = _axis(longitudinal_axis)
    angle = _strict_int(
        reference_angle_microdegrees,
        name="reference_angle_microdegrees",
        minimum=MIN_STRIP_ANGLE_MICRODEGREES,
        maximum=MAX_STRIP_ANGLE_MICRODEGREES_EXCLUSIVE - 1,
    )
    width = (
        None
        if width_um is None
        else _strict_int(
            width_um,
            name="width_um",
            minimum=MIN_STRIP_WIDTH_UM,
            maximum=MAX_STRIP_WIDTH_UM,
        )
    )
    minimum_height = (
        None
        if minimum_height_um is None
        else _strict_int(
            minimum_height_um,
            name="minimum_height_um",
            minimum=-MAX_STRIP_HEIGHT_UM,
            maximum=MAX_STRIP_HEIGHT_UM,
        )
    )
    maximum_height = (
        None
        if maximum_height_um is None
        else _strict_int(
            maximum_height_um,
            name="maximum_height_um",
            minimum=-MAX_STRIP_HEIGHT_UM,
            maximum=MAX_STRIP_HEIGHT_UM,
        )
    )
    if (
        minimum_height is not None
        and maximum_height is not None
        and minimum_height >= maximum_height
    ):
        raise ArtifactSurfaceStripError(
            "minimum_height_um must be below maximum_height_um"
        )
    return {
        "kind": STRIP_SELECTION_KIND,
        "longitudinal_axis": axis,
        "maximum_height_um": maximum_height,
        "maximum_normal_angle_microdegrees": _strict_int(
            maximum_normal_angle_microdegrees,
            name="maximum_normal_angle_microdegrees",
            minimum=MIN_STRIP_NORMAL_ANGLE_MICRODEGREES,
            maximum=MAX_STRIP_NORMAL_ANGLE_MICRODEGREES,
        ),
        "minimum_height_um": minimum_height,
        "reference_angle_microdegrees": angle,
        "schema_version": STRIP_SELECTION_SCHEMA_VERSION,
        "width_um": width,
    }


def validate_strip_parameters(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactSurfaceStripError("strip parameters must be an object")
    expected = {
        "kind",
        "longitudinal_axis",
        "maximum_height_um",
        "maximum_normal_angle_microdegrees",
        "minimum_height_um",
        "reference_angle_microdegrees",
        "schema_version",
        "width_um",
    }
    keys = set(value.keys())
    if keys != expected:
        missing = sorted(expected - keys)
        unknown = sorted(keys - expected)
        raise ArtifactSurfaceStripError(
            f"strip parameter keys are invalid (missing={missing}, unknown={unknown})"
        )
    if value["kind"] != STRIP_SELECTION_KIND:
        raise ArtifactSurfaceStripError("strip parameter kind is unsupported")
    if value["schema_version"] != STRIP_SELECTION_SCHEMA_VERSION:
        raise ArtifactSurfaceStripError("strip parameter schema is unsupported")
    return strip_parameters(
        longitudinal_axis=value["longitudinal_axis"],  # type: ignore[arg-type]
        reference_angle_microdegrees=value["reference_angle_microdegrees"],  # type: ignore[arg-type]
        width_um=value["width_um"],  # type: ignore[arg-type]
        minimum_height_um=value["minimum_height_um"],  # type: ignore[arg-type]
        maximum_height_um=value["maximum_height_um"],  # type: ignore[arg-type]
        maximum_normal_angle_microdegrees=value[  # type: ignore[arg-type]
            "maximum_normal_angle_microdegrees"
        ],
    )


@dataclass(frozen=True, slots=True)
class SurfaceStripSelection:
    """Sorted source-face indices of one outer-surface strip, and what it cost."""

    face_indices: np.ndarray
    parameters: Mapping[str, Any]
    qc: Mapping[str, Any]

    def __post_init__(self) -> None:
        indices = np.asarray(self.face_indices, dtype=np.int64).reshape(-1)
        if indices.size < 1:
            raise ArtifactSurfaceStripError("strip selection is empty")
        if np.any(np.diff(indices) <= 0):
            raise ArtifactSurfaceStripError(
                "strip face indices must be sorted and unique"
            )
        indices.setflags(write=False)
        object.__setattr__(self, "face_indices", indices)
        object.__setattr__(
            self, "parameters", MappingProxyType(dict(self.parameters))
        )
        object.__setattr__(self, "qc", MappingProxyType(dict(self.qc)))

    @property
    def face_count(self) -> int:
        return int(self.face_indices.shape[0])

    def parameters_dict(self) -> dict[str, Any]:
        return dict(self.parameters)

    def qc_dict(self) -> dict[str, Any]:
        return dict(self.qc)


def _validated_arrays(
    vertices_world_mm: object,
    faces: object,
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(vertices_world_mm, dtype=np.float64)
    triangles = np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or vertices.shape[0] < 3:
        raise ArtifactSurfaceStripError("vertices must be an (N, 3) array")
    if vertices.shape[0] > MAX_STRIP_SOURCE_VERTICES:
        raise ArtifactSurfaceStripError("mesh exceeds the strip vertex safety limit")
    if triangles.dtype.kind not in {"i", "u"} or triangles.ndim != 2:
        raise ArtifactSurfaceStripError("faces must be an integer (M, 3) array")
    if triangles.shape[1] != 3 or triangles.shape[0] < 1:
        raise ArtifactSurfaceStripError("faces must be an integer (M, 3) array")
    if triangles.shape[0] > MAX_STRIP_SOURCE_FACES:
        raise ArtifactSurfaceStripError("mesh exceeds the strip face safety limit")
    triangles = np.asarray(triangles, dtype=np.int64)
    if np.any(triangles < 0) or np.any(triangles >= vertices.shape[0]):
        raise ArtifactSurfaceStripError("face index is out of range")
    if not bool(np.isfinite(vertices).all()):
        raise ArtifactSurfaceStripError("mesh contains non-finite coordinates")
    return vertices, triangles


def _component_labels(
    faces: np.ndarray,
    *,
    cancellation_probe: CancellationProbe | None,
) -> np.ndarray:
    """Label each face with its edge-connected piece; a shared vertex is not a join."""

    raise_if_cancelled(cancellation_probe)
    face_count = int(faces.shape[0])
    edges = np.sort(
        np.concatenate(
            (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0
        ),
        axis=1,
    )
    raise_if_cancelled(cancellation_probe)
    _unique, inverse, counts = np.unique(
        edges, axis=0, return_inverse=True, return_counts=True
    )
    inverse = np.asarray(inverse).reshape(-1)
    raise_if_cancelled(cancellation_probe)
    owners = np.tile(np.arange(face_count, dtype=np.int64), 3)
    shared = counts[inverse] > 1
    order = np.argsort(inverse[shared], kind="stable")
    grouped_edge = inverse[shared][order]
    grouped_face = owners[shared][order]
    raise_if_cancelled(cancellation_probe)

    parent = list(range(face_count))

    def find(index: int) -> int:
        root = index
        while parent[root] != root:
            root = parent[root]
        while parent[index] != index:
            parent[index], index = root, parent[index]
        return root

    start = 0
    total = int(grouped_edge.shape[0])
    while start < total:
        stop = start + 1
        while stop < total and grouped_edge[stop] == grouped_edge[start]:
            stop += 1
        first = find(int(grouped_face[start]))
        for offset in range(start + 1, stop):
            other = find(int(grouped_face[offset]))
            if other != first:
                parent[other] = first
        start = stop
    raise_if_cancelled(cancellation_probe)
    return np.fromiter(
        (find(index) for index in range(face_count)),
        dtype=np.int64,
        count=face_count,
    )


def _inconsistent_oriented_edge_count(faces: np.ndarray) -> int:
    """Directed edges that repeat: the patch disagrees with itself about front."""

    directed = np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0
    )
    _unique, counts = np.unique(directed, axis=0, return_counts=True)
    return int(np.count_nonzero(counts > 1))


def select_surface_strip(
    vertices_world_mm: object,
    faces: object,
    parameters: Mapping[str, Any],
    *,
    largest_component: bool = False,
    cancellation_probe: CancellationProbe | None = None,
) -> SurfaceStripSelection:
    """Cut one meridional strip of the outer surface about the canonical axis.

    ``vertices_world_mm`` must already be in the canonical frame the artifact
    was positioned into, because the axis this measures about is the canonical
    origin's.  Under a manual drag that origin means nothing, and the caller
    is responsible for refusing there.
    """

    validated = validate_strip_parameters(parameters)
    vertices, triangles = _validated_arrays(vertices_world_mm, faces)
    raise_if_cancelled(cancellation_probe)

    axis_index = "xyz".index(str(validated["longitudinal_axis"]))
    first_index, second_index = _AXIS_BASIS[str(validated["longitudinal_axis"])]
    corners = vertices[triangles]
    centroids = corners.mean(axis=1)
    raise_if_cancelled(cancellation_probe)
    heights = centroids[:, axis_index]
    plane = np.column_stack((centroids[:, first_index], centroids[:, second_index]))
    radii = np.hypot(plane[:, 0], plane[:, 1])
    raise_if_cancelled(cancellation_probe)

    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    degenerate = lengths <= 0.0
    safe_lengths = np.where(degenerate, 1.0, lengths)
    normals = normals / safe_lengths[:, None]
    raise_if_cancelled(cancellation_probe)

    near_axis = radii <= _MINIMUM_RADIUS_MM
    safe_radii = np.where(near_axis, 1.0, radii)
    outward = (
        normals[:, first_index] * plane[:, 0]
        + normals[:, second_index] * plane[:, 1]
    ) / safe_radii
    raise_if_cancelled(cancellation_probe)

    # The window is tested at the vertices, not at the face centre.  A
    # triangle that holds any point of the strip must have a vertex on the
    # inside of it, so taking every face with a vertex in the window covers
    # the whole width the caller asked for.  Testing centres instead lets the
    # boundary land wherever the facet spacing falls - and that spacing is the
    # arc r * dtheta, wider where the body swells - so the strip came out with
    # its width quantised per row: stacked trapezoids rather than a band.
    vertex_heights = vertices[:, axis_index]
    vertex_plane = np.column_stack(
        (vertices[:, first_index], vertices[:, second_index])
    )
    vertex_radii = np.hypot(vertex_plane[:, 0], vertex_plane[:, 1])
    inside = np.ones((vertices.shape[0],), dtype=bool)
    minimum_height = validated["minimum_height_um"]
    maximum_height = validated["maximum_height_um"]
    if minimum_height is not None:
        inside &= vertex_heights >= float(minimum_height) / 1000.0
    if maximum_height is not None:
        inside &= vertex_heights <= float(maximum_height) / 1000.0
    width = validated["width_um"]
    if width is not None:
        reference = math.radians(
            float(validated["reference_angle_microdegrees"]) / 1_000_000.0
        )
        vertex_angles = np.arctan2(vertex_plane[:, 1], vertex_plane[:, 0])
        offset = np.abs(
            np.mod(vertex_angles - reference + math.pi, 2.0 * math.pi) - math.pi
        )
        window_arc = offset * vertex_radii
        half_width_mm = float(width) / 2000.0
        # A vertex on the axis has no meridian of its own, so it cannot put a
        # face inside a strip that is about one.
        inside &= (window_arc <= half_width_mm) & (
            vertex_radii > _MINIMUM_RADIUS_MM
        )
    raise_if_cancelled(cancellation_probe)
    window = ~near_axis & ~degenerate & inside[triangles].any(axis=1)
    raise_if_cancelled(cancellation_probe)

    candidate_count = int(np.count_nonzero(window))
    if candidate_count == 0:
        raise ArtifactSurfaceStripError(
            "no face lies in this strip; check the meridian angle, the width, "
            "and the height range against the artifact's own extent"
        )

    threshold = math.cos(
        math.radians(
            float(validated["maximum_normal_angle_microdegrees"]) / 1_000_000.0
        )
    )
    outer = window & (outward >= threshold)
    inner = window & (outward <= -threshold)
    outer_count = int(np.count_nonzero(outer))
    inner_count = int(np.count_nonzero(inner))
    raise_if_cancelled(cancellation_probe)

    if outer_count == 0:
        raise ArtifactSurfaceStripError(
            f"none of the {candidate_count} faces in this strip faces away from "
            "the axis: the strip may be too narrow to catch a whole face, the "
            "artifact may not be positioned on its measured rotation axis, or "
            "the mesh may be wound inside out"
        )

    outer_radius_mean = float(np.mean(radii[outer]))
    inner_radius_mean = float(np.mean(radii[inner])) if inner_count else float("nan")
    if inner_count and inner_radius_mean >= outer_radius_mean:
        # Both sheets are here and the outward-facing one is the nearer to the
        # axis.  That is an inside-out mesh, and taking the rubbing from it
        # would put the wrong surface on the drawing.
        raise ArtifactSurfaceStripError(
            "the outward-facing faces of this strip sit closer to the axis "
            f"({outer_radius_mean:.3f} mm) than the inward-facing ones "
            f"({inner_radius_mean:.3f} mm); this mesh appears to be wound "
            "inside out, so its outer surface cannot be told from its inner"
        )

    selected_faces = triangles[outer]
    inconsistent = _inconsistent_oriented_edge_count(selected_faces)
    if inconsistent:
        raise ArtifactSurfaceStripError(
            f"{inconsistent} directed edges of this strip repeat, so the patch "
            "either carries duplicated faces or is wound inconsistently and "
            "which side faces out cannot be decided; repair the mesh first"
        )
    raise_if_cancelled(cancellation_probe)

    indices = np.flatnonzero(outer).astype(np.int64)
    if indices.size > MAX_STRIP_FACES:
        raise ArtifactSurfaceStripError(
            f"this strip selects {indices.size} faces, above the "
            f"{MAX_STRIP_FACES}-face limit an unwrap can carry; narrow the "
            "width or the height range"
        )
    labels = _component_labels(
        selected_faces,
        cancellation_probe=cancellation_probe,
    )
    unique_labels, sizes = np.unique(labels, return_counts=True)
    components = int(unique_labels.size)
    dropped = 0
    if components != 1:
        ordered = np.sort(sizes)[::-1]
        if not largest_component:
            # A height range cut across a triangulated band can leave a single
            # face hanging, and so can a window that caught two real surfaces.
            # The sizes are what tells those apart, so name them.
            listed = ", ".join(str(int(size)) for size in ordered[:8])
            more = "" if ordered.size <= 8 else ", ..."
            raise ArtifactSurfaceStripError(
                f"this strip falls into {components} separate pieces of "
                f"{listed}{more} faces; an unwrap needs one connected surface, "
                "so adjust the window or keep the largest piece explicitly"
            )
        best = _largest_label(unique_labels, sizes, labels=labels, indices=indices)
        keep = labels == best
        dropped = int(indices.size) - int(np.count_nonzero(keep))
        indices = np.asarray(indices[keep], dtype=np.int64)
        selected_faces = triangles[indices]
    raise_if_cancelled(cancellation_probe)

    selected_radii = radii[indices]
    selected_heights = heights[indices]
    qc = {
        "candidate_face_count": candidate_count,
        "component_count": components,
        "degenerate_face_count": int(np.count_nonzero(degenerate)),
        "discarded_component_face_count": dropped,
        "inward_face_count": inner_count,
        "inward_mean_radius_um": (
            int(round(inner_radius_mean * 1000.0)) if inner_count else None
        ),
        "maximum_height_um": int(round(float(np.max(selected_heights)) * 1000.0)),
        "maximum_radius_um": int(round(float(np.max(selected_radii)) * 1000.0)),
        "minimum_height_um": int(round(float(np.min(selected_heights)) * 1000.0)),
        "minimum_radius_um": int(round(float(np.min(selected_radii)) * 1000.0)),
        "near_axis_face_count": int(np.count_nonzero(near_axis)),
        "orientation_consistent": True,
        "outward_face_count": outer_count,
        "outward_mean_radius_um": int(round(outer_radius_mean * 1000.0)),
        "selected_face_count": int(indices.size),
        "source_face_count": int(triangles.shape[0]),
    }
    return SurfaceStripSelection(
        face_indices=indices,
        parameters=validated,
        qc=qc,
    )


def _largest_label(
    unique_labels: np.ndarray,
    sizes: np.ndarray,
    *,
    labels: np.ndarray,
    indices: np.ndarray,
) -> int:
    """The biggest piece, ties broken by the lowest source face index.

    Without the tie-break the answer would depend on how numpy happened to
    order two pieces of equal size, and one mesh could give two strips.
    """

    candidates = unique_labels[sizes == sizes.max()]
    if int(candidates.size) == 1:
        return int(candidates[0])
    return int(
        min(
            (int(label) for label in candidates),
            key=lambda label: int(np.min(indices[labels == label])),
        )
    )


def select_positioned_surface_strip(
    session: Any,
    parameters: Mapping[str, Any],
    *,
    largest_component: bool = False,
    cancellation_probe: CancellationProbe | None = None,
) -> SurfaceStripSelection:
    """Cut the strip on a session, refusing one that was not stood on its axis.

    The meridian angle, the width, and the height range are all measured about
    the canonical origin.  Under a manual drag that origin is wherever the drag
    left it, so a strip cut there would name a place on the pot that does not
    exist.
    """

    from .artifact_axis_alignment import AXIS_ALIGN_RECIPE_KIND  # noqa: PLC0415
    from .artifact_session import (  # noqa: PLC0415
        ArtifactSession,
        ArtifactSessionError,
    )

    if not isinstance(session, ArtifactSession):
        raise ArtifactSurfaceStripError("session must be an ArtifactSession")
    align_id = session.document.active_align_revision_id
    align = (
        session.document.align_revision_index.get(align_id)
        if isinstance(align_id, str)
        else None
    )
    if align is None or align.recipe.get("kind") != AXIS_ALIGN_RECIPE_KIND:
        raise ArtifactSurfaceStripError(
            "cutting a strip about the axis needs an artifact positioned on its "
            "measured rotation axis; the active Align was not made from one"
        )
    try:
        projection = session.materialize()
    except ArtifactSessionError as exc:
        raise ArtifactSurfaceStripError(str(exc)) from exc
    return select_surface_strip(
        projection.mesh.vertices,
        projection.mesh.faces,
        parameters,
        largest_component=largest_component,
        cancellation_probe=cancellation_probe,
    )


__all__ = [
    "ArtifactSurfaceStripError",
    "DEFAULT_STRIP_NORMAL_ANGLE_MICRODEGREES",
    "MAX_STRIP_FACES",
    "MAX_STRIP_WIDTH_UM",
    "STRIP_SELECTION_KIND",
    "STRIP_SELECTION_SCHEMA_VERSION",
    "SurfaceStripSelection",
    "select_positioned_surface_strip",
    "select_surface_strip",
    "strip_parameters",
    "validate_strip_parameters",
]
