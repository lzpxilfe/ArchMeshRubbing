"""Put an artifact's rotation axis on +Z, from two measured circles.

Positioning is the first step of recording an artifact and, until now, the only
native Align recipe was `manual_scene_trs_delta`: the archaeologist dragged the
pot until the rotation axis looked upright.  Everything downstream inherits that
guess, because an elevation and a section are cut against the aligned frame.

A wheel-thrown vessel gives a better answer than the eye can.  Its rim and its
base are circles about one axis, and `measurement.circle_diameter.v1` already
fits each one and records the centre and the plane normal.  The line joining the
two centres *is* the rotation axis.  This module turns two such records into an
AlignRevision that carries the axis to +Z and the lower circle's centre to the
origin, so the vessel stands on Z = 0 the way it stands on a table.

Two properties make the result checkable rather than merely computed:

* **The recipe carries its own inputs.**  The two centres and normals are
  embedded as the same fixed decimals the records store, so the matrix can be
  recomputed from the recipe alone.  An offline verifier reading an export
  package has no access to the records, and a recipe that only named record ids
  would be an assertion rather than a derivation.
* **The arithmetic is exact.**  Nothing here re-fits anything.  The centres are
  numbers read from finished records, and the rotation is built by Rodrigues
  from them, so a recomputation reproduces the matrix bit for bit.  There is no
  eigendecomposition and no least squares in this path, and therefore no BLAS
  dependence to make re-verification fragile.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .alignment_utils import (
    compose_align_matrices,
    require_rigid_matrix4x4,
    rotation_matrix_align_vectors,
)
from .artifact_document import ArtifactDocument, DerivedRecord
from .artifact_surface_measurement import (
    SURFACE_DIAMETER_RECORD_TYPE,
    ArtifactSurfaceMeasurementError,
    surface_measurement_receipt_from_record,
)


AXIS_ALIGN_RECIPE_KIND = "rotation_axis_from_circle_records/v1"
AXIS_ALIGN_CONVENTION = "delta @ parent"

# The canonical axis a positioned artifact stands on.
CANONICAL_AXIS = (0.0, 0.0, 1.0)

# Below this the direction between two circle centres is dominated by their own
# fit error rather than by the artifact's shape.  Both gates apply: a small pot
# needs the absolute floor, a wide shallow bowl needs the radius-relative one.
MINIMUM_CENTER_SEPARATION_MM = 5.0
MINIMUM_SEPARATION_TO_RADIUS_RATIO = 0.25

# Two circles about one axis have normals parallel to it.  Past this, they are
# not coaxial and the line joining their centres is not the rotation axis.
MAXIMUM_NORMAL_DISAGREEMENT_DEG = 20.0

_QC_DECIMALS = 6


class ArtifactAxisAlignmentError(ValueError):
    """Two circle records cannot establish a rotation axis."""


def _decimal_vector(value: object, *, field_name: str) -> np.ndarray:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ArtifactAxisAlignmentError(f"{field_name} must be three decimal strings")
    if len(value) != 3:
        raise ArtifactAxisAlignmentError(f"{field_name} must have three components")
    numbers: list[float] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ArtifactAxisAlignmentError(
                f"{field_name}[{index}] must be a decimal string"
            )
        try:
            number = float(Decimal(item))
        except (InvalidOperation, ValueError) as exc:
            raise ArtifactAxisAlignmentError(
                f"{field_name}[{index}] is not a decimal number"
            ) from exc
        if not math.isfinite(number):
            raise ArtifactAxisAlignmentError(f"{field_name}[{index}] must be finite")
        numbers.append(number)
    return np.asarray(numbers, dtype=np.float64)


def _unit(vector: np.ndarray, *, field_name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ArtifactAxisAlignmentError(f"{field_name} must have non-zero length")
    return vector / norm


def _angle_between_deg(first: np.ndarray, second: np.ndarray) -> float:
    dot = float(np.clip(np.dot(_unit(first, field_name="vector"),
                               _unit(second, field_name="vector")), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def _undirected_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    """Angle between two lines, ignoring which way each vector points.

    A circle's normal has no inherent up: the fit canonicalises its sign by the
    dominant component, so a rim and a base of the same pot can come back
    pointing opposite ways.  What matters is whether the two planes are
    parallel, which is a question about lines, not arrows.
    """

    angle = _angle_between_deg(first, second)
    return min(angle, 180.0 - angle)


def _circle_from_record(
    record: DerivedRecord,
    *,
    field_name: str,
) -> tuple[np.ndarray, np.ndarray, float, list[str], list[str]]:
    if record.type != SURFACE_DIAMETER_RECORD_TYPE:
        raise ArtifactAxisAlignmentError(
            f"{field_name} must be a {SURFACE_DIAMETER_RECORD_TYPE} record, "
            f"not {record.type!r}"
        )
    try:
        receipt = surface_measurement_receipt_from_record(record)
    except ArtifactSurfaceMeasurementError as exc:
        raise ArtifactAxisAlignmentError(f"{field_name}: {exc}") from exc
    measurement = receipt["measurement"]
    center_decimal = list(measurement["center_mm_decimal"])
    normal_decimal = list(measurement["normal_unit_decimal"])
    center = _decimal_vector(center_decimal, field_name=f"{field_name}.center")
    normal = _decimal_vector(normal_decimal, field_name=f"{field_name}.normal")
    radius = float(Decimal(str(measurement["radius_mm_decimal"])))
    return center, normal, radius, center_decimal, normal_decimal


def _delta_from_axis(axis: np.ndarray, origin_mm: np.ndarray) -> np.ndarray:
    """Return the rigid delta taking `axis` to +Z and `origin_mm` to the origin."""

    rotation = rotation_matrix_align_vectors(axis, np.asarray(CANONICAL_AXIS))
    delta = np.eye(4, dtype=np.float64)
    delta[:3, :3] = rotation
    delta[:3, 3] = -(rotation @ origin_mm)
    return require_rigid_matrix4x4(delta, field_name="axis_align_delta")


def axis_align_delta_from_recipe(recipe: Mapping[str, Any]) -> np.ndarray:
    """Recompute the rigid delta from the recipe alone.

    This is the derivation an offline verifier runs.  It reads only the recipe,
    never the records, because an export package carries the Align ancestry
    without the measurements it came from.
    """

    if not isinstance(recipe, Mapping):
        raise ArtifactAxisAlignmentError("axis align recipe must be a mapping")
    if recipe.get("kind") != AXIS_ALIGN_RECIPE_KIND:
        raise ArtifactAxisAlignmentError(
            f"axis align recipe kind must be {AXIS_ALIGN_RECIPE_KIND!r}"
        )
    if recipe.get("convention") != AXIS_ALIGN_CONVENTION:
        raise ArtifactAxisAlignmentError("axis align recipe convention is invalid")
    top_center = _decimal_vector(
        recipe.get("top_center_mm_decimal"), field_name="recipe.top_center_mm_decimal"
    )
    bottom_center = _decimal_vector(
        recipe.get("bottom_center_mm_decimal"),
        field_name="recipe.bottom_center_mm_decimal",
    )
    separation = top_center - bottom_center
    if float(np.linalg.norm(separation)) <= 0.0:
        raise ArtifactAxisAlignmentError(
            "axis align recipe centres coincide, so it names no axis"
        )
    return _delta_from_axis(separation, bottom_center)


def _quantized(value: float) -> float:
    return float(round(float(value), _QC_DECIMALS))


def build_axis_alignment(
    document: ArtifactDocument,
    *,
    top_record_id: str,
    bottom_record_id: str,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Return the composed Align matrix, its recipe and its QC.

    `top` and `bottom` name which circle sits higher on the finished drawing;
    the axis runs from bottom to top, so the vessel ends up standing rather than
    inverted.
    """

    if not isinstance(document, ArtifactDocument):
        raise ArtifactAxisAlignmentError("document must be an ArtifactDocument")
    top_id = str(top_record_id)
    bottom_id = str(bottom_record_id)
    if top_id == bottom_id:
        raise ArtifactAxisAlignmentError(
            "the upper and lower circle must be two different records"
        )
    records = document.record_index
    for field_name, record_id in (("top_record_id", top_id), ("bottom_record_id", bottom_id)):
        if record_id not in records:
            raise ArtifactAxisAlignmentError(
                f"{field_name} {record_id!r} does not exist in this document"
            )
    top_record = records[top_id]
    bottom_record = records[bottom_id]

    parent_id = document.active_align_revision_id
    if parent_id is None:
        raise ArtifactAxisAlignmentError("an active Align revision is required")
    parent = document.align_revision_index[parent_id]
    # Both centres have to be numbers in one frame.  Circles measured under
    # different alignments are in different coordinate systems, and the line
    # between them would mean nothing.
    for field_name, record in (("top", top_record), ("bottom", bottom_record)):
        if record.align_revision_id != parent_id:
            raise ArtifactAxisAlignmentError(
                f"the {field_name} circle was measured under a different Align "
                f"({record.align_revision_id!r}); re-measure it under the active "
                "Align before using it to set the axis"
            )

    top_center, top_normal, top_radius, top_center_decimal, top_normal_decimal = (
        _circle_from_record(top_record, field_name="top circle")
    )
    (
        bottom_center,
        bottom_normal,
        bottom_radius,
        bottom_center_decimal,
        bottom_normal_decimal,
    ) = _circle_from_record(bottom_record, field_name="bottom circle")

    separation_vector = top_center - bottom_center
    separation = float(np.linalg.norm(separation_vector))
    largest_radius = max(top_radius, bottom_radius)
    if separation < MINIMUM_CENTER_SEPARATION_MM or (
        largest_radius > 0.0
        and separation < largest_radius * MINIMUM_SEPARATION_TO_RADIUS_RATIO
    ):
        raise ArtifactAxisAlignmentError(
            f"the two circle centres are {separation:.3f} mm apart, which is too "
            f"close to fix an axis (needs at least "
            f"{MINIMUM_CENTER_SEPARATION_MM:.3f} mm and at least "
            f"{MINIMUM_SEPARATION_TO_RADIUS_RATIO:.0%} of the {largest_radius:.3f} mm "
            "radius). Measure circles further apart, such as the rim and the base."
        )

    axis = _unit(separation_vector, field_name="axis")
    normal_disagreement = _undirected_angle_deg(top_normal, bottom_normal)
    top_axis_disagreement = _undirected_angle_deg(top_normal, axis)
    bottom_axis_disagreement = _undirected_angle_deg(bottom_normal, axis)
    worst = max(normal_disagreement, top_axis_disagreement, bottom_axis_disagreement)
    if worst > MAXIMUM_NORMAL_DISAGREEMENT_DEG:
        raise ArtifactAxisAlignmentError(
            f"the two circles are not coaxial: their planes and the line joining "
            f"their centres disagree by up to {worst:.2f}°, over the "
            f"{MAXIMUM_NORMAL_DISAGREEMENT_DEG:.2f}° limit. The line between these "
            "centres is not the rotation axis."
        )

    recipe = {
        "bottom_center_mm_decimal": bottom_center_decimal,
        "bottom_normal_unit_decimal": bottom_normal_decimal,
        "bottom_record_id": bottom_id,
        "convention": AXIS_ALIGN_CONVENTION,
        "kind": AXIS_ALIGN_RECIPE_KIND,
        "top_center_mm_decimal": top_center_decimal,
        "top_normal_unit_decimal": top_normal_decimal,
        "top_record_id": top_id,
    }
    delta = axis_align_delta_from_recipe(recipe)
    matrix = compose_align_matrices(delta, parent.matrix)

    qc = {
        "axis_tilt_corrected_deg": _quantized(
            _angle_between_deg(axis, np.asarray(CANONICAL_AXIS))
        ),
        "center_separation_mm": _quantized(separation),
        "circle_normal_disagreement_deg": _quantized(worst),
        "proper_rigid": True,
    }
    return matrix, recipe, qc


def verify_axis_alignment_matrix(
    *,
    recipe: Mapping[str, Any],
    parent_matrix: np.ndarray,
    matrix: np.ndarray,
) -> None:
    """Prove a stored Align matrix is the one its recipe derives.

    Exact equality is the right standard here.  The recipe holds fixed decimals
    and the derivation is Rodrigues arithmetic, so any difference means the
    matrix was not produced by this recipe.
    """

    delta = axis_align_delta_from_recipe(recipe)
    recomputed = compose_align_matrices(delta, np.asarray(parent_matrix, dtype=np.float64))
    stored = np.asarray(matrix, dtype=np.float64)
    if stored.shape != recomputed.shape or not np.array_equal(stored, recomputed):
        raise ArtifactAxisAlignmentError(
            "the stored Align matrix is not the one its axis recipe derives"
        )


__all__ = [
    "AXIS_ALIGN_CONVENTION",
    "AXIS_ALIGN_RECIPE_KIND",
    "ArtifactAxisAlignmentError",
    "CANONICAL_AXIS",
    "MAXIMUM_NORMAL_DISAGREEMENT_DEG",
    "MINIMUM_CENTER_SEPARATION_MM",
    "MINIMUM_SEPARATION_TO_RADIUS_RATIO",
    "axis_align_delta_from_recipe",
    "build_axis_alignment",
    "verify_axis_alignment_matrix",
]
