"""와통 기준 축: the drum a roof tile was formed on, measured from its surface.

A wheel-thrown pot tells its axis through two circles, and
`rotation_axis_from_circle_records/v1` turns a rim and a base into an Align.
A roof tile has no rim and no base.  What it has is the 와통 - the drum the
clay slab was wrapped on - and every section of the tile is an arc of it.

Two arc sections do not fix that drum.  A 암키와 spans sixty to eighty degrees
of it, and on an arc that narrow a fitted circle's centre slides along the
arc's bisector while its radius changes to match.  On 204 암키와편 two section
circles fitted through the app's own surface anchors came back at radius 149
and 230 mm, and at 161 and 290 mm when the picks were spread wider, for a drum
whose inner wall a cylinder fit puts at 182.  The planes of those circles
agreed to 0.02 degrees; their centres did not, and a tile unrolled about the
line through them folds over.

This module measures the drum from the recording surface itself - every vertex
of the faces the recorder selected, along the whole length and across the
whole arc - and publishes it as a record:

* **Direction** from the surface normals.  Every normal of a cylinder is
  perpendicular to its axis, so the axis is the direction the area-weighted
  normal covariance has least of.  The length of the surface fixes it, which
  the arc alone cannot.
* **The cylinder** by Levenberg-Marquardt on the radial residual, from that
  direction and a normalized algebraic circle in the plane across it.  The
  algebraic fit is biased on a narrow arc; the geometric one is not, and
  thousands of points spread over the whole arc and the whole length
  condition what two hand-picked sections could not.
* **Two end sections** - the drum's own circles where the surface begins and
  ends along it: centre on the axis, plane across it, radius the drum's.
  They are what `rotation_axis_from_circle_records/v1` reads, so the Align
  that consumes them (`build_mandrel_axis_alignment`) is the existing,
  offline-verifiable one: its recipe carries two centres and two normals as
  fixed decimals and its matrix is Rodrigues arithmetic on them.

The selection is written in exactly the encoding a development uses
(`tile_unwrap_selection`), so when the drum is measured on the surface that is
then unrolled, the two records carry one `selection_sha256` and can be shown
to name one face set.

What validation does not do is re-run the fit: a solve over a hundred thousand
points is not bit-stable across BLAS builds.  `mandrel_receipt_from_record`
checks the receipt's structure, hashes and internal geometry;
`verify_mandrel_record_against_mesh` checks the claim - that the stored
cylinder fits the stored surface as closely as the receipt says - by
recomputing the residual from the stored parameters with an exactly rounded
sum.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, DecimalException, ROUND_HALF_EVEN, localcontext
import json
import math
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from .artifact_cancellation import CancellationProbe, poll_cancellation, raise_if_cancelled
from .artifact_document import (
    ArtifactDocument,
    ArtifactDocumentError,
    DerivedRecord,
    OperationContext,
    RecordLifecycleStatus,
    canonical_recipe_hash,
)
from .artifact_scene_adapter import ArtifactProjectionSnapshot
from .artifact_session import ArtifactSession, ArtifactSessionError
from .artifact_tile_unwrap_extractor import (
    ArtifactTileUnwrapError,
    selection_face_indices,
    tile_unwrap_selection,
    validate_tile_unwrap_selection,
)
from .canonical_json import canonical_json_bytes, canonical_json_sha256


MANDREL_RECORD_TYPE = "measurement.mandrel_cylinder.v1"
#: The measurement operation this recipe belongs to; every measurement
#: recipe names its kind so a work item cannot run the wrong one.
MANDREL_RECIPE_KIND = "mandrel_cylinder"
MANDREL_ALGORITHM = "archmeshrubbing.mandrel_cylinder_from_recording_surface"
MANDREL_ALGORITHM_VERSION = "1.0.0"
MANDREL_RECIPE_SCHEMA_VERSION = "1.0.0"
MANDREL_RECEIPT_SCHEMA_VERSION = "1.0.0"
MANDREL_FIT_POLICY = "normal_covariance_axis+radial_levenberg_marquardt_cylinder/v1"
MANDREL_COORDINATE_SPACE = "canonical_aligned_mm/v1"
MANDREL_EXTENSION_KEY = "org.archmeshrubbing:mandrel-cylinder-v1"
MANDREL_MEDIA_TYPE = "application/vnd.archmeshrubbing.mandrel-cylinder-receipt+json"
MANDREL_REF_PREFIX = "urn:archmeshrubbing:mandrel-cylinder:sha256:"

FACING_TOWARD_AXIS = "toward_axis"
FACING_AWAY_FROM_AXIS = "away_from_axis"
FACINGS = (FACING_TOWARD_AXIS, FACING_AWAY_FROM_AXIS)

#: Below this many faces a recording surface is a patch, not a drum.
MIN_MANDREL_FACES = 64
#: A strip narrower than this around the drum does not fix its radius; the
#: development refuses the same narrow arc for the same reason
#: (``section_arc_span_too_small``).
MIN_MANDREL_ARC_SPAN_DEG = 20.0
#: The same floor the circle alignment puts under two centres: shorter than
#: this along the drum, the two end sections name no direction.
MIN_MANDREL_LENGTH_MM = 5.0
MAX_MANDREL_RADIUS_MM = 100_000.0
MAX_MANDREL_ITERATIONS = 80
#: A recording surface holds one face of the wall.  Selected across both, the
#: fit averages two radii a slab's thickness apart and names neither; past
#: this share of area facing the other way, refuse rather than guess.
MAX_MANDREL_OPPOSITE_FACING_SHARE = 0.30
MAX_MANDREL_RECEIPT_BYTES = 64 * 1024

RESULT_DECIMAL_PLACES = 6
UNIT_DECIMAL_PLACES = 9
_RESULT_QUANTUM = Decimal("0.000001")
_UNIT_QUANTUM = Decimal("0.000000001")
_SIGNED_DECIMAL_RE = re.compile(r"^-?(0|[1-9][0-9]*)\.[0-9]{6}$")
_UNSIGNED_DECIMAL_RE = re.compile(r"^(0|[1-9][0-9]*)\.[0-9]{6}$")
_SIGNED_UNIT_RE = re.compile(r"^-?(0|[1-9][0-9]*)\.[0-9]{9}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
#: A section centre is the axis point moved along the axis by its station,
#: each of the three stored to a micrometre, so it may sit up to a few
#: micrometres off the line recomputed from them.
_SECTION_ON_AXIS_TOLERANCE_MM = 0.000004


class ArtifactMandrelError(ValueError):
    """A recording surface cannot establish the drum a tile was formed on."""


# ---------------------------------------------------------------------------
# Fixed decimals and frozen JSON
# ---------------------------------------------------------------------------


def _fixed_decimal(value: Decimal | float, quantum: Decimal = _RESULT_QUANTUM) -> str:
    number = value if isinstance(value, Decimal) else Decimal(str(float(value)))
    if not number.is_finite():
        raise ArtifactMandrelError("mandrel result must be finite")
    try:
        with localcontext() as context:
            context.prec = max(80, len(number.as_tuple().digits) + 20)
            quantized = number.quantize(quantum, rounding=ROUND_HALF_EVEN)
    except DecimalException as exc:
        raise ArtifactMandrelError(
            "mandrel result cannot be represented by the fixed decimal policy"
        ) from exc
    if quantized == 0:
        quantized = Decimal(0).quantize(quantum)
    return format(quantized, "f")


def _decimal_text(value: object, *, name: str, signed: bool = True, unit: bool = False) -> Decimal:
    if not isinstance(value, str):
        raise ArtifactMandrelError(f"{name} must be a fixed decimal string")
    pattern = _SIGNED_UNIT_RE if unit else (_SIGNED_DECIMAL_RE if signed else _UNSIGNED_DECIMAL_RE)
    if pattern.fullmatch(value) is None:
        raise ArtifactMandrelError(f"{name} is not a canonical fixed decimal")
    number = Decimal(value)
    if value.startswith("-") and number == 0:
        raise ArtifactMandrelError(f"{name} writes zero with a sign")
    return number


def _decimal_vector(value: object, *, name: str, unit: bool = False) -> list[Decimal]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ArtifactMandrelError(f"{name} must be three fixed decimals")
    return [
        _decimal_text(item, name=f"{name}[{index}]", unit=unit)
        for index, item in enumerate(value)
    ]


def _strict_int(value: object, *, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactMandrelError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ArtifactMandrelError(f"{name} must be within [{minimum}, {maximum}]")
    return int(value)


def _exact_mapping(value: object, keys: set[str] | frozenset[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactMandrelError(f"{name} must be an object")
    missing = sorted(set(keys) - set(value))
    unexpected = sorted(set(value) - set(keys))
    if missing:
        raise ArtifactMandrelError(f"{name} is missing fields: {', '.join(missing)}")
    if unexpected:
        raise ArtifactMandrelError(f"{name} has unsupported fields: {', '.join(unexpected)}")
    return value


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(value[key]) for key in sorted(value)})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _frozen_mapping(value: Mapping[str, Any], *, name: str) -> Mapping[str, Any]:
    try:
        decoded = json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ArtifactMandrelError(f"{name} is not strict JSON") from exc
    if not isinstance(decoded, dict):
        raise ArtifactMandrelError(f"{name} must be an object")
    frozen = _freeze_json(decoded)
    assert isinstance(frozen, Mapping)
    return frozen


# ---------------------------------------------------------------------------
# Recipe
# ---------------------------------------------------------------------------

_RECIPE_KEYS = frozenset(
    {
        "algorithm",
        "algorithm_version",
        "coordinate_space",
        "fit_policy",
        "kind",
        "schema_version",
        "selection",
    }
)


def mandrel_recipe(
    *,
    total_face_count: int,
    selected_face_indices: Sequence[int] | np.ndarray,
) -> dict[str, Any]:
    """The complete recipe for one 와통 measurement on a recording surface."""

    try:
        selection = tile_unwrap_selection(
            total_face_count=total_face_count,
            selected_face_indices=selected_face_indices,
        )
    except ArtifactTileUnwrapError as exc:
        raise ArtifactMandrelError(str(exc)) from exc
    if int(selection["selected_face_count"]) < MIN_MANDREL_FACES:
        raise ArtifactMandrelError(
            f"a 와통 is measured on at least {MIN_MANDREL_FACES} faces of one "
            f"recording surface; {selection['selected_face_count']} were selected"
        )
    return {
        "algorithm": MANDREL_ALGORITHM,
        "algorithm_version": MANDREL_ALGORITHM_VERSION,
        "coordinate_space": MANDREL_COORDINATE_SPACE,
        "fit_policy": MANDREL_FIT_POLICY,
        "kind": MANDREL_RECIPE_KIND,
        "schema_version": MANDREL_RECIPE_SCHEMA_VERSION,
        "selection": selection,
    }


def validate_mandrel_recipe(value: object) -> dict[str, Any]:
    recipe = _exact_mapping(value, _RECIPE_KEYS, name="mandrel recipe")
    for key, expected in (
        ("algorithm", MANDREL_ALGORITHM),
        ("algorithm_version", MANDREL_ALGORITHM_VERSION),
        ("coordinate_space", MANDREL_COORDINATE_SPACE),
        ("fit_policy", MANDREL_FIT_POLICY),
        ("kind", MANDREL_RECIPE_KIND),
        ("schema_version", MANDREL_RECIPE_SCHEMA_VERSION),
    ):
        if recipe[key] != expected:
            raise ArtifactMandrelError(f"mandrel recipe {key} is unsupported")
    try:
        selection = validate_tile_unwrap_selection(recipe["selection"])
    except ArtifactTileUnwrapError as exc:
        raise ArtifactMandrelError(f"mandrel recipe selection: {exc}") from exc
    if int(selection["selected_face_count"]) < MIN_MANDREL_FACES:
        raise ArtifactMandrelError("mandrel recipe selection is too small to fit a drum")
    output = {key: recipe[key] for key in _RECIPE_KEYS if key != "selection"}
    output["selection"] = selection
    return output


def mandrel_selection_hash(recipe: Mapping[str, Any]) -> str:
    return str(validate_mandrel_recipe(recipe)["selection"]["selection_sha256"])


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


def _canonical_sign(vector: np.ndarray) -> np.ndarray:
    """Point an undirected axis the one way, so a refit cannot flip it."""
    dominant = int(np.argmax(np.abs(vector)))
    return -vector if float(vector[dominant]) < 0.0 else vector


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.zeros(3, dtype=np.float64)
    helper[int(np.argmin(np.abs(normal)))] = 1.0
    first = np.cross(normal, helper)
    first /= float(np.linalg.norm(first))
    second = np.cross(normal, first)
    return first, second / float(np.linalg.norm(second))


def _exact_mean(values: np.ndarray) -> float:
    return math.fsum(float(value) for value in values) / len(values)


def _radial_residuals(
    points: np.ndarray, axis_point: np.ndarray, direction: np.ndarray, radius: float
) -> np.ndarray:
    offset = points - axis_point
    along = offset @ direction
    return np.linalg.norm(offset - np.outer(along, direction), axis=1) - radius


def _arc_span_deg(angles: np.ndarray) -> float:
    """How far round the drum the surface reaches: 360 minus the widest gap."""
    ordered = np.sort(np.mod(angles, 2.0 * math.pi))
    gaps = np.diff(np.concatenate([ordered, [ordered[0] + 2.0 * math.pi]]))
    return math.degrees(2.0 * math.pi - float(gaps.max()))


def _fit_drum(
    points: np.ndarray,
    face_normals: np.ndarray,
    face_areas: np.ndarray,
    *,
    cancellation_probe: CancellationProbe | None,
) -> dict[str, Any]:
    covariance = (face_normals * face_areas[:, None]).T @ face_normals
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    except np.linalg.LinAlgError as exc:
        raise ArtifactMandrelError("the surface normals give no axis") from exc
    if not np.isfinite(eigenvalues).all() or float(eigenvalues[-1]) <= 0.0:
        raise ArtifactMandrelError("the surface normals give no axis")
    if float(eigenvalues[1]) <= float(eigenvalues[-1]) * 1e-6:
        raise ArtifactMandrelError(
            "the recording surface is flat: its normals do not turn about any "
            "axis, so it is not a section of a drum"
        )
    start_direction = _canonical_sign(np.asarray(eigenvectors[:, 0], dtype=np.float64))
    e1, e2 = _plane_basis(start_direction)
    centroid = np.array([_exact_mean(points[:, axis]) for axis in range(3)])

    # A normalized algebraic circle across the start direction: biased on a
    # narrow arc, but a start the geometric refinement can walk from.
    local = points - centroid
    x = local @ e1
    y = local @ e2
    scale = float(np.max(np.hypot(x, y)))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ArtifactMandrelError("the recording surface has no extent across its axis")
    xn, yn = x / scale, y / scale
    design = np.column_stack((2.0 * xn, 2.0 * yn, np.ones(len(xn))))
    solution, _residual, rank, _singular = np.linalg.lstsq(design, xn * xn + yn * yn, rcond=None)
    if int(rank) != 3:
        raise ArtifactMandrelError("the section across the axis is circle-fit degenerate")
    radius_squared = float(solution[2] + solution[0] ** 2 + solution[1] ** 2)
    if not math.isfinite(radius_squared) or radius_squared <= 0.0:
        raise ArtifactMandrelError("the section across the axis has no positive radius")

    # Parameters: axis point offset across (a1, a2), direction tilt (b1, b2),
    # radius - all measured in the start frame.
    parameters = np.array(
        [float(solution[0]) * scale, float(solution[1]) * scale, 0.0, 0.0,
         math.sqrt(radius_squared) * scale]
    )
    steps = np.array([1e-4, 1e-4, 1e-7, 1e-7, 1e-4])

    def unpack(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        direction = start_direction + values[2] * e1 + values[3] * e2
        direction = direction / float(np.linalg.norm(direction))
        return centroid + values[0] * e1 + values[1] * e2, direction, float(values[4])

    def residuals(values: np.ndarray) -> np.ndarray:
        axis_point, direction, radius = unpack(values)
        return _radial_residuals(points, axis_point, direction, radius)

    current = residuals(parameters)
    cost = float(current @ current)
    damping = 1e-3
    iterations = 0
    converged = False
    jacobian = np.empty((len(points), 5), dtype=np.float64)
    for iterations in range(1, MAX_MANDREL_ITERATIONS + 1):
        poll_cancellation(cancellation_probe, iterations)
        for column in range(5):
            forward = parameters.copy()
            backward = parameters.copy()
            forward[column] += steps[column]
            backward[column] -= steps[column]
            jacobian[:, column] = (residuals(forward) - residuals(backward)) / (2.0 * steps[column])
        normal_matrix = jacobian.T @ jacobian
        gradient = jacobian.T @ current
        accepted = False
        for _attempt in range(12):
            damped = normal_matrix + damping * np.diag(np.diag(normal_matrix))
            try:
                delta = np.linalg.solve(damped, -gradient)
            except np.linalg.LinAlgError:
                damping *= 10.0
                continue
            trial = parameters + delta
            trial_residuals = residuals(trial)
            trial_cost = float(trial_residuals @ trial_residuals)
            if math.isfinite(trial_cost) and trial_cost <= cost:
                improvement = cost - trial_cost
                parameters, current, cost = trial, trial_residuals, trial_cost
                damping = max(damping / 10.0, 1e-12)
                accepted = True
                break
            damping *= 10.0
        if not accepted:
            # No step lowers the cost: the fit is at its minimum to the
            # precision the residual can resolve.
            converged = True
            break
        small_position = max(abs(float(delta[0])), abs(float(delta[1])), abs(float(delta[4]))) < 1e-7
        small_tilt = max(abs(float(delta[2])), abs(float(delta[3]))) < 1e-10
        if (small_position and small_tilt) or improvement <= cost * 1e-15:
            converged = True
            break
    if not converged:
        raise ArtifactMandrelError(
            f"the drum fit did not settle in {MAX_MANDREL_ITERATIONS} iterations"
        )

    axis_point, direction, radius = unpack(parameters)
    if not math.isfinite(radius) or radius <= 0.0 or radius > MAX_MANDREL_RADIUS_MM:
        raise ArtifactMandrelError(f"the drum fit produced an unusable radius ({radius!r} mm)")
    direction = _canonical_sign(direction)
    # The point on the axis nearest the surface: one well-defined point, so
    # two fits of the same surface name the same one.
    axis_point = axis_point + float((centroid - axis_point) @ direction) * direction

    scaled = jacobian / np.maximum(np.linalg.norm(jacobian, axis=0), 1e-300)
    singular = np.linalg.svd(scaled, compute_uv=False)
    condition = float(singular[0] / singular[-1]) if float(singular[-1]) > 0.0 else math.inf
    return {
        "axis_point": axis_point,
        "condition": condition,
        "direction": direction,
        "iterations": iterations,
        "radius": radius,
    }


def extract_mandrel(
    vertices: object,
    faces: object,
    recipe: Mapping[str, Any],
    *,
    cancellation_probe: CancellationProbe | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit the drum to the recipe's recording surface; return (receipt, qc)."""

    raise_if_cancelled(cancellation_probe)
    validated = validate_mandrel_recipe(recipe)
    selection = validated["selection"]
    vertex_array = np.asarray(vertices, dtype=np.float64)
    face_array = np.asarray(faces, dtype=np.int64)
    if vertex_array.ndim != 2 or vertex_array.shape[1] != 3 or not np.isfinite(vertex_array).all():
        raise ArtifactMandrelError("mesh vertices must be finite (N, 3) coordinates")
    if face_array.ndim != 2 or face_array.shape[1] != 3:
        raise ArtifactMandrelError("mesh faces must be triangles")
    if int(selection["total_face_count"]) != int(face_array.shape[0]):
        raise ArtifactMandrelError("mandrel selection does not match the mesh it is applied to")
    if face_array.size and (int(face_array.min()) < 0 or int(face_array.max()) >= len(vertex_array)):
        raise ArtifactMandrelError("mesh faces reference missing vertices")

    selected = face_array[selection_face_indices(selection)]
    corners = vertex_array[selected]
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    twice_area = np.linalg.norm(cross, axis=1)
    keep = twice_area > 0.0
    if int(keep.sum()) < MIN_MANDREL_FACES:
        raise ArtifactMandrelError("too few faces of the recording surface have area")
    normals = cross[keep] / twice_area[keep, None]
    areas = 0.5 * twice_area[keep]
    used = np.unique(selected.reshape(-1))
    points = vertex_array[used]

    fit = _fit_drum(points, normals, areas, cancellation_probe=cancellation_probe)
    axis_point = fit["axis_point"]
    direction = fit["direction"]
    radius = float(fit["radius"])

    residuals = _radial_residuals(points, axis_point, direction, radius)
    offsets = points - axis_point
    stations = offsets @ direction
    first, second = _plane_basis(direction)
    angles = np.arctan2(offsets @ second, offsets @ first)
    arc_span = _arc_span_deg(angles)
    if arc_span < MIN_MANDREL_ARC_SPAN_DEG:
        raise ArtifactMandrelError(
            f"the recording surface reaches {arc_span:.1f} degrees round its drum; "
            f"at least {MIN_MANDREL_ARC_SPAN_DEG:.0f} are needed to fix the radius"
        )
    bottom_station = float(stations.min())
    top_station = float(stations.max())
    if top_station - bottom_station < MIN_MANDREL_LENGTH_MM:
        raise ArtifactMandrelError(
            f"the recording surface runs {top_station - bottom_station:.3f} mm along "
            f"its drum; at least {MIN_MANDREL_LENGTH_MM:.0f} mm are needed"
        )

    # Which face of the wall is this?  Outward normals of the concave face
    # point toward the axis, of the convex face away from it.
    face_centres = corners[keep].mean(axis=1) - axis_point
    radial = face_centres - np.outer(face_centres @ direction, direction)
    radial_length = np.linalg.norm(radial, axis=1)
    radial_length[radial_length == 0.0] = 1.0
    facing = np.einsum("ij,ij->i", normals, radial / radial_length[:, None])
    toward = float(areas[facing < 0.0].sum())
    away = float(areas[facing >= 0.0].sum())
    total_area = toward + away
    opposite_share = min(toward, away) / total_area if total_area > 0.0 else 1.0
    if opposite_share > MAX_MANDREL_OPPOSITE_FACING_SHARE:
        raise ArtifactMandrelError(
            f"{opposite_share:.0%} of the selected area faces the other way: the "
            "selection holds both faces of the wall, and a drum fitted to both "
            "names neither radius; select the 내면 or the 외면 alone"
        )
    surface_facing = FACING_TOWARD_AXIS if toward >= away else FACING_AWAY_FROM_AXIS

    def section(station: float) -> dict[str, Any]:
        centre = axis_point + station * direction
        return {
            "center_mm_decimal": [_fixed_decimal(value) for value in centre],
            "normal_unit_decimal": [_fixed_decimal(value, _UNIT_QUANTUM) for value in direction],
            "radius_mm_decimal": _fixed_decimal(radius),
            "station_mm_decimal": _fixed_decimal(station),
        }

    squares = [float(value) * float(value) for value in residuals]
    receipt = {
        "algorithm": MANDREL_ALGORITHM,
        "algorithm_version": MANDREL_ALGORITHM_VERSION,
        "axis": {
            "point_mm_decimal": [_fixed_decimal(value) for value in axis_point],
            "unit_decimal": [_fixed_decimal(value, _UNIT_QUANTUM) for value in direction],
        },
        "coordinate_space": MANDREL_COORDINATE_SPACE,
        "fit_policy": MANDREL_FIT_POLICY,
        "input_face_count": int(face_array.shape[0]),
        "input_vertex_count": int(vertex_array.shape[0]),
        "quality": {
            "arc_span_deg_decimal": _fixed_decimal(arc_span),
            "fit_condition_decimal": _fixed_decimal(min(fit["condition"], 1e30)),
            "iterations": int(fit["iterations"]),
            "length_along_axis_mm_decimal": _fixed_decimal(
                Decimal(_fixed_decimal(top_station)) - Decimal(_fixed_decimal(bottom_station))
            ),
            "opposite_facing_area_share_decimal": _fixed_decimal(opposite_share),
            "radial_max_residual_mm_decimal": _fixed_decimal(float(np.abs(residuals).max())),
            "radial_p95_residual_mm_decimal": _fixed_decimal(
                float(np.percentile(np.abs(residuals), 95.0))
            ),
            "radial_rms_residual_mm_decimal": _fixed_decimal(
                math.sqrt(math.fsum(squares) / len(squares))
            ),
        },
        "radius_mm_decimal": _fixed_decimal(radius),
        "schema_version": MANDREL_RECEIPT_SCHEMA_VERSION,
        "sections": {"bottom": section(bottom_station), "top": section(top_station)},
        "selected_face_count": int(selection["selected_face_count"]),
        "selection_sha256": str(selection["selection_sha256"]),
        "surface_facing": surface_facing,
        "used_vertex_count": int(len(points)),
    }
    receipt = validate_mandrel_receipt(receipt)
    return receipt, _qc_from_receipt(receipt)


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------

_RECEIPT_KEYS = frozenset(
    {
        "algorithm",
        "algorithm_version",
        "axis",
        "coordinate_space",
        "fit_policy",
        "input_face_count",
        "input_vertex_count",
        "quality",
        "radius_mm_decimal",
        "schema_version",
        "sections",
        "selected_face_count",
        "selection_sha256",
        "surface_facing",
        "used_vertex_count",
    }
)
_QUALITY_KEYS = frozenset(
    {
        "arc_span_deg_decimal",
        "fit_condition_decimal",
        "iterations",
        "length_along_axis_mm_decimal",
        "opposite_facing_area_share_decimal",
        "radial_max_residual_mm_decimal",
        "radial_p95_residual_mm_decimal",
        "radial_rms_residual_mm_decimal",
    }
)
_SECTION_KEYS = frozenset(
    {"center_mm_decimal", "normal_unit_decimal", "radius_mm_decimal", "station_mm_decimal"}
)


def validate_mandrel_receipt(value: object) -> dict[str, Any]:
    """Check a receipt's structure and that its geometry holds together."""

    receipt = _exact_mapping(value, _RECEIPT_KEYS, name="mandrel receipt")
    for key, expected in (
        ("algorithm", MANDREL_ALGORITHM),
        ("algorithm_version", MANDREL_ALGORITHM_VERSION),
        ("coordinate_space", MANDREL_COORDINATE_SPACE),
        ("fit_policy", MANDREL_FIT_POLICY),
        ("schema_version", MANDREL_RECEIPT_SCHEMA_VERSION),
    ):
        if receipt[key] != expected:
            raise ArtifactMandrelError(f"mandrel receipt {key} is unsupported")
    if not isinstance(receipt["selection_sha256"], str) or _SHA256_RE.fullmatch(
        receipt["selection_sha256"]
    ) is None:
        raise ArtifactMandrelError("mandrel receipt selection_sha256 is invalid")
    if receipt["surface_facing"] not in FACINGS:
        raise ArtifactMandrelError("mandrel receipt surface_facing is invalid")
    input_faces = _strict_int(receipt["input_face_count"], name="input_face_count",
                              minimum=1, maximum=2**53 - 1)
    _strict_int(receipt["input_vertex_count"], name="input_vertex_count", minimum=3, maximum=2**53 - 1)
    _strict_int(receipt["selected_face_count"], name="selected_face_count",
                minimum=MIN_MANDREL_FACES, maximum=input_faces)
    _strict_int(receipt["used_vertex_count"], name="used_vertex_count", minimum=3,
                maximum=int(receipt["input_vertex_count"]))

    axis = _exact_mapping(receipt["axis"], {"point_mm_decimal", "unit_decimal"}, name="axis")
    point = _decimal_vector(axis["point_mm_decimal"], name="axis.point_mm_decimal")
    unit = _decimal_vector(axis["unit_decimal"], name="axis.unit_decimal", unit=True)
    unit_float = np.array([float(item) for item in unit])
    if abs(float(np.linalg.norm(unit_float)) - 1.0) > 1e-8:
        raise ArtifactMandrelError("mandrel axis unit is not a unit vector")
    radius = _decimal_text(receipt["radius_mm_decimal"], name="radius_mm_decimal", signed=False)
    if radius <= 0 or float(radius) > MAX_MANDREL_RADIUS_MM:
        raise ArtifactMandrelError("mandrel radius is out of range")

    sections = _exact_mapping(receipt["sections"], {"bottom", "top"}, name="sections")
    stations: dict[str, Decimal] = {}
    for name in ("bottom", "top"):
        section = _exact_mapping(sections[name], _SECTION_KEYS, name=f"sections.{name}")
        if list(section["normal_unit_decimal"]) != list(axis["unit_decimal"]):
            raise ArtifactMandrelError(f"sections.{name} does not lie across the axis")
        if section["radius_mm_decimal"] != receipt["radius_mm_decimal"]:
            raise ArtifactMandrelError(f"sections.{name} is not a section of this drum")
        station = _decimal_text(section["station_mm_decimal"], name=f"sections.{name}.station")
        centre = _decimal_vector(section["center_mm_decimal"], name=f"sections.{name}.center")
        expected = [point[i] + station * unit[i] for i in range(3)]
        if max(abs(float(centre[i] - expected[i])) for i in range(3)) > _SECTION_ON_AXIS_TOLERANCE_MM:
            raise ArtifactMandrelError(f"sections.{name} centre is not on the axis")
        stations[name] = station
    if stations["top"] <= stations["bottom"]:
        raise ArtifactMandrelError("mandrel sections are not ordered along the axis")

    quality = _exact_mapping(receipt["quality"], _QUALITY_KEYS, name="quality")
    _strict_int(quality["iterations"], name="quality.iterations", minimum=1,
                maximum=MAX_MANDREL_ITERATIONS)
    length = _decimal_text(quality["length_along_axis_mm_decimal"], name="length", signed=False)
    if length != stations["top"] - stations["bottom"]:
        raise ArtifactMandrelError("mandrel length does not match its sections")
    if float(length) < MIN_MANDREL_LENGTH_MM:
        raise ArtifactMandrelError("mandrel surface is too short along its axis")
    arc = _decimal_text(quality["arc_span_deg_decimal"], name="arc span", signed=False)
    if not (Decimal(str(MIN_MANDREL_ARC_SPAN_DEG)) <= arc <= Decimal(360)):
        raise ArtifactMandrelError("mandrel arc span is out of range")
    rms = _decimal_text(quality["radial_rms_residual_mm_decimal"], name="rms", signed=False)
    p95 = _decimal_text(quality["radial_p95_residual_mm_decimal"], name="p95", signed=False)
    maximum = _decimal_text(quality["radial_max_residual_mm_decimal"], name="max", signed=False)
    if rms > maximum or p95 > maximum:
        raise ArtifactMandrelError("mandrel residual summary is inconsistent")
    share = _decimal_text(
        quality["opposite_facing_area_share_decimal"], name="opposite share", signed=False
    )
    if share > Decimal(str(MAX_MANDREL_OPPOSITE_FACING_SHARE)):
        raise ArtifactMandrelError("mandrel surface holds both faces of the wall")
    _decimal_text(quality["fit_condition_decimal"], name="fit condition", signed=False)

    frozen = _frozen_mapping(receipt, name="mandrel receipt")
    thawed = _thaw_json(frozen)
    assert isinstance(thawed, dict)
    return thawed


def _qc_from_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    quality = receipt["quality"]
    return {
        "arc_span_deg": float(Decimal(quality["arc_span_deg_decimal"])),
        "length_along_axis_mm": float(Decimal(quality["length_along_axis_mm_decimal"])),
        "radial_rms_residual_mm": float(Decimal(quality["radial_rms_residual_mm_decimal"])),
        "radius_mm": float(Decimal(receipt["radius_mm_decimal"])),
        "surface_facing": str(receipt["surface_facing"]),
        "used_vertex_count": int(receipt["used_vertex_count"]),
    }


def _validate_qc_against_receipt(qc: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
    if _thaw_json(_freeze_json(dict(qc))) != _qc_from_receipt(receipt):
        raise ArtifactMandrelError("mandrel QC does not match its receipt")


def _validate_recipe_against_receipt(
    recipe: Mapping[str, Any], receipt: Mapping[str, Any]
) -> None:
    selection = recipe["selection"]
    if (
        selection["selection_sha256"] != receipt["selection_sha256"]
        or int(selection["selected_face_count"]) != int(receipt["selected_face_count"])
        or int(selection["total_face_count"]) != int(receipt["input_face_count"])
    ):
        raise ArtifactMandrelError("mandrel recipe and receipt name different surfaces")


# ---------------------------------------------------------------------------
# Computation, commit, and records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MandrelComputation:
    context: OperationContext
    projection_snapshot: ArtifactProjectionSnapshot
    receipt: Mapping[str, Any]
    recipe: Mapping[str, Any]
    qc: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.context, OperationContext):
            raise ArtifactMandrelError("context must be OperationContext")
        if not isinstance(self.projection_snapshot, ArtifactProjectionSnapshot):
            raise ArtifactMandrelError("projection_snapshot must be ArtifactProjectionSnapshot")
        recipe = validate_mandrel_recipe(self.recipe)
        receipt = validate_mandrel_receipt(self.receipt)
        _validate_recipe_against_receipt(recipe, receipt)
        if canonical_recipe_hash(recipe) != self.context.recipe_hash:
            raise ArtifactMandrelError("mandrel recipe does not match its OperationContext")
        if self.context.selection_hash != recipe["selection"]["selection_sha256"]:
            raise ArtifactMandrelError("mandrel selection hash does not match its surface")
        snapshot = self.projection_snapshot
        if (
            tuple(self.context.source_asset_ids) != (snapshot.source_asset_id,)
            or self.context.geometry_revision_id != snapshot.geometry_revision_id
            or self.context.source_metadata_revision_id != snapshot.source_metadata_revision_id
            or self.context.align_revision_id != snapshot.align_revision_id
        ):
            raise ArtifactMandrelError("projection snapshot does not match the mandrel context")
        qc = _frozen_mapping(self.qc, name="mandrel.qc")
        _validate_qc_against_receipt(qc, receipt)
        object.__setattr__(self, "recipe", _frozen_mapping(recipe, name="recipe"))
        object.__setattr__(self, "receipt", _frozen_mapping(receipt, name="receipt"))
        object.__setattr__(self, "qc", qc)

    @property
    def record_type(self) -> str:
        return MANDREL_RECORD_TYPE

    @property
    def geometry_ref(self) -> str:
        return MANDREL_REF_PREFIX + canonical_json_sha256(self.receipt_dict())

    def recipe_dict(self) -> dict[str, Any]:
        value = _thaw_json(self.recipe)
        assert isinstance(value, dict)
        return value

    def receipt_dict(self) -> dict[str, Any]:
        value = _thaw_json(self.receipt)
        assert isinstance(value, dict)
        return value

    def qc_dict(self) -> dict[str, Any]:
        value = _thaw_json(self.qc)
        assert isinstance(value, dict)
        return value


def mandrel_computation_matches_active_projection(
    session: ArtifactSession,
    computation: MandrelComputation,
) -> bool:
    if not isinstance(session, ArtifactSession) or not isinstance(computation, MandrelComputation):
        return False
    try:
        current = session.projection_snapshot()
    except ArtifactSessionError:
        return False
    return current.render_key == computation.projection_snapshot.render_key


def commit_mandrel_measurement(
    session: ArtifactSession,
    computation: MandrelComputation,
    *,
    record_id: str,
    created_at: str,
    operator: str,
    depends_on_record_ids: Sequence[str] = (),
) -> ArtifactSession:
    if not mandrel_computation_matches_active_projection(session, computation):
        raise ArtifactMandrelError("mandrel computation is stale for the active projection")
    receipt = computation.receipt_dict()
    qc = computation.qc_dict()
    _validate_qc_against_receipt(qc, receipt)
    receipt_bytes = canonical_json_bytes(receipt)
    if len(receipt_bytes) > MAX_MANDREL_RECEIPT_BYTES:
        raise ArtifactMandrelError("mandrel receipt exceeds its limit")
    receipt_sha256 = canonical_json_sha256(receipt)
    extensions = {
        MANDREL_EXTENSION_KEY: {
            "media_type": MANDREL_MEDIA_TYPE,
            "receipt": receipt,
            "receipt_byte_length": len(receipt_bytes),
            "receipt_sha256": receipt_sha256,
            "schema_version": MANDREL_RECEIPT_SCHEMA_VERSION,
        }
    }
    try:
        document = session.document.append_record_from_context(
            context=computation.context,
            id=record_id,
            type=MANDREL_RECORD_TYPE,
            geometry_ref=MANDREL_REF_PREFIX + receipt_sha256,
            recipe=computation.recipe_dict(),
            qc=qc,
            lifecycle_status=RecordLifecycleStatus.READY,
            created_at=created_at,
            operator=operator,
            depends_on_record_ids=depends_on_record_ids,
            extensions=extensions,
        )
    except ArtifactDocumentError as exc:
        raise ArtifactMandrelError(str(exc)) from exc
    return session.with_document(document)


def mandrel_receipt_from_record(record: DerivedRecord) -> dict[str, Any]:
    if not isinstance(record, DerivedRecord):
        raise ArtifactMandrelError("record must be a DerivedRecord")
    if record.type != MANDREL_RECORD_TYPE:
        raise ArtifactMandrelError("record is not a 와통 measurement")
    descriptor = _exact_mapping(
        record.extensions.get(MANDREL_EXTENSION_KEY),
        {"media_type", "receipt", "receipt_byte_length", "receipt_sha256", "schema_version"},
        name="mandrel descriptor",
    )
    if (
        descriptor["media_type"] != MANDREL_MEDIA_TYPE
        or descriptor["schema_version"] != MANDREL_RECEIPT_SCHEMA_VERSION
    ):
        raise ArtifactMandrelError("mandrel descriptor is invalid")
    declared_length = _strict_int(
        descriptor["receipt_byte_length"],
        name="mandrel receipt_byte_length",
        minimum=2,
        maximum=MAX_MANDREL_RECEIPT_BYTES,
    )
    receipt = validate_mandrel_receipt(_thaw_json(descriptor["receipt"]))
    receipt_bytes = canonical_json_bytes(receipt)
    if len(receipt_bytes) != declared_length:
        raise ArtifactMandrelError("mandrel receipt length is invalid")
    receipt_sha256 = canonical_json_sha256(receipt)
    if descriptor["receipt_sha256"] != receipt_sha256:
        raise ArtifactMandrelError("mandrel receipt hash is invalid")
    if record.geometry_ref != MANDREL_REF_PREFIX + receipt_sha256:
        raise ArtifactMandrelError("mandrel geometry_ref is invalid")
    recipe = validate_mandrel_recipe(_thaw_json(record.recipe))
    _validate_recipe_against_receipt(recipe, receipt)
    if record.selection_hash != recipe["selection"]["selection_sha256"]:
        raise ArtifactMandrelError("mandrel selection hash is invalid")
    _validate_qc_against_receipt(record.qc, receipt)
    return receipt


def validate_mandrel_records(document: ArtifactDocument) -> None:
    if not isinstance(document, ArtifactDocument):
        raise ArtifactMandrelError("document must be an ArtifactDocument")
    for record in document.records:
        if record.type == MANDREL_RECORD_TYPE:
            mandrel_receipt_from_record(record)


def verify_mandrel_record_against_mesh(
    record: DerivedRecord,
    vertices: object,
    faces: object,
) -> dict[str, Any]:
    """Recheck the claim: the stored drum fits the stored surface as stated.

    ``vertices`` must be the canonical projection under the record's own Align
    revision - the frame it was measured in.  The residual is recomputed from
    the stored axis and radius, not refitted, and summed exactly, so the check
    does not depend on which linear-algebra build ran the original fit.
    """

    receipt = mandrel_receipt_from_record(record)
    recipe = validate_mandrel_recipe(_thaw_json(record.recipe))
    vertex_array = np.asarray(vertices, dtype=np.float64)
    face_array = np.asarray(faces, dtype=np.int64)
    if int(recipe["selection"]["total_face_count"]) != int(face_array.shape[0]):
        raise ArtifactMandrelError("mandrel record does not match this mesh")
    used = np.unique(face_array[selection_face_indices(recipe["selection"])].reshape(-1))
    if int(len(used)) != int(receipt["used_vertex_count"]):
        raise ArtifactMandrelError("mandrel record used a different set of vertices")
    point = np.array([float(item) for item in receipt["axis"]["point_mm_decimal"]])
    unit = np.array([float(item) for item in receipt["axis"]["unit_decimal"]])
    unit = unit / float(np.linalg.norm(unit))
    radius = float(receipt["radius_mm_decimal"])
    residuals = _radial_residuals(vertex_array[used], point, unit, radius)
    rms = math.sqrt(math.fsum(float(value) * float(value) for value in residuals) / len(residuals))
    stored = float(receipt["quality"]["radial_rms_residual_mm_decimal"])
    # The stored parameters are rounded to a micrometre and a nanoradian, which
    # moves each residual by at most about that much.
    if abs(rms - stored) > 0.002:
        raise ArtifactMandrelError(
            f"the stored drum fits this surface at rms {rms:.6f} mm, not the "
            f"{stored:.6f} mm its receipt claims"
        )
    return receipt


__all__ = [
    "ArtifactMandrelError",
    "FACINGS",
    "FACING_AWAY_FROM_AXIS",
    "FACING_TOWARD_AXIS",
    "MANDREL_ALGORITHM",
    "MANDREL_EXTENSION_KEY",
    "MANDREL_RECIPE_KIND",
    "MANDREL_RECORD_TYPE",
    "MIN_MANDREL_ARC_SPAN_DEG",
    "MIN_MANDREL_FACES",
    "MandrelComputation",
    "commit_mandrel_measurement",
    "extract_mandrel",
    "mandrel_computation_matches_active_projection",
    "mandrel_receipt_from_record",
    "mandrel_recipe",
    "mandrel_selection_hash",
    "validate_mandrel_receipt",
    "validate_mandrel_recipe",
    "validate_mandrel_records",
    "verify_mandrel_record_against_mesh",
]
