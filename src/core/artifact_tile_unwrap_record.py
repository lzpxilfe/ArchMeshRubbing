"""Immutable, content-addressed roof-tile unwrap receipts."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

from .artifact_document import (
    ArtifactDocument,
    ArtifactDocumentError,
    DerivedRecord,
    OperationContext,
    RecordLifecycleStatus,
)
from .artifact_tile_unwrap_extractor import (
    MAX_TILE_UNWRAP_COORDINATE_UM,
    MAX_TILE_UNWRAP_FACES,
    MAX_TILE_UNWRAP_QC_FACES,
    MAX_TILE_UNWRAP_VERTICES,
    SECTION_CENTER_CANONICAL_AXIS,
    SECTION_CENTER_FIT_PER_SECTION,
    TILE_UNWRAP_COORDINATE_QUANTUM_UM,
    TILE_UNWRAP_COORDINATE_SPACE,
    TILE_UNWRAP_GEOMETRY_REF_PREFIX,
    TILE_UNWRAP_HASH_SCOPE,
    TILE_UNWRAP_OUTPUT_SCHEMA_VERSION,
    ArtifactTileUnwrapError,
    TileUnwrapMesh,
    validate_tile_unwrap_recipe,
)
from .canonical_json import canonical_json_bytes, canonical_json_sha256


TILE_UNWRAP_RECORD_TYPE = "surface.tile_unwrap.v1"
TILE_UNWRAP_RECEIPT_EXTENSION_KEY = "org.archmeshrubbing:tile-unwrap-v1"
TILE_UNWRAP_RECEIPT_MEDIA_TYPE = (
    "application/vnd.archmeshrubbing.tile-unwrap-receipt+json"
)
MAX_TILE_UNWRAP_RECEIPT_BYTES = 64 * 1024

#: Distortion a fitted-centre record may not exceed.  A record unrolled about
#: the measured axis reports all three and is gated on none of them: there
#: they measure relief on the wall, not a centre in the wrong place.
TILE_UNWRAP_DISTORTION_FACE_MAX_MILLIONTHS = 250_000
TILE_UNWRAP_DISTORTION_MEAN_MAX_MILLIONTHS = 75_000
TILE_UNWRAP_DISTORTION_P95_MAX_MILLIONTHS = 150_000

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_QC_FIELDS = {
    "boundary_loop_count",
    "connected_component_count",
    "degenerate_uv_face_count",
    "distortion_max_millionths",
    "distortion_mean_millionths",
    "distortion_median_millionths",
    "distortion_p95_millionths",
    "face_count",
    "foldover_face_count",
    "height_um",
    "negative_orientation_face_count",
    "duplicate_face_count",
    "inconsistent_oriented_edge_count",
    "nonmanifold_edge_count",
    "positive_orientation_face_count",
    "section_centerline_length_um",
    "section_count",
    "section_fit_valid_count",
    "section_mean_radius_um",
    "section_mean_span_microdegrees",
    "section_row_shift_applied",
    "section_row_shift_max_um",
    "section_row_shift_station_count",
    "selected_face_count",
    "selection_sha256",
    "unwrap_sha256",
    "uv_overlap_pair_count",
    "vertex_count",
    "width_um",
}


class ArtifactTileUnwrapRecordError(ValueError):
    """A durable tile-unwrapping record violates its public contract."""


def _exact_keys(
    value: object,
    expected: set[str],
    *,
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactTileUnwrapRecordError(f"{name} must be an object")
    observed = set(value)
    missing = sorted(expected - observed)
    unknown = sorted(observed - expected)
    if missing:
        raise ArtifactTileUnwrapRecordError(
            f"{name} is missing fields: {', '.join(missing)}"
        )
    if unknown:
        raise ArtifactTileUnwrapRecordError(
            f"{name} has unknown fields: {', '.join(unknown)}"
        )
    return value


def _strict_int(
    value: object,
    *,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArtifactTileUnwrapRecordError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ArtifactTileUnwrapRecordError(
            f"{name} must be in the inclusive range {minimum}..{maximum}"
        )
    return value


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ArtifactTileUnwrapRecordError(f"{name} must be a lowercase SHA-256")
    return value


def _dimension(
    value: object,
    *,
    name: str,
    expected_um: int,
) -> dict[str, int]:
    rational = _exact_keys(
        value,
        {"denominator", "numerator"},
        name=name,
    )
    denominator = _strict_int(
        rational["denominator"],
        name=f"{name}.denominator",
        minimum=1000,
        maximum=1000,
    )
    numerator = _strict_int(
        rational["numerator"],
        name=f"{name}.numerator",
        minimum=1,
        maximum=MAX_TILE_UNWRAP_COORDINATE_UM,
    )
    if numerator != expected_um:
        raise ArtifactTileUnwrapRecordError(f"{name} does not match bounds_um")
    return {"denominator": denominator, "numerator": numerator}


def validate_tile_unwrap_receipt(value: object) -> dict[str, Any]:
    receipt = _exact_keys(
        value,
        {
            "axis",
            "bounds_um",
            "component_sha256",
            "coordinate_quantum_um",
            "coordinate_space",
            "face_count",
            "hash_scope",
            "height_mm_exact",
            "record_view",
            "schema_version",
            "selection_sha256",
            "source_face_count",
            "source_vertex_count",
            "unwrap_sha256",
            "vertex_count",
            "width_mm_exact",
        },
        name="tile unwrap receipt",
    )
    literal_fields = {
        "coordinate_quantum_um": TILE_UNWRAP_COORDINATE_QUANTUM_UM,
        "coordinate_space": TILE_UNWRAP_COORDINATE_SPACE,
        "hash_scope": TILE_UNWRAP_HASH_SCOPE,
        "schema_version": TILE_UNWRAP_OUTPUT_SCHEMA_VERSION,
    }
    for key, expected in literal_fields.items():
        if receipt[key] != expected:
            raise ArtifactTileUnwrapRecordError(
                f"tile unwrap receipt field {key!r} is invalid"
            )
    axis = receipt["axis"]
    if axis not in {"x", "y", "z"}:
        raise ArtifactTileUnwrapRecordError("tile unwrap receipt axis is invalid")
    record_view = receipt["record_view"]
    if record_view not in {"top", "bottom"}:
        raise ArtifactTileUnwrapRecordError(
            "tile unwrap receipt record_view is invalid"
        )
    vertex_count = _strict_int(
        receipt["vertex_count"],
        name="vertex_count",
        minimum=3,
        maximum=MAX_TILE_UNWRAP_VERTICES,
    )
    face_count = _strict_int(
        receipt["face_count"],
        name="face_count",
        minimum=1,
        maximum=MAX_TILE_UNWRAP_QC_FACES,
    )
    source_vertex_count = _strict_int(
        receipt["source_vertex_count"],
        name="source_vertex_count",
        minimum=3,
        maximum=MAX_TILE_UNWRAP_VERTICES,
    )
    source_face_count = _strict_int(
        receipt["source_face_count"],
        name="source_face_count",
        minimum=1,
        maximum=MAX_TILE_UNWRAP_QC_FACES,
    )
    if source_vertex_count != vertex_count or source_face_count != face_count:
        raise ArtifactTileUnwrapRecordError(
            "tile unwrap receipt correspondence counts are inconsistent"
        )
    bounds = _exact_keys(
        receipt["bounds_um"],
        {"maximum_u", "maximum_v", "minimum_u", "minimum_v"},
        name="bounds_um",
    )
    minimum_u = _strict_int(
        bounds["minimum_u"], name="bounds_um.minimum_u", minimum=0, maximum=0
    )
    minimum_v = _strict_int(
        bounds["minimum_v"], name="bounds_um.minimum_v", minimum=0, maximum=0
    )
    maximum_u = _strict_int(
        bounds["maximum_u"],
        name="bounds_um.maximum_u",
        minimum=1,
        maximum=MAX_TILE_UNWRAP_COORDINATE_UM,
    )
    maximum_v = _strict_int(
        bounds["maximum_v"],
        name="bounds_um.maximum_v",
        minimum=1,
        maximum=MAX_TILE_UNWRAP_COORDINATE_UM,
    )
    component_hashes = _exact_keys(
        receipt["component_sha256"],
        {
            "faces_i32le",
            "source_face_indices_i64le",
            "source_vertex_indices_i64le",
            "uv_um_i64le",
        },
        name="component_sha256",
    )
    canonical_components = {
        key: _sha256(component_hashes[key], name=f"component_sha256.{key}")
        for key in sorted(component_hashes)
    }
    selection_sha = _sha256(receipt["selection_sha256"], name="selection_sha256")
    unwrap_sha = _sha256(receipt["unwrap_sha256"], name="unwrap_sha256")
    width = maximum_u - minimum_u
    height = maximum_v - minimum_v
    width_exact = _dimension(
        receipt["width_mm_exact"], name="width_mm_exact", expected_um=width
    )
    height_exact = _dimension(
        receipt["height_mm_exact"], name="height_mm_exact", expected_um=height
    )
    return {
        **literal_fields,
        "axis": axis,
        "bounds_um": {
            "maximum_u": maximum_u,
            "maximum_v": maximum_v,
            "minimum_u": minimum_u,
            "minimum_v": minimum_v,
        },
        "component_sha256": canonical_components,
        "face_count": face_count,
        "height_mm_exact": height_exact,
        "record_view": record_view,
        "selection_sha256": selection_sha,
        "source_face_count": source_face_count,
        "source_vertex_count": source_vertex_count,
        "unwrap_sha256": unwrap_sha,
        "vertex_count": vertex_count,
        "width_mm_exact": width_exact,
    }


def _validate_qc_against_receipt(
    qc: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    section_center_policy: str = SECTION_CENTER_FIT_PER_SECTION,
) -> dict[str, Any]:
    if section_center_policy not in {
        SECTION_CENTER_FIT_PER_SECTION,
        SECTION_CENTER_CANONICAL_AXIS,
    }:
        raise ArtifactTileUnwrapRecordError(
            "tile unwrap QC gate needs a known section centre policy"
        )
    value = _exact_keys(qc, _QC_FIELDS, name="tile unwrap QC")
    expected = {
        "face_count": receipt["face_count"],
        "height_um": receipt["height_mm_exact"]["numerator"],
        "selected_face_count": receipt["source_face_count"],
        "selection_sha256": receipt["selection_sha256"],
        "unwrap_sha256": receipt["unwrap_sha256"],
        "vertex_count": receipt["vertex_count"],
        "width_um": receipt["width_mm_exact"]["numerator"],
    }
    for key, expected_value in expected.items():
        if value[key] != expected_value:
            raise ArtifactTileUnwrapRecordError(
                f"tile unwrap QC field {key!r} does not match its receipt"
            )
    face_count = int(receipt["face_count"])
    degenerate = _strict_int(
        value["degenerate_uv_face_count"],
        name="qc.degenerate_uv_face_count",
        minimum=0,
        maximum=face_count,
    )
    foldovers = _strict_int(
        value["foldover_face_count"],
        name="qc.foldover_face_count",
        minimum=0,
        maximum=face_count,
    )
    positive = _strict_int(
        value["positive_orientation_face_count"],
        name="qc.positive_orientation_face_count",
        minimum=0,
        maximum=face_count,
    )
    negative = _strict_int(
        value["negative_orientation_face_count"],
        name="qc.negative_orientation_face_count",
        minimum=0,
        maximum=face_count,
    )
    if degenerate + positive + negative != face_count:
        raise ArtifactTileUnwrapRecordError(
            "tile unwrap orientation QC is inconsistent"
        )
    if degenerate != 0 or foldovers != 0 or min(positive, negative) != 0:
        raise ArtifactTileUnwrapRecordError(
            "authoritative tile unwrap cannot contain collapsed or folded faces"
        )
    component_count = _strict_int(
        value["connected_component_count"],
        name="qc.connected_component_count",
        minimum=1,
        maximum=face_count,
    )
    _strict_int(
        value["boundary_loop_count"],
        name="qc.boundary_loop_count",
        minimum=1,
        maximum=face_count,
    )
    topology_zero_fields = (
        "duplicate_face_count",
        "inconsistent_oriented_edge_count",
        "nonmanifold_edge_count",
        "uv_overlap_pair_count",
    )
    topology_zero = {
        key: _strict_int(
            value[key],
            name=f"qc.{key}",
            minimum=0,
            maximum=face_count,
        )
        for key in topology_zero_fields
    }
    if component_count != 1 or any(topology_zero.values()):
        raise ArtifactTileUnwrapRecordError(
            "authoritative tile unwrap topology/overlap QC did not pass"
        )
    distortion_fields = (
        "distortion_max_millionths",
        "distortion_mean_millionths",
        "distortion_median_millionths",
        "distortion_p95_millionths",
    )
    distortion = {
        key: _strict_int(value[key], name=f"qc.{key}", minimum=0, maximum=1_000_000)
        for key in distortion_fields
    }
    # A fitted centre is estimated, so the three distortion numbers report
    # how well it was estimated and all three gate.  Unrolled about the
    # measured axis nothing is estimated - the arc and the station both come
    # from the axis - so what the numbers measure is the surface's own
    # relief: a corded tile's back has more area than the cylinder it
    # develops onto, and the finer the mesh the more of it is resolved.  All
    # three are reported and none of them gates there; the topology, the
    # orientation, the overlap and the section rules are untouched.
    axis_centred = section_center_policy == SECTION_CENTER_CANONICAL_AXIS
    if (
        not axis_centred
        and distortion["distortion_max_millionths"]
        > TILE_UNWRAP_DISTORTION_FACE_MAX_MILLIONTHS
    ):
        raise ArtifactTileUnwrapRecordError("tile unwrap max distortion exceeds gate")
    if (
        not axis_centred
        and distortion["distortion_mean_millionths"]
        > TILE_UNWRAP_DISTORTION_MEAN_MAX_MILLIONTHS
    ):
        raise ArtifactTileUnwrapRecordError("tile unwrap mean distortion exceeds gate")
    if (
        not axis_centred
        and distortion["distortion_p95_millionths"]
        > TILE_UNWRAP_DISTORTION_P95_MAX_MILLIONTHS
    ):
        raise ArtifactTileUnwrapRecordError("tile unwrap p95 distortion exceeds gate")
    if (
        distortion["distortion_median_millionths"]
        > distortion["distortion_max_millionths"]
        or distortion["distortion_p95_millionths"]
        > distortion["distortion_max_millionths"]
        or distortion["distortion_mean_millionths"]
        > distortion["distortion_max_millionths"]
        or distortion["distortion_median_millionths"]
        > distortion["distortion_p95_millionths"]
    ):
        raise ArtifactTileUnwrapRecordError("tile unwrap distortion QC is inconsistent")
    section_count = _strict_int(
        value["section_count"], name="qc.section_count", minimum=12, maximum=96
    )
    valid_count = _strict_int(
        value["section_fit_valid_count"],
        name="qc.section_fit_valid_count",
        minimum=4,
        maximum=section_count,
    )
    if valid_count < max(4, int(section_count * 0.35)):
        raise ArtifactTileUnwrapRecordError("tile unwrap section fit is too sparse")
    _strict_int(
        value["section_centerline_length_um"],
        name="qc.section_centerline_length_um",
        minimum=1,
        maximum=MAX_TILE_UNWRAP_COORDINATE_UM,
    )
    _strict_int(
        value["section_mean_radius_um"],
        name="qc.section_mean_radius_um",
        minimum=1,
        maximum=MAX_TILE_UNWRAP_COORDINATE_UM,
    )
    _strict_int(
        value["section_mean_span_microdegrees"],
        name="qc.section_mean_span_microdegrees",
        minimum=20_000_000,
        maximum=360_000_000,
    )
    row_shift_applied = value["section_row_shift_applied"]
    if not isinstance(row_shift_applied, bool):
        raise ArtifactTileUnwrapRecordError(
            "qc.section_row_shift_applied must be a boolean"
        )
    row_shift_max_um = _strict_int(
        value["section_row_shift_max_um"],
        name="qc.section_row_shift_max_um",
        minimum=0,
        maximum=MAX_TILE_UNWRAP_COORDINATE_UM,
    )
    row_shift_station_count = _strict_int(
        value["section_row_shift_station_count"],
        name="qc.section_row_shift_station_count",
        minimum=0,
        maximum=section_count,
    )
    if row_shift_applied:
        if row_shift_max_um < 1 or row_shift_station_count != section_count:
            raise ArtifactTileUnwrapRecordError(
                "tile unwrap applied row-shift QC is inconsistent"
            )
    elif row_shift_max_um != 0 or row_shift_station_count != 0:
        raise ArtifactTileUnwrapRecordError(
            "tile unwrap inactive row-shift QC is inconsistent"
        )
    return dict(value)


def validate_tile_unwrap_qc(
    value: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    section_center_policy: str = SECTION_CENTER_FIT_PER_SECTION,
) -> dict[str, Any]:
    """Validate the closed QC shape against a validated public receipt.

    The distortion gate depends on where the section centres came from, so
    the recipe's ``section_center_policy`` is passed alongside.
    """

    validated_receipt = validate_tile_unwrap_receipt(receipt)
    return _validate_qc_against_receipt(
        value, validated_receipt, section_center_policy=section_center_policy
    )


def append_tile_unwrap_record_from_context(
    document: ArtifactDocument,
    *,
    context: OperationContext,
    unwrap: TileUnwrapMesh,
    recipe: Mapping[str, Any],
    record_id: str,
    created_at: str,
    operator: str,
    depends_on_record_ids: Sequence[str] = (),
    qc: Mapping[str, Any],
) -> ArtifactDocument:
    if not isinstance(document, ArtifactDocument):
        raise ArtifactTileUnwrapRecordError("document must be an ArtifactDocument")
    if not isinstance(context, OperationContext):
        raise ArtifactTileUnwrapRecordError("context must be an OperationContext")
    if not isinstance(unwrap, TileUnwrapMesh):
        raise ArtifactTileUnwrapRecordError("unwrap must be a TileUnwrapMesh")
    try:
        validated_recipe = validate_tile_unwrap_recipe(recipe)
    except ArtifactTileUnwrapError as exc:
        raise ArtifactTileUnwrapRecordError(str(exc)) from exc
    selection = validated_recipe["selection"]
    assert isinstance(selection, Mapping)
    selection_sha = str(selection["selection_sha256"])
    if context.selection_hash != selection_sha:
        raise ArtifactTileUnwrapRecordError(
            "tile unwrap context selection does not match recipe"
        )
    receipt = validate_tile_unwrap_receipt(
        unwrap.receipt(selection_sha256=selection_sha)
    )
    if receipt["axis"] != validated_recipe["longitudinal_axis"]:
        raise ArtifactTileUnwrapRecordError(
            "tile unwrap receipt axis differs from recipe"
        )
    if receipt["record_view"] != validated_recipe["record_view"]:
        raise ArtifactTileUnwrapRecordError(
            "tile unwrap receipt record_view differs from recipe"
        )
    if receipt["source_face_count"] != selection["selected_face_count"]:
        raise ArtifactTileUnwrapRecordError(
            "tile unwrap receipt selection count differs from recipe"
        )
    validated_qc = _validate_qc_against_receipt(
        qc,
        receipt,
        section_center_policy=str(
            validated_recipe.get("section_center_policy", SECTION_CENTER_FIT_PER_SECTION)
        ),
    )
    if validated_qc["section_count"] != validated_recipe["n_sections"]:
        raise ArtifactTileUnwrapRecordError(
            "tile unwrap section QC differs from recipe"
        )
    receipt_bytes = canonical_json_bytes(receipt)
    if len(receipt_bytes) > MAX_TILE_UNWRAP_RECEIPT_BYTES:
        raise ArtifactTileUnwrapRecordError("tile unwrap receipt exceeds size limit")
    extensions = {
        TILE_UNWRAP_RECEIPT_EXTENSION_KEY: {
            "media_type": TILE_UNWRAP_RECEIPT_MEDIA_TYPE,
            "receipt": receipt,
            "receipt_byte_length": len(receipt_bytes),
            "receipt_sha256": canonical_json_sha256(receipt),
            "schema_version": TILE_UNWRAP_OUTPUT_SCHEMA_VERSION,
        }
    }
    try:
        return document.append_record_from_context(
            context=context,
            id=record_id,
            type=TILE_UNWRAP_RECORD_TYPE,
            geometry_ref=f"{TILE_UNWRAP_GEOMETRY_REF_PREFIX}{receipt['unwrap_sha256']}",
            recipe=validated_recipe,
            qc=validated_qc,
            lifecycle_status=RecordLifecycleStatus.READY,
            created_at=created_at,
            operator=operator,
            depends_on_record_ids=depends_on_record_ids,
            extensions=extensions,
        )
    except ArtifactDocumentError as exc:
        raise ArtifactTileUnwrapRecordError(str(exc)) from exc


def tile_unwrap_receipt_from_record(record: DerivedRecord) -> dict[str, Any]:
    if not isinstance(record, DerivedRecord):
        raise ArtifactTileUnwrapRecordError("record must be a DerivedRecord")
    if record.type != TILE_UNWRAP_RECORD_TYPE:
        raise ArtifactTileUnwrapRecordError("record is not a tile unwrap record")
    descriptor = _exact_keys(
        record.extensions.get(TILE_UNWRAP_RECEIPT_EXTENSION_KEY),
        {
            "media_type",
            "receipt",
            "receipt_byte_length",
            "receipt_sha256",
            "schema_version",
        },
        name="tile unwrap descriptor",
    )
    if descriptor["media_type"] != TILE_UNWRAP_RECEIPT_MEDIA_TYPE:
        raise ArtifactTileUnwrapRecordError("tile unwrap media type is invalid")
    if descriptor["schema_version"] != TILE_UNWRAP_OUTPUT_SCHEMA_VERSION:
        raise ArtifactTileUnwrapRecordError("tile unwrap descriptor schema is invalid")
    receipt = validate_tile_unwrap_receipt(descriptor["receipt"])
    receipt_bytes = canonical_json_bytes(receipt)
    if descriptor["receipt_byte_length"] != len(receipt_bytes):
        raise ArtifactTileUnwrapRecordError(
            "tile unwrap receipt byte length is invalid"
        )
    if descriptor["receipt_sha256"] != canonical_json_sha256(receipt):
        raise ArtifactTileUnwrapRecordError("tile unwrap receipt SHA-256 is invalid")
    if record.geometry_ref != (
        f"{TILE_UNWRAP_GEOMETRY_REF_PREFIX}{receipt['unwrap_sha256']}"
    ):
        raise ArtifactTileUnwrapRecordError(
            "tile unwrap geometry_ref does not match receipt"
        )
    try:
        recipe = validate_tile_unwrap_recipe(record.recipe)
    except ArtifactTileUnwrapError as exc:
        raise ArtifactTileUnwrapRecordError(str(exc)) from exc
    selection = recipe["selection"]
    assert isinstance(selection, Mapping)
    if receipt["axis"] != recipe["longitudinal_axis"]:
        raise ArtifactTileUnwrapRecordError("tile unwrap record axis mismatch")
    if receipt["record_view"] != recipe["record_view"]:
        raise ArtifactTileUnwrapRecordError("tile unwrap record view mismatch")
    if receipt["selection_sha256"] != selection["selection_sha256"]:
        raise ArtifactTileUnwrapRecordError("tile unwrap selection digest mismatch")
    if receipt["source_face_count"] != selection["selected_face_count"]:
        raise ArtifactTileUnwrapRecordError("tile unwrap selection count mismatch")
    if record.selection_hash != selection["selection_sha256"]:
        raise ArtifactTileUnwrapRecordError(
            "tile unwrap record selection_hash mismatch"
        )
    record_qc = record.to_dict()["qc"]
    assert isinstance(record_qc, dict)
    validated_qc = _validate_qc_against_receipt(
        record_qc,
        receipt,
        section_center_policy=str(
            recipe.get("section_center_policy", SECTION_CENTER_FIT_PER_SECTION)
        ),
    )
    if validated_qc["section_count"] != recipe["n_sections"]:
        raise ArtifactTileUnwrapRecordError(
            "tile unwrap section QC differs from recipe"
        )
    return receipt


@dataclass(frozen=True, slots=True)
class DevelopedCylinder:
    """The cylinder a developed wall was measured to lie on.

    A 기와 was formed on a 와통, and the wall that touched it - the 내면 of a
    암키와, carrying the 포목흔 - still has the mould's radius.  Unrolling that
    wall fits a circle to every section, so the radius is already measured
    and this only reads it back.

    It is the radius of *the wall that was developed*, and nothing else: the
    same reading taken from a tile's 등면 gives the outer form, one wall
    thickness larger, which is not the 와통.  Which wall was developed is the
    drafter's to know from the selection they made, so the number is offered
    for the title block rather than written onto the drawing.
    """

    #: Radius of the fitted section circles, and twice it.
    radius_um: int
    diameter_um: int
    #: How the reading was made, so a reader can weigh it: a diameter fitted
    #: from twelve sections of a small fragment is not the one fitted from
    #: ninety-six of a whole tile.
    section_count: int
    section_fit_valid_count: int
    section_mean_span_microdegrees: int
    developed_length_um: int

    @property
    def radius_mm(self) -> float:
        return self.radius_um / 1000.0

    @property
    def diameter_mm(self) -> float:
        return self.diameter_um / 1000.0


def developed_cylinder_from_record(record: DerivedRecord) -> DevelopedCylinder:
    """Read back the cylinder one tile unwrap record was developed from.

    The record is validated first, so the reading cannot come from a receipt
    that does not hold.
    """

    tile_unwrap_receipt_from_record(record)
    qc = record.to_dict()["qc"]
    assert isinstance(qc, Mapping)
    radius_um = _strict_int(
        qc["section_mean_radius_um"],
        name="qc.section_mean_radius_um",
        minimum=1,
        maximum=MAX_TILE_UNWRAP_COORDINATE_UM,
    )
    return DevelopedCylinder(
        radius_um=radius_um,
        diameter_um=2 * radius_um,
        section_count=int(qc["section_count"]),
        section_fit_valid_count=int(qc["section_fit_valid_count"]),
        section_mean_span_microdegrees=int(qc["section_mean_span_microdegrees"]),
        developed_length_um=int(qc["section_centerline_length_um"]),
    )


def validate_tile_unwrap_records(document: ArtifactDocument) -> None:
    if not isinstance(document, ArtifactDocument):
        raise ArtifactTileUnwrapRecordError("document must be an ArtifactDocument")
    for record in document.records:
        if record.type == TILE_UNWRAP_RECORD_TYPE:
            receipt = tile_unwrap_receipt_from_record(record)
            recipe = validate_tile_unwrap_recipe(record.recipe)
            selection = recipe["selection"]
            assert isinstance(selection, Mapping)
            geometry = document.geometry_revision_index[record.geometry_revision_id]
            geometry_qc = geometry.qc
            face_count = _strict_int(
                geometry_qc.get("face_count"),
                name="geometry.qc.face_count",
                minimum=1,
                maximum=MAX_TILE_UNWRAP_FACES,
            )
            vertex_count = _strict_int(
                geometry_qc.get("vertex_count"),
                name="geometry.qc.vertex_count",
                minimum=3,
                maximum=MAX_TILE_UNWRAP_VERTICES,
            )
            if selection["total_face_count"] != face_count:
                raise ArtifactTileUnwrapRecordError(
                    "tile unwrap selection does not match source geometry face count"
                )
            if receipt["source_vertex_count"] > vertex_count:
                raise ArtifactTileUnwrapRecordError(
                    "tile unwrap correspondence exceeds source geometry vertex count"
                )


__all__ = [
    "ArtifactTileUnwrapRecordError",
    "DevelopedCylinder",
    "MAX_TILE_UNWRAP_RECEIPT_BYTES",
    "TILE_UNWRAP_RECEIPT_EXTENSION_KEY",
    "TILE_UNWRAP_RECEIPT_MEDIA_TYPE",
    "TILE_UNWRAP_RECORD_TYPE",
    "append_tile_unwrap_record_from_context",
    "developed_cylinder_from_record",
    "tile_unwrap_receipt_from_record",
    "validate_tile_unwrap_qc",
    "validate_tile_unwrap_receipt",
    "validate_tile_unwrap_records",
]
