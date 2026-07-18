from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np
import pytest

from src.core.artifact_document import ArtifactDocument
from src.core.artifact_record_validation import (
    ArtifactKnownRecordError,
    validate_known_records,
)
from src.core.artifact_session import ArtifactSession
from src.core.artifact_surface_measurement import (
    ArtifactSurfaceMeasurementComputation,
    ArtifactSurfaceMeasurementError,
    SURFACE_DIAMETER_RECORD_TYPE,
    SURFACE_DISTANCE_RECORD_TYPE,
    commit_artifact_surface_measurement,
    extract_surface_measurement,
    extract_surface_measurement_from_source,
    resolve_surface_anchor_from_ray,
    surface_diameter_recipe,
    surface_distance_recipe,
    surface_measurement_receipt_from_record,
    surface_measurement_selection_hash,
    validate_surface_measurement_receipt,
)
from src.core.mesh_import_recipe import current_mesh_import_recipe
from src.core.mesh_loader import MeshData
from src.core.source_identity import SourceFingerprint


ROOT = Path(__file__).resolve().parents[1]
STAMP = "2026-07-18T00:00:00Z"


def _triangle(*, offset_x: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(
            [
                [-2.0 + offset_x, -2.0, 0.0],
                [2.0 + offset_x, -2.0, 0.0],
                [0.0 + offset_x, 3.0, 0.0],
            ],
            dtype=np.float64,
        ),
        np.asarray([[0, 1, 2]], dtype=np.int32),
    )


def _anchor(
    vertices: np.ndarray,
    faces: np.ndarray,
    point: tuple[float, float, float],
    *,
    source_faces: np.ndarray | None = None,
) -> dict[str, Any]:
    depth = np.asarray(point, dtype=np.float64)
    return resolve_surface_anchor_from_ray(
        vertices,
        faces,
        source_faces=faces if source_faces is None else source_faces,
        ray_origin_world_mm=depth + np.asarray([0.0, 0.0, 5.0]),
        ray_direction_world=[0.0, 0.0, -1.0],
        depth_point_world_mm=depth,
        pixel_footprint_um=10,
    )


def _session(vertices: np.ndarray, faces: np.ndarray) -> ArtifactSession:
    mesh = MeshData(
        vertices=vertices,
        faces=faces,
        unit="mm",
        filepath=Path("/private/surface/scan.ply"),
        source_identity=SourceFingerprint(
            sha256="7" * 64,
            size_bytes=1024,
            mtime_ns=1,
            original_name="surface.ply",
            format="ply",
        ),
        source_format="ply",
        source_import_recipe=current_mesh_import_recipe("ply"),
    )
    return ArtifactSession.create_from_source(
        mesh,
        resolved_source_path="/private/surface/scan.ply",
        unit="mm",
        axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
        handedness="right",
        software_version="surface-test",
        operator="tester",
        created_at=STAMP,
        document_id="artifact:surface-measurement",
        metadata_revision_id="metadata:surface-measurement",
        align_revision_id="align:surface-measurement",
    )


def _distance_recipe(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> dict[str, Any]:
    anchors = [
        _anchor(vertices, faces, (-0.5 + float(vertices[0, 0] + 2.0), 0.0, 0.0)),
        _anchor(vertices, faces, (0.5 + float(vertices[0, 0] + 2.0), 0.0, 0.0)),
    ]
    return surface_distance_recipe(
        anchors,
        source_vertex_count=int(vertices.shape[0]),
        source_face_count=int(faces.shape[0]),
    )


def test_global_ray_pick_returns_source_order_barycentric_anchor() -> None:
    vertices, source_faces = _triangle()
    projected_faces = source_faces[:, [0, 2, 1]]
    anchor = _anchor(
        vertices,
        projected_faces,
        (0.5, 0.0, 0.0),
        source_faces=source_faces,
    )

    assert anchor["face_index"] == 0
    assert anchor["face_vertex_indices"] == [0, 1, 2]
    assert sum(anchor["barycentric_numerators"]) == 1_000_000_000
    weights = np.asarray(anchor["barycentric_numerators"], dtype=np.float64) / 1e9
    np.testing.assert_allclose(weights @ vertices[source_faces[0]], [0.5, 0.0, 0.0], atol=1e-9)
    assert anchor["depth_search_offset_px"] == [0, 0]

    with pytest.raises(ArtifactSurfaceMeasurementError, match="source_faces is required"):
        resolve_surface_anchor_from_ray(
            vertices,
            projected_faces,
            source_faces=None,  # type: ignore[arg-type]
            ray_origin_world_mm=(0.5, 0.0, 5.0),
            ray_direction_world=(0.0, 0.0, -1.0),
            depth_point_world_mm=(0.5, 0.0, 0.0),
            pixel_footprint_um=10,
        )


def test_distance_is_exact_one_millimetre_and_offset_stable() -> None:
    for offset in (0.0, 1_000_000_000.0):
        vertices, faces = _triangle(offset_x=offset)
        recipe = _distance_recipe(vertices, faces)
        receipt, qc = extract_surface_measurement(vertices, faces, recipe)

        measurement = receipt["measurement"]
        assert measurement["distance_mm_decimal"] == "1.000000"
        assert measurement["squared_distance_exact_mm2"] == {
            "denominator": "1",
            "numerator": "1",
        }
        assert receipt["quality"]["status"] == "pass"
        assert qc["distance_mm_decimal"] == "1.000000"
        assert qc["diameter_mm_decimal"] is None


def test_large_quantized_triangle_area_does_not_overflow_int64() -> None:
    extent_mm = float(2**32) / 1000.0
    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [extent_mm, 0.0, 0.0], [0.0, extent_mm, 0.0]],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int32)
    recipe = surface_distance_recipe(
        [
            _anchor(vertices, faces, (0.0, 0.0, 0.0)),
            _anchor(vertices, faces, (extent_mm, 0.0, 0.0)),
        ],
        source_vertex_count=3,
        source_face_count=1,
    )
    receipt, _qc = extract_surface_measurement(vertices, faces, recipe)
    assert receipt["measurement"]["distance_mm_decimal"] == "4294967.296000"


def test_unreferenced_extreme_vertex_does_not_change_local_anchor_policy() -> None:
    triangle, faces = _triangle()
    vertices = np.vstack((triangle, np.asarray([[10_000_000_000_000.0, 0.0, 0.0]])))
    recipe = surface_distance_recipe(
        [
            _anchor(vertices, faces, (-0.5, 0.0, 0.0)),
            _anchor(vertices, faces, (0.5, 0.0, 0.0)),
        ],
        source_vertex_count=4,
        source_face_count=1,
    )
    direct, _direct_qc = extract_surface_measurement(vertices, faces, recipe)
    source, _source_qc = extract_surface_measurement_from_source(
        vertices,
        faces,
        np.eye(4, dtype=np.float64),
        recipe,
    )
    assert direct == source
    assert direct["measurement"]["distance_mm_decimal"] == "1.000000"


@pytest.mark.parametrize("offset", [0.0, 100_000_000_000.0])
def test_four_anchor_best_fit_circle_has_exact_one_millimetre_diameter(
    offset: float,
) -> None:
    vertices, faces = _triangle(offset_x=offset)
    anchors = [
        _anchor(vertices, faces, (offset + 0.5, 0.0, 0.0)),
        _anchor(vertices, faces, (offset, 0.5, 0.0)),
        _anchor(vertices, faces, (offset - 0.5, 0.0, 0.0)),
        _anchor(vertices, faces, (offset, -0.5, 0.0)),
    ]
    recipe = surface_diameter_recipe(
        anchors,
        source_vertex_count=3,
        source_face_count=1,
    )
    receipt, qc = extract_surface_measurement(vertices, faces, recipe)
    measurement = receipt["measurement"]

    assert measurement["diameter_mm_decimal"] == "1.000000"
    assert measurement["radius_mm_decimal"] == "0.500000"
    assert measurement["center_mm_decimal"] == [
        f"{offset:.6f}",
        "0.000000",
        "0.000000",
    ]
    assert measurement["plane_rms_residual_mm_decimal"] == "0.000000"
    assert measurement["radial_rms_residual_mm_decimal"] == "0.000000"
    assert measurement["sample_count"] == 4
    assert receipt["quality"]["status"] == "pass"
    assert qc["diameter_mm_decimal"] == "1.000000"


def test_invalid_or_degenerate_anchor_sets_fail_closed() -> None:
    vertices, faces = _triangle()
    distance = _distance_recipe(vertices, faces)

    duplicate = copy.deepcopy(distance)
    duplicate["anchors"][1] = copy.deepcopy(duplicate["anchors"][0])
    with pytest.raises(ArtifactSurfaceMeasurementError, match="distinct"):
        extract_surface_measurement(vertices, faces, duplicate)

    mismatched = copy.deepcopy(distance)
    mismatched["anchors"][0]["face_vertex_indices"] = [0, 2, 1]
    with pytest.raises(ArtifactSurfaceMeasurementError, match="identity changed"):
        extract_surface_measurement(vertices, faces, mismatched)

    bad_barycentric = copy.deepcopy(distance)
    bad_barycentric["anchors"][0]["barycentric_numerators"][0] += 1
    with pytest.raises(ArtifactSurfaceMeasurementError, match="sum exactly"):
        extract_surface_measurement(vertices, faces, bad_barycentric)

    collinear_anchors = [
        _anchor(vertices, faces, (-0.5, 0.0, 0.0)),
        _anchor(vertices, faces, (0.0, 0.0, 0.0)),
        _anchor(vertices, faces, (0.5, 0.0, 0.0)),
    ]
    collinear = surface_diameter_recipe(
        collinear_anchors,
        source_vertex_count=3,
        source_face_count=1,
    )
    with pytest.raises(ArtifactSurfaceMeasurementError, match="collinear"):
        extract_surface_measurement(vertices, faces, collinear)


def test_schema_runtime_and_known_record_validation_reject_tampering() -> None:
    vertices, faces = _triangle()
    recipe = _distance_recipe(vertices, faces)
    receipt, qc = extract_surface_measurement(vertices, faces, recipe)
    schema = json.loads(
        (ROOT / "schemas/surface_measurement_receipt-1.0.0.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    assert list(validator.iter_errors(receipt)) == []

    tampered = copy.deepcopy(receipt)
    tampered["measurement"]["distance_mm_decimal"] = "9.000000"
    assert list(validator.iter_errors(tampered)) == []
    with pytest.raises(ArtifactSurfaceMeasurementError, match="exact squared"):
        validate_surface_measurement_receipt(tampered)

    coherent_result_forgery = copy.deepcopy(receipt)
    coherent_result_forgery["measurement"]["distance_mm_decimal"] = "9.000000"
    coherent_result_forgery["measurement"]["squared_distance_exact_mm2"] = {
        "denominator": "1",
        "numerator": "81",
    }
    with pytest.raises(ArtifactSurfaceMeasurementError, match="derived from its anchors"):
        validate_surface_measurement_receipt(coherent_result_forgery)

    forged_edge = copy.deepcopy(receipt)
    forged_edge["anchors"][0]["edge_status"] = "near_edge"
    forged_edge["quality"]["near_edge_anchor_count"] = 1
    forged_edge["quality"]["review_reasons"] = ["anchor_near_triangle_edge"]
    forged_edge["quality"]["status"] = "review"
    with pytest.raises(ArtifactSurfaceMeasurementError, match="edge status"):
        validate_surface_measurement_receipt(forged_edge)

    forged_point = copy.deepcopy(receipt)
    forged_point["anchors"][0]["resolved_point_numerator_grid_bary"][0] = (
        "9" * 128
    )
    with pytest.raises(ArtifactSurfaceMeasurementError, match="safe source/grid"):
        validate_surface_measurement_receipt(forged_point)

    forged_tolerance = copy.deepcopy(receipt)
    forged_tolerance["anchors"][0]["depth_match_tolerance_um"] += 1
    with pytest.raises(ArtifactSurfaceMeasurementError, match="depth tolerance"):
        validate_surface_measurement_receipt(forged_tolerance)

    session = _session(vertices, faces)
    projection = session.materialize()
    context = session.capture_operation(
        recipe=recipe,
        selection_hash=surface_measurement_selection_hash(recipe),
    )
    computation = ArtifactSurfaceMeasurementComputation(
        context=context,
        projection_snapshot=projection.snapshot,
        receipt=receipt,
        recipe=recipe,
        qc=qc,
    )
    committed = commit_artifact_surface_measurement(
        session,
        computation,
        record_id="record:surface-distance",
        created_at=STAMP,
        operator="tester",
    )
    record = committed.document.record_index["record:surface-distance"]
    assert record.type == SURFACE_DISTANCE_RECORD_TYPE
    assert surface_measurement_receipt_from_record(record) == receipt
    validate_known_records(committed.document)
    reopened = ArtifactDocument.from_json_bytes(committed.document.canonical_json_bytes())
    rebound = ArtifactSession.bind_loaded_document(
        reopened,
        session.source_mesh,
        resolved_source_path=session.resolved_source_path,
    )
    assert surface_measurement_receipt_from_record(
        rebound.document.record_index[record.id]
    ) == receipt

    oversized_value = copy.deepcopy(committed.document.to_dict())
    oversized_value["records"][0]["extensions"][
        "org.archmeshrubbing:surface-measurement-v1"
    ]["receipt_byte_length"] = 128 * 1024 + 1
    oversized = ArtifactDocument.from_dict(oversized_value)
    with pytest.raises(ArtifactKnownRecordError, match="receipt"):
        validate_known_records(oversized)

    value = committed.document.to_dict()
    value["records"][0]["extensions"][
        "org.archmeshrubbing:surface-measurement-v1"
    ]["receipt"]["anchors"][0]["face_vertex_indices"] = [0, 2, 1]
    forged = ArtifactDocument.from_dict(value)
    with pytest.raises(ArtifactKnownRecordError, match="receipt"):
        validate_known_records(forged)


def test_diameter_record_type_and_cancellation_are_closed() -> None:
    vertices, faces = _triangle()
    anchors = [
        _anchor(vertices, faces, (0.5, 0.0, 0.0)),
        _anchor(vertices, faces, (0.0, 0.5, 0.0)),
        _anchor(vertices, faces, (-0.5, 0.0, 0.0)),
    ]
    recipe = surface_diameter_recipe(
        anchors,
        source_vertex_count=3,
        source_face_count=1,
    )
    calls = 0

    def cancel() -> bool:
        nonlocal calls
        calls += 1
        return calls >= 2

    from src.core.artifact_cancellation import ArtifactComputationCancelledError

    with pytest.raises(ArtifactComputationCancelledError):
        extract_surface_measurement(
            vertices,
            faces,
            recipe,
            cancellation_probe=cancel,
        )

    receipt, qc = extract_surface_measurement(vertices, faces, recipe)
    session = _session(vertices, faces)
    context = session.capture_operation(
        recipe=recipe,
        selection_hash=surface_measurement_selection_hash(recipe),
    )
    computation = ArtifactSurfaceMeasurementComputation(
        context=context,
        projection_snapshot=session.projection_snapshot(),
        receipt=receipt,
        recipe=recipe,
        qc=qc,
    )
    committed = commit_artifact_surface_measurement(
        session,
        computation,
        record_id="record:surface-diameter",
        created_at=STAMP,
        operator="tester",
    )
    assert committed.document.record_index["record:surface-diameter"].type == SURFACE_DIAMETER_RECORD_TYPE
