from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from src.core.artifact_geometry_metrics import (
    ArtifactGeometryMetricsError,
    GEOMETRY_METRICS_RECORD_TYPE,
    commit_artifact_geometry_metrics,
    compute_artifact_geometry_metrics,
    extract_geometry_metrics,
    geometry_metrics_receipt_from_record,
    geometry_metrics_recipe,
    validate_geometry_metrics_receipt,
)
from src.core.artifact_document import ArtifactDocument
from src.core.artifact_record_validation import (
    ArtifactKnownRecordError,
    validate_known_records,
)
from src.core.artifact_session import ArtifactSession
from src.core.mesh_import_recipe import current_mesh_import_recipe
from src.core.mesh_loader import MeshData
from src.core.source_identity import SourceFingerprint


ROOT = Path(__file__).resolve().parents[1]
STAMP = "2026-07-18T00:00:00Z"


def _tetrahedron(*, offset_x: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [
            [0.0 + offset_x, 0.0, 0.0],
            [1.0 + offset_x, 0.0, 0.0],
            [0.0 + offset_x, 1.0, 0.0],
            [0.0 + offset_x, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
        dtype=np.int32,
    )
    return vertices, faces


def _session(vertices: np.ndarray, faces: np.ndarray) -> ArtifactSession:
    mesh = MeshData(
        vertices=vertices,
        faces=faces,
        unit="mm",
        filepath=Path("/private/metrics/scan.ply"),
        source_identity=SourceFingerprint(
            sha256="8" * 64,
            size_bytes=1024,
            mtime_ns=1,
            original_name="metrics.ply",
            format="ply",
        ),
        source_format="ply",
        source_import_recipe=current_mesh_import_recipe("ply"),
    )
    return ArtifactSession.create_from_source(
        mesh,
        resolved_source_path="/private/metrics/scan.ply",
        unit="mm",
        axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
        handedness="right",
        software_version="metrics-test",
        operator="tester",
        created_at=STAMP,
        document_id="artifact:geometry-metrics",
        metadata_revision_id="metadata:geometry-metrics",
        align_revision_id="align:geometry-metrics",
    )


def test_closed_tetrahedron_records_exact_guarded_volume_and_area() -> None:
    vertices, faces = _tetrahedron()
    session = _session(vertices, faces)
    computation = compute_artifact_geometry_metrics(session)
    receipt = computation.receipt_dict()

    assert receipt["surface_area"] == {
        "decimal_mm2": "2.366025",
        "decimal_places": 6,
        "status": "available",
    }
    assert receipt["volume"] == {
        "decimal_mm3": "0.166666667",
        "decimal_places": 9,
        "exact_rational_mm3": {"denominator": "6", "numerator": "1"},
        "policy": "single_closed_consistently_oriented_edge_manifold_component/v1",
        "signed_six_grid_units3": "1000000000",
        "status": "available",
        "winding": "positive",
    }
    assert receipt["topology"]["closed_edge_manifold"] is True
    assert receipt["topology"]["consistently_oriented"] is True

    committed = commit_artifact_geometry_metrics(
        session,
        computation,
        record_id="record:metrics:tetrahedron",
        created_at=STAMP,
        operator="tester",
    )
    record = committed.document.record_index["record:metrics:tetrahedron"]
    assert record.type == GEOMETRY_METRICS_RECORD_TYPE
    assert geometry_metrics_receipt_from_record(record) == receipt
    validate_known_records(committed.document)

    reopened = committed.document.from_json_bytes(
        committed.document.canonical_json_bytes()
    )
    validate_known_records(reopened)
    assert geometry_metrics_receipt_from_record(
        reopened.record_index[record.id]
    ) == receipt


def test_open_surface_keeps_area_but_refuses_volume() -> None:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 2.0, 0.0]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    receipt, qc = extract_geometry_metrics(
        vertices,
        faces,
        geometry_metrics_recipe(),
    )

    assert receipt["surface_area"]["decimal_mm2"] == "4.000000"
    assert receipt["topology"]["boundary_edge_count"] == 4
    assert receipt["topology"]["closed_edge_manifold"] is False
    assert receipt["volume"]["status"] == "unavailable_topology"
    assert receipt["volume"]["decimal_mm3"] is None
    assert qc["volume_status"] == "unavailable_topology"


def test_disconnected_closed_shells_refuse_ambiguous_combined_volume() -> None:
    first_vertices, first_faces = _tetrahedron()
    second_vertices, second_faces = _tetrahedron(offset_x=5.0)
    vertices = np.vstack((first_vertices, second_vertices))
    faces = np.vstack((first_faces, second_faces + 4))

    receipt, _qc = extract_geometry_metrics(
        vertices,
        faces,
        geometry_metrics_recipe(),
    )

    assert receipt["topology"]["closed_edge_manifold"] is True
    assert receipt["topology"]["consistently_oriented"] is True
    assert receipt["topology"]["connected_component_count"] == 2
    assert receipt["volume"]["status"] == "unavailable_topology"


def test_orientation_mismatch_refuses_volume() -> None:
    vertices, faces = _tetrahedron()
    faces[1] = faces[1, ::-1]
    receipt, _qc = extract_geometry_metrics(
        vertices,
        faces,
        geometry_metrics_recipe(),
    )

    assert receipt["topology"]["closed_edge_manifold"] is True
    assert receipt["topology"]["orientation_mismatch_edge_count"] == 3
    assert receipt["topology"]["consistently_oriented"] is False
    assert receipt["volume"]["status"] == "unavailable_topology"


def test_quantization_is_explicit_and_repeatable() -> None:
    vertices, faces = _tetrahedron()
    vertices[1, 0] = 1.0004
    recipe = geometry_metrics_recipe(coordinate_grid_um=1)

    first, first_qc = extract_geometry_metrics(vertices, faces, recipe)
    second, second_qc = extract_geometry_metrics(vertices.copy(), faces.copy(), recipe)

    assert first == second
    assert first_qc == second_qc
    assert first["quantization"]["changed_vertex_count"] == 1
    assert first["quantization"]["maximum_displacement_um"] == "0.400000"
    assert first["bounds_grid"]["maximum"] == [1000, 1000, 1000]


def test_receipt_schema_and_runtime_validator_reject_tampering() -> None:
    vertices, faces = _tetrahedron()
    receipt, _qc = extract_geometry_metrics(
        vertices,
        faces,
        geometry_metrics_recipe(),
    )
    schema = json.loads(
        (ROOT / "schemas/geometry_metrics_receipt-1.0.0.schema.json").read_text(
            encoding="utf-8"
        )
    )
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)
    assert list(validator.iter_errors(receipt)) == []

    tampered = copy.deepcopy(receipt)
    tampered["volume"]["decimal_mm3"] = "9.999999999"
    assert list(validator.iter_errors(tampered)) == []
    with pytest.raises(ArtifactGeometryMetricsError, match="does not match"):
        validate_geometry_metrics_receipt(tampered)

    unknown = copy.deepcopy(receipt)
    unknown["topology"]["sampled"] = False
    assert list(validator.iter_errors(unknown))
    with pytest.raises(ArtifactGeometryMetricsError, match="unknown fields"):
        validate_geometry_metrics_receipt(unknown)


def test_known_record_registry_rejects_project_receipt_tampering() -> None:
    vertices, faces = _tetrahedron()
    session = _session(vertices, faces)
    computation = compute_artifact_geometry_metrics(session)
    committed = commit_artifact_geometry_metrics(
        session,
        computation,
        record_id="record:metrics:tamper",
        created_at=STAMP,
        operator="tester",
    )
    document_value = committed.document.to_dict()
    descriptor = document_value["records"][0]["extensions"][
        "org.archmeshrubbing:geometry-metrics-v1"
    ]
    descriptor["receipt"]["surface_area"]["decimal_mm2"] = "99.000000"
    tampered = ArtifactDocument.from_dict(document_value)

    with pytest.raises(ArtifactKnownRecordError, match="receipt"):
        validate_known_records(tampered)


def test_computation_cancellation_does_not_create_a_partial_result() -> None:
    vertices, faces = _tetrahedron()
    calls = 0

    def cancel() -> bool:
        nonlocal calls
        calls += 1
        return calls >= 2

    from src.core.artifact_cancellation import ArtifactComputationCancelledError

    with pytest.raises(ArtifactComputationCancelledError):
        extract_geometry_metrics(
            vertices,
            faces,
            geometry_metrics_recipe(),
            cancellation_probe=cancel,
        )
