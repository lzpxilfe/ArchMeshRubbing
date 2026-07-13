from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from src.core.artifact_document import ArtifactDocument, DerivedRecord
from src.core.artifact_record_validation import (
    ArtifactKnownRecordError,
    validate_known_records,
)
from src.core.artifact_session import ArtifactSession
from src.core.artifact_tile_unwrap_extractor import (
    ArtifactTileUnwrapError,
    TileUnwrapMesh,
    commit_artifact_tile_unwrap,
    compute_artifact_tile_unwrap,
    compute_artifact_tile_unwrap_from_recipe,
    extract_tile_unwrap,
    require_current_tile_unwrap_computation,
    tile_unwrap_recipe,
    validate_tile_unwrap_recipe,
)
from src.core.artifact_tile_unwrap_record import (
    TILE_UNWRAP_RECEIPT_EXTENSION_KEY,
    TILE_UNWRAP_RECORD_TYPE,
    tile_unwrap_receipt_from_record,
)
from src.core.mesh_import_recipe import current_mesh_import_recipe
from src.core.mesh_loader import MeshData
from src.core.source_identity import SourceFingerprint
from src.core.tile_synthetic import (
    SyntheticTileGroundTruth,
    generate_synthetic_tile,
    synthetic_tile_spec_from_preset,
)


STAMP = "2026-07-13T00:00:00Z"


def _source_mesh(*, seed: int = 7) -> tuple[MeshData, SyntheticTileGroundTruth]:
    artifact = generate_synthetic_tile(
        synthetic_tile_spec_from_preset("sugkiwa_quarter", seed=seed)
    )
    mesh = MeshData(
        vertices=np.asarray(artifact.mesh.vertices, dtype=np.float64),
        faces=np.asarray(artifact.mesh.faces, dtype=np.int32),
        unit="mm",
        filepath=Path("/source/synthetic-tile.ply"),
        source_identity=SourceFingerprint(
            sha256=f"{seed:064x}",
            size_bytes=123_456,
            mtime_ns=1,
            original_name="synthetic-tile.ply",
            format="ply",
        ),
        source_format="ply",
        source_import_recipe=current_mesh_import_recipe("ply"),
    )
    return mesh, artifact.truth


def _aligned_session(
    *, seed: int = 7
) -> tuple[ArtifactSession, SyntheticTileGroundTruth]:
    mesh, truth = _source_mesh(seed=seed)
    session = ArtifactSession.create_from_source(
        mesh,
        resolved_source_path="/source/synthetic-tile.ply",
        unit="mm",
        axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
        handedness="right",
        software_version="0.7-test",
        operator="pytest",
        created_at=STAMP,
        document_id=f"artifact:tile:{seed}",
        metadata_revision_id="metadata:initial",
        align_revision_id="align:initial",
    )
    return (
        session.commit_preview(
            translation_mm=(0.0, 0.0, 0.0),
            rotation_deg=(0.0, 0.0, 0.0),
            scale=1.0,
            pivot_mm=(0.0, 0.0, 0.0),
            operator="pytest",
            created_at="2026-07-13T00:00:01Z",
            revision_id="align:confirmed",
        ),
        truth,
    )


def _recorded_session(
    *, seed: int = 7
) -> tuple[ArtifactSession, SyntheticTileGroundTruth]:
    session, truth = _aligned_session(seed=seed)
    computation = compute_artifact_tile_unwrap(
        session,
        longitudinal_axis="y",
        record_view="top",
        n_sections=32,
    )
    return (
        commit_artifact_tile_unwrap(
            session,
            computation,
            record_id="record:tile-unwrap",
            created_at="2026-07-13T00:00:02Z",
            operator="pytest",
        ),
        truth,
    )


def test_recipe_persists_canonical_face_ranges_and_rejects_axis_guessing() -> None:
    recipe = tile_unwrap_recipe(
        longitudinal_axis="y",
        record_view="top",
        total_face_count=10,
        selected_face_indices=[4, 0, 1, 3, 4, 8],
        n_sections=24,
    )

    assert recipe["selection"]["face_ranges"] == [[0, 2], [3, 5], [8, 9]]
    assert recipe["selection"]["selected_face_count"] == 5
    assert validate_tile_unwrap_recipe(recipe) == recipe

    with pytest.raises(ArtifactTileUnwrapError, match="explicit canonical"):
        tile_unwrap_recipe(
            longitudinal_axis="auto",
            record_view="top",
            total_face_count=10,
        )


def test_authoritative_unwrap_is_deterministic_and_tracks_tile_scale() -> None:
    session, truth = _aligned_session()

    first = compute_artifact_tile_unwrap(
        session,
        longitudinal_axis="y",
        record_view="top",
        n_sections=32,
    )
    second = compute_artifact_tile_unwrap_from_recipe(session, first.recipe)

    selection_sha = first.context.selection_hash
    assert selection_sha is not None
    first_receipt = first.unwrap.receipt(selection_sha256=selection_sha)
    second_receipt = second.unwrap.receipt(
        selection_sha256=second.context.selection_hash or ""
    )
    assert first_receipt == second_receipt
    assert first.unwrap.uv_um.flags.writeable is False
    assert first.unwrap.faces.flags.writeable is False
    assert first.qc["foldover_face_count"] == 0
    assert first.qc["degenerate_uv_face_count"] == 0
    assert first.qc["distortion_p95_millionths"] < 100_000

    expected_length_um = int(round(float(truth.spec.length_world) * 1000.0))
    measured_length_um = int(first.qc["height_um"])
    assert abs(measured_length_um - expected_length_um) / expected_length_um < 0.02


def test_commit_creates_strict_content_addressed_record_and_roundtrips() -> None:
    session, _truth = _recorded_session()
    record = session.document.record_index["record:tile-unwrap"]
    receipt = tile_unwrap_receipt_from_record(record)

    assert record.type == TILE_UNWRAP_RECORD_TYPE
    assert record.selection_hash == receipt["selection_sha256"]
    assert record.geometry_ref.endswith(receipt["unwrap_sha256"])
    assert receipt["width_mm_exact"]["denominator"] == 1000
    assert receipt["height_mm_exact"]["denominator"] == 1000
    validate_known_records(session.document)

    reparsed = ArtifactDocument.from_dict(session.document.to_dict())
    validate_known_records(reparsed)
    assert reparsed.canonical_json_bytes() == session.document.canonical_json_bytes()


def test_generated_receipt_matches_public_json_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    session, _truth = _recorded_session(seed=8)
    record = session.document.record_index["record:tile-unwrap"]
    receipt = tile_unwrap_receipt_from_record(record)
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "schemas"
        / "tile_unwrap_receipt-1.0.0.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)

    assert list(validator.iter_errors(receipt)) == []
    tampered = dict(receipt)
    tampered["axis"] = "auto"
    assert list(validator.iter_errors(tampered))


@pytest.mark.parametrize("tamper", ["receipt", "qc", "section_recipe"])
def test_known_record_validation_rejects_tampering(tamper: str) -> None:
    session, _truth = _recorded_session(seed=9)
    record = session.document.record_index["record:tile-unwrap"]
    record_dict = record.to_dict()
    if tamper == "receipt":
        descriptor = record_dict["extensions"][TILE_UNWRAP_RECEIPT_EXTENSION_KEY]
        descriptor["receipt"]["unwrap_sha256"] = "0" * 64
    elif tamper == "qc":
        record_dict["qc"]["distortion_mean_millionths"] = 999_999
    else:
        record_dict["qc"]["section_count"] = 24
        record_dict["qc"]["section_fit_valid_count"] = 24
    broken = DerivedRecord.from_dict(record_dict)
    document = replace(session.document, records=(broken,))

    with pytest.raises(ArtifactKnownRecordError):
        validate_known_records(document)


def test_known_record_selection_is_bound_to_geometry_counts() -> None:
    session, _truth = _recorded_session(seed=10)
    geometry = session.document.geometry_revisions[0]
    geometry_qc = dict(geometry.qc)
    geometry_qc["face_count"] = int(geometry_qc["face_count"]) + 1
    broken_geometry = replace(geometry, qc=geometry_qc)
    document = replace(
        session.document,
        geometry_revisions=(broken_geometry,),
    )

    with pytest.raises(ArtifactKnownRecordError, match="source geometry face count"):
        validate_known_records(document)


def test_tile_unwrap_mesh_rejects_integer_narrowing_overflow() -> None:
    with pytest.raises(ArtifactTileUnwrapError, match="face index"):
        TileUnwrapMesh(
            uv_um=np.asarray([[0, 0], [1, 0], [0, 1]], dtype=np.int64),
            faces=np.asarray([[0, 1, 2**32]], dtype=np.uint64),
            source_vertex_indices=np.asarray([0, 1, 2], dtype=np.int64),
            source_face_indices=np.asarray([0], dtype=np.int64),
            axis="y",
            record_view="top",
        )


def test_authoritative_unwrap_rejects_internal_algorithm_fallback() -> None:
    vertices = np.asarray(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4]],
        dtype=np.int32,
    )
    mesh = MeshData(vertices=vertices, faces=faces, unit="mm")
    recipe = tile_unwrap_recipe(
        longitudinal_axis="y",
        record_view="top",
        total_face_count=4,
        n_sections=12,
    )

    with pytest.raises(ArtifactTileUnwrapError, match="rejected algorithm fallback"):
        extract_tile_unwrap(mesh, recipe)


def test_align_change_makes_computation_stale_without_deleting_it() -> None:
    session, _truth = _aligned_session(seed=11)
    computation = compute_artifact_tile_unwrap(
        session,
        longitudinal_axis="y",
        record_view="top",
        n_sections=24,
    )
    changed = session.commit_preview(
        translation_mm=(1.0, 0.0, 0.0),
        rotation_deg=(0.0, 0.0, 0.0),
        scale=1.0,
        pivot_mm=(0.0, 0.0, 0.0),
        operator="pytest",
        created_at="2026-07-13T00:00:03Z",
        revision_id="align:changed",
    )

    with pytest.raises(ArtifactTileUnwrapError, match="stale"):
        require_current_tile_unwrap_computation(changed, computation)

    historical = commit_artifact_tile_unwrap(
        changed,
        computation,
        record_id="record:historical-tile-unwrap",
        created_at="2026-07-13T00:00:04Z",
        operator="pytest",
    )
    assert (
        historical.document.record_freshness("record:historical-tile-unwrap").value
        == "stale_alignment"
    )


def test_top_and_bottom_are_explicit_distinct_measurement_records() -> None:
    session, _truth = _aligned_session(seed=13)
    top = compute_artifact_tile_unwrap(
        session,
        longitudinal_axis="y",
        record_view="top",
        n_sections=24,
    )
    bottom = compute_artifact_tile_unwrap(
        session,
        longitudinal_axis="y",
        record_view="bottom",
        n_sections=24,
    )
    top_receipt = top.unwrap.receipt(selection_sha256=top.context.selection_hash or "")
    bottom_receipt = bottom.unwrap.receipt(
        selection_sha256=bottom.context.selection_hash or ""
    )

    assert top_receipt["unwrap_sha256"] != bottom_receipt["unwrap_sha256"]
    assert top_receipt["width_mm_exact"] == bottom_receipt["width_mm_exact"]
    assert top_receipt["height_mm_exact"] == bottom_receipt["height_mm_exact"]
