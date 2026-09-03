from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from src.core.artifact_document import (
    ArtifactDocument,
    DerivedRecord,
    canonical_recipe_hash,
)
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
    _orientation_qc,
    _uv_overlap_pair_count,
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
from src.core.flatten_models_sectionwise import (
    sectionwise_cylindrical_parameterization,
)
from src.core.tile_synthetic import (
    SyntheticTileGroundTruth,
    generate_synthetic_tile,
    synthetic_tile_spec_from_preset,
)


STAMP = "2026-07-13T00:00:00Z"


class _PlainMesh:
    """Minimal vertices/faces carrier for the Qt-free sectionwise unwrap.

    The parameterisation reads only these three attributes, so a full MeshData
    would drag in source identity and an import receipt that this test has no
    use for. `cast` at the call site records that this is deliberate rather
    than an unnoticed type hole.
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        self.vertices = vertices
        self.faces = faces
        self.n_vertices = int(vertices.shape[0])


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
    assert recipe["algorithm_version"] == "1.2.0"
    assert recipe["schema_version"] == "1.2.0"
    assert recipe["seam_policy"] == "minimum_angular_range_auto"
    assert recipe["seam_angle_microdegrees"] is None
    assert validate_tile_unwrap_recipe(recipe) == recipe

    upper_bound_recipe = tile_unwrap_recipe(
        longitudinal_axis="y",
        record_view="top",
        total_face_count=10,
        n_sections=96,
    )
    assert upper_bound_recipe["n_sections"] == 96

    with pytest.raises(
        ArtifactTileUnwrapError,
        match=r"n_sections must be in the inclusive range 12\.\.96",
    ):
        tile_unwrap_recipe(
            longitudinal_axis="y",
            record_view="top",
            total_face_count=10,
            n_sections=97,
        )

    with pytest.raises(ArtifactTileUnwrapError, match="explicit canonical"):
        tile_unwrap_recipe(
            longitudinal_axis="auto",
            record_view="top",
            total_face_count=10,
        )
    with pytest.raises(ArtifactTileUnwrapError, match="250000-face QC limit"):
        tile_unwrap_recipe(
            longitudinal_axis="y",
            record_view="top",
            total_face_count=250_001,
        )


def test_legacy_1_1_recipe_hash_and_recompute_are_not_upgraded() -> None:
    hash_fixture = tile_unwrap_recipe(
        longitudinal_axis="y",
        record_view="top",
        total_face_count=10,
        selected_face_indices=[4, 0, 1, 3, 4, 8],
        n_sections=24,
    )
    hash_fixture.pop("seam_angle_microdegrees")
    hash_fixture["algorithm_version"] = "1.1.0"
    hash_fixture["schema_version"] = "1.1.0"

    assert validate_tile_unwrap_recipe(hash_fixture) == hash_fixture
    assert canonical_recipe_hash(hash_fixture) == (
        "0318c7c5c5b4c901d5e22f2b1941f894200ac3f221e2f1f8b0a638cf7da1ed3b"
    )

    session, _truth = _aligned_session()
    face_count = int(np.asarray(session.source_mesh.faces).shape[0])
    current_auto = tile_unwrap_recipe(
        longitudinal_axis="y",
        record_view="top",
        total_face_count=face_count,
        n_sections=32,
    )
    legacy = dict(current_auto)
    legacy.pop("seam_angle_microdegrees")
    legacy["algorithm_version"] = "1.1.0"
    legacy["schema_version"] = "1.1.0"

    legacy_result = compute_artifact_tile_unwrap_from_recipe(session, legacy)
    current_result = compute_artifact_tile_unwrap_from_recipe(session, current_auto)

    assert dict(legacy_result.recipe) == legacy
    assert legacy_result.context.recipe_hash == canonical_recipe_hash(legacy)
    assert np.array_equal(legacy_result.unwrap.uv_um, current_result.unwrap.uv_um)
    assert np.array_equal(legacy_result.unwrap.faces, current_result.unwrap.faces)


@pytest.mark.parametrize(
    "invalid",
    [
        True,
        0.0,
        float("nan"),
        float("inf"),
        "0",
        -180_000_001,
        180_000_000,
    ],
)
def test_explicit_seam_recipe_rejects_noncanonical_values(invalid: object) -> None:
    with pytest.raises(ArtifactTileUnwrapError, match="seam_angle_microdegrees"):
        tile_unwrap_recipe(
            longitudinal_axis="y",
            record_view="top",
            total_face_count=10,
            seam_angle_microdegrees=invalid,  # type: ignore[arg-type]
        )


def test_explicit_seam_recipe_is_closed_and_policy_bound() -> None:
    recipe = tile_unwrap_recipe(
        longitudinal_axis="y",
        record_view="top",
        total_face_count=10,
        seam_angle_microdegrees=-180_000_000,
    )

    assert recipe["seam_policy"] == "fixed_angle_microdegrees"
    assert recipe["seam_angle_microdegrees"] == -180_000_000
    assert validate_tile_unwrap_recipe(recipe) == recipe

    inconsistent = dict(recipe)
    inconsistent["seam_policy"] = "minimum_angular_range_auto"
    with pytest.raises(ArtifactTileUnwrapError, match="inconsistent"):
        validate_tile_unwrap_recipe(inconsistent)

    missing = dict(recipe)
    missing.pop("seam_angle_microdegrees")
    with pytest.raises(ArtifactTileUnwrapError, match="missing fields"):
        validate_tile_unwrap_recipe(missing)

    unknown = dict(recipe)
    unknown["seam_angle_degrees"] = 0
    with pytest.raises(ArtifactTileUnwrapError, match="unknown fields"):
        validate_tile_unwrap_recipe(unknown)


def test_explicit_seam_is_deterministic_and_changes_parameterization() -> None:
    session, _truth = _aligned_session()
    zero = compute_artifact_tile_unwrap(
        session,
        longitudinal_axis="y",
        record_view="top",
        n_sections=32,
        seam_angle_microdegrees=0,
    )
    repeated = compute_artifact_tile_unwrap_from_recipe(session, zero.recipe)
    quarter_turn = compute_artifact_tile_unwrap(
        session,
        longitudinal_axis="y",
        record_view="top",
        n_sections=32,
        seam_angle_microdegrees=90_000_000,
    )

    assert np.array_equal(zero.unwrap.uv_um, repeated.unwrap.uv_um)
    assert zero.qc == repeated.qc
    assert not np.array_equal(zero.unwrap.uv_um, quarter_turn.unwrap.uv_um)
    assert zero.context.recipe_hash != quarter_turn.context.recipe_hash
    assert (
        zero.unwrap.receipt(selection_sha256=zero.context.selection_hash or "")[
            "unwrap_sha256"
        ]
        != quarter_turn.unwrap.receipt(
            selection_sha256=quarter_turn.context.selection_hash or ""
        )["unwrap_sha256"]
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
        / "tile_unwrap_receipt-1.1.0.schema.json"
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


@pytest.mark.parametrize(
    ("field", "value", "expected_message"),
    [
        ("distortion_max_millionths", 250_001, "max distortion"),
        ("distortion_p95_millionths", 150_001, "p95 distortion"),
        ("distortion_mean_millionths", 75_001, "mean distortion"),
    ],
)
def test_known_record_validation_enforces_current_distortion_gates(
    field: str,
    value: int,
    expected_message: str,
) -> None:
    session, _truth = _recorded_session(seed=9)
    record = session.document.record_index["record:tile-unwrap"]
    record_dict = record.to_dict()
    qc = record_dict["qc"]
    qc["distortion_median_millionths"] = 0
    qc["distortion_mean_millionths"] = 0
    qc["distortion_p95_millionths"] = 0
    qc["distortion_max_millionths"] = value
    qc[field] = value
    broken = DerivedRecord.from_dict(record_dict)
    document = replace(session.document, records=(broken,))

    with pytest.raises(ArtifactKnownRecordError, match=expected_message):
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


def test_authoritative_unwrap_rejects_disconnected_recording_surfaces() -> None:
    mesh, _truth = _source_mesh(seed=17)
    vertices = np.concatenate(
        [mesh.vertices, mesh.vertices + np.asarray([500.0, 0.0, 0.0])],
        axis=0,
    )
    faces = np.concatenate(
        [mesh.faces, mesh.faces + mesh.n_vertices],
        axis=0,
    )
    duplicated = MeshData(vertices=vertices, faces=faces, unit="mm")
    recipe = tile_unwrap_recipe(
        longitudinal_axis="y",
        record_view="top",
        total_face_count=duplicated.n_faces,
        n_sections=24,
    )

    with pytest.raises(ArtifactTileUnwrapError, match="one edge-connected component"):
        extract_tile_unwrap(duplicated, recipe)


def test_authoritative_unwrap_rejects_duplicate_faces_before_ready() -> None:
    mesh, _truth = _source_mesh(seed=18)
    faces = np.concatenate([mesh.faces, mesh.faces[:1]], axis=0)
    duplicated = MeshData(vertices=mesh.vertices, faces=faces, unit="mm")
    recipe = tile_unwrap_recipe(
        longitudinal_axis="y",
        record_view="top",
        total_face_count=duplicated.n_faces,
        n_sections=24,
    )

    with pytest.raises(ArtifactTileUnwrapError, match="duplicate faces"):
        extract_tile_unwrap(duplicated, recipe)


def test_authoritative_unwrap_tolerates_small_axial_scan_noise() -> None:
    mesh, _truth = _source_mesh(seed=18)
    vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
    pattern = ((np.arange(vertices.shape[0], dtype=np.int64) * 17) % 11) - 5
    vertices[:, 1] += pattern.astype(np.float64) * 0.0002
    noisy = MeshData(vertices=vertices, faces=mesh.faces, unit="mm")
    recipe = tile_unwrap_recipe(
        longitudinal_axis="y",
        record_view="top",
        total_face_count=noisy.n_faces,
        n_sections=32,
    )

    _unwrap, qc = extract_tile_unwrap(noisy, recipe)

    assert qc["distortion_max_millionths"] <= 250_000
    assert qc["distortion_p95_millionths"] <= 150_000
    assert qc["distortion_mean_millionths"] <= 75_000


def test_global_uv_overlap_qc_detects_positive_area_overlap() -> None:
    uv_um = np.asarray(
        [
            [0, 0],
            [10, 0],
            [0, 10],
            [1, 1],
            [2, 1],
            [1, 2],
        ],
        dtype=np.int64,
    )
    faces = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int32)

    assert _uv_overlap_pair_count(
        uv_um,
        faces,
        cancellation_probe=None,
    ) == 1


def test_orientation_qc_is_exact_near_maximum_coordinate_extent() -> None:
    extent = 2**52
    uv_um = np.asarray(
        [[0, 0], [extent, extent - 1], [extent - 1, extent - 2]],
        dtype=np.int64,
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int32)

    assert _orientation_qc(uv_um, faces) == {
        "degenerate_uv_face_count": 0,
        "foldover_face_count": 0,
        "negative_orientation_face_count": 1,
        "positive_orientation_face_count": 0,
    }


def test_global_uv_overlap_qc_allows_shared_triangle_edges() -> None:
    uv_um = np.asarray(
        [[0, 0], [10, 0], [0, 10], [10, 10]],
        dtype=np.int64,
    )
    faces = np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int32)

    assert _uv_overlap_pair_count(
        uv_um,
        faces,
        cancellation_probe=None,
    ) == 0


def test_global_uv_overlap_budget_ignores_bounding_box_rejected_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Four disjoint triangles that share one broad-phase grid cell.  Every pair
    # is rejected by the cheap bounding-box test, so none of them costs an
    # exact overlap evaluation and none may consume the budget.  Counting these
    # made the budget scale with grid occupancy instead of real work, which
    # tripped at roughly 55,000 faces and put the documented 250,000-face QC
    # limit out of reach.
    uv_um = np.asarray(
        [
            [0, 0],
            [1, 0],
            [0, 1],
            [10, 0],
            [11, 0],
            [10, 1],
            [0, 10],
            [1, 10],
            [0, 11],
            [10, 10],
            [11, 10],
            [10, 11],
        ],
        dtype=np.int64,
    )
    faces = np.arange(12, dtype=np.int32).reshape(4, 3)
    monkeypatch.setattr(
        "src.core.artifact_tile_unwrap_extractor.MAX_TILE_UNWRAP_OVERLAP_CANDIDATES",
        1,
    )

    assert (
        _uv_overlap_pair_count(uv_um, faces, cancellation_probe=None) == 0
    )


def test_requesting_more_sections_than_the_surface_supports_fails_early() -> None:
    # Stations come from mesh quantiles, so a coarse surface collapses ties and
    # cannot deliver the requested count.  The failure must name the achievable
    # count instead of discarding the finished unwrap at commit time.
    session, _truth = _aligned_session(seed=13)
    with pytest.raises(ArtifactTileUnwrapError) as excinfo:
        compute_artifact_tile_unwrap(
            session,
            longitudinal_axis="y",
            record_view="top",
            n_sections=96,
        )
    message = str(excinfo.value)
    assert "requested 96 sections" in message
    assert "set n_sections to" in message


def test_global_uv_overlap_qc_handles_a_realistic_roof_tile_face_count() -> None:
    # A real scan of a roof tile lands well past the ~55,000 faces at which the
    # candidate-counted budget used to trip, even though the documented QC
    # limit is 250,000.  Drive the actual unwrap coordinates of a dense
    # synthetic tile through the overlap gate.
    spec = replace(
        synthetic_tile_spec_from_preset("sugkiwa_quarter", seed=5),
        axial_samples=175,
        angular_samples=175,
    )
    artifact = generate_synthetic_tile(spec)
    vertices = np.asarray(artifact.mesh.vertices, dtype=np.float64)
    faces = np.asarray(artifact.mesh.faces, dtype=np.int32)
    assert faces.shape[0] > 55_000

    unwrapped, _meta = sectionwise_cylindrical_parameterization(
        cast(MeshData, _PlainMesh(vertices, faces)),
        axis="y",
        n_sections=32,
        record_view="top",
        return_meta=True,
    )
    uv_mm = np.asarray(unwrapped, dtype=np.float64)
    uv_mm = uv_mm - np.min(uv_mm, axis=0, keepdims=True)
    uv_um = np.rint(uv_mm * 1000.0).astype(np.int64)

    assert _uv_overlap_pair_count(uv_um, faces, cancellation_probe=None) == 0


def test_global_uv_overlap_budget_still_bounds_exact_overlap_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Three quads, each split into two edge-sharing triangles.  The two halves
    # of a quad share a bounding box, so each quad costs exactly one exact
    # overlap evaluation while never actually overlapping.  The budget must
    # still fire on that real work.
    vertices: list[list[int]] = []
    faces: list[list[int]] = []
    for quad_index in range(3):
        base = quad_index * 4
        left = quad_index * 100
        vertices.extend(
            [
                [left, 0],
                [left + 10, 0],
                [left, 10],
                [left + 10, 10],
            ]
        )
        faces.append([base + 0, base + 1, base + 2])
        faces.append([base + 1, base + 3, base + 2])
    uv_um = np.asarray(vertices, dtype=np.int64)
    face_array = np.asarray(faces, dtype=np.int32)

    # Without a budget the surface is clean: shared edges are not overlap.
    assert (
        _uv_overlap_pair_count(uv_um, face_array, cancellation_probe=None) == 0
    )

    monkeypatch.setattr(
        "src.core.artifact_tile_unwrap_extractor.MAX_TILE_UNWRAP_OVERLAP_CANDIDATES",
        2,
    )
    with pytest.raises(ArtifactTileUnwrapError, match="examined-pair budget"):
        _uv_overlap_pair_count(uv_um, face_array, cancellation_probe=None)


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
