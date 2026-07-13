from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest

import src.core.artifact_tile_unwrap_export as tile_export
from src.application.artifact_measurements import (
    ArtifactMeasurementController,
    MeasurementOperationKind,
    MeasurementOperationState,
)
from src.application.artifact_workbench import (
    ArtifactWorkbench,
    RecordBindingTransition,
)
from src.core.artifact_session import ArtifactSession
from src.core.artifact_tile_unwrap_export import (
    ArtifactTileUnwrapExportError,
    TILE_UNWRAP_EXPORT_SIDECAR_NAME,
    build_tile_unwrap_export,
    discard_prepared_tile_unwrap_package,
    export_tile_unwrap_package,
    prepare_staged_tile_unwrap_publication,
    publish_prepared_tile_unwrap_package,
    stage_tile_unwrap_package,
    validate_tile_unwrap_export_bytes,
    validate_tile_unwrap_export_package,
)
from src.core.artifact_tile_unwrap_extractor import (
    ArtifactTileUnwrapComputation,
    TileUnwrapMesh,
    commit_artifact_tile_unwrap,
    compute_artifact_tile_unwrap,
)
from src.core.artifact_tile_unwrap_record import tile_unwrap_receipt_from_record
from src.core.mesh_import_recipe import current_mesh_import_recipe
from src.core.mesh_loader import MeshData
from src.core.source_identity import SourceFingerprint
from src.core.tile_synthetic import (
    generate_synthetic_tile,
    synthetic_tile_spec_from_preset,
)


STAMP = "2026-07-13T00:00:00Z"
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _confirmed_tile_unwrap_directory_fsync():
    with patch.object(tile_export, "fsync_export_directory", return_value=True):
        yield


class _SequentialIds:
    def __init__(self) -> None:
        self.value = 0

    def __call__(self, prefix: str) -> str:
        self.value += 1
        return f"{prefix}:tile-test-{self.value}"


def _recorded() -> tuple[ArtifactSession, ArtifactTileUnwrapComputation]:
    synthetic = generate_synthetic_tile(
        synthetic_tile_spec_from_preset("sugkiwa_quarter", seed=17)
    )
    mesh = MeshData(
        vertices=np.asarray(synthetic.mesh.vertices, dtype=np.float64),
        faces=np.asarray(synthetic.mesh.faces, dtype=np.int32),
        unit="mm",
        filepath=Path("/private/scans/tile.ply"),
        source_identity=SourceFingerprint(
            sha256="f" * 64,
            size_bytes=777_777,
            mtime_ns=1,
            original_name="tile.ply",
            format="ply",
        ),
        source_format="ply",
        source_import_recipe=current_mesh_import_recipe("ply"),
    )
    session = ArtifactSession.create_from_source(
        mesh,
        resolved_source_path="/private/scans/tile.ply",
        unit="mm",
        axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
        handedness="right",
        software_version="0.7-test",
        operator="pytest",
        created_at=STAMP,
        document_id="artifact:tile-export",
        metadata_revision_id="metadata:initial",
        align_revision_id="align:initial",
    ).commit_preview(
        translation_mm=(0.0, 0.0, 0.0),
        rotation_deg=(0.0, 0.0, 0.0),
        scale=1.0,
        pivot_mm=(0.0, 0.0, 0.0),
        operator="pytest",
        created_at="2026-07-13T00:00:01Z",
        revision_id="align:confirmed",
    )
    computation = compute_artifact_tile_unwrap(
        session,
        longitudinal_axis="y",
        record_view="top",
        n_sections=32,
    )
    committed = commit_artifact_tile_unwrap(
        session,
        computation,
        record_id="record:tile-export",
        created_at="2026-07-13T00:00:02Z",
        operator="pytest",
    )
    return committed, computation


def test_bundle_is_deterministic_portable_and_exactly_one_to_one() -> None:
    session, computation = _recorded()

    first = build_tile_unwrap_export(
        session.document, "record:tile-export", computation.unwrap
    )
    second = build_tile_unwrap_export(
        session.document, "record:tile-export", computation.unwrap
    )
    sidecar = validate_tile_unwrap_export_bytes(
        first.payload_bytes,
        first.obj_bytes,
        first.svg_bytes,
        first.sidecar_bytes,
    )
    record = session.document.record_index["record:tile-export"]
    receipt = tile_unwrap_receipt_from_record(record)

    assert first == second
    assert hashlib.sha256(first.payload_bytes).hexdigest() == receipt["unwrap_sha256"]
    assert b'physical_scale":"1:1"' in first.sidecar_bytes
    assert b'width="126.478mm"' in first.svg_bytes
    assert b'height="166.266mm"' in first.svg_bytes
    assert b"/private/scans" not in first.sidecar_bytes
    assert sidecar["presentation"]["boundary_loop_count"] == 1


def test_application_controller_computes_commits_and_exports_tile_unwrap() -> None:
    session, _existing = _recorded()
    workbench = ArtifactWorkbench(session=session, id_factory=_SequentialIds())
    controller = ArtifactMeasurementController(
        workbench,
        id_factory=_SequentialIds(),
    )
    item = controller.begin_tile_unwrap(
        longitudinal_axis="y",
        record_view="bottom",
        n_sections=24,
        record_id="record:tile-controller",
        created_at="2026-07-13T00:00:03Z",
        operator="pytest",
    )

    result = controller.execute(item)

    assert item.kind is MeasurementOperationKind.TILE_UNWRAP
    assert isinstance(result.computation, ArtifactTileUnwrapComputation)
    assert "record:tile-controller" not in session.document.record_index

    def publish(transition: RecordBindingTransition) -> None:
        activation = workbench.activate_record_binding(transition)
        workbench.finalize_record_binding(activation)

    publication = controller.publish_result(item, result, publish)

    assert publication.record_id == "record:tile-controller"
    assert controller.summary(item).state is MeasurementOperationState.COMPLETED
    bundle = build_tile_unwrap_export(
        publication.session.document,
        publication.record_id,
        result.computation.unwrap,
    )
    validate_tile_unwrap_export_bytes(
        bundle.payload_bytes,
        bundle.obj_bytes,
        bundle.svg_bytes,
        bundle.sidecar_bytes,
        document=publication.session.document,
    )


def test_generated_sidecar_matches_closed_public_json_schema() -> None:
    jsonschema = importlib.import_module("jsonschema")
    referencing = importlib.import_module("referencing")

    def load_schema(name: str) -> dict[str, object]:
        value = json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8"))
        assert isinstance(value, dict)
        return value

    export_schema = load_schema("tile_unwrap_export-1.0.0.schema.json")
    receipt_schema = load_schema("tile_unwrap_receipt-1.0.0.schema.json")
    rubbing_export_schema = load_schema("rubbing_export-1.0.0.schema.json")
    import_recipe_schema = load_schema("mesh_import_recipe-1.0.0.schema.json")
    import_recipe_v2_schema = load_schema("mesh_import_recipe-2.0.0.schema.json")
    for schema in (
        export_schema,
        receipt_schema,
        rubbing_export_schema,
        import_recipe_schema,
        import_recipe_v2_schema,
    ):
        jsonschema.Draft202012Validator.check_schema(schema)

    registry = referencing.Registry()
    for schema in (
        receipt_schema,
        rubbing_export_schema,
        import_recipe_schema,
        import_recipe_v2_schema,
    ):
        schema_id = schema["$id"]
        assert isinstance(schema_id, str)
        registry = registry.with_resource(
            schema_id,
            referencing.Resource.from_contents(schema),
        )

    validator = jsonschema.Draft202012Validator(
        export_schema,
        registry=registry,
    )
    session, computation = _recorded()
    bundle = build_tile_unwrap_export(
        session.document,
        "record:tile-export",
        computation.unwrap,
    )
    sidecar = json.loads(bundle.sidecar_bytes)
    assert isinstance(sidecar, dict)
    assert list(validator.iter_errors(sidecar)) == []

    tampered = copy.deepcopy(sidecar)
    recipe = tampered["recipe"]
    assert isinstance(recipe, dict)
    recipe["longitudinal_axis"] = "auto"
    assert list(validator.iter_errors(tampered))


def test_canonical_payload_roundtrips_and_rejects_trailing_bytes() -> None:
    session, computation = _recorded()
    record = session.document.record_index["record:tile-export"]
    receipt = tile_unwrap_receipt_from_record(record)
    payload = computation.unwrap.canonical_payload_bytes(
        selection_sha256=receipt["selection_sha256"]
    )

    parsed, header = TileUnwrapMesh.from_canonical_payload_bytes(
        payload,
        expected_selection_sha256=receipt["selection_sha256"],
    )

    assert parsed.receipt(selection_sha256=header["selection_sha256"]) == receipt
    with pytest.raises(ValueError, match="trailing bytes"):
        TileUnwrapMesh.from_canonical_payload_bytes(payload + b"x")


@pytest.mark.parametrize("artifact", ["payload", "obj", "svg", "sidecar"])
def test_offline_verifier_rejects_every_artifact_tamper(artifact: str) -> None:
    session, computation = _recorded()
    bundle = build_tile_unwrap_export(
        session.document, "record:tile-export", computation.unwrap
    )
    values = [
        bundle.payload_bytes,
        bundle.obj_bytes,
        bundle.svg_bytes,
        bundle.sidecar_bytes,
    ]
    index = {"payload": 0, "obj": 1, "svg": 2, "sidecar": 3}[artifact]
    damaged = bytearray(values[index])
    damaged[max(0, len(damaged) // 2)] ^= 1
    values[index] = bytes(damaged)

    with pytest.raises(ArtifactTileUnwrapExportError):
        validate_tile_unwrap_export_bytes(*values)


def test_package_publishes_without_overwrite_and_verifies_offline(
    tmp_path: Path,
) -> None:
    session, computation = _recorded()
    destination = tmp_path / "roof-tile.amr-unwrap"

    publication = export_tile_unwrap_package(
        destination,
        session.document,
        "record:tile-export",
        computation.unwrap,
    )

    assert publication.destination == destination
    assert destination.is_dir()
    assert TILE_UNWRAP_EXPORT_SIDECAR_NAME in {
        entry.name for entry in destination.iterdir()
    }
    validate_tile_unwrap_export_package(destination)
    validate_tile_unwrap_export_package(destination, document=session.document)
    with pytest.raises(ArtifactTileUnwrapExportError, match="already exists"):
        export_tile_unwrap_package(
            destination,
            session.document,
            "record:tile-export",
            computation.unwrap,
        )


def test_staged_package_is_hidden_until_exact_prepared_capability_publishes(
    tmp_path: Path,
) -> None:
    session, computation = _recorded()
    destination = tmp_path / "staged.amr-unwrap"

    stage = stage_tile_unwrap_package(
        destination,
        session.document,
        "record:tile-export",
        computation.unwrap,
    )

    assert stage.is_dir()
    assert not destination.exists()
    validate_tile_unwrap_export_package(stage, document=session.document)
    prepared = prepare_staged_tile_unwrap_publication(
        stage,
        destination,
        document=session.document,
    )
    published = publish_prepared_tile_unwrap_package(prepared)

    assert published == destination
    assert destination.is_dir()
    assert not stage.exists()
    validate_tile_unwrap_export_package(destination, document=session.document)
    with pytest.raises(
        ArtifactTileUnwrapExportError,
        match="already visible|invalid or consumed",
    ):
        publish_prepared_tile_unwrap_package(prepared)


def test_prepared_package_rejects_tamper_and_owned_stage_can_be_discarded(
    tmp_path: Path,
) -> None:
    session, computation = _recorded()
    destination = tmp_path / "tamper.amr-unwrap"
    stage = stage_tile_unwrap_package(
        destination,
        session.document,
        "record:tile-export",
        computation.unwrap,
    )
    prepared = prepare_staged_tile_unwrap_publication(
        stage,
        destination,
        document=session.document,
    )
    sidecar = stage / TILE_UNWRAP_EXPORT_SIDECAR_NAME
    sidecar.write_bytes(sidecar.read_bytes() + b"\n")

    with pytest.raises(ArtifactTileUnwrapExportError, match="changed after preparation"):
        publish_prepared_tile_unwrap_package(prepared)

    assert discard_prepared_tile_unwrap_package(prepared) is True
    assert not stage.exists()
    assert not destination.exists()


def test_relocated_package_validates_in_independent_offline_process(
    tmp_path: Path,
) -> None:
    session, computation = _recorded()
    package = tmp_path / "source.amr-unwrap"
    export_tile_unwrap_package(
        package,
        session.document,
        "record:tile-export",
        computation.unwrap,
    )
    relocated = tmp_path / "이동된 기와 기록.amr-unwrap"
    package.rename(relocated)
    project_root = str(ROOT)
    script = (
        "import json,sys;sys.path.insert(0,sys.argv[2]);"
        "from src.core.artifact_tile_unwrap_export import "
        "validate_tile_unwrap_export_package;"
        "s=validate_tile_unwrap_export_package(sys.argv[1]);"
        "print(json.dumps({'unwrap':s['geometry']['unwrap_sha256'],"
        "'scale':s['presentation']['physical_scale']},sort_keys=True))"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, str(relocated), project_root],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    verified = json.loads(completed.stdout)
    receipt = computation.unwrap.receipt(
        selection_sha256=computation.context.selection_hash or ""
    )
    assert verified == {
        "scale": "1:1",
        "unwrap": receipt["unwrap_sha256"],
    }


def test_package_rejects_extra_entries_and_symlinks(tmp_path: Path) -> None:
    session, computation = _recorded()
    destination = tmp_path / "roof-tile.amr-unwrap"
    export_tile_unwrap_package(
        destination,
        session.document,
        "record:tile-export",
        computation.unwrap,
    )
    (destination / "extra.txt").write_text("unexpected", encoding="utf-8")

    with pytest.raises(ArtifactTileUnwrapExportError, match="closed contract"):
        validate_tile_unwrap_export_package(destination)


def test_publish_race_preserves_winner_and_removes_owned_stage(
    tmp_path: Path,
) -> None:
    session, computation = _recorded()
    destination = tmp_path / "raced.amr-unwrap"

    def winning_race(_stage: Path, target: Path) -> None:
        target.mkdir()
        (target / "winner.txt").write_text("other process", encoding="utf-8")
        raise tile_export.ArtifactVectorExportError("export destination already exists")

    with patch.object(
        tile_export,
        "publish_export_directory_noreplace",
        side_effect=winning_race,
    ):
        with pytest.raises(ArtifactTileUnwrapExportError, match="already exists"):
            export_tile_unwrap_package(
                destination,
                session.document,
                "record:tile-export",
                computation.unwrap,
            )

    assert (destination / "winner.txt").read_text(encoding="utf-8") == "other process"
    assert not list(tmp_path.glob(".*.staging-*"))


def test_stale_align_record_cannot_export() -> None:
    session, computation = _recorded()
    changed = session.commit_preview(
        translation_mm=(1.0, 0.0, 0.0),
        rotation_deg=(0.0, 0.0, 0.0),
        scale=1.0,
        pivot_mm=(0.0, 0.0, 0.0),
        operator="pytest",
        created_at="2026-07-13T00:00:03Z",
        revision_id="align:changed",
    )

    with pytest.raises(ArtifactTileUnwrapExportError, match="FRESH"):
        build_tile_unwrap_export(
            changed.document,
            "record:tile-export",
            computation.unwrap,
        )
