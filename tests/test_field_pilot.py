from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from src import build_info
from src.application.artifact_workflow_self_test import _run_in_directory
from src.core.canonical_json import canonical_json_bytes, canonical_json_sha256
from src.core.field_pilot import (
    FIELD_PILOT_CHECKS,
    FieldPilotError,
    build_field_pilot_report,
    build_field_pilot_verification_report,
    default_field_pilot_review,
    load_field_pilot_report,
    load_field_pilot_review,
    validate_field_pilot_report,
    validate_field_pilot_review,
    write_field_pilot_report,
    write_field_pilot_review_template,
)


ROOT = Path(__file__).resolve().parents[1]
STAMP = "2026-07-14T00:00:00Z"


def _machine(*, system: str) -> dict[str, Any]:
    return {
        "frozen": system == "Windows",
        "logical_cpu_count": 8,
        "machine": "AMD64" if system == "Windows" else "x86_64",
        "peak_working_set_bytes": 256 * 1024 * 1024,
        "process_bits": 64,
        "python_version": "3.12.10",
        "release": "11" if system == "Windows" else "test-kernel",
        "system": system,
        "total_physical_memory_bytes": 16 * 1024 * 1024 * 1024,
    }


def _passing_review(
    *,
    project_document_sha256: str = "d" * 64,
    survey_artifact_set_sha256: str = "e" * 64,
) -> dict[str, Any]:
    review = default_field_pilot_review()
    review.update(
        {
            "artifact_label": "registered-tile-001",
            "project_document_sha256": project_document_sha256,
            "reviewed_at_utc": STAMP,
            "reviewer_id": "archaeologist-01",
            "survey_artifact_set_sha256": survey_artifact_set_sha256,
        }
    )
    review["checks"] = {name: "pass" for name in FIELD_PILOT_CHECKS}
    review["measurements"] = {
        "scale_expected_mm": 100.0,
        "scale_observed_mm": 100.1,
        "scale_tolerance_mm": 0.2,
        "workflow_elapsed_minutes": 18.5,
    }
    return review


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def _passing_opengl_report() -> dict[str, Any]:
    metadata = build_info.build_metadata()
    return {
        "checks": [
            {"id": "qt.native_platform", "ok": True},
            {"id": "driver.depth_bits", "ok": True},
        ],
        "cleanup_errors": [],
        "context": {
            "depth_bits": 24,
            "qt_platform": "windows",
            "renderer": "Mesa D3D12 test renderer",
            "software_renderer": True,
            "vendor": "Mesa",
            "version": "4.1 compatibility",
        },
        "ok": True,
        "render_modes": [{"mode": "perspective"}, {"mode": "top_orthographic"}],
        "schema": "archmeshrubbing.opengl_driver_smoke",
        "schema_version": 1,
        "source": {
            "commit": metadata["commit"],
            "runtime_lock_sha256": metadata["dependency_lock_sha256"],
        },
        "tested_at_utc": STAMP,
    }


@pytest.fixture(scope="module")
def completed_artifacts(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, Path]:
    directory = tmp_path_factory.mktemp("field-pilot-artifacts")
    _run_in_directory(directory)
    return (
        directory / "workflow-fixture-recovered.amr",
        directory / "relocated-complete-workflow.amr-survey",
    )


@pytest.fixture(scope="module")
def incomplete_report(
    completed_artifacts: tuple[Path, Path],
) -> dict[str, Any]:
    project, survey = completed_artifacts
    return build_field_pilot_report(
        project,
        survey,
        created_at_utc=STAMP,
        machine_snapshot=_machine(system="Linux"),
    )


def _schema(name: str) -> dict[str, Any]:
    return json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8"))


def _rehash_report(report: dict[str, Any]) -> None:
    unsigned = dict(report)
    unsigned.pop("pilot_sha256")
    report["pilot_sha256"] = canonical_json_sha256(unsigned)


def test_review_template_is_schema_valid_and_cannot_accidentally_pass() -> None:
    jsonschema = importlib.import_module("jsonschema")
    schema = _schema("field_pilot_review-1.0.0.schema.json")
    jsonschema.Draft202012Validator.check_schema(schema)

    review = validate_field_pilot_review(default_field_pilot_review())

    assert list(jsonschema.Draft202012Validator(schema).iter_errors(review)) == []
    assert set(review["checks"]) == set(FIELD_PILOT_CHECKS)
    assert set(review["checks"].values()) == {"not_tested"}
    assert review["reviewed_at_utc"] is None


def test_review_cross_fields_fail_closed() -> None:
    review = _passing_review()
    review["measurements"]["scale_observed_mm"] = 101.0
    with pytest.raises(FieldPilotError, match="contradicts"):
        validate_field_pilot_review(review)

    review = _passing_review()
    review["measurements"]["workflow_elapsed_minutes"] = None
    with pytest.raises(FieldPilotError, match="elapsed minutes"):
        validate_field_pilot_review(review)

    review = default_field_pilot_review()
    review["measurements"]["scale_expected_mm"] = 100.0
    with pytest.raises(FieldPilotError, match="must be null"):
        validate_field_pilot_review(review)

    review = default_field_pilot_review()
    review["project_document_sha256"] = "d" * 64
    with pytest.raises(FieldPilotError, match="both be supplied"):
        validate_field_pilot_review(review)


def test_review_loader_rejects_non_finite_json(tmp_path: Path) -> None:
    path = tmp_path / "review.json"
    path.write_text(
        json.dumps(default_field_pilot_review()).replace(
            '"workflow_elapsed_minutes": null',
            '"workflow_elapsed_minutes": NaN',
        ),
        encoding="utf-8",
    )

    with pytest.raises(FieldPilotError, match="strict UTF-8 JSON"):
        load_field_pilot_review(path)


def test_review_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "review.json"
    payload = canonical_json_bytes(default_field_pilot_review()).decode("utf-8")
    payload = payload.replace(
        '"artifact_label":"replace-with-local-artifact-label"',
        '"artifact_label":"first","artifact_label":"second"',
        1,
    )
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(FieldPilotError, match="strict UTF-8 JSON"):
        load_field_pilot_review(path)


def test_real_workflow_pair_builds_an_honest_incomplete_report(
    incomplete_report: dict[str, Any],
) -> None:
    report = incomplete_report

    assert report["outcome"] == {
        "artifact_verification": "pass",
        "human_review": "incomplete",
        "opengl_driver": "not_provided",
        "pilot": "incomplete",
        "scope": "single_artifact_single_machine",
        "windows_runtime": "not_target",
    }
    assert report["project_verification"]["ok"] is True
    assert report["survey_verification"]["ok"] is True
    assert report["authentication"] == {
        "kind": "none",
        "signature_present": False,
    }
    assert report["release_claim"] == "single_pilot_only_not_release_approval"
    assert report["inputs"] == {
        "opengl_report_name": None,
        "project_name": "workflow-fixture-recovered.amr",
        "review_name": None,
        "survey_name": "relocated-complete-workflow.amr-survey",
    }


def test_complete_windows_evidence_can_verify_one_pilot_only(
    completed_artifacts: tuple[Path, Path],
    incomplete_report: dict[str, Any],
    tmp_path: Path,
) -> None:
    project, survey = completed_artifacts
    review_path = tmp_path / "archaeologist-review.json"
    opengl_path = tmp_path / "windows-opengl.json"
    project_digest = incomplete_report["project_verification"]["evidence"][
        "document_sha256"
    ]
    survey_digest = incomplete_report["survey_verification"]["evidence"][
        "artifact_set_sha256"
    ]
    _write_json(
        review_path,
        _passing_review(
            project_document_sha256=project_digest,
            survey_artifact_set_sha256=survey_digest,
        ),
    )
    _write_json(opengl_path, _passing_opengl_report())

    report = build_field_pilot_report(
        project,
        survey,
        review=review_path,
        opengl_report=opengl_path,
        created_at_utc=STAMP,
        machine_snapshot=_machine(system="Windows"),
    )

    assert report["outcome"] == {
        "artifact_verification": "pass",
        "human_review": "pass",
        "opengl_driver": "pass",
        "pilot": "verified",
        "scope": "single_artifact_single_machine",
        "windows_runtime": "pass",
    }
    assert report["opengl_driver"]["check_count"] == 2
    assert report["opengl_driver"]["input_name"] == opengl_path.name
    assert report["review"]["reviewer_id"] == "archaeologist-01"
    assert validate_field_pilot_report(report) == report


def test_review_bound_to_another_artifact_fails_the_pilot(
    completed_artifacts: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    project, survey = completed_artifacts
    review_path = tmp_path / "wrong-artifact-review.json"
    _write_json(review_path, _passing_review())

    report = build_field_pilot_report(
        project,
        survey,
        review=review_path,
        created_at_utc=STAMP,
        machine_snapshot=_machine(system="Windows"),
    )

    assert report["outcome"]["artifact_verification"] == "pass"
    assert report["outcome"]["human_review"] == "fail"
    assert report["outcome"]["pilot"] == "failed"


def test_report_and_verification_receipt_match_public_schemas(
    incomplete_report: dict[str, Any],
    tmp_path: Path,
) -> None:
    jsonschema = importlib.import_module("jsonschema")
    referencing = importlib.import_module("referencing")
    report_schema = _schema("field_pilot_report-1.0.0.schema.json")
    review_schema = _schema("field_pilot_review-1.0.0.schema.json")
    offline_schema = _schema("offline_verification_report-1.0.0.schema.json")
    receipt_schema = _schema("field_pilot_verification-1.0.0.schema.json")
    jsonschema.Draft202012Validator.check_schema(report_schema)
    jsonschema.Draft202012Validator.check_schema(review_schema)
    jsonschema.Draft202012Validator.check_schema(offline_schema)
    jsonschema.Draft202012Validator.check_schema(receipt_schema)
    registry = referencing.Registry()
    for schema in (review_schema, offline_schema):
        registry = registry.with_resource(
            schema["$id"],
            referencing.Resource.from_contents(schema),
        )
    report_validator = jsonschema.Draft202012Validator(
        report_schema,
        registry=registry,
    )
    receipt_validator = jsonschema.Draft202012Validator(receipt_schema)

    report_path = tmp_path / "pilot.json"
    write_field_pilot_report(report_path, incomplete_report)
    receipt = build_field_pilot_verification_report(report_path)
    failed_receipt = build_field_pilot_verification_report(
        tmp_path / "missing-pilot.json"
    )

    assert list(report_validator.iter_errors(incomplete_report)) == []
    assert list(receipt_validator.iter_errors(receipt)) == []
    assert list(receipt_validator.iter_errors(failed_receipt)) == []
    assert receipt["ok"] is True
    assert receipt["evidence"]["pilot_outcome"] == "incomplete"
    assert receipt["evidence"]["pilot_sha256"] == incomplete_report["pilot_sha256"]
    assert failed_receipt["ok"] is False
    assert str(tmp_path) not in failed_receipt["error"]["message"]


def test_report_publication_is_canonical_atomic_and_no_overwrite(
    incomplete_report: dict[str, Any],
    tmp_path: Path,
) -> None:
    destination = tmp_path / "pilot.json"
    publication = write_field_pilot_report(destination, incomplete_report)

    assert publication.path == destination
    assert load_field_pilot_report(destination) == incomplete_report
    assert destination.read_bytes() == canonical_json_bytes(incomplete_report) + b"\n"
    assert not list(tmp_path.glob(".amr-field-pilot-*.tmp"))
    with pytest.raises(FieldPilotError, match="already exists"):
        write_field_pilot_report(destination, incomplete_report)


def test_report_publication_has_one_concurrent_winner(
    incomplete_report: dict[str, Any],
    tmp_path: Path,
) -> None:
    destination = tmp_path / "pilot.json"

    def publish() -> str:
        try:
            write_field_pilot_report(destination, incomplete_report)
        except FieldPilotError:
            return "lost"
        return "won"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _index: publish(), range(2)))

    assert outcomes.count("won") == 1
    assert outcomes.count("lost") == 1
    assert load_field_pilot_report(destination) == incomplete_report
    assert not list(tmp_path.glob(".amr-field-pilot-*.tmp"))


def test_loader_rejects_tampering_and_noncanonical_bytes(
    incomplete_report: dict[str, Any],
    tmp_path: Path,
) -> None:
    tampered = json.loads(json.dumps(incomplete_report))
    tampered["application"]["name"] = "Modified"
    tampered_path = tmp_path / "tampered.json"
    tampered_path.write_bytes(canonical_json_bytes(tampered) + b"\n")
    with pytest.raises(FieldPilotError, match="pilot_sha256"):
        load_field_pilot_report(tampered_path)

    pretty_path = tmp_path / "pretty.json"
    pretty_path.write_text(
        json.dumps(incomplete_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(FieldPilotError, match="not canonical"):
        load_field_pilot_report(pretty_path)


def test_semantic_validation_rejects_rehashed_nested_receipt_extensions(
    incomplete_report: dict[str, Any],
) -> None:
    modified = json.loads(json.dumps(incomplete_report))
    modified["project_verification"]["unexpected"] = True
    _rehash_report(modified)

    with pytest.raises(FieldPilotError, match="fields are not closed"):
        validate_field_pilot_report(modified)


def test_semantic_validation_rejects_rehashed_automatic_absolute_paths(
    incomplete_report: dict[str, Any],
) -> None:
    modified = json.loads(json.dumps(incomplete_report))
    modified["project_verification"]["evidence"]["source"][
        "original_name"
    ] = "/Users/researcher/private/artifact.ply"
    _rehash_report(modified)

    with pytest.raises(FieldPilotError, match="absolute path"):
        validate_field_pilot_report(modified)


def test_invalid_input_report_fails_without_disclosing_absolute_paths(
    tmp_path: Path,
) -> None:
    secret_directory = tmp_path / "sensitive-collection-name"
    project = secret_directory / "missing.amr"
    survey = secret_directory / "missing.amr-survey"

    report = build_field_pilot_report(
        project,
        survey,
        created_at_utc=STAMP,
        machine_snapshot=_machine(system="Windows"),
    )
    encoded = canonical_json_bytes(report).decode("utf-8")

    assert report["outcome"]["artifact_verification"] == "fail"
    assert report["outcome"]["pilot"] == "failed"
    assert str(secret_directory) not in encoded
    assert report["inputs"]["project_name"] == "missing.amr"
    assert report["inputs"]["survey_name"] == "missing.amr-survey"


def test_cross_platform_input_names_never_retain_parent_components() -> None:
    report = build_field_pilot_report(
        r"C:\sensitive-collection\missing.amr",
        r"C:\sensitive-collection\missing.amr-survey",
        created_at_utc=STAMP,
        machine_snapshot=_machine(system="Windows"),
    )
    encoded = canonical_json_bytes(report).decode("utf-8")

    assert "sensitive-collection" not in encoded
    assert report["inputs"]["project_name"] == "missing.amr"
    assert report["inputs"]["survey_name"] == "missing.amr-survey"
    assert report["project_verification"]["input_name"] == "missing.amr"
    assert report["survey_verification"]["input_name"] == "missing.amr-survey"


def test_review_template_writer_is_canonical_and_no_overwrite(
    tmp_path: Path,
) -> None:
    path = tmp_path / "review.json"

    write_field_pilot_review_template(path)
    review, input_sha256 = load_field_pilot_review(path)

    assert review == default_field_pilot_review()
    assert len(input_sha256) == 64
    assert path.read_bytes() == canonical_json_bytes(review) + b"\n"
    with pytest.raises(FieldPilotError, match="already exists"):
        write_field_pilot_review_template(path)


def test_public_cli_publishes_incomplete_report_then_verifies_its_integrity(
    completed_artifacts: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    project, survey = completed_artifacts
    report_path = tmp_path / "cli-pilot.json"
    receipt_path = tmp_path / "cli-verification.json"

    pilot = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--field-pilot",
            str(project),
            str(survey),
            "--report",
            str(report_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert pilot.returncode == 1, pilot.stderr
    report = load_field_pilot_report(report_path)
    assert report["outcome"]["pilot"] == "incomplete"
    assert json.loads(pilot.stdout)["pilot_sha256"] == report["pilot_sha256"]

    verification = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--verify-field-pilot",
            str(report_path),
            "--report",
            str(receipt_path),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert verification.returncode == 0, verification.stderr
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["ok"] is True
    assert receipt["evidence"]["pilot_outcome"] == "incomplete"
