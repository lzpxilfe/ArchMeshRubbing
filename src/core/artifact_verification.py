"""Unified, deterministic offline verification receipts for public artifacts.

The format-specific validators remain the authority for bytes, physical scale,
recipes, QC, and provenance.  This module only detects a public artifact,
invokes the corresponding validator, and returns one small machine-readable
receipt suitable for the frozen command-line application.

No timestamp or absolute input path is included, so a successful receipt for
the same artifact and authority mode has the same JSON value offline.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping

from .artifact_document import ArtifactDocument, RecordLifecycleStatus
from .artifact_rubbing_export import (
    RUBBING_EXPORT_SIDECAR_NAME,
    validate_rubbing_export_package,
)
from .artifact_session import ArtifactSession
from .artifact_survey_export import (
    SURVEY_EXPORT_MANIFEST_NAME,
    validate_survey_export_package,
)
from .artifact_tile_unwrap_export import (
    TILE_UNWRAP_EXPORT_SIDECAR_NAME,
    validate_tile_unwrap_export_package,
)
from .artifact_vector_export import (
    VECTOR_EXPORT_SIDECAR_NAME,
    validate_vector_export_package,
)
from .canonical_json import canonical_json_bytes
from .project_file import load_artifact_session_project


OFFLINE_VERIFICATION_FORMAT = "archmeshrubbing_offline_verification"
OFFLINE_VERIFICATION_SCHEMA_VERSION = "1.0.0"

ARTIFACT_KIND_PROJECT = "project"
ARTIFACT_KIND_VECTOR_EXPORT = "vector_export"
ARTIFACT_KIND_RUBBING_EXPORT = "rubbing_export"
ARTIFACT_KIND_SURVEY_EXPORT = "survey_export"
ARTIFACT_KIND_TILE_UNWRAP_EXPORT = "tile_unwrap_export"
ARTIFACT_KIND_UNKNOWN = "unknown"

AUTHORITY_SELF_CONTAINED = "self_contained"
AUTHORITY_MATCHED_PROJECT = "matched_project"

_DIRECTORY_MARKERS = {
    VECTOR_EXPORT_SIDECAR_NAME: ARTIFACT_KIND_VECTOR_EXPORT,
    RUBBING_EXPORT_SIDECAR_NAME: ARTIFACT_KIND_RUBBING_EXPORT,
    SURVEY_EXPORT_MANIFEST_NAME: ARTIFACT_KIND_SURVEY_EXPORT,
    TILE_UNWRAP_EXPORT_SIDECAR_NAME: ARTIFACT_KIND_TILE_UNWRAP_EXPORT,
}
_WINDOWS_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/]|\\\\)[^\r\n\t\"']+"
)
_POSIX_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9:])/(?:[^/\s\"']+/)*[^/\s\"']*"
)


class ArtifactVerificationError(ValueError):
    """A stable preflight failure before a format-specific validator runs."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        artifact_kind: str = ARTIFACT_KIND_UNKNOWN,
    ) -> None:
        super().__init__(message)
        self.code = str(code)
        self.artifact_kind = str(artifact_kind)


def _safe_input_name(path: str | os.PathLike[str]) -> str:
    try:
        name = Path(os.fspath(path)).name
    except (TypeError, ValueError, OSError):
        return "invalid-input"
    if not name:
        name = "current-directory"
    safe = "".join(
        "_" if character in "/\\" or ord(character) < 32 or ord(character) == 127
        else character
        for character in name
    )
    return safe[:255] or "unnamed-input"


def _report_base(
    *,
    input_name: str,
    artifact_kind: str,
    authority: str,
    ok: bool,
) -> dict[str, Any]:
    return {
        "artifact_kind": artifact_kind,
        "authority": authority,
        "format": OFFLINE_VERIFICATION_FORMAT,
        "input_name": input_name,
        "ok": bool(ok),
        "schema_version": OFFLINE_VERIFICATION_SCHEMA_VERSION,
    }


def _redacted_error_message(exc: Exception, paths: tuple[Path, ...]) -> str:
    text = f"{type(exc).__name__}: {exc}"
    replacements: set[str] = set()
    for path in paths:
        for candidate in (
            os.fspath(path),
            os.path.abspath(os.fspath(path)),
            str(path.expanduser()),
            str(path.expanduser().resolve(strict=False)),
        ):
            if len(candidate) > 1:
                replacements.add(candidate)
                replacements.add(candidate.replace("\\", "/"))
    for candidate in sorted(replacements, key=len, reverse=True):
        text = text.replace(candidate, "<path>")
    text = _WINDOWS_ABSOLUTE_PATH_RE.sub("<path>", text)
    text = _POSIX_ABSOLUTE_PATH_RE.sub("<path>", text)
    text = " ".join(text.split())
    return (text or "verification failed")[:1024]


def _failure_report(
    *,
    input_name: str,
    artifact_kind: str,
    authority: str,
    code: str,
    message: str,
) -> dict[str, Any]:
    report = _report_base(
        input_name=input_name,
        artifact_kind=artifact_kind,
        authority=authority,
        ok=False,
    )
    report["error"] = {"code": code, "message": message}
    return report


def _path_identity(path: Path, *, prefix: str) -> os.stat_result:
    try:
        identity = path.stat(follow_symlinks=False)
    except FileNotFoundError as exc:
        raise ArtifactVerificationError(
            f"{prefix}_missing",
            "input does not exist" if prefix == "input" else "authority project does not exist",
        ) from exc
    except OSError as exc:
        raise ArtifactVerificationError(
            f"{prefix}_unreadable",
            "input cannot be inspected"
            if prefix == "input"
            else "authority project cannot be inspected",
        ) from exc
    if stat.S_ISLNK(identity.st_mode):
        raise ArtifactVerificationError(
            f"{prefix}_symlink",
            "symbolic-link inputs are not accepted",
        )
    return identity


def _detect_artifact(path: Path) -> str:
    identity = _path_identity(path, prefix="input")
    if stat.S_ISREG(identity.st_mode):
        if path.suffix.lower() == ".amr":
            return ARTIFACT_KIND_PROJECT
        raise ArtifactVerificationError(
            "input_unsupported",
            "regular-file input must be an .amr project",
        )
    if not stat.S_ISDIR(identity.st_mode):
        raise ArtifactVerificationError(
            "input_not_regular",
            "input must be a real .amr file or export-package directory",
        )
    try:
        markers = sorted(
            entry.name for entry in path.iterdir() if entry.name in _DIRECTORY_MARKERS
        )
    except OSError as exc:
        raise ArtifactVerificationError(
            "input_unreadable",
            "export-package directory cannot be enumerated",
        ) from exc
    if not markers:
        raise ArtifactVerificationError(
            "input_unsupported",
            "directory has no recognized ArchMeshRubbing export sidecar",
        )
    if len(markers) != 1:
        raise ArtifactVerificationError(
            "input_ambiguous",
            "directory contains more than one export-package marker",
        )
    return _DIRECTORY_MARKERS[markers[0]]


def _load_authority_project(path: Path) -> ArtifactSession:
    identity = _path_identity(path, prefix="authority_project")
    if not stat.S_ISREG(identity.st_mode) or path.suffix.lower() != ".amr":
        raise ArtifactVerificationError(
            "authority_project_unsupported",
            "authority project must be a real .amr file",
        )
    return load_artifact_session_project(path)


def _sidecar_mapping(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:  # pragma: no cover
        raise RuntimeError(f"validated {label} sidecar could not be decoded") from exc
    if not isinstance(value, dict):  # pragma: no cover - validator guarantees this
        raise RuntimeError(f"validated {label} sidecar root is not an object")
    return value


def _bound_document_sha256(document: ArtifactDocument | None) -> str | None:
    return None if document is None else document.canonical_sha256


def _export_claims(
    sidecar: Mapping[str, Any],
    *,
    bound_document_sha256: str | None,
) -> dict[str, Any]:
    return {
        "bound_project_document_sha256": bound_document_sha256,
        "presentation": sidecar["presentation"],
        "provenance": sidecar["provenance"],
        "qc": sidecar["qc"],
        "recipe": sidecar["recipe"],
    }


def _verify_vector(
    path: Path,
    *,
    document: ArtifactDocument | None,
) -> dict[str, Any]:
    bundle = validate_vector_export_package(path, document=document)
    sidecar = _sidecar_mapping(bundle.sidecar_bytes, label="vector export")
    return {
        **_export_claims(
            sidecar,
            bound_document_sha256=_bound_document_sha256(document),
        ),
        "height_mm": bundle.height_mm,
        "sidecar_sha256": bundle.sidecar_sha256,
        "svg_sha256": bundle.svg_sha256,
        "vector_payload_sha256": bundle.vector_payload_sha256,
        "width_mm": bundle.width_mm,
    }


def _verify_rubbing(
    path: Path,
    *,
    document: ArtifactDocument | None,
) -> dict[str, Any]:
    bundle = validate_rubbing_export_package(path, document=document)
    sidecar = _sidecar_mapping(bundle.sidecar_bytes, label="rubbing export")
    return {
        **_export_claims(
            sidecar,
            bound_document_sha256=_bound_document_sha256(document),
        ),
        "height_pixels": bundle.height_pixels,
        "pixels_per_meter": bundle.pixels_per_meter,
        "png_sha256": bundle.png_sha256,
        "raster_sha256": bundle.raster_sha256,
        "raw_pixel_sha256": bundle.raw_pixel_sha256,
        "sidecar_sha256": bundle.sidecar_sha256,
        "width_pixels": bundle.width_pixels,
    }


def _verify_tile_unwrap(
    path: Path,
    *,
    document: ArtifactDocument | None,
) -> dict[str, Any]:
    sidecar = validate_tile_unwrap_export_package(path, document=document)
    sidecar_bytes = canonical_json_bytes(sidecar) + b"\n"
    return {
        **_export_claims(
            sidecar,
            bound_document_sha256=_bound_document_sha256(document),
        ),
        "artifacts": sidecar["artifacts"],
        "claims_sha256": sidecar["claims_sha256"],
        "geometry": sidecar["geometry"],
        "sidecar_sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
    }


def _verify_survey(
    path: Path,
    *,
    document: ArtifactDocument | None,
) -> dict[str, Any]:
    bundle = validate_survey_export_package(path, document=document)
    return {
        "artifact_count": bundle.artifact_count,
        "artifact_set_sha256": bundle.artifact_set_sha256,
        "authority": bundle.manifest["authority"],
        "bound_project_document_sha256": _bound_document_sha256(document),
        "manifest_sha256": bundle.manifest_sha256,
        "qc": bundle.manifest["qc"],
        "rubbing_count": bundle.rubbing_count,
        "vector_count": bundle.vector_count,
        "workflow": bundle.manifest["workflow"],
    }


def _sorted_counter(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _verify_project(path: Path) -> dict[str, Any]:
    session = load_artifact_session_project(path)
    document = session.document
    active_metadata_id = document.active_source_metadata_revision_id
    active_align_id = document.active_align_revision_id
    if active_metadata_id is None or active_align_id is None:  # pragma: no cover
        raise RuntimeError("materialized project has no active metadata and Align")
    metadata = document.source_metadata_revision_index[active_metadata_id]
    align = document.align_revision_index[active_align_id]
    geometry = document.geometry_revision_index[
        session.verified_geometry.geometry_revision_id
    ]
    source = document.source_asset_index[session.verified_geometry.source_asset_id]
    metadata_value = metadata.to_dict()
    align_value = align.to_dict()
    geometry_value = geometry.to_dict()
    freshnesses = document.record_freshnesses()
    return {
        "active_canonical_matrix": [
            [float(cell) for cell in row]
            for row in document.active_canonical_matrix().tolist()
        ],
        "align": {
            "id": align_value["id"],
            "matrix4x4": align_value["matrix4x4"],
            "qc": align_value["qc"],
            "recipe": align_value["recipe"],
        },
        "document_id": document.document_id,
        "document_schema_version": document.schema_version,
        "document_sha256": document.canonical_sha256,
        "embedded_source_materialized": True,
        "geometry": {
            "face_count": session.source_mesh.n_faces,
            "geometry_hash_scope": session.verified_geometry.geometry_hash_scope,
            "geometry_revision_id": session.verified_geometry.geometry_revision_id,
            "geometry_sha256": session.verified_geometry.geometry_sha256,
            "import_recipe": geometry_value["import_recipe"],
            "qc": geometry_value["qc"],
            "vertex_count": session.source_mesh.n_vertices,
        },
        "records": {
            "count": len(document.records),
            "freshness": _sorted_counter(
                [freshness.value for freshness in freshnesses.values()]
            ),
            "lifecycle": _sorted_counter(
                [RecordLifecycleStatus(record.lifecycle_status).value for record in document.records]
            ),
            "types": _sorted_counter([record.type for record in document.records]),
        },
        "software_version": document.software_version,
        "source": {
            "identity_scope": source.identity_scope,
            "media_type": source.media_type,
            "original_name": source.original_name,
            "sha256": source.sha256,
            "size_bytes": source.size_bytes,
        },
        "source_metadata": {
            "axes": metadata_value["axes"],
            "confirmation_status": metadata_value["confirmation_status"],
            "handedness": metadata_value["handedness"],
            "id": metadata_value["id"],
            "source_to_canonical_mm": metadata_value["source_to_canonical_mm"],
            "unit": metadata_value["unit"],
        },
    }


def build_artifact_verification_report(
    path: str | os.PathLike[str],
    *,
    against_project: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Return a deterministic success or fail-closed offline verification report.

    A supplied authority project is itself fully reopened from embedded source,
    then its exact ``ArtifactDocument`` is passed into the export validator.
    ``.amr`` input verification also requires complete embedded-source
    materialization; a manifest-only project never receives a success receipt.
    """

    input_name = _safe_input_name(path)
    authority = (
        AUTHORITY_MATCHED_PROJECT
        if against_project is not None
        else AUTHORITY_SELF_CONTAINED
    )
    artifact_kind = ARTIFACT_KIND_UNKNOWN
    try:
        input_path = Path(os.fspath(path))
        artifact_kind = _detect_artifact(input_path)
        if artifact_kind == ARTIFACT_KIND_PROJECT:
            if against_project is not None:
                raise ArtifactVerificationError(
                    "project_binding_not_applicable",
                    "--against-project applies only to export-package directories",
                    artifact_kind=artifact_kind,
                )
            evidence = _verify_project(input_path)
        else:
            document: ArtifactDocument | None = None
            if against_project is not None:
                authority_session = _load_authority_project(
                    Path(os.fspath(against_project))
                )
                document = authority_session.document
            if artifact_kind == ARTIFACT_KIND_VECTOR_EXPORT:
                evidence = _verify_vector(input_path, document=document)
            elif artifact_kind == ARTIFACT_KIND_RUBBING_EXPORT:
                evidence = _verify_rubbing(input_path, document=document)
            elif artifact_kind == ARTIFACT_KIND_TILE_UNWRAP_EXPORT:
                evidence = _verify_tile_unwrap(input_path, document=document)
            elif artifact_kind == ARTIFACT_KIND_SURVEY_EXPORT:
                evidence = _verify_survey(input_path, document=document)
            else:  # pragma: no cover - closed detector result
                raise RuntimeError("unsupported detected artifact kind")
        report = _report_base(
            input_name=input_name,
            artifact_kind=artifact_kind,
            authority=authority,
            ok=True,
        )
        report["evidence"] = evidence
        return report
    except ArtifactVerificationError as exc:
        return _failure_report(
            input_name=input_name,
            artifact_kind=(
                exc.artifact_kind
                if exc.artifact_kind != ARTIFACT_KIND_UNKNOWN
                else artifact_kind
            ),
            authority=authority,
            code=exc.code,
            message=str(exc),
        )
    except Exception as exc:
        paths: list[Path] = []
        try:
            paths.append(Path(os.fspath(path)))
        except (TypeError, ValueError, OSError):
            pass
        if against_project is not None:
            try:
                paths.append(Path(os.fspath(against_project)))
            except (TypeError, ValueError, OSError):
                pass
        return _failure_report(
            input_name=input_name,
            artifact_kind=artifact_kind,
            authority=authority,
            code="verification_failed",
            message=_redacted_error_message(exc, tuple(paths)),
        )


__all__ = [
    "ARTIFACT_KIND_PROJECT",
    "ARTIFACT_KIND_RUBBING_EXPORT",
    "ARTIFACT_KIND_SURVEY_EXPORT",
    "ARTIFACT_KIND_TILE_UNWRAP_EXPORT",
    "ARTIFACT_KIND_UNKNOWN",
    "ARTIFACT_KIND_VECTOR_EXPORT",
    "AUTHORITY_MATCHED_PROJECT",
    "AUTHORITY_SELF_CONTAINED",
    "OFFLINE_VERIFICATION_FORMAT",
    "OFFLINE_VERIFICATION_SCHEMA_VERSION",
    "ArtifactVerificationError",
    "build_artifact_verification_report",
]
