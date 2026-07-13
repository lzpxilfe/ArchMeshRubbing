"""Offline, privacy-bounded evidence for one real-world field pilot.

A field-pilot report combines four deliberately separate claims:

* strict materialization of one ``.amr`` project;
* exact-project verification of one complete ``.amr-survey`` package;
* one native Windows OpenGL driver-smoke receipt; and
* a closed human-review form completed by an archaeologist.

The report is canonical JSON with a semantic self-hash and an explicit
``authentication=none`` declaration.  It is evidence for one artifact on one
machine, never a signature or a release approval.  Absolute input paths,
hostnames and user names are not recorded.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import stat
import struct
import sys
import tempfile
import time
from typing import Any

from .artifact_verification import (
    ARTIFACT_KIND_PROJECT,
    ARTIFACT_KIND_SURVEY_EXPORT,
    ARTIFACT_KIND_UNKNOWN,
    AUTHORITY_MATCHED_PROJECT,
    AUTHORITY_SELF_CONTAINED,
    OFFLINE_VERIFICATION_FORMAT,
    OFFLINE_VERIFICATION_SCHEMA_VERSION,
    build_artifact_verification_report,
)
from .artifact_vector_export import fsync_export_directory
from .canonical_json import canonical_json_bytes, canonical_json_sha256


FIELD_PILOT_REVIEW_FORMAT = "archmeshrubbing_field_pilot_review"
FIELD_PILOT_REPORT_FORMAT = "archmeshrubbing_field_pilot_report"
FIELD_PILOT_VERIFICATION_FORMAT = (
    "archmeshrubbing_field_pilot_verification"
)
FIELD_PILOT_SCHEMA_VERSION = "1.0.0"
FIELD_PILOT_SCOPE = "single_artifact_single_machine"
FIELD_PILOT_RELEASE_CLAIM = "single_pilot_only_not_release_approval"

FIELD_PILOT_CHECKS = (
    "source_unit",
    "align_grounding",
    "cutline_fidelity",
    "outline_fidelity",
    "rubbing_legibility",
    "physical_scale_1_1",
    "original_source_preserved",
    "offline_operation",
    "workflow_stability",
    "workflow_usability",
)

REVIEW_STATUS_PASS = "pass"
REVIEW_STATUS_FAIL = "fail"
REVIEW_STATUS_NOT_TESTED = "not_tested"
REVIEW_STATUSES = frozenset(
    {
        REVIEW_STATUS_PASS,
        REVIEW_STATUS_FAIL,
        REVIEW_STATUS_NOT_TESTED,
    }
)

MAX_FIELD_PILOT_REVIEW_BYTES = 64 * 1024
MAX_FIELD_PILOT_REPORT_BYTES = 16 * 1024 * 1024
MAX_OPENGL_REPORT_BYTES = 32 * 1024 * 1024

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64}|unknown)$")
_CHANNEL_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,31}$")
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_MULTILINE_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SAFE_NAME_RE = re.compile(r"^[^/\\\x00-\x1f\x7f]{1,255}$")
_ERROR_CODE_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_WINDOWS_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/]|\\\\)[^\r\n\t\"']+"
)
_POSIX_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9:])/(?:[^/\s\"']+/)*[^/\s\"']*"
)

_TEMPLATE_ARTIFACT_LABEL = "replace-with-local-artifact-label"
_TEMPLATE_REVIEWER_ID = "replace-with-reviewer-pseudonym"


class FieldPilotError(ValueError):
    """A field-pilot input or report violates its closed contract."""


@dataclass(frozen=True, slots=True)
class FieldPilotPublication:
    path: Path
    sha256: str
    durability_confirmed: bool
    warning_message: str | None = None


def _exact_mapping(
    value: object,
    keys: set[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FieldPilotError(f"{label} must be an object")
    observed = set(value)
    if any(not isinstance(key, str) for key in observed):
        raise FieldPilotError(f"{label} contains a non-string field name")
    if observed != keys:
        raise FieldPilotError(
            f"{label} fields differ; missing={sorted(keys - observed)}, "
            f"unexpected={sorted(observed - keys)}"
        )
    return value


def _required_text(
    value: object,
    *,
    label: str,
    maximum: int,
    multiline: bool = False,
) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise FieldPilotError(
            f"{label} must be a non-empty string no longer than {maximum} characters"
        )
    if value != value.strip():
        raise FieldPilotError(f"{label} must not have surrounding whitespace")
    matcher = _MULTILINE_CONTROL_RE if multiline else _CONTROL_RE
    if matcher.search(value):
        raise FieldPilotError(f"{label} contains a forbidden control character")
    return value


def _optional_text(
    value: object,
    *,
    label: str,
    maximum: int,
    multiline: bool = False,
) -> str:
    if value == "":
        return ""
    return _required_text(
        value,
        label=label,
        maximum=maximum,
        multiline=multiline,
    )


def _safe_input_name(value: str | os.PathLike[str]) -> str:
    try:
        raw = os.fspath(value)
        if not isinstance(raw, str):
            return "invalid-input"
        normalized = raw.replace("\\", "/").rstrip("/")
        name = normalized.rsplit("/", 1)[-1] if normalized else ""
    except (OSError, TypeError, ValueError):
        return "invalid-input"
    if not name:
        return "current-directory"
    safe = "".join(
        "_"
        if character in "/\\" or ord(character) < 32 or ord(character) == 127
        else character
        for character in name
    )
    return safe[:255] or "unnamed-input"


def _validate_safe_name(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SAFE_NAME_RE.fullmatch(value) is None:
        raise FieldPilotError(f"{label} is not a private basename")
    return value


def _validate_timestamp(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _TIMESTAMP_RE.fullmatch(value) is None:
        raise FieldPilotError(f"{label} must be UTC whole seconds ending in Z")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise FieldPilotError(f"{label} is not a real timestamp") from exc
    normalized = parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat()
    if normalized.replace("+00:00", "Z") != value:
        raise FieldPilotError(f"{label} is not normalized UTC")
    return value


def _utc_now_seconds() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _optional_measurement(
    value: object,
    *,
    label: str,
    strictly_positive: bool,
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FieldPilotError(f"{label} must be a finite number or null")
    number = float(value)
    if not math.isfinite(number):
        raise FieldPilotError(f"{label} must be finite")
    if strictly_positive and number <= 0.0:
        raise FieldPilotError(f"{label} must be greater than zero")
    if not strictly_positive and number < 0.0:
        raise FieldPilotError(f"{label} must be non-negative")
    return number


def default_field_pilot_review() -> dict[str, Any]:
    """Return a schema-valid checklist that cannot accidentally pass."""

    return {
        "artifact_label": _TEMPLATE_ARTIFACT_LABEL,
        "checks": {name: REVIEW_STATUS_NOT_TESTED for name in FIELD_PILOT_CHECKS},
        "format": FIELD_PILOT_REVIEW_FORMAT,
        "measurements": {
            "scale_expected_mm": None,
            "scale_observed_mm": None,
            "scale_tolerance_mm": None,
            "workflow_elapsed_minutes": None,
        },
        "notes": "",
        "project_document_sha256": None,
        "reviewed_at_utc": None,
        "reviewer_id": _TEMPLATE_REVIEWER_ID,
        "schema_version": FIELD_PILOT_SCHEMA_VERSION,
        "survey_artifact_set_sha256": None,
    }


def validate_field_pilot_review(value: object) -> dict[str, Any]:
    """Validate and normalize one editable human-review document."""

    root = _exact_mapping(
        value,
        {
            "artifact_label",
            "checks",
            "format",
            "measurements",
            "notes",
            "project_document_sha256",
            "reviewed_at_utc",
            "reviewer_id",
            "schema_version",
            "survey_artifact_set_sha256",
        },
        label="field-pilot review",
    )
    if root.get("format") != FIELD_PILOT_REVIEW_FORMAT:
        raise FieldPilotError("field-pilot review format is invalid")
    if root.get("schema_version") != FIELD_PILOT_SCHEMA_VERSION:
        raise FieldPilotError("field-pilot review schema version is invalid")
    artifact_label = _required_text(
        root.get("artifact_label"),
        label="artifact_label",
        maximum=128,
    )
    reviewer_id = _required_text(
        root.get("reviewer_id"),
        label="reviewer_id",
        maximum=128,
    )
    reviewed_at = root.get("reviewed_at_utc")
    if reviewed_at is not None:
        reviewed_at = _validate_timestamp(
            reviewed_at,
            label="reviewed_at_utc",
        )

    checks_value = _exact_mapping(
        root.get("checks"),
        set(FIELD_PILOT_CHECKS),
        label="field-pilot checks",
    )
    checks: dict[str, str] = {}
    for name in FIELD_PILOT_CHECKS:
        status = checks_value.get(name)
        if not isinstance(status, str) or status not in REVIEW_STATUSES:
            raise FieldPilotError(
                f"field-pilot check {name} must be pass, fail, or not_tested"
            )
        checks[name] = status

    measurements_value = _exact_mapping(
        root.get("measurements"),
        {
            "scale_expected_mm",
            "scale_observed_mm",
            "scale_tolerance_mm",
            "workflow_elapsed_minutes",
        },
        label="field-pilot measurements",
    )
    expected = _optional_measurement(
        measurements_value.get("scale_expected_mm"),
        label="scale_expected_mm",
        strictly_positive=True,
    )
    observed = _optional_measurement(
        measurements_value.get("scale_observed_mm"),
        label="scale_observed_mm",
        strictly_positive=True,
    )
    tolerance = _optional_measurement(
        measurements_value.get("scale_tolerance_mm"),
        label="scale_tolerance_mm",
        strictly_positive=False,
    )
    scale_values = (expected, observed, tolerance)
    scale_status = checks["physical_scale_1_1"]
    if scale_status == REVIEW_STATUS_NOT_TESTED:
        if any(item is not None for item in scale_values):
            raise FieldPilotError(
                "physical scale measurements must be null while the check is not_tested"
            )
    elif any(item is None for item in scale_values):
        raise FieldPilotError(
            "physical scale pass/fail requires expected, observed, and tolerance mm"
        )
    else:
        assert expected is not None and observed is not None and tolerance is not None
        within_tolerance = abs(observed - expected) <= tolerance
        expected_status = (
            REVIEW_STATUS_PASS if within_tolerance else REVIEW_STATUS_FAIL
        )
        if scale_status != expected_status:
            raise FieldPilotError(
                "physical scale status contradicts the measured tolerance"
            )

    elapsed = _optional_measurement(
        measurements_value.get("workflow_elapsed_minutes"),
        label="workflow_elapsed_minutes",
        strictly_positive=True,
    )
    usability_status = checks["workflow_usability"]
    if usability_status == REVIEW_STATUS_NOT_TESTED and elapsed is not None:
        raise FieldPilotError(
            "workflow elapsed time must be null while usability is not_tested"
        )
    if usability_status != REVIEW_STATUS_NOT_TESTED and elapsed is None:
        raise FieldPilotError(
            "workflow usability pass/fail requires elapsed minutes"
        )

    notes = _optional_text(
        root.get("notes"),
        label="notes",
        maximum=4096,
        multiline=True,
    )
    project_document_sha256 = root.get("project_document_sha256")
    survey_artifact_set_sha256 = root.get("survey_artifact_set_sha256")
    for digest, label in (
        (project_document_sha256, "project_document_sha256"),
        (survey_artifact_set_sha256, "survey_artifact_set_sha256"),
    ):
        if digest is not None:
            _validate_digest(digest, label=label)
    if (project_document_sha256 is None) != (
        survey_artifact_set_sha256 is None
    ):
        raise FieldPilotError(
            "project and survey review bindings must both be supplied or both be null"
        )
    return {
        "artifact_label": artifact_label,
        "checks": checks,
        "format": FIELD_PILOT_REVIEW_FORMAT,
        "measurements": {
            "scale_expected_mm": expected,
            "scale_observed_mm": observed,
            "scale_tolerance_mm": tolerance,
            "workflow_elapsed_minutes": elapsed,
        },
        "notes": notes,
        "project_document_sha256": project_document_sha256,
        "reviewed_at_utc": reviewed_at,
        "reviewer_id": reviewer_id,
        "schema_version": FIELD_PILOT_SCHEMA_VERSION,
        "survey_artifact_set_sha256": survey_artifact_set_sha256,
    }


def _read_regular_file(
    path: Path,
    *,
    label: str,
    maximum: int,
) -> bytes:
    try:
        identity = path.stat(follow_symlinks=False)
    except FileNotFoundError as exc:
        raise FieldPilotError(f"{label} does not exist") from exc
    except OSError as exc:
        raise FieldPilotError(f"{label} cannot be inspected") from exc
    if not stat.S_ISREG(identity.st_mode) or path.is_symlink():
        raise FieldPilotError(f"{label} must be a real regular file")
    if identity.st_size <= 0 or identity.st_size > maximum:
        raise FieldPilotError(f"{label} size is outside the safety limit")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise FieldPilotError(f"{label} cannot be read") from exc
    if len(payload) != identity.st_size:
        raise FieldPilotError(f"{label} changed while being read")
    return payload


def _strict_json(payload: bytes, *, label: str) -> object:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON number: {value}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise FieldPilotError(f"{label} is not strict UTF-8 JSON") from exc
    return value


def load_field_pilot_review(
    path: str | os.PathLike[str],
) -> tuple[dict[str, Any], str]:
    source = Path(os.fspath(path))
    payload = _read_regular_file(
        source,
        label="field-pilot review",
        maximum=MAX_FIELD_PILOT_REVIEW_BYTES,
    )
    review = validate_field_pilot_review(
        _strict_json(payload, label="field-pilot review")
    )
    return review, hashlib.sha256(payload).hexdigest()


def _human_review_status(
    review: Mapping[str, Any],
    *,
    project_document_sha256: str | None,
    survey_artifact_set_sha256: str | None,
    report_created_at_utc: str,
) -> str:
    checks = review["checks"]
    assert isinstance(checks, Mapping)
    statuses = [checks[name] for name in FIELD_PILOT_CHECKS]
    if REVIEW_STATUS_FAIL in statuses:
        return REVIEW_STATUS_FAIL
    review_project = review.get("project_document_sha256")
    review_survey = review.get("survey_artifact_set_sha256")
    if review_project is not None and (
        project_document_sha256 is None
        or review_project != project_document_sha256
        or survey_artifact_set_sha256 is None
        or review_survey != survey_artifact_set_sha256
    ):
        return REVIEW_STATUS_FAIL
    reviewed_at = review.get("reviewed_at_utc")
    if isinstance(reviewed_at, str) and reviewed_at > report_created_at_utc:
        return REVIEW_STATUS_FAIL
    if (
        all(status == REVIEW_STATUS_PASS for status in statuses)
        and reviewed_at is not None
        and review.get("artifact_label") != _TEMPLATE_ARTIFACT_LABEL
        and review.get("reviewer_id") != _TEMPLATE_REVIEWER_ID
        and review_project == project_document_sha256
        and review_survey == survey_artifact_set_sha256
        and project_document_sha256 is not None
        and survey_artifact_set_sha256 is not None
    ):
        return REVIEW_STATUS_PASS
    return "incomplete"


def _nullable_positive_int(value: object, *, label: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value <= 0:
        raise FieldPilotError(f"{label} must be a positive integer or null")
    return value


def _total_physical_memory_bytes() -> int | None:
    if sys.platform == "win32":
        try:
            class MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status_value = MemoryStatusEx()
            status_value.dwLength = ctypes.sizeof(status_value)
            windll = getattr(ctypes, "windll", None)
            if windll is None:
                return None
            if not windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status_value)):
                return None
            return int(status_value.ullTotalPhys)
        except (AttributeError, OSError, TypeError, ValueError):
            return None
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        pages = int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None
    total = page_size * pages
    return total if total > 0 else None


def _peak_working_set_bytes() -> int | None:
    if sys.platform == "win32":
        try:
            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            windll = getattr(ctypes, "windll", None)
            if windll is None:
                return None
            process = windll.kernel32.GetCurrentProcess()
            ok = windll.psapi.GetProcessMemoryInfo(
                process,
                ctypes.byref(counters),
                counters.cb,
            )
            return int(counters.PeakWorkingSetSize) if ok else None
        except (AttributeError, OSError, TypeError, ValueError):
            return None
    try:
        import resource

        usage = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, OSError, ValueError):
        return None
    if usage <= 0:
        return None
    return usage if sys.platform == "darwin" else usage * 1024


def field_pilot_machine_snapshot() -> dict[str, Any]:
    """Return a host summary that deliberately excludes identity and paths."""

    logical_cpus = os.cpu_count()
    return {
        "frozen": bool(getattr(sys, "frozen", False)),
        "logical_cpu_count": logical_cpus if logical_cpus and logical_cpus > 0 else None,
        "machine": (platform.machine() or "unknown")[:128],
        "peak_working_set_bytes": _peak_working_set_bytes(),
        "process_bits": struct.calcsize("P") * 8,
        "python_version": platform.python_version(),
        "release": (platform.release() or "unknown")[:128],
        "system": platform.system() or "unknown",
        "total_physical_memory_bytes": _total_physical_memory_bytes(),
    }


def _validate_machine(value: object) -> dict[str, Any]:
    root = _exact_mapping(
        value,
        {
            "frozen",
            "logical_cpu_count",
            "machine",
            "peak_working_set_bytes",
            "process_bits",
            "python_version",
            "release",
            "system",
            "total_physical_memory_bytes",
        },
        label="field-pilot machine",
    )
    if type(root.get("frozen")) is not bool:
        raise FieldPilotError("machine.frozen must be boolean")
    process_bits = root.get("process_bits")
    if process_bits not in {32, 64}:
        raise FieldPilotError("machine.process_bits must be 32 or 64")
    result = {
        "frozen": bool(root["frozen"]),
        "logical_cpu_count": _nullable_positive_int(
            root.get("logical_cpu_count"),
            label="logical_cpu_count",
        ),
        "machine": _required_text(
            root.get("machine"),
            label="machine",
            maximum=128,
        ),
        "peak_working_set_bytes": _nullable_positive_int(
            root.get("peak_working_set_bytes"),
            label="peak_working_set_bytes",
        ),
        "process_bits": int(process_bits),
        "python_version": _required_text(
            root.get("python_version"),
            label="python_version",
            maximum=64,
        ),
        "release": _required_text(
            root.get("release"),
            label="release",
            maximum=128,
        ),
        "system": _required_text(
            root.get("system"),
            label="system",
            maximum=64,
        ),
        "total_physical_memory_bytes": _nullable_positive_int(
            root.get("total_physical_memory_bytes"),
            label="total_physical_memory_bytes",
        ),
    }
    return result


def _validate_build(value: object) -> dict[str, Any]:
    root = _exact_mapping(
        value,
        {
            "channel",
            "commit",
            "dependency_lock_sha256",
            "manifest_present",
            "source_tree",
            "windows_wheel_lock_sha256",
        },
        label="field-pilot build",
    )
    channel = root.get("channel")
    commit = root.get("commit")
    dependency = root.get("dependency_lock_sha256")
    wheel = root.get("windows_wheel_lock_sha256")
    if not isinstance(channel, str) or _CHANNEL_RE.fullmatch(channel) is None:
        raise FieldPilotError("build.channel is invalid")
    if not isinstance(commit, str) or _COMMIT_RE.fullmatch(commit) is None:
        raise FieldPilotError("build.commit is invalid")
    for label, digest in (
        ("dependency_lock_sha256", dependency),
        ("windows_wheel_lock_sha256", wheel),
    ):
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise FieldPilotError(f"build.{label} is invalid")
    if type(root.get("manifest_present")) is not bool:
        raise FieldPilotError("build.manifest_present must be boolean")
    source_tree = root.get("source_tree")
    if source_tree not in {"clean", "dirty", "unknown"}:
        raise FieldPilotError("build.source_tree is invalid")
    return {
        "channel": channel,
        "commit": commit,
        "dependency_lock_sha256": dependency,
        "manifest_present": bool(root["manifest_present"]),
        "source_tree": source_tree,
        "windows_wheel_lock_sha256": wheel,
    }


def _not_provided_opengl() -> dict[str, Any]:
    return {
        "check_count": 0,
        "context": None,
        "error": None,
        "input_name": None,
        "runtime_lock_sha256": None,
        "sha256": None,
        "size_bytes": None,
        "source_commit": None,
        "status": "not_provided",
        "tested_at_utc": None,
    }


def _failed_opengl(input_name: str, message: str) -> dict[str, Any]:
    return {
        "check_count": 0,
        "context": None,
        "error": _required_text(message, label="OpenGL error", maximum=512),
        "input_name": input_name,
        "runtime_lock_sha256": None,
        "sha256": None,
        "size_bytes": None,
        "source_commit": None,
        "status": "fail",
        "tested_at_utc": None,
    }


def _load_opengl_descriptor(
    path: str | os.PathLike[str],
    *,
    build: Mapping[str, Any],
) -> dict[str, Any]:
    source_path = Path(os.fspath(path))
    input_name = _safe_input_name(source_path)
    try:
        payload = _read_regular_file(
            source_path,
            label="OpenGL driver-smoke report",
            maximum=MAX_OPENGL_REPORT_BYTES,
        )
        value = _strict_json(payload, label="OpenGL driver-smoke report")
        if not isinstance(value, Mapping):
            raise FieldPilotError("OpenGL driver-smoke report root is not an object")
        if value.get("schema") != "archmeshrubbing.opengl_driver_smoke":
            raise FieldPilotError("OpenGL driver-smoke schema is invalid")
        if value.get("schema_version") != 1 or value.get("ok") is not True:
            raise FieldPilotError("OpenGL driver-smoke did not pass")
        checks = value.get("checks")
        if not isinstance(checks, list) or not checks:
            raise FieldPilotError("OpenGL driver-smoke has no checks")
        check_ids: set[str] = set()
        for entry in checks:
            if not isinstance(entry, Mapping) or entry.get("ok") is not True:
                raise FieldPilotError("OpenGL driver-smoke contains a failed check")
            check_id = entry.get("id")
            if not isinstance(check_id, str) or not check_id or check_id in check_ids:
                raise FieldPilotError("OpenGL driver-smoke check IDs are invalid")
            check_ids.add(check_id)
        context = value.get("context")
        if not isinstance(context, Mapping):
            raise FieldPilotError("OpenGL driver-smoke context is missing")
        if context.get("qt_platform") != "windows":
            raise FieldPilotError("OpenGL driver-smoke did not use qwindows")
        vendor = _required_text(
            context.get("vendor"),
            label="OpenGL vendor",
            maximum=256,
        )
        renderer = _required_text(
            context.get("renderer"),
            label="OpenGL renderer",
            maximum=256,
        )
        version = _required_text(
            context.get("version"),
            label="OpenGL version",
            maximum=256,
        )
        depth_bits = context.get("depth_bits")
        if type(depth_bits) is not int or depth_bits < 24:
            raise FieldPilotError("OpenGL driver-smoke depth buffer is below 24 bits")
        if type(context.get("software_renderer")) is not bool:
            raise FieldPilotError("OpenGL software-renderer flag is invalid")
        render_modes = value.get("render_modes")
        if not isinstance(render_modes, list) or len(render_modes) != 2:
            raise FieldPilotError("OpenGL driver-smoke omitted a projection mode")
        cleanup_errors = value.get("cleanup_errors")
        if not isinstance(cleanup_errors, list) or cleanup_errors:
            raise FieldPilotError("OpenGL driver-smoke cleanup is not clean")
        tested_at = _validate_timestamp(
            value.get("tested_at_utc"),
            label="OpenGL tested_at_utc",
        )
        source = value.get("source")
        if not isinstance(source, Mapping):
            raise FieldPilotError("OpenGL driver-smoke source receipt is missing")
        source_commit = source.get("commit")
        runtime_lock = source.get("runtime_lock_sha256")
        if not isinstance(source_commit, str) or _COMMIT_RE.fullmatch(source_commit) is None:
            raise FieldPilotError("OpenGL driver-smoke source commit is invalid")
        if not isinstance(runtime_lock, str) or _SHA256_RE.fullmatch(runtime_lock) is None:
            raise FieldPilotError("OpenGL driver-smoke runtime lock is invalid")
        if runtime_lock != build["dependency_lock_sha256"]:
            raise FieldPilotError("OpenGL driver-smoke runtime lock differs from this build")
        build_commit = build["commit"]
        if (
            build_commit != "unknown"
            and source_commit != "unknown"
            and source_commit != build_commit
        ):
            raise FieldPilotError("OpenGL driver-smoke commit differs from this build")
        return {
            "check_count": len(checks),
            "context": {
                "depth_bits": depth_bits,
                "qt_platform": "windows",
                "renderer": renderer,
                "software_renderer": bool(context["software_renderer"]),
                "vendor": vendor,
                "version": version,
            },
            "error": None,
            "input_name": input_name,
            "runtime_lock_sha256": runtime_lock,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
            "source_commit": source_commit,
            "status": REVIEW_STATUS_PASS,
            "tested_at_utc": tested_at,
        }
    except FieldPilotError as exc:
        return _failed_opengl(input_name, str(exc))


def _validate_opengl_descriptor(value: object) -> dict[str, Any]:
    root = _exact_mapping(
        value,
        {
            "check_count",
            "context",
            "error",
            "input_name",
            "runtime_lock_sha256",
            "sha256",
            "size_bytes",
            "source_commit",
            "status",
            "tested_at_utc",
        },
        label="field-pilot OpenGL descriptor",
    )
    status_value = root.get("status")
    if status_value not in {REVIEW_STATUS_PASS, REVIEW_STATUS_FAIL, "not_provided"}:
        raise FieldPilotError("OpenGL descriptor status is invalid")
    check_count = root.get("check_count")
    if type(check_count) is not int or check_count < 0:
        raise FieldPilotError("OpenGL descriptor check_count is invalid")
    if status_value == "not_provided":
        if root != _not_provided_opengl():
            raise FieldPilotError("not-provided OpenGL descriptor is not canonical")
        return _not_provided_opengl()
    input_name = _validate_safe_name(root.get("input_name"), label="OpenGL input_name")
    if status_value == REVIEW_STATUS_FAIL:
        if check_count != 0 or root.get("context") is not None:
            raise FieldPilotError("failed OpenGL descriptor carries success evidence")
        if any(
            root.get(key) is not None
            for key in (
                "runtime_lock_sha256",
                "sha256",
                "size_bytes",
                "source_commit",
                "tested_at_utc",
            )
        ):
            raise FieldPilotError("failed OpenGL descriptor carries partial evidence")
        error = _required_text(
            root.get("error"),
            label="OpenGL error",
            maximum=512,
        )
        return _failed_opengl(input_name, error)
    if check_count <= 0 or root.get("error") is not None:
        raise FieldPilotError("passing OpenGL descriptor is incomplete")
    for key in ("runtime_lock_sha256", "sha256"):
        digest = root.get(key)
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise FieldPilotError(f"OpenGL descriptor {key} is invalid")
    source_commit = root.get("source_commit")
    if not isinstance(source_commit, str) or _COMMIT_RE.fullmatch(source_commit) is None:
        raise FieldPilotError("OpenGL descriptor source_commit is invalid")
    size_bytes = root.get("size_bytes")
    if type(size_bytes) is not int or size_bytes <= 0:
        raise FieldPilotError("OpenGL descriptor size_bytes is invalid")
    tested_at = _validate_timestamp(
        root.get("tested_at_utc"),
        label="OpenGL tested_at_utc",
    )
    context = _exact_mapping(
        root.get("context"),
        {
            "depth_bits",
            "qt_platform",
            "renderer",
            "software_renderer",
            "vendor",
            "version",
        },
        label="OpenGL context",
    )
    depth_bits = context.get("depth_bits")
    if type(depth_bits) is not int or depth_bits < 24:
        raise FieldPilotError("OpenGL context depth_bits is invalid")
    if context.get("qt_platform") != "windows":
        raise FieldPilotError("OpenGL context is not qwindows")
    if type(context.get("software_renderer")) is not bool:
        raise FieldPilotError("OpenGL context software_renderer is invalid")
    normalized_context = {
        "depth_bits": depth_bits,
        "qt_platform": "windows",
        "renderer": _required_text(
            context.get("renderer"), label="OpenGL renderer", maximum=256
        ),
        "software_renderer": bool(context["software_renderer"]),
        "vendor": _required_text(
            context.get("vendor"), label="OpenGL vendor", maximum=256
        ),
        "version": _required_text(
            context.get("version"), label="OpenGL version", maximum=256
        ),
    }
    return {
        "check_count": check_count,
        "context": normalized_context,
        "error": None,
        "input_name": input_name,
        "runtime_lock_sha256": root["runtime_lock_sha256"],
        "sha256": root["sha256"],
        "size_bytes": size_bytes,
        "source_commit": source_commit,
        "status": REVIEW_STATUS_PASS,
        "tested_at_utc": tested_at,
    }


def _validate_offline_report_shape(
    value: object,
    *,
    artifact_kind: str,
    authority: str,
    input_name: str,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FieldPilotError(f"{label} must be an object")
    base_fields = {
        "artifact_kind",
        "authority",
        "format",
        "input_name",
        "ok",
        "schema_version",
    }
    ok = value.get("ok")
    if type(ok) is not bool:
        raise FieldPilotError(f"{label} ok flag is invalid")
    expected_fields = base_fields | ({"evidence"} if ok else {"error"})
    if set(value) != expected_fields:
        raise FieldPilotError(f"{label} fields are not closed")
    if value.get("format") != OFFLINE_VERIFICATION_FORMAT:
        raise FieldPilotError(f"{label} format is invalid")
    if value.get("schema_version") != OFFLINE_VERIFICATION_SCHEMA_VERSION:
        raise FieldPilotError(f"{label} schema version is invalid")
    observed_kind = value.get("artifact_kind")
    if value.get("authority") != authority:
        raise FieldPilotError(f"{label} authority is invalid")
    if value.get("input_name") != input_name:
        raise FieldPilotError(f"{label} input name is inconsistent")
    if ok:
        if observed_kind != artifact_kind:
            raise FieldPilotError(f"{label} artifact kind is invalid")
        evidence = _validate_offline_success_evidence(
            value.get("evidence"),
            artifact_kind=artifact_kind,
            label=f"{label} evidence",
        )
    else:
        if observed_kind not in {artifact_kind, ARTIFACT_KIND_UNKNOWN}:
            raise FieldPilotError(f"{label} failure artifact kind is invalid")
        error = _exact_mapping(
            value.get("error"),
            {"code", "message"},
            label=f"{label} error",
        )
        code = error.get("code")
        if not isinstance(code, str) or _ERROR_CODE_RE.fullmatch(code) is None:
            raise FieldPilotError(f"{label} error code is invalid")
        _required_text(
            error.get("message"),
            label=f"{label} error message",
            maximum=1024,
        )
        evidence = None
    _reject_automatic_absolute_paths(value, label=label)
    normalized = dict(value)
    if evidence is not None:
        normalized["evidence"] = evidence
    return normalized


def _reject_automatic_absolute_paths(value: object, *, label: str) -> None:
    if isinstance(value, str):
        if (
            _WINDOWS_ABSOLUTE_PATH_RE.search(value) is not None
            or _POSIX_ABSOLUTE_PATH_RE.search(value) is not None
        ):
            raise FieldPilotError(f"{label} contains an absolute path")
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_automatic_absolute_paths(item, label=label)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _reject_automatic_absolute_paths(item, label=label)


def _validate_digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise FieldPilotError(f"{label} is not a SHA-256 digest")
    return value


def _validate_nonnegative_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise FieldPilotError(f"{label} must be a non-negative integer")
    return value


def _validate_matrix4x4(value: object, *, label: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != 4:
        raise FieldPilotError(f"{label} must be a 4x4 matrix")
    matrix: list[list[float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 4:
            raise FieldPilotError(f"{label} must be a 4x4 matrix")
        normalized_row: list[float] = []
        for cell in row:
            if isinstance(cell, bool) or not isinstance(cell, (int, float)):
                raise FieldPilotError(f"{label} contains a non-number")
            number = float(cell)
            if not math.isfinite(number):
                raise FieldPilotError(f"{label} contains a non-finite number")
            normalized_row.append(number)
        matrix.append(normalized_row)
    return matrix


def _validate_counter(value: object, *, label: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise FieldPilotError(f"{label} must be an object")
    result: dict[str, int] = {}
    for key, count in value.items():
        name = _required_text(key, label=f"{label} key", maximum=128)
        if type(count) is not int or count <= 0:
            raise FieldPilotError(f"{label} values must be positive integers")
        result[name] = count
    return result


def _validate_project_evidence(value: object, *, label: str) -> dict[str, Any]:
    evidence = _exact_mapping(
        value,
        {
            "active_canonical_matrix",
            "align",
            "document_id",
            "document_schema_version",
            "document_sha256",
            "embedded_source_materialized",
            "geometry",
            "records",
            "software_version",
            "source",
            "source_metadata",
        },
        label=label,
    )
    if evidence.get("embedded_source_materialized") is not True:
        raise FieldPilotError(f"{label} did not materialize its embedded source")
    align = _exact_mapping(
        evidence.get("align"),
        {"id", "matrix4x4", "qc", "recipe"},
        label=f"{label}.align",
    )
    if not isinstance(align.get("qc"), Mapping) or not isinstance(
        align.get("recipe"), Mapping
    ):
        raise FieldPilotError(f"{label}.align claims are invalid")
    geometry = _exact_mapping(
        evidence.get("geometry"),
        {
            "face_count",
            "geometry_hash_scope",
            "geometry_revision_id",
            "geometry_sha256",
            "import_recipe",
            "qc",
            "vertex_count",
        },
        label=f"{label}.geometry",
    )
    if not isinstance(geometry.get("import_recipe"), Mapping) or not isinstance(
        geometry.get("qc"), Mapping
    ):
        raise FieldPilotError(f"{label}.geometry claims are invalid")
    records = _exact_mapping(
        evidence.get("records"),
        {"count", "freshness", "lifecycle", "types"},
        label=f"{label}.records",
    )
    source = _exact_mapping(
        evidence.get("source"),
        {"identity_scope", "media_type", "original_name", "sha256", "size_bytes"},
        label=f"{label}.source",
    )
    metadata = _exact_mapping(
        evidence.get("source_metadata"),
        {
            "axes",
            "confirmation_status",
            "handedness",
            "id",
            "source_to_canonical_mm",
            "unit",
        },
        label=f"{label}.source_metadata",
    )
    axes = _exact_mapping(
        metadata.get("axes"),
        {"source_x", "source_y", "source_z"},
        label=f"{label}.source_metadata.axes",
    )
    for key, axis in axes.items():
        _required_text(axis, label=f"{label}.axes.{key}", maximum=8)
    for key in (
        "document_id",
        "document_schema_version",
        "software_version",
    ):
        _required_text(evidence.get(key), label=f"{label}.{key}", maximum=256)
    _required_text(align.get("id"), label=f"{label}.align.id", maximum=256)
    _validate_matrix4x4(
        evidence.get("active_canonical_matrix"),
        label=f"{label}.active_canonical_matrix",
    )
    _validate_matrix4x4(align.get("matrix4x4"), label=f"{label}.align.matrix4x4")
    _validate_digest(evidence.get("document_sha256"), label=f"{label}.document_sha256")
    _validate_digest(geometry.get("geometry_sha256"), label=f"{label}.geometry_sha256")
    _validate_nonnegative_int(geometry.get("face_count"), label=f"{label}.face_count")
    _validate_nonnegative_int(
        geometry.get("vertex_count"),
        label=f"{label}.vertex_count",
    )
    for key in ("geometry_hash_scope", "geometry_revision_id"):
        _required_text(geometry.get(key), label=f"{label}.{key}", maximum=256)
    _validate_nonnegative_int(records.get("count"), label=f"{label}.records.count")
    for key in ("freshness", "lifecycle", "types"):
        _validate_counter(records.get(key), label=f"{label}.records.{key}")
    for key in ("identity_scope", "media_type", "original_name"):
        _required_text(source.get(key), label=f"{label}.source.{key}", maximum=256)
    _validate_digest(source.get("sha256"), label=f"{label}.source.sha256")
    _validate_nonnegative_int(
        source.get("size_bytes"),
        label=f"{label}.source.size_bytes",
    )
    for key in ("confirmation_status", "handedness", "id", "unit"):
        _required_text(metadata.get(key), label=f"{label}.metadata.{key}", maximum=256)
    _validate_matrix4x4(
        metadata.get("source_to_canonical_mm"),
        label=f"{label}.source_to_canonical_mm",
    )
    return dict(evidence)


def _validate_survey_evidence(value: object, *, label: str) -> dict[str, Any]:
    evidence = _exact_mapping(
        value,
        {
            "artifact_count",
            "artifact_set_sha256",
            "authority",
            "bound_project_document_sha256",
            "manifest_sha256",
            "qc",
            "rubbing_count",
            "vector_count",
            "workflow",
        },
        label=label,
    )
    if (
        evidence.get("artifact_count"),
        evidence.get("vector_count"),
        evidence.get("rubbing_count"),
    ) != (15, 9, 6):
        raise FieldPilotError(f"{label} does not contain the required 15 artifacts")
    for key in (
        "artifact_set_sha256",
        "bound_project_document_sha256",
        "manifest_sha256",
    ):
        _validate_digest(evidence.get(key), label=f"{label}.{key}")
    for key in ("authority", "qc", "workflow"):
        if not isinstance(evidence.get(key), Mapping):
            raise FieldPilotError(f"{label}.{key} must be an object")
    return dict(evidence)


def _validate_offline_success_evidence(
    value: object,
    *,
    artifact_kind: str,
    label: str,
) -> dict[str, Any]:
    if artifact_kind == ARTIFACT_KIND_PROJECT:
        return _validate_project_evidence(value, label=label)
    if artifact_kind == ARTIFACT_KIND_SURVEY_EXPORT:
        return _validate_survey_evidence(value, label=label)
    raise FieldPilotError(f"{label} has an unsupported artifact kind")


def _artifact_verification_status(
    project: Mapping[str, Any],
    survey: Mapping[str, Any],
) -> str:
    if project.get("ok") is not True or survey.get("ok") is not True:
        return REVIEW_STATUS_FAIL
    project_evidence = project.get("evidence")
    survey_evidence = survey.get("evidence")
    if not isinstance(project_evidence, Mapping) or not isinstance(
        survey_evidence, Mapping
    ):
        return REVIEW_STATUS_FAIL
    document_sha256 = project_evidence.get("document_sha256")
    if not isinstance(document_sha256, str) or _SHA256_RE.fullmatch(document_sha256) is None:
        return REVIEW_STATUS_FAIL
    if survey_evidence.get("bound_project_document_sha256") != document_sha256:
        return REVIEW_STATUS_FAIL
    if (
        survey_evidence.get("artifact_count"),
        survey_evidence.get("vector_count"),
        survey_evidence.get("rubbing_count"),
    ) != (15, 9, 6):
        return REVIEW_STATUS_FAIL
    for key in ("artifact_set_sha256", "manifest_sha256"):
        digest = survey_evidence.get(key)
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            return REVIEW_STATUS_FAIL
    return REVIEW_STATUS_PASS


def _verified_artifact_digests(
    project: Mapping[str, Any],
    survey: Mapping[str, Any],
) -> tuple[str | None, str | None]:
    if project.get("ok") is not True or survey.get("ok") is not True:
        return None, None
    project_evidence = project.get("evidence")
    survey_evidence = survey.get("evidence")
    if not isinstance(project_evidence, Mapping) or not isinstance(
        survey_evidence, Mapping
    ):
        return None, None
    project_digest = project_evidence.get("document_sha256")
    survey_digest = survey_evidence.get("artifact_set_sha256")
    if (
        not isinstance(project_digest, str)
        or _SHA256_RE.fullmatch(project_digest) is None
        or not isinstance(survey_digest, str)
        or _SHA256_RE.fullmatch(survey_digest) is None
    ):
        return None, None
    return project_digest, survey_digest


def _windows_runtime_status(machine: Mapping[str, Any]) -> str:
    return (
        REVIEW_STATUS_PASS
        if machine.get("system") == "Windows" and machine.get("process_bits") == 64
        else "not_target"
    )


def _pilot_outcome(
    *,
    artifact_status: str,
    human_status: str,
    opengl_status: str,
    windows_status: str,
) -> str:
    if REVIEW_STATUS_FAIL in {artifact_status, human_status, opengl_status}:
        return "failed"
    if (
        artifact_status == REVIEW_STATUS_PASS
        and human_status == REVIEW_STATUS_PASS
        and opengl_status == REVIEW_STATUS_PASS
        and windows_status == REVIEW_STATUS_PASS
    ):
        return "verified"
    return "incomplete"


def _duration_ms(start_ns: int, end_ns: int) -> int:
    if type(start_ns) is not int or type(end_ns) is not int or end_ns < start_ns:
        raise FieldPilotError("monotonic pilot timing source is invalid")
    return (end_ns - start_ns) // 1_000_000


def build_field_pilot_report(
    project: str | os.PathLike[str],
    survey: str | os.PathLike[str],
    *,
    review: str | os.PathLike[str] | None = None,
    opengl_report: str | os.PathLike[str] | None = None,
    created_at_utc: str | None = None,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    machine_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify one project/survey pair and build a canonical pilot record."""

    from src import build_info

    project_name = _safe_input_name(project)
    survey_name = _safe_input_name(survey)
    review_name = None if review is None else _safe_input_name(review)
    opengl_name = (
        None if opengl_report is None else _safe_input_name(opengl_report)
    )
    review_value: dict[str, Any]
    review_input_sha256: str | None
    if review is None:
        review_value = default_field_pilot_review()
        review_input_sha256 = None
    else:
        review_value, review_input_sha256 = load_field_pilot_review(review)

    build = _validate_build(build_info.build_metadata())
    machine = _validate_machine(
        field_pilot_machine_snapshot()
        if machine_snapshot is None
        else machine_snapshot
    )
    opengl = (
        _not_provided_opengl()
        if opengl_report is None
        else _load_opengl_descriptor(opengl_report, build=build)
    )

    project_start = clock_ns()
    project_verification = build_artifact_verification_report(project)
    project_verification["input_name"] = project_name
    project_end = clock_ns()
    survey_start = clock_ns()
    survey_verification = build_artifact_verification_report(
        survey,
        against_project=project,
    )
    survey_verification["input_name"] = survey_name
    survey_end = clock_ns()
    project_ms = _duration_ms(project_start, project_end)
    survey_ms = _duration_ms(survey_start, survey_end)

    created_at = _validate_timestamp(
        created_at_utc or _utc_now_seconds(),
        label="created_at_utc",
    )
    artifact_status = _artifact_verification_status(
        project_verification,
        survey_verification,
    )
    project_document_sha256, survey_artifact_set_sha256 = (
        _verified_artifact_digests(
            project_verification,
            survey_verification,
        )
    )
    human_status = _human_review_status(
        review_value,
        project_document_sha256=project_document_sha256,
        survey_artifact_set_sha256=survey_artifact_set_sha256,
        report_created_at_utc=created_at,
    )
    opengl_status = str(opengl["status"])
    windows_status = _windows_runtime_status(machine)
    pilot_status = _pilot_outcome(
        artifact_status=artifact_status,
        human_status=human_status,
        opengl_status=opengl_status,
        windows_status=windows_status,
    )
    payload: dict[str, Any] = {
        "application": {
            "name": build_info.APP_NAME,
            "version": build_info.APP_VERSION,
        },
        "authentication": {
            "kind": "none",
            "signature_present": False,
        },
        "build": build,
        "created_at_utc": created_at,
        "format": FIELD_PILOT_REPORT_FORMAT,
        "inputs": {
            "opengl_report_name": opengl_name,
            "project_name": project_name,
            "review_name": review_name,
            "survey_name": survey_name,
        },
        "machine": machine,
        "opengl_driver": opengl,
        "outcome": {
            "artifact_verification": artifact_status,
            "human_review": human_status,
            "opengl_driver": opengl_status,
            "pilot": pilot_status,
            "scope": FIELD_PILOT_SCOPE,
            "windows_runtime": windows_status,
        },
        "performance": {
            "project_verification_ms": project_ms,
            "survey_verification_ms": survey_ms,
            "total_verification_ms": project_ms + survey_ms,
        },
        "project_verification": project_verification,
        "release_claim": FIELD_PILOT_RELEASE_CLAIM,
        "review": review_value,
        "review_input_sha256": review_input_sha256,
        "review_normalized_sha256": canonical_json_sha256(review_value),
        "schema_version": FIELD_PILOT_SCHEMA_VERSION,
        "survey_verification": survey_verification,
    }
    payload["pilot_sha256"] = canonical_json_sha256(payload)
    return validate_field_pilot_report(payload)


def _validate_application(value: object) -> dict[str, str]:
    root = _exact_mapping(
        value,
        {"name", "version"},
        label="field-pilot application",
    )
    return {
        "name": _required_text(root.get("name"), label="application.name", maximum=128),
        "version": _required_text(
            root.get("version"),
            label="application.version",
            maximum=64,
        ),
    }


def validate_field_pilot_report(value: object) -> dict[str, Any]:
    """Validate report structure, cross-claims, and semantic self-hash."""

    root = _exact_mapping(
        value,
        {
            "application",
            "authentication",
            "build",
            "created_at_utc",
            "format",
            "inputs",
            "machine",
            "opengl_driver",
            "outcome",
            "performance",
            "pilot_sha256",
            "project_verification",
            "release_claim",
            "review",
            "review_input_sha256",
            "review_normalized_sha256",
            "schema_version",
            "survey_verification",
        },
        label="field-pilot report",
    )
    if root.get("format") != FIELD_PILOT_REPORT_FORMAT:
        raise FieldPilotError("field-pilot report format is invalid")
    if root.get("schema_version") != FIELD_PILOT_SCHEMA_VERSION:
        raise FieldPilotError("field-pilot report schema version is invalid")
    if root.get("release_claim") != FIELD_PILOT_RELEASE_CLAIM:
        raise FieldPilotError("field-pilot release claim is invalid")
    authentication = _exact_mapping(
        root.get("authentication"),
        {"kind", "signature_present"},
        label="field-pilot authentication",
    )
    if authentication != {"kind": "none", "signature_present": False}:
        raise FieldPilotError("field-pilot report must declare no authentication")
    application = _validate_application(root.get("application"))
    build = _validate_build(root.get("build"))
    created_at = _validate_timestamp(
        root.get("created_at_utc"),
        label="created_at_utc",
    )
    inputs = _exact_mapping(
        root.get("inputs"),
        {"opengl_report_name", "project_name", "review_name", "survey_name"},
        label="field-pilot inputs",
    )
    project_name = _validate_safe_name(
        inputs.get("project_name"),
        label="project_name",
    )
    survey_name = _validate_safe_name(
        inputs.get("survey_name"),
        label="survey_name",
    )
    review_name = inputs.get("review_name")
    if review_name is not None:
        review_name = _validate_safe_name(review_name, label="review_name")
    opengl_name = inputs.get("opengl_report_name")
    if opengl_name is not None:
        opengl_name = _validate_safe_name(opengl_name, label="opengl_report_name")

    machine = _validate_machine(root.get("machine"))
    opengl = _validate_opengl_descriptor(root.get("opengl_driver"))
    if opengl_name != opengl.get("input_name"):
        raise FieldPilotError("OpenGL input name does not match its descriptor")
    review = validate_field_pilot_review(root.get("review"))
    review_input_sha256 = root.get("review_input_sha256")
    if review_name is None:
        if review_input_sha256 is not None or review != default_field_pilot_review():
            raise FieldPilotError("review-less report carries human input")
    elif (
        not isinstance(review_input_sha256, str)
        or _SHA256_RE.fullmatch(review_input_sha256) is None
    ):
        raise FieldPilotError("review_input_sha256 is invalid")
    normalized_review_sha256 = root.get("review_normalized_sha256")
    if normalized_review_sha256 != canonical_json_sha256(review):
        raise FieldPilotError("review_normalized_sha256 does not match the review")

    project = _validate_offline_report_shape(
        root.get("project_verification"),
        artifact_kind=ARTIFACT_KIND_PROJECT,
        authority=AUTHORITY_SELF_CONTAINED,
        input_name=project_name,
        label="project verification",
    )
    survey = _validate_offline_report_shape(
        root.get("survey_verification"),
        artifact_kind=ARTIFACT_KIND_SURVEY_EXPORT,
        authority=AUTHORITY_MATCHED_PROJECT,
        input_name=survey_name,
        label="survey verification",
    )
    performance = _exact_mapping(
        root.get("performance"),
        {
            "project_verification_ms",
            "survey_verification_ms",
            "total_verification_ms",
        },
        label="field-pilot performance",
    )
    durations: dict[str, int] = {}
    for key in (
        "project_verification_ms",
        "survey_verification_ms",
        "total_verification_ms",
    ):
        duration = performance.get(key)
        if type(duration) is not int or duration < 0:
            raise FieldPilotError(f"performance.{key} must be a non-negative integer")
        durations[key] = duration
    if durations["total_verification_ms"] != (
        durations["project_verification_ms"]
        + durations["survey_verification_ms"]
    ):
        raise FieldPilotError("total verification duration is inconsistent")

    artifact_status = _artifact_verification_status(project, survey)
    project_document_sha256, survey_artifact_set_sha256 = (
        _verified_artifact_digests(project, survey)
    )
    human_status = _human_review_status(
        review,
        project_document_sha256=project_document_sha256,
        survey_artifact_set_sha256=survey_artifact_set_sha256,
        report_created_at_utc=created_at,
    )
    opengl_status = str(opengl["status"])
    windows_status = _windows_runtime_status(machine)
    pilot_status = _pilot_outcome(
        artifact_status=artifact_status,
        human_status=human_status,
        opengl_status=opengl_status,
        windows_status=windows_status,
    )
    outcome = _exact_mapping(
        root.get("outcome"),
        {
            "artifact_verification",
            "human_review",
            "opengl_driver",
            "pilot",
            "scope",
            "windows_runtime",
        },
        label="field-pilot outcome",
    )
    expected_outcome = {
        "artifact_verification": artifact_status,
        "human_review": human_status,
        "opengl_driver": opengl_status,
        "pilot": pilot_status,
        "scope": FIELD_PILOT_SCOPE,
        "windows_runtime": windows_status,
    }
    if dict(outcome) != expected_outcome:
        raise FieldPilotError("field-pilot outcome contradicts its evidence")

    pilot_sha256 = root.get("pilot_sha256")
    if not isinstance(pilot_sha256, str) or _SHA256_RE.fullmatch(pilot_sha256) is None:
        raise FieldPilotError("pilot_sha256 is invalid")
    unsigned = dict(root)
    unsigned.pop("pilot_sha256")
    if canonical_json_sha256(unsigned) != pilot_sha256:
        raise FieldPilotError("pilot_sha256 does not match the report")

    normalized = dict(root)
    normalized.update(
        {
            "application": application,
            "build": build,
            "created_at_utc": created_at,
            "inputs": {
                "opengl_report_name": opengl_name,
                "project_name": project_name,
                "review_name": review_name,
                "survey_name": survey_name,
            },
            "machine": machine,
            "opengl_driver": opengl,
            "outcome": expected_outcome,
            "performance": durations,
            "project_verification": dict(project),
            "review": review,
            "survey_verification": dict(survey),
        }
    )
    return normalized


def field_pilot_report_bytes(value: object) -> bytes:
    report = validate_field_pilot_report(value)
    return canonical_json_bytes(report) + b"\n"


def load_field_pilot_report(
    path: str | os.PathLike[str],
) -> dict[str, Any]:
    source = Path(os.fspath(path))
    payload = _read_regular_file(
        source,
        label="field-pilot report",
        maximum=MAX_FIELD_PILOT_REPORT_BYTES,
    )
    value = _strict_json(payload, label="field-pilot report")
    report = validate_field_pilot_report(value)
    if payload != canonical_json_bytes(report) + b"\n":
        raise FieldPilotError(
            "field-pilot report is not canonical RFC 8785 JSON plus one newline"
        )
    return report


def _atomic_write_json_noreplace(
    destination: Path,
    payload: bytes,
) -> FieldPilotPublication:
    try:
        destination = Path(os.path.abspath(os.fspath(destination.expanduser())))
    except (OSError, TypeError, ValueError) as exc:
        raise FieldPilotError("report destination is invalid") from exc
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise FieldPilotError("report destination parent cannot be created") from exc
    if os.path.lexists(destination):
        raise FieldPilotError("report destination already exists")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".amr-field-pilot-",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    committed = False
    cleanup_warning: str | None = None
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
            committed = True
        except FileExistsError as exc:
            raise FieldPilotError("report destination already exists") from exc
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                raise FieldPilotError("report destination already exists") from exc
            raise FieldPilotError(
                "atomic no-replace report publication is unavailable"
            ) from exc
        try:
            temporary.unlink()
        except OSError:
            cleanup_warning = (
                "report published but its hidden staging hard link could not be removed"
            )
        try:
            published = destination.read_bytes()
        except OSError as exc:
            raise FieldPilotError(
                "report was published but cannot be re-read"
            ) from exc
        if published != payload:
            raise FieldPilotError("published report bytes changed")
        directory_fsync = fsync_export_directory(destination.parent)
        durability_confirmed = directory_fsync and cleanup_warning is None
        warning = cleanup_warning
        if not directory_fsync:
            warning = (
                "report published but directory fsync is unsupported; "
                "crash durability is uncertain"
                if warning is None
                else warning + "; directory fsync is unsupported"
            )
        return FieldPilotPublication(
            path=destination,
            sha256=hashlib.sha256(payload).hexdigest(),
            durability_confirmed=durability_confirmed,
            warning_message=warning,
        )
    finally:
        if not committed or temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


def write_field_pilot_report(
    path: str | os.PathLike[str],
    report: object,
) -> FieldPilotPublication:
    payload = field_pilot_report_bytes(report)
    return _atomic_write_json_noreplace(Path(os.fspath(path)), payload)


def write_field_pilot_review_template(
    path: str | os.PathLike[str],
) -> FieldPilotPublication:
    review = validate_field_pilot_review(default_field_pilot_review())
    payload = canonical_json_bytes(review) + b"\n"
    return _atomic_write_json_noreplace(Path(os.fspath(path)), payload)


def _redacted_failure_message(exc: Exception) -> str:
    if isinstance(exc, FieldPilotError):
        text = str(exc)
    else:
        text = f"{type(exc).__name__}: field-pilot verification failed"
    text = " ".join(text.split())
    return text[:512] or "field-pilot verification failed"


def build_field_pilot_verification_report(
    path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Return a deterministic integrity receipt for one pilot report."""

    input_name = _safe_input_name(path)
    base: dict[str, Any] = {
        "format": FIELD_PILOT_VERIFICATION_FORMAT,
        "input_name": input_name,
        "ok": False,
        "schema_version": FIELD_PILOT_SCHEMA_VERSION,
    }
    try:
        report = load_field_pilot_report(path)
        project_evidence = report["project_verification"].get("evidence") or {}
        survey_evidence = report["survey_verification"].get("evidence") or {}
        opengl = report["opengl_driver"]
        base["ok"] = True
        base["evidence"] = {
            "artifact_document_sha256": project_evidence.get("document_sha256"),
            "opengl_report_sha256": opengl.get("sha256"),
            "pilot_outcome": report["outcome"]["pilot"],
            "pilot_sha256": report["pilot_sha256"],
            "review_normalized_sha256": report["review_normalized_sha256"],
            "scope": FIELD_PILOT_SCOPE,
            "survey_artifact_set_sha256": survey_evidence.get(
                "artifact_set_sha256"
            ),
        }
        return base
    except Exception as exc:
        base["error"] = {
            "code": "verification_failed",
            "message": _redacted_failure_message(exc),
        }
        return base


__all__ = [
    "FIELD_PILOT_CHECKS",
    "FIELD_PILOT_RELEASE_CLAIM",
    "FIELD_PILOT_REPORT_FORMAT",
    "FIELD_PILOT_REVIEW_FORMAT",
    "FIELD_PILOT_SCHEMA_VERSION",
    "FIELD_PILOT_SCOPE",
    "FIELD_PILOT_VERIFICATION_FORMAT",
    "FieldPilotError",
    "FieldPilotPublication",
    "build_field_pilot_report",
    "build_field_pilot_verification_report",
    "default_field_pilot_review",
    "field_pilot_machine_snapshot",
    "field_pilot_report_bytes",
    "load_field_pilot_report",
    "load_field_pilot_review",
    "validate_field_pilot_report",
    "validate_field_pilot_review",
    "write_field_pilot_report",
    "write_field_pilot_review_template",
]
