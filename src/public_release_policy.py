"""Fail-closed policy for public Windows binary distribution.

This module records and verifies release-policy facts.  It deliberately does
not decide copyright ownership or provide legal advice.  Schema 1.0.0 only
supports a blocked public-binary decision; enabling distribution requires an
intentional code and policy revision after the relevant rights are confirmed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping
from urllib.parse import urlsplit


PUBLIC_RELEASE_POLICY_SCHEMA_VERSION = "1.0.0"
PUBLIC_RELEASE_POLICY_PARTS = ("requirements", "public-release-policy.json")

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_CANONICAL_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_REASON_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class PublicReleasePolicyError(RuntimeError):
    """Raised when release policy evidence is absent, altered, or insufficient."""


@dataclass(frozen=True, slots=True)
class ProjectLicense:
    expression: str
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class RuntimeLicenseObservation:
    canonical_name: str
    evidence_field: str
    license_expression: str
    version: str


@dataclass(frozen=True, slots=True)
class PolicySource:
    label: str
    url: str


@dataclass(frozen=True, slots=True)
class PublicReleasePolicy:
    artifact_scope: str
    decision: str
    legal_advice: bool
    project_license: ProjectLicense
    reason_code: str
    reviewed_on: str
    rights_holder_authorization: str
    runtime_license_observations: tuple[RuntimeLicenseObservation, ...]
    schema_version: str
    sources: tuple[PolicySource, ...]

    def detail(self) -> str:
        observations = ",".join(
            f"{item.canonical_name}=={item.version}:{item.license_expression}"
            for item in self.runtime_license_observations
        )
        return (
            f"public-binary={self.decision}; reason={self.reason_code}; "
            f"authorization={self.rights_holder_authorization}; "
            f"project={self.project_license.expression}; runtime={observations}"
        )


def canonical_json_bytes(value: object) -> bytes:
    """Serialize policy JSON with one stable byte representation."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _nonempty_string(value: object, *, label: str, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise PublicReleasePolicyError(f"{label} is invalid")
    if any(ord(character) < 0x20 for character in value):
        raise PublicReleasePolicyError(f"{label} contains a control character")
    return value


def _exact_object(value: object, fields: set[str], *, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise PublicReleasePolicyError(f"{label} fields are invalid")
    return value


def _parse_project_license(value: object) -> ProjectLicense:
    item = _exact_object(
        value,
        {"expression", "path", "sha256"},
        label="project license",
    )
    expression = _nonempty_string(item["expression"], label="project license expression")
    path = _nonempty_string(item["path"], label="project license path")
    parsed_path = Path(path)
    if (
        "\\" in path
        or parsed_path.is_absolute()
        or parsed_path.as_posix() != path
        or any(part in {"", ".", ".."} for part in parsed_path.parts)
    ):
        raise PublicReleasePolicyError("project license path is not portable")
    sha256 = _nonempty_string(item["sha256"], label="project license SHA-256")
    if _HASH_RE.fullmatch(sha256) is None:
        raise PublicReleasePolicyError("project license SHA-256 is invalid")
    return ProjectLicense(expression=expression, path=path, sha256=sha256)


def _parse_observations(value: object) -> tuple[RuntimeLicenseObservation, ...]:
    if not isinstance(value, list) or not value:
        raise PublicReleasePolicyError("runtime license observations are invalid")
    observations: list[RuntimeLicenseObservation] = []
    for value_item in value:
        item = _exact_object(
            value_item,
            {"canonical_name", "evidence_field", "license_expression", "version"},
            label="runtime license observation",
        )
        canonical_name = _nonempty_string(
            item["canonical_name"], label="runtime canonical name"
        )
        if _CANONICAL_NAME_RE.fullmatch(canonical_name) is None:
            raise PublicReleasePolicyError("runtime canonical name is invalid")
        evidence_field = _nonempty_string(
            item["evidence_field"], label="runtime license evidence field"
        )
        if evidence_field != "License-Expression":
            raise PublicReleasePolicyError(
                "runtime license observation must use License-Expression metadata"
            )
        observations.append(
            RuntimeLicenseObservation(
                canonical_name=canonical_name,
                evidence_field=evidence_field,
                license_expression=_nonempty_string(
                    item["license_expression"],
                    label="runtime license expression",
                ),
                version=_nonempty_string(item["version"], label="runtime version"),
            )
        )
    names = [item.canonical_name for item in observations]
    if names != sorted(names) or len(names) != len(set(names)):
        raise PublicReleasePolicyError(
            "runtime license observations must be unique and sorted"
        )
    return tuple(observations)


def _parse_sources(value: object) -> tuple[PolicySource, ...]:
    if not isinstance(value, list) or not value:
        raise PublicReleasePolicyError("public release policy sources are invalid")
    sources: list[PolicySource] = []
    for value_item in value:
        item = _exact_object(value_item, {"label", "url"}, label="policy source")
        label = _nonempty_string(item["label"], label="policy source label")
        url = _nonempty_string(item["url"], label="policy source URL", maximum=2048)
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or not parsed.netloc
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise PublicReleasePolicyError("policy source must be an HTTPS URL")
        sources.append(PolicySource(label=label, url=url))
    urls = [item.url for item in sources]
    if urls != sorted(urls) or len(urls) != len(set(urls)):
        raise PublicReleasePolicyError("policy sources must be unique and URL-sorted")
    return tuple(sources)


def load_public_release_policy(
    path: Path,
) -> tuple[PublicReleasePolicy, bytes]:
    """Load one canonical policy document and reject any permissive v1 variant."""

    if path.is_symlink() or not path.is_file():
        raise PublicReleasePolicyError("public release policy is missing or linked")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise PublicReleasePolicyError("public release policy is unreadable") from exc
    if not raw or len(raw) > 256 * 1024:
        raise PublicReleasePolicyError("public release policy size is invalid")
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublicReleasePolicyError(
            "public release policy is not strict UTF-8 JSON"
        ) from exc
    fields = {
        "artifact_scope",
        "decision",
        "legal_advice",
        "project_license",
        "reason_code",
        "reviewed_on",
        "rights_holder_authorization",
        "runtime_license_observations",
        "schema_version",
        "sources",
    }
    root = _exact_object(value, fields, label="public release policy")
    canonical = canonical_json_bytes(root)
    if raw not in {canonical, canonical + b"\n"}:
        raise PublicReleasePolicyError("public release policy is not canonical JSON")
    schema_version = _nonempty_string(
        root["schema_version"], label="public release policy schema"
    )
    if schema_version != PUBLIC_RELEASE_POLICY_SCHEMA_VERSION:
        raise PublicReleasePolicyError("public release policy schema is unsupported")
    artifact_scope = _nonempty_string(root["artifact_scope"], label="artifact scope")
    if artifact_scope != "windows-x64-pyinstaller-portable":
        raise PublicReleasePolicyError("public release policy scope is unsupported")
    decision = _nonempty_string(root["decision"], label="public release decision")
    if decision != "blocked":
        raise PublicReleasePolicyError(
            "schema 1.0.0 cannot authorize public binary distribution"
        )
    if root["legal_advice"] is not False:
        raise PublicReleasePolicyError("policy must state that it is not legal advice")
    reason_code = _nonempty_string(root["reason_code"], label="policy reason code")
    if _REASON_RE.fullmatch(reason_code) is None:
        raise PublicReleasePolicyError("policy reason code is invalid")
    reviewed_on = _nonempty_string(root["reviewed_on"], label="policy review date")
    try:
        date.fromisoformat(reviewed_on)
    except ValueError as exc:
        raise PublicReleasePolicyError("policy review date is invalid") from exc
    authorization = _nonempty_string(
        root["rights_holder_authorization"],
        label="rights-holder authorization status",
    )
    if authorization != "not-recorded":
        raise PublicReleasePolicyError(
            "schema 1.0.0 requires rights-holder authorization to remain unrecorded"
        )
    return (
        PublicReleasePolicy(
            artifact_scope=artifact_scope,
            decision=decision,
            legal_advice=False,
            project_license=_parse_project_license(root["project_license"]),
            reason_code=reason_code,
            reviewed_on=reviewed_on,
            rights_holder_authorization=authorization,
            runtime_license_observations=_parse_observations(
                root["runtime_license_observations"]
            ),
            schema_version=schema_version,
            sources=_parse_sources(root["sources"]),
        ),
        raw,
    )


def verify_project_license(policy: PublicReleasePolicy, license_path: Path) -> None:
    """Bind the policy's SPDX expression to the exact bundled license bytes."""

    if license_path.is_symlink() or not license_path.is_file():
        raise PublicReleasePolicyError("project license file is missing or linked")
    if license_path.name != Path(policy.project_license.path).name:
        raise PublicReleasePolicyError("project license file path does not match policy")
    try:
        raw = license_path.read_bytes()
    except OSError as exc:
        raise PublicReleasePolicyError("project license file is unreadable") from exc
    if hashlib.sha256(raw).hexdigest() != policy.project_license.sha256:
        raise PublicReleasePolicyError("project license SHA-256 does not match policy")


def verify_runtime_license_observations(
    policy: PublicReleasePolicy,
    observed: Mapping[str, tuple[str, str | None]],
) -> None:
    """Compare pinned package version and License-Expression to actual metadata."""

    for requirement in policy.runtime_license_observations:
        actual = observed.get(requirement.canonical_name)
        if actual is None:
            raise PublicReleasePolicyError(
                "runtime license observation is missing for "
                f"{requirement.canonical_name}"
            )
        version, expression = actual
        if version != requirement.version:
            raise PublicReleasePolicyError(
                f"runtime version differs from policy for {requirement.canonical_name}"
            )
        if expression != requirement.license_expression:
            raise PublicReleasePolicyError(
                "runtime License-Expression differs from policy for "
                f"{requirement.canonical_name}"
            )


def require_public_binary_distribution(policy: PublicReleasePolicy) -> None:
    """Fail unless a future, explicitly implemented schema authorizes release."""

    raise PublicReleasePolicyError(
        "public binary distribution is blocked by policy: " + policy.reason_code
    )
