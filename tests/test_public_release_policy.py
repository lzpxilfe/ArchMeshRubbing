from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.public_release_policy import (
    PublicReleasePolicyError,
    canonical_json_bytes,
    load_public_release_policy,
    require_public_binary_distribution,
    verify_project_license,
    verify_runtime_license_observations,
)


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "requirements" / "public-release-policy.json"


def test_committed_public_release_policy_binds_license_and_runtime_lock() -> None:
    policy, raw = load_public_release_policy(POLICY_PATH)
    verify_project_license(policy, ROOT / "LICENSE")
    verify_runtime_license_observations(
        policy,
        {"pyqt6": ("6.11.0", "GPL-3.0-only")},
    )

    assert raw in {
        canonical_json_bytes(json.loads(raw.decode("utf-8"))),
        canonical_json_bytes(json.loads(raw.decode("utf-8"))) + b"\n",
    }
    assert policy.decision == "blocked"
    assert policy.rights_holder_authorization == "not-recorded"
    assert policy.project_license.expression == "GPL-2.0-only"
    assert policy.project_license.sha256 == hashlib.sha256(
        (ROOT / "LICENSE").read_bytes()
    ).hexdigest()
    assert policy.runtime_license_observations[0].license_expression == (
        "GPL-3.0-only"
    )
    assert "public-binary=blocked" in policy.detail()
    with pytest.raises(PublicReleasePolicyError, match="blocked by policy"):
        require_public_binary_distribution(policy)


def test_public_release_policy_rejects_permissive_or_noncanonical_variants(
    tmp_path: Path,
) -> None:
    value = json.loads(POLICY_PATH.read_text("utf-8"))
    value["decision"] = "allowed"
    path = tmp_path / "allowed.json"
    path.write_bytes(canonical_json_bytes(value))
    with pytest.raises(PublicReleasePolicyError, match="cannot authorize"):
        load_public_release_policy(path)

    value["decision"] = "blocked"
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    with pytest.raises(PublicReleasePolicyError, match="not canonical"):
        load_public_release_policy(path)


def test_public_release_policy_detects_license_and_metadata_drift(
    tmp_path: Path,
) -> None:
    policy, _raw = load_public_release_policy(POLICY_PATH)
    altered_license = tmp_path / "LICENSE"
    altered_license.write_bytes((ROOT / "LICENSE").read_bytes() + b"changed\n")
    with pytest.raises(PublicReleasePolicyError, match="SHA-256"):
        verify_project_license(policy, altered_license)
    with pytest.raises(PublicReleasePolicyError, match="runtime version"):
        verify_runtime_license_observations(
            policy,
            {"pyqt6": ("6.10.2", "GPL-3.0-only")},
        )
    with pytest.raises(PublicReleasePolicyError, match="License-Expression"):
        verify_runtime_license_observations(
            policy,
            {"pyqt6": ("6.11.0", "LGPL-3.0-only")},
        )
