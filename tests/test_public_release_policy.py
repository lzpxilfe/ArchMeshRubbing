from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.public_release_policy import (
    PublicReleasePolicyError,
    canonical_json_bytes,
    derive_combined_work_expression,
    load_public_release_policy,
    require_public_binary_distribution,
    verify_combined_work_license,
    verify_project_license,
    verify_runtime_license_observations,
)


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "requirements" / "public-release-policy.json"


def test_committed_public_release_policy_binds_license_and_runtime_lock() -> None:
    policy, raw = load_public_release_policy(POLICY_PATH)
    verify_project_license(policy, ROOT / "LICENSE")
    verify_combined_work_license(policy, ROOT)
    verify_runtime_license_observations(
        policy,
        {"pyqt6": ("6.11.0", "GPL-3.0-only")},
    )

    assert raw in {
        canonical_json_bytes(json.loads(raw.decode("utf-8"))),
        canonical_json_bytes(json.loads(raw.decode("utf-8"))) + b"\n",
    }
    assert policy.schema_version == "1.1.0"
    assert policy.decision == "allowed"
    assert policy.allows_public_binary is True
    assert policy.rights_holder_authorization != "not-recorded"
    assert policy.project_license.expression == "Apache-2.0"
    assert policy.project_license.sha256 == hashlib.sha256(
        (ROOT / "LICENSE").read_bytes()
    ).hexdigest()
    assert policy.combined_work_license is not None
    assert policy.combined_work_license.expression == "GPL-3.0-only"
    assert policy.combined_work_license.sha256 == hashlib.sha256(
        (ROOT / policy.combined_work_license.path).read_bytes()
    ).hexdigest()
    assert policy.runtime_license_observations[0].license_expression == (
        "GPL-3.0-only"
    )
    assert "public-binary=allowed" in policy.detail()
    assert "combined=GPL-3.0-only" in policy.detail()
    require_public_binary_distribution(policy)


def test_combined_work_expression_is_derived_not_declared() -> None:
    # A permissive source bundled with a copyleft runtime is conveyed under
    # the runtime's terms.
    assert (
        derive_combined_work_expression("Apache-2.0", ("GPL-3.0-only",))
        == "GPL-3.0-only"
    )
    assert derive_combined_work_expression("MIT", ("MIT", "BSD-3-Clause")) == "MIT"
    assert (
        derive_combined_work_expression("Apache-2.0", ("MIT", "LGPL-3.0-only"))
        == "LGPL-3.0-only"
    )
    # GPLv2-only cannot absorb a GPLv3 obligation at all.
    with pytest.raises(PublicReleasePolicyError, match="cannot be combined"):
        derive_combined_work_expression("GPL-2.0-only", ("GPL-3.0-only",))
    # Unknown expressions fail closed instead of being guessed at.
    with pytest.raises(PublicReleasePolicyError, match="conveyance table"):
        derive_combined_work_expression("Apache-2.0", ("Proprietary",))
    with pytest.raises(PublicReleasePolicyError, match="conveyance table"):
        derive_combined_work_expression("SomeoneElsesLicense", ("MIT",))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        pytest.param(
            lambda value: value.__setitem__(
                "project_license",
                {**value["project_license"], "expression": "GPL-2.0-only"},
            ),
            "cannot be combined",
            id="gpl2-only-source-cannot-ship-gpl3-runtime",
        ),
        pytest.param(
            lambda value: value.__setitem__(
                "rights_holder_authorization", "not-recorded"
            ),
            "must record rights-holder authorization",
            id="allowed-requires-recorded-authorization",
        ),
        pytest.param(
            lambda value: value.__setitem__(
                "combined_work_license",
                {**value["combined_work_license"], "expression": "MIT"},
            ),
            "does not match the expression",
            id="declared-conveyance-must-match-derivation",
        ),
        pytest.param(
            lambda value: value.__setitem__(
                "runtime_license_observations",
                [
                    {
                        **value["runtime_license_observations"][0],
                        "license_expression": "Proprietary",
                    }
                ],
            ),
            "conveyance table",
            id="unknown-runtime-expression-fails-closed",
        ),
    ],
)
def test_allowed_policy_cannot_be_forged_by_editing_one_field(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    value = json.loads(POLICY_PATH.read_text("utf-8"))
    mutate(value)
    path = tmp_path / "forged.json"
    path.write_bytes(canonical_json_bytes(value))
    with pytest.raises(PublicReleasePolicyError, match=message):
        load_public_release_policy(path)


def test_public_release_policy_rejects_noncanonical_or_legacy_permissive(
    tmp_path: Path,
) -> None:
    value = json.loads(POLICY_PATH.read_text("utf-8"))
    path = tmp_path / "noncanonical.json"
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    with pytest.raises(PublicReleasePolicyError, match="not canonical"):
        load_public_release_policy(path)

    # Schema 1.0.0 still refuses to authorize distribution, so an archived
    # policy cannot be re-read as permissive.
    legacy = {
        key: value[key] for key in value if key != "combined_work_license"
    }
    legacy["schema_version"] = "1.0.0"
    legacy["decision"] = "allowed"
    legacy_path = tmp_path / "legacy-allowed.json"
    legacy_path.write_bytes(canonical_json_bytes(legacy))
    with pytest.raises(PublicReleasePolicyError, match="cannot authorize"):
        load_public_release_policy(legacy_path)


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


def test_combined_work_license_text_must_match_policy(tmp_path: Path) -> None:
    policy, _raw = load_public_release_policy(POLICY_PATH)
    assert policy.combined_work_license is not None
    fake_root = tmp_path / "root"
    target = fake_root / policy.combined_work_license.path
    target.parent.mkdir(parents=True)
    target.write_bytes(b"not the GPL\n")
    with pytest.raises(PublicReleasePolicyError, match="SHA-256"):
        verify_combined_work_license(policy, fake_root)
