"""Inspect or enforce the fail-closed public binary release policy."""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path
import re
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.public_release_policy import (  # noqa: E402
    PUBLIC_RELEASE_POLICY_PARTS,
    PublicReleasePolicyError,
    load_public_release_policy,
    require_public_binary_distribution,
    verify_project_license,
    verify_runtime_license_observations,
)


def _canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _installed_observations(
    canonical_names: Sequence[str],
) -> dict[str, tuple[str, str | None]]:
    result: dict[str, tuple[str, str | None]] = {}
    for canonical_name in canonical_names:
        try:
            metadata = importlib.metadata.metadata(canonical_name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise PublicReleasePolicyError(
                f"runtime distribution is not installed: {canonical_name}"
            ) from exc
        observed_name = str(metadata.get("Name") or canonical_name)
        key = _canonical_name(observed_name)
        if key != canonical_name:
            raise PublicReleasePolicyError(
                f"runtime metadata name differs from policy: {observed_name}"
            )
        result[key] = (
            str(metadata.get("Version") or ""),
            (
                str(metadata.get("License-Expression")).strip()
                if metadata.get("License-Expression")
                else None
            ),
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify the ArchMeshRubbing public binary release policy."
    )
    parser.add_argument(
        "command",
        choices=("status", "assert-blocked", "require-public"),
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=ROOT.joinpath(*PUBLIC_RELEASE_POLICY_PARTS),
    )
    parser.add_argument("--license", type=Path, default=ROOT / "LICENSE")
    args = parser.parse_args(argv)
    try:
        policy, _raw = load_public_release_policy(args.policy)
        verify_project_license(policy, args.license)
        observations = _installed_observations(
            [item.canonical_name for item in policy.runtime_license_observations]
        )
        verify_runtime_license_observations(policy, observations)
        if args.command == "require-public":
            require_public_binary_distribution(policy)
        elif args.command == "assert-blocked" and policy.decision != "blocked":
            raise PublicReleasePolicyError(
                "public binary release policy unexpectedly permits distribution"
            )
    except PublicReleasePolicyError as exc:
        print(f"Public release policy failed: {exc}", file=sys.stderr)
        return 2
    print(policy.detail())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
