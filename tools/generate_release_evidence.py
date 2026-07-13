"""Generate or verify deterministic evidence for a frozen payload."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.release_evidence import (  # noqa: E402
    EVIDENCE_DIRECTORY_NAME,
    ReleaseEvidenceError,
    generate_release_evidence,
    verify_release_evidence,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--payload", type=Path, required=True)
    generate.add_argument("--created-at", required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--payload", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            result = generate_release_evidence(
                args.payload,
                args.payload / EVIDENCE_DIRECTORY_NAME,
                created_at=args.created_at,
            )
        else:
            result = verify_release_evidence(args.payload)
    except ReleaseEvidenceError as exc:
        parser.exit(1, f"release evidence failed: {exc}\n")
    print(result.detail())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
