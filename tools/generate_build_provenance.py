"""Generate or verify an unsigned, offline build-provenance record."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.build_provenance import (  # noqa: E402
    BuildProvenanceError,
    generate_build_provenance,
    github_actions_invocation,
    verify_build_provenance,
)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate")
    _common(generate)
    generate.add_argument("--output", type=Path, required=True)

    verify = subparsers.add_parser("verify")
    _common(verify)
    verify.add_argument("--provenance", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            result = generate_build_provenance(
                args.archive,
                args.manifest,
                args.payload,
                args.output,
                invocation=github_actions_invocation(os.environ),
            )
        else:
            result = verify_build_provenance(
                args.provenance,
                args.archive,
                args.manifest,
                args.payload,
            )
    except BuildProvenanceError as exc:
        parser.exit(1, f"build provenance failed: {exc}\n")
    print(result.detail())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
