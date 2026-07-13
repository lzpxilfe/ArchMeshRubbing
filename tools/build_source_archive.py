"""Build or verify the exact corresponding-source ZIP for one Git commit."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.source_archive import (  # noqa: E402
    SourceArchiveError,
    build_source_archive,
    verify_source_archive,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--repository", type=Path, default=ROOT)
    build.add_argument("--archive", type=Path, required=True)
    build.add_argument("--sidecar", type=Path, required=True)
    build.add_argument("--commit")

    verify = subparsers.add_parser("verify")
    verify.add_argument("--archive", type=Path, required=True)
    verify.add_argument("--sidecar", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            result = build_source_archive(
                args.repository,
                args.archive,
                args.sidecar,
                commit=args.commit,
            )
        else:
            result = verify_source_archive(args.archive, args.sidecar)
    except SourceArchiveError as exc:
        parser.exit(1, f"source archive failed: {exc}\n")
    print(result.detail())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
