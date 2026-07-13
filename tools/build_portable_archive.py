"""Build, verify, or safely extract an ArchMeshRubbing portable ZIP."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.portable_archive import (  # noqa: E402
    PortableArchiveError,
    build_portable_archive,
    extract_portable_archive,
    verify_portable_archive,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--payload", type=Path, required=True)
    build.add_argument("--archive", type=Path, required=True)
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument("--source-date-epoch", type=int, required=True)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--archive", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)

    extract = subparsers.add_parser("extract")
    extract.add_argument("--archive", type=Path, required=True)
    extract.add_argument("--manifest", type=Path, required=True)
    extract.add_argument("--destination", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            result = build_portable_archive(
                args.payload,
                args.archive,
                args.manifest,
                source_date_epoch=args.source_date_epoch,
            )
        elif args.command == "verify":
            result = verify_portable_archive(args.archive, args.manifest)
        else:
            result = extract_portable_archive(
                args.archive,
                args.manifest,
                args.destination,
            )
    except PortableArchiveError as exc:
        parser.exit(1, f"portable archive failed: {exc}\n")
    print(result.detail())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
