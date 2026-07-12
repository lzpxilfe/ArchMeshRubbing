"""Compatibility entry point for the safe cross-platform native build tool.

The historical script deleted ``build/`` and ``dist/`` and created a Windows
desktop shortcut automatically.  Those side effects are intentionally retired.
Use ``tools/build_native.py`` directly for new automation.
"""

from __future__ import annotations

import sys

from tools.build_native import main as build_native_main


def main() -> int:
    print(
        "build_and_shortcut.py is deprecated; delegating to "
        "tools/build_native.py without creating a shortcut.",
        file=sys.stderr,
    )
    return build_native_main()


if __name__ == "__main__":
    raise SystemExit(main())
