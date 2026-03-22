"""
Compatibility launcher for the interactive application.

The old lightweight GUI had drifted away from the main recording-surface
workflow. Keep `python app_gui.py` working by forwarding to the maintained
interactive application instead of carrying a second interface.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from main import launch_gui  # noqa: E402


def main() -> None:
    launch_gui()


if __name__ == "__main__":
    main()
