"""Fail-closed Windows client runtime contract shared by product entrypoints.

The supported product runtime is deliberately narrower than ``sys.platform ==
"win32"``: Windows 10 build 17763 or newer (including Windows 11), workstation
editions only, a native AMD64 host, a 64-bit AMD64 process, and CPython 3.12.
Wine/Proton compatibility layers are excluded. Collection and evaluation are
separate so durable evidence can be evaluated deterministically without
consulting the machine that later reads it.
"""

from __future__ import annotations

from collections.abc import Mapping
import ctypes
import platform
import re
import struct
import sys
from typing import Any


MINIMUM_WINDOWS_CLIENT_BUILD = 17_763
SUPPORTED_PYTHON = (3, 12)
WINDOWS_WORKSTATION_PRODUCT_TYPE = 1
WINDOWS_NATIVE_AMD64 = "AMD64"

_IMAGE_FILE_MACHINE_NAMES = {
    0x014C: "x86",
    0x0200: "IA64",
    0x8664: WINDOWS_NATIVE_AMD64,
    0xAA64: "ARM64",
}
_PYTHON_VERSION_RE = re.compile(r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)$")


class UnsupportedWindowsRuntimeError(RuntimeError):
    """The live process does not meet the supported Windows GUI contract."""


def _windows_version_claims() -> tuple[int | None, int | None, int | None, int | None]:
    if sys.platform != "win32":
        return None, None, None, None
    getwindowsversion = getattr(sys, "getwindowsversion", None)
    if not callable(getwindowsversion):
        return None, None, None, None
    try:
        value = getwindowsversion()
        major = int(value.major)
        minor = int(value.minor)
        build = int(value.build)
        product_type = int(value.product_type)
    except (AttributeError, OSError, TypeError, ValueError):
        return None, None, None, None
    if major < 0 or minor < 0 or build <= 0 or product_type not in {1, 2, 3}:
        return None, None, None, None
    return major, minor, build, product_type


def _windows_native_machine() -> str | None:
    """Read the native host architecture, detecting x64-on-ARM64 emulation."""

    if sys.platform != "win32":
        return None
    win_dll = getattr(ctypes, "WinDLL", None)
    if win_dll is None:
        return None
    try:
        kernel32: Any = win_dll("kernel32", use_last_error=True)
        get_current_process = kernel32.GetCurrentProcess
        get_current_process.argtypes = []
        get_current_process.restype = ctypes.c_void_p
        is_wow64_process2 = kernel32.IsWow64Process2
        is_wow64_process2.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ushort),
            ctypes.POINTER(ctypes.c_ushort),
        ]
        is_wow64_process2.restype = ctypes.c_int
        process_machine = ctypes.c_ushort()
        native_machine = ctypes.c_ushort()
        if not is_wow64_process2(
            get_current_process(),
            ctypes.byref(process_machine),
            ctypes.byref(native_machine),
        ):
            return None
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    return _IMAGE_FILE_MACHINE_NAMES.get(
        int(native_machine.value),
        f"unknown-0x{int(native_machine.value):04x}",
    )


def _windows_compatibility_layer() -> str | None:
    """Return ``wine`` when the Win32 process is hosted by Wine/Proton."""

    if sys.platform != "win32":
        return None
    win_dll = getattr(ctypes, "WinDLL", None)
    if win_dll is None:
        return None
    try:
        ntdll: Any = win_dll("ntdll", use_last_error=True)
    except (OSError, TypeError, ValueError):
        return None
    try:
        getattr(ntdll, "wine_get_version")
    except AttributeError:
        return "none"
    return "wine"


def collect_windows_runtime_claims() -> dict[str, object]:
    """Collect the bounded claims needed to evaluate the Windows contract."""

    major, minor, build, product_type = _windows_version_claims()
    return {
        "machine": platform.machine() or "unknown",
        "process_bits": struct.calcsize("P") * 8,
        "python_implementation": platform.python_implementation() or "unknown",
        "python_version": platform.python_version() or "unknown",
        "release": platform.release() or "unknown",
        "system": platform.system() or "unknown",
        "windows_build_number": build,
        "windows_compatibility_layer": _windows_compatibility_layer(),
        "windows_major_version": major,
        "windows_minor_version": minor,
        "windows_native_machine": _windows_native_machine(),
        "windows_product_type": product_type,
    }


def windows_client_runtime_failures(claims: Mapping[str, object]) -> tuple[str, ...]:
    """Return stable reason codes for every unsupported or missing claim."""

    failures: list[str] = []
    if claims.get("system") != "Windows":
        failures.append("not-windows")

    machine = claims.get("machine")
    if not isinstance(machine, str) or machine.casefold() not in {"amd64", "x86_64"}:
        failures.append("process-not-amd64")
    process_bits = claims.get("process_bits")
    if type(process_bits) is not int or process_bits != 64:
        failures.append("process-not-64-bit")

    if claims.get("python_implementation") != "CPython":
        failures.append("python-not-cpython")
    python_version = claims.get("python_version")
    match = (
        _PYTHON_VERSION_RE.fullmatch(python_version)
        if isinstance(python_version, str)
        else None
    )
    if match is None or (
        int(match.group("major")),
        int(match.group("minor")),
    ) != SUPPORTED_PYTHON:
        failures.append("python-not-3.12")

    major = claims.get("windows_major_version")
    minor = claims.get("windows_minor_version")
    build = claims.get("windows_build_number")
    if (
        type(major) is not int
        or type(minor) is not int
        or type(build) is not int
        or (major, minor) != (10, 0)
        or build < MINIMUM_WINDOWS_CLIENT_BUILD
    ):
        failures.append("windows-version-unsupported")
    if claims.get("windows_product_type") != WINDOWS_WORKSTATION_PRODUCT_TYPE:
        failures.append("windows-server-or-unknown")
    if claims.get("windows_native_machine") != WINDOWS_NATIVE_AMD64:
        failures.append("native-machine-not-amd64")
    if claims.get("windows_compatibility_layer") != "none":
        failures.append("windows-compatibility-layer")
    return tuple(failures)


def is_supported_windows_client_runtime(claims: Mapping[str, object]) -> bool:
    """Return whether stored or live claims meet the full product contract."""

    return not windows_client_runtime_failures(claims)


def require_supported_windows_client_runtime() -> dict[str, object]:
    """Return live claims or raise with stable fail-closed reason codes."""

    claims = collect_windows_runtime_claims()
    failures = windows_client_runtime_failures(claims)
    if failures:
        raise UnsupportedWindowsRuntimeError(
            "ArchMeshRubbing GUI requires Windows 10 build 17763+ or Windows 11 "
            "Workstation on native AMD64 with 64-bit CPython 3.12; unsupported "
            "runtime: "
            + ", ".join(failures)
        )
    return claims


__all__ = [
    "MINIMUM_WINDOWS_CLIENT_BUILD",
    "SUPPORTED_PYTHON",
    "UnsupportedWindowsRuntimeError",
    "WINDOWS_NATIVE_AMD64",
    "WINDOWS_WORKSTATION_PRODUCT_TYPE",
    "collect_windows_runtime_claims",
    "is_supported_windows_client_runtime",
    "require_supported_windows_client_runtime",
    "windows_client_runtime_failures",
]
