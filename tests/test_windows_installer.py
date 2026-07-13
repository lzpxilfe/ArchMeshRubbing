from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INSTALLER = ROOT / "installer" / "ArchMeshRubbing.iss"


def _script() -> str:
    return INSTALLER.read_text(encoding="utf-8")


def test_installer_requires_build_identity_and_has_stable_app_id() -> None:
    script = _script()

    for symbol in ("AppVersion", "SourceDir", "OutputDir", "SourceCommit"):
        assert f"#ifndef {symbol}" in script
        assert f"/D{symbol}=" in script

    assert "AppId={{B274D884-EE82-4B03-BB97-8EE62B89323A}" in script
    assert script.count("AppId=") == 1
    assert (
        "AppComments=Unsigned verification build from source commit "
        "{#SourceCommit}" in script
    )


def test_installer_is_windows_x64_per_user_and_explicitly_unsigned() -> None:
    script = _script()

    assert "MinVersion=10.0" in script
    assert "ArchitecturesAllowed=x64compatible" in script
    assert "ArchitecturesInstallIn64BitMode=x64compatible" in script
    assert "PrivilegesRequired=lowest" in script
    assert "DefaultDirName={localappdata}\\Programs\\ArchMeshRubbing" in script
    assert (
        "OutputBaseFilename=ArchMeshRubbing-{#AppVersion}-Windows-x64-unsigned"
        in script
    )
    assert "SignTool=" not in script
    assert "SignedUninstaller=yes" not in script


def test_installer_copies_only_frozen_payload_and_adds_no_runtime_actions() -> None:
    script = _script()

    assert (
        'Source: "{#SourceDir}\\*"; DestDir: "{app}"; '
        "Flags: ignoreversion recursesubdirs createallsubdirs" in script
    )
    assert "LicenseFile={#SourceDir}\\_internal\\LICENSE" in script
    assert "AppReadmeFile={app}\\_internal\\README.md" in script
    assert "[Run]" not in script
    assert "[UninstallRun]" not in script
    assert "[Registry]" not in script
    assert "ChangesAssociations=no" in script
    assert "downloadtemporaryfile" not in script.lower()


def test_installer_creates_only_a_start_menu_shortcut() -> None:
    script = _script()

    assert (
        'Name: "{autoprograms}\\ArchMeshRubbing\\ArchMeshRubbing"' in script
    )
    assert "{autodesktop}" not in script
    assert "{commondesktop}" not in script
    assert "{userdesktop}" not in script
