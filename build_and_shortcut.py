import os
import subprocess
import shutil
import ctypes
from pathlib import Path


def get_desktop_dir() -> Path:
    """Return Desktop path without assuming a literal 'Desktop' folder name."""
    if os.name == "nt":
        csidl_desktopdirectory = 0x0010
        shgfp_type_current = 0
        buf = ctypes.create_unicode_buffer(260)
        result = ctypes.windll.shell32.SHGetFolderPathW(
            None,
            csidl_desktopdirectory,
            None,
            shgfp_type_current,
            buf,
        )
        if result == 0 and buf.value:
            return Path(buf.value)
    return Path.home() / "Desktop"


def delete_old_shortcuts(shortcut_name):
    """Delete existing shortcuts matching the name pattern on Desktop."""
    desktop = get_desktop_dir()
    
    # Find all shortcuts that start with the app name
    deleted_count = 0
    for shortcut in desktop.glob(f"{shortcut_name}*.lnk"):
        try:
            shortcut.unlink()
            print(f"Deleted old shortcut: {shortcut.name}")
            deleted_count += 1
        except Exception as e:
            print(f"Warning: Could not delete {shortcut.name}: {e}")
    
    if deleted_count == 0:
        print("No old shortcuts found.")
    else:
        print(f"Deleted {deleted_count} old shortcut(s).")

def create_shortcut(target_exe, shortcut_name):
    """Windows Desktop shortcut creator."""
    desktop = get_desktop_dir()
    shortcut_path = desktop / f"{shortcut_name}.lnk"
    
    # Delete old shortcuts first
    delete_old_shortcuts(shortcut_name)
    
    powershell_cmd = f"""
    $s = New-Object -ComObject WScript.Shell
    $shortcut = $s.CreateShortcut('{shortcut_path}')
    $shortcut.TargetPath = '{target_exe}'
    $shortcut.WorkingDirectory = '{os.path.dirname(target_exe)}'
    $shortcut.Save()
    """
    
    try:
        subprocess.run(["powershell", "-Command", powershell_cmd], check=True)
        print(f"Success: Shortcut created at {shortcut_path}")
    except Exception as e:
        print(f"Error: Failed to create shortcut: {e}")

def main():
    project_dir = Path(__file__).parent.absolute()
    spec_file = project_dir / "ArchMeshRubbing.spec"
    dist_dir = project_dir / "dist"
    exe_path = dist_dir / "ArchMeshRubbing.exe"
    
    print("Cleaning old build artifacts...")
    for folder in ["build", "dist"]:
        path = project_dir / folder
        if path.exists():
            print(f"Removing {path}...")
            shutil.rmtree(path)

    print("Building (PyInstaller)...")
    try:
        subprocess.run(["pyinstaller", "--noconfirm", str(spec_file)], check=True)
        print("Build Complete!")
        
        if exe_path.exists():
            print("Creating Desktop Shortcut...")
            create_shortcut(str(exe_path), "ArchMeshRubbing")
        else:
            print(f"Error: EXE not found at {exe_path}")
            
    except subprocess.CalledProcessError as e:
        print(f"Error during build: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
