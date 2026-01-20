import os
import subprocess
import shutil
from pathlib import Path

def create_shortcut(target_exe, shortcut_name):
    """Windows Desktop shortcut creator."""
    desktop = Path(os.path.expanduser("~")) / "Desktop"
    shortcut_path = desktop / f"{shortcut_name}.lnk"
    
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
