"""
ArchMeshRubbing - 3D Mesh Flattening Tool for Archaeological Documentation
怨좉퀬???좊Ъ 3D 硫붿돩 ?됰㈃???꾧뎄

Main entry point
"""

import sys
import os
import logging
from pathlib import Path

# Ensure repository root is on sys.path so "src" is importable.
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.runtime_defaults import DEFAULTS
from src.core.output_paths import (
    rubbing_output_path,
    projection_output_path,
    inner_surface_path,
    outer_surface_path,
)

_LOGGER = logging.getLogger(__name__)
DEFAULT_EXPORT_DPI = DEFAULTS.export_dpi
DEFAULT_RENDER_RESOLUTION = DEFAULTS.render_resolution
DEFAULT_ARAP_MAX_ITERATIONS = DEFAULTS.arap_max_iterations
DEFAULT_MESH_UNIT = "mm"


def run_cli():
    """而ㅻ㎤?쒕씪???명꽣?섏씠???ㅽ뻾"""
    try:
        from src.core.logging_utils import setup_logging

        setup_logging()
    except Exception as e:
        _LOGGER.debug("Failed to initialize logging: %s", e, exc_info=True)

    if len(sys.argv) < 2:
        print_help()
        return
    
    cmd = sys.argv[1]
    
    if cmd == '--help' or cmd == '-h':
        print_help()
        return
    
    if cmd == '--info' and len(sys.argv) > 2:
        show_file_info(sys.argv[2])
        return
    
    if cmd == '--gui':
        launch_gui()
        return

    if cmd == '--open-project' and len(sys.argv) > 2:
        launch_gui(open_project=sys.argv[2])
        return
    
    if cmd == '--flatten' and len(sys.argv) > 2:
        flatten_mesh(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
        return
    
    if cmd == '--project' and len(sys.argv) > 2:
        project_mesh(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
        return
    
    if cmd == '--separate' and len(sys.argv) > 2:
        separate_mesh(sys.argv[2])
        return
    
    # 湲곕낯: ?뚯씪 泥섎━
    if os.path.exists(cmd):
        process_mesh(cmd)
    else:
        print(f"Error: Unknown command or file not found: {cmd}")
        print("Use --help for usage information")


def print_help():
    """?꾩?留?異쒕젰"""
    from src.core.mesh_loader import MeshLoader
    
    print("=" * 60)
    print("ArchMeshRubbing - 3D Mesh Flattening Tool")
    print("怨좉퀬???좊Ъ 3D 硫붿돩 ?됰㈃???꾧뎄")
    print("=" * 60)
    print()
    print("Usage:")
    print("  python main.py <mesh_file>              # Full processing")
    print("  python main.py --info <mesh_file>       # Show file info")
    print("  python main.py --flatten <mesh_file> [output]    # Flatten only")
    print("  python main.py --project <mesh_file> [output]    # Orthographic projection")
    print("  python main.py --separate <mesh_file>   # Separate inner/outer surfaces")
    print("  python main.py --gui                    # Launch GUI (interactive)")
    print("  python main.py --open-project <project.amr>  # Launch GUI and open project")
    print()
    print(f"Supported formats: {list(MeshLoader.SUPPORTED_FORMATS.keys())}")
    print()
    print("Examples:")
    print("  python main.py roof_tile.obj")
    print("  python main.py --flatten roof_tile.ply rubbing.tiff")
    print("  python main.py --project roof_tile.stl planview.png")


def show_file_info(filepath: str):
    """?뚯씪 ?뺣낫 ?쒖떆"""
    from src.core.mesh_loader import MeshLoader
    
    print(f"\nFile Info: {filepath}")
    print("-" * 40)
    
    try:
        loader = MeshLoader(default_unit=DEFAULT_MESH_UNIT)
        info = loader.get_file_info(filepath)
        for key, value in info.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"  Error: {e}")


def process_mesh(filepath: str):
    """硫붿돩 ?꾩껜 泥섎━ (濡쒕뱶 ???됰㈃?????대?吏 ?앹꽦)"""
    from src.core.mesh_loader import MeshLoader
    from src.core.flattener import ARAPFlattener
    from src.core.surface_visualizer import SurfaceVisualizer
    
    print(f"\n{'='*60}")
    print(f"Processing: {filepath}")
    print(f"{'='*60}")
    
    try:
        # 1. 濡쒕뱶
        print("\n[1/4] Loading mesh...")
        loader = MeshLoader(default_unit=DEFAULT_MESH_UNIT)
        mesh = loader.load(filepath)
        
        print(f"      Vertices: {mesh.n_vertices:,}")
        print(f"      Faces: {mesh.n_faces:,}")
        print(f"      Surface Area: {mesh.surface_area:,.2f} {mesh.unit}^2")
        print(f"      Size: {mesh.extents[0]:.1f} x {mesh.extents[1]:.1f} x {mesh.extents[2]:.1f} {mesh.unit}")
        print(f"      Has Texture: {mesh.has_texture}")
        
        # 2. Flatten
        print("\n[2/4] Flattening mesh (ARAP algorithm)...")
        flattener = ARAPFlattener(max_iterations=DEFAULT_ARAP_MAX_ITERATIONS)
        flattened = flattener.flatten(mesh)
        
        print(f"      Flattened size: {flattened.width:.2f} x {flattened.height:.2f} {mesh.unit}")
        print(f"      Mean distortion: {flattened.mean_distortion:.1%}")
        print(f"      Max distortion: {flattened.max_distortion:.1%}")
        
        # 3. ?곷낯 ?대?吏 ?앹꽦
        print("\n[3/4] Generating rubbing image...")
        visualizer = SurfaceVisualizer(default_dpi=DEFAULT_EXPORT_DPI)
        rubbing = visualizer.generate_rubbing(flattened, width_pixels=DEFAULT_RENDER_RESOLUTION)
        
        print(f"      Image size: {rubbing.width_pixels} x {rubbing.height_pixels} pixels")
        print(f"      Real size: {rubbing.width_real:.1f} x {rubbing.height_real:.1f} {rubbing.unit}")
        
        # 4. ???
        print("\n[4/4] Saving output...")
        output_path = rubbing_output_path(filepath)
        rubbing.save(str(output_path), include_scale_bar=True)
        
        print(f"      Saved: {output_path}")
        
        print(f"\n{'='*60}")
        print("Done!")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


def flatten_mesh(filepath: str, output_path: str | None = None):
    """硫붿돩 ?됰㈃?붾쭔 ?섑뻾"""
    from src.core.mesh_loader import MeshLoader
    from src.core.flattener import ARAPFlattener
    from src.core.surface_visualizer import SurfaceVisualizer
    
    print(f"\nFlattening: {filepath}")
    print("-" * 40)
    
    try:
        loader = MeshLoader(default_unit=DEFAULT_MESH_UNIT)
        mesh = loader.load(filepath)
        
        print(f"  Loaded: {mesh.n_vertices:,} vertices, {mesh.n_faces:,} faces")

        flattener = ARAPFlattener(max_iterations=DEFAULT_ARAP_MAX_ITERATIONS)
        flattened = flattener.flatten(mesh)
        
        print(f"  Flattened: {flattened.width:.2f} x {flattened.height:.2f} {mesh.unit}")
        print(f"  Distortion: {flattened.mean_distortion:.1%} (mean), {flattened.max_distortion:.1%} (max)")
        
        # ?대?吏 ?앹꽦
        visualizer = SurfaceVisualizer()
        rubbing = visualizer.generate_rubbing(flattened, width_pixels=DEFAULT_RENDER_RESOLUTION)
        
        save_path = rubbing_output_path(filepath, output_path)
        rubbing.save(str(save_path), include_scale_bar=True)
        
        print(f"  Saved: {save_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def project_mesh(filepath: str, output_path: str | None = None):
    """?뺤궗?ъ쁺 ?대?吏 ?앹꽦"""
    from src.core.mesh_loader import MeshLoader
    from src.core.orthographic_projector import OrthographicProjector
    
    print(f"\nProjecting: {filepath}")
    print("-" * 40)
    
    try:
        loader = MeshLoader(default_unit=DEFAULT_MESH_UNIT)
        mesh = loader.load(filepath)
        
        print(f"  Loaded: {mesh.n_vertices:,} vertices, {mesh.n_faces:,} faces")
        
        projector = OrthographicProjector(resolution=DEFAULT_RENDER_RESOLUTION)
        
        # ?뺣젹
        aligned = projector.align_mesh(mesh, method='pca')
        print("  Aligned mesh using PCA")
        
        # ?ъ쁺
        result = projector.project(aligned, direction='top', render_mode='depth')
        
        print(f"  Projected: {result.image.shape[1]} x {result.image.shape[0]} pixels")
        print(f"  Real size: {result.width_real:.2f} x {result.height_real:.2f} {result.unit}")
        
        save_path = projection_output_path(filepath, output_path)
        result.save(str(save_path), dpi=DEFAULT_EXPORT_DPI)
        
        print(f"  Saved: {save_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def separate_mesh(filepath: str):
    """?대㈃/?몃㈃ 遺꾨━"""
    from src.core.mesh_loader import MeshLoader
    from src.core.surface_separator import SurfaceSeparator
    
    print(f"\nSeparating surfaces: {filepath}")
    print("-" * 40)
    
    try:
        loader = MeshLoader(default_unit=DEFAULT_MESH_UNIT)
        mesh = loader.load(filepath)
        
        print(f"  Loaded: {mesh.n_vertices:,} vertices, {mesh.n_faces:,} faces")
        
        separator = SurfaceSeparator()
        result = separator.auto_detect_surfaces(mesh)
        
        if result.inner_surface:
            print(f"  Inner surface: {result.inner_surface.n_faces:,} faces")
        if result.outer_surface:
            print(f"  Outer surface: {result.outer_surface.n_faces:,} faces")
        
        # ???(trimesh ?ъ슜)
        if result.inner_surface:
            inner_path = inner_surface_path(filepath)
            inner_mesh = result.inner_surface.to_trimesh()
            inner_mesh.export(str(inner_path))
            print(f"  Saved: {inner_path}")
        
        if result.outer_surface:
            outer_path = outer_surface_path(filepath)
            outer_mesh = result.outer_surface.to_trimesh()
            outer_mesh.export(str(outer_path))
            print(f"  Saved: {outer_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def launch_gui(*, open_project: str | None = None) -> None:
    """Launch the interactive GUI (app_interactive.py)."""
    try:
        import app_interactive
    except Exception as e:
        print("Failed to import GUI.")
        print("Make sure PyQt6 is installed: pip install -r requirements.txt")
        print(f"Error: {type(e).__name__}: {e}")
        return

    if open_project:
        # Let app_interactive access the argument via sys.argv.
        sys.argv = [sys.argv[0], "--open-project", str(open_project)]
    else:
        sys.argv = [sys.argv[0]]

    try:
        app_interactive.main()
    except Exception as e:
        print(f"GUI failed to start: {type(e).__name__}: {e}")


def run_gui():
    """GUI ?좏뵆由ъ??댁뀡 ?ㅽ뻾"""
    import importlib.util

    if importlib.util.find_spec("PyQt6") is None:
        print("PyQt6 is not installed. Install with: pip install PyQt6")
        return

    print("GUI is under development. Check back soon!")


if __name__ == '__main__':
    run_cli()
