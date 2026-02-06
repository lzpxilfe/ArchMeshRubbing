"""
ArchMeshRubbing - 3D Mesh Flattening Tool for Archaeological Documentation
고고학 유물 3D 메쉬 평면화 도구

Main entry point
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

_LOGGER = logging.getLogger(__name__)


def run_cli():
    """커맨드라인 인터페이스 실행"""
    try:
        from src.core.logging_utils import setup_logging

        setup_logging()
    except Exception:
        pass

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
        print("GUI mode coming soon...")
        print("For now, use command line: python main.py <mesh_file>")
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
    
    # 기본: 파일 처리
    if os.path.exists(cmd):
        process_mesh(cmd)
    else:
        print(f"Error: Unknown command or file not found: {cmd}")
        print("Use --help for usage information")


def print_help():
    """도움말 출력"""
    from src.core.mesh_loader import MeshLoader
    
    print("=" * 60)
    print("ArchMeshRubbing - 3D Mesh Flattening Tool")
    print("고고학 유물 3D 메쉬 평면화 도구")
    print("=" * 60)
    print()
    print("Usage:")
    print("  python main.py <mesh_file>              # Full processing")
    print("  python main.py --info <mesh_file>       # Show file info")
    print("  python main.py --flatten <mesh_file> [output]    # Flatten only")
    print("  python main.py --project <mesh_file> [output]    # Orthographic projection")
    print("  python main.py --separate <mesh_file>   # Separate inner/outer surfaces")
    print("  python main.py --gui                    # Launch GUI (coming soon)")
    print()
    print(f"Supported formats: {list(MeshLoader.SUPPORTED_FORMATS.keys())}")
    print()
    print("Examples:")
    print("  python main.py roof_tile.obj")
    print("  python main.py --flatten roof_tile.ply rubbing.tiff")
    print("  python main.py --project roof_tile.stl planview.png")


def show_file_info(filepath: str):
    """파일 정보 표시"""
    from src.core.mesh_loader import MeshLoader
    
    print(f"\nFile Info: {filepath}")
    print("-" * 40)
    
    try:
        loader = MeshLoader()
        info = loader.get_file_info(filepath)
        for key, value in info.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"  Error: {e}")


def process_mesh(filepath: str):
    """메쉬 전체 처리 (로드 → 평면화 → 이미지 생성)"""
    from src.core.mesh_loader import MeshLoader
    from src.core.flattener import ARAPFlattener
    from src.core.surface_visualizer import SurfaceVisualizer
    
    print(f"\n{'='*60}")
    print(f"Processing: {filepath}")
    print(f"{'='*60}")
    
    try:
        # 1. 로드
        print("\n[1/4] Loading mesh...")
        loader = MeshLoader()
        mesh = loader.load(filepath)
        
        print(f"      Vertices: {mesh.n_vertices:,}")
        print(f"      Faces: {mesh.n_faces:,}")
        print(f"      Surface Area: {mesh.surface_area:,.2f} {mesh.unit}²")
        print(f"      Size: {mesh.extents[0]:.1f} x {mesh.extents[1]:.1f} x {mesh.extents[2]:.1f} {mesh.unit}")
        print(f"      Has Texture: {mesh.has_texture}")
        
        # 2. 평면화
        print("\n[2/4] Flattening mesh (ARAP algorithm)...")
        flattener = ARAPFlattener(max_iterations=30)
        flattened = flattener.flatten(mesh)
        
        print(f"      Flattened size: {flattened.width:.2f} x {flattened.height:.2f} {mesh.unit}")
        print(f"      Mean distortion: {flattened.mean_distortion:.1%}")
        print(f"      Max distortion: {flattened.max_distortion:.1%}")
        
        # 3. 탁본 이미지 생성
        print("\n[3/4] Generating rubbing image...")
        visualizer = SurfaceVisualizer(default_dpi=300)
        rubbing = visualizer.generate_rubbing(flattened, width_pixels=2000)
        
        print(f"      Image size: {rubbing.width_pixels} x {rubbing.height_pixels} pixels")
        print(f"      Real size: {rubbing.width_real:.1f} x {rubbing.height_real:.1f} {rubbing.unit}")
        
        # 4. 저장
        print("\n[4/4] Saving output...")
        output_path = Path(filepath).with_suffix('.rubbing.png')
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
    """메쉬 평면화만 수행"""
    from src.core.mesh_loader import MeshLoader
    from src.core.flattener import ARAPFlattener
    from src.core.surface_visualizer import SurfaceVisualizer
    
    print(f"\nFlattening: {filepath}")
    print("-" * 40)
    
    try:
        loader = MeshLoader()
        mesh = loader.load(filepath)
        
        print(f"  Loaded: {mesh.n_vertices:,} vertices, {mesh.n_faces:,} faces")
        
        flattener = ARAPFlattener(max_iterations=30)
        flattened = flattener.flatten(mesh)
        
        print(f"  Flattened: {flattened.width:.2f} x {flattened.height:.2f} {mesh.unit}")
        print(f"  Distortion: {flattened.mean_distortion:.1%} (mean), {flattened.max_distortion:.1%} (max)")
        
        # 이미지 생성
        visualizer = SurfaceVisualizer()
        rubbing = visualizer.generate_rubbing(flattened, width_pixels=2000)
        
        output_path = output_path or str(Path(filepath).with_suffix('.rubbing.png'))
        rubbing.save(output_path, include_scale_bar=True)
        
        print(f"  Saved: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def project_mesh(filepath: str, output_path: str | None = None):
    """정사투영 이미지 생성"""
    from src.core.mesh_loader import MeshLoader
    from src.core.orthographic_projector import OrthographicProjector
    
    print(f"\nProjecting: {filepath}")
    print("-" * 40)
    
    try:
        loader = MeshLoader()
        mesh = loader.load(filepath)
        
        print(f"  Loaded: {mesh.n_vertices:,} vertices, {mesh.n_faces:,} faces")
        
        projector = OrthographicProjector(resolution=2000)
        
        # 정렬
        aligned = projector.align_mesh(mesh, method='pca')
        print("  Aligned mesh using PCA")
        
        # 투영
        result = projector.project(aligned, direction='top', render_mode='depth')
        
        print(f"  Projected: {result.image.shape[1]} x {result.image.shape[0]} pixels")
        print(f"  Real size: {result.width_real:.2f} x {result.height_real:.2f} {result.unit}")
        
        output_path = output_path or str(Path(filepath).with_suffix('.projection.png'))
        result.save(output_path, dpi=300)
        
        print(f"  Saved: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def separate_mesh(filepath: str):
    """내면/외면 분리"""
    from src.core.mesh_loader import MeshLoader
    from src.core.surface_separator import SurfaceSeparator
    
    print(f"\nSeparating surfaces: {filepath}")
    print("-" * 40)
    
    try:
        loader = MeshLoader()
        mesh = loader.load(filepath)
        
        print(f"  Loaded: {mesh.n_vertices:,} vertices, {mesh.n_faces:,} faces")
        
        separator = SurfaceSeparator()
        result = separator.auto_detect_surfaces(mesh)
        
        if result.inner_surface:
            print(f"  Inner surface: {result.inner_surface.n_faces:,} faces")
        if result.outer_surface:
            print(f"  Outer surface: {result.outer_surface.n_faces:,} faces")
        
        # 저장 (trimesh 사용)
        base_path = Path(filepath)
        
        if result.inner_surface:
            inner_path = base_path.with_suffix('.inner.ply')
            inner_mesh = result.inner_surface.to_trimesh()
            inner_mesh.export(str(inner_path))
            print(f"  Saved: {inner_path}")
        
        if result.outer_surface:
            outer_path = base_path.with_suffix('.outer.ply')
            outer_mesh = result.outer_surface.to_trimesh()
            outer_mesh.export(str(outer_path))
            print(f"  Saved: {outer_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def run_gui():
    """GUI 애플리케이션 실행"""
    import importlib.util

    if importlib.util.find_spec("PyQt6") is None:
        print("PyQt6 is not installed. Install with: pip install PyQt6")
        return

    print("GUI is under development. Check back soon!")


if __name__ == '__main__':
    run_cli()
