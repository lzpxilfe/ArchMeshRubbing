"""
ArchMeshRubbing - Archaeological Recording-Surface Unwrap Tool
고고학 기록용 3D 메쉬 기록면 전개 도구

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

from src.core.runtime_defaults import DEFAULTS  # noqa: E402
from src.core.output_paths import (  # noqa: E402
    rubbing_output_path,
    projection_output_path,
    review_sheet_output_path,
    inner_surface_path,
    outer_surface_path,
)

_LOGGER = logging.getLogger(__name__)
DEFAULT_EXPORT_DPI = DEFAULTS.export_dpi
DEFAULT_RENDER_RESOLUTION = DEFAULTS.render_resolution
DEFAULT_ARAP_MAX_ITERATIONS = DEFAULTS.arap_max_iterations
DEFAULT_MESH_UNIT = "mm"
SUPPORTED_FORMATS = ["OBJ", "PLY", "STL", "OFF", "glTF (.gltf)", "glTF Binary (.glb)"]


def run_cli():
    """명령줄 인터페이스 실행."""
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

    if cmd == '--review' and len(sys.argv) > 2:
        review_mesh(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
        return

    if cmd == '--generate-synthetic' and len(sys.argv) > 2:
        try:
            seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        except Exception:
            seed = 1
        generate_synthetic_bundle(
            sys.argv[2],
            seed=seed,
            output_path=sys.argv[4] if len(sys.argv) > 4 else None,
        )
        return

    if cmd == '--benchmark-synthetic' and len(sys.argv) > 2:
        benchmark_synthetic_tiles(
            sys.argv[2],
            seeds_arg=(sys.argv[3] if len(sys.argv) > 3 else "1"),
        )
        return
    
    if cmd == '--project' and len(sys.argv) > 2:
        project_mesh(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
        return
    
    if cmd == '--separate' and len(sys.argv) > 2:
        separate_mesh(sys.argv[2])
        return
    
    # 기본: 파일 경로가 들어오면 전체 처리
    if os.path.exists(cmd):
        process_mesh(cmd)
    else:
        print(f"Error: Unknown command or file not found: {cmd}")
        print("Use --help for usage information")


def print_help():
    """도움말 출력."""
    print("=" * 60)
    print("ArchMeshRubbing - Archaeological Recording-Surface Unwrap Tool")
    print("고고학 기록용 3D 메쉬 기록면 전개 도구")
    print("=" * 60)
    print()
    print("Usage:")
    print("  python main.py <mesh_file>              # Full processing")
    print("  python main.py --info <mesh_file>       # Show file info")
    print("  python main.py --flatten <mesh_file> [output]    # Recording-surface unwrap only")
    print("  python main.py --review <mesh_file> [output]     # Recording-surface review sheet")
    print("  python main.py --generate-synthetic <preset> [seed] [output]  # Synthetic tile benchmark bundle + review")
    print("  python main.py --benchmark-synthetic <output_dir> [seeds]     # Synthetic benchmark suite + review sheets")
    print("  python main.py --project <mesh_file> [output]    # Orthographic projection")
    print("  python main.py --separate <mesh_file>   # Separate inner/outer surfaces")
    print("  python main.py --gui                    # Launch GUI (interactive)")
    print("  python main.py --open-project <project.amr>  # Launch GUI and open project")
    print()
    print(f"Supported formats: {SUPPORTED_FORMATS}")
    print()
    print("Examples:")
    print("  python main.py roof_tile.obj")
    print("  python main.py --flatten roof_tile.ply rubbing.tiff")
    print("  python main.py --review roof_tile.ply review.png")
    print("  python main.py --generate-synthetic sugkiwa_quarter 7 synthetic_tile.obj")
    print("  python main.py --benchmark-synthetic ./benchmarks 1,2,3")
    print("  python main.py --project roof_tile.stl planview.png")


def show_file_info(filepath: str):
    """파일 정보 표시."""
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
    """메쉬 전체 처리 (로드 -> 기록면 전개 -> 탁본 이미지 생성)."""
    from src.core.mesh_loader import MeshLoader
    from src.core.flattener import ARAPFlattener
    from src.core.surface_visualizer import SurfaceVisualizer
    
    print(f"\n{'='*60}")
    print(f"Processing: {filepath}")
    print(f"{'='*60}")
    
    try:
        # 1. Load
        print("\n[1/4] Loading mesh...")
        loader = MeshLoader(default_unit=DEFAULT_MESH_UNIT)
        mesh = loader.load(filepath)
        
        print(f"      Vertices: {mesh.n_vertices:,}")
        print(f"      Faces: {mesh.n_faces:,}")
        print(f"      Surface Area: {mesh.surface_area:,.2f} {mesh.unit}^2")
        print(f"      Size: {mesh.extents[0]:.1f} x {mesh.extents[1]:.1f} x {mesh.extents[2]:.1f} {mesh.unit}")
        print(f"      Has Texture: {mesh.has_texture}")
        
        # 2. Recording-surface unwrap
        print("\n[2/4] Unwrapping recording surface (ARAP algorithm)...")
        flattener = ARAPFlattener(max_iterations=DEFAULT_ARAP_MAX_ITERATIONS)
        flattened = flattener.flatten(mesh)
        
        print(f"      Unwrapped size: {flattened.width:.2f} x {flattened.height:.2f} {mesh.unit}")
        print(f"      Mean distortion: {flattened.mean_distortion:.1%}")
        print(f"      Max distortion: {flattened.max_distortion:.1%}")
        
        # 3. Generate rubbing image
        print("\n[3/4] Generating rubbing image...")
        visualizer = SurfaceVisualizer(default_dpi=DEFAULT_EXPORT_DPI)
        rubbing = visualizer.generate_rubbing(flattened, width_pixels=DEFAULT_RENDER_RESOLUTION)
        
        print(f"      Image size: {rubbing.width_pixels} x {rubbing.height_pixels} pixels")
        print(f"      Real size: {rubbing.width_real:.1f} x {rubbing.height_real:.1f} {rubbing.unit}")
        
        # 4. Save output
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
    """기록면 전개만 수행."""
    from src.core.mesh_loader import MeshLoader
    from src.core.flattener import ARAPFlattener
    from src.core.surface_visualizer import SurfaceVisualizer
    
    print(f"\nUnwrapping recording surface: {filepath}")
    print("-" * 40)
    
    try:
        loader = MeshLoader(default_unit=DEFAULT_MESH_UNIT)
        mesh = loader.load(filepath)
        
        print(f"  Loaded: {mesh.n_vertices:,} vertices, {mesh.n_faces:,} faces")

        flattener = ARAPFlattener(max_iterations=DEFAULT_ARAP_MAX_ITERATIONS)
        flattened = flattener.flatten(mesh)
        
        print(f"  Unwrapped: {flattened.width:.2f} x {flattened.height:.2f} {mesh.unit}")
        print(f"  Distortion: {flattened.mean_distortion:.1%} (mean), {flattened.max_distortion:.1%} (max)")
        
        # 이미지 생성
        visualizer = SurfaceVisualizer()
        rubbing = visualizer.generate_rubbing(flattened, width_pixels=DEFAULT_RENDER_RESOLUTION)
        
        save_path = rubbing_output_path(filepath, output_path)
        rubbing.save(str(save_path), include_scale_bar=True)
        
        print(f"  Saved: {save_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def review_mesh(filepath: str, output_path: str | None = None):
    """기록면 검토 시트 생성."""
    from src.core.flattener import ARAPFlattener
    from src.core.mesh_loader import MeshLoader
    from src.core.recording_surface_review import (
        build_recording_surface_summary_lines,
        RecordingSurfaceReviewOptions,
        render_recording_surface_review,
    )
    from src.core.surface_visualizer import SurfaceVisualizer

    print(f"\nBuilding recording-surface review sheet: {filepath}")
    print("-" * 40)

    try:
        loader = MeshLoader(default_unit=DEFAULT_MESH_UNIT)
        mesh = loader.load(filepath)

        print(f"  Loaded: {mesh.n_vertices:,} vertices, {mesh.n_faces:,} faces")

        flattener = ARAPFlattener(max_iterations=DEFAULT_ARAP_MAX_ITERATIONS)
        flattened = flattener.flatten(mesh)

        print(f"  Unwrapped: {flattened.width:.2f} x {flattened.height:.2f} {mesh.unit}")
        print(f"  Distortion: {flattened.mean_distortion:.1%} (mean), {flattened.max_distortion:.1%} (max)")

        visualizer = SurfaceVisualizer(default_dpi=DEFAULT_EXPORT_DPI)
        rubbing = visualizer.generate_rubbing(flattened, width_pixels=DEFAULT_RENDER_RESOLUTION)

        review = render_recording_surface_review(
            flattened,
            options=RecordingSurfaceReviewOptions(
                dpi=DEFAULT_EXPORT_DPI,
                width_pixels=DEFAULT_RENDER_RESOLUTION,
                summary_lines=build_recording_surface_summary_lines(
                    flattened,
                    record_label="전체 기록면",
                    target_label="전체 메쉬",
                    mode_label="일반 전개",
                ),
            ),
            rubbing_image=rubbing.to_pil_image(),
        )

        save_path = review_sheet_output_path(filepath, output_path)
        review.combined_image.save(str(save_path))

        print(f"  Saved: {save_path}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def generate_synthetic_bundle(preset: str, *, seed: int = 1, output_path: str | None = None):
    """합성 기와 벤치마크 묶음 생성."""
    from src.core.tile_synthetic import (
        evaluate_tile_interpretation,
        generate_synthetic_tile,
        save_synthetic_tile_bundle,
        synthetic_tile_spec_from_preset,
    )

    print(f"\nGenerating synthetic tile benchmark: {preset} (seed {int(seed)})")
    print("-" * 40)

    try:
        spec = synthetic_tile_spec_from_preset(preset, seed=int(seed))
        artifact = generate_synthetic_tile(spec)
        report = evaluate_tile_interpretation(artifact.truth.ground_truth_state, artifact.truth)

        if not output_path:
            output_path = str(Path.cwd() / f"{artifact.name}.obj")

        bundle_paths = save_synthetic_tile_bundle(
            artifact,
            output_path,
            interpretation_state=artifact.truth.ground_truth_state,
            evaluation_report=report,
        )

        print(f"  Mesh: {bundle_paths.get('mesh', '')}")
        print(f"  Truth: {bundle_paths.get('truth', '')}")
        print(f"  Interpretation: {bundle_paths.get('interpretation', '')}")
        print(f"  Evaluation: {bundle_paths.get('evaluation', '')}")
        print(f"  Review: {bundle_paths.get('review', '')}")
        print(f"  Bundle: {bundle_paths.get('bundle', '')}")
        print(f"  Baseline score: {report.overall_score * 100.0:.0f} / 100")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def benchmark_synthetic_tiles(output_dir: str, *, seeds_arg: str = "1"):
    """합성 기와 벤치마크 suite 생성."""
    from src.core.tile_synthetic import save_synthetic_benchmark_suite

    print(f"\nBuilding synthetic benchmark suite: {output_dir}")
    print("-" * 40)

    try:
        seeds: list[int] = []
        for token in str(seeds_arg or "1").split(","):
            token = str(token or "").strip()
            if not token:
                continue
            seeds.append(int(token))
        if not seeds:
            seeds = [1]

        report = save_synthetic_benchmark_suite(output_dir, seeds=tuple(seeds))
        print(f"  Cases: {report.case_count}")
        print(f"  Average score: {report.average_score * 100.0:.1f} / 100")
        print(f"  Pass threshold: {report.pass_threshold * 100.0:.0f} / 100")
        print(f"  Passed / Failed: {report.pass_count} / {report.fail_count}")
        print(f"  Summary JSON: {Path(output_dir) / 'synthetic_benchmark_summary.json'}")
        print(f"  Summary CSV: {Path(output_dir) / 'synthetic_benchmark_summary.csv'}")
        if report.cases:
            first_review = str(report.cases[0].review_path or "").strip()
            if first_review:
                print(f"  Review sheets: {Path(first_review).parent}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def project_mesh(filepath: str, output_path: str | None = None):
    """정사영 이미지 생성."""
    from src.core.mesh_loader import MeshLoader
    from src.core.orthographic_projector import OrthographicProjector
    
    print(f"\nProjecting: {filepath}")
    print("-" * 40)
    
    try:
        loader = MeshLoader(default_unit=DEFAULT_MESH_UNIT)
        mesh = loader.load(filepath)
        
        print(f"  Loaded: {mesh.n_vertices:,} vertices, {mesh.n_faces:,} faces")
        
        projector = OrthographicProjector(resolution=DEFAULT_RENDER_RESOLUTION)
        
        # 정렬
        aligned = projector.align_mesh(mesh, method='pca')
        print("  Aligned mesh using PCA")
        
        # 투영
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
    """내면/외면 분리."""
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
        
        # 저장 (trimesh 사용)
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
    """GUI 애플리케이션 실행."""
    import importlib.util

    if importlib.util.find_spec("PyQt6") is None:
        print("PyQt6 is not installed. Install with: pip install PyQt6")
        return

    print("GUI is under development. Check back soon!")


if __name__ == '__main__':
    run_cli()
