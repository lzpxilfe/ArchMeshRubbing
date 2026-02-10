from __future__ import annotations

import argparse
import ctypes
import gc
import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.mesh_loader import MeshLoader
from src.core.surface_separator import SurfaceSeparator

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore


def _rss_mb() -> float:
    if psutil is None:
        # Windows fallback without psutil.
        if sys.platform.startswith("win"):
            try:
                class _PMCEX(ctypes.Structure):
                    _fields_ = [
                        ("cb", ctypes.c_ulong),
                        ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t),
                        ("PrivateUsage", ctypes.c_size_t),
                    ]

                pmc = _PMCEX()
                pmc.cb = ctypes.sizeof(_PMCEX)
                proc = ctypes.windll.kernel32.GetCurrentProcess()
                ok = ctypes.windll.psapi.GetProcessMemoryInfo(proc, ctypes.byref(pmc), pmc.cb)
                if ok:
                    return float(pmc.WorkingSetSize) / (1024.0 * 1024.0)
            except Exception:
                return float("nan")
        return float("nan")
    try:
        proc = psutil.Process()
        return float(proc.memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        return float("nan")


def _to_int_set(values: Any) -> set[int]:
    arr = np.asarray(values if values is not None else np.zeros((0,), dtype=np.int32), dtype=np.int64).reshape(-1)
    if arr.size <= 0:
        return set()
    out: set[int] = set()
    for x in arr.tolist():
        try:
            out.add(int(x))
        except Exception:
            continue
    return out


def _sample_seed(arr: np.ndarray, *, stride: int, min_seed: int, max_seed: int = 20000) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.int32).reshape(-1)
    if arr.size <= 0:
        return np.zeros((0,), dtype=np.int32)
    arr = np.unique(arr)
    stride_i = max(1, int(stride))
    seed = arr[::stride_i]
    if seed.size < int(min_seed):
        step = max(1, int(arr.size // max(1, int(min_seed))))
        seed = arr[::step]
    if seed.size > int(max_seed):
        step = max(1, int(seed.size // int(max_seed)))
        seed = seed[::step]
    return np.unique(seed.astype(np.int32, copy=False))


def _apply_large_mesh_stability_presets(mesh) -> None:
    try:
        n_faces = int(getattr(mesh, "n_faces", 0) or 0)
    except Exception:
        n_faces = 0
    if n_faces < 1_000_000:
        return
    if not hasattr(mesh, "_views_fallback_use_normals"):
        mesh._views_fallback_use_normals = False
    if not hasattr(mesh, "_views_migu_absdot_max"):
        mesh._views_migu_absdot_max = 1.0
    if not hasattr(mesh, "_views_migu_max_frac"):
        mesh._views_migu_max_frac = 0.05
    if not hasattr(mesh, "_views_visibility_neighborhood"):
        mesh._views_visibility_neighborhood = 2


def _build_overlay_indices(
    cache: dict[str, np.ndarray],
    key: str,
    face_set: set[int],
    max_faces: int,
) -> np.ndarray | None:
    n = int(len(face_set))
    if n <= 0 or n > int(max_faces):
        return None
    arr = cache.get(key)
    if isinstance(arr, np.ndarray) and arr.dtype == np.uint32 and arr.ndim == 1 and arr.size == n * 3:
        return arr
    face_idx = np.fromiter(face_set, dtype=np.int32, count=n)
    base = face_idx.astype(np.uint32, copy=False) * np.uint32(3)
    out = np.empty((n * 3,), dtype=np.uint32)
    out[0::3] = base
    out[1::3] = base + np.uint32(1)
    out[2::3] = base + np.uint32(2)
    cache[key] = out
    return out


def _fmt_sec(x: float) -> str:
    return f"{x:.4f}s"


def _fmt_mb(x: float) -> str:
    if not np.isfinite(x):
        return "n/a"
    return f"{x:.1f}MB"


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    mesh_path = Path(args.mesh).expanduser().resolve()
    if not mesh_path.exists():
        raise FileNotFoundError(str(mesh_path))

    print(f"[load] mesh={mesh_path}")
    loader = MeshLoader(default_unit=str(args.unit))
    t0 = time.perf_counter()
    mesh = loader.load(str(mesh_path))
    load_sec = float(time.perf_counter() - t0)
    print(f"[load] faces={int(getattr(mesh, 'n_faces', 0) or 0):,}, verts={int(getattr(mesh, 'n_vertices', 0) or 0):,}, time={_fmt_sec(load_sec)}")

    if args.unresolved_keep_max >= 0:
        mesh._assist_unresolved_keep_max = int(args.unresolved_keep_max)
        print(f"[config] _assist_unresolved_keep_max={int(args.unresolved_keep_max):,}")
    _apply_large_mesh_stability_presets(mesh)

    sep = SurfaceSeparator()
    t1 = time.perf_counter()
    auto = sep.auto_detect_surfaces(mesh, method=str(args.bootstrap_method), return_submeshes=False)
    boot_sec = float(time.perf_counter() - t1)
    outer_all = _to_int_set(getattr(auto, "outer_face_indices", None))
    inner_all = _to_int_set(getattr(auto, "inner_face_indices", None))
    migu_all = _to_int_set(getattr(auto, "migu_face_indices", None))
    outer_all.difference_update(migu_all)
    inner_all.difference_update(migu_all)
    overlap = outer_all.intersection(inner_all)
    if overlap:
        inner_all.difference_update(overlap)
        migu_all.update(overlap)
    print(
        "[bootstrap] "
        f"method={args.bootstrap_method}, time={_fmt_sec(boot_sec)}, "
        f"outer={len(outer_all):,}, inner={len(inner_all):,}, migu={len(migu_all):,}"
    )

    outer_seed = _sample_seed(
        np.asarray(sorted(outer_all), dtype=np.int32),
        stride=int(args.seed_stride),
        min_seed=int(args.min_seed),
        max_seed=int(args.max_seed),
    )
    inner_seed = _sample_seed(
        np.asarray(sorted(inner_all), dtype=np.int32),
        stride=int(args.seed_stride),
        min_seed=int(args.min_seed),
        max_seed=int(args.max_seed),
    )
    migu_idx = np.asarray(sorted(migu_all), dtype=np.int32)

    print(
        "[seed] "
        f"outer_seed={int(outer_seed.size):,}, inner_seed={int(inner_seed.size):,}, "
        f"min_seed={int(args.min_seed):,}, stride={int(args.seed_stride)}"
    )

    if outer_seed.size < int(args.min_seed) or inner_seed.size < int(args.min_seed):
        raise RuntimeError("Seed count is too small for assist profiling. Lower --min-seed or --seed-stride.")

    tracemalloc.start()
    rows: list[dict[str, Any]] = []

    for i in range(int(args.iterations)):
        gc.collect()
        rss_before = _rss_mb()

        t_assist_0 = time.perf_counter()
        out_outer, out_inner, meta = sep.assist_outer_inner_from_seeds(
            mesh,
            outer_face_indices=outer_seed,
            inner_face_indices=inner_seed,
            migu_face_indices=migu_idx,
            method=str(args.assist_method),
            conservative=not bool(args.aggressive),
            min_seed=int(args.min_seed),
        )
        assist_sec = float(time.perf_counter() - t_assist_0)

        t_sets_0 = time.perf_counter()
        outer_set = _to_int_set(out_outer)
        inner_set = _to_int_set(out_inner)
        migu_set = _to_int_set(migu_idx)
        outer_set.difference_update(migu_set)
        inner_set.difference_update(migu_set)
        overlap2 = outer_set.intersection(inner_set)
        if overlap2:
            inner_set.difference_update(overlap2)

        unresolved_count = int((meta or {}).get("unresolved_count", 0) or 0)
        unresolved_truncated = bool((meta or {}).get("unresolved_truncated", False))
        unresolved_raw = (meta or {}).get("unresolved_indices", None)
        if unresolved_raw is None:
            unresolved_set: set[int] = set()
        else:
            unresolved_set = _to_int_set(np.asarray(unresolved_raw, dtype=np.int32).reshape(-1))
        unresolved_set.difference_update(outer_set)
        unresolved_set.difference_update(inner_set)
        unresolved_set.difference_update(migu_set)
        set_sec = float(time.perf_counter() - t_sets_0)

        t_overlay_0 = time.perf_counter()
        cache: dict[str, np.ndarray] = {}
        idx_outer = _build_overlay_indices(cache, "outer", outer_set, int(args.overlay_max_faces))
        idx_inner = _build_overlay_indices(cache, "inner", inner_set, int(args.overlay_max_faces))
        idx_migu = _build_overlay_indices(cache, "migu", migu_set, int(args.overlay_max_faces))
        idx_unresolved = _build_overlay_indices(cache, "unresolved", unresolved_set, int(args.overlay_max_faces))
        overlay_cold_sec = float(time.perf_counter() - t_overlay_0)

        t_overlay_hot_0 = time.perf_counter()
        _ = _build_overlay_indices(cache, "outer", outer_set, int(args.overlay_max_faces))
        _ = _build_overlay_indices(cache, "inner", inner_set, int(args.overlay_max_faces))
        _ = _build_overlay_indices(cache, "migu", migu_set, int(args.overlay_max_faces))
        _ = _build_overlay_indices(cache, "unresolved", unresolved_set, int(args.overlay_max_faces))
        overlay_hot_sec = float(time.perf_counter() - t_overlay_hot_0)

        overlay_bytes = int(
            sum(
                int(x.nbytes)
                for x in (idx_outer, idx_inner, idx_migu, idx_unresolved)
                if isinstance(x, np.ndarray)
            )
        )

        rss_after = _rss_mb()
        tm_cur, tm_peak = tracemalloc.get_traced_memory()
        tm_peak_mb = float(tm_peak) / (1024.0 * 1024.0)
        row = {
            "iter": i + 1,
            "assist_sec": assist_sec,
            "set_sec": set_sec,
            "overlay_cold_sec": overlay_cold_sec,
            "overlay_hot_sec": overlay_hot_sec,
            "outer_count": int(len(outer_set)),
            "inner_count": int(len(inner_set)),
            "migu_count": int(len(migu_set)),
            "unresolved_count": int(unresolved_count),
            "unresolved_drawn_count": int(len(unresolved_set)),
            "unresolved_truncated": bool(unresolved_truncated),
            "overlay_index_mb": float(overlay_bytes) / (1024.0 * 1024.0),
            "rss_before_mb": rss_before,
            "rss_after_mb": rss_after,
            "tracemalloc_current_mb": float(tm_cur) / (1024.0 * 1024.0),
            "tracemalloc_peak_mb": tm_peak_mb,
            "status": str((meta or {}).get("status", "")),
            "assist_mode": str((meta or {}).get("assist_mode", "")),
            "auto_mapping": str((meta or {}).get("auto_mapping", "")),
        }
        rows.append(row)
        print(
            f"[iter {i+1}] assist={_fmt_sec(assist_sec)}, sets={_fmt_sec(set_sec)}, "
            f"overlay(cold/hot)={_fmt_sec(overlay_cold_sec)}/{_fmt_sec(overlay_hot_sec)}, "
            f"unresolved={row['unresolved_count']:,} (draw={row['unresolved_drawn_count']:,}), "
            f"overlay_idx={_fmt_mb(row['overlay_index_mb'])}, "
            f"rss={_fmt_mb(rss_before)}->{_fmt_mb(rss_after)}, "
            f"trace_peak={_fmt_mb(tm_peak_mb)}"
        )

    tm_cur, tm_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    def _agg(key: str) -> dict[str, float]:
        vals = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size <= 0:
            nan = float("nan")
            return {"mean": nan, "p95": nan, "max": nan}
        return {
            "mean": float(np.mean(vals)),
            "p95": float(np.percentile(vals, 95.0)),
            "max": float(np.max(vals)),
        }

    summary = {
        "mesh": str(mesh_path),
        "load_sec": load_sec,
        "bootstrap_sec": boot_sec,
        "faces": int(getattr(mesh, "n_faces", 0) or 0),
        "vertices": int(getattr(mesh, "n_vertices", 0) or 0),
        "iterations": int(args.iterations),
        "assist_method": str(args.assist_method),
        "aggressive": bool(args.aggressive),
        "seed_stride": int(args.seed_stride),
        "min_seed": int(args.min_seed),
        "max_seed": int(args.max_seed),
        "overlay_max_faces": int(args.overlay_max_faces),
        "metrics": {
            "assist_sec": _agg("assist_sec"),
            "set_sec": _agg("set_sec"),
            "overlay_cold_sec": _agg("overlay_cold_sec"),
            "overlay_hot_sec": _agg("overlay_hot_sec"),
            "overlay_index_mb": _agg("overlay_index_mb"),
            "rss_after_mb": _agg("rss_after_mb"),
            "tracemalloc_peak_mb": _agg("tracemalloc_peak_mb"),
        },
        "rows": rows,
        "tracemalloc_final_current_mb": float(tm_cur) / (1024.0 * 1024.0),
        "tracemalloc_final_peak_mb": float(tm_peak) / (1024.0 * 1024.0),
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile manual seeded assist-surface pipeline (assist + unresolved overlay build)."
    )
    default_mesh = Path("수키와편.stl")
    parser.add_argument(
        "mesh",
        nargs="?",
        default=str(default_mesh) if default_mesh.exists() else None,
        help="Path to mesh file (stl/obj/ply/off/glb/gltf).",
    )
    parser.add_argument("--unit", default="cm", help="Default unit for mesh loader.")
    parser.add_argument("--iterations", type=int, default=3, help="Assist repeat count.")
    parser.add_argument("--bootstrap-method", default="auto", help="Method for initial outer/inner seed bootstrap.")
    parser.add_argument("--assist-method", default="auto", help="Method used in assist (auto/views/cylinder).")
    parser.add_argument("--aggressive", action="store_true", help="Use aggressive assist mode (conservative=False).")
    parser.add_argument("--seed-stride", type=int, default=120, help="Stride for sparse seed sampling.")
    parser.add_argument("--min-seed", type=int, default=48, help="Minimum seed count for each side.")
    parser.add_argument("--max-seed", type=int, default=20000, help="Maximum seeds per side after sampling.")
    parser.add_argument(
        "--overlay-max-faces",
        type=int,
        default=3_000_000,
        help="Max faces for overlay index build per label set.",
    )
    parser.add_argument(
        "--unresolved-keep-max",
        type=int,
        default=-1,
        help="Override mesh._assist_unresolved_keep_max (negative keeps mesh default).",
    )
    parser.add_argument("--json-out", default="", help="Optional JSON output path.")
    args = parser.parse_args()
    if not args.mesh:
        parser.error("mesh path is required (or place '수키와편.stl' in current directory).")
    return args


def main() -> int:
    args = parse_args()
    result = run_profile(args)
    print(
        "[summary] "
        f"assist mean={_fmt_sec(result['metrics']['assist_sec']['mean'])}, "
        f"overlay cold mean={_fmt_sec(result['metrics']['overlay_cold_sec']['mean'])}, "
        f"overlay hot mean={_fmt_sec(result['metrics']['overlay_hot_sec']['mean'])}, "
        f"rss_after max={_fmt_mb(result['metrics']['rss_after_mb']['max'])}, "
        f"trace_peak max={_fmt_mb(result['metrics']['tracemalloc_peak_mb']['max'])}"
    )
    out = str(args.json_out or "").strip()
    if out:
        out_path = Path(out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[saved] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
