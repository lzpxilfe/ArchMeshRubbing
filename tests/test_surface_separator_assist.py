import unittest

import numpy as np

from src.core.surface_separator import SeparatedSurfaces, SurfaceSeparator
from tests.test_surface_separator_cylinder import _make_thin_cylinder_patch


class TestSurfaceSeparatorAssist(unittest.TestCase):
    def test_assist_from_manual_seeds_fills_unknown_faces(self):
        mesh, true_outer, true_inner, true_migu = _make_thin_cylinder_patch()

        # Sparse user seeds on both sides.
        outer_seed = np.asarray(sorted(true_outer)[::40], dtype=np.int32)
        inner_seed = np.asarray(sorted(true_inner)[::40], dtype=np.int32)
        migu_idx = np.asarray(sorted(true_migu), dtype=np.int32)

        sep = SurfaceSeparator()
        outer_idx, inner_idx, meta = sep.assist_outer_inner_from_seeds(
            mesh,
            outer_face_indices=outer_seed,
            inner_face_indices=inner_seed,
            migu_face_indices=migu_idx,
            method="auto",
            conservative=True,
            min_seed=5,
        )

        self.assertEqual(str((meta or {}).get("status", "")), "ok")
        pred_outer = set(map(int, np.asarray(outer_idx, dtype=np.int32).reshape(-1)))
        pred_inner = set(map(int, np.asarray(inner_idx, dtype=np.int32).reshape(-1)))
        true_migu_set = set(true_migu)

        self.assertTrue(set(map(int, outer_seed.tolist())).issubset(pred_outer))
        self.assertTrue(set(map(int, inner_seed.tolist())).issubset(pred_inner))
        self.assertTrue(pred_outer.isdisjoint(pred_inner))
        self.assertTrue(pred_outer.isdisjoint(true_migu_set))
        self.assertTrue(pred_inner.isdisjoint(true_migu_set))

        outer_recall = len(pred_outer & true_outer) / max(1, len(true_outer))
        inner_recall = len(pred_inner & true_inner) / max(1, len(true_inner))
        self.assertGreaterEqual(outer_recall, 0.95)
        self.assertGreaterEqual(inner_recall, 0.95)

    def test_assist_requires_both_side_seeds(self):
        mesh, true_outer, _true_inner, true_migu = _make_thin_cylinder_patch()
        outer_seed = np.asarray(sorted(true_outer)[::30], dtype=np.int32)
        migu_idx = np.asarray(sorted(true_migu), dtype=np.int32)

        sep = SurfaceSeparator()
        outer_idx, inner_idx, meta = sep.assist_outer_inner_from_seeds(
            mesh,
            outer_face_indices=outer_seed,
            inner_face_indices=np.zeros((0,), dtype=np.int32),
            migu_face_indices=migu_idx,
            method="auto",
            conservative=True,
            min_seed=5,
        )

        self.assertEqual(str((meta or {}).get("status", "")), "missing_seeds")
        self.assertGreaterEqual(int((meta or {}).get("seed_outer_count", 0)), 1)
        self.assertEqual(int((meta or {}).get("seed_inner_count", -1)), 0)
        self.assertGreaterEqual(int(np.asarray(outer_idx).size), 1)
        self.assertEqual(int(np.asarray(inner_idx).size), 0)

    def test_assist_unresolved_indices_truncated_when_limit_is_low(self):
        class SparseAutoSeparator(SurfaceSeparator):
            def auto_detect_surfaces(self, mesh, method="auto", return_submeshes=False):
                # Intentionally classify only a few faces so unresolved remains large.
                outer = np.asarray([0, 2, 4], dtype=np.int32)
                inner = np.asarray([1, 3, 5], dtype=np.int32)
                return SeparatedSurfaces(
                    inner_surface=None,
                    outer_surface=None,
                    inner_face_indices=inner,
                    outer_face_indices=outer,
                    migu_face_indices=None,
                    meta={"method": "sparse_fake"},
                )

        mesh, true_outer, true_inner, true_migu = _make_thin_cylinder_patch()
        mesh._assist_unresolved_keep_max = 8
        outer_seed = np.asarray(sorted(true_outer)[:8], dtype=np.int32)
        inner_seed = np.asarray(sorted(true_inner)[:8], dtype=np.int32)
        migu_idx = np.asarray(sorted(true_migu), dtype=np.int32)

        sep = SparseAutoSeparator()
        _outer_idx, _inner_idx, meta = sep.assist_outer_inner_from_seeds(
            mesh,
            outer_face_indices=outer_seed,
            inner_face_indices=inner_seed,
            migu_face_indices=migu_idx,
            method="auto",
            conservative=False,
            min_seed=5,
        )

        self.assertEqual(str((meta or {}).get("status", "")), "ok")
        self.assertGreater(int((meta or {}).get("unresolved_count", 0)), 8)
        self.assertTrue(bool((meta or {}).get("unresolved_truncated", False)))
        self.assertIsNone((meta or {}).get("unresolved_indices", "not-none"))


if __name__ == "__main__":
    unittest.main()
