"""메우기 찾기: telling a computed lid from a measured surface.

The fixture is a drum whose two ends were left open by the scanner and then
closed by software, exactly the way a real file arrives: the rim of each
hole rebuilt as an even ring, one new point in its middle, one triangle to
every point of the rim.  The wall between them keeps the uneven triangles a
scanner actually produces, so that "the fan's faces are all the same size"
is a statement about the fill and not about the fixture being tidy
everywhere.

The drum is then squashed across one axis, which is what makes the fixture
resemble the file it was written for.  In 24ET0021 each fill's rim runs from
15.4 to 17.5 mm out from its centre - it is nowhere near a circle - and yet
every triangle of the fan has the same area to six decimal places.  A
squashed regular fan does exactly that, and it stops the test from passing
merely because the fixture's holes are round.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.core.artifact_mesh_fills import (
    DEFAULT_AREA_SPREAD_MILLIONTHS,
    ArtifactMeshFillsError,
    describe_filled_holes,
    face_areas_mm2,
    fill_face_indices,
    find_filled_holes,
)


def _ring(radius: float, height: float, columns: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, columns, endpoint=False)
    return np.column_stack(
        [radius * np.cos(angles), radius * np.sin(angles), np.full(columns, height)]
    )


def _scanned_drum(
    columns: int = 32, rows: int = 8, *, capped: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """A drum whose wall is unevenly triangulated, optionally with fans on it."""

    rng = np.random.default_rng(20260909)
    heights = np.linspace(0.0, 40.0, rows + 1)
    rings = [_ring(20.0, z, columns) for z in heights]
    for ring in rings[1:-1]:
        ring += rng.normal(0.0, 0.6, ring.shape)
    vertices = np.concatenate(rings)
    faces: list[list[int]] = []
    for row in range(rows):
        for i in range(columns):
            j = (i + 1) % columns
            a = row * columns + i
            b = row * columns + j
            c = (row + 1) * columns + j
            d = (row + 1) * columns + i
            faces.append([a, b, c])
            faces.append([a, c, d])
    if capped:
        bottom_rim = np.arange(columns)
        top_rim = np.arange(rows * columns, (rows + 1) * columns)
        bottom_apex = vertices.shape[0]
        top_apex = bottom_apex + 1
        vertices = np.concatenate(
            [vertices, np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 40.0]])]
        )
        for i in range(columns):
            j = (i + 1) % columns
            faces.append([bottom_apex, int(bottom_rim[j]), int(bottom_rim[i])])
            faces.append([top_apex, int(top_rim[i]), int(top_rim[j])])
    squashed = vertices.copy()
    squashed[:, 1] *= 0.88
    return squashed, np.asarray(faces, dtype=np.int64)


def _fan_of_equal_faces(vertices: np.ndarray, faces: np.ndarray, apex: int) -> bool:
    fan = np.flatnonzero((faces == apex).any(axis=1))
    areas = face_areas_mm2(vertices, faces[fan])
    return bool(areas.std() / areas.mean() < 1e-9)


class TestMeshFills(unittest.TestCase):
    def test_the_two_lids_are_found_and_the_scanned_wall_is_not(self):
        vertices, faces = _scanned_drum()
        report = find_filled_holes(vertices, faces)

        self.assertEqual(len(report.fills), 2)
        self.assertEqual(report.filled_face_count, 64)
        self.assertTrue(report.needs_decision)
        for fill in report.fills:
            self.assertEqual(fill.face_count, 32)
            self.assertEqual(fill.rim_point_count, 32)
            # The rim is nowhere near round - it runs from 17.6 to 20.0 mm
            # out from its centre - and yet every triangle of the fan has
            # the same area, because they were laid out rather than measured.
            self.assertLess(fill.area_spread_millionths, 10)
            self.assertGreater(fill.coarseness_millionths, 2_000_000)

    def test_a_scan_with_its_holes_still_open_reports_nothing(self):
        vertices, faces = _scanned_drum(capped=False)
        report = find_filled_holes(vertices, faces)

        self.assertEqual(report.fills, ())
        self.assertFalse(report.needs_decision)
        self.assertEqual(describe_filled_holes(report), ("메우기로 보이는 면은 없습니다.",))

    def test_the_fill_is_the_faces_a_repair_would_drop(self):
        vertices, faces = _scanned_drum()
        report = find_filled_holes(vertices, faces)

        dropped = np.concatenate(
            [fill_face_indices(faces, fill) for fill in report.fills]
        )
        self.assertEqual(dropped.size, 64)
        remaining = np.delete(np.arange(faces.shape[0]), dropped)
        # What is left is the wall the scanner measured, and it is open at
        # both ends again - which is the state the file arrived in.
        self.assertEqual(remaining.size, faces.shape[0] - 64)
        kept = faces[remaining]
        for fill in report.fills:
            self.assertFalse((kept == fill.apex_vertex).any())

    def test_a_fill_from_another_mesh_is_refused_rather_than_guessed_at(self):
        vertices, faces = _scanned_drum()
        report = find_filled_holes(vertices, faces)
        fill = report.fills[0]

        trimmed = np.delete(faces, fill_face_indices(faces, fill)[:4], axis=0)
        with self.assertRaises(ArtifactMeshFillsError) as caught:
            fill_face_indices(trimmed, fill)
        self.assertIn("not the mesh the fill was found in", str(caught.exception))

    def test_an_even_fan_that_is_no_coarser_than_the_mesh_is_left_alone(self):
        """Evenness alone is not enough to call a surface fabricated.

        The same drum with a coarse wall: the caps are still perfect fans,
        but they no longer span in one triangle what the scan spends many on,
        and a patch like that is more likely a flat facet of the artifact
        than an invention.  It is left for the archaeologist to notice.
        """

        vertices, faces = _scanned_drum(rows=2)
        report = find_filled_holes(vertices, faces)

        self.assertEqual(report.fills, ())
        for apex in (vertices.shape[0] - 2, vertices.shape[0] - 1):
            self.assertTrue(_fan_of_equal_faces(vertices, faces, apex))

    def test_a_fan_of_uneven_faces_is_not_a_fill(self):
        vertices, faces = _scanned_drum()
        moved = vertices.copy()
        # Push one lid's centre off the axis: the fan is still a fan and
        # still coarse, but its triangles no longer agree in area.
        moved[-1] += np.array([6.0, 0.0, 0.0])
        report = find_filled_holes(moved, faces)

        self.assertEqual(len(report.fills), 1)
        self.assertEqual(report.fills[0].apex_um[2], 0)

    def test_the_read_out_names_the_evidence_it_found_them_by(self):
        vertices, faces = _scanned_drum()
        lines = describe_filled_holes(find_filled_holes(vertices, faces))
        joined = "\n".join(lines)

        self.assertIn("2곳", joined)
        self.assertIn("면 64개", joined)
        self.assertIn("넓이가 모두 같습니다", joined)
        self.assertIn("점선", joined)

    def test_the_thresholds_are_checked_before_any_work(self):
        vertices, faces = _scanned_drum()
        for kwargs, expected in (
            ({"minimum_fan_faces": 2}, "minimum_fan_faces"),
            ({"area_spread_millionths": 0}, "area_spread_millionths"),
            ({"coarseness_millionths": 10}, "coarseness_millionths"),
            ({"minimum_fan_faces": True}, "minimum_fan_faces"),
        ):
            with self.assertRaises(ArtifactMeshFillsError) as caught:
                find_filled_holes(vertices, faces, **kwargs)
            self.assertIn(expected, str(caught.exception))

    def test_a_mesh_it_cannot_read_is_refused(self):
        vertices, faces = _scanned_drum()
        with self.assertRaises(ArtifactMeshFillsError):
            find_filled_holes(vertices[:, :2], faces)
        with self.assertRaises(ArtifactMeshFillsError):
            find_filled_holes(vertices, faces[:0])
        with self.assertRaises(ArtifactMeshFillsError):
            find_filled_holes(vertices, faces + vertices.shape[0])

    def test_the_report_carries_the_numbers_the_finding_rests_on(self):
        vertices, faces = _scanned_drum()
        qc = find_filled_holes(vertices, faces).qc_dict()

        self.assertEqual(qc["fill_count"], 2)
        self.assertEqual(qc["filled_face_count"], 64)
        self.assertEqual(qc["area_spread_millionths"], DEFAULT_AREA_SPREAD_MILLIONTHS)
        self.assertEqual(qc["searched_face_count"], faces.shape[0])
        self.assertGreater(qc["mesh_median_area_um2"], 0)
        self.assertEqual(len(qc["fills"]), 2)
        self.assertEqual(qc["fills"][0]["rim_point_count"], 32)


if __name__ == "__main__":
    unittest.main()
