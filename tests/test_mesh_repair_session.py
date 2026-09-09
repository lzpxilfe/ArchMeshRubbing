"""보정의 기록: a repaired scan that can be opened again and still be the same.

A repaired mesh is never stored.  What the document keeps is the decision -
these edges, onto that body - beside the hash of what applying it to the same
file produces, so reopening parses the file, does the steps again, and refuses
if the answer is a different mesh.  These tests hold that round trip, and the
two refusals that keep it honest: a document that already carries records is
not repaired under them, and a replay that no longer applies stops the open.

The fixture is in centimetres on purpose.  A repair measures in millimetres -
how wide the scanner's shadow is - so running it on source arrays would search
ten times too far, and the drum here is sized so that mistake could not pass
unnoticed.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import hashlib
import unittest

import numpy as np

from src.core.artifact_document import (
    ArtifactDocument,
    DerivedRecord,
    RecordLifecycleStatus,
)
from src.core.artifact_mesh_bodies import boundary_rings, diagnose_mesh_bodies, face_bodies
from src.core.artifact_mesh_repair import (
    MESH_REPAIR_EXTENSION_KEY,
    RingPairJoin,
    body_sha256,
    ring_sha256,
)
from src.core.artifact_session import ArtifactSession, ArtifactSessionError
from src.core.mesh_import_recipe import current_mesh_import_recipe
from src.core.mesh_loader import MeshData
from src.core.source_identity import SourceFingerprint


STAMP = "2026-09-09T00:00:00Z"


def _ring(radius: float, height: float, columns: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, columns, endpoint=False)
    return np.column_stack(
        [radius * np.cos(angles), radius * np.sin(angles), np.full(columns, height)]
    )


def _defective_scan(columns: int = 24, rows: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """A hollow drum in centimetres with a two-skinned piece standing in it.

    The piece's two edges stop 0.01 cm - a tenth of a millimetre - short of
    the inner wall, which is the ring of shadow a scanner's head leaves where
    it cannot reach into a joint.
    """

    heights = np.linspace(0.0, 2.0, rows + 1)
    outer = [_ring(1.00, z, columns) for z in heights]
    inner = [_ring(0.95, z, columns) for z in heights]
    piece_top = _ring(0.94, 1.06, columns)
    piece_under = _ring(0.94, 0.94, columns)
    vertices = np.concatenate(
        [
            *outer,
            *inner,
            piece_top,
            np.array([[0.0, 0.0, 1.30]]),
            piece_under,
            np.array([[0.0, 0.0, 1.20]]),
        ]
    )
    faces: list[list[int]] = []

    def quad(a: int, b: int, c: int, d: int) -> None:
        faces.append([a, b, c])
        faces.append([a, c, d])

    outer_base = 0
    inner_base = (rows + 1) * columns
    for row in range(rows):
        for i in range(columns):
            j = (i + 1) % columns
            quad(
                outer_base + row * columns + i,
                outer_base + row * columns + j,
                outer_base + (row + 1) * columns + j,
                outer_base + (row + 1) * columns + i,
            )
            quad(
                inner_base + row * columns + j,
                inner_base + row * columns + i,
                inner_base + (row + 1) * columns + i,
                inner_base + (row + 1) * columns + j,
            )
    for i in range(columns):
        j = (i + 1) % columns
        quad(outer_base + j, outer_base + i, inner_base + i, inner_base + j)
        quad(
            outer_base + rows * columns + i,
            outer_base + rows * columns + j,
            inner_base + rows * columns + j,
            inner_base + rows * columns + i,
        )
    top_base = 2 * (rows + 1) * columns
    top_apex = top_base + columns
    under_base = top_apex + 1
    under_apex = under_base + columns
    for i in range(columns):
        j = (i + 1) % columns
        faces.append([top_apex, top_base + i, top_base + j])
        faces.append([under_apex, under_base + j, under_base + i])
    return vertices, np.asarray(faces, dtype=np.int64)


def _mesh() -> MeshData:
    vertices, faces = _defective_scan()
    return MeshData(
        vertices=vertices,
        faces=np.asarray(faces, dtype=np.int32),
        unit="cm",
        filepath=Path("/source/split.ply"),
        source_identity=SourceFingerprint(
            sha256="b" * 64,
            size_bytes=4096,
            mtime_ns=1,
            original_name="split.ply",
            format="ply",
        ),
        source_format="ply",
        source_import_recipe=current_mesh_import_recipe("ply"),
    )


def _session() -> ArtifactSession:
    return ArtifactSession.create_from_source(
        _mesh(),
        resolved_source_path="/source/split.ply",
        unit="cm",
        axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
        handedness="right",
        software_version="0.7.0",
        operator="tester",
        created_at=STAMP,
        document_id="artifact:repair-test",
        metadata_revision_id="metadata:m1",
        align_revision_id="align:a1",
    )


def _join(session: ArtifactSession) -> RingPairJoin:
    faces = np.asarray(session.source_mesh.faces, dtype=np.int64)
    rings = boundary_rings(faces)
    labels = face_bodies(faces)
    wall = int(np.argmax(np.bincount(labels)))
    return RingPairJoin(
        first_ring_sha256=ring_sha256(rings[0]),
        second_ring_sha256=ring_sha256(rings[1]),
        body_sha256=body_sha256(faces, np.flatnonzero(labels == wall)),
    )


class TestMeshRepairSession(unittest.TestCase):
    def test_the_scan_arrives_split_and_the_repair_makes_it_one_body(self):
        session = _session()
        before = diagnose_mesh_bodies(
            np.asarray(session.materialize().mesh.vertices, dtype=np.float64),
            np.asarray(session.source_mesh.faces, dtype=np.int64),
        )
        self.assertEqual(len(before.bodies), 3)
        self.assertTrue(before.needs_decision)

        repaired = session.commit_mesh_repair(
            [_join(session)], operator="tester", created_at=STAMP
        )
        after = diagnose_mesh_bodies(
            np.asarray(repaired.materialize().mesh.vertices, dtype=np.float64),
            np.asarray(repaired.source_mesh.faces, dtype=np.int64),
        )
        self.assertTrue(after.one_body)
        self.assertFalse(after.needs_decision)

        # The repaired geometry is the one the session now works on, not a
        # second one sitting unused beside the split original.
        active = repaired.document.source_metadata_revision_index[
            repaired.document.active_source_metadata_revision_id or ""
        ]
        self.assertNotEqual(
            active.geometry_revision_id,
            session.document.geometry_revisions[0].id,
        )
        self.assertEqual(
            repaired.verified_geometry.geometry_revision_id,
            active.geometry_revision_id,
        )

    def test_the_document_keeps_the_decision_not_the_mesh(self):
        repaired = _session().commit_mesh_repair(
            [_join(_session())], operator="tester", created_at=STAMP
        )
        geometry = repaired.document.geometry_revision_index[
            repaired.document.source_metadata_revision_index[
                repaired.document.active_source_metadata_revision_id or ""
            ].geometry_revision_id
        ]
        stored = dict(geometry.extensions[MESH_REPAIR_EXTENSION_KEY])

        parent = repaired.document.geometry_revisions[0]
        self.assertEqual(stored["parent_geometry_revision_id"], parent.id)
        self.assertNotEqual(geometry.id, parent.id)
        self.assertEqual(len(stored["joins"]), 1)
        self.assertEqual(stored["joins"][0]["kind"], "sew_ring_pair_to_body/v1")
        self.assertEqual(stored["receipt"]["body_count_before"], 3)
        self.assertEqual(stored["receipt"]["body_count_after"], 1)
        # Nothing in the document is the mesh itself: it is the file, the
        # decision, and the hash of what the two make.
        self.assertNotIn("faces", stored)
        self.assertEqual(len(geometry.geometry_sha256), 64)

    def test_a_repaired_document_reopens_as_the_same_mesh(self):
        repaired = _session().commit_mesh_repair(
            [_join(_session())], operator="tester", created_at=STAMP
        )
        saved = ArtifactDocument.from_dict(repaired.document.to_dict())

        # Reopening starts from the file, exactly as it was parsed.
        reopened = ArtifactSession.bind_loaded_document(
            saved, _mesh(), resolved_source_path="/source/split.ply"
        )
        self.assertTrue(
            np.array_equal(
                np.asarray(reopened.source_mesh.faces),
                np.asarray(repaired.source_mesh.faces),
            )
        )
        self.assertEqual(
            reopened.verified_geometry.geometry_sha256,
            repaired.verified_geometry.geometry_sha256,
        )

    def test_the_standing_on_its_axis_is_carried_because_no_vertex_moved(self):
        session = _session()
        repaired = session.commit_mesh_repair(
            [_join(session)], operator="tester", created_at=STAMP
        )
        active = repaired.document.align_revision_index[
            repaired.document.active_align_revision_id or ""
        ]
        parent = session.document.align_revision_index[
            session.document.active_align_revision_id or ""
        ]
        self.assertEqual(active.matrix4x4, parent.matrix4x4)
        self.assertEqual(active.recipe["kind"], "carried_through_mesh_repair")
        self.assertIs(active.qc["vertices_unchanged"], True)

    def test_a_document_that_already_carries_records_is_not_repaired_under(self):
        session = _session()
        geometry_id = session.document.geometry_revisions[0].id
        align_id = session.document.active_align_revision_id or ""
        measured = replace(
            session.document,
            records=(
                DerivedRecord(
                    id="record:measured-before-the-repair",
                    type="derived.test",
                    geometry_revision_id=geometry_id,
                    align_revision_id=align_id,
                    depends_on_record_ids=(),
                    geometry_ref="payload:measured",
                    recipe={"kind": "test"},
                    recipe_hash=hashlib.sha256(b'{"kind":"test"}').hexdigest(),
                    selection_hash=None,
                    qc={},
                    lifecycle_status=RecordLifecycleStatus.READY,
                    created_at=STAMP,
                    operator="tester",
                ),
            ),
        )
        carrying = session.with_document(measured)

        with self.assertRaises(ArtifactSessionError) as caught:
            carrying.commit_mesh_repair(
                [_join(session)], operator="tester", created_at=STAMP
            )
        message = str(caught.exception)
        self.assertIn("already carries 1 records", message)
        self.assertIn("before anything is measured", message)

    def test_a_repair_with_no_joins_changes_nothing_and_is_refused(self):
        with self.assertRaises(ArtifactSessionError) as caught:
            _session().commit_mesh_repair([], operator="tester", created_at=STAMP)
        self.assertIn("no joins", str(caught.exception))

    def test_a_replay_that_no_longer_applies_stops_the_open(self):
        repaired = _session().commit_mesh_repair(
            [_join(_session())], operator="tester", created_at=STAMP
        )
        saved = ArtifactDocument.from_dict(repaired.document.to_dict())

        # The same file, one triangle turned over: the edges the decision
        # named are no longer the edges that are there.
        altered = _mesh()
        faces = np.asarray(altered.faces).copy()
        faces[0] = faces[0][::-1]
        altered.faces = faces

        with self.assertRaises(ArtifactSessionError) as caught:
            ArtifactSession.bind_loaded_document(
                saved, altered, resolved_source_path="/source/split.ply"
            )
        self.assertIn("repair", str(caught.exception).lower())


if __name__ == "__main__":
    unittest.main()
