from __future__ import annotations

from typing import Any, cast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_UNIT = "cm"
PLY_BYTES = textwrap.dedent(
    """\
    ply
    format ascii 1.0
    comment independent-process artifact roundtrip fixture
    element vertex 5
    property float x
    property float y
    property float z
    element face 4
    property list uchar int vertex_indices
    end_header
    1.25 -2.5 0.75
    4.5 -1.25 1.5
    3.75 2.0 2.25
    -0.5 1.5 -1.0
    2.0 0.25 4.0
    3 0 1 4
    3 1 2 4
    3 2 3 4
    3 3 0 4
    """
).encode("ascii")


PROCESS_A = textwrap.dedent(
    """\
    import json
    import os
    from pathlib import Path
    import sys

    import numpy as np

    from src.core.artifact_session import ArtifactSession
    from src.core.geometry_identity import mesh_geometry_sha256
    from src.core.mesh_loader import MeshLoader
    from src.core.project_file import save_artifact_project


    source_path = Path(sys.argv[1]).resolve()
    project_path = Path(sys.argv[2]).resolve()
    mesh = MeshLoader(default_unit="mm").load(source_path, unit="cm")
    source_identity = mesh.source_identity
    if source_identity is None:
        raise RuntimeError("source fingerprint was not captured")

    session = ArtifactSession.create_from_source(
        mesh,
        resolved_source_path=str(source_path),
        unit="cm",
        axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
        handedness="right",
        software_version="0.1.0-test",
        operator="new-process-gate",
        created_at="2026-07-11T00:00:00Z",
        document_id="artifact:new-process-roundtrip",
        metadata_revision_id="metadata:source-cm",
        align_revision_id="align:identity",
    )
    session = session.commit_preview(
        translation_mm=(13.25, -7.5, 4.75),
        rotation_deg=(17.0, -23.0, 61.0),
        scale=1.0,
        pivot_mm=(12.5, -5.0, 7.5),
        operator="new-process-gate",
        created_at="2026-07-11T00:00:01Z",
        revision_id="align:nontrivial",
    )

    computed_geometry_hash = mesh_geometry_sha256(mesh)
    if computed_geometry_hash != session.verified_geometry.geometry_sha256:
        raise RuntimeError("session geometry identity was not computed from decoded geometry")

    save_artifact_project(project_path, session.document)
    projection = session.materialize()
    active_metadata_id = session.document.active_source_metadata_revision_id
    active_align_id = session.document.active_align_revision_id
    if active_metadata_id is None or active_align_id is None:
        raise RuntimeError("active metadata/Align context is missing")
    metadata = session.document.source_metadata_revision_index[active_metadata_id]
    active_align = session.document.align_revision_index[active_align_id]

    json.dump(
        {
            "pid": os.getpid(),
            "source_path": str(source_path),
            "source_sha256": source_identity.sha256,
            "source_size_bytes": source_identity.size_bytes,
            "geometry_sha256": computed_geometry_hash,
            "active_align_id": active_align.id,
            "active_align_matrix": np.asarray(active_align.matrix, dtype=np.float64).tolist(),
            "world_vertices": np.asarray(
                projection.mesh.vertices, dtype=np.float64
            ).tolist(),
            "metadata_unit": metadata.unit,
            "parser_format": mesh.source_format,
        },
        sys.stdout,
        sort_keys=True,
    )
    """
)


PROCESS_B = textwrap.dedent(
    """\
    import json
    import os
    from pathlib import Path
    import sys

    import numpy as np

    from src.core.artifact_session import ArtifactSession
    from src.core.geometry_identity import mesh_geometry_sha256
    from src.core.mesh_loader import MeshLoader
    from src.core.project_file import load_artifact_project


    project_path = Path(sys.argv[1]).resolve()
    relocated_source_path = Path(sys.argv[2]).resolve()
    document = load_artifact_project(project_path)

    active_metadata_id = document.active_source_metadata_revision_id
    active_align_id = document.active_align_revision_id
    if active_metadata_id is None or active_align_id is None:
        raise RuntimeError("active metadata/Align context is missing")
    metadata = document.source_metadata_revision_index[active_metadata_id]
    geometry = document.geometry_revision_index[metadata.geometry_revision_id]
    parser_format = geometry.import_recipe.get("format")
    if not isinstance(parser_format, str) or not parser_format:
        raise RuntimeError("saved parser format is missing")

    # Deliberately re-open and re-hash the relocated source. The geometry digest
    # below comes from the newly decoded mesh, never from GeometryRevision.
    mesh = MeshLoader(default_unit="mm").load(
        relocated_source_path,
        unit=metadata.unit,
        source_format=parser_format,
    )
    source_identity = mesh.source_identity
    if source_identity is None:
        raise RuntimeError("relocated source fingerprint was not captured")
    computed_geometry_hash = mesh_geometry_sha256(
        mesh,
        scope=geometry.geometry_hash_scope,
    )

    rebound = ArtifactSession.bind_loaded_document(
        document,
        mesh,
        resolved_source_path=str(relocated_source_path),
    )
    if computed_geometry_hash != rebound.verified_geometry.geometry_sha256:
        raise RuntimeError("rebound session did not retain independently computed identity")
    projection = rebound.materialize()
    active_align = rebound.document.align_revision_index[active_align_id]

    json.dump(
        {
            "pid": os.getpid(),
            "source_path": str(relocated_source_path),
            "source_sha256": source_identity.sha256,
            "source_size_bytes": source_identity.size_bytes,
            "geometry_sha256": computed_geometry_hash,
            "active_align_id": active_align.id,
            "active_align_matrix": np.asarray(active_align.matrix, dtype=np.float64).tolist(),
            "world_vertices": np.asarray(
                projection.mesh.vertices, dtype=np.float64
            ).tolist(),
            "metadata_unit": metadata.unit,
            "parser_format": parser_format,
            "loaded_mesh_unit": mesh.unit,
            "loaded_mesh_parser_format": mesh.source_format,
        },
        sys.stdout,
        sort_keys=True,
    )
    """
)


class TestArtifactIndependentProcessRoundtrip(unittest.TestCase):
    def _run_worker(self, program: str, *arguments: Path) -> dict[str, Any]:
        environment = os.environ.copy()
        inherited_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(REPOSITORY_ROOT), inherited_pythonpath) if part
        )
        completed = subprocess.run(
            [sys.executable, "-c", program, *(str(path) for path in arguments)],
            cwd=REPOSITORY_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if completed.returncode != 0:
            self.fail(
                "independent worker failed "
                f"(exit={completed.returncode})\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            self.fail(
                "independent worker did not emit one JSON result\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}\n"
                f"decode error: {exc}"
            )
        if not isinstance(payload, dict):
            self.fail(
                f"independent worker emitted {type(payload).__name__}, expected object"
            )
        return cast(dict[str, Any], payload)

    def test_open_align_save_relocate_reopen_materialize_across_processes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            workspace = Path(temporary_directory)
            source_path = workspace / "capture" / "excavated-fragment.ply"
            project_path = workspace / "project" / "excavated-fragment.amr"
            source_path.parent.mkdir(parents=True)
            project_path.parent.mkdir(parents=True)
            source_path.write_bytes(PLY_BYTES)

            process_a = self._run_worker(PROCESS_A, source_path, project_path)

            relocated_source_path = (
                workspace / "archive" / "renamed" / "fragment.raw-scan"
            )
            relocated_source_path.parent.mkdir(parents=True)
            source_path.replace(relocated_source_path)
            self.assertFalse(source_path.exists())
            self.assertTrue(project_path.is_file())
            self.assertEqual(relocated_source_path.read_bytes(), PLY_BYTES)

            process_b = self._run_worker(
                PROCESS_B,
                project_path,
                relocated_source_path,
            )

        self.assertNotEqual(process_a["pid"], process_b["pid"])
        self.assertNotEqual(process_a["source_path"], process_b["source_path"])
        self.assertTrue(str(process_a["source_path"]).endswith(".ply"))
        self.assertTrue(str(process_b["source_path"]).endswith(".raw-scan"))

        self.assertEqual(process_a["source_sha256"], process_b["source_sha256"])
        self.assertEqual(
            process_a["source_size_bytes"],
            process_b["source_size_bytes"],
        )
        self.assertEqual(process_a["source_size_bytes"], len(PLY_BYTES))
        self.assertEqual(process_a["geometry_sha256"], process_b["geometry_sha256"])
        self.assertEqual(process_a["active_align_id"], "align:nontrivial")
        self.assertEqual(process_a["active_align_id"], process_b["active_align_id"])
        np.testing.assert_allclose(
            process_a["active_align_matrix"],
            process_b["active_align_matrix"],
            rtol=0.0,
            atol=1e-12,
        )
        self.assertFalse(
            np.allclose(
                np.asarray(process_a["active_align_matrix"], dtype=np.float64),
                np.eye(4, dtype=np.float64),
                rtol=0.0,
                atol=1e-12,
            )
        )
        np.testing.assert_allclose(
            process_a["world_vertices"],
            process_b["world_vertices"],
            rtol=0.0,
            atol=1e-12,
        )

        self.assertEqual(process_a["metadata_unit"], SOURCE_UNIT)
        self.assertEqual(process_b["metadata_unit"], SOURCE_UNIT)
        self.assertEqual(process_a["parser_format"], "ply")
        self.assertEqual(process_b["parser_format"], "ply")
        self.assertEqual(process_b["loaded_mesh_unit"], SOURCE_UNIT)
        self.assertEqual(process_b["loaded_mesh_parser_format"], "ply")


if __name__ == "__main__":
    unittest.main()
