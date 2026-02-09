import json
import tempfile
import unittest
from pathlib import Path

from src.core.project_file import load_project, save_project, PROJECT_FORMAT, PROJECT_VERSION


class TestProjectFile(unittest.TestCase):
    def test_roundtrip_zip_manifest(self):
        state = {
            "viewport": {"slice": {"enabled": True, "z": 12.34}},
            "objects": [
                {
                    "name": "Test",
                    "mesh": {"path": "C:/tmp/foo.stl", "source_scale_factor": 0.1},
                    "transform": {"t": [1.0, 2.0, 3.0], "r": [10.0, 20.0, 30.0], "s": 1.5},
                    "faces": {"outer": [1, 2, 3], "inner": [], "migu": []},
                }
            ],
        }
        meta = {"app": "ArchMeshRubbing", "version": "0.0.0"}

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sample.amr"
            save_project(path, state, meta=meta)
            doc = load_project(path)

        self.assertEqual(doc.get("format"), PROJECT_FORMAT)
        self.assertEqual(doc.get("version"), PROJECT_VERSION)
        self.assertEqual(doc.get("meta"), meta)
        self.assertEqual(doc.get("state"), state)

    def test_load_plain_json_fallback(self):
        state = {"hello": "world"}
        doc = {
            "format": PROJECT_FORMAT,
            "version": PROJECT_VERSION,
            "meta": {},
            "saved_at": "2026-01-01T00:00:00Z",
            "state": state,
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sample.json"
            path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            loaded = load_project(path)

        self.assertEqual(loaded.get("state"), state)

