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

    def test_roundtrip_state_with_tile_slots(self):
        state = {
            "objects": [
                {
                    "name": "Tile",
                    "mesh": {"path": "C:/tmp/tile.obj", "source_scale_factor": 1.0},
                    "faces": {"selected": [1, 2, 3], "outer": [], "inner": [], "migu": []},
                    "tile_interpretation": {
                        "tile_class": "sugkiwa",
                        "split_scheme": "quarter",
                        "record_view": "top",
                        "record_strategy": "canonical_visible",
                        "saved_slots": [
                            {
                                "slot_key": "slot_1",
                                "label": "상면 기록 | 선택 3면",
                                "selected_faces": [1, 2, 3],
                                "tile_class": "sugkiwa",
                                "split_scheme": "quarter",
                                "axis_hint": {
                                    "source": "selected_patch_pca",
                                    "vector_world": [0.0, 1.0, 0.0],
                                    "origin_world": [0.0, 0.0, 0.0],
                                    "confidence": 0.8,
                                    "face_count": 3,
                                    "note": "slot axis",
                                },
                                "section_observations": [],
                                "mandrel_fit": {
                                    "radius_world": 22.5,
                                    "radius_spread_world": 0.6,
                                    "axis_origin_world": [0.0, 0.0, 0.0],
                                    "axis_vector_world": [0.0, 1.0, 0.0],
                                    "confidence": 0.7,
                                    "used_sections": 3,
                                    "used_points": 72,
                                    "scope": "현재 선택 표면",
                                    "note": "slot fit",
                                },
                                "record_view": "top",
                                "record_strategy": "canonical_visible",
                                "workflow_stage": "record_surface",
                                "note": "",
                                "updated_at_iso": "2026-03-20T12:00:00",
                            }
                        ],
                    },
                }
            ]
        }

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "tile_slots.amr"
            save_project(path, state, meta={})
            doc = load_project(path)

        loaded_state = doc.get("state", {})
        loaded_slots = (
            loaded_state.get("objects", [{}])[0]
            .get("tile_interpretation", {})
            .get("saved_slots", [])
        )
        self.assertEqual(len(loaded_slots), 1)
        self.assertEqual(loaded_slots[0]["slot_key"], "slot_1")
        self.assertEqual(loaded_slots[0]["selected_faces"], [1, 2, 3])
        self.assertEqual(loaded_slots[0]["record_view"], "top")
