"""The grid closing: hairline slivers are the grid's, not the artifact's.

Near a silhouette the projected triangles are thinner than a lattice cell,
and snapping each one to the grid leaves gaps between them - holes one cell
wide and a few cells long, and pinches where the outline touches itself.
Every surface with relief has them, a finer grid only finds smaller ones,
and on a rough wall they made the outline refuse itself.  Outline algorithm
1.1.0 closes the lattice union by one cell; 1.0.0 records recompute as they
were.
"""

from __future__ import annotations

import pytest

from src.core.artifact_document import canonical_recipe_hash
from src.core.artifact_outline_extractor import (
    OUTLINE_ALGORITHM_VERSION,
    OUTLINE_GRID_CLOSING_RADIUS_CELLS,
    OUTLINE_LEGACY_ALGORITHM_VERSION,
    ArtifactVectorExtractionError,
    compute_artifact_outline,
    extract_outline_geometry,
    outline_recipe,
    validate_outline_record_contract,
)
from src.core.artifact_outline_topology import validate_outline_topology
from synthetic_vessel import grained_surface, positioned_vessel_session


@pytest.fixture(scope="module")
def grained():
    session, _vertices, _faces = positioned_vessel_session(
        segments=48, rings=24, relief=grained_surface
    )
    # The outline is taken of the artifact as positioned, not of the source
    # arrays, so read the mesh back through the document.
    mesh = session.materialize().mesh
    return session, mesh.vertices, mesh.faces


def test_the_lattice_union_leaves_slivers_and_the_closing_removes_them(grained) -> None:
    """The lattice union of a closed surface shows hairline holes it does not
    have; after the closing there are none, and the area has moved by parts
    per million."""

    _session, vertices, faces = grained
    legacy = extract_outline_geometry(
        vertices, faces, "front", precision_grid_mm=0.02,
        algorithm_version=OUTLINE_LEGACY_ALGORITHM_VERSION,
    )
    closed = extract_outline_geometry(vertices, faces, "front", precision_grid_mm=0.02)

    assert legacy.qc["hole_count"] > 0, legacy.qc
    assert closed.qc["hole_count"] == 0
    assert closed.qc["grid_closing_hole_fill_count"] == legacy.qc["hole_count"]
    assert closed.qc["grid_closing_component_merge_count"] == 0
    assert closed.qc["grid_closing_radius_cells"] == OUTLINE_GRID_CLOSING_RADIUS_CELLS
    assert abs(closed.qc["grid_closing_area_delta_mm2"]) < 1e-3 * closed.qc["outline_area_mm2"]
    assert closed.qc["component_count"] == legacy.qc["component_count"] == 1
    # Both are outlines in their own right.
    validate_outline_topology(legacy.payload)
    validate_outline_topology(closed.payload)
    assert closed.payload.sha256 != legacy.payload.sha256


def test_each_version_reproduces_its_own_recipe_and_contract(grained) -> None:
    session, _vertices, _faces = grained
    computation = compute_artifact_outline(session, "front", precision_grid_mm=0.02)
    recipe = computation.recipe

    assert recipe["algorithm_version"] == OUTLINE_ALGORITHM_VERSION
    assert recipe["grid_closing"] == {
        "join_style": "mitre",
        "operation": "buffer_out_then_in_then_set_precision/v1",
        "radius_cells": OUTLINE_GRID_CLOSING_RADIUS_CELLS,
    }
    validate_outline_record_contract(computation.payload, recipe)

    legacy_recipe = outline_recipe(
        "front", precision_grid_mm=0.02, algorithm_version=OUTLINE_LEGACY_ALGORITHM_VERSION
    )
    assert "grid_closing" not in legacy_recipe
    assert canonical_recipe_hash(legacy_recipe) != canonical_recipe_hash(recipe)
    # The legacy recipe is a contract of its own and still hashes as it did.
    assert legacy_recipe["algorithm_version"] == OUTLINE_LEGACY_ALGORITHM_VERSION
    assert legacy_recipe["precision_model"].endswith("then_balanced_union_all/v1")
    with pytest.raises(ArtifactVectorExtractionError, match="algorithm_version"):
        outline_recipe("front", precision_grid_mm=0.02, algorithm_version="0.9.0")
