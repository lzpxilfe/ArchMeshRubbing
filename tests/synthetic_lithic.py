"""뗀석기: a flaked stone tool as a closed solid, for tests that need one.

Everything the program was built on so far turns on an axis - a pot, a
tile's cylinder.  A stone tool does not.  The guidelines put it flat on the
table in the direction it was used ([K2] 2014, 석기 실측 방법: an arrowhead
tang down, a hoe blade down), draw the plan, and cut a long and a cross
section - or, better, draw all six projections in third-angle ([K2]), and
a dagger always with its side view.  On the plan the flake scars are drawn
as inner lines (내선, [K1] 2013 p. 48): the ridges where one scar meets the
next are what tell a reader how the tool was made.

This tool is a leaf-shaped flake worked on one face.  Its dorsal face is
the upper envelope of a handful of planes, so the scars are flat and the
ridges between them are sharp, and its ventral face is one smooth surface
with the bulb of percussion swelling near the platform end.  The two meet
at the margin.  Sizes are those of a hand-sized biface, not a copy of any
particular tool, and nothing here is random.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from src.core.artifact_session import ArtifactSession
from src.core.mesh_import_recipe import current_mesh_import_recipe
from src.core.mesh_loader import MeshData
from src.core.source_identity import SourceFingerprint


@dataclass(frozen=True, slots=True)
class Facet:
    """One flake scar on the dorsal face: a plane, in millimetres.

    ``height_mm`` is the plane's height over the centre of the plan and
    ``slope`` how steeply it falls, per millimetre, in the direction
    ``direction_deg`` measured in the plan.
    """

    height_mm: float
    slope: float
    direction_deg: float


@dataclass(frozen=True, slots=True)
class LithicShape:
    #: Half the length, along +X, and the plan's fullness: 0 is an ellipse
    #: and more makes the platform end (-X) broader than the tip (+X).
    half_length_mm: float = 42.0
    half_width_mm: float = 28.0
    taper: float = 0.28
    #: The ventral face's greatest depth below the margin, and the bulb's.
    ventral_depth_mm: float = 5.0
    bulb_mm: float = 2.5
    #: The dorsal scars.  Their upper envelope, tapered to nothing at the
    #: margin, is the dorsal face.
    facets: tuple[Facet, ...] = (
        Facet(height_mm=14.0, slope=0.30, direction_deg=90.0),
        Facet(height_mm=14.0, slope=0.30, direction_deg=-90.0),
        Facet(height_mm=12.5, slope=0.22, direction_deg=0.0),
        Facet(height_mm=12.0, slope=0.26, direction_deg=180.0),
        Facet(height_mm=11.0, slope=0.34, direction_deg=45.0),
        Facet(height_mm=11.0, slope=0.34, direction_deg=-135.0),
    )

    def __post_init__(self) -> None:
        for name in ("half_length_mm", "half_width_mm", "ventral_depth_mm"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not (0.0 <= self.taper < 0.6):
            raise ValueError("taper must be between 0 and 0.6")
        if not self.facets:
            raise ValueError("a flaked tool has at least one scar")


BIFACE_SHAPE = LithicShape()


def plan_radius(shape: LithicShape, angle_rad: float) -> float:
    """The margin's distance from the plan's centre in one direction."""

    c, s = math.cos(angle_rad), math.sin(angle_rad)
    # An ellipse, pinched toward the tip: the platform end stays broad.
    ellipse = 1.0 / math.sqrt(
        (c / shape.half_length_mm) ** 2 + (s / shape.half_width_mm) ** 2
    )
    return ellipse * (1.0 - shape.taper * 0.5 * (1.0 + c) * abs(s))


def dorsal_height(shape: LithicShape, x_mm: float, y_mm: float) -> float:
    """The upper envelope of the scars over one point of the plan."""

    best = -math.inf
    for facet in shape.facets:
        along = x_mm * math.cos(math.radians(facet.direction_deg)) + y_mm * math.sin(
            math.radians(facet.direction_deg)
        )
        best = max(best, facet.height_mm - facet.slope * along)
    return best


def ventral_depth(shape: LithicShape, x_mm: float, y_mm: float) -> float:
    """How far below the margin the ventral face lies at one point."""

    bulb_centre = (-shape.half_length_mm * 0.55, 0.0)
    spread = shape.half_width_mm * 0.45
    reach = ((x_mm - bulb_centre[0]) ** 2 + (y_mm - bulb_centre[1]) ** 2) / (
        2.0 * spread * spread
    )
    return shape.ventral_depth_mm + shape.bulb_mm * math.exp(-reach)


def flaked_tool(
    shape: LithicShape = BIFACE_SHAPE,
    *,
    rings: int = 36,
    segments: int = 120,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (vertices, faces) for one closed flaked tool.

    The plan is sampled on rings from the centre to the margin; the dorsal
    and ventral faces share the margin ring, so the solid is watertight.
    Both faces are drawn to the margin by a taper, so the edge is an edge.
    """

    if rings < 2 or segments < 8:
        raise ValueError("a tool needs at least 2 rings and 8 segments")
    vertices: list[list[float]] = []
    faces: list[list[int]] = []

    def taper(s: float) -> float:
        # 1 at the centre, 0 at the margin, and flat at the centre.
        return 1.0 - s ** 4

    def surface(*, dorsal: bool) -> tuple[int, int]:
        centre = len(vertices)
        z = dorsal_height(shape, 0.0, 0.0) if dorsal else -ventral_depth(shape, 0.0, 0.0)
        vertices.append([0.0, 0.0, float(z)])
        first_ring = len(vertices)
        for ring in range(1, rings + 1):
            s = ring / rings
            for segment in range(segments):
                angle = 2.0 * math.pi * segment / segments
                r = s * plan_radius(shape, angle)
                x, y = r * math.cos(angle), r * math.sin(angle)
                if dorsal:
                    z = dorsal_height(shape, x, y) * taper(s)
                else:
                    z = -ventral_depth(shape, x, y) * taper(s)
                vertices.append([x, y, float(z)])
        return centre, first_ring

    dorsal_centre, dorsal_first = surface(dorsal=True)
    ventral_centre, ventral_first = surface(dorsal=False)
    # The margin ring is one ring: the ventral face's last ring is dropped
    # and its faces sewn to the dorsal margin instead.
    margin_first = dorsal_first + (rings - 1) * segments
    ventral_last = ventral_first + (rings - 1) * segments
    kept = vertices[:ventral_last]
    vertices = kept

    def ring_index(first: int, ring: int, segment: int, *, ventral: bool) -> int:
        if ventral and ring == rings:
            return margin_first + segment % segments
        return first + (ring - 1) * segments + segment % segments

    for dorsal in (True, False):
        centre = dorsal_centre if dorsal else ventral_centre
        first = dorsal_first if dorsal else ventral_first
        for segment in range(segments):
            a = ring_index(first, 1, segment, ventral=not dorsal)
            b = ring_index(first, 1, segment + 1, ventral=not dorsal)
            faces.append([centre, a, b] if dorsal else [centre, b, a])
        for ring in range(1, rings):
            for segment in range(segments):
                a = ring_index(first, ring, segment, ventral=not dorsal)
                b = ring_index(first, ring, segment + 1, ventral=not dorsal)
                c = ring_index(first, ring + 1, segment + 1, ventral=not dorsal)
                d = ring_index(first, ring + 1, segment, ventral=not dorsal)
                # Wound to face out of the stone: up on the dorsal face,
                # down on the ventral.
                if dorsal:
                    faces.append([a, d, c])
                    faces.append([a, c, b])
                else:
                    faces.append([a, c, d])
                    faces.append([a, b, c])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def plan_area_mm2(shape: LithicShape, *, steps: int = 20000) -> float:
    """The plan's area, by quadrature, to compare an outline against."""

    total = 0.0
    for index in range(steps):
        angle = 2.0 * math.pi * (index + 0.5) / steps
        total += 0.5 * plan_radius(shape, angle) ** 2
    return total * (2.0 * math.pi / steps)


def lithic_session(
    shape: LithicShape = BIFACE_SHAPE,
    *,
    rings: int = 36,
    segments: int = 120,
    document_id: str = "artifact:biface",
) -> tuple[ArtifactSession, np.ndarray, np.ndarray]:
    """One tool in a session, lying as it would on the table: no Align.

    A stone tool is positioned by its use direction, by hand, and this one
    already lies that way - the tip on +X, the platform on -X, the dorsal
    face up - so the canonical frame is the source frame.
    """

    vertices, faces = flaked_tool(shape, rings=rings, segments=segments)
    mesh = MeshData(
        vertices=vertices,
        faces=faces,
        unit="mm",
        filepath=Path("/source/biface.ply"),
        source_identity=SourceFingerprint(
            sha256="b" * 64,
            size_bytes=int(vertices.size),
            mtime_ns=1,
            original_name="biface.ply",
            format="ply",
        ),
        source_format="ply",
        source_import_recipe=current_mesh_import_recipe("ply"),
    )
    session = ArtifactSession.create_from_source(
        mesh,
        resolved_source_path="/source/biface.ply",
        unit="mm",
        axes={"source_x": "+X", "source_y": "+Y", "source_z": "+Z"},
        handedness="right",
        software_version="lithic-test",
        operator="tester",
        created_at="2026-09-05T00:00:00Z",
        document_id=document_id,
        metadata_revision_id=f"metadata:{document_id}",
        align_revision_id=f"align:{document_id}",
    )
    return session, vertices, faces


__all__ = [
    "BIFACE_SHAPE",
    "Facet",
    "LithicShape",
    "dorsal_height",
    "flaked_tool",
    "lithic_session",
    "plan_area_mm2",
    "plan_radius",
    "ventral_depth",
]
