"""Shared result types for mesh flattening."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, NotRequired, Optional, TypedDict

import numpy as np

from .logging_utils import log_once
from .mesh_loader import MeshData


class FlattenResultMeta(TypedDict, total=False):
    flatten_method: str
    requested_flatten_method: str
    initial_method: str
    iterations: int
    smooth_iters: int
    smooth_strength: float
    recommended_method: str
    recommended_label: str
    recommendation_reason: str
    recommendation_confidence: float
    recommendation_badge: str
    recommendation_fallback_chain: list[str]
    recommendation_alternatives: list[dict[str, str]]
    fallback_from: str
    fallback_reason: str
    fallback_used_method: str
    fallback_chain: list[str]
    sectionwise: bool
    sectionwise_fallback: bool
    sectionwise_reason: str
    flatten_size_warning: bool
    flatten_size_guard_applied: bool
    flatten_size_guard_scale: float
    distortion_summary: dict[str, Any]
    extra: NotRequired[dict[str, Any]]


@dataclass
class FlattenedMesh:
    """Flattened mesh result with UVs and optional metadata."""

    uv: np.ndarray
    faces: np.ndarray
    original_mesh: MeshData
    distortion_per_face: Optional[np.ndarray] = None
    scale: float = 1.0
    meta: dict[str, Any] = field(default_factory=dict)

    _bounds: Optional[np.ndarray] = field(default=None, repr=False)

    def __post_init__(self):
        uv = np.asarray(self.uv, dtype=np.float64)
        if uv.ndim == 1:
            if uv.size % 2 == 0:
                uv = uv.reshape(-1, 2)
            else:
                uv = uv.reshape(0, 2)
        elif uv.ndim == 2:
            if uv.shape[1] >= 2:
                uv = uv[:, :2]
            else:
                uv = uv.reshape(0, 2)
        else:
            uv = uv.reshape(0, 2)
        self.uv = uv

        faces = np.asarray(self.faces, dtype=np.int32)
        if faces.size == 0:
            faces = faces.reshape(0, 3)
        elif faces.ndim == 1:
            if faces.size % 3 == 0:
                faces = faces.reshape(-1, 3)
            else:
                faces = faces.reshape(0, 3)
        elif faces.ndim == 2:
            if faces.shape[1] < 3:
                faces = faces.reshape(0, 3)
            else:
                faces = faces[:, :3]
        else:
            faces = faces.reshape(0, 3)

        try:
            if self.uv.shape[0] > 0 and faces.shape[0] > 0:
                valid = (faces >= 0) & (faces < int(self.uv.shape[0]))
                keep = np.all(valid, axis=1)
                faces = faces[keep]
        except Exception:
            log_once(
                logging.getLogger(__name__),
                "flattener:FlattenedMesh_post_init_face_filter",
                logging.WARNING,
                "FlattenedMesh.__post_init__ face index filtering failed",
                exc_info=True,
            )
        self.faces = faces

    @property
    def n_vertices(self) -> int:
        return len(self.uv)

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    @property
    def bounds(self) -> np.ndarray:
        if self._bounds is None:
            uv = np.asarray(self.uv, dtype=np.float64)
            if uv.ndim != 2 or uv.size == 0:
                self._bounds = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
                return self._bounds

            finite = np.all(np.isfinite(uv), axis=1)
            uv_f = uv[finite]
            if uv_f.size == 0:
                self._bounds = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float64)
                return self._bounds

            self._bounds = np.array([uv_f.min(axis=0), uv_f.max(axis=0)])
        return self._bounds

    @property
    def extents(self) -> np.ndarray:
        return self.bounds[1] - self.bounds[0]

    @property
    def width(self) -> float:
        return self.extents[0] * self.scale

    @property
    def height(self) -> float:
        return self.extents[1] * self.scale

    @property
    def mean_distortion(self) -> float:
        if self.distortion_per_face is None:
            return 0.0
        return float(np.mean(self.distortion_per_face))

    @property
    def max_distortion(self) -> float:
        if self.distortion_per_face is None:
            return 0.0
        return float(np.max(self.distortion_per_face))

    def normalize(self) -> "FlattenedMesh":
        if self.uv.ndim != 2 or self.uv.size == 0:
            return FlattenedMesh(
                uv=self.uv.copy(),
                faces=self.faces,
                original_mesh=self.original_mesh,
                distortion_per_face=self.distortion_per_face,
                scale=float(self.scale),
                meta=dict(getattr(self, "meta", {}) or {}),
            )

        min_uv = self.uv.min(axis=0)
        max_uv = self.uv.max(axis=0)
        extent = max_uv - min_uv
        extent[extent == 0] = 1.0

        normalized_uv = (self.uv - min_uv) / extent
        new_scale = self.scale * max(extent)

        return FlattenedMesh(
            uv=normalized_uv,
            faces=self.faces,
            original_mesh=self.original_mesh,
            distortion_per_face=self.distortion_per_face,
            scale=new_scale,
            meta=dict(getattr(self, "meta", {}) or {}),
        )

    def get_pixel_coordinates(self, width: int, height: int) -> np.ndarray:
        normalized = self.normalize()
        pixels = normalized.uv.copy()
        pixels[:, 0] *= width - 1
        pixels[:, 1] *= height - 1
        pixels[:, 1] = (height - 1) - pixels[:, 1]
        return pixels.astype(np.int32)
