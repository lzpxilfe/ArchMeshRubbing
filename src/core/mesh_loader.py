"""
Mesh Loader Module
메쉬 파일 로딩 및 데이터 구조 정의

Supports: OBJ, PLY, STL, OFF formats
"""

from dataclasses import dataclass, field
import hashlib
import os
import stat
import tempfile
from pathlib import Path
from typing import BinaryIO, Callable, Mapping, Optional, List, Union, cast
import logging
import numpy as np

from .logging_utils import log_once
from .mesh_import_recipe import (
    current_mesh_import_recipe,
    mesh_import_recipe_with_manifest,
    validate_mesh_import_recipe,
)
from .source_identity import SourceFingerprint, open_fingerprinted_file
from .source_manifest import (
    DEPENDENCY_RESOURCE_ROLE,
    MAX_SOURCE_MANIFEST_ENTRIES,
    PRIMARY_RESOURCE_ROLE,
    ResolvedSourceResource,
    SourceManifest,
    SourceManifestEntry,
    SourceManifestError,
    canonical_logical_path,
    fixed_media_type,
    resolve_logical_reference,
)

_LOGGER = logging.getLogger(__name__)

_VERIFIED_STREAM_CHUNK_SIZE = 1024 * 1024
_VERIFIED_STREAM_SPOOL_LIMIT = 8 * _VERIFIED_STREAM_CHUNK_SIZE

try:
    import trimesh
    from trimesh.resolvers import Resolver
except ImportError:
    raise ImportError("trimesh is required. Install with: pip install trimesh")


class ExternalMeshDependencyError(ValueError):
    """An authoritative primary mesh attempted to resolve another asset."""


class _RecordingDenyResolver(Resolver):
    """Prevent Trimesh from constructing an unbounded filesystem resolver."""

    def __init__(
        self,
        namespace: str = "",
        requests: list[str] | None = None,
    ) -> None:
        self._namespace = str(namespace)
        self._requests = requests if requests is not None else []

    @property
    def requests(self) -> tuple[str, ...]:
        return tuple(self._requests)

    def _record(self, key: object) -> None:
        text = str(key)
        if self._namespace:
            text = f"{self._namespace}/{text}"
        self._requests.append(text)

    def get(self, key: object) -> bytes:
        self._record(key)
        raise FileNotFoundError(
            "external mesh dependencies are denied for authoritative imports"
        )

    def write(self, name: str, data: object) -> None:
        del data
        self._record(name)
        raise PermissionError(
            "authoritative import resolver is read-only and denies dependencies"
        )

    def namespaced(self, namespace: str) -> "_RecordingDenyResolver":
        prefix = "/".join(
            part for part in (self._namespace, str(namespace)) if part
        )
        return _RecordingDenyResolver(prefix, self._requests)

    def keys(self) -> tuple[str, ...]:
        return ()

    def __contains__(self, key: object) -> bool:
        self._record(key)
        return False

    def validate_after_load(self) -> None:
        if self.requests:
            raise ExternalMeshDependencyError(
                "authoritative source requested external sidecar assets; "
                "dependency_policy=deny_external"
            )


@dataclass(slots=True)
class _CapturedResolverState:
    root: Path
    primary_logical_path: str
    payloads: dict[str, bytes] = field(default_factory=dict)
    resources: dict[str, ResolvedSourceResource] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)


def _read_resolved_resource(
    root: Path,
    logical_path: str,
) -> tuple[bytes, str, SourceFingerprint]:
    """Read one contained regular file and bind the returned bytes to its hash."""

    candidate = root.joinpath(*logical_path.split("/"))
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise SourceManifestError(
            f"dependency is missing or escapes the source root: {logical_path!r}"
        ) from exc
    if not resolved.is_file():
        raise SourceManifestError(
            f"dependency is not a regular file: {logical_path!r}"
        )
    with open_fingerprinted_file(resolved) as (stream, fingerprint):
        try:
            opened_target = resolved.resolve(strict=True)
            opened_target.relative_to(root)
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise SourceManifestError(
                f"dependency escaped the source root while opening: {logical_path!r}"
            ) from exc
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise SourceManifestError(
                f"dependency is not an opened regular file: {logical_path!r}"
            )
        payload = stream.read()
        if not isinstance(payload, bytes):
            payload = bytes(payload)
        if len(payload) != fingerprint.size_bytes:
            raise SourceManifestError(
                f"dependency size changed while reading: {logical_path!r}"
            )
    return payload, str(resolved), fingerprint


class _CapturingDirectoryResolver(Resolver):
    """Read only parser-requested files contained by the primary source root."""

    def __init__(
        self,
        root: Path,
        primary_logical_path: str,
        *,
        namespace: str = "",
        state: _CapturedResolverState | None = None,
    ) -> None:
        resolved_root = root.resolve(strict=True)
        if not resolved_root.is_dir():
            raise SourceManifestError("source resolver root must be a directory")
        self._state = state or _CapturedResolverState(
            root=resolved_root,
            primary_logical_path=canonical_logical_path(primary_logical_path),
        )
        self._namespace = namespace

    @property
    def resources(self) -> tuple[ResolvedSourceResource, ...]:
        return tuple(
            self._state.resources[path]
            for path in sorted(self._state.resources)
        )

    def _logical_path(self, key: object) -> str:
        return resolve_logical_reference(self._namespace, key)

    def get(self, key: object) -> bytes:
        try:
            logical_path = self._logical_path(key)
            if logical_path == self._state.primary_logical_path:
                raise SourceManifestError(
                    "a sidecar reference resolves to the primary mesh path"
                )
            cached = self._state.payloads.get(logical_path)
            if cached is not None:
                return cached
            if len(self._state.resources) >= MAX_SOURCE_MANIFEST_ENTRIES - 1:
                raise SourceManifestError(
                    "source dependency closure exceeds the portable entry budget"
                )
            payload, locator, fingerprint = _read_resolved_resource(
                self._state.root,
                logical_path,
            )
            entry = SourceManifestEntry(
                logical_path=logical_path,
                media_type=fixed_media_type(logical_path),
                role=DEPENDENCY_RESOURCE_ROLE,
                sha256=fingerprint.sha256,
                size_bytes=fingerprint.size_bytes,
            )
            self._state.payloads[logical_path] = payload
            self._state.resources[logical_path] = ResolvedSourceResource(
                entry=entry,
                locator=locator,
            )
            return payload
        except Exception as exc:
            self._state.failures.append(str(exc))
            raise FileNotFoundError(str(exc)) from exc

    def write(self, name: str, data: object) -> None:
        del data
        self._state.failures.append(f"parser attempted resolver write: {name!r}")
        raise PermissionError("authoritative source resolver is read-only")

    def namespaced(self, namespace: str) -> "_CapturingDirectoryResolver":
        try:
            namespace_text = str(namespace).strip()
            combined = (
                self._namespace
                if not namespace_text
                else resolve_logical_reference(self._namespace, namespace_text)
            )
        except Exception as exc:
            self._state.failures.append(str(exc))
            raise
        return _CapturingDirectoryResolver(
            self._state.root,
            self._state.primary_logical_path,
            namespace=combined,
            state=self._state,
        )

    def keys(self) -> tuple[str, ...]:
        prefix = f"{self._namespace}/" if self._namespace else ""
        return tuple(
            path.removeprefix(prefix)
            for path in sorted(self._state.payloads)
            if path.startswith(prefix)
        )

    def __contains__(self, key: object) -> bool:
        try:
            self.get(key)
        except FileNotFoundError:
            return False
        return True

    def validate_after_load(self) -> None:
        if self._state.failures:
            raise ExternalMeshDependencyError(
                "authoritative source requested an unsafe, missing, or unreadable "
                "sidecar under resolver_profile=relative-contained-v1"
            )


ResourcePayloadLoader = Callable[[SourceManifestEntry], tuple[bytes, str]]


@dataclass(slots=True)
class _ClosedResolverState:
    entries: dict[str, SourceManifestEntry]
    loader: ResourcePayloadLoader
    payloads: dict[str, bytes] = field(default_factory=dict)
    resources: dict[str, ResolvedSourceResource] = field(default_factory=dict)
    requested: set[str] = field(default_factory=set)
    failures: list[str] = field(default_factory=list)


class _ClosedManifestResolver(Resolver):
    """Replay only the dependency bytes declared by a strict v2 receipt."""

    def __init__(
        self,
        manifest: SourceManifest,
        loader: ResourcePayloadLoader,
        *,
        namespace: str = "",
        state: _ClosedResolverState | None = None,
    ) -> None:
        if not isinstance(manifest, SourceManifest):
            raise SourceManifestError("closed resolver needs a SourceManifest")
        self._manifest = manifest
        self._namespace = namespace
        self._state = state or _ClosedResolverState(
            entries={entry.logical_path: entry for entry in manifest.dependency_entries},
            loader=loader,
        )

    @property
    def resources(self) -> tuple[ResolvedSourceResource, ...]:
        return tuple(
            self._state.resources[path]
            for path in sorted(self._state.resources)
        )

    def get(self, key: object) -> bytes:
        try:
            logical_path = resolve_logical_reference(self._namespace, key)
            entry = self._state.entries.get(logical_path)
            if entry is None:
                raise SourceManifestError(
                    f"parser requested undeclared dependency: {logical_path!r}"
                )
            self._state.requested.add(logical_path)
            cached = self._state.payloads.get(logical_path)
            if cached is not None:
                return cached
            payload, locator = self._state.loader(entry)
            if not isinstance(payload, bytes):
                payload = bytes(payload)
            observed_digest = hashlib.sha256(payload).hexdigest()
            if len(payload) != entry.size_bytes or observed_digest != entry.sha256:
                raise SourceManifestError(
                    f"dependency bytes do not match the source manifest: {logical_path!r}"
                )
            self._state.payloads[logical_path] = payload
            self._state.resources[logical_path] = ResolvedSourceResource(
                entry=entry,
                locator=locator,
            )
            return payload
        except Exception as exc:
            self._state.failures.append(str(exc))
            raise FileNotFoundError(str(exc)) from exc

    def write(self, name: str, data: object) -> None:
        del data
        self._state.failures.append(f"parser attempted resolver write: {name!r}")
        raise PermissionError("authoritative source resolver is read-only")

    def namespaced(self, namespace: str) -> "_ClosedManifestResolver":
        try:
            namespace_text = str(namespace).strip()
            combined = (
                self._namespace
                if not namespace_text
                else resolve_logical_reference(self._namespace, namespace_text)
            )
        except Exception as exc:
            self._state.failures.append(str(exc))
            raise
        return _ClosedManifestResolver(
            self._manifest,
            self._state.loader,
            namespace=combined,
            state=self._state,
        )

    def keys(self) -> tuple[str, ...]:
        prefix = f"{self._namespace}/" if self._namespace else ""
        return tuple(
            path.removeprefix(prefix)
            for path in sorted(self._state.entries)
            if path.startswith(prefix)
        )

    def __contains__(self, key: object) -> bool:
        try:
            self.get(key)
        except FileNotFoundError:
            return False
        return True

    def validate_after_load(self) -> None:
        missing = sorted(set(self._state.entries) - self._state.requested)
        if self._state.failures or missing:
            raise ExternalMeshDependencyError(
                "parser dependency requests do not exactly match the closed source manifest"
            )


def _primary_resource_entry(
    fingerprint: SourceFingerprint,
    logical_path: str,
) -> SourceManifestEntry:
    return SourceManifestEntry(
        logical_path=logical_path,
        media_type=fixed_media_type(logical_path),
        role=PRIMARY_RESOURCE_ROLE,
        sha256=fingerprint.sha256,
        size_bytes=fingerprint.size_bytes,
    )


def _require_manifest_primary(
    manifest: SourceManifest,
    fingerprint: SourceFingerprint,
) -> None:
    primary = manifest.primary_entry
    if (
        primary.sha256 != fingerprint.sha256
        or primary.size_bytes != fingerprint.size_bytes
    ):
        raise ExternalMeshDependencyError(
            "primary source bytes do not match the closed source manifest"
        )


def _directory_payload_loader(root: Path) -> ResourcePayloadLoader:
    resolved_root = root.resolve(strict=True)

    def load(entry: SourceManifestEntry) -> tuple[bytes, str]:
        payload, locator, _fingerprint = _read_resolved_resource(
            resolved_root,
            entry.logical_path,
        )
        return payload, locator

    return load


def _load_authoritative_trimesh(
    source_stream: BinaryIO,
    *,
    source_format: str,
    resolver: Resolver | None = None,
) -> object:
    """Execute the closed parser profile with an explicit resolver boundary."""

    active_resolver = resolver or _RecordingDenyResolver()
    try:
        loaded = trimesh.load(
            source_stream,
            file_type=source_format,
            resolver=active_resolver,
            allow_remote=False,
            force="mesh",
            process=False,
            maintain_order=True,
        )
    except Exception as exc:
        try:
            active_resolver.validate_after_load()  # type: ignore[attr-defined]
        except ExternalMeshDependencyError as dependency_exc:
            raise dependency_exc from exc
        raise
    active_resolver.validate_after_load()  # type: ignore[attr-defined]
    return loaded


@dataclass
class MeshData:
    """
    3D 메쉬 데이터 컨테이너
    
    Attributes:
        vertices: (N, 3) 정점 좌표 배열
        faces: (M, 3) 면 인덱스 배열 (삼각형)
        normals: (N, 3) 정점 법선 벡터 (선택)
        face_normals: (M, 3) 면 법선 벡터 (선택)
        uv_coords: (N, 2) UV 좌표 (선택)
        texture: 텍스처 이미지 (선택)
        unit: 좌표 단위 ('mm', 'cm', 'm')
        filepath: 원본 파일 경로
    """
    vertices: np.ndarray
    faces: np.ndarray
    normals: Optional[np.ndarray] = None
    face_normals: Optional[np.ndarray] = None
    uv_coords: Optional[np.ndarray] = None
    texture: Optional[np.ndarray] = None
    unit: str = 'mm'
    filepath: Optional[Path] = None
    # Identity of the raw primary file before unit scaling, centering, or any
    # other geometry mutation.
    source_identity: Optional[SourceFingerprint] = None
    # Parser recipe is separate from filename/extension hints so a byte-identical
    # relocated file can be renamed without changing how its bytes are decoded.
    source_format: Optional[str] = None
    # Exact parser/runtime receipt used to create authoritative source geometry.
    # This is runtime provenance; the durable copy lives in GeometryRevision.
    source_import_recipe: Optional[Mapping[str, object]] = None
    # Runtime locators for the primary and every manifest-bound dependency.
    # Locators are never serialized into the ArtifactDocument.
    source_resources: tuple[ResolvedSourceResource, ...] = ()
    
    # Computed properties cache
    _bounds: Optional[np.ndarray] = field(default=None, repr=False)
    _centroid: Optional[np.ndarray] = field(default=None, repr=False)
    _surface_area: Optional[float] = field(default=None, repr=False)
    _normals_chunk_faces: Optional[int] = field(default=None, repr=False)

    # Optional runtime tuning knobs used by surface separation / assist.
    _assist_unresolved_keep_max: Optional[int] = field(default=None, repr=False)
    _views_use_topology_assignment: bool = field(default=True, repr=False)
    _views_fallback_use_normals: bool = field(default=False, repr=False)
    _views_migu_absdot_max: Optional[float] = field(default=None, repr=False)
    _views_migu_max_frac: Optional[float] = field(default=None, repr=False)
    _views_visibility_neighborhood: Optional[int] = field(default=None, repr=False)
    
    def __post_init__(self):
        """데이터 검증 및 타입 변환"""
        # vertices: (N, 3) 보장
        vertices = np.asarray(self.vertices, dtype=np.float64)
        if vertices.ndim == 1:
            if vertices.size % 3 == 0:
                vertices = vertices.reshape(-1, 3)
            else:
                vertices = vertices.reshape(0, 3)
        elif vertices.ndim == 2:
            if vertices.shape[1] == 2:
                vertices = np.hstack([vertices, np.zeros((vertices.shape[0], 1), dtype=np.float64)])
            elif vertices.shape[1] >= 3:
                vertices = vertices[:, :3]
            else:
                vertices = vertices.reshape(0, 3)
        else:
            vertices = vertices.reshape(0, 3)
        self.vertices = vertices

        # faces: (M, 3) 보장
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

        # 인덱스가 깨진 경우(파일 손상/비정상 로드) 크래시 방지용 필터링
        try:
            if self.vertices.shape[0] > 0 and faces.shape[0] > 0:
                valid = (faces >= 0) & (faces < int(self.vertices.shape[0]))
                keep = np.all(valid, axis=1)
                faces = faces[keep]
        except Exception:
            log_once(
                _LOGGER,
                "mesh_loader:post_init_face_filter",
                logging.WARNING,
                "MeshData.__post_init__ face index filtering failed",
                exc_info=True,
            )
        # 퇴화 면(무너진 삼각형/중복 정점 면) 제거.
        # 이 값들은 후속 분리/정규화 단계에서 수치 불안정을 키울 수 있어 명시적으로 제외합니다.
        try:
            if self.vertices.shape[0] > 0 and faces.shape[0] > 0:
                f = np.asarray(faces, dtype=np.int32, copy=False)
                v0 = self.vertices[f[:, 0]]
                v1 = self.vertices[f[:, 1]]
                v2 = self.vertices[f[:, 2]]
                duplicate_vert = (f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 2] == f[:, 0])
                area2 = np.cross(v1 - v0, v2 - v0)
                area2 = np.sum(area2 * area2, axis=1)
                degenerate = duplicate_vert | (area2 <= 0.0) | ~np.isfinite(area2)
                if np.any(degenerate):
                    keep = ~degenerate
                    removed = int(faces.shape[0]) - int(np.count_nonzero(keep))
                    if removed > 0:
                        faces = f[keep]
                        log_once(
                            _LOGGER,
                            "mesh_loader:post_init_prune_degenerate",
                            logging.WARNING,
                            "MeshData.__post_init__ removed %d degenerate faces",
                            removed,
                        )
        except Exception:
            log_once(
                _LOGGER,
                "mesh_loader:post_init_face_degenerate_filter",
                logging.DEBUG,
                "MeshData.__post_init__ degenerate-face filtering skipped",
                exc_info=True,
            )
        self.faces = faces
        
        if self.normals is not None:
            self.normals = np.asarray(self.normals, dtype=np.float64)
        if self.face_normals is not None:
            self.face_normals = np.asarray(self.face_normals, dtype=np.float32)
        if self.uv_coords is not None:
            self.uv_coords = np.asarray(self.uv_coords, dtype=np.float64)
        if self.source_import_recipe is not None:
            if not isinstance(self.source_import_recipe, Mapping):
                raise TypeError("source_import_recipe must be a mapping or None")
            self.source_import_recipe = dict(self.source_import_recipe)
        try:
            resources = tuple(self.source_resources)
        except TypeError as exc:
            raise TypeError("source_resources must be iterable") from exc
        if not all(isinstance(item, ResolvedSourceResource) for item in resources):
            raise TypeError(
                "source_resources must contain only ResolvedSourceResource values"
            )
        resource_keys = [
            (item.entry.logical_path, item.entry.sha256) for item in resources
        ]
        if len(resource_keys) != len(set(resource_keys)):
            raise ValueError("source_resources must not contain duplicate entries")
        self.source_resources = tuple(
            sorted(
                resources,
                key=lambda item: (item.entry.logical_path, item.entry.sha256),
            )
        )
    
    @property
    def n_vertices(self) -> int:
        """정점 개수"""
        return len(self.vertices)
    
    @property
    def n_faces(self) -> int:
        """면 개수"""
        return len(self.faces)
    
    @property
    def bounds(self) -> np.ndarray:
        """경계 박스 [[min_x, min_y, min_z], [max_x, max_y, max_z]]"""
        if self._bounds is None:
            if self.vertices.ndim != 2 or self.vertices.size == 0:
                self._bounds = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64)
                return self._bounds
            self._bounds = np.array([
                self.vertices.min(axis=0),
                self.vertices.max(axis=0)
            ])
        return self._bounds
    
    @property
    def extents(self) -> np.ndarray:
        """경계 박스 크기 [width, height, depth]"""
        return self.bounds[1] - self.bounds[0]
    
    @property
    def centroid(self) -> np.ndarray:
        """메쉬 중심점"""
        if self._centroid is None:
            self._centroid = self.vertices.mean(axis=0)
        assert self._centroid is not None
        return self._centroid
    
    @property
    def surface_area(self) -> float:
        """총 표면적 계산 (대형 메쉬 안전 처리)"""
        if self._surface_area is None:
            try:
                faces = np.asarray(self.faces)
                if (
                    self.vertices.ndim != 2
                    or self.vertices.size == 0
                    or faces.ndim != 2
                    or faces.shape[1] < 3
                    or faces.size == 0
                ):
                    self._surface_area = 0.0
                    return self._surface_area

                # 면이 너무 많으면 (100만 이상) 추정값 사용
                if len(self.faces) > 1000000:
                    # 샘플링으로 추정 (10만 면만 계산)
                    sample_size = 100000
                    indices = np.random.choice(len(self.faces), sample_size, replace=False)
                    sample_faces = self.faces[indices]
                    
                    v0 = self.vertices[sample_faces[:, 0]]
                    v1 = self.vertices[sample_faces[:, 1]]
                    v2 = self.vertices[sample_faces[:, 2]]
                    
                    cross = np.cross(v1 - v0, v2 - v0)
                    sample_area = np.linalg.norm(cross, axis=1).sum() / 2.0
                    # 비율로 전체 추정
                    self._surface_area = float(sample_area * len(self.faces) / sample_size)
                else:
                    # 정상 계산
                    v0 = self.vertices[self.faces[:, 0]]
                    v1 = self.vertices[self.faces[:, 1]]
                    v2 = self.vertices[self.faces[:, 2]]
                    
                    cross = np.cross(v1 - v0, v2 - v0)
                    areas = np.linalg.norm(cross, axis=1) / 2.0
                    self._surface_area = float(areas.sum())
            except MemoryError:
                # 메모리 부족 시 추정값 반환
                self._surface_area = -1.0  # 계산 불가 표시
        return self._surface_area
    
    @property
    def has_texture(self) -> bool:
        """텍스처 존재 여부"""
        return self.texture is not None and self.uv_coords is not None
    
    def compute_normals(self, *, compute_vertex_normals: bool = True, force: bool = False) -> None:
        """법선 벡터 계산 (없는 경우)"""
        if force:
            self.face_normals = None
            self.normals = None

        faces = np.asarray(self.faces)
        if (
            self.vertices.ndim != 2
            or faces.ndim != 2
            or faces.shape[1] < 3
            or self.vertices.size == 0
            or faces.size == 0
        ):
            if self.face_normals is None:
                self.face_normals = np.zeros((0, 3), dtype=np.float32)
            if compute_vertex_normals and self.normals is None:
                self.normals = np.zeros_like(self.vertices, dtype=np.float64)
            return

        if self.face_normals is None:
            # For huge meshes, compute in chunks to avoid large temporaries / OOM.
            try:
                n_faces = int(faces.shape[0])
            except Exception:
                n_faces = int(len(faces))

            try:
                chunk = int(getattr(self, "_normals_chunk_faces", 250000) or 250000)
            except Exception:
                chunk = 250000
            chunk = max(10000, min(chunk, 500000))

            if n_faces > chunk:
                out = np.empty((n_faces, 3), dtype=np.float32)
                for start in range(0, n_faces, chunk):
                    end = min(n_faces, start + chunk)
                    f = np.asarray(faces[start:end, :3], dtype=np.int32)
                    if f.size == 0:
                        continue
                    v0 = self.vertices[f[:, 0]]
                    v1 = self.vertices[f[:, 1]]
                    v2 = self.vertices[f[:, 2]]

                    cross = np.cross(v1 - v0, v2 - v0)
                    norms = np.linalg.norm(cross, axis=1, keepdims=True)
                    out_chunk = np.zeros((f.shape[0], 3), dtype=np.float32)
                    valid = np.isfinite(norms[:, 0]) & (norms[:, 0] > 0.0)
                    if np.any(valid):
                        out_chunk[valid] = (cross[valid] / norms[valid]).astype(np.float32, copy=False)
                    out[start:end] = out_chunk
                self.face_normals = out

        if self.face_normals is None:
            v0 = self.vertices[self.faces[:, 0]]
            v1 = self.vertices[self.faces[:, 1]]
            v2 = self.vertices[self.faces[:, 2]]
            
            cross = np.cross(v1 - v0, v2 - v0)
            norms = np.linalg.norm(cross, axis=1, keepdims=True)
            self.face_normals = np.zeros((cross.shape[0], 3), dtype=np.float32)
            valid = np.isfinite(norms[:, 0]) & (norms[:, 0] > 0.0)
            if np.any(valid):
                self.face_normals[valid] = (cross[valid] / norms[valid]).astype(np.float32, copy=False)
        
        if compute_vertex_normals and self.normals is None:
            # 정점 법선 = 인접 면 법선의 평균
            self.normals = np.zeros_like(self.vertices, dtype=np.float64)
            faces = self.faces
            face_normals = np.asarray(self.face_normals, dtype=np.float64)
            # Python loop 대신 벡터화된 누적 (대용량 메쉬 로딩 속도 개선)
            np.add.at(self.normals, faces[:, 0], face_normals)
            np.add.at(self.normals, faces[:, 1], face_normals)
            np.add.at(self.normals, faces[:, 2], face_normals)
            
            norms = np.linalg.norm(self.normals, axis=1, keepdims=True)
            valid = np.isfinite(norms[:, 0]) & (norms[:, 0] > 0.0)
            normals = self.normals
            if np.any(valid):
                normals[valid] = normals[valid] / norms[valid]
            if np.any(~valid):
                normals[~valid] = 0.0
            self.normals = normals
    
    def get_edges(self) -> np.ndarray:
        """모든 엣지 목록 반환 (N, 2)"""
        edges = set()
        for face in self.faces:
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
                edges.add(edge)
        return np.array(list(edges), dtype=np.int32)

    def get_boundary_edges(self) -> np.ndarray:
        """
        경계 엣지 목록 반환 (K, 2)

        열린 메쉬(open surface)에서 한 면에만 속하는 엣지를 경계로 간주합니다.
        """
        edge_count: dict[tuple[int, int], int] = {}
        for face in self.faces:
            for i in range(3):
                a = int(face[i])
                b = int(face[(i + 1) % 3])
                edge = (a, b) if a < b else (b, a)
                edge_count[edge] = edge_count.get(edge, 0) + 1

        boundary_edges = [e for e, c in edge_count.items() if c == 1]
        if not boundary_edges:
            return np.zeros((0, 2), dtype=np.int32)
        return np.asarray(boundary_edges, dtype=np.int32)

    def get_boundary_loops(self) -> List[np.ndarray]:
        """
        경계 루프(들)을 정렬된 정점 인덱스 배열로 반환합니다.

        Returns:
            List[np.ndarray]: 각 루프는 (L,) 형태의 정점 인덱스 배열 (반복된 시작점 없음)
        """
        boundary_edges = self.get_boundary_edges()
        if len(boundary_edges) == 0:
            return []

        # 인접 리스트 구성 (경계 그래프)
        adjacency: dict[int, list[int]] = {}
        unused_edges: set[tuple[int, int]] = set()
        for a, b in boundary_edges:
            a_i = int(a)
            b_i = int(b)
            adjacency.setdefault(a_i, []).append(b_i)
            adjacency.setdefault(b_i, []).append(a_i)
            unused_edges.add((a_i, b_i) if a_i < b_i else (b_i, a_i))

        loops: list[np.ndarray] = []

        while unused_edges:
            # 시작 엣지 하나 선택
            start_edge = next(iter(unused_edges))
            a0, b0 = start_edge
            unused_edges.remove(start_edge)

            loop = [a0, b0]
            prev = a0
            curr = b0

            # 경계 루프 추적
            while True:
                neighbors = adjacency.get(curr, [])
                if not neighbors:
                    break

                # prev가 아닌 이웃 중 아직 사용되지 않은 엣지를 우선 선택
                next_v = None
                for cand in neighbors:
                    if cand == prev:
                        continue
                    e = (curr, cand) if curr < cand else (cand, curr)
                    if e in unused_edges:
                        next_v = cand
                        unused_edges.remove(e)
                        break

                if next_v is None:
                    # 더 이상 진행 불가 (비정상/분기 경계 등)
                    break

                if next_v == loop[0]:
                    # 루프 닫힘
                    break

                loop.append(next_v)
                prev, curr = curr, next_v

                # 안전장치: 무한루프 방지
                if len(loop) > len(boundary_edges) + 1:
                    break

            # 너무 짧은 체인은 제외 (삼각형 이상)
            if len(loop) >= 3:
                loops.append(np.asarray(loop, dtype=np.int32))

        return loops
    
    def get_boundary_vertices(self) -> np.ndarray:
        """
        경계 정점 인덱스 반환 (열린 메쉬의 경우)

        - 가능하면 가장 큰 경계 루프를 정렬된 순서로 반환합니다.
        - 루프 추적이 불가한 경우(비정상 메쉬 등)에는 유니크 정점 집합을 반환합니다.
        """
        loops = self.get_boundary_loops()
        if loops:
            # 가장 긴 루프 선택 (정점 수 기준)
            main = max(loops, key=lambda a: int(a.size))
            return main.copy()

        boundary_edges = self.get_boundary_edges()
        if len(boundary_edges) == 0:
            return np.zeros((0,), dtype=np.int32)

        boundary_verts = np.unique(boundary_edges.reshape(-1))
        return boundary_verts.astype(np.int32)
    
    def center_at_origin(self) -> 'MeshData':
        """중심을 원점으로 이동한 새 메쉬 반환"""
        centered_vertices = self.vertices - self.centroid
        return MeshData(
            vertices=centered_vertices,
            faces=self.faces.copy(),
            normals=self.normals.copy() if self.normals is not None else None,
            face_normals=self.face_normals.copy() if self.face_normals is not None else None,
            uv_coords=self.uv_coords.copy() if self.uv_coords is not None else None,
            texture=self.texture.copy() if self.texture is not None else None,
            unit=self.unit,
            filepath=self.filepath,
            source_identity=self.source_identity,
            source_format=self.source_format,
            source_import_recipe=self.source_import_recipe,
            source_resources=self.source_resources,
        )
    
    def to_trimesh(self) -> 'trimesh.Trimesh':
        """trimesh 객체로 변환"""
        mesh = trimesh.Trimesh(
            vertices=self.vertices,
            faces=self.faces,
            vertex_normals=self.normals,
            process=False
        )
        try:
            meta = getattr(mesh, "metadata", None)
            if not isinstance(meta, dict):
                meta = {}
                mesh.metadata = meta
            meta["unit"] = str(self.unit)
            if self.filepath is not None:
                meta["filepath"] = str(self.filepath)
        except Exception:
            log_once(
                _LOGGER,
                "mesh_loader:to_trimesh_metadata",
                logging.DEBUG,
                "Failed to attach metadata to trimesh mesh",
                exc_info=True,
            )
        return mesh
    
    @classmethod
    def from_trimesh(cls, mesh: 'trimesh.Trimesh',
                     filepath: Optional[Path] = None,
                     unit: str = 'mm',
                     source_identity: Optional[SourceFingerprint] = None,
                     source_format: Optional[str] = None,
                     source_import_recipe: Optional[Mapping[str, object]] = None,
                     source_resources: tuple[ResolvedSourceResource, ...] = ()) -> 'MeshData':
        """trimesh 객체에서 생성"""
        # 텍스처 추출 시도
        texture = None
        uv_coords = None

        visual = getattr(mesh, "visual", None)
        uv = getattr(visual, "uv", None) if visual is not None else None
        if uv is not None:
            uv_coords = uv

        material = getattr(visual, "material", None) if visual is not None else None
        image = getattr(material, "image", None) if material is not None else None
        if image is not None:
            texture = np.array(image)
        
        return cls(
            vertices=mesh.vertices,
            faces=mesh.faces,
            # NOTE: huge mesh(특히 STL)에서 vertex_normals 계산이 로딩 시간을 크게 증가시킵니다.
            #       기본은 skip하고, 필요할 때 compute_normals()로 생성합니다.
            normals=None,
            face_normals=None,
            uv_coords=uv_coords,
            texture=texture,
            unit=unit,
            filepath=filepath,
            source_identity=source_identity,
            source_format=source_format,
            source_import_recipe=source_import_recipe,
            source_resources=source_resources,
        )
    
    def extract_submesh(self, face_indices: np.ndarray) -> 'MeshData':
        """선택된 면으로 서브메쉬 추출"""
        face_indices = np.asarray(face_indices, dtype=np.int32).reshape(-1)

        faces = np.asarray(self.faces, dtype=np.int32)
        vertices = np.asarray(self.vertices, dtype=np.float64)
        n_faces = int(faces.shape[0]) if faces.ndim == 2 else 0

        if face_indices.size == 0 or n_faces == 0:
            return MeshData(
                vertices=np.zeros((0, 3), dtype=np.float64),
                faces=np.zeros((0, 3), dtype=np.int32),
                normals=None,
                face_normals=None,
                uv_coords=None,
                texture=self.texture,
                unit=self.unit,
                filepath=self.filepath,
                source_identity=self.source_identity,
                source_format=self.source_format,
                source_import_recipe=self.source_import_recipe,
                source_resources=self.source_resources,
            )

        # 크래시 방지: 인덱스 범위 밖 제거
        valid = (face_indices >= 0) & (face_indices < n_faces)
        if not bool(np.all(valid)):
            face_indices = face_indices[valid]
        if face_indices.size == 0:
            return MeshData(
                vertices=np.zeros((0, 3), dtype=np.float64),
                faces=np.zeros((0, 3), dtype=np.int32),
                normals=None,
                face_normals=None,
                uv_coords=None,
                texture=self.texture,
                unit=self.unit,
                filepath=self.filepath,
                source_identity=self.source_identity,
                source_format=self.source_format,
                source_import_recipe=self.source_import_recipe,
                source_resources=self.source_resources,
            )

        selected_faces = faces[face_indices]

        # 사용된 정점 인덱스 (sorted)
        unique_verts = np.unique(selected_faces.reshape(-1)).astype(np.int32, copy=False)

        # 새 정점/속성
        new_vertices = vertices[unique_verts]
        new_normals = None
        if self.normals is not None:
            try:
                normals = np.asarray(self.normals, dtype=np.float64)
                if normals.shape[0] == vertices.shape[0]:
                    new_normals = normals[unique_verts]
            except Exception:
                new_normals = None

        new_face_normals = None
        if self.face_normals is not None:
            try:
                fn = np.asarray(self.face_normals, dtype=np.float32)
                if fn.shape[0] == faces.shape[0]:
                    new_face_normals = fn[face_indices]
            except Exception:
                new_face_normals = None

        new_uv = None
        if self.uv_coords is not None:
            try:
                uv = np.asarray(self.uv_coords, dtype=np.float64)
                if uv.shape[0] == vertices.shape[0]:
                    new_uv = uv[unique_verts]
            except Exception:
                new_uv = None

        # 새 인덱스 매핑 (vectorized)
        # unique_verts는 sorted이므로 searchsorted로 빠르게 remap 가능
        new_faces = np.searchsorted(unique_verts, selected_faces).astype(np.int32, copy=False)

        return MeshData(
            vertices=new_vertices,
            faces=new_faces,
            normals=new_normals,
            face_normals=new_face_normals,
            uv_coords=new_uv,
            texture=self.texture,  # 텍스처는 공유
            unit=self.unit,
            filepath=self.filepath,
            source_identity=self.source_identity,
            source_format=self.source_format,
            source_import_recipe=self.source_import_recipe,
            source_resources=self.source_resources,
        )


class MeshLoader:
    """
    다양한 3D 포맷의 메쉬 파일 로더
    
    Supported formats:
        - OBJ (Wavefront)
        - PLY (Polygon File Format)
        - STL (Stereolithography)
        - OFF (Object File Format)
        - GLTF/GLB (GL Transmission Format)
    """
    
    SUPPORTED_FORMATS: dict[str, str] = {
        '.obj': 'Wavefront OBJ',
        '.ply': 'Polygon File Format',
        '.stl': 'Stereolithography',
        '.off': 'Object File Format',
        '.gltf': 'GL Transmission Format',
        '.glb': 'GL Transmission Format (Binary)',
    }
    
    def __init__(self, default_unit: str = 'mm'):
        """
        Args:
            default_unit: 기본 좌표 단위 ('mm', 'cm', 'm')
        """
        self.default_unit = default_unit
    
    @classmethod
    def get_supported_formats(cls) -> dict[str, str]:
        """지원 포맷 목록 반환"""
        return cls.SUPPORTED_FORMATS.copy()
    
    @classmethod
    def get_file_filter(cls) -> str:
        """파일 다이얼로그용 필터 문자열 생성"""
        all_exts = ' '.join(f'*{ext}' for ext in cls.SUPPORTED_FORMATS.keys())
        filters = [f"All 3D Formats ({all_exts})"]
        
        for ext, name in cls.SUPPORTED_FORMATS.items():
            filters.append(f"{name} (*{ext})")
        
        return ';;'.join(filters)
    
    def load(
        self,
        filepath: Union[str, Path],
        unit: Optional[str] = None,
        *,
        source_format: Optional[str] = None,
        import_recipe: Mapping[str, object] | None = None,
        capture_dependencies: bool | None = None,
    ) -> MeshData:
        """
        메쉬 파일 로드
        
        Args:
            filepath: 메쉬 파일 경로
            unit: 좌표 단위 (None이면 default_unit 사용)
            source_format: relocated project source의 원래 parser 형식 hint.
                파일 이름은 identity가 아니므로 suffix가 바뀌어도 저장된
                primary format으로 동일 바이트를 파싱할 수 있습니다.
            import_recipe: 저장된 closed parser recipe. 생략하면 현재 runtime의
                strict recipe를 생성해 신규 import에 사용합니다.
            
        Returns:
            MeshData: 로드된 메쉬 데이터
            
        Raises:
            FileNotFoundError: 파일이 존재하지 않음
            ValueError: 지원하지 않는 포맷
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        path_ext = filepath.suffix.lower()
        format_hint = str(source_format or "").strip().lower().removeprefix('.')
        parse_ext = f".{format_hint}" if format_hint else path_ext
        if parse_ext not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {parse_ext or path_ext}\n"
                f"Supported formats: {list(self.SUPPORTED_FORMATS.keys())}"
            )

        unit = unit or self.default_unit
        file_type = parse_ext.removeprefix('.')
        recipe = (
            current_mesh_import_recipe(file_type)
            if import_recipe is None
            else import_recipe
        )
        execution = validate_mesh_import_recipe(recipe, allow_legacy=True)
        if execution.source_format != file_type:
            raise ValueError(
                "import recipe format does not match requested parser format: "
                f"{execution.source_format!r} != {file_type!r}"
            )
        capture = import_recipe is None if capture_dependencies is None else bool(
            capture_dependencies
        )
        if capture and execution.source_manifest is not None:
            raise ValueError(
                "dependency capture cannot replace an existing closed source manifest"
            )

        if execution.source_manifest is not None:
            primary_logical_path = execution.source_manifest.primary_logical_path
        else:
            primary_logical_path = canonical_logical_path(
                filepath.name,
                field_name="primary source logical path",
            )
        resolver: Resolver
        resolved_primary_path = filepath.resolve(strict=True)
        resolver_root = resolved_primary_path.parent
        if capture:
            resolver = _CapturingDirectoryResolver(
                resolver_root,
                primary_logical_path,
            )
        elif execution.source_manifest is not None:
            resolver = _ClosedManifestResolver(
                execution.source_manifest,
                _directory_payload_loader(resolver_root),
            )
        else:
            resolver = _RecordingDenyResolver()

        # Hash and parse the exact same open descriptor. Reopening by path here
        # would allow a same-size/same-mtime replacement to pair one file's
        # hash with another file's geometry. This work runs in MeshLoadThread
        # for the GUI, so large-file hashing never blocks the UI thread.
        with open_fingerprinted_file(filepath) as (source_stream, source_identity):
            # These keyword values are the executable strict/legacy profile
            # validated above. There is no compatibility fallback which could
            # silently execute different parser flags.
            mesh = _load_authoritative_trimesh(
                source_stream,
                source_format=file_type,
                resolver=resolver,
            )

        primary_entry = _primary_resource_entry(
            source_identity,
            primary_logical_path,
        )
        primary_resource = ResolvedSourceResource(
            entry=primary_entry,
            locator=str(resolved_primary_path),
        )
        if isinstance(resolver, _CapturingDirectoryResolver):
            manifest = SourceManifest(
                primary_logical_path=primary_logical_path,
                entries=(primary_entry, *(item.entry for item in resolver.resources)),
            )
            receipt = mesh_import_recipe_with_manifest(recipe, manifest)
            source_resources = (primary_resource, *resolver.resources)
        elif isinstance(resolver, _ClosedManifestResolver):
            assert execution.source_manifest is not None
            _require_manifest_primary(execution.source_manifest, source_identity)
            receipt = dict(recipe)
            source_resources = (primary_resource, *resolver.resources)
        else:
            receipt = dict(recipe)
            source_resources = (primary_resource,)
        
        # Scene인 경우 단일 메쉬로 병합
        if isinstance(mesh, trimesh.Scene):
            meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if len(meshes) == 0:
                raise ValueError(f"No valid mesh found in: {filepath}")
            mesh = trimesh.util.concatenate(meshes)

        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Expected trimesh.Trimesh, got {type(mesh).__name__}")

        # MeshData로 변환
        mesh_data = MeshData.from_trimesh(
            mesh,
            filepath=filepath,
            unit=unit,
            source_identity=source_identity,
            source_format=file_type,
            source_import_recipe=receipt,
            source_resources=source_resources,
        )
        
        # 법선 계산 (없는 경우)
        # 로딩 시점에는 face normals만 계산 (vertex normals는 필요 시점에 계산)
        mesh_data.compute_normals(compute_vertex_normals=False)
        
        return mesh_data

    def load_verified_stream(
        self,
        source_stream: BinaryIO,
        *,
        unit: str,
        source_format: str,
        expected_sha256: str,
        expected_size_bytes: int,
        original_name: str,
        import_recipe: Mapping[str, object] | None = None,
        dependency_loader: ResourcePayloadLoader | None = None,
        primary_locator: str | None = None,
    ) -> MeshData:
        """Verify and load one primary mesh stream without trusting a path.

        The incoming bytes are copied incrementally into a bounded in-memory
        spool which rolls over to a temporary file for larger sources.  The
        digest and length are checked before the parser sees any bytes, then
        the exact verified spool descriptor is rewound and passed to trimesh.
        The primary stream is verified before parsing. Strict v1 receipts deny
        every sidecar; strict v2 receipts can resolve only manifest-declared
        bytes supplied by ``dependency_loader``.
        """
        format_hint = str(source_format or "").strip().lower().removeprefix(".")
        parse_ext = f".{format_hint}" if format_hint else ""
        if parse_ext not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {parse_ext}\n"
                f"Supported formats: {list(self.SUPPORTED_FORMATS.keys())}"
            )

        display_name = str(original_name)
        expected_identity = SourceFingerprint(
            sha256=expected_sha256,
            size_bytes=expected_size_bytes,
            mtime_ns=0,
            original_name=display_name,
            format=format_hint,
        )
        recipe = (
            current_mesh_import_recipe(format_hint)
            if import_recipe is None
            else import_recipe
        )
        execution = validate_mesh_import_recipe(recipe, allow_legacy=True)
        if execution.source_format != format_hint:
            raise ValueError(
                "import recipe format does not match source_format: "
                f"{execution.source_format!r} != {format_hint!r}"
            )
        resolver: Resolver
        if execution.source_manifest is not None:
            if dependency_loader is None:
                raise ValueError(
                    "closed source manifest requires a dependency payload loader"
                )
            resolver = _ClosedManifestResolver(
                execution.source_manifest,
                dependency_loader,
            )
        else:
            resolver = _RecordingDenyResolver()

        digest = hashlib.sha256()
        observed_size = 0
        with tempfile.SpooledTemporaryFile(
            max_size=_VERIFIED_STREAM_SPOOL_LIMIT,
            mode="w+b",
        ) as verified_stream:
            while True:
                chunk = source_stream.read(_VERIFIED_STREAM_CHUNK_SIZE)
                if not isinstance(chunk, (bytes, bytearray, memoryview)):
                    raise TypeError("source_stream.read() must return bytes")
                if not chunk:
                    break
                if len(chunk) > _VERIFIED_STREAM_CHUNK_SIZE:
                    raise ValueError("source_stream returned a chunk larger than 1 MiB")
                next_size = observed_size + len(chunk)
                if next_size > expected_identity.size_bytes:
                    raise ValueError(
                        "Source size mismatch before mesh parsing: "
                        f"expected {expected_identity.size_bytes}, observed at least {next_size}"
                    )
                digest.update(chunk)
                written = verified_stream.write(chunk)
                if written != len(chunk):
                    raise OSError("failed to spool the complete source stream")
                observed_size = next_size

            observed_sha256 = digest.hexdigest()
            if observed_size != expected_identity.size_bytes:
                raise ValueError(
                    "Source size mismatch before mesh parsing: "
                    f"expected {expected_identity.size_bytes}, observed {observed_size}"
                )
            if observed_sha256 != expected_identity.sha256:
                raise ValueError(
                    "Source SHA-256 mismatch before mesh parsing: "
                    f"expected {expected_identity.sha256}, observed {observed_sha256}"
                )

            source_identity = SourceFingerprint(
                sha256=observed_sha256,
                size_bytes=observed_size,
                mtime_ns=0,
                original_name=display_name,
                format=format_hint,
            )
            if execution.source_manifest is not None:
                _require_manifest_primary(execution.source_manifest, source_identity)
            verified_stream.seek(0)
            mesh = _load_authoritative_trimesh(
                cast(BinaryIO, verified_stream),
                source_format=format_hint,
                resolver=resolver,
            )

        display_path = Path(display_name)
        if isinstance(mesh, trimesh.Scene):
            meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if len(meshes) == 0:
                raise ValueError(f"No valid mesh found in: {display_path}")
            mesh = trimesh.util.concatenate(meshes)

        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Expected trimesh.Trimesh, got {type(mesh).__name__}")

        primary_logical_path = (
            execution.source_manifest.primary_logical_path
            if execution.source_manifest is not None
            else canonical_logical_path(
                display_path.name,
                field_name="primary source logical path",
            )
        )
        primary_resource = ResolvedSourceResource(
            entry=_primary_resource_entry(source_identity, primary_logical_path),
            locator=str(primary_locator or display_name),
        )
        dependency_resources = (
            resolver.resources
            if isinstance(resolver, _ClosedManifestResolver)
            else ()
        )
        mesh_data = MeshData.from_trimesh(
            mesh,
            filepath=display_path,
            unit=unit,
            source_identity=source_identity,
            source_format=format_hint,
            source_import_recipe=dict(recipe),
            source_resources=(primary_resource, *dependency_resources),
        )
        mesh_data.compute_normals(compute_vertex_normals=False)
        return mesh_data
    
    def load_multiple(self, filepaths: List[Union[str, Path]], 
                      unit: Optional[str] = None) -> List[MeshData]:
        """
        여러 메쉬 파일 로드
        
        Args:
            filepaths: 파일 경로 목록
            unit: 좌표 단위
            
        Returns:
            List[MeshData]: 로드된 메쉬 목록
        """
        return [self.load(fp, unit) for fp in filepaths]
    
    def get_file_info(self, filepath: Union[str, Path]) -> dict[str, object]:
        """
        파일 정보 미리보기 (전체 로드 없이)
        
        Args:
            filepath: 메쉬 파일 경로
            
        Returns:
            dict: 파일 정보 딕셔너리
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        ext = filepath.suffix.lower()
        file_size = filepath.stat().st_size
        
        info: dict[str, object] = {
            'filename': filepath.name,
            'format': self.SUPPORTED_FORMATS.get(ext, 'Unknown'),
            'extension': ext,
            'file_size_mb': round(file_size / (1024 * 1024), 2),
        }
        
        # Geometry metadata must use the same authoritative parser gate as an
        # actual Open.  Calling ``trimesh.load(path)`` here would auto-create a
        # filesystem resolver and could read untracked MTL/image/buffer files.
        try:
            mesh = self.load(filepath)
            info['n_vertices'] = mesh.n_vertices
            info['n_faces'] = mesh.n_faces
            info['has_texture'] = mesh.has_texture
            
        except Exception as e:
            info['error'] = str(e)
        
        return info


class MeshProcessor:
    """메쉬 처리 및 저장 유틸리티"""
    
    def save_mesh(self, mesh_data: Union[MeshData, 'trimesh.Trimesh'], filepath: str):
        """
        메쉬를 파일로 저장
        
        Args:
            mesh_data: MeshData 또는 trimesh.Trimesh 객체
            filepath: 저장할 파일 경로
        """
        filepath = str(filepath)
        
        if isinstance(mesh_data, MeshData):
            # trimesh 객체로 변환
            mesh = mesh_data.to_trimesh()
        else:
            mesh = mesh_data
            
        # trimesh export 기능 사용
        mesh.export(filepath)


# 간단한 테스트용
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        loader = MeshLoader()
        filepath = sys.argv[1]
        
        print(f"Loading: {filepath}")
        print("-" * 40)
        
        # 파일 정보
        info = loader.get_file_info(filepath)
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        print("-" * 40)
        
        # 전체 로드
        mesh = loader.load(filepath)
        print(f"  Vertices: {mesh.n_vertices:,}")
        print(f"  Faces: {mesh.n_faces:,}")
        print(f"  Bounds: {mesh.bounds}")
        print(f"  Extents: {mesh.extents}")
        print(f"  Surface Area: {mesh.surface_area:,.2f} {mesh.unit}²")
        print(f"  Has Texture: {mesh.has_texture}")
    else:
        print("Usage: python mesh_loader.py <mesh_file>")
        print(f"Supported formats: {list(MeshLoader.SUPPORTED_FORMATS.keys())}")
