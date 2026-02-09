"""
ARAP (As-Rigid-As-Possible) Mesh Flattening Module
메쉬 평면화 알고리즘 - 왜곡을 최소화하며 3D 표면을 2D로 펼침

Based on: "As-Rigid-As-Possible Surface Modeling" (Sorkine & Alexa, 2007)
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import logging
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, splu

from .logging_utils import log_once
from .mesh_loader import MeshData

_LOGGER = logging.getLogger(__name__)


def _log_ignored_exception(context: str = "Ignored exception") -> None:
    try:
        _LOGGER.debug("%s", context, exc_info=True)
    except Exception:
        pass


@dataclass
class FlattenedMesh:
    """
    평면화된 메쉬 결과
    
    Attributes:
        uv: (N, 2) 평면화된 2D 좌표
        faces: (M, 3) 면 인덱스 (원본과 동일)
        original_mesh: 원본 3D 메쉬 참조
        distortion_per_face: 각 면의 왜곡도 (0=왜곡없음, 1=100% 왜곡)
        scale: UV 좌표의 스케일 (원본 단위 기준)
    """
    uv: np.ndarray
    faces: np.ndarray
    original_mesh: MeshData
    distortion_per_face: Optional[np.ndarray] = None
    scale: float = 1.0
    meta: dict[str, Any] = field(default_factory=dict)
    
    # 캐시
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

        # UV 인덱스가 깨진 경우(부분 실패/손상) 크래시 방지용 필터링
        try:
            if self.uv.shape[0] > 0 and faces.shape[0] > 0:
                valid = (faces >= 0) & (faces < int(self.uv.shape[0]))
                keep = np.all(valid, axis=1)
                faces = faces[keep]
        except Exception:
            log_once(
                _LOGGER,
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
        """2D 경계 [[min_u, min_v], [max_u, max_v]]"""
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
        """2D 크기 [width, height]"""
        return self.bounds[1] - self.bounds[0]
    
    @property
    def width(self) -> float:
        """실제 너비 (원본 단위)"""
        return self.extents[0] * self.scale
    
    @property
    def height(self) -> float:
        """실제 높이 (원본 단위)"""
        return self.extents[1] * self.scale
    
    @property
    def mean_distortion(self) -> float:
        """평균 왜곡도"""
        if self.distortion_per_face is None:
            return 0.0
        return float(np.mean(self.distortion_per_face))
    
    @property
    def max_distortion(self) -> float:
        """최대 왜곡도"""
        if self.distortion_per_face is None:
            return 0.0
        return float(np.max(self.distortion_per_face))
    
    def normalize(self) -> 'FlattenedMesh':
        """UV 좌표를 [0, 1] 범위로 정규화"""
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
        extent[extent == 0] = 1  # 0으로 나누기 방지
        
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
        """
        UV를 픽셀 좌표로 변환
        
        Args:
            width: 출력 이미지 너비
            height: 출력 이미지 높이
            
        Returns:
            (N, 2) 픽셀 좌표 배열
        """
        normalized = self.normalize()
        pixels = normalized.uv.copy()
        pixels[:, 0] *= (width - 1)
        pixels[:, 1] *= (height - 1)
        # Y축 뒤집기 (이미지 좌표계)
        pixels[:, 1] = (height - 1) - pixels[:, 1]
        return pixels.astype(np.int32)


class ARAPFlattener:
    """
    ARAP (As-Rigid-As-Possible) 기반 메쉬 평면화
    
    3D 표면을 2D로 펼치면서 로컬 강성을 최대한 유지합니다.
    """
    
    def __init__(self, max_iterations: int = 50, tolerance: float = 1e-6):
        """
        Args:
            max_iterations: ARAP 반복 최대 횟수
            tolerance: 수렴 판정 임계값
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
    
    def flatten(
        self,
        mesh: MeshData,
        boundary_type: str = "free",
        initial_method: str = "lscm",
        *,
        initial_uv: np.ndarray | None = None,
        smooth_iters: int = 0,
        smooth_strength: float = 0.15,
        pack_compact: bool = True,
    ) -> FlattenedMesh:
        """
        3D 메쉬를 2D로 평면화
        
        Args:
            mesh: 입력 3D 메쉬
            boundary_type: 경계 조건 ('free', 'fixed_circle', 'fixed_rect')
            initial_method: 초기 파라미터화 방법 ('lscm', 'tutte')
            initial_uv: 초기 UV를 직접 지정할 경우 사용
            smooth_iters: 후처리 UV 스무딩 반복 횟수 (0이면 비활성)
            smooth_strength: 스무딩 강도 (0~1)
            pack_compact: 다중 컴포넌트 패킹 방식(컴팩트 선반 패킹)
            
        Returns:
            FlattenedMesh: 평면화 결과
        """
        if mesh is None:
            raise ValueError("mesh is None")

        # 입력 메쉬 정리(퇴화 삼각형/NaN 제거 + 사용 정점만 유지)
        mesh = self._sanitize_mesh(mesh)
        if mesh.n_vertices == 0 or mesh.n_faces == 0:
            return FlattenedMesh(
                uv=np.zeros((mesh.n_vertices, 2), dtype=np.float64),
                faces=mesh.faces,
                original_mesh=mesh,
                distortion_per_face=np.zeros((mesh.n_faces,), dtype=np.float64),
                scale=1.0,
            )

        components = self._face_connected_components(mesh.faces)
        if len(components) <= 1:
            # 1) 초기 파라미터화 (안정적인 fallback 포함)
            if initial_uv is None:
                initial_uv = self._safe_initial_parameterization(mesh, initial_method)
            else:
                initial_uv = np.asarray(initial_uv, dtype=np.float64)

            # 2) ARAP 최적화 (UV는 mesh 단위 스케일 유지)
            optimized_uv = self._arap_optimize(mesh, initial_uv, boundary_type)
            optimized_uv = self._orient_uv_pca(optimized_uv)

            # 2.5) 가벼운 UV 스무딩 (원자화/각진 느낌 완화)
            if int(smooth_iters) > 0 and float(smooth_strength) > 0:
                try:
                    edge_i, edge_j, edge_w = self._compute_cotangent_edge_weights(mesh)
                    anchors = self._pick_anchor_pair(mesh, np.arange(mesh.n_vertices, dtype=np.int32))
                    optimized_uv = self._smooth_uv_laplacian(
                        optimized_uv,
                        edge_i,
                        edge_j,
                        edge_w,
                        iterations=int(smooth_iters),
                        strength=float(smooth_strength),
                        fixed_indices=anchors,
                    )
                except Exception:
                    _log_ignored_exception()

            # 3) 왜곡도 계산 (UV가 mesh 단위이므로 scale=1.0 고정)
            distortion = self._compute_distortion(mesh, optimized_uv)

            return FlattenedMesh(
                uv=optimized_uv,
                faces=mesh.faces,
                original_mesh=mesh,
                distortion_per_face=distortion,
                scale=1.0,
            )

        # 여러 연결 컴포넌트: 각각 펼친 뒤 패킹
        components = sorted(components, key=lambda a: int(a.size), reverse=True)
        uv_all = np.zeros((mesh.n_vertices, 2), dtype=np.float64)
        gap = 1.0
        try:
            ext = np.asarray(mesh.extents, dtype=np.float64).reshape(-1)
            if ext.size >= 1 and np.isfinite(ext).all():
                gap = float(max(0.5, 0.005 * float(np.max(ext))))
        except Exception:
            gap = 1.0

        packed: list[tuple[np.ndarray, np.ndarray, float, float]] = []
        for face_indices in components:
            comp_mesh, vmap = self._extract_submesh_with_mapping(mesh, face_indices)
            if comp_mesh.n_vertices == 0 or comp_mesh.n_faces == 0:
                continue

            if initial_uv is None:
                comp_initial = self._safe_initial_parameterization(comp_mesh, initial_method)
            else:
                try:
                    comp_initial = np.asarray(initial_uv, dtype=np.float64)[vmap]
                except Exception:
                    comp_initial = self._safe_initial_parameterization(comp_mesh, initial_method)
            comp_uv = self._arap_optimize(comp_mesh, comp_initial, boundary_type)
            comp_uv = np.asarray(comp_uv, dtype=np.float64)
            if comp_uv.ndim != 2 or comp_uv.shape[0] == 0 or comp_uv.shape[1] < 2:
                continue
            comp_uv = comp_uv[:, :2].copy()
            comp_uv = self._orient_uv_pca(comp_uv)

            if int(smooth_iters) > 0 and float(smooth_strength) > 0:
                try:
                    edge_i, edge_j, edge_w = self._compute_cotangent_edge_weights(comp_mesh)
                    anchors = self._pick_anchor_pair(comp_mesh, np.arange(comp_mesh.n_vertices, dtype=np.int32))
                    comp_uv = self._smooth_uv_laplacian(
                        comp_uv,
                        edge_i,
                        edge_j,
                        edge_w,
                        iterations=int(smooth_iters),
                        strength=float(smooth_strength),
                        fixed_indices=anchors,
                    )
                except Exception:
                    _log_ignored_exception()

            finite = np.all(np.isfinite(comp_uv), axis=1)
            if np.any(finite):
                min_uv = comp_uv[finite].min(axis=0)
                max_uv = comp_uv[finite].max(axis=0)
            else:
                min_uv = np.zeros((2,), dtype=np.float64)
                max_uv = np.zeros((2,), dtype=np.float64)
            comp_uv[~finite] = 0.0
            comp_uv -= min_uv
            width = float(max_uv[0] - min_uv[0]) if np.any(finite) else 0.0
            height = float(max_uv[1] - min_uv[1]) if np.any(finite) else 0.0
            packed.append((comp_uv, vmap, width, height))

        if packed:
            packed.sort(key=lambda x: float(x[2] * x[3]), reverse=True)
            max_w = max(p[2] for p in packed)
            total_area = sum(float(p[2] * p[3]) for p in packed)
            row_limit = max(max_w, float(np.sqrt(total_area)) * (1.25 if pack_compact else 2.0))
            cursor_x = 0.0
            cursor_y = 0.0
            row_h = 0.0
            for comp_uv, vmap, width, height in packed:
                if cursor_x > 0.0 and (cursor_x + width) > row_limit:
                    cursor_x = 0.0
                    cursor_y += row_h + gap
                    row_h = 0.0
                comp_uv[:, 0] += cursor_x
                comp_uv[:, 1] += cursor_y
                uv_all[vmap] = comp_uv
                cursor_x += width + gap
                row_h = max(row_h, height)

        distortion = self._compute_distortion(mesh, uv_all)

        return FlattenedMesh(
            uv=uv_all,
            faces=mesh.faces,
            original_mesh=mesh,
            distortion_per_face=distortion,
            scale=1.0,
        )

    def _safe_initial_parameterization(self, mesh: MeshData, initial_method: str) -> np.ndarray:
        """LSCM/Tutte 초기 UV 생성이 실패해도 항상 UV를 반환합니다."""
        method = str(initial_method or "lscm").lower().strip()

        # Safety: large meshes make LSCM/Tutte extremely slow and memory-heavy.
        try:
            max_verts = int(getattr(self, "_initial_param_max_vertices", 200000))
        except Exception:
            max_verts = 200000
        try:
            max_faces = int(getattr(self, "_initial_param_max_faces", 400000))
        except Exception:
            max_faces = 400000
        n_verts = int(getattr(mesh, "n_vertices", 0) or 0)
        n_faces = int(getattr(mesh, "n_faces", 0) or 0)
        if n_verts > max_verts or n_faces > max_faces:
            log_once(
                _LOGGER,
                "flattener:initial_param_fallback_large",
                logging.INFO,
                "Initial UV fallback to PCA projection (verts=%d, faces=%d)",
                n_verts,
                n_faces,
            )
            basis = self._compute_reference_basis(mesh)  # (2, 3)
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            return (basis @ vertices.T).T

        try:
            if method == "tutte":
                return self._tutte_parameterization(mesh)
            if method == "lscm":
                return self._lscm_parameterization(mesh)
        except Exception:
            _LOGGER.debug("Initial UV parameterization failed (method=%s)", method, exc_info=True)

        # fallback: Tutte -> LSCM -> PCA projection
        try:
            return self._tutte_parameterization(mesh)
        except Exception:
            _LOGGER.debug("Fallback Tutte parameterization failed", exc_info=True)
        try:
            return self._lscm_parameterization(mesh)
        except Exception:
            _LOGGER.debug("Fallback LSCM parameterization failed", exc_info=True)

        basis = self._compute_reference_basis(mesh)  # (2, 3)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        uv = (basis @ vertices.T).T
        return np.asarray(uv, dtype=np.float64)

    def _sanitize_mesh(self, mesh: MeshData) -> MeshData:
        """
        펼침/선택 등 후처리 안정성을 위해 메쉬를 정리합니다.
        - NaN/Inf 정점이 포함된 face 제거
        - 중복 인덱스(face 퇴화) 제거
        - 면적이 거의 0인 삼각형 제거
        - 사용 정점만 남기도록 reindex
        """
        if mesh is None:
            raise ValueError("mesh is None")

        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        if vertices.ndim != 2 or vertices.shape[0] == 0 or faces.ndim != 2 or faces.shape[0] == 0:
            return mesh

        faces = faces[:, :3].astype(np.int32, copy=False)
        a = faces[:, 0]
        b = faces[:, 1]
        c = faces[:, 2]

        mask = (a != b) & (b != c) & (a != c)
        finite_v = np.all(np.isfinite(vertices), axis=1)
        mask &= finite_v[a] & finite_v[b] & finite_v[c]

        faces = faces[mask]
        if faces.shape[0] == 0:
            return MeshData(
                vertices=np.zeros((0, 3), dtype=np.float64),
                faces=np.zeros((0, 3), dtype=np.int32),
                normals=None,
                face_normals=None,
                uv_coords=None,
                texture=mesh.texture,
                unit=mesh.unit,
                filepath=mesh.filepath,
            )

        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        area2 = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        faces = faces[area2 > 1e-12]
        if faces.shape[0] == 0:
            return MeshData(
                vertices=np.zeros((0, 3), dtype=np.float64),
                faces=np.zeros((0, 3), dtype=np.int32),
                normals=None,
                face_normals=None,
                uv_coords=None,
                texture=mesh.texture,
                unit=mesh.unit,
                filepath=mesh.filepath,
            )

        unique_verts = np.unique(faces.reshape(-1)).astype(np.int32, copy=False)
        new_vertices = vertices[unique_verts]
        new_faces = np.searchsorted(unique_verts, faces).astype(np.int32, copy=False)

        new_uv = None
        if mesh.uv_coords is not None:
            try:
                uv = np.asarray(mesh.uv_coords, dtype=np.float64)
                if uv.ndim == 2 and uv.shape[0] == vertices.shape[0] and uv.shape[1] >= 2:
                    new_uv = uv[unique_verts][:, :2].copy()
            except Exception:
                new_uv = None

        return MeshData(
            vertices=new_vertices,
            faces=new_faces,
            normals=None,
            face_normals=None,
            uv_coords=new_uv,
            texture=mesh.texture,
            unit=mesh.unit,
            filepath=mesh.filepath,
        )

    def _orient_uv_pca(self, uv: np.ndarray) -> np.ndarray:
        """UV를 2D PCA로 회전시켜 axis-aligned 배치에 가깝게 정렬합니다."""
        uv = np.asarray(uv, dtype=np.float64)
        if uv.ndim != 2 or uv.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.float64)

        out = uv[:, :2].copy()
        finite = np.all(np.isfinite(out), axis=1)
        if np.count_nonzero(finite) < 2:
            out[~finite] = 0.0
            return out

        pts = out[finite]
        mean = pts.mean(axis=0)
        centered = pts - mean

        cov = centered.T @ centered
        if cov.shape == (2, 2) and centered.shape[0] > 0:
            cov = cov / float(centered.shape[0])

        try:
            evals, evecs = np.linalg.eigh(cov)
        except np.linalg.LinAlgError:
            out[~finite] = 0.0
            return out

        order = np.argsort(evals)[::-1]
        axes = evecs[:, order]

        # 부호 고정(결정적) + 우수(right-handed) 유지
        for k in range(2):
            axis = axes[:, k]
            idx = int(np.argmax(np.abs(axis)))
            if axis[idx] < 0:
                axes[:, k] *= -1
        if float(np.linalg.det(axes)) < 0:
            axes[:, 1] *= -1

        out[finite] = (pts - mean) @ axes
        out[~finite] = 0.0
        return out

    def _face_connected_components(self, faces: np.ndarray) -> list[np.ndarray]:
        """면(face) adjacency(공유 edge) 기준 연결 컴포넌트를 찾습니다."""
        faces = np.asarray(faces, dtype=np.int32)
        if faces.ndim != 2 or faces.shape[0] == 0:
            return []

        faces = faces[:, :3].astype(np.int32, copy=False)
        m = int(faces.shape[0])

        # Very large meshes: Python-side component split is too slow.
        try:
            max_faces = int(getattr(self, "_component_split_max_faces", 400000))
        except Exception:
            max_faces = 400000
        if m > max_faces:
            log_once(
                _LOGGER,
                "flattener:skip_face_components",
                logging.INFO,
                "Skipping face connected-components split (faces=%d > %d)",
                m,
                int(max_faces),
            )
            return [np.arange(m, dtype=np.int32)]

        parent = np.arange(m, dtype=np.int32)
        rank = np.zeros(m, dtype=np.int8)

        def find(x: int) -> int:
            x = int(x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = int(parent[x])
            return x

        def union(a_idx: int, b_idx: int) -> None:
            ra = find(a_idx)
            rb = find(b_idx)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] = np.int8(rank[ra] + 1)

        edge_to_face: dict[tuple[int, int], int] = {}
        for fi, face in enumerate(faces):
            a, b, c = int(face[0]), int(face[1]), int(face[2])
            for u, v in ((a, b), (b, c), (c, a)):
                if u > v:
                    u, v = v, u
                key = (u, v)
                prev = edge_to_face.get(key)
                if prev is None:
                    edge_to_face[key] = int(fi)
                else:
                    union(int(fi), int(prev))

        groups: dict[int, list[int]] = {}
        for fi in range(m):
            root = find(fi)
            groups.setdefault(root, []).append(fi)

        return [np.asarray(g, dtype=np.int32) for g in groups.values()]

    def _extract_submesh_with_mapping(
        self, mesh: MeshData, face_indices: np.ndarray
    ) -> tuple[MeshData, np.ndarray]:
        """face_indices로 submesh를 만들고, (submesh_vertex -> original_vertex) 매핑을 함께 반환합니다."""
        face_indices = np.asarray(face_indices, dtype=np.int32).reshape(-1)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)

        if face_indices.size == 0 or faces.ndim != 2 or faces.shape[0] == 0:
            empty = MeshData(
                vertices=np.zeros((0, 3), dtype=np.float64),
                faces=np.zeros((0, 3), dtype=np.int32),
                normals=None,
                face_normals=None,
                uv_coords=None,
                texture=mesh.texture,
                unit=mesh.unit,
                filepath=mesh.filepath,
            )
            return empty, np.zeros((0,), dtype=np.int32)

        valid = (face_indices >= 0) & (face_indices < int(faces.shape[0]))
        face_indices = face_indices[valid]
        if face_indices.size == 0:
            empty = MeshData(
                vertices=np.zeros((0, 3), dtype=np.float64),
                faces=np.zeros((0, 3), dtype=np.int32),
                normals=None,
                face_normals=None,
                uv_coords=None,
                texture=mesh.texture,
                unit=mesh.unit,
                filepath=mesh.filepath,
            )
            return empty, np.zeros((0,), dtype=np.int32)

        selected_faces = faces[face_indices][:, :3]
        unique_verts = np.unique(selected_faces.reshape(-1)).astype(np.int32, copy=False)
        new_vertices = vertices[unique_verts]
        new_faces = np.searchsorted(unique_verts, selected_faces).astype(np.int32, copy=False)

        new_uv = None
        if mesh.uv_coords is not None:
            try:
                uv = np.asarray(mesh.uv_coords, dtype=np.float64)
                if uv.ndim == 2 and uv.shape[0] == vertices.shape[0] and uv.shape[1] >= 2:
                    new_uv = uv[unique_verts][:, :2].copy()
            except Exception:
                new_uv = None

        submesh = MeshData(
            vertices=new_vertices,
            faces=new_faces,
            normals=None,
            face_normals=None,
            uv_coords=new_uv,
            texture=mesh.texture,
            unit=mesh.unit,
            filepath=mesh.filepath,
        )
        return submesh, unique_verts
    
    def _lscm_parameterization(self, mesh: MeshData) -> np.ndarray:
        """
        LSCM (Least Squares Conformal Maps) 초기 파라미터화
        각도를 보존하는 등각 매핑
        """
        n = mesh.n_vertices
        m = mesh.n_faces
        
        vertices = mesh.vertices
        faces = mesh.faces
        
        # 경계 정점 찾기 (없으면 전체 정점에서 앵커 선택)
        boundary = np.asarray(mesh.get_boundary_vertices(), dtype=np.int32).reshape(-1)
        candidates = boundary if boundary.size >= 2 else np.arange(n, dtype=np.int32)

        fixed_verts = self._pick_anchor_pair(mesh, candidates)
        dist = float(np.linalg.norm(vertices[fixed_verts[1]] - vertices[fixed_verts[0]]))
        if not np.isfinite(dist) or dist < 1e-9:
            dist = 1.0

        # 두 앵커 정점을 mesh 단위로 고정 (0, 0)과 (dist, 0)
        fixed_pos = np.array([[0.0, 0.0], [dist, 0.0]], dtype=np.float64)
        
        # LSCM 방정식 구성
        # 각 삼각형에 대해 등각 조건 설정
        
        # 로컬 좌표계에서의 삼각형 정점
        rows = []
        cols = []
        vals_real = []

        for fi, face in enumerate(faces):
            v0, v1, v2 = vertices[face]
            
            # 로컬 2D 좌표 (삼각형을 xy 평면에 배치)
            e1 = v1 - v0
            e2 = v2 - v0
            
            # 첫 번째 엣지를 x축에 정렬
            len_e1 = np.linalg.norm(e1)
            if len_e1 < 1e-10:
                continue
            
            x1 = len_e1
            y1 = 0.0
            
            # 두 번째 정점의 로컬 좌표
            x2 = np.dot(e2, e1) / len_e1
            
            # e1에 수직인 성분
            e1_normalized = e1 / len_e1
            perp = e2 - x2 * e1_normalized
            y2 = np.linalg.norm(perp)
            
            # 삼각형 면적
            area = 0.5 * x1 * y2
            if area < 1e-10:
                continue
            
            # LSCM 가중치 (Levy et al.)
            # W_j = (v_{j+1} - v_{j-1}) / (2 * sqrt(Area))
            sqrt_area = np.sqrt(area)
            
            # 복소수 좌표로 변환
            z0 = complex(0, 0)
            z1 = complex(x1, y1)
            z2 = complex(x2, y2)
            
            # 각 정점에 대한 가중치
            w0 = (z2 - z1) / (2 * sqrt_area)
            w1 = (z0 - z2) / (2 * sqrt_area)
            w2 = (z1 - z0) / (2 * sqrt_area)
            
            for vi, w in zip(face, [w0, w1, w2]):
                rows.extend([2*fi, 2*fi, 2*fi+1, 2*fi+1])
                cols.extend([2*vi, 2*vi+1, 2*vi, 2*vi+1])
                vals_real.extend([w.real, -w.imag, w.imag, w.real])
        
        # 희소 행렬 구성
        A = sparse.coo_matrix(
            (vals_real, (rows, cols)),
            shape=(2*m, 2*n)
        ).tocsr()
        
        # 고정 정점 제약 추가
        # 고정된 열을 제거하고 우변으로 이동
        free_mask = np.ones(n, dtype=bool)
        free_mask[fixed_verts] = False
        free_indices = np.where(free_mask)[0]
        
        # 열 인덱스 재매핑
        col_map = np.zeros(n, dtype=int)
        col_map[free_indices] = np.arange(len(free_indices))
        
        # 자유 정점만 포함하는 시스템 구성
        free_cols = []
        for idx in free_indices:
            free_cols.extend([2*idx, 2*idx+1])
        
        A_free = A[:, free_cols]
        
        # 고정 정점 기여도를 우변으로
        fixed_cols = []
        for idx in fixed_verts:
            fixed_cols.extend([2*idx, 2*idx+1])
        
        A_fixed = A[:, fixed_cols]
        b_fixed = np.zeros(len(fixed_verts) * 2)
        for i, (idx, pos) in enumerate(zip(fixed_verts, fixed_pos)):
            b_fixed[2*i] = pos[0]
            b_fixed[2*i+1] = pos[1]
        
        b = -A_fixed @ b_fixed
        
        # 최소자승 풀이
        AtA = A_free.T @ A_free
        Atb = A_free.T @ b
        
        # 정규화 추가 (수치 안정성)
        AtA += sparse.eye(AtA.shape[0]) * 1e-8
        
        x_free = spsolve(AtA.tocsr(), Atb)
        
        # 결과 조립
        uv = np.zeros((n, 2))
        for i, idx in enumerate(fixed_verts):
            uv[idx] = fixed_pos[i]
        for i, idx in enumerate(free_indices):
            uv[idx, 0] = x_free[2*i]
            uv[idx, 1] = x_free[2*i+1]
        
        return uv
    
    def _tutte_parameterization(self, mesh: MeshData) -> np.ndarray:
        """
        Tutte 임베딩 - 경계를 원에 고정하고 내부를 조화 매핑
        """
        n = mesh.n_vertices

        loops = mesh.get_boundary_loops()
        boundary = loops[0] if loops else mesh.get_boundary_vertices()
        if loops:
            # 가장 큰 경계 루프를 사용
            boundary = max(loops, key=lambda a: int(a.size))
        
        if len(boundary) < 3:
            # 닫힌 메쉬는 LSCM으로 대체
            return self._lscm_parameterization(mesh)
        
        # 경계 정점을 원에 배치 (mesh 단위 스케일 유지: perimeter 기반 radius)
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        b_pts = verts[boundary]
        if b_pts.shape[0] >= 2:
            diffs = b_pts[(np.arange(len(boundary)) + 1) % len(boundary)] - b_pts
            perim = float(np.linalg.norm(diffs, axis=1).sum())
        else:
            perim = 0.0
        radius = perim / (2.0 * np.pi) if perim > 1e-12 else 1.0

        angles = np.linspace(0, 2*np.pi, len(boundary), endpoint=False)
        boundary_uv = np.column_stack([np.cos(angles), np.sin(angles)]) * radius
        
        # 라플라시안 행렬 구성
        is_boundary = np.zeros(n, dtype=bool)
        is_boundary[boundary] = True
        
        # 인접 정보 구성
        adjacency = [set() for _ in range(n)]
        for face in mesh.faces:
            for i in range(3):
                adjacency[face[i]].add(face[(i+1)%3])
                adjacency[face[i]].add(face[(i+2)%3])
        
        # 균등 가중치 라플라시안
        rows, cols, vals = [], [], []
        b = np.zeros((n, 2))
        
        for i in range(n):
            if is_boundary[i]:
                # 경계 정점: 고정
                rows.append(i)
                cols.append(i)
                vals.append(1.0)
                idx = np.where(boundary == i)[0][0]
                b[i] = boundary_uv[idx]
            else:
                # 내부 정점: 이웃의 평균
                neighbors = list(adjacency[i])
                if len(neighbors) == 0:
                    rows.append(i)
                    cols.append(i)
                    vals.append(1.0)
                else:
                    rows.append(i)
                    cols.append(i)
                    vals.append(1.0)
                    for j in neighbors:
                        rows.append(i)
                        cols.append(j)
                        vals.append(-1.0 / len(neighbors))
        
        L = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()

        # 풀이
        uv = np.zeros((n, 2))
        uv[:, 0] = np.asarray(spsolve(L, b[:, 0])).ravel()
        uv[:, 1] = np.asarray(spsolve(L, b[:, 1])).ravel()

        return uv
    
    def _arap_optimize(self, mesh: MeshData, initial_uv: np.ndarray,
                       boundary_type: str) -> np.ndarray:
        """
        ARAP 최적화 반복 (3D -> 2D ARAP 파라미터화)

        - (u_j - u_i) 2D 엣지와 (p_j - p_i) 3D 엣지를 직접 매칭하고,
          정점별 2x3 회전(R_i)을 SVD(Procrustes)로 추정합니다.
        - 기존처럼 3D를 먼저 어떤 평면으로 투영하면(예: PCA) 곡률이 큰 메쉬에서
          엣지 길이 정보가 크게 훼손될 수 있어, 곡면 펼침에 부적합합니다.
        """
        n = mesh.n_vertices
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        if (
            n <= 0
            or vertices.ndim != 2
            or vertices.shape[0] == 0
            or faces.ndim != 2
            or faces.shape[0] == 0
        ):
            return np.zeros((n, 2), dtype=np.float64)

        uv = np.asarray(initial_uv, dtype=np.float64)
        if uv.ndim != 2 or uv.shape[0] != n or uv.shape[1] < 2:
            uv = np.zeros((n, 2), dtype=np.float64)
        else:
            uv = uv[:, :2].copy()

        # fallback용 기준 투영(고립 정점/SVD 실패 시 사용)
        reference_basis = self._compute_reference_basis(mesh)  # (2,3)

        # Guard: ARAP is expensive on huge meshes (LU + per-vertex SVD).
        try:
            max_verts = int(getattr(self, "_arap_max_vertices", 150000))
        except Exception:
            max_verts = 150000
        try:
            max_faces = int(getattr(self, "_arap_max_faces", 300000))
        except Exception:
            max_faces = 300000
        if n > max_verts or int(faces.shape[0]) > max_faces:
            log_once(
                _LOGGER,
                "flattener:arap_skip_large",
                logging.INFO,
                "Skipping ARAP optimize (verts=%d, faces=%d)",
                int(n),
                int(faces.shape[0]),
            )
            return uv

        edge_i, edge_j, edge_w = self._compute_cotangent_edge_weights(mesh)
        if edge_i.size == 0:
            return (reference_basis @ vertices.T).T

        edge_vec_3d = vertices[edge_j] - vertices[edge_i]

        bt = str(boundary_type or "free").lower().strip()
        if bt in {"fixed_circle", "fixed_rect"}:
            boundary = np.asarray(mesh.get_boundary_vertices(), dtype=np.int32).reshape(-1)
        else:
            boundary = np.zeros((0,), dtype=np.int32)
        fixed_indices, fixed_uv, uv = self._build_fixed_constraints(
            mesh,
            uv,
            boundary_type=bt,
            boundary=boundary,
        )
        fixed_indices = np.asarray(fixed_indices, dtype=np.int32).reshape(-1)
        fixed_uv = np.asarray(fixed_uv, dtype=np.float64).reshape(-1, 2)

        if fixed_indices.size > 0:
            uv[fixed_indices] = fixed_uv

        # 라플라시안(고정) + reduced system factorization (반복마다 RHS만 변경)
        L = self._build_laplacian(n, edge_i, edge_j, edge_w).tocsc()
        all_idx = np.arange(n, dtype=np.int32)

        fixed_mask = np.zeros(n, dtype=bool)
        if fixed_indices.size > 0:
            fixed_mask[fixed_indices] = True
        free_indices = all_idx[~fixed_mask]
        if free_indices.size == 0:
            return uv

        L_ff = L[free_indices][:, free_indices].tocsc()
        reg = 1e-8
        L_ff_reg = L_ff + sparse.eye(L_ff.shape[0], format="csc") * reg
        try:
            lu = splu(L_ff_reg)
            use_lu = True
        except Exception:
            lu = None
            use_lu = False
            L_ff_reg = L_ff_reg.tocsr()

        L_fc = None
        if fixed_indices.size > 0:
            try:
                L_fc = L[free_indices][:, fixed_indices].tocsc()
            except Exception:
                L_fc = None

        max_iter = int(max(0, self.max_iterations))
        tol = float(self.tolerance)

        for _ in range(max_iter):
            # Step 1: 로컬 회전 최적화
            rotations = self._compute_local_rotations(
                vertices,
                uv,
                edge_i=edge_i,
                edge_j=edge_j,
                edge_w=edge_w,
                edge_vec_3d=edge_vec_3d,
                reference_basis=reference_basis,
            )

            # Step 2: 글로벌 위치 최적화
            b = self._compute_global_rhs(
                rotations,
                edge_i=edge_i,
                edge_j=edge_j,
                edge_w=edge_w,
                edge_vec_3d=edge_vec_3d,
                n_vertices=n,
            )
            rhs = b[free_indices].copy()
            if fixed_indices.size > 0 and L_fc is not None:
                rhs -= (L_fc @ uv[fixed_indices])

            new_uv = uv.copy()
            if use_lu and lu is not None:
                new_uv[free_indices, 0] = np.asarray(lu.solve(rhs[:, 0])).ravel()
                new_uv[free_indices, 1] = np.asarray(lu.solve(rhs[:, 1])).ravel()
            else:
                new_uv[free_indices, 0] = np.asarray(spsolve(L_ff_reg, rhs[:, 0])).ravel()
                new_uv[free_indices, 1] = np.asarray(spsolve(L_ff_reg, rhs[:, 1])).ravel()

            diff = new_uv[free_indices] - uv[free_indices]
            if fixed_indices.size > 0:
                new_uv[fixed_indices] = fixed_uv

            # 수렴 확인 (UV 변화량 기반, 에너지 계산 스킵으로 속도 개선)
            try:
                max_delta = float(np.max(np.abs(diff))) if diff.size else 0.0
                scale_ref = float(np.max(np.abs(uv[free_indices]))) if uv[free_indices].size else 1.0
            except Exception:
                max_delta = 0.0
                scale_ref = 1.0
            uv = new_uv
            if max_delta <= tol * max(1.0, scale_ref):
                break

        return uv

    def _pick_anchor_pair(self, mesh: MeshData, candidates: np.ndarray) -> np.ndarray:
        """후보 정점 집합에서 대략적인 지름(가장 먼 두 점) 쌍을 선택합니다."""
        candidates = np.asarray(candidates, dtype=np.int32).reshape(-1)
        n = int(mesh.n_vertices)
        if n <= 0:
            return np.array([0, 0], dtype=np.int32)
        if candidates.size < 2:
            a = int(candidates[0]) if candidates.size == 1 else 0
            b = 1 if n > 1 else a
            return np.array([a, b], dtype=np.int32)

        verts = np.asarray(mesh.vertices, dtype=np.float64)
        a0 = int(candidates[0])
        d0 = np.linalg.norm(verts[candidates] - verts[a0], axis=1)
        b = int(candidates[int(np.argmax(d0))])
        d1 = np.linalg.norm(verts[candidates] - verts[b], axis=1)
        a = int(candidates[int(np.argmax(d1))])
        if a == b:
            for cand in candidates:
                if int(cand) != b:
                    a = int(cand)
                    break
        return np.array([b, a], dtype=np.int32)

    def _align_uv_to_anchors(self, uv: np.ndarray, a: int, b: int, target_dist: float) -> np.ndarray:
        """UV를 (a)->(0,0), (b)->(target_dist,0)로 오도록 similarity 정렬합니다."""
        uv = np.asarray(uv, dtype=np.float64)
        if uv.ndim != 2 or uv.shape[0] == 0 or uv.shape[1] < 2:
            return np.zeros((int(uv.shape[0]) if uv.ndim == 2 else 0, 2), dtype=np.float64)

        a = int(a)
        b = int(b)
        if a < 0 or b < 0 or a >= uv.shape[0] or b >= uv.shape[0]:
            return uv[:, :2].copy()

        p0 = uv[a, :2].copy()
        p1 = uv[b, :2].copy()
        v = p1 - p0
        norm = float(np.linalg.norm(v))
        if not np.isfinite(norm) or norm < 1e-12:
            out = uv[:, :2].copy()
            out[a] = (0.0, 0.0)
            out[b] = (float(target_dist), 0.0)
            return out

        angle = float(np.arctan2(v[1], v[0]))
        c = float(np.cos(-angle))
        s = float(np.sin(-angle))
        rot = np.array([[c, -s], [s, c]], dtype=np.float64)

        scale = float(target_dist) / norm if np.isfinite(target_dist) and float(target_dist) > 0 else 1.0
        out = (uv[:, :2] - p0) @ rot.T
        out *= scale
        out[a] = (0.0, 0.0)
        out[b] = (float(target_dist), 0.0)
        return out

    def _build_fixed_constraints(
        self,
        mesh: MeshData,
        uv: np.ndarray,
        *,
        boundary_type: str,
        boundary: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """boundary_type에 따라 고정 정점/좌표를 구성하고, 필요 시 UV를 정렬합니다."""
        n = int(mesh.n_vertices)
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        boundary_type = str(boundary_type or "free").lower().strip()
        boundary = np.asarray(boundary, dtype=np.int32).reshape(-1)

        if boundary_type == "fixed_circle" and boundary.size >= 3:
            b_pts = verts[boundary]
            diffs = b_pts[(np.arange(len(boundary)) + 1) % len(boundary)] - b_pts
            perim = float(np.linalg.norm(diffs, axis=1).sum())
            radius = perim / (2.0 * np.pi) if perim > 1e-12 else 1.0
            angles = np.linspace(0.0, 2.0 * np.pi, len(boundary), endpoint=False)
            boundary_uv = np.column_stack([np.cos(angles), np.sin(angles)]) * radius
            new_uv = np.asarray(uv, dtype=np.float64).copy()
            new_uv[boundary] = boundary_uv
            return boundary.copy(), boundary_uv.astype(np.float64, copy=False), new_uv

        if boundary_type == "fixed_rect" and boundary.size >= 4:
            b_pts = verts[boundary]
            seg = b_pts[(np.arange(len(boundary)) + 1) % len(boundary)] - b_pts
            seg_len = np.linalg.norm(seg, axis=1)
            perim = float(seg_len.sum())
            if perim <= 1e-12:
                return boundary.copy(), uv[boundary][:, :2].copy(), np.asarray(uv, dtype=np.float64).copy()

            basis = self._compute_reference_basis(mesh)
            proj = (basis @ b_pts.T).T
            ext = proj.max(axis=0) - proj.min(axis=0)
            w = float(max(ext[0], 1e-6))
            h = float(max(ext[1], 1e-6))
            rect_perim = 2.0 * (w + h)
            s = perim / rect_perim if rect_perim > 1e-12 else 1.0
            w *= s
            h *= s

            cum = np.concatenate([[0.0], np.cumsum(seg_len)])
            t = cum[:-1] / perim * (2.0 * (w + h))
            out = np.zeros((len(boundary), 2), dtype=np.float64)
            for i, ti in enumerate(t):
                if ti < w:
                    out[i] = (ti, 0.0)
                elif ti < w + h:
                    out[i] = (w, ti - w)
                elif ti < 2.0 * w + h:
                    out[i] = (w - (ti - (w + h)), h)
                else:
                    out[i] = (0.0, h - (ti - (2.0 * w + h)))

            new_uv = np.asarray(uv, dtype=np.float64).copy()
            new_uv[boundary] = out
            return boundary.copy(), out, new_uv

        # 자유 경계: 두 정점만 고정 (회전/이동 방지)
        candidates = boundary if boundary.size >= 2 else np.arange(n, dtype=np.int32)
        anchors = self._pick_anchor_pair(mesh, candidates)
        dist = float(np.linalg.norm(verts[anchors[1]] - verts[anchors[0]]))
        if not np.isfinite(dist) or dist < 1e-9:
            dist = 1.0
        new_uv = self._align_uv_to_anchors(uv, anchors[0], anchors[1], dist)
        fixed_uv = new_uv[anchors][:, :2].copy()
        return anchors.astype(np.int32, copy=False), fixed_uv, new_uv

    def _compute_reference_basis(self, mesh: MeshData) -> np.ndarray:
        """
        3D 엣지를 2D로 투영하기 위한 기준 축(2x3)을 계산합니다.

        현재 구현은 PCA 기반으로 "가장 잘 맞는" 2D 평면을 잡아,
        메쉬의 방향과 무관하게 ARAP이 동작하도록 합니다.
        """
        vertices = mesh.vertices.astype(np.float64, copy=False)
        centered = vertices - vertices.mean(axis=0)

        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = np.argsort(eigenvalues)[::-1]
        axes = eigenvectors[:, order]  # columns

        u = axes[:, 0].copy()
        v = axes[:, 1].copy()

        # 축 부호를 결정적으로 고정 (PCA 부호 뒤집힘 방지)
        for axis in (u, v):
            idx = int(np.argmax(np.abs(axis)))
            if axis[idx] < 0:
                axis *= -1

        # 2x3 투영 행렬
        return np.vstack([u, v])
    
    def _compute_cotangent_edge_weights(self, mesh: MeshData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Symmetric cotangent weights for undirected edges.

        Returns:
            (edge_i, edge_j, edge_w) with edge_i < edge_j.
        """
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        if (
            vertices.ndim != 2
            or vertices.shape[0] == 0
            or faces.ndim != 2
            or faces.shape[0] == 0
            or faces.shape[1] < 3
        ):
            return (
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.float64),
            )

        f = faces[:, :3].astype(np.int32, copy=False)
        a = f[:, 0]
        b = f[:, 1]
        c = f[:, 2]

        va = vertices[a]
        vb = vertices[b]
        vc = vertices[c]

        eps = 1e-10

        # Cotangent at the opposite vertex for each edge in the triangle.
        u0 = va - vc
        v0 = vb - vc
        cot_c = np.einsum("ij,ij->i", u0, v0) / (np.linalg.norm(np.cross(u0, v0), axis=1) + eps)

        u1 = vb - va
        v1 = vc - va
        cot_a = np.einsum("ij,ij->i", u1, v1) / (np.linalg.norm(np.cross(u1, v1), axis=1) + eps)

        u2 = vc - vb
        v2 = va - vb
        cot_b = np.einsum("ij,ij->i", u2, v2) / (np.linalg.norm(np.cross(u2, v2), axis=1) + eps)

        edge_i = np.concatenate([a, b, c]).astype(np.int32, copy=False)
        edge_j = np.concatenate([b, c, a]).astype(np.int32, copy=False)
        edge_w = (0.5 * np.concatenate([cot_c, cot_a, cot_b])).astype(np.float64, copy=False)

        finite = np.isfinite(edge_w)
        if not bool(np.all(finite)):
            edge_w = edge_w.copy()
            edge_w[~finite] = 0.0

        # Undirected edge key (i < j)
        ii = np.minimum(edge_i, edge_j)
        jj = np.maximum(edge_i, edge_j)
        keep = ii != jj
        ii = ii[keep]
        jj = jj[keep]
        edge_w = edge_w[keep]
        if ii.size == 0:
            return (
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.float64),
            )

        order = np.lexsort((jj, ii))
        ii = ii[order]
        jj = jj[order]
        edge_w = edge_w[order]

        # Sum duplicate edges (same undirected pair).
        n = int(vertices.shape[0])
        key = ii.astype(np.int64) * np.int64(n) + jj.astype(np.int64)
        starts = np.r_[0, np.nonzero(key[1:] != key[:-1])[0] + 1]
        w_sum = np.add.reduceat(edge_w, starts)
        ii_u = ii[starts]
        jj_u = jj[starts]

        # Avoid non-positive weights for stability (matches prior behavior).
        w_sum = np.maximum(w_sum, 1e-6)
        return (
            np.asarray(ii_u, dtype=np.int32),
            np.asarray(jj_u, dtype=np.int32),
            np.asarray(w_sum, dtype=np.float64),
        )

    def _smooth_uv_laplacian(
        self,
        uv: np.ndarray,
        edge_i: np.ndarray,
        edge_j: np.ndarray,
        edge_w: np.ndarray,
        *,
        iterations: int = 3,
        strength: float = 0.15,
        fixed_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        """Lightweight Laplacian smoothing on UV to reduce facet-like noise."""
        uv = np.asarray(uv, dtype=np.float64)
        if uv.ndim != 2 or uv.shape[0] == 0:
            return uv
        iters = int(max(0, iterations))
        if iters <= 0:
            return uv
        w = float(strength)
        if not np.isfinite(w) or w <= 0.0:
            return uv
        w = min(0.95, w)

        n = int(uv.shape[0])
        edge_i = np.asarray(edge_i, dtype=np.int32).reshape(-1)
        edge_j = np.asarray(edge_j, dtype=np.int32).reshape(-1)
        edge_w = np.asarray(edge_w, dtype=np.float64).reshape(-1)
        if edge_i.size == 0:
            return uv

        m = int(min(edge_i.size, edge_j.size, edge_w.size))
        ei = edge_i[:m]
        ej = edge_j[:m]
        ew = edge_w[:m]

        fixed_mask = np.zeros((n,), dtype=bool)
        if fixed_indices is not None:
            idx = np.asarray(fixed_indices, dtype=np.int32).reshape(-1)
            valid = (idx >= 0) & (idx < n)
            fixed_mask[idx[valid]] = True

        out = uv.copy()
        for _ in range(iters):
            sum_w = np.zeros((n,), dtype=np.float64)
            sum_uv = np.zeros((n, 2), dtype=np.float64)
            np.add.at(sum_w, ei, ew)
            np.add.at(sum_w, ej, ew)
            np.add.at(sum_uv, ei, ew[:, None] * out[ej])
            np.add.at(sum_uv, ej, ew[:, None] * out[ei])

            avg = np.zeros_like(out)
            mask = sum_w > 1e-12
            if np.any(mask):
                avg[mask] = sum_uv[mask] / sum_w[mask][:, None]
            new_uv = out.copy()
            movable = ~fixed_mask
            new_uv[movable] = (1.0 - w) * out[movable] + w * avg[movable]
            out = new_uv

        return out
    
    def _build_laplacian(
        self,
        n: int,
        edge_i: np.ndarray,
        edge_j: np.ndarray,
        edge_w: np.ndarray,
    ) -> sparse.csr_matrix:
        """Cotangent-weight Laplacian (symmetric)."""
        n = int(n)
        edge_i = np.asarray(edge_i, dtype=np.int32).reshape(-1)
        edge_j = np.asarray(edge_j, dtype=np.int32).reshape(-1)
        edge_w = np.asarray(edge_w, dtype=np.float64).reshape(-1)
        if n <= 0 or edge_i.size == 0:
            return sparse.csr_matrix((n, n), dtype=np.float64)

        m = int(min(edge_i.size, edge_j.size, edge_w.size))
        edge_i = edge_i[:m]
        edge_j = edge_j[:m]
        edge_w = edge_w[:m]

        rows = np.concatenate([edge_i, edge_j, edge_i, edge_j])
        cols = np.concatenate([edge_j, edge_i, edge_i, edge_j])
        vals = np.concatenate([-edge_w, -edge_w, edge_w, edge_w])

        return sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    
    def _compute_local_rotations(
        self,
        vertices: np.ndarray,
        uv: np.ndarray,
        *,
        edge_i: np.ndarray,
        edge_j: np.ndarray,
        edge_w: np.ndarray,
        edge_vec_3d: np.ndarray,
        reference_basis: np.ndarray,
    ) -> np.ndarray:
        """
        각 정점에 대한 최적 로컬 회전(R_i: 2x3) 계산
        - SVD(Procrustes)로 row-orthonormal 2x3 회전 추정
        """
        vertices = np.asarray(vertices, dtype=np.float64)
        uv = np.asarray(uv, dtype=np.float64)
        n = int(vertices.shape[0]) if vertices.ndim == 2 else 0
        if n <= 0:
            return np.zeros((0, 2, 3), dtype=np.float64)

        edge_i = np.asarray(edge_i, dtype=np.int32).reshape(-1)
        edge_j = np.asarray(edge_j, dtype=np.int32).reshape(-1)
        edge_w = np.asarray(edge_w, dtype=np.float64).reshape(-1)
        edge_vec_3d = np.asarray(edge_vec_3d, dtype=np.float64).reshape(-1, 3)

        default_R = np.asarray(reference_basis, dtype=np.float64).reshape(2, 3)
        if edge_i.size == 0:
            return np.tile(default_R.reshape(1, 2, 3), (n, 1, 1))

        m = int(min(edge_i.size, edge_j.size, edge_w.size, edge_vec_3d.shape[0]))
        ei = edge_i[:m]
        ej = edge_j[:m]
        w = edge_w[:m]
        e3 = edge_vec_3d[:m]

        e_uv = uv[ej, :2] - uv[ei, :2]  # (m,2)
        w0 = (w * e_uv[:, 0]).astype(np.float64, copy=False)  # (m,)
        w1 = (w * e_uv[:, 1]).astype(np.float64, copy=False)  # (m,)

        c0 = (w0[:, None] * e3).astype(np.float64, copy=False)  # (m,3)
        c1 = (w1[:, None] * e3).astype(np.float64, copy=False)  # (m,3)
        contrib = np.concatenate([c0, c1], axis=1)  # (m,6)

        s_flat = np.zeros((n, 6), dtype=np.float64)
        np.add.at(s_flat, ei, contrib)
        np.add.at(s_flat, ej, contrib)
        s_mat = s_flat.reshape(n, 2, 3)

        try:
            U, _s, Vt = np.linalg.svd(s_mat, full_matrices=True)
        except np.linalg.LinAlgError:
            return np.tile(default_R.reshape(1, 2, 3), (n, 1, 1))

        M = np.zeros((2, 3), dtype=np.float64)
        M[0, 0] = 1.0
        M[1, 1] = 1.0
        rotations = (U @ M) @ Vt

        # Isolated vertices: use the reference basis.
        deg = np.bincount(np.concatenate([ei, ej]).astype(np.int64, copy=False), minlength=n)
        rotations[deg == 0] = default_R

        return rotations
    
    def _compute_global_rhs(
        self,
        rotations: np.ndarray,
        *,
        edge_i: np.ndarray,
        edge_j: np.ndarray,
        edge_w: np.ndarray,
        edge_vec_3d: np.ndarray,
        n_vertices: int,
    ) -> np.ndarray:
        """Global step RHS b (n,2): sum w_ij * 0.5*(R_i + R_j) @ (p_j - p_i)."""
        n = int(n_vertices)
        b = np.zeros((n, 2), dtype=np.float64)
        if n <= 0:
            return b

        edge_i = np.asarray(edge_i, dtype=np.int32).reshape(-1)
        edge_j = np.asarray(edge_j, dtype=np.int32).reshape(-1)
        edge_w = np.asarray(edge_w, dtype=np.float64).reshape(-1)
        edge_vec_3d = np.asarray(edge_vec_3d, dtype=np.float64).reshape(-1, 3)
        rotations = np.asarray(rotations, dtype=np.float64)

        if edge_i.size == 0:
            return b

        m = int(min(edge_i.size, edge_j.size, edge_w.size, edge_vec_3d.shape[0]))
        ei = edge_i[:m]
        ej = edge_j[:m]
        w = edge_w[:m]
        e3 = edge_vec_3d[:m]

        R_avg = 0.5 * (rotations[ei] + rotations[ej])  # (m,2,3)
        rotated_edge = np.einsum("mij,mj->mi", R_avg, e3)  # (m,2)
        contrib = (w[:, None] * rotated_edge).astype(np.float64, copy=False)
        np.add.at(b, ei, -contrib)
        np.add.at(b, ej, contrib)
        return b
    
    def _compute_arap_energy(
        self,
        uv: np.ndarray,
        rotations: np.ndarray,
        *,
        edge_i: np.ndarray,
        edge_j: np.ndarray,
        edge_w: np.ndarray,
        edge_vec_3d: np.ndarray,
    ) -> float:
        """ARAP 에너지 계산"""
        uv = np.asarray(uv, dtype=np.float64)
        rotations = np.asarray(rotations, dtype=np.float64)

        edge_i = np.asarray(edge_i, dtype=np.int32).reshape(-1)
        edge_j = np.asarray(edge_j, dtype=np.int32).reshape(-1)
        edge_w = np.asarray(edge_w, dtype=np.float64).reshape(-1)
        edge_vec_3d = np.asarray(edge_vec_3d, dtype=np.float64).reshape(-1, 3)
        if edge_i.size == 0:
            return 0.0

        m = int(min(edge_i.size, edge_j.size, edge_w.size, edge_vec_3d.shape[0]))
        ei = edge_i[:m]
        ej = edge_j[:m]
        w = edge_w[:m]
        e3 = edge_vec_3d[:m]

        e_uv = uv[ej, :2] - uv[ei, :2]
        pred = np.einsum("mij,mj->mi", rotations[ei], e3)
        diff = e_uv - pred
        energy = float(np.sum(w * np.einsum("mi,mi->m", diff, diff)))
        return energy
    
    def _compute_scale(self, mesh: MeshData, uv: np.ndarray) -> float:
        """원본 표면적을 보존하는 스케일 계산"""
        area_3d = float(getattr(mesh, "surface_area", 0.0) or 0.0)
        if not np.isfinite(area_3d) or area_3d <= 1e-12:
            return 1.0

        faces = np.asarray(mesh.faces, dtype=np.int32)
        uv = np.asarray(uv, dtype=np.float64)
        if faces.ndim != 2 or faces.shape[0] == 0 or faces.shape[1] < 3 or uv.ndim != 2 or uv.shape[0] == 0:
            return 1.0

        f = faces[:, :3].astype(np.int32, copy=False)
        try:
            tri = uv[f][:, :, :2].astype(np.float64, copy=False)
        except Exception:
            return 1.0

        e1 = tri[:, 1, :] - tri[:, 0, :]
        e2 = tri[:, 2, :] - tri[:, 0, :]
        area2 = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
        area_2d = float(np.sum(area2))
        if not np.isfinite(area_2d) or area_2d <= 1e-10:
            return 1.0

        return float(np.sqrt(area_3d / area_2d))
    
    def _compute_distortion(self, mesh: MeshData, uv: np.ndarray) -> np.ndarray:
        """각 면의 왜곡도 계산 (0 = 왜곡 없음)"""
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        uv = np.asarray(uv, dtype=np.float64)

        if faces.ndim != 2 or faces.shape[0] == 0 or faces.shape[1] < 3:
            return np.zeros((0,), dtype=np.float64)

        f = faces[:, :3].astype(np.int32, copy=False)
        n_faces = int(f.shape[0])
        distortions = np.ones((n_faces,), dtype=np.float64)

        try:
            v0 = vertices[f[:, 0]]
            v1 = vertices[f[:, 1]]
            v2 = vertices[f[:, 2]]
            uv2 = uv[:, :2]
            t0 = uv2[f[:, 0]]
            t1 = uv2[f[:, 1]]
            t2 = uv2[f[:, 2]]
        except Exception:
            return distortions

        e1_3d = v1 - v0
        e2_3d = v2 - v0
        area_3d = 0.5 * np.linalg.norm(np.cross(e1_3d, e2_3d), axis=1)

        e1_2d = t1 - t0
        e2_2d = t2 - t0
        cross2 = e1_2d[:, 0] * e2_2d[:, 1] - e1_2d[:, 1] * e2_2d[:, 0]
        area_2d = 0.5 * np.abs(cross2)

        valid_area = np.isfinite(area_3d) & np.isfinite(area_2d) & (area_3d > 1e-10) & (area_2d > 1e-10)
        if not np.any(valid_area):
            return distortions

        a3 = area_3d[valid_area]
        a2 = area_2d[valid_area]
        area_ratio = np.minimum(a2 / a3, a3 / a2)
        area_distortion = 1.0 - area_ratio

        len_e1_3d = np.linalg.norm(e1_3d, axis=1)
        len_e2_3d = np.linalg.norm(e2_3d, axis=1)
        len_e1_2d = np.linalg.norm(e1_2d, axis=1)
        len_e2_2d = np.linalg.norm(e2_2d, axis=1)

        stretch_distortion = np.ones((n_faces,), dtype=np.float64)
        valid_len = (
            (len_e1_3d > 1e-10)
            & (len_e2_3d > 1e-10)
            & (len_e1_2d > 1e-10)
            & (len_e2_2d > 1e-10)
            & np.isfinite(len_e1_3d)
            & np.isfinite(len_e2_3d)
            & np.isfinite(len_e1_2d)
            & np.isfinite(len_e2_2d)
        )
        if np.any(valid_len):
            r1 = len_e1_2d[valid_len] / len_e1_3d[valid_len]
            r2 = len_e2_2d[valid_len] / len_e2_3d[valid_len]
            ratio1 = np.minimum(r1, 1.0 / r1)
            ratio2 = np.minimum(r2, 1.0 / r2)
            stretch_distortion[valid_len] = 1.0 - 0.5 * (ratio1 + ratio2)

        distortions[valid_area] = 0.5 * (area_distortion + stretch_distortion[valid_area])
        return np.clip(distortions, 0.0, 1.0)


def _similarity_align_2d(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Similarity-align `source` to `target` (2D) using least-squares Procrustes.

    Returns:
        Aligned source with same shape as input.
    """
    src = np.asarray(source, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    if src.ndim != 2 or tgt.ndim != 2 or src.shape[0] != tgt.shape[0] or src.shape[1] < 2 or tgt.shape[1] < 2:
        return src[:, :2].copy() if src.ndim == 2 and src.shape[1] >= 2 else np.zeros((0, 2), dtype=np.float64)

    src2 = src[:, :2].copy()
    tgt2 = tgt[:, :2].copy()
    finite = np.all(np.isfinite(src2), axis=1) & np.all(np.isfinite(tgt2), axis=1)
    if int(finite.sum()) < 2:
        return src2

    a = src2[finite]
    b = tgt2[finite]
    a_mean = a.mean(axis=0)
    b_mean = b.mean(axis=0)
    a0 = a - a_mean
    b0 = b - b_mean

    denom = float(np.sum(a0 * a0))
    if not np.isfinite(denom) or denom < 1e-12:
        out = src2
        out[finite] = b
        return out

    h = a0.T @ b0
    try:
        u, s, vt = np.linalg.svd(h)
    except Exception:
        return src2

    r = u @ vt
    if float(np.linalg.det(r)) < 0:
        u[:, -1] *= -1.0
        r = u @ vt

    scale = float(np.sum(s)) / denom
    if not np.isfinite(scale) or abs(scale) < 1e-12:
        scale = 1.0

    aligned = (src2 - a_mean) @ r.T
    aligned *= scale
    aligned += b_mean
    return aligned


def _mesh_total_area_3d(mesh: MeshData) -> float:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[0] == 0 or faces.ndim != 2 or faces.shape[0] == 0:
        return 0.0

    tri = vertices[faces[:, :3]]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    a = np.linalg.norm(np.cross(e1, e2), axis=1) * 0.5
    a = a[np.isfinite(a)]
    return float(a.sum()) if a.size else 0.0


def _mesh_total_area_2d(mesh: MeshData, uv: np.ndarray) -> float:
    pts = np.asarray(uv, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if pts.ndim != 2 or pts.shape[0] == 0 or faces.ndim != 2 or faces.shape[0] == 0:
        return 0.0

    tri = pts[faces[:, :3], :2]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    cross = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    a = np.abs(cross) * 0.5
    a = a[np.isfinite(a)]
    return float(a.sum()) if a.size else 0.0


def _normalize_cylinder_axis_choice(choice: str) -> str:
    c = str(choice or "").strip().lower()
    if c in {"x", "x축", "x축 기준", "x axis"}:
        return "x"
    if c in {"y", "y축", "y축 기준", "y axis"}:
        return "y"
    if c in {"z", "z축", "z축 기준", "z axis"}:
        return "z"
    if c in {"auto", "자동", "자동 감지", "automatic"}:
        return "auto"
    return "auto"


def _axis_unit_vector(axis: str) -> np.ndarray:
    a = _normalize_cylinder_axis_choice(axis)
    if a == "x":
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if a == "y":
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return np.array([0.0, 0.0, 1.0], dtype=np.float64)


def _choose_best_axis_xyz(vertices: np.ndarray) -> str:
    v = np.asarray(vertices, dtype=np.float64)
    if v.ndim != 2 or v.shape[0] < 4 or v.shape[1] < 3:
        return "z"

    best_axis = "z"
    best_score = float("inf")
    for ax in ("x", "y", "z"):
        axis = _axis_unit_vector(ax)
        t = v @ axis
        perp = v - t[:, None] * axis[None, :]
        center = perp.mean(axis=0)
        r = np.linalg.norm(perp - center[None, :], axis=1)
        r = r[np.isfinite(r)]
        if r.size < 4:
            continue
        med = float(np.median(r))
        if not np.isfinite(med) or med < 1e-9:
            continue
        score = float(np.std(r) / med)
        if score < best_score:
            best_score = score
            best_axis = ax

    return best_axis


def _angles_to_min_range(
    angles: np.ndarray, *, seam_hint: float | None = None
) -> tuple[np.ndarray, float, float]:
    """
    Shift angles by a seam angle to minimize the covered range, then wrap to [0, 2pi).

    Returns:
        (angles_wrapped, seam_angle, span)
    """
    a = np.asarray(angles, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    if a.size < 2:
        seam = 0.0
        out = np.zeros_like(np.asarray(angles, dtype=np.float64).reshape(-1))
        return out, seam, 0.0

    if seam_hint is not None:
        try:
            seam_val = float(seam_hint)
        except Exception:
            seam_val = None
        if seam_val is not None and np.isfinite(seam_val):
            raw = np.asarray(angles, dtype=np.float64).reshape(-1) - seam_val
            wrapped = np.mod(raw, 2.0 * np.pi)
            span = float(np.nanmax(wrapped) - np.nanmin(wrapped)) if wrapped.size else 0.0
            return wrapped, float(seam_val), span

    a_sorted = np.sort(a)
    diffs = np.diff(a_sorted)
    wrap_gap = (a_sorted[0] + 2.0 * np.pi) - a_sorted[-1]
    gaps = np.concatenate([diffs, [wrap_gap]])
    k = int(np.argmax(gaps))

    if k == a_sorted.size - 1:
        start = float(a_sorted[-1])
        gap = float(wrap_gap)
        seam = start + gap * 0.5
    else:
        start = float(a_sorted[k])
        end = float(a_sorted[k + 1])
        seam = (start + end) * 0.5

    # Shift and wrap to [0, 2pi)
    raw = np.asarray(angles, dtype=np.float64).reshape(-1) - seam
    wrapped = np.mod(raw, 2.0 * np.pi)
    span = float(np.nanmax(wrapped) - np.nanmin(wrapped)) if wrapped.size else 0.0
    return wrapped, float(seam), span


def _flatten_cut_lines_points(
    cut_lines_world: list[list[list[float]]] | None,
) -> np.ndarray:
    if not cut_lines_world:
        return np.zeros((0, 3), dtype=np.float64)
    pts: list[np.ndarray] = []
    for line in cut_lines_world:
        if line is None:
            continue
        arr = np.asarray(line, dtype=np.float64).reshape(-1, 3)
        if arr.size:
            pts.append(arr)
    if not pts:
        return np.zeros((0, 3), dtype=np.float64)
    out = np.concatenate(pts, axis=0)
    if not np.isfinite(out).all():
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _seam_hint_from_cut_lines(
    cut_lines_world: list[list[list[float]]] | None,
    *,
    axis: np.ndarray,
    b1: np.ndarray,
    b2: np.ndarray,
    center: np.ndarray,
) -> float | None:
    pts = _flatten_cut_lines_points(cut_lines_world)
    if pts.size == 0:
        return None
    try:
        axis = np.asarray(axis, dtype=np.float64).reshape(3)
        b1 = np.asarray(b1, dtype=np.float64).reshape(3)
        b2 = np.asarray(b2, dtype=np.float64).reshape(3)
        center = np.asarray(center, dtype=np.float64).reshape(3)
    except Exception:
        return None

    t = pts @ axis
    perp = pts - t[:, None] * axis[None, :]
    r_vec = perp - center[None, :]
    x = r_vec @ b1
    y = r_vec @ b2
    theta = np.arctan2(y, x)
    if theta.size == 0:
        return None
    s = np.sin(theta)
    c = np.cos(theta)
    s_sum = float(np.sum(s))
    c_sum = float(np.sum(c))
    if not np.isfinite(s_sum) or not np.isfinite(c_sum):
        return None
    if abs(s_sum) < 1e-9 and abs(c_sum) < 1e-9:
        return None
    return float(np.arctan2(s_sum, c_sum))


def cylindrical_parameterization(
    mesh: MeshData,
    *,
    axis: str = "auto",
    radius: float | None = None,
    cut_lines_world: list[list[list[float]]] | None = None,
    seam_hint: float | None = None,
    return_meta: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """
    Simple cylindrical unwrapping (developable approximation).

    Args:
        mesh: MeshData
        axis: 'auto' | 'x' | 'y' | 'z' (Korean UI strings are also accepted)
        radius: cylinder radius in mesh/world units. If None, estimate from vertices.

    Returns:
        (N, 2) UV in mesh/world units.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[0] == 0 or vertices.shape[1] < 3:
        return np.zeros((0, 2), dtype=np.float64)

    axis_choice = _normalize_cylinder_axis_choice(axis)
    if axis_choice == "auto":
        axis_choice = _choose_best_axis_xyz(vertices)

    a = _axis_unit_vector(axis_choice)
    # Orthonormal basis (b1, b2) spanning plane perpendicular to axis a
    temp = np.array([1.0, 0.0, 0.0], dtype=np.float64) if abs(float(a[0])) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float64)
    b1 = np.cross(a, temp)
    b1_norm = float(np.linalg.norm(b1))
    if not np.isfinite(b1_norm) or b1_norm < 1e-12:
        b1 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        b1_norm = float(np.linalg.norm(b1))
    b1 /= b1_norm
    b2 = np.cross(a, b1)

    # Project to axis and perpendicular plane
    t = vertices @ a  # (N,)
    perp = vertices - t[:, None] * a[None, :]
    center = perp.mean(axis=0)
    r_vec = perp - center[None, :]

    x = r_vec @ b1
    y = r_vec @ b2
    theta = np.arctan2(y, x)  # [-pi, pi]

    seam_from_lines = _seam_hint_from_cut_lines(
        cut_lines_world,
        axis=a,
        b1=b1,
        b2=b2,
        center=center,
    )
    seam_pref = seam_hint if seam_hint is not None else seam_from_lines
    theta_wrapped, seam_angle, span = _angles_to_min_range(theta, seam_hint=seam_pref)

    r = np.linalg.norm(r_vec, axis=1)
    if radius is None:
        r_f = r[np.isfinite(r)]
        radius_val = float(np.median(r_f)) if r_f.size else 1.0
    else:
        try:
            radius_val = float(radius)
        except Exception:
            radius_val = 0.0
        if not np.isfinite(radius_val) or abs(radius_val) < 1e-12:
            r_f = r[np.isfinite(r)]
            radius_val = float(np.median(r_f)) if r_f.size else 1.0

    radius_val = abs(float(radius_val))
    if radius_val < 1e-6:
        radius_val = 1.0

    u = theta_wrapped * radius_val
    v = t.astype(np.float64, copy=False)

    uv = np.stack([u, v], axis=1)
    if not np.isfinite(uv).all():
        uv = np.nan_to_num(uv, nan=0.0, posinf=0.0, neginf=0.0)
    if bool(return_meta):
        meta: dict[str, Any] = {
            "cylinder_axis_choice": str(axis_choice),
            "cylinder_axis": np.asarray(a, dtype=np.float64).reshape(3),
            "cylinder_center": np.asarray(center, dtype=np.float64).reshape(3),
            "cylinder_radius": float(radius_val),
            "cylinder_seam_angle": float(seam_angle),
            "cylinder_span": float(span),
        }
        return uv, meta
    return uv


def flatten_with_method(
    mesh: MeshData,
    *,
    method: str = "arap",
    iterations: int = 30,
    distortion: float = 0.5,
    boundary_type: str = "free",
    initial_method: str = "lscm",
    cylinder_axis: str = "auto",
    cylinder_radius: float | None = None,
    cut_lines_world: list[list[list[float]]] | None = None,
    smooth_iters: int | None = None,
    smooth_strength: float = 0.15,
    pack_compact: bool = True,
) -> FlattenedMesh:
    """
    Convenience wrapper to flatten a mesh with selectable models.

    Supported methods:
      - 'arap': shape-preserving (LSCM init + ARAP optimize)
      - 'lscm': conformal (angle-preserving) init only
      - 'area': area-prioritized blend (Tutte ↔ LSCM), controlled by `distortion`
      - 'cylinder': cylindrical unwrapping

    Notes:
      - `distortion` in [0..1]: 0=area priority, 1=angle priority (for method='area').
      - `cylinder_radius` is in mesh/world units; if None, it is estimated.
      - `cut_lines_world` can align cylindrical seams to user cut lines.
      - `smooth_iters`/`smooth_strength` apply lightweight UV smoothing.
      - `pack_compact` selects compact packing for multi-component ARAP.
    """
    raw_method = str(method or "arap").strip()
    m_text = raw_method.lower()
    if "arap" in m_text:
        m = "arap"
    elif "lscm" in m_text:
        m = "lscm"
    elif ("area" in m_text) or ("면적" in raw_method):
        m = "area"
    elif ("cyl" in m_text) or ("원통" in raw_method):
        m = "cylinder"
    else:
        m = m_text
    iters = int(iterations)
    if iters < 0:
        iters = 0

    flattener = ARAPFlattener(max_iterations=max(1, iters) if m == "arap" else 0)

    mesh_s = flattener._sanitize_mesh(mesh)
    if mesh_s.n_vertices == 0 or mesh_s.n_faces == 0:
        return FlattenedMesh(
            uv=np.zeros((mesh_s.n_vertices, 2), dtype=np.float64),
            faces=mesh_s.faces,
            original_mesh=mesh_s,
            distortion_per_face=np.zeros((mesh_s.n_faces,), dtype=np.float64),
            scale=1.0,
        )

    smooth_iters_val: int
    if smooth_iters is None:
        smooth_iters_val = 3 if m == "arap" else 0
    else:
        try:
            smooth_iters_val = int(smooth_iters)
        except Exception:
            smooth_iters_val = 0
        if smooth_iters_val < 0:
            smooth_iters_val = 0

    try:
        smooth_strength_val = float(smooth_strength)
    except Exception:
        smooth_strength_val = 0.0
    if not np.isfinite(smooth_strength_val) or smooth_strength_val < 0.0:
        smooth_strength_val = 0.0

    def _maybe_smooth_uv(uv_in: np.ndarray) -> np.ndarray:
        if smooth_iters_val <= 0 or smooth_strength_val <= 0.0:
            return uv_in
        try:
            edge_i, edge_j, edge_w = flattener._compute_cotangent_edge_weights(mesh_s)
            anchors = flattener._pick_anchor_pair(
                mesh_s, np.arange(mesh_s.n_vertices, dtype=np.int32)
            )
            return flattener._smooth_uv_laplacian(
                uv_in,
                edge_i,
                edge_j,
                edge_w,
                iterations=smooth_iters_val,
                strength=smooth_strength_val,
                fixed_indices=anchors,
            )
        except Exception:
            _log_ignored_exception()
            return uv_in

    if m == "arap":
        init_text = str(initial_method or "lscm")
        init_t = init_text.lower().strip()
        initial_uv = None
        cyl_meta: dict[str, Any] = {}
        if ("cyl" in init_t) or ("원통" in init_text):
            initial_uv_res = cylindrical_parameterization(
                mesh_s,
                axis=cylinder_axis,
                radius=cylinder_radius,
                cut_lines_world=cut_lines_world,
                return_meta=True,
            )
            if isinstance(initial_uv_res, tuple):
                initial_uv, cyl_meta = initial_uv_res
            else:
                initial_uv = initial_uv_res
                cyl_meta = {}
        out = flattener.flatten(
            mesh_s,
            boundary_type=str(boundary_type or "free"),
            initial_method=init_text,
            initial_uv=initial_uv,
            smooth_iters=smooth_iters_val,
            smooth_strength=smooth_strength_val,
            pack_compact=bool(pack_compact),
        )
        try:
            out.meta["flatten_method"] = "arap"
            out.meta["initial_method"] = init_text
            out.meta["iterations"] = int(iters)
            out.meta["smooth_iters"] = int(smooth_iters_val)
            out.meta["smooth_strength"] = float(smooth_strength_val)
            if cyl_meta:
                out.meta.update(cyl_meta)
        except Exception:
            _log_ignored_exception()
        return out

    if m == "lscm":
        uv = flattener._safe_initial_parameterization(mesh_s, "lscm")
        uv = np.asarray(uv, dtype=np.float64)
        if uv.ndim != 2 or uv.shape[0] != mesh_s.n_vertices or uv.shape[1] < 2:
            uv = np.zeros((mesh_s.n_vertices, 2), dtype=np.float64)
        else:
            uv = uv[:, :2].copy()
        uv = flattener._orient_uv_pca(uv)
        uv = _maybe_smooth_uv(uv)
        distortion_arr = flattener._compute_distortion(mesh_s, uv)
        return FlattenedMesh(
            uv=uv,
            faces=mesh_s.faces,
            original_mesh=mesh_s,
            distortion_per_face=distortion_arr,
            scale=1.0,
            meta={"flatten_method": "lscm"},
        )

    if m in {"area", "area_preserve", "area-preserve", "area_preserving"}:
        w = float(distortion)
        if not np.isfinite(w):
            w = 0.5
        w = float(np.clip(w, 0.0, 1.0))

        uv_area = flattener._safe_initial_parameterization(mesh_s, "tutte")
        uv_angle = flattener._safe_initial_parameterization(mesh_s, "lscm")
        uv_area = np.asarray(uv_area, dtype=np.float64)
        uv_angle = np.asarray(uv_angle, dtype=np.float64)
        if uv_area.ndim != 2 or uv_area.shape[0] != mesh_s.n_vertices or uv_area.shape[1] < 2:
            uv_area = np.zeros((mesh_s.n_vertices, 2), dtype=np.float64)
        else:
            uv_area = uv_area[:, :2].copy()
        if uv_angle.ndim != 2 or uv_angle.shape[0] != mesh_s.n_vertices or uv_angle.shape[1] < 2:
            uv_angle = np.zeros((mesh_s.n_vertices, 2), dtype=np.float64)
        else:
            uv_angle = uv_angle[:, :2].copy()

        uv_area_aligned = _similarity_align_2d(uv_area, uv_angle)
        uv = (1.0 - w) * uv_area_aligned + w * uv_angle
        uv = np.asarray(uv, dtype=np.float64)
        uv = flattener._orient_uv_pca(uv)
        uv = _maybe_smooth_uv(uv)

        # Global area scale match (helps keep real-world sizing consistent).
        a3 = _mesh_total_area_3d(mesh_s)
        a2 = _mesh_total_area_2d(mesh_s, uv)
        if a3 > 1e-12 and a2 > 1e-12:
            s = float(np.sqrt(a3 / a2))
            if np.isfinite(s) and s > 1e-9:
                uv *= s

        distortion_arr = flattener._compute_distortion(mesh_s, uv)
        return FlattenedMesh(
            uv=uv,
            faces=mesh_s.faces,
            original_mesh=mesh_s,
            distortion_per_face=distortion_arr,
            scale=1.0,
            meta={"flatten_method": "area", "distortion_weight": float(w)},
        )

    if m in {"cylinder", "cyl", "cylindrical"}:
        uv_res = cylindrical_parameterization(
            mesh_s,
            axis=cylinder_axis,
            radius=cylinder_radius,
            cut_lines_world=cut_lines_world,
            return_meta=True,
        )
        if isinstance(uv_res, tuple):
            uv, cyl_meta = uv_res
        else:
            uv = uv_res
            cyl_meta = {}
        uv = np.asarray(uv, dtype=np.float64)
        if uv.ndim != 2 or uv.shape[0] != mesh_s.n_vertices or uv.shape[1] < 2:
            uv = np.zeros((mesh_s.n_vertices, 2), dtype=np.float64)
        else:
            uv = uv[:, :2].copy()
        uv = flattener._orient_uv_pca(uv)
        uv = _maybe_smooth_uv(uv)
        distortion_arr = flattener._compute_distortion(mesh_s, uv)
        meta = {
            "flatten_method": "cylinder",
            "cylinder_axis_input": str(cylinder_axis),
            "cylinder_radius_input": None if cylinder_radius is None else float(cylinder_radius),
            **(cyl_meta or {}),
        }
        return FlattenedMesh(
            uv=uv,
            faces=mesh_s.faces,
            original_mesh=mesh_s,
            distortion_per_face=distortion_arr,
            scale=1.0,
            meta=meta,
        )

    raise NotImplementedError(f"Unsupported flatten method: {method}")


# 테스트용
if __name__ == '__main__':
    print("ARAP Flattener module loaded successfully")
    print("Use: flattener = ARAPFlattener(); result = flattener.flatten(mesh)")
