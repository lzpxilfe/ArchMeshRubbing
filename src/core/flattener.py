"""
ARAP (As-Rigid-As-Possible) Mesh Flattening Module
메쉬 평면화 알고리즘 - 왜곡을 최소화하며 3D 표면을 2D로 펼침

Based on: "As-Rigid-As-Possible Surface Modeling" (Sorkine & Alexa, 2007)
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .mesh_loader import MeshData


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
    
    # 캐시
    _bounds: Optional[np.ndarray] = field(default=None, repr=False)
    
    def __post_init__(self):
        self.uv = np.asarray(self.uv, dtype=np.float64)
        self.faces = np.asarray(self.faces, dtype=np.int32)
    
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
            self._bounds = np.array([
                self.uv.min(axis=0),
                self.uv.max(axis=0)
            ])
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
            scale=new_scale
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
    
    def flatten(self, mesh: MeshData, 
                boundary_type: str = 'free',
                initial_method: str = 'lscm') -> FlattenedMesh:
        """
        3D 메쉬를 2D로 평면화
        
        Args:
            mesh: 입력 3D 메쉬
            boundary_type: 경계 조건 ('free', 'fixed_circle', 'fixed_rect')
            initial_method: 초기 파라미터화 방법 ('lscm', 'tutte')
            
        Returns:
            FlattenedMesh: 평면화 결과
        """
        # 1. 초기 파라미터화
        if initial_method == 'lscm':
            initial_uv = self._lscm_parameterization(mesh)
        elif initial_method == 'tutte':
            initial_uv = self._tutte_parameterization(mesh)
        else:
            raise ValueError(f"Unknown initial method: {initial_method}")
        
        # 2. ARAP 최적화
        optimized_uv = self._arap_optimize(mesh, initial_uv, boundary_type)
        
        # 3. 스케일 계산 (원본 표면적 보존)
        scale = self._compute_scale(mesh, optimized_uv)
        
        # 4. 왜곡도 계산
        distortion = self._compute_distortion(mesh, optimized_uv)
        
        return FlattenedMesh(
            uv=optimized_uv,
            faces=mesh.faces,
            original_mesh=mesh,
            distortion_per_face=distortion,
            scale=scale
        )
    
    def _lscm_parameterization(self, mesh: MeshData) -> np.ndarray:
        """
        LSCM (Least Squares Conformal Maps) 초기 파라미터화
        각도를 보존하는 등각 매핑
        """
        n = mesh.n_vertices
        m = mesh.n_faces
        
        vertices = mesh.vertices
        faces = mesh.faces
        
        # 경계 정점 찾기
        boundary = mesh.get_boundary_vertices()
        
        if len(boundary) < 2:
            # 닫힌 메쉬: 임의로 두 정점 고정
            boundary = np.array([0, n // 2])
        
        # 두 경계 정점을 고정 (0, 0)과 (1, 0)
        fixed_verts = boundary[:2]
        fixed_pos = np.array([[0.0, 0.0], [1.0, 0.0]])
        
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
        
        # 경계 정점을 원에 배치
        angles = np.linspace(0, 2*np.pi, len(boundary), endpoint=False)
        boundary_uv = np.column_stack([np.cos(angles), np.sin(angles)])
        
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
        ARAP 최적화 반복
        로컬 회전과 글로벌 위치를 번갈아 최적화
        """
        n = mesh.n_vertices

        uv = initial_uv.copy()
        
        # 기준 2D 좌표계 (메쉬가 어떤 방향을 향하든 안정적으로 동작하도록)
        reference_basis = self._compute_reference_basis(mesh)
        
        # 코탄젠트 가중치 계산
        weights = self._compute_cotangent_weights(mesh)
        
        # 경계 정점
        boundary = mesh.get_boundary_vertices()
        is_boundary = np.zeros(n, dtype=bool)
        is_boundary[boundary] = True
        
        # 경계 조건에 따른 고정 정점 설정
        if boundary_type == 'fixed_circle' and len(boundary) >= 3:
            # 경계를 원에 고정
            fixed_mask = is_boundary
        elif boundary_type == 'fixed_rect' and len(boundary) >= 4:
            # 경계를 사각형에 고정
            fixed_mask = is_boundary
        else:
            # 자유 경계: 두 정점만 고정 (회전/이동 방지)
            fixed_mask = np.zeros(n, dtype=bool)
            if len(boundary) >= 2:
                fixed_mask[boundary[0]] = True
                fixed_mask[boundary[len(boundary)//2]] = True
            else:
                fixed_mask[0] = True
                fixed_mask[n//2] = True
        
        fixed_indices = np.where(fixed_mask)[0]
        free_indices = np.where(~fixed_mask)[0]
        
        # ARAP 반복
        prev_energy = float('inf')
        
        for iteration in range(self.max_iterations):
            # Step 1: 로컬 회전 최적화
            rotations = self._compute_local_rotations(mesh, uv, weights, reference_basis)
            
            # Step 2: 글로벌 위치 최적화
            uv = self._solve_global_step(mesh, uv, rotations, weights, 
                                          fixed_indices, free_indices, reference_basis)
            
            # 에너지 계산 및 수렴 확인
            energy = self._compute_arap_energy(mesh, uv, rotations, weights, reference_basis)
            
            if abs(prev_energy - energy) < self.tolerance:
                break
            
            prev_energy = energy
        
        return uv

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
    
    def _compute_cotangent_weights(self, mesh: MeshData) -> Dict[Tuple[int, int], float]:
        """코탄젠트 가중치 계산"""
        vertices = mesh.vertices
        faces = mesh.faces
        
        weights = {}
        
        for face in faces:
            for i in range(3):
                vi = face[i]
                vj = face[(i + 1) % 3]
                vk = face[(i + 2) % 3]
                
                # vk에서의 각도에 대한 코탄젠트
                e1 = vertices[vi] - vertices[vk]
                e2 = vertices[vj] - vertices[vk]
                
                cos_angle = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-10)
                cos_angle = np.clip(cos_angle, -1, 1)
                sin_angle = np.sqrt(1 - cos_angle**2) + 1e-10
                cot = cos_angle / sin_angle
                
                edge = tuple(sorted([vi, vj]))
                weights[edge] = weights.get(edge, 0) + cot * 0.5
        
        # 음수 가중치 방지
        for edge in weights:
            weights[edge] = max(weights[edge], 1e-6)
        
        return weights
    
    def _build_laplacian(self, mesh: MeshData,
                         weights: Dict[Tuple[int, int], float]) -> sparse.csr_matrix:
        """코탄젠트 가중치 라플라시안 행렬"""
        n = mesh.n_vertices
        
        rows, cols, vals = [], [], []
        
        for (i, j), w in weights.items():
            rows.extend([i, j, i, j])
            cols.extend([j, i, i, j])
            vals.extend([-w, -w, w, w])
        
        coo = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n))
        return sparse.csr_matrix(coo)
    
    def _compute_local_rotations(self, mesh: MeshData, uv: np.ndarray,
                                  weights: Dict[Tuple[int, int], float],
                                  reference_basis: np.ndarray) -> np.ndarray:
        """
        각 정점에 대한 최적 로컬 회전 계산
        SVD를 사용한 Procrustes 분석
        """
        n = mesh.n_vertices
        vertices = mesh.vertices
        
        rotations = np.zeros((n, 2, 2))
        
        # 각 정점의 이웃 정보
        neighbors = [[] for _ in range(n)]
        for (i, j), w in weights.items():
            neighbors[i].append((j, w))
            neighbors[j].append((i, w))
        
        for i in range(n):
            if len(neighbors[i]) == 0:
                rotations[i] = np.eye(2)
                continue
            
            # 공분산 행렬 계산
            S = np.zeros((2, 2))
            
            for j, w in neighbors[i]:
                # 원본 3D 엣지 (xy 평면에 투영)
                e_3d = vertices[j] - vertices[i]
                e_orig = reference_basis @ e_3d
                
                # 현재 2D 엣지
                e_curr = uv[j] - uv[i]
                
                S += w * np.outer(e_curr, e_orig)
            
            # SVD로 최적 회전 추출
            U, _, Vt = np.linalg.svd(S)
            R = U @ Vt
            
            # 반사 방지
            if np.linalg.det(R) < 0:
                U[:, -1] *= -1
                R = U @ Vt
            
            rotations[i] = R
        
        return rotations
    
    def _solve_global_step(self, mesh: MeshData, uv: np.ndarray,
                           rotations: np.ndarray,
                           weights: Dict[Tuple[int, int], float],
                           fixed_indices: np.ndarray,
                           free_indices: np.ndarray,
                           reference_basis: np.ndarray) -> np.ndarray:
        """글로벌 위치 최적화 단계"""
        n = mesh.n_vertices
        vertices = mesh.vertices
        
        # 우변 계산
        b = np.zeros((n, 2))
        
        for (i, j), w in weights.items():
            e_3d = vertices[j] - vertices[i]
            e_orig = reference_basis @ e_3d
            
            # 두 정점의 회전 평균 적용
            R_avg = (rotations[i] + rotations[j]) / 2
            rotated_edge = R_avg @ e_orig
            
            b[i] -= w * rotated_edge
            b[j] += w * rotated_edge
        
        # 라플라시안 행렬
        L = self._build_laplacian(mesh, weights)
        
        # 고정 정점 처리
        if len(fixed_indices) > 0:
            # 고정 정점 값을 우변으로 이동
            for idx in fixed_indices:
                b -= L[:, idx].toarray().flatten()[:, np.newaxis] * uv[idx]
            
            # 시스템 축소
            L_reduced = L[np.ix_(free_indices, free_indices)]
            b_reduced = b[free_indices]
            
            # 정규화 추가
            L_reduced += sparse.eye(len(free_indices)) * 1e-8
            
            # 풀이
            new_uv = uv.copy()
            new_uv[free_indices, 0] = np.asarray(spsolve(L_reduced.tocsr(), b_reduced[:, 0])).ravel()
            new_uv[free_indices, 1] = np.asarray(spsolve(L_reduced.tocsr(), b_reduced[:, 1])).ravel()

            return new_uv
        else:
            # 모든 정점이 자유 (원점 고정)
            L += sparse.eye(n) * 1e-8
            new_uv = np.zeros((n, 2))
            new_uv[:, 0] = np.asarray(spsolve(L.tocsr(), b[:, 0])).ravel()
            new_uv[:, 1] = np.asarray(spsolve(L.tocsr(), b[:, 1])).ravel()
            return new_uv
    
    def _compute_arap_energy(self, mesh: MeshData, uv: np.ndarray,
                             rotations: np.ndarray,
                             weights: Dict[Tuple[int, int], float],
                             reference_basis: np.ndarray) -> float:
        """ARAP 에너지 계산"""
        vertices = mesh.vertices
        energy = 0.0
        
        for (i, j), w in weights.items():
            e_3d = vertices[j] - vertices[i]
            e_orig = reference_basis @ e_3d
            e_curr = uv[j] - uv[i]
            
            R = rotations[i]
            diff = e_curr - R @ e_orig
            energy += w * np.dot(diff, diff)
        
        return energy
    
    def _compute_scale(self, mesh: MeshData, uv: np.ndarray) -> float:
        """원본 표면적을 보존하는 스케일 계산"""
        # 3D 표면적
        area_3d = mesh.surface_area
        
        # 2D 면적
        area_2d = 0.0
        for face in mesh.faces:
            v0, v1, v2 = uv[face]
            # 2D 삼각형 면적 (외적의 z 성분)
            area_2d += abs(np.cross(v1 - v0, v2 - v0)) / 2
        
        if area_2d < 1e-10:
            return 1.0

        # 스케일 = sqrt(3D 면적 / 2D 면적)
        return float(np.sqrt(area_3d / area_2d))
    
    def _compute_distortion(self, mesh: MeshData, uv: np.ndarray) -> np.ndarray:
        """각 면의 왜곡도 계산 (0 = 왜곡 없음)"""
        vertices = mesh.vertices
        faces = mesh.faces
        
        distortions = np.zeros(len(faces))
        
        for fi, face in enumerate(faces):
            # 3D 삼각형
            v0_3d, v1_3d, v2_3d = vertices[face]
            e1_3d = v1_3d - v0_3d
            e2_3d = v2_3d - v0_3d
            
            # 3D 면적
            area_3d = np.linalg.norm(np.cross(e1_3d, e2_3d)) / 2
            
            # 2D 삼각형
            v0_2d, v1_2d, v2_2d = uv[face]
            e1_2d = v1_2d - v0_2d
            e2_2d = v2_2d - v0_2d
            
            # 2D 면적
            area_2d = abs(np.cross(e1_2d, e2_2d)) / 2
            
            if area_3d < 1e-10 or area_2d < 1e-10:
                distortions[fi] = 1.0
                continue
            
            # 면적 비율 왜곡
            area_ratio = area_2d / area_3d if area_3d > area_2d else area_3d / area_2d
            area_distortion = 1.0 - area_ratio
            
            # 각도 왜곡 (간단한 근사)
            # 원본 엣지 길이
            len_e1_3d = np.linalg.norm(e1_3d)
            len_e2_3d = np.linalg.norm(e2_3d)
            
            # 2D 엣지 길이
            len_e1_2d = np.linalg.norm(e1_2d)
            len_e2_2d = np.linalg.norm(e2_2d)
            
            # 길이 비율
            if len_e1_3d > 1e-10 and len_e2_3d > 1e-10:
                ratio1 = len_e1_2d / len_e1_3d if len_e1_3d > len_e1_2d else len_e1_3d / len_e1_2d
                ratio2 = len_e2_2d / len_e2_3d if len_e2_3d > len_e2_2d else len_e2_3d / len_e2_2d
                stretch_distortion = 1.0 - (ratio1 + ratio2) / 2
            else:
                stretch_distortion = 1.0
            
            # 종합 왜곡도
            distortions[fi] = (area_distortion + stretch_distortion) / 2
        
        return np.clip(distortions, 0, 1)


# 테스트용
if __name__ == '__main__':
    print("ARAP Flattener module loaded successfully")
    print("Use: flattener = ARAPFlattener(); result = flattener.flatten(mesh)")
