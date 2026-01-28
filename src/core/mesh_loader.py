"""
Mesh Loader Module
메쉬 파일 로딩 및 데이터 구조 정의

Supports: OBJ, PLY, STL, OFF formats
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Union
import numpy as np

try:
    import trimesh
except ImportError:
    raise ImportError("trimesh is required. Install with: pip install trimesh")


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
    
    # Computed properties cache
    _bounds: Optional[np.ndarray] = field(default=None, repr=False)
    _centroid: Optional[np.ndarray] = field(default=None, repr=False)
    _surface_area: Optional[float] = field(default=None, repr=False)
    
    def __post_init__(self):
        """데이터 검증 및 타입 변환"""
        self.vertices = np.asarray(self.vertices, dtype=np.float64)
        self.faces = np.asarray(self.faces, dtype=np.int32)
        
        if self.normals is not None:
            self.normals = np.asarray(self.normals, dtype=np.float64)
        if self.face_normals is not None:
            self.face_normals = np.asarray(self.face_normals, dtype=np.float32)
        if self.uv_coords is not None:
            self.uv_coords = np.asarray(self.uv_coords, dtype=np.float64)
    
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

        if self.face_normals is None:
            v0 = self.vertices[self.faces[:, 0]]
            v1 = self.vertices[self.faces[:, 1]]
            v2 = self.vertices[self.faces[:, 2]]
            
            cross = np.cross(v1 - v0, v2 - v0)
            norms = np.linalg.norm(cross, axis=1, keepdims=True)
            norms[norms == 0] = 1  # 0으로 나누기 방지
            self.face_normals = (cross / norms).astype(np.float32, copy=False)
        
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
            norms[norms == 0] = 1
            self.normals = self.normals / norms
    
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
            filepath=self.filepath
        )
    
    def to_trimesh(self) -> 'trimesh.Trimesh':
        """trimesh 객체로 변환"""
        mesh = trimesh.Trimesh(
            vertices=self.vertices,
            faces=self.faces,
            vertex_normals=self.normals,
            process=False
        )
        return mesh
    
    @classmethod
    def from_trimesh(cls, mesh: 'trimesh.Trimesh',
                     filepath: Optional[Path] = None,
                     unit: str = 'mm') -> 'MeshData':
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
            filepath=filepath
        )
    
    def extract_submesh(self, face_indices: np.ndarray) -> 'MeshData':
        """선택된 면으로 서브메쉬 추출"""
        face_indices = np.asarray(face_indices, dtype=np.int32)
        
        # 선택된 면들
        selected_faces = self.faces[face_indices]
        
        # 사용된 정점 인덱스
        unique_verts = np.unique(selected_faces.flatten())
        
        # 새 인덱스 매핑
        vert_map = {old: new for new, old in enumerate(unique_verts)}
        new_faces = np.array([[vert_map[v] for v in face] for face in selected_faces])
        
        # 새 정점
        new_vertices = self.vertices[unique_verts]
        
        # 새 법선 (있는 경우)
        new_normals = self.normals[unique_verts] if self.normals is not None else None
        new_face_normals = self.face_normals[face_indices] if self.face_normals is not None else None
        
        # 새 UV (있는 경우)
        new_uv = self.uv_coords[unique_verts] if self.uv_coords is not None else None
        
        return MeshData(
            vertices=new_vertices,
            faces=new_faces,
            normals=new_normals,
            face_normals=new_face_normals,
            uv_coords=new_uv,
            texture=self.texture,  # 텍스처는 공유
            unit=self.unit,
            filepath=self.filepath
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
    
    SUPPORTED_FORMATS = {
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
    def get_supported_formats(cls) -> dict:
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
    
    def load(self, filepath: Union[str, Path], unit: Optional[str] = None) -> MeshData:
        """
        메쉬 파일 로드
        
        Args:
            filepath: 메쉬 파일 경로
            unit: 좌표 단위 (None이면 default_unit 사용)
            
        Returns:
            MeshData: 로드된 메쉬 데이터
            
        Raises:
            FileNotFoundError: 파일이 존재하지 않음
            ValueError: 지원하지 않는 포맷
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        ext = filepath.suffix.lower()
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {ext}\n"
                f"Supported formats: {list(self.SUPPORTED_FORMATS.keys())}"
            )
        
        unit = unit or self.default_unit
        
        # trimesh로 로드
        try:
            # 대용량 메쉬 로드 성능: 불필요한 후처리(process) 비활성화
            mesh = trimesh.load(str(filepath), force='mesh', process=False, maintain_order=True)
        except TypeError:
            # 구버전 trimesh 호환
            mesh = trimesh.load(str(filepath), force='mesh')
        
        # Scene인 경우 단일 메쉬로 병합
        if isinstance(mesh, trimesh.Scene):
            meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if len(meshes) == 0:
                raise ValueError(f"No valid mesh found in: {filepath}")
            mesh = trimesh.util.concatenate(meshes)

        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Expected trimesh.Trimesh, got {type(mesh).__name__}")

        # MeshData로 변환
        mesh_data = MeshData.from_trimesh(mesh, filepath=filepath, unit=unit)
        
        # 법선 계산 (없는 경우)
        # 로딩 시점에는 face normals만 계산 (vertex normals는 필요 시점에 계산)
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
    
    def get_file_info(self, filepath: Union[str, Path]) -> dict:
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
        
        info = {
            'filename': filepath.name,
            'format': self.SUPPORTED_FORMATS.get(ext, 'Unknown'),
            'extension': ext,
            'file_size_mb': round(file_size / (1024 * 1024), 2),
        }
        
        # 가능하면 정점/면 수도 빠르게 파악
        try:
            try:
                mesh = trimesh.load(str(filepath), force='mesh', process=False, maintain_order=True)
            except TypeError:
                mesh = trimesh.load(str(filepath), force='mesh')
            if isinstance(mesh, trimesh.Scene):
                meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
                total_verts = sum(m.vertices.shape[0] for m in meshes)
                total_faces = sum(m.faces.shape[0] for m in meshes)
            elif isinstance(mesh, trimesh.Trimesh):
                total_verts = mesh.vertices.shape[0]
                total_faces = mesh.faces.shape[0]
            else:
                total_verts = 0
                total_faces = 0

            info['n_vertices'] = total_verts
            info['n_faces'] = total_faces
            visual = getattr(mesh, "visual", None)
            uv = getattr(visual, "uv", None) if visual is not None else None
            info['has_texture'] = uv is not None
            
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
