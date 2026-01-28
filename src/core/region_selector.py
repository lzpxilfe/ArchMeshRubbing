"""
Region Selector Module
영역 선택 - 미구 등 특정 부분만 추출

3D 또는 2D 뷰에서 선택한 영역만 추출하여 별도 처리할 수 있습니다.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .mesh_loader import MeshData


@dataclass
class SelectionResult:
    """
    선택 결과
    
    Attributes:
        selected_mesh: 선택된 영역의 메쉬
        remaining_mesh: 나머지 영역의 메쉬
        selected_face_indices: 선택된 면 인덱스
    """
    selected_mesh: Optional[MeshData]
    remaining_mesh: Optional[MeshData]
    selected_face_indices: np.ndarray


class RegionSelector:
    """
    영역 선택 도구
    
    다양한 방식으로 메쉬의 특정 영역을 선택할 수 있습니다.
    """
    
    def select_by_plane(self, mesh: MeshData,
                        plane_point: np.ndarray,
                        plane_normal: np.ndarray,
                        keep_side: str = 'positive') -> SelectionResult:
        """
        평면으로 영역 선택 (미구 등의 단진 부분 추출)
        
        Args:
            mesh: 입력 메쉬
            plane_point: 평면 위의 한 점
            plane_normal: 평면 법선 벡터
            keep_side: 유지할 쪽 ('positive', 'negative')
            
        Returns:
            SelectionResult: 선택 결과
        """
        plane_point = np.asarray(plane_point, dtype=np.float64)
        plane_normal = np.asarray(plane_normal, dtype=np.float64)
        plane_normal /= np.linalg.norm(plane_normal)
        
        # 면 중심 계산
        face_centers = np.zeros((mesh.n_faces, 3))
        for i, face in enumerate(mesh.faces):
            face_centers[i] = mesh.vertices[face].mean(axis=0)
        
        # 평면에서의 부호 있는 거리
        distances = np.dot(face_centers - plane_point, plane_normal)
        
        # 선택
        if keep_side == 'positive':
            selected_mask = distances > 0
        else:
            selected_mask = distances < 0
        
        selected_indices = np.where(selected_mask)[0]
        remaining_indices = np.where(~selected_mask)[0]
        
        selected_mesh = mesh.extract_submesh(selected_indices)
        remaining_mesh = mesh.extract_submesh(remaining_indices) if len(remaining_indices) > 0 else None
        
        return SelectionResult(
            selected_mesh=selected_mesh,
            remaining_mesh=remaining_mesh,
            selected_face_indices=selected_indices
        )
    
    def select_by_box(self, mesh: MeshData,
                      box_min: np.ndarray,
                      box_max: np.ndarray) -> SelectionResult:
        """
        바운딩 박스로 영역 선택
        
        Args:
            mesh: 입력 메쉬
            box_min: 박스 최소점 [x, y, z]
            box_max: 박스 최대점 [x, y, z]
            
        Returns:
            SelectionResult: 선택 결과
        """
        box_min = np.asarray(box_min, dtype=np.float64)
        box_max = np.asarray(box_max, dtype=np.float64)
        
        # 면 중심 계산
        face_centers = np.zeros((mesh.n_faces, 3))
        for i, face in enumerate(mesh.faces):
            face_centers[i] = mesh.vertices[face].mean(axis=0)
        
        # 박스 내부 체크
        in_box = np.all((face_centers >= box_min) & (face_centers <= box_max), axis=1)
        
        selected_indices = np.where(in_box)[0]
        remaining_indices = np.where(~in_box)[0]
        
        selected_mesh = mesh.extract_submesh(selected_indices)
        remaining_mesh = mesh.extract_submesh(remaining_indices) if len(remaining_indices) > 0 else None
        
        return SelectionResult(
            selected_mesh=selected_mesh,
            remaining_mesh=remaining_mesh,
            selected_face_indices=selected_indices
        )
    
    def select_by_height_range(self, mesh: MeshData,
                               min_height: float,
                               max_height: float,
                               axis: int = 2) -> SelectionResult:
        """
        높이 범위로 영역 선택 (미구 단 추출에 유용)
        
        Args:
            mesh: 입력 메쉬
            min_height: 최소 높이
            max_height: 최대 높이
            axis: 높이 축 (0=X, 1=Y, 2=Z)
            
        Returns:
            SelectionResult: 선택 결과
        """
        # 면 중심의 높이
        face_heights = np.zeros(mesh.n_faces)
        for i, face in enumerate(mesh.faces):
            face_heights[i] = mesh.vertices[face, axis].mean()
        
        # 범위 내 선택
        in_range = (face_heights >= min_height) & (face_heights <= max_height)
        
        selected_indices = np.where(in_range)[0]
        remaining_indices = np.where(~in_range)[0]
        
        selected_mesh = mesh.extract_submesh(selected_indices)
        remaining_mesh = mesh.extract_submesh(remaining_indices) if len(remaining_indices) > 0 else None
        
        return SelectionResult(
            selected_mesh=selected_mesh,
            remaining_mesh=remaining_mesh,
            selected_face_indices=selected_indices
        )
    
    def select_by_distance_from_boundary(self, mesh: MeshData,
                                          max_distance: float) -> SelectionResult:
        """
        경계로부터의 거리로 선택 (가장자리 영역 추출)
        
        Args:
            mesh: 입력 메쉬
            max_distance: 경계로부터 최대 거리
            
        Returns:
            SelectionResult: 선택 결과
        """
        # 경계 정점 찾기
        boundary_verts = set(mesh.get_boundary_vertices())
        
        if len(boundary_verts) == 0:
            # 닫힌 메쉬: 전체 선택
            return SelectionResult(
                selected_mesh=mesh,
                remaining_mesh=None,
                selected_face_indices=np.arange(mesh.n_faces)
            )
        
        # 각 정점의 경계까지 거리 계산 (BFS)
        n = mesh.n_vertices
        distances = np.full(n, np.inf)
        
        # 인접 정점
        adjacency = [set() for _ in range(n)]
        for face in mesh.faces:
            for i in range(3):
                adjacency[face[i]].add(face[(i+1) % 3])
                adjacency[face[i]].add(face[(i+2) % 3])
        
        # BFS로 거리 계산
        queue = list(boundary_verts)
        for v in boundary_verts:
            distances[v] = 0
        
        while queue:
            v = queue.pop(0)
            for neighbor in adjacency[v]:
                edge_dist = np.linalg.norm(mesh.vertices[neighbor] - mesh.vertices[v])
                new_dist = distances[v] + edge_dist
                
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    queue.append(neighbor)
        
        # 면 선택 (모든 정점이 거리 내에 있는 면)
        selected = []
        for i, face in enumerate(mesh.faces):
            face_dist = distances[face].mean()
            if face_dist <= max_distance:
                selected.append(i)
        
        selected_indices = np.array(selected)
        all_indices = set(range(mesh.n_faces))
        remaining_indices = np.array(list(all_indices - set(selected)))
        
        selected_mesh = mesh.extract_submesh(selected_indices) if len(selected_indices) > 0 else None
        remaining_mesh = mesh.extract_submesh(remaining_indices) if len(remaining_indices) > 0 else None
        
        return SelectionResult(
            selected_mesh=selected_mesh,
            remaining_mesh=remaining_mesh,
            selected_face_indices=selected_indices
        )
    
    def select_by_curvature(self, mesh: MeshData,
                            min_curvature: float = 0.0,
                            max_curvature: float = np.inf) -> SelectionResult:
        """
        곡률 범위로 선택 (평평한 부분 vs 곡면 부분)
        
        Args:
            mesh: 입력 메쉬
            min_curvature: 최소 곡률
            max_curvature: 최대 곡률
            
        Returns:
            SelectionResult: 선택 결과
        """
        mesh.compute_normals()
        normals = mesh.normals
        if normals is None:
            raise RuntimeError("Mesh normals are required for curvature selection")
        
        # 정점별 곡률 추정
        n = mesh.n_vertices
        curvatures = np.zeros(n)
        
        adjacency = [set() for _ in range(n)]
        for face in mesh.faces:
            for i in range(3):
                adjacency[face[i]].add(face[(i+1) % 3])
                adjacency[face[i]].add(face[(i+2) % 3])
        
        for i in range(n):
            neighbors = list(adjacency[i])
            if len(neighbors) == 0:
                continue
            
            normal_i = normals[i]
            angle_diffs = []
            for j in neighbors:
                normal_j = normals[j]
                dot = np.clip(np.dot(normal_i, normal_j), -1, 1)
                angle = np.arccos(dot)
                angle_diffs.append(angle)
            
            curvatures[i] = np.mean(angle_diffs)
        
        # 면별 곡률
        face_curvatures = np.zeros(mesh.n_faces)
        for i, face in enumerate(mesh.faces):
            face_curvatures[i] = curvatures[face].mean()
        
        # 범위 내 선택
        in_range = (face_curvatures >= min_curvature) & (face_curvatures <= max_curvature)
        
        selected_indices = np.where(in_range)[0]
        remaining_indices = np.where(~in_range)[0]
        
        selected_mesh = mesh.extract_submesh(selected_indices) if len(selected_indices) > 0 else None
        remaining_mesh = mesh.extract_submesh(remaining_indices) if len(remaining_indices) > 0 else None
        
        return SelectionResult(
            selected_mesh=selected_mesh,
            remaining_mesh=remaining_mesh,
            selected_face_indices=selected_indices
        )
    
    def select_by_face_indices(self, mesh: MeshData,
                               face_indices: np.ndarray) -> SelectionResult:
        """
        면 인덱스 목록으로 직접 선택
        
        Args:
            mesh: 입력 메쉬
            face_indices: 선택할 면 인덱스 배열
            
        Returns:
            SelectionResult: 선택 결과
        """
        face_indices = np.asarray(face_indices, dtype=np.int32)
        
        all_indices = set(range(mesh.n_faces))
        selected_set = set(face_indices)
        remaining_indices = np.array(list(all_indices - selected_set))
        
        selected_mesh = mesh.extract_submesh(face_indices)
        remaining_mesh = mesh.extract_submesh(remaining_indices) if len(remaining_indices) > 0 else None
        
        return SelectionResult(
            selected_mesh=selected_mesh,
            remaining_mesh=remaining_mesh,
            selected_face_indices=face_indices
        )
    
    def grow_selection(self, mesh: MeshData,
                       initial_faces: np.ndarray,
                       iterations: int = 1) -> np.ndarray:
        """
        선택 영역 확장
        
        Args:
            mesh: 입력 메쉬
            initial_faces: 초기 선택 면 인덱스
            iterations: 확장 반복 횟수
            
        Returns:
            확장된 면 인덱스 배열
        """
        # 면 인접성 구성
        m = mesh.n_faces
        face_adjacency = [set() for _ in range(m)]
        
        edge_to_faces = {}
        for fi, face in enumerate(mesh.faces):
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i+1) % 3]]))
                if edge in edge_to_faces:
                    other = edge_to_faces[edge]
                    face_adjacency[fi].add(other)
                    face_adjacency[other].add(fi)
                else:
                    edge_to_faces[edge] = fi
        
        # 선택 확장
        selected = set(initial_faces)
        
        for _ in range(iterations):
            new_selection = set()
            for face in selected:
                new_selection.update(face_adjacency[face])
            selected.update(new_selection)
        
        return np.array(sorted(selected))
    
    def shrink_selection(self, mesh: MeshData,
                         initial_faces: np.ndarray,
                         iterations: int = 1) -> np.ndarray:
        """
        선택 영역 축소
        
        Args:
            mesh: 입력 메쉬
            initial_faces: 초기 선택 면 인덱스
            iterations: 축소 반복 횟수
            
        Returns:
            축소된 면 인덱스 배열
        """
        # 면 인접성 구성
        m = mesh.n_faces
        face_adjacency = [set() for _ in range(m)]
        
        edge_to_faces = {}
        for fi, face in enumerate(mesh.faces):
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i+1) % 3]]))
                if edge in edge_to_faces:
                    other = edge_to_faces[edge]
                    face_adjacency[fi].add(other)
                    face_adjacency[other].add(fi)
                else:
                    edge_to_faces[edge] = fi
        
        # 선택 축소 (경계 제거)
        selected = set(initial_faces)
        
        for _ in range(iterations):
            boundary_faces = set()
            for face in selected:
                # 인접 면 중 선택되지 않은 것이 있으면 경계
                if any(adj not in selected for adj in face_adjacency[face]):
                    boundary_faces.add(face)
            
            selected -= boundary_faces
        
        return np.array(sorted(selected))


# 테스트용
if __name__ == '__main__':
    print("Region Selector module loaded successfully")
    print("Use: selector = RegionSelector(); result = selector.select_by_height_range(mesh, 0, 10)")
