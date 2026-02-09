"""
Surface Separator Module
표면 분리 - 기와의 내면/외면 자동 감지 및 분리

법선 방향을 기준으로 메쉬를 내면과 외면으로 분리합니다.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import logging
import numpy as np

from .logging_utils import log_once
from .mesh_loader import MeshData

_LOGGER = logging.getLogger(__name__)


@dataclass
class SeparatedSurfaces:
    """
    분리된 표면 결과
    
    Attributes:
        inner_surface: 내면 메쉬
        outer_surface: 외면 메쉬
        inner_face_indices: 내면에 속하는 원본 면 인덱스
        outer_face_indices: 외면에 속하는 원본 면 인덱스
    """
    inner_surface: Optional[MeshData]
    outer_surface: Optional[MeshData]
    inner_face_indices: np.ndarray
    outer_face_indices: np.ndarray
    
    @property
    def has_both(self) -> bool:
        """내면과 외면 모두 존재하는지"""
        return self.inner_surface is not None and self.outer_surface is not None


class SurfaceSeparator:
    """
    표면 분리기
    
    기와처럼 내면/외면이 구분되는 메쉬를 자동으로 분리합니다.
    """
    
    def __init__(self, angle_threshold: float = 90.0):
        """
        Args:
            angle_threshold: 표면 분리 각도 임계값 (도)
        """
        self.angle_threshold = angle_threshold
    
    def auto_detect_surfaces(
        self,
        mesh: MeshData,
        reference_direction: Optional[np.ndarray] = None,
        *,
        method: str = "normals",
        projector_resolution: int = 768,
        view_depth_tol: float | None = None,
    ) -> SeparatedSurfaces:
        """
        법선 방향으로 내면/외면 자동 분리
        
        Args:
            mesh: 입력 메쉬
            reference_direction: 기준 방향 (None이면 자동 감지)
            
        Returns:
            SeparatedSurfaces: 분리된 표면들
        """
        m = str(method or "normals").strip().lower()
        if m in {"view", "views", "topbottom", "top_bottom", "visible"}:
            return self._auto_detect_surfaces_by_views(
                mesh,
                resolution=int(projector_resolution),
                depth_tol=view_depth_tol,
            )

        # 법선 계산 (내/외면 분리는 face normals만 필요)
        mesh.compute_normals(compute_vertex_normals=False)
        
        if mesh.face_normals is None:
            raise ValueError("Failed to compute face normals")
        
        # 기준 방향 결정
        if reference_direction is None:
            reference_direction = self._estimate_reference_direction(mesh)
        
        reference_direction = np.asarray(reference_direction, dtype=np.float64)
        reference_direction /= np.linalg.norm(reference_direction)
        
        # 각 면의 방향 분류
        dots = np.dot(mesh.face_normals, reference_direction)

        # angle_threshold(도): 기준 방향(outer) / 반대(inner)와 이루는 각도가 이 값보다 작으면
        # "확실한" 면으로 간주합니다. 나머지(애매한) 면은 부호(내적)로 분류합니다.
        try:
            theta = float(self.angle_threshold)
        except Exception:
            theta = 90.0
        theta = float(np.clip(theta, 0.0, 90.0))
        cos_thr = float(np.cos(np.deg2rad(theta)))

        outer_strong = dots > cos_thr
        inner_strong = dots < -cos_thr
        ambiguous = ~(outer_strong | inner_strong)

        outer_mask = outer_strong | (ambiguous & (dots > 0.0))
        inner_mask = ~outer_mask
        
        outer_indices = np.where(outer_mask)[0]
        inner_indices = np.where(inner_mask)[0]
        
        # 서브메쉬 추출
        outer_surface = mesh.extract_submesh(outer_indices) if len(outer_indices) > 0 else None
        inner_surface = mesh.extract_submesh(inner_indices) if len(inner_indices) > 0 else None
        
        return SeparatedSurfaces(
            inner_surface=inner_surface,
            outer_surface=outer_surface,
            inner_face_indices=inner_indices,
            outer_face_indices=outer_indices
        )

    def _auto_detect_surfaces_by_views(
        self,
        mesh: MeshData,
        *,
        resolution: int = 768,
        depth_tol: float | None = None,
    ) -> SeparatedSurfaces:
        """
        상면/하면(Top/Bottom)에서 '보이는 면'을 기준으로 outer/inner를 분리합니다.
        얇은 쉘(기와 등)에서 법선 기반 분리가 불안정할 때 사용합니다.

        구현 메모:
        - 정사투영 "카메라"를 사용하는 대신, face centroid를 픽셀 그리드에 binning 하여
          간이 Z-buffer(최소 depth)를 구성합니다. (triangle 래스터화 대비 훨씬 빠르고,
          대용량 메쉬에서도 세부(주름) 면이 downsample 때문에 누락되는 문제를 줄입니다.)
        - Top/Bottom에서 둘 다/둘 다 아닌(가려진) 면은 기준축(PCA 두께축) 좌표 + (가능하면) 면 법선 prior로 보정합니다.
        """

        vertices = np.asarray(getattr(mesh, "vertices", np.zeros((0, 3))), dtype=np.float64)
        faces = np.asarray(getattr(mesh, "faces", np.zeros((0, 3))), dtype=np.int32)
        if (
            vertices.ndim != 2
            or vertices.shape[0] == 0
            or faces.ndim != 2
            or faces.shape[0] == 0
            or faces.shape[1] < 3
        ):
            outer_idx = np.zeros((0,), dtype=np.int32)
            inner_idx = np.zeros((0,), dtype=np.int32)
            return SeparatedSurfaces(
                inner_surface=None,
                outer_surface=None,
                inner_face_indices=inner_idx,
                outer_face_indices=outer_idx,
            )

        faces = faces[:, :3]

        # Precompute face centroids once (world coords)
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        cent = (v0 + v1 + v2) / 3.0

        axis_map = {
            "top": (2, 0, 1, False),
            "bottom": (2, 0, 1, True),
            "front": (1, 0, 2, False),
            "back": (1, 0, 2, True),
            "left": (0, 1, 2, True),
            "right": (0, 1, 2, False),
        }

        def _compute_depth_buffer(direction: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
            depth_axis, x_axis, y_axis, flip_depth = axis_map.get(direction, (2, 0, 1, False))

            x_all = vertices[:, x_axis]
            y_all = vertices[:, y_axis]
            x_min, x_max = float(np.nanmin(x_all)), float(np.nanmax(x_all))
            y_min, y_max = float(np.nanmin(y_all)), float(np.nanmax(y_all))

            width = float(x_max - x_min)
            height = float(y_max - y_min)
            if not np.isfinite(width) or not np.isfinite(height) or (abs(width) < 1e-12 and abs(height) < 1e-12):
                width = 1.0
                height = 1.0

            res = int(max(64, int(resolution)))
            if width > height:
                img_w = int(res)
                img_h = int(res * height / max(1e-12, width))
            else:
                img_h = int(res)
                img_w = int(res * width / max(1e-12, height))

            img_h = max(1, int(img_h))
            img_w = max(1, int(img_w))

            # Total pixels cap (keep it small; we don't need photo-real depth here)
            max_total = 8_000_000
            total = int(img_h) * int(img_w)
            if total > max_total:
                s = float(np.sqrt(float(max_total) / float(max(1, total))))
                img_w = max(1, int(float(img_w) * s))
                img_h = max(1, int(float(img_h) * s))

            dx = float(x_max - x_min)
            dy = float(y_max - y_min)
            if abs(dx) < 1e-12:
                dx = 1e-12
            if abs(dy) < 1e-12:
                dy = 1e-12

            x = cent[:, x_axis]
            y = cent[:, y_axis]
            z = cent[:, depth_axis]
            if flip_depth:
                z = -z

            sx = (x - x_min) / dx * float(img_w - 1)
            sy = (y - y_min) / dy * float(img_h - 1)
            sy = float(img_h - 1) - sy

            px = np.clip(np.rint(sx), 0, img_w - 1).astype(np.int32, copy=False)
            py = np.clip(np.rint(sy), 0, img_h - 1).astype(np.int32, copy=False)

            depth_buf = np.full((img_h, img_w), np.inf, dtype=np.float64)
            try:
                np.minimum.at(depth_buf, (py, px), z)
            except Exception:
                # best-effort: keep inf buffer (=> no visible faces)
                pass

            depth_buf[np.isinf(depth_buf)] = np.nan
            bounds = np.array([[x_min, y_min], [x_max, y_max]], dtype=np.float64)
            scale = float(max(width / float(img_w), height / float(img_h)))
            return depth_buf, bounds, px, py, scale

        def visible_faces(direction: str) -> np.ndarray:
            depth_map, _bounds, px, py, scale = _compute_depth_buffer(direction)
            if depth_map.ndim != 2:
                return np.zeros((faces.shape[0],), dtype=bool)

            depth_axis, _x_axis, _y_axis, flip_depth = axis_map.get(direction, (2, 0, 1, False))
            z = cent[:, depth_axis]
            if flip_depth:
                z = -z

            try:
                d = depth_map[py, px].astype(np.float64, copy=False)
            except Exception:
                d = np.full((faces.shape[0],), np.nan, dtype=np.float64)

            tol_val = depth_tol
            if tol_val is None:
                tol_val = float(scale) * 0.75
            tol_f = float(max(0.0, tol_val))

            return np.isfinite(d) & np.isfinite(z) & (z <= (d + tol_f))

        vis_top = visible_faces("top")
        vis_bottom = visible_faces("bottom")

        seed_outer = vis_top & ~vis_bottom
        seed_inner = vis_bottom & ~vis_top

        # Resolve ambiguous faces using the PCA thickness axis (+ normal prior).
        d = self._estimate_reference_direction(mesh)
        d = np.asarray(d, dtype=np.float64).reshape(-1)
        if d.size < 3 or not np.isfinite(d[:3]).all():
            d = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        d = d[:3] / (float(np.linalg.norm(d[:3])) + 1e-12)

        t = np.asarray(cent @ d.reshape(3, 1), dtype=np.float64).reshape(-1)
        # Choose d sign so that outer seed tends to have larger t (if both seeds exist).
        if bool(seed_outer.any()) and bool(seed_inner.any()):
            try:
                med_o = float(np.nanmedian(t[seed_outer]))
                med_i = float(np.nanmedian(t[seed_inner]))
                if np.isfinite(med_o) and np.isfinite(med_i) and (med_o < med_i):
                    d = -d
                    t = -t
            except Exception:
                pass
        try:
            med = float(np.nanmedian(t))
        except Exception:
            med = 0.0

        # Normal prior for occluded folds: if a face normal has a clear sign along thickness axis,
        # prefer that over global median split.
        dotn = None
        try:
            if getattr(mesh, "face_normals", None) is None:
                mesh.compute_normals(compute_vertex_normals=False)
            fn = np.asarray(getattr(mesh, "face_normals", None), dtype=np.float64)
            if fn.ndim == 2 and fn.shape[0] == faces.shape[0] and fn.shape[1] >= 3:
                fn = fn[:, :3]
                nrm = np.linalg.norm(fn, axis=1, keepdims=True)
                fn = fn / (nrm + 1e-12)
                dotn = np.asarray(fn @ d.reshape(3,), dtype=np.float64).reshape(-1)
        except Exception:
            dotn = None

        # Fix view seeds when a patch is only visible from one side, but has a very strong
        # opposite normal sign along the thickness axis.
        #
        # This addresses "wrinkles/folds" where an outer patch can be occluded in the top view
        # and end up being visible in the bottom view (thus mis-seeded as inner).
        if isinstance(dotn, np.ndarray) and dotn.shape == (faces.shape[0],):
            seed_fix_thr = 0.35  # require a strong normal agreement to override view seeds
            finite = np.isfinite(dotn)
            fix_to_outer = seed_inner & finite & (dotn >= float(seed_fix_thr))
            fix_to_inner = seed_outer & finite & (dotn <= -float(seed_fix_thr))
            if bool(fix_to_outer.any()):
                seed_outer = seed_outer | fix_to_outer
                seed_inner = seed_inner & ~fix_to_outer
            if bool(fix_to_inner.any()):
                seed_inner = seed_inner | fix_to_inner
                seed_outer = seed_outer & ~fix_to_inner

        ambiguous = ~(seed_outer | seed_inner)

        if isinstance(dotn, np.ndarray) and dotn.shape == (faces.shape[0],):
            thr = 0.12  # small but non-zero: ignore near-perpendicular faces (side walls)
            confident = ambiguous & np.isfinite(dotn) & (np.abs(dotn) >= float(thr))
            assign_outer = (ambiguous & ~confident & (t >= med)) | (confident & (dotn >= 0.0))
        else:
            assign_outer = ambiguous & (t >= med)
        assign_inner = ambiguous & ~assign_outer

        outer_mask = seed_outer | assign_outer
        inner_mask = seed_inner | assign_inner

        outer_indices = np.where(outer_mask)[0].astype(np.int32, copy=False)
        inner_indices = np.where(inner_mask)[0].astype(np.int32, copy=False)

        outer_surface = mesh.extract_submesh(outer_indices) if outer_indices.size > 0 else None
        inner_surface = mesh.extract_submesh(inner_indices) if inner_indices.size > 0 else None

        return SeparatedSurfaces(
            inner_surface=inner_surface,
            outer_surface=outer_surface,
            inner_face_indices=inner_indices,
            outer_face_indices=outer_indices,
        )
    
    def _estimate_reference_direction(self, mesh: MeshData) -> np.ndarray:
        """
        기준 방향 자동 추정

        기와처럼 얇은 쉘/대칭 형상에서는 면적 가중 법선 합이 0에 가까워지기 쉬워서,
        먼저 PCA로 "두께 방향"(최소 분산 축)을 추정하고 실패 시 기존 방식으로 fallback 합니다.
        """
        try:
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            faces = np.asarray(mesh.faces, dtype=np.int32)
            if vertices.ndim != 2 or vertices.shape[0] == 0:
                return np.array([0.0, 0.0, 1.0], dtype=np.float64)
            if faces.ndim != 2 or faces.shape[0] == 0 or faces.shape[1] < 3:
                return np.array([0.0, 0.0, 1.0], dtype=np.float64)

            # 1) PCA: smallest-variance axis ~= sheet/thickness normal (robust for tiles).
            try:
                v = np.asarray(vertices[:, :3], dtype=np.float64)
                v = v[np.isfinite(v).all(axis=1)]
                if v.shape[0] >= 8:
                    c = np.mean(v, axis=0)
                    x = v - c
                    cov = (x.T @ x) / float(max(1, x.shape[0] - 1))
                    w, vecs = np.linalg.eigh(cov)
                    ref = vecs[:, int(np.argsort(w)[0])]
                    ref_norm = float(np.linalg.norm(ref))
                    if ref_norm > 1e-12:
                        ref = (ref / ref_norm).astype(np.float64, copy=False)
                        # Prefer +Z-ish direction to avoid random flips.
                        if float(ref[2]) < 0.0:
                            ref = -ref
                        return ref
            except Exception:
                pass

            # 2) Area-weighted normal sum fallback.
            v0 = vertices[faces[:, 0]]
            v1 = vertices[faces[:, 1]]
            v2 = vertices[faces[:, 2]]

            cross = np.cross(v1 - v0, v2 - v0)
            weighted_normal = np.sum(cross, axis=0)
            norm = float(np.linalg.norm(weighted_normal))
            if norm > 1e-10:
                return (weighted_normal / norm).astype(np.float64, copy=False)
        except Exception:
            log_once(
                _LOGGER,
                "surface_separator:estimate_reference_direction",
                logging.DEBUG,
                "Failed to estimate reference direction; falling back to +Z",
                exc_info=True,
            )

        # 기본값: Z축
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    
    def separate_by_selection(self, mesh: MeshData,
                              selected_face_indices: np.ndarray) -> Tuple[MeshData, MeshData | None]:
        """
        선택된 면으로 수동 분리
        
        Args:
            mesh: 입력 메쉬
            selected_face_indices: 선택된 면 인덱스 (첫 번째 그룹)
            
        Returns:
            (선택된 메쉬, 나머지 메쉬)
        """
        selected = np.asarray(selected_face_indices, dtype=np.int32).reshape(-1)
        n_faces = int(getattr(mesh, "n_faces", 0) or 0)
        if n_faces <= 0:
            return mesh, None

        valid = (selected >= 0) & (selected < n_faces)
        selected_valid = selected[valid]

        mask = np.ones((n_faces,), dtype=bool)
        if selected_valid.size:
            mask[selected_valid] = False
        remaining_indices = np.where(mask)[0].astype(np.int32, copy=False)
        
        selected_mesh = mesh.extract_submesh(selected_valid)
        remaining_mesh = mesh.extract_submesh(remaining_indices) if remaining_indices.size > 0 else None
        
        return selected_mesh, remaining_mesh
    
    def separate_by_curvature(self, mesh: MeshData,
                              curvature_threshold: float = 0.0) -> SeparatedSurfaces:
        """
        곡률 기반 분리 (볼록/오목)
        
        Args:
            mesh: 입력 메쉬
            curvature_threshold: 곡률 임계값
            
        Returns:
            SeparatedSurfaces: 분리된 표면들
        """
        # 정점별 곡률 추정
        curvatures = self._estimate_vertex_curvature(mesh)
        
        # 면별 평균 곡률
        face_curvatures = np.zeros(mesh.n_faces)
        for i, face in enumerate(mesh.faces):
            face_curvatures[i] = curvatures[face].mean()
        
        # 분류: 양의 곡률 (볼록) = 외면, 음의 곡률 (오목) = 내면
        outer_mask = face_curvatures >= curvature_threshold
        inner_mask = ~outer_mask
        
        outer_indices = np.where(outer_mask)[0]
        inner_indices = np.where(inner_mask)[0]
        
        outer_surface = mesh.extract_submesh(outer_indices) if len(outer_indices) > 0 else None
        inner_surface = mesh.extract_submesh(inner_indices) if len(inner_indices) > 0 else None
        
        return SeparatedSurfaces(
            inner_surface=inner_surface,
            outer_surface=outer_surface,
            inner_face_indices=inner_indices,
            outer_face_indices=outer_indices
        )
    
    def _estimate_vertex_curvature(self, mesh: MeshData) -> np.ndarray:
        """정점별 곡률 추정 (간단한 방법)"""
        mesh.compute_normals()
        normals = mesh.normals
        if normals is None:
            raise RuntimeError("Mesh normals are required for curvature estimation")
        
        n = mesh.n_vertices
        curvatures = np.zeros(n)
        
        # 인접 정점 정보
        adjacency = [set() for _ in range(n)]
        for face in mesh.faces:
            for i in range(3):
                adjacency[face[i]].add(face[(i+1) % 3])
                adjacency[face[i]].add(face[(i+2) % 3])
        
        # 각 정점의 곡률 = 이웃 법선과의 평균 각도 차이
        for i in range(n):
            neighbors = list(adjacency[i])
            if len(neighbors) == 0:
                continue
            
            normal_i = normals[i]
            
            # 이웃 법선과의 각도 차이
            angle_diffs = []
            for j in neighbors:
                normal_j = normals[j]
                dot = np.clip(np.dot(normal_i, normal_j), -1, 1)
                angle = np.arccos(dot)
                angle_diffs.append(angle)
            
            curvatures[i] = np.mean(angle_diffs)
        
        return curvatures
    
    def separate_connected_components(self, mesh: MeshData) -> List[MeshData]:
        """
        연결 컴포넌트별로 분리
        
        Args:
            mesh: 입력 메쉬
            
        Returns:
            분리된 메쉬 목록
        """
        m = mesh.n_faces
        
        # 면 인접성 그래프 구성
        face_adjacency = [set() for _ in range(m)]
        
        # 엣지 -> 면 매핑
        edge_to_faces = {}
        for fi, face in enumerate(mesh.faces):
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i+1) % 3]]))
                if edge in edge_to_faces:
                    # 인접 면 연결
                    other_face = edge_to_faces[edge]
                    face_adjacency[fi].add(other_face)
                    face_adjacency[other_face].add(fi)
                else:
                    edge_to_faces[edge] = fi
        
        # BFS로 연결 컴포넌트 찾기
        visited = np.zeros(m, dtype=bool)
        components = []
        
        from collections import deque

        for start_face in range(m):
            if visited[start_face]:
                continue
            
            # BFS
            component = []
            queue = deque([int(start_face)])
            visited[start_face] = True
            
            while queue:
                face = int(queue.popleft())
                component.append(face)
                
                for neighbor in face_adjacency[face]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(int(neighbor))
            
            components.append(np.array(component))
        
        # 각 컴포넌트를 서브메쉬로 추출
        return [mesh.extract_submesh(comp) for comp in components]


# 테스트용
if __name__ == '__main__':
    print("Surface Separator module loaded successfully")
    print("Use: separator = SurfaceSeparator(); result = separator.auto_detect_surfaces(mesh)")
