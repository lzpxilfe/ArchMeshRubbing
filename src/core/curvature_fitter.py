"""
Curvature Fitter - 3D 점들로부터 원호 피팅
메쉬 외면에 찍은 점들을 이용해 곡률 반경을 계산
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class FittedArc:
    """피팅된 원호 결과"""
    center: np.ndarray      # 원의 중심 (3D)
    radius: float           # 반지름 (cm)
    normal: np.ndarray      # 원이 놓인 평면의 법선
    points_2d: np.ndarray   # 평면에 투영된 2D 점들
    plane_origin: np.ndarray # 평면 원점
    plane_u: np.ndarray     # 평면 U축
    plane_v: np.ndarray     # 평면 V축


class CurvatureFitter:
    """
    3D 점들로부터 원호(Arc)를 피팅하는 클래스
    
    워크플로우:
    1. 3D 점들을 최적 평면에 투영
    2. 2D에서 원 피팅 (최소자승법)
    3. 결과를 다시 3D로 변환
    """
    
    @staticmethod
    def fit_plane(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        점들을 지나는 최적 평면 계산 (PCA)
        
        Args:
            points: (N, 3) 3D 점들
            
        Returns:
            (centroid, normal): 평면의 중심과 법선
        """
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        
        # SVD로 주성분 분석
        _, _, vh = np.linalg.svd(centered)
        
        # 가장 작은 특이값에 해당하는 벡터 = 법선
        normal = vh[-1]
        
        return centroid, normal
    
    @staticmethod
    def project_to_plane(points: np.ndarray, origin: np.ndarray, normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        3D 점들을 평면에 투영하여 2D 좌표로 변환
        
        Returns:
            (points_2d, u_axis, v_axis): 2D 좌표와 평면 축
        """
        # 평면의 로컬 좌표계 생성
        # 임의의 벡터와 외적하여 U축 생성
        if abs(normal[0]) < 0.9:
            temp = np.array([1, 0, 0])
        else:
            temp = np.array([0, 1, 0])
        
        u_axis = np.cross(normal, temp)
        u_axis = u_axis / np.linalg.norm(u_axis)
        
        v_axis = np.cross(normal, u_axis)
        v_axis = v_axis / np.linalg.norm(v_axis)
        
        # 점들을 2D로 투영
        centered = points - origin
        points_2d = np.zeros((len(points), 2))
        points_2d[:, 0] = np.dot(centered, u_axis)
        points_2d[:, 1] = np.dot(centered, v_axis)
        
        return points_2d, u_axis, v_axis
    
    @staticmethod
    def fit_circle_2d(points_2d: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        2D 점들에 원을 피팅 (최소자승법)
        
        Algebraic circle fit (Kasa method)
        
        Args:
            points_2d: (N, 2) 2D 점들
            
        Returns:
            (center_2d, radius): 원의 중심과 반지름
        """
        n = len(points_2d)
        if n < 3:
            raise ValueError("원 피팅에는 최소 3개의 점이 필요합니다")
        
        # 선형 최소자승법으로 원 피팅
        # (x - a)^2 + (y - b)^2 = r^2
        # x^2 + y^2 = 2ax + 2by + (r^2 - a^2 - b^2)
        
        x = points_2d[:, 0]
        y = points_2d[:, 1]
        
        # A * [a, b, c]^T = b
        # where c = r^2 - a^2 - b^2
        A = np.column_stack([2*x, 2*y, np.ones(n)])
        b = x**2 + y**2
        
        # 최소자승법 풀이
        result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        a, b_val, c = [float(v) for v in result]
        center_2d = np.array([a, b_val], dtype=np.float64)
        radius2 = float(c + a * a + b_val * b_val)
        radius = float(np.sqrt(max(radius2, 0.0)))
        
        return center_2d, radius
    
    def fit_arc(self, points: np.ndarray) -> Optional[FittedArc]:
        """
        3D 점들에 원호를 피팅
        
        Args:
            points: (N, 3) 3D 점들, 최소 3개 필요
            
        Returns:
            FittedArc 객체 또는 None (실패 시)
        """
        points = np.array(points)
        
        if len(points) < 3:
            return None
        
        try:
            # 1. 최적 평면 찾기
            origin, normal = self.fit_plane(points)
            
            # 2. 평면에 투영
            points_2d, u_axis, v_axis = self.project_to_plane(points, origin, normal)
            
            # 3. 2D 원 피팅
            center_2d, radius = self.fit_circle_2d(points_2d)
            
            # 4. 중심을 3D로 변환
            center_3d = origin + center_2d[0] * u_axis + center_2d[1] * v_axis
            
            return FittedArc(
                center=center_3d,
                radius=radius,
                normal=normal,
                points_2d=points_2d,
                plane_origin=origin,
                plane_u=u_axis,
                plane_v=v_axis
            )
            
        except Exception as e:
            print(f"원호 피팅 실패: {e}")
            return None
    
    def generate_arc_points(self, arc: FittedArc, n_points: int = 64) -> np.ndarray:
        """
        피팅된 원호를 시각화하기 위한 3D 점들 생성
        
        Args:
            arc: FittedArc 객체
            n_points: 생성할 점의 개수
            
        Returns:
            (n_points, 3) 원호 위의 3D 점들
        """
        angles = np.linspace(0, 2 * np.pi, n_points)
        
        # 2D 원 점들
        circle_2d = np.zeros((n_points, 2))
        circle_2d[:, 0] = arc.radius * np.cos(angles)
        circle_2d[:, 1] = arc.radius * np.sin(angles)
        
        # 3D로 변환
        circle_3d = np.zeros((n_points, 3))
        for i, (u, v) in enumerate(circle_2d):
            circle_3d[i] = arc.center + u * arc.plane_u + v * arc.plane_v
        
        return circle_3d


# 테스트
if __name__ == '__main__':
    fitter = CurvatureFitter()
    
    # 반지름 10cm 원 위의 점들 생성
    theta = np.array([0, np.pi/4, np.pi/2])
    test_points = np.zeros((3, 3))
    test_points[:, 0] = 10 * np.cos(theta)
    test_points[:, 1] = 10 * np.sin(theta)
    test_points[:, 2] = 0
    
    result = fitter.fit_arc(test_points)
    
    if result:
        print("피팅 결과:")
        print(f"  중심: {result.center}")
        print(f"  반지름: {result.radius:.2f} cm")
        print(f"  법선: {result.normal}")
