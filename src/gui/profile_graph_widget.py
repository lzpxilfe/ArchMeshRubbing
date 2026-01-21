"""
단면 프로파일 그래프 위젯
메쉬 표면의 높이 변화를 그래프로 시각화합니다.
"""
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPainter, QPen, QColor, QFont
import numpy as np


class ProfileGraphWidget(QWidget):
    """높이 프로파일을 그리는 위젯"""
    
    def __init__(self, title="프로파일", parent=None):
        super().__init__(parent)
        self.title = title
        self.points = []  # (dist, height) 리스트
        self.setMinimumHeight(150)
        self.setStyleSheet("background-color: white; border: 1px solid #ddd;")
        
    def set_data(self, points):
        """
        그래프 데이터 설정
        Args:
            points: [(dist1, h1), (dist2, h2), ...] 형태의 넘파이 배열 또는 리스트
        """
        self.points = points
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w, h = self.width(), self.height()
        margin = 30
        graph_w = w - 2 * margin
        graph_h = h - 2 * margin
        
        # 배경
        painter.fillRect(self.rect(), QColor(255, 255, 255))
        
        # 제목
        painter.setFont(QFont("Malgun Gothic", 9, QFont.Weight.Bold))
        painter.drawText(margin, 20, self.title)
        
        if not self.points or len(self.points) < 2:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "데이터 없음")
            return
            
        data = np.array(self.points)
        dists = data[:, 0]
        heights = data[:, 1]
        
        min_d, max_d = dists.min(), dists.max()
        min_h, max_h = heights.min(), heights.max()
        
        # 범위가 너무 작으면 보정
        if max_d == min_d: max_d += 1
        if max_h == min_h: 
            min_h -= 1
            max_h += 1
        
        # 여유 공간 추가
        h_range = max_h - min_h
        min_h -= h_range * 0.1
        max_h += h_range * 0.1
        h_range = max_h - min_h
        
        # 축 그리기
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        # X축
        painter.drawLine(margin, h - margin, w - margin, h - margin)
        # Y축
        painter.drawLine(margin, margin, margin, h - margin)
        
        # 눈금 및 라벨 (Y축)
        painter.setFont(QFont("Arial", 7))
        painter.drawText(5, margin, f"{max_h:.1f}")
        painter.drawText(5, h - margin, f"{min_h:.1f}")
        
        # 데이터 그리기
        painter.setPen(QPen(QColor(255, 0, 0), 2))
        
        def to_screen(d, h_val):
            sx = margin + (d - min_d) / (max_d - min_d) * graph_w
            sy = h - margin - (h_val - min_h) / h_range * graph_h
            return sx, sy
            
        points_q = []
        for d, hv in self.points:
            sx, sy = to_screen(d, hv)
            points_q.append((sx, sy))
            
        for i in range(len(points_q) - 1):
            p1 = points_q[i]
            p2 = points_q[i+1]
            painter.drawLine(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))
            
        # 0점 라인 (있다면)
        if min_h <= 0 <= max_h:
            painter.setPen(QPen(QColor(200, 200, 200), 1, Qt.PenStyle.DashLine))
            _, sy0 = to_screen(min_d, 0)
            painter.drawLine(margin, int(sy0), w - margin, int(sy0))
