"""
단면 프로파일 그래프 위젯
메쉬 표면의 높이 변화를 그래프로 시각화합니다.
"""
import logging

from PyQt6.QtWidgets import QWidget, QFileDialog, QMenu
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QPen, QColor, QFont
import numpy as np

from ..core.logging_utils import log_once

_LOGGER = logging.getLogger(__name__)


class ProfileGraphWidget(QWidget):
    """높이 프로파일을 그리는 위젯"""
    
    def __init__(self, title="프로파일", parent=None):
        super().__init__(parent)
        self.title = title
        self.points = []  # (dist, height) 리스트
        self._view_range = None  # (min_d, max_d, min_h, max_h) or None(auto)
        self.setMinimumHeight(150)
        self.setStyleSheet("background-color: white; border: 1px solid #ddd;")
        
    def set_data(self, points):
        """
        그래프 데이터 설정
        Args:
            points: [(dist1, h1), (dist2, h2), ...] 형태의 넘파이 배열 또는 리스트
        """
        self.points = points
        self._view_range = None
        self.update()

    def _compute_auto_range(self):
        if not self.points or len(self.points) < 2:
            return None

        data = np.asarray(self.points, dtype=np.float64)
        if data.ndim != 2 or data.shape[1] < 2:
            return None

        dists = data[:, 0]
        heights = data[:, 1]

        if dists.size < 2 or heights.size < 2:
            return None

        min_d, max_d = float(np.nanmin(dists)), float(np.nanmax(dists))
        min_h, max_h = float(np.nanmin(heights)), float(np.nanmax(heights))

        # 범위가 너무 작으면 보정
        if abs(max_d - min_d) < 1e-12:
            max_d = min_d + 1.0
        if abs(max_h - min_h) < 1e-12:
            min_h -= 1.0
            max_h += 1.0

        # 여유 공간(높이 방향)
        h_range = max_h - min_h
        min_h -= h_range * 0.1
        max_h += h_range * 0.1

        return (min_d, max_d, min_h, max_h)

    def reset_view(self):
        self._view_range = None
        self.update()

    def _save_image_dialog(self):
        default_name = "profile.png"
        try:
            safe_title = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(self.title))[:40]
            if safe_title:
                default_name = f"{safe_title}.png"
        except Exception:
            log_once(
                _LOGGER,
                "profile_graph_widget.safe_title",
                logging.DEBUG,
                "Failed to derive safe default filename from title: %r",
                self.title,
                exc_info=True,
            )

        path, _ = QFileDialog.getSaveFileName(
            self,
            "그래프 이미지 저장",
            default_name,
            "PNG Images (*.png);;JPEG Images (*.jpg *.jpeg);;All Files (*)",
        )
        if not path:
            return
        try:
            pix = self.grab()
            pix.save(path)
        except Exception:
            _LOGGER.exception("Failed to save profile graph image to %s", path)
            return

    def contextMenuEvent(self, a0):
        if a0 is None:
            return
        event = a0
        menu = QMenu(self)
        act_save = menu.addAction("이미지로 저장...")
        act_reset = menu.addAction("줌/뷰 초기화")
        chosen = menu.exec(event.globalPos())
        if chosen == act_save:
            self._save_image_dialog()
        elif chosen == act_reset:
            self.reset_view()

    def mouseDoubleClickEvent(self, a0):
        if a0 is None:
            return
        # 더블클릭: 뷰 초기화
        self.reset_view()

    def wheelEvent(self, a0):
        if a0 is None:
            return
        event = a0
        if not self.points or len(self.points) < 2:
            return

        auto = self._compute_auto_range()
        if auto is None:
            return

        if self._view_range is None:
            self._view_range = auto

        min_d, max_d, min_h, max_h = [float(v) for v in self._view_range]
        if abs(max_d - min_d) < 1e-12 or abs(max_h - min_h) < 1e-12:
            return

        delta = event.angleDelta().y()
        if delta == 0:
            return

        steps = float(delta) / 120.0
        zoom = float(np.power(0.9, steps))  # wheel up: zoom in (<1), wheel down: zoom out (>1)

        w, h = self.width(), self.height()
        margin = 30
        graph_w = max(1.0, float(w - 2 * margin))
        graph_h = max(1.0, float(h - 2 * margin))

        # 마우스 위치를 데이터 좌표로 변환(그래프 밖이면 중심 기준)
        mx = float(event.position().x())
        my = float(event.position().y())
        if mx < margin or mx > (w - margin) or my < margin or my > (h - margin):
            pivot_d = (min_d + max_d) * 0.5
            pivot_h = (min_h + max_h) * 0.5
        else:
            rx = (mx - margin) / graph_w
            ry = (my - margin) / graph_h
            pivot_d = min_d + rx * (max_d - min_d)
            pivot_h = max_h - ry * (max_h - min_h)

        # Ctrl: X만, Shift: Y만, 기본: 둘 다
        mods = event.modifiers()
        zoom_x = zoom_y = zoom
        if mods & Qt.KeyboardModifier.ControlModifier:
            zoom_y = 1.0
        if mods & Qt.KeyboardModifier.ShiftModifier:
            zoom_x = 1.0

        new_min_d = pivot_d - (pivot_d - min_d) * zoom_x
        new_max_d = pivot_d + (max_d - pivot_d) * zoom_x
        new_min_h = pivot_h - (pivot_h - min_h) * zoom_y
        new_max_h = pivot_h + (max_h - pivot_h) * zoom_y

        # 최소 스팬 보정
        if abs(new_max_d - new_min_d) < 1e-6:
            mid = (new_min_d + new_max_d) * 0.5
            new_min_d = mid - 5e-7
            new_max_d = mid + 5e-7
        if abs(new_max_h - new_min_h) < 1e-6:
            mid = (new_min_h + new_max_h) * 0.5
            new_min_h = mid - 5e-7
            new_max_h = mid + 5e-7

        self._view_range = (float(new_min_d), float(new_max_d), float(new_min_h), float(new_max_h))
        self.update()
        
    def paintEvent(self, a0):
        if a0 is None:
            return
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
            
        auto = self._compute_auto_range()
        if auto is None:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "데이터 없음")
            return

        if self._view_range is None:
            min_d, max_d, min_h, max_h = auto
        else:
            try:
                min_d, max_d, min_h, max_h = [float(v) for v in self._view_range]
            except Exception:
                min_d, max_d, min_h, max_h = auto

            if abs(max_d - min_d) < 1e-12 or abs(max_h - min_h) < 1e-12:
                min_d, max_d, min_h, max_h = auto

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
