"""
ArchMeshRubbing GUI - Main Window
媛꾨떒???쒕옒洹????쒕∼ ?명꽣?섏씠??
"""

import sys
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QTextEdit,
    QGroupBox, QRadioButton, QSpinBox, QComboBox, QMessageBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDragLeaveEvent, QDropEvent, QFont

# Ensure repository root is on sys.path so "src" is importable.
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.runtime_defaults import DEFAULTS  # noqa: E402
from src.core.output_paths import (  # noqa: E402
    rubbing_output_path,
    projection_output_path,
    inner_surface_path,
    outer_surface_path,
)

APP_NAME = "ArchMeshRubbing"
APP_VERSION = "0.1.0"
DEFAULT_EXPORT_DPI = DEFAULTS.export_dpi
DEFAULT_RENDER_RESOLUTION = DEFAULTS.render_resolution
DEFAULT_ARAP_MAX_ITERATIONS = DEFAULTS.arap_max_iterations
GUI_MIN_RESOLUTION = DEFAULTS.gui_min_resolution
GUI_MAX_RESOLUTION = DEFAULTS.gui_max_resolution
DEFAULT_MESH_UNIT = "mm"
try:
    import src as _amr_src

    APP_VERSION = str(getattr(_amr_src, "__version__", APP_VERSION))
except Exception:
    pass


def _format_optional_count(value: object) -> str:
    if isinstance(value, bool):
        return "?"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{int(value):,}"
    if isinstance(value, str):
        try:
            return f"{int(value.strip()):,}"
        except ValueError:
            return "?"
    return "?"


class ProcessingThread(QThread):
    """Background worker thread."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, filepath, mode, options):
        super().__init__()
        self.filepath = filepath
        self.mode = mode
        self.options = options
    
    def run(self):
        try:
            from src.core.mesh_loader import MeshLoader
            from src.core.flattener import ARAPFlattener
            from src.core.orthographic_projector import OrthographicProjector
            from src.core.surface_visualizer import SurfaceVisualizer
            from src.core.surface_separator import SurfaceSeparator
            
            # 1. 濡쒕뱶
            self.progress.emit(10, "硫붿돩 濡쒕뵫 以?..")
            loader = MeshLoader(default_unit=DEFAULT_MESH_UNIT)
            mesh = loader.load(self.filepath)
            
            self.progress.emit(20, f"Loaded: {mesh.n_vertices:,} vertices, {mesh.n_faces:,} faces")
            
            output_path = Path(self.filepath)
            
            if self.mode == 'flatten':
                # Flatten
                self.progress.emit(30, "Flattening mesh (ARAP)...")
                flattener = ARAPFlattener(max_iterations=DEFAULT_ARAP_MAX_ITERATIONS)
                flattened = flattener.flatten(mesh)
                
                self.progress.emit(60, "?곷낯 ?대?吏 ?앹꽦 以?..")
                visualizer = SurfaceVisualizer(default_dpi=DEFAULT_EXPORT_DPI)
                rubbing = visualizer.generate_rubbing(
                    flattened, 
                    width_pixels=self.options.get('resolution', DEFAULT_RENDER_RESOLUTION),
                    style=self.options.get('style', 'traditional')
                )
                
                self.progress.emit(90, "???以?..")
                save_path = rubbing_output_path(output_path)
                rubbing.save(str(save_path), include_scale_bar=True)
                
                self.progress.emit(100, "?꾨즺!")
                self.finished.emit(True, str(save_path))
                
            elif self.mode == 'project':
                # ?뺤궗?ъ쁺
                self.progress.emit(40, "?뺤궗?ъ쁺 ?앹꽦 以?..")
                projector = OrthographicProjector(
                    resolution=self.options.get('resolution', DEFAULT_RENDER_RESOLUTION)
                )
                aligned = projector.align_mesh(mesh, method='pca')
                result = projector.project(aligned, direction='top', render_mode='depth')
                
                self.progress.emit(90, "???以?..")
                save_path = projection_output_path(output_path)
                result.save(str(save_path), dpi=DEFAULT_EXPORT_DPI)
                
                self.progress.emit(100, "?꾨즺!")
                self.finished.emit(True, str(save_path))
                
            elif self.mode == 'separate':
                # ?쒕㈃ 遺꾨━
                self.progress.emit(50, "?쒕㈃ 遺꾨━ 以?..")
                separator = SurfaceSeparator()
                result = separator.auto_detect_surfaces(mesh)
                
                self.progress.emit(80, "???以?..")
                
                saved_files = []
                if result.inner_surface:
                    inner_path = inner_surface_path(output_path)
                    result.inner_surface.to_trimesh().export(str(inner_path))
                    saved_files.append(str(inner_path))
                
                if result.outer_surface:
                    outer_path = outer_surface_path(output_path)
                    result.outer_surface.to_trimesh().export(str(outer_path))
                    saved_files.append(str(outer_path))
                
                self.progress.emit(100, "?꾨즺!")
                self.finished.emit(True, ", ".join(saved_files))
            
        except Exception as e:
            self.finished.emit(False, str(e))


class DropArea(QLabel):
    """?쒕옒洹????쒕∼ ?곸뿭"""
    fileDropped = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("3D ?뚯씪???ш린???뚯뼱???볦쑝?몄슂\n\n(.obj, .ply, .stl, .off)")
        self.setStyleSheet("""
            QLabel {
                border: 3px dashed #aaa;
                border-radius: 15px;
                background-color: #f5f5f5;
                font-size: 16px;
                color: #666;
                min-height: 150px;
                padding: 20px;
            }
            QLabel:hover {
                border-color: #4a90d9;
                background-color: #e8f4fc;
            }
        """)
        self.setAcceptDrops(True)
    
    def dragEnterEvent(self, a0: QDragEnterEvent | None):
        if a0 is None:
            return
        event = a0
        mime = event.mimeData()
        if mime is not None and mime.hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(self.styleSheet().replace("#f5f5f5", "#d4edda"))

    def dragLeaveEvent(self, a0: QDragLeaveEvent | None):
        if a0 is None:
            return
        self.setStyleSheet(self.styleSheet().replace("#d4edda", "#f5f5f5"))

    def dropEvent(self, a0: QDropEvent | None):
        if a0 is None:
            return
        event = a0
        self.setStyleSheet(self.styleSheet().replace("#d4edda", "#f5f5f5"))
        mime = event.mimeData()
        urls = mime.urls() if mime is not None else []
        if urls:
            filepath = urls[0].toLocalFile()
            self.fileDropped.emit(filepath)


class MainWindow(QMainWindow):
    """Main window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION} - 怨좉퀬??硫붿돩 ?곷낯 ?꾧뎄")
        self.setMinimumSize(800, 600)
        self.current_file = None
        self.processing_thread = None
        
        self.init_ui()
    
    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # ?쒕ぉ
        title = QLabel(f"{APP_NAME} v{APP_VERSION}")
        title.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        subtitle = QLabel("3D 硫붿돩瑜??곷낯泥섎읆 ?쇱퀜二쇰뒗 ?꾧뎄")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #666; font-size: 14px;")
        layout.addWidget(subtitle)
        
        # ?쒕옒洹????쒕∼ ?곸뿭
        self.drop_area = DropArea()
        self.drop_area.fileDropped.connect(self.on_file_dropped)
        layout.addWidget(self.drop_area)
        
        # ?뚯씪 ?좏깮 踰꾪듉
        btn_layout = QHBoxLayout()
        self.btn_open = QPushButton("?뱛 ?뚯씪 ?좏깮")
        self.btn_open.setStyleSheet("""
            QPushButton {
                background-color: #4a90d9;
                color: white;
                border: none;
                padding: 12px 30px;
                font-size: 14px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
        """)
        self.btn_open.clicked.connect(self.open_file)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_open)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # ?뚯씪 ?뺣낫
        self.file_info = QLabel("")
        self.file_info.setStyleSheet("font-size: 12px; color: #333;")
        self.file_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.file_info)
        
        # 泥섎━ ?듭뀡
        options_group = QGroupBox("泥섎━ ?듭뀡")
        options_layout = QVBoxLayout(options_group)
        
        # 紐⑤뱶 ?좏깮
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("泥섎━ 紐⑤뱶:"))
        
        self.radio_flatten = QRadioButton("?곷낯 (?쇱튂湲?")
        self.radio_flatten.setChecked(True)
        self.radio_project = QRadioButton("?뺤궗?ъ쁺 (?됰㈃??")
        self.radio_separate = QRadioButton("?쒕㈃ 遺꾨━ (?대㈃/?몃㈃)")
        
        mode_layout.addWidget(self.radio_flatten)
        mode_layout.addWidget(self.radio_project)
        mode_layout.addWidget(self.radio_separate)
        mode_layout.addStretch()
        options_layout.addLayout(mode_layout)
        
        # ?댁긽??
        res_layout = QHBoxLayout()
        res_layout.addWidget(QLabel("異쒕젰 ?댁긽??"))
        self.spin_resolution = QSpinBox()
        self.spin_resolution.setRange(GUI_MIN_RESOLUTION, GUI_MAX_RESOLUTION)
        self.spin_resolution.setValue(DEFAULT_RENDER_RESOLUTION)
        self.spin_resolution.setSuffix(" px")
        res_layout.addWidget(self.spin_resolution)
        
        res_layout.addWidget(QLabel("  ?ㅽ???"))
        self.combo_style = QComboBox()
        self.combo_style.addItems(["traditional", "modern", "relief"])
        res_layout.addWidget(self.combo_style)
        res_layout.addStretch()
        options_layout.addLayout(res_layout)
        
        layout.addWidget(options_group)
        
        # 吏꾪뻾 ?곹깭
        progress_group = QGroupBox("吏꾪뻾 ?곹깭")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("?湲?以?..")
        self.status_label.setStyleSheet("color: #666;")
        progress_layout.addWidget(self.status_label)
        
        layout.addWidget(progress_group)
        
        # 泥섎━ 踰꾪듉
        self.btn_process = QPushButton("?? 泥섎━ ?쒖옉")
        self.btn_process.setEnabled(False)
        self.btn_process.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 15px 40px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:disabled {
                background-color: #ccc;
            }
        """)
        self.btn_process.clicked.connect(self.start_processing)
        layout.addWidget(self.btn_process)
        
        # 寃곌낵 濡쒓렇
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(100)
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("font-family: monospace; font-size: 11px;")
        layout.addWidget(self.log_text)
    
    def open_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "3D 硫붿돩 ?뚯씪 ?좏깮",
            "",
            "3D Files (*.obj *.ply *.stl *.off *.gltf *.glb);;All Files (*)"
        )
        if filepath:
            self.on_file_dropped(filepath)
    
    def on_file_dropped(self, filepath):
        self.current_file = filepath
        self.drop_area.setText(f"?좏깮???뚯씪:\n{Path(filepath).name}")
        
        # ?뚯씪 ?뺣낫 ?쒖떆
        try:
            from src.core.mesh_loader import MeshLoader
            loader = MeshLoader(default_unit=DEFAULT_MESH_UNIT)
            info = loader.get_file_info(filepath)

            n_vertices_text = _format_optional_count(info.get("n_vertices"))
            n_faces_text = _format_optional_count(info.get("n_faces"))
            file_size = info.get("file_size_mb")
            size_text = f"{file_size}" if isinstance(file_size, (int, float)) else "?"

            info_text = f"?뺤젏: {n_vertices_text}媛? |  "
            info_text += f"硫? {n_faces_text}媛? |  "
            info_text += f"?ш린: {size_text} MB"
            self.file_info.setText(info_text)

            self.log(f"?뚯씪 濡쒕뱶: {filepath}")
            self.log(f"  - Vertices: {n_vertices_text}")
            self.log(f"  - Faces: {n_faces_text}")
            
        except Exception as e:
            self.file_info.setText(f"?ㅻ쪟: {e}")
        
        self.btn_process.setEnabled(True)
    
    def start_processing(self):
        if not self.current_file:
            return
        
        # 紐⑤뱶 寃곗젙
        if self.radio_flatten.isChecked():
            mode = 'flatten'
        elif self.radio_project.isChecked():
            mode = 'project'
        else:
            mode = 'separate'
        
        options = {
            'resolution': self.spin_resolution.value(),
            'style': self.combo_style.currentText()
        }
        
        self.btn_process.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("泥섎━ ?쒖옉...")
        
        self.processing_thread = ProcessingThread(self.current_file, mode, options)
        self.processing_thread.progress.connect(self.on_progress)
        self.processing_thread.finished.connect(self.on_finished)
        self.processing_thread.start()
    
    def on_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
        self.log(message)
    
    def on_finished(self, success, result):
        self.btn_process.setEnabled(True)
        
        if success:
            self.status_label.setText("???꾨즺!")
            self.log(f"????λ맖: {result}")
            QMessageBox.information(self, "?꾨즺", f"泥섎━媛 ?꾨즺?섏뿀?듬땲??\n\n????꾩튂:\n{result}")
        else:
            self.status_label.setText("???ㅻ쪟 諛쒖깮")
            self.log(f"???ㅻ쪟: {result}")
            QMessageBox.critical(self, "?ㅻ쪟", f"泥섎━ 以??ㅻ쪟媛 諛쒖깮?덉뒿?덈떎:\n\n{result}")

        if self.processing_thread is not None:
            self.processing_thread.deleteLater()
            self.processing_thread = None
    
    def log(self, message):
        self.log_text.append(message)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
