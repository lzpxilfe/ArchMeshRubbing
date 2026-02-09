"""
ArchMeshRubbing GUI - Main Window
간단한 드래그 앤 드롭 인터페이스
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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

APP_NAME = "ArchMeshRubbing"
APP_VERSION = "0.1.0"
try:
    import src as _amr_src

    APP_VERSION = str(getattr(_amr_src, "__version__", APP_VERSION))
except Exception:
    pass


class ProcessingThread(QThread):
    """백그라운드 처리 스레드"""
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
            
            # 1. 로드
            self.progress.emit(10, "메쉬 로딩 중...")
            loader = MeshLoader()
            mesh = loader.load(self.filepath)
            
            self.progress.emit(20, f"로드 완료: {mesh.n_vertices:,} 정점, {mesh.n_faces:,} 면")
            
            output_path = Path(self.filepath)
            
            if self.mode == 'flatten':
                # 평면화
                self.progress.emit(30, "ARAP 평면화 중...")
                flattener = ARAPFlattener(max_iterations=30)
                flattened = flattener.flatten(mesh)
                
                self.progress.emit(60, "탁본 이미지 생성 중...")
                visualizer = SurfaceVisualizer(default_dpi=300)
                rubbing = visualizer.generate_rubbing(
                    flattened, 
                    width_pixels=self.options.get('resolution', 2000),
                    style=self.options.get('style', 'traditional')
                )
                
                self.progress.emit(90, "저장 중...")
                save_path = output_path.with_suffix('.rubbing.png')
                rubbing.save(str(save_path), include_scale_bar=True)
                
                self.progress.emit(100, "완료!")
                self.finished.emit(True, str(save_path))
                
            elif self.mode == 'project':
                # 정사투영
                self.progress.emit(40, "정사투영 생성 중...")
                projector = OrthographicProjector(resolution=self.options.get('resolution', 2000))
                aligned = projector.align_mesh(mesh, method='pca')
                result = projector.project(aligned, direction='top', render_mode='depth')
                
                self.progress.emit(90, "저장 중...")
                save_path = output_path.with_suffix('.projection.png')
                result.save(str(save_path), dpi=300)
                
                self.progress.emit(100, "완료!")
                self.finished.emit(True, str(save_path))
                
            elif self.mode == 'separate':
                # 표면 분리
                self.progress.emit(50, "표면 분리 중...")
                separator = SurfaceSeparator()
                result = separator.auto_detect_surfaces(mesh)
                
                self.progress.emit(80, "저장 중...")
                
                saved_files = []
                if result.inner_surface:
                    inner_path = output_path.with_suffix('.inner.ply')
                    result.inner_surface.to_trimesh().export(str(inner_path))
                    saved_files.append(str(inner_path))
                
                if result.outer_surface:
                    outer_path = output_path.with_suffix('.outer.ply')
                    result.outer_surface.to_trimesh().export(str(outer_path))
                    saved_files.append(str(outer_path))
                
                self.progress.emit(100, "완료!")
                self.finished.emit(True, ", ".join(saved_files))
            
        except Exception as e:
            self.finished.emit(False, str(e))


class DropArea(QLabel):
    """드래그 앤 드롭 영역"""
    fileDropped = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("3D 파일을 여기에 끌어다 놓으세요\n\n(.obj, .ply, .stl, .off)")
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
    """메인 윈도우"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION} - 고고학 메쉬 탁본 도구")
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
        
        # 제목
        title = QLabel(f"{APP_NAME} v{APP_VERSION}")
        title.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        subtitle = QLabel("3D 메쉬를 탁본처럼 펼쳐주는 도구")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #666; font-size: 14px;")
        layout.addWidget(subtitle)
        
        # 드래그 앤 드롭 영역
        self.drop_area = DropArea()
        self.drop_area.fileDropped.connect(self.on_file_dropped)
        layout.addWidget(self.drop_area)
        
        # 파일 선택 버튼
        btn_layout = QHBoxLayout()
        self.btn_open = QPushButton("📂 파일 선택")
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
        
        # 파일 정보
        self.file_info = QLabel("")
        self.file_info.setStyleSheet("font-size: 12px; color: #333;")
        self.file_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.file_info)
        
        # 처리 옵션
        options_group = QGroupBox("처리 옵션")
        options_layout = QVBoxLayout(options_group)
        
        # 모드 선택
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("처리 모드:"))
        
        self.radio_flatten = QRadioButton("탁본 (펼치기)")
        self.radio_flatten.setChecked(True)
        self.radio_project = QRadioButton("정사투영 (평면도)")
        self.radio_separate = QRadioButton("표면 분리 (내면/외면)")
        
        mode_layout.addWidget(self.radio_flatten)
        mode_layout.addWidget(self.radio_project)
        mode_layout.addWidget(self.radio_separate)
        mode_layout.addStretch()
        options_layout.addLayout(mode_layout)
        
        # 해상도
        res_layout = QHBoxLayout()
        res_layout.addWidget(QLabel("출력 해상도:"))
        self.spin_resolution = QSpinBox()
        self.spin_resolution.setRange(500, 8000)
        self.spin_resolution.setValue(2000)
        self.spin_resolution.setSuffix(" px")
        res_layout.addWidget(self.spin_resolution)
        
        res_layout.addWidget(QLabel("  스타일:"))
        self.combo_style = QComboBox()
        self.combo_style.addItems(["traditional", "modern", "relief"])
        res_layout.addWidget(self.combo_style)
        res_layout.addStretch()
        options_layout.addLayout(res_layout)
        
        layout.addWidget(options_group)
        
        # 진행 상태
        progress_group = QGroupBox("진행 상태")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("대기 중...")
        self.status_label.setStyleSheet("color: #666;")
        progress_layout.addWidget(self.status_label)
        
        layout.addWidget(progress_group)
        
        # 처리 버튼
        self.btn_process = QPushButton("🚀 처리 시작")
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
        
        # 결과 로그
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(100)
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("font-family: monospace; font-size: 11px;")
        layout.addWidget(self.log_text)
    
    def open_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "3D 메쉬 파일 선택",
            "",
            "3D Files (*.obj *.ply *.stl *.off *.gltf *.glb);;All Files (*)"
        )
        if filepath:
            self.on_file_dropped(filepath)
    
    def on_file_dropped(self, filepath):
        self.current_file = filepath
        self.drop_area.setText(f"선택된 파일:\n{Path(filepath).name}")
        
        # 파일 정보 표시
        try:
            from src.core.mesh_loader import MeshLoader
            loader = MeshLoader()
            info = loader.get_file_info(filepath)
            
            info_text = f"정점: {info.get('n_vertices', '?'):,}개  |  "
            info_text += f"면: {info.get('n_faces', '?'):,}개  |  "
            info_text += f"크기: {info.get('file_size_mb', '?')} MB"
            self.file_info.setText(info_text)
            
            self.log(f"파일 로드: {filepath}")
            self.log(f"  - 정점: {info.get('n_vertices', '?'):,}개")
            self.log(f"  - 면: {info.get('n_faces', '?'):,}개")
            
        except Exception as e:
            self.file_info.setText(f"오류: {e}")
        
        self.btn_process.setEnabled(True)
    
    def start_processing(self):
        if not self.current_file:
            return
        
        # 모드 결정
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
        self.status_label.setText("처리 시작...")
        
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
            self.status_label.setText("✅ 완료!")
            self.log(f"✅ 저장됨: {result}")
            QMessageBox.information(self, "완료", f"처리가 완료되었습니다!\n\n저장 위치:\n{result}")
        else:
            self.status_label.setText("❌ 오류 발생")
            self.log(f"❌ 오류: {result}")
            QMessageBox.critical(self, "오류", f"처리 중 오류가 발생했습니다:\n\n{result}")
    
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
