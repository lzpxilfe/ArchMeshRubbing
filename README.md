# 🏺 ArchMeshRubbing

ArchMeshRubbing은 고고학·건축 유물의 3D 메쉬를 정렬하고, 단면을 확인하고, 기록면을 전개하고, 탁본 스타일 이미지와 실측 SVG로 내보내기 위한 데스크톱 도구입니다.

ArchMeshRubbing is a desktop tool for aligning archaeological 3D meshes, inspecting sections, unwrapping recording surfaces, and exporting rubbing-style images and measured SVG outputs.

이 프로젝트는 특히 기와, 곡면 파편, 얇은 쉘 구조처럼 내면/외면이 중요한 유물을 다루는 워크플로우를 목표로 합니다.

## ✨ 할 수 있는 일

- 🧭 메쉬 정렬
  - 바닥면 지정 및 자세 맞춤
  - 6방향 정규 뷰
  - X-Ray, Flat Shading 등 표시 옵션

- ✂️ 단면과 윤곽 확인
  - 실시간 단면 슬라이싱
  - ROI 기반 외곽 추출
  - 절단선/단면선 저장

- 🗺️ 메쉬 펼침
  - ARAP
  - LSCM
  - 면적 보존 계열
  - 원통 펼침
  - 단면 기반 펼침(기와/U자형 곡면 대응용)

- 🪨 표면 분리
  - 내면/외면 자동 분리
  - 미구(thickness/edge faces) 추정 보조

- 🖼️ 결과물 내보내기
  - 탁본 이미지 `PNG / TIFF / JPEG`
  - 디지털 탁본(상/하면 2장)
  - 현재 뷰 기반 원통 이미지
  - 정사투영 이미지
  - 펼침 결과 SVG
  - 실측 SVG
  - 통합 SVG(실측 + 단면 + 탁본)
  - 내면/외면/펼쳐진 메쉬 저장

- 💾 프로젝트 저장
  - `.amr` 프로젝트 저장/불러오기
  - 카메라, 객체, 레이어, 일부 옵션 복원

## 🧪 핵심 워크플로우

1. 메쉬 파일을 연다.
2. 바닥면을 맞추고 자세를 정리한다.
3. 필요하면 단면선, ROI, 표면 선택 정보를 지정한다.
4. 펼침 방법을 선택한다.
5. 탁본 이미지, 실측 SVG, 통합 SVG 등 필요한 결과를 저장한다.

## 🧪 Synthetic Benchmark / 합성 벤치마크

실제 기와 샘플이 없더라도, 프로그램 안에서 `합성 기와(synthetic tile)`를 생성해 기준 데이터를 만들 수 있습니다. 이 synthetic workflow는 와통 반경, 2분할/4분할, 길이축, 대표 단면, 기록면(top/bottom) 정답이 포함된 회귀 세트를 만드는 데 사용됩니다.

Even without real tile scans, you can generate synthetic roof-tile meshes to build a baseline regression suite. Each synthetic bundle can include the mesh, ground-truth interpretation, evaluation report, and a recording-surface review sheet.

### CLI 예시 / CLI examples

```bash
python main.py --generate-synthetic sugkiwa_quarter 7 synthetic_tile.obj
python main.py --benchmark-synthetic ./benchmarks 1,2,3
```

생성 결과에는 보통 다음 파일이 포함됩니다.

The generated bundle usually contains:

- `*.obj`
- `*.truth.json`
- `*.interpretation.json`
- `*.evaluation.json`
- `*.review.png`
- `*.bundle.json`

benchmark suite를 만들면 `synthetic_benchmark_summary.json`과 `synthetic_benchmark_summary.csv`도 함께 저장됩니다.

When you build a benchmark suite, `synthetic_benchmark_summary.json` and `synthetic_benchmark_summary.csv` are also written.

## 📦 지원 포맷

- `OBJ`
- `PLY`
- `STL`
- `OFF`
- `glTF (.gltf)`
- `glTF Binary (.glb)`

## 🚀 빠른 시작

### Windows

```bat
py -3 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-optional.txt
python app_interactive.py
```

또는:

```bat
python main.py --gui
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-optional.txt
python app_interactive.py
```

`python` 명령이 없으면 `python3 app_interactive.py`를 사용하세요.

## 🖥️ 실행 방법

### GUI

```bash
python app_interactive.py
```

또는:

```bash
python main.py --gui
python main.py --open-project sample.amr
```

### CLI

```bash
python main.py --help
python main.py --info mesh.obj
python main.py --flatten mesh.obj
python main.py --project mesh.obj
python main.py --separate mesh.obj
```

## 🛠️ 설치 의존성

### 기본

```bash
pip install -r requirements.txt
```

포함되는 주요 패키지:

- `numpy`
- `scipy`
- `trimesh`
- `Pillow`
- `PyQt6`
- `PyOpenGL`

### 선택

```bash
pip install -r requirements-optional.txt
```

선택 패키지는 OpenCV 기반 외곽 추출과 일부 성능 개선에 도움을 줍니다.

## 📁 주요 결과물

- `*.rubbing.png`
  - 기본 탁본 이미지

- `*.projection.png`
  - 정사투영 이미지

- `*.inner.ply`
  - 내면 메쉬

- `*.outer.ply`
  - 외면 메쉬

- `*.amr`
  - 프로젝트 파일

디지털 탁본(상/하면 2장) 저장 시에는 보통 다음처럼 생성됩니다.

- `<name>_top.<ext>`
- `<name>_bottom.<ext>`

## ⌨️ 자주 쓰는 단축키

- `1`~`6`: 6방향 뷰
- `F`: 선택 객체 화면 맞춤
- `Ctrl+S`: 프로젝트 저장
- `Ctrl+Shift+O`: 프로젝트 열기

## ⚙️ 환경 변수

- `ARCHMESHRUBBING_EXPORT_DPI`
- `ARCHMESHRUBBING_RENDER_RESOLUTION`
- `ARCHMESHRUBBING_ARAP_MAX_ITERATIONS`
- `ARCHMESHRUBBING_GUI_MIN_RESOLUTION`
- `ARCHMESHRUBBING_GUI_MAX_RESOLUTION`
- `ARCHMESHRUBBING_PROFILE_EXPORT_SAFE`
- `ARCHMESHRUBBING_DISABLE_OPENCV`
- `ARCHMESHRUBBING_CV2_IMPORT_TIMEOUT`
- `ARCHMESHRUBBING_LOG_LEVEL`

## 🧭 문제 해결

### `python: command not found`

macOS / Linux에서는 `python` 대신 `python3`만 있는 경우가 많습니다.

```bash
python3 app_interactive.py
```

### `ModuleNotFoundError: No module named 'PyQt6'`

GUI 의존성이 설치되지 않은 상태입니다.

```bash
pip install -r requirements.txt
```

### OpenCV 없이 실행하고 싶을 때

OpenCV는 선택 사항입니다. 설치하지 않아도 대부분 기능은 동작합니다.

```bash
pip install -r requirements.txt
```

필요하면 나중에만:

```bash
pip install -r requirements-optional.txt
```

## 📚 문서

- [레퍼런스 목록](docs/REFERENCES.md)
- [기능-레퍼런스 매핑](docs/FEATURE_REFERENCES.md)
- [합성 벤치마크 가이드 / Synthetic Benchmark Guide](docs/SYNTHETIC_BENCHMARKS.md)

## 📌 현재 상태

이 프로젝트는 연구/제작 워크플로우를 빠르게 반복하기 위한 도구로 계속 개선 중입니다. 특히 곡면 유물의 기록면 전개 품질, 기와류 단면 기반 전개, synthetic benchmark 기반 회귀 검증을 계속 다듬는 중입니다.

This project is under active development as a research-first archaeology tool. Current focus areas include recording-surface unwrap quality for curved artifacts, fabrication-aware tile workflows, and synthetic-benchmark-based regression checks.

## 📄 라이선스

GNU General Public License v2.0 (GPLv2)
