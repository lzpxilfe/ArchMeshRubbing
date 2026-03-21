# 🏺 ArchMeshRubbing

ArchMeshRubbing은 고고학 3D 메쉬를 `기록면(recording surface)` 관점에서 다루기 위한 데스크톱 도구입니다.  
메쉬를 단순히 펼치는 것이 아니라, 연구자가 실제로 기록하고 싶은 표면을 정렬하고, 확인하고, 전개하고, 검토 시트와 SVG로 남기는 흐름을 목표로 합니다.

ArchMeshRubbing is a desktop tool for archaeology-first 3D mesh work.  
Instead of treating the mesh as generic CG geometry, it focuses on recording surfaces that archaeologists want to inspect, unwrap, review, and export.

## ✨ What It Does / 무엇을 할 수 있나

### 🧭 1. 3D 기록 작업의 기본 흐름

- 3D 메쉬 불러오기
- 바닥면 정렬과 자세 교정
- `상/하/전/후/좌/우` 6방향 정규 뷰
- 단면 슬라이싱, ROI, 외곽 확인
- 현재 시점의 가시면 선택
- 프로젝트 저장/복원 (`.amr`)

### 🗺️ 2. 기록면 전개 (Recording-Surface Unwrap)

- ARAP
- LSCM
- 면적 보존 계열
- 원통 전개
- `단면 기반 section-wise 전개`
- 기와용 `상면 / 하면 기록면` 자동 준비
- `연속 기록면` 미리보기
- 기록면 검토 시트(review sheet) 저장

### 🧱 3. 기와 모드 (Tile Interpretation Workflow)

- 수키와 / 암키와 / 미상
- 2분할 / 4분할 / 미상 가설
- 길이축 힌트 추정
- 대표 단면 후보 생성과 채택
- 단면 프로파일 분석
- 와통 반경 초벌 피팅
- 작업 슬롯 저장 / 복원
- 슬롯별 검토 시트 묶음 저장

### 🧪 4. Synthetic Benchmark

실제 기와 스캔이 부족해도 synthetic tile을 생성해 기준 데이터를 만들 수 있습니다.

You can generate synthetic roof-tile cases even when real scans are not yet available.

- 합성 기와 1건 생성
- 합성 정답(ground truth) 연결
- 현재 해석 상태 평가
- 정답 가설 즉시 복원
- synthetic bundle 저장
- `preset × seed` benchmark suite 생성
- case별 `review sheet` 저장
- pass/fail threshold 기반 회귀 점검

### 🖼️ 5. 산출물

- 탁본형 이미지 `PNG / TIFF / JPEG`
- 디지털 탁본
- 기록면 검토 시트 `*.review.png`
- 펼침 결과 SVG
- 실측 SVG
- 통합 SVG
- 정사투영 이미지
- 프로젝트 파일 `*.amr`
- synthetic benchmark sidecar JSON/CSV

## 🧪 Main Workflows / 핵심 작업 흐름

### A. 범용 기록 workflow

1. 메쉬를 연다.
2. 바닥면과 자세를 정리한다.
3. 6방향 뷰 또는 ROI로 기록할 표면을 확인한다.
4. 기록면 전개를 실행한다.
5. 검토 시트, 탁본 이미지, SVG를 저장한다.

### B. 기와 workflow

1. 기와 유형과 분할 가설을 정한다.
2. 길이축 힌트를 잡는다.
3. 대표 단면 후보를 만든다.
4. 단면 프로파일과 와통 반경을 본다.
5. 상면 또는 하면 기록면을 준비한다.
6. 전개, 검토 시트, SVG를 저장한다.

### C. synthetic benchmark workflow

1. preset과 seed를 고른다.
2. synthetic tile 또는 benchmark suite를 생성한다.
3. 평가 점수와 실패 케이스를 확인한다.
4. `*.review.png`를 보며 seam, 단면 가이드, 기록면 방향을 점검한다.

## 📦 Supported Formats / 지원 포맷

- `OBJ`
- `PLY`
- `STL`
- `OFF`
- `glTF (.gltf)`
- `glTF Binary (.glb)`

## 🚀 Quick Start / 빠른 시작

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

## 🖥️ CLI Examples / CLI 예시

```bash
python main.py --help
python main.py --info mesh.obj
python main.py --flatten mesh.obj
python main.py --review mesh.obj review.png
python main.py --generate-synthetic sugkiwa_quarter 7 synthetic_tile.obj
python main.py --benchmark-synthetic ./benchmarks 1,2,3
python main.py --project mesh.obj planview.png
```

## 🧾 Typical Outputs / 대표 결과물

- `*.rubbing.png`
- `*.review.png`
- `*.projection.png`
- `*.svg`
- `*.amr`
- `*.truth.json`
- `*.interpretation.json`
- `*.evaluation.json`
- `synthetic_benchmark_summary.json`
- `synthetic_benchmark_summary.csv`

## ⌨️ Shortcuts / 자주 쓰는 단축키

- `1`~`6`: 6방향 뷰
- `F`: 선택 객체 화면 맞춤
- `Ctrl+S`: 프로젝트 저장
- `Ctrl+Shift+O`: 프로젝트 열기

## 🛠️ Dependencies / 설치 의존성

### 기본

```bash
pip install -r requirements.txt
```

주요 패키지:

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

선택 패키지는 OpenCV 기반 보조 기능과 일부 성능 개선에 도움을 줍니다.

## 🧭 Troubleshooting / 문제 해결

### `python: command not found`

macOS / Linux에서는 `python3`만 있는 경우가 많습니다.

```bash
python3 app_interactive.py
```

### `ModuleNotFoundError: No module named 'PyQt6'`

GUI 의존성이 설치되지 않은 상태입니다.

```bash
pip install -r requirements.txt
```

### OpenCV 없이 실행하고 싶을 때

OpenCV는 선택 사항입니다.

```bash
pip install -r requirements.txt
```

필요할 때만:

```bash
pip install -r requirements-optional.txt
```

## 📚 Docs / 문서

- [레퍼런스 목록](docs/REFERENCES.md)
- [기능-레퍼런스 매핑](docs/FEATURE_REFERENCES.md)
- [합성 벤치마크 가이드 / Synthetic Benchmark Guide](docs/SYNTHETIC_BENCHMARKS.md)

## 📌 Current Status / 현재 상태

이 프로젝트는 `고고학 기록면 전개 도구`로 계속 진화 중입니다.  
현재는 특히 다음 영역을 강하게 밀고 있습니다.

- 곡면 유물의 연속 기록면 전개
- 기와 제작형을 고려한 section-wise workflow
- synthetic benchmark 기반 회귀 검증
- 검토 시트 중심의 연구용 산출물

This project is still evolving as an archaeology-first recording-surface tool, with current focus on continuous unwraps for curved artifacts, fabrication-aware tile workflows, synthetic-benchmark regression, and review-sheet-centered outputs.

## 📄 License / 라이선스

GNU General Public License v2.0 (GPLv2)
