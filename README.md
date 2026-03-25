# 🏺 ArchMeshRubbing

ArchMeshRubbing은 고고학 3D 메쉬를 `기록면(recording surface)` 관점에서 다루는 데스크톱 도구입니다.

It is an archaeology-first desktop tool for turning 3D meshes into usable recording outputs rather than treating them as generic CG assets.

핵심은 단순한 mesh flatten이 아닙니다.

- `정위치`: 유물을 도면 기준에 맞게 놓기
- `실측용 도면`: 단면, 외곽, 기와 제작 가설을 정리하기
- `탁본`: 상면/하면 기록면을 준비하고 relief를 읽기 좋은 이미지로 만들기
- `제원측정`: 거리, 지름, 면적, 부피 같은 값을 확인하기

## ✨ Current Focus / 현재 방향

- `4축 기본 워크플로우`
  - `정위치 -> 실측용 도면 -> 탁본 -> 제원측정`
- `기록면(recording surface)` 중심 처리
  - triangle explode가 아니라 연속 표면 기록을 목표로 함
- `기와 제작형 해석`
  - 수키와/암키와
  - 2분할/4분할 가설
  - 길이축, 대표 단면, 와통 추정
- `탁본형 시각화`
  - 검토 시트(review sheet)
  - 기록면 전개 SVG
  - 6방향 실측 도면 패키지

## 🧭 Main Workflow / 메인 워크플로우

### 1. 정위치

- 메쉬 불러오기
- 바닥면 정렬
- 표준 6방향 뷰
- 프로젝트 저장/복원 (`.amr`)

### 2. 실측용 도면

- 기와 유형 / 분할 가설 입력
- 길이축 자동 추정
- 대표 단면 자동 준비
- 단면 프로파일 분석
- 와통 추정 실행
- 실측용 SVG 저장
- 6방향 도면 패키지 저장

### 3. 탁본

- 상면 기록 준비
- 하면 기록 준비
- 기록면 미리보기
- 기록면 검토 시트 저장

기록면 검토 시트와 미리보기는 전개된 2D 기록면 이미지에서 뒤에서 다음 렌더 모드를 지원합니다.

- `자동`
- `다중광(기록면)`
- `자연(이미지)+CLAHE`
- `로컬 대비(텍스처)`
- `하이브리드(형상+텍스처)`
- `텍스처 판독(실사)`
- `노멀 언샵`
- `스펙큘러 강조`
- `노멀 보기`
- `자연(이미지)`

기본 출력 패널에서 렌더 모드 위에 `질감 강조`, `추가 스무딩`, `텍스처 보정`을 덧붙여
기록면의 미세한 음각/양각 표현을 직접 조절할 수 있습니다.
텍스처가 포함된 메쉬는 `하이브리드`/`텍스처 판독` 모드에서 실제 UV 텍스처를 평면화 결과에 합성합니다.

### 4. 제원측정

- 거리 측정
- 지름/원호 보조
- 면적/부피 확인용 기본 측정 도구

## 🧱 Tile Workflow / 기와 해석 워크플로우

기와는 단순한 U자 곡면으로 보지 않고, 제작 과정을 가진 유물로 다룹니다.

- 수키와 / 암키와 / 미상
- 2분할 / 4분할 / 미상
- 길이축 힌트
- 대표 단면 후보
- 단면 프로파일 분석
- 와통 초벌 피팅
- 상면 / 하면 기록면 준비
- 작업 슬롯 저장 / 복원

기본 화면에서는 핵심 단계만 보이고,
수동 단면 조정, 연구용 라벨링, synthetic benchmark 같은 기능은 세부 토글 뒤에 숨겨 둡니다.

## 🖼️ Primary Outputs / 기본 산출물

기본 UI는 지금 아래 3가지를 중심으로 설계되어 있습니다.

- `기록면 검토 시트`
- `기록면 전개 SVG`
- `6방향 도면 패키지`

기본 UI에서 제거된 오래된 우회 출력은 코드상 호환 안내만 남아 있습니다.

## 🧪 Synthetic Benchmark / 합성 벤치마크

실제 기와 스캔이 부족해도 합성 데이터로 워크플로우를 검증할 수 있습니다.

- synthetic tile 생성
- ground truth 연결
- 현재 해석 상태 평가
- synthetic bundle 저장
- `preset × seed` benchmark suite 저장
- case별 `*.review.png` 생성
- threshold 기반 pass/fail 요약

자세한 내용은 [Synthetic Benchmark Guide](docs/SYNTHETIC_BENCHMARKS.md)를 참고하세요.

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

`python`이 없으면 `python3 app_interactive.py`를 사용하세요.

## 🖥️ CLI Examples / CLI 예시

```bash
python main.py --help
python main.py mesh.obj
python main.py --open-project sample.amr
python main.py --info mesh.obj
python main.py --flatten mesh.obj unwrap.png
python main.py --review mesh.obj review.png
python main.py --generate-synthetic sugkiwa_quarter 7 synthetic_tile.obj
python main.py --benchmark-synthetic ./benchmarks 1,2,3
python main.py --project mesh.obj planview.png
```

CLI의 `--flatten`과 `--review`는 `quick full-surface` 경로입니다.
상면/하면 기록면을 유도형으로 준비하려면 GUI를 쓰는 것이 좋습니다.

## 📚 Docs / 문서

- [합성 벤치마크 가이드 / Synthetic Benchmark Guide](docs/SYNTHETIC_BENCHMARKS.md)
- [레퍼런스 목록](docs/REFERENCES.md)
- [기능-레퍼런스 매핑](docs/FEATURE_REFERENCES.md)

## 🧭 Philosophy / 철학

이 프로젝트는 메쉬를 단순한 CG 자산으로 다루지 않습니다.

- 메쉬 체계: 정위치, 실측용 도면, 제원측정
- 기록면 체계: 탁본, 검토 시트, 전개 SVG

즉, 내부 계산은 메쉬를 써도
사용자가 다루는 결과는 `기록 가능한 표면`이어야 한다는 관점을 따릅니다.

## 📌 Current Status / 현재 상태

현재 버전은 다음에 집중하고 있습니다.

- 4축 기본 UI 정리
- 기와용 제작형 해석 흐름
- 기록면 검토 시트 중심 산출물
- 다중광/노멀 기반 탁본형 셰이딩
- synthetic benchmark 기반 회귀 검증

## 📄 License / 라이선스

GNU General Public License v2.0 (GPLv2)
