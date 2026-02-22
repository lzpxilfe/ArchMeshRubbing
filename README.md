# ArchMeshRubbing

ArchMeshRubbing은 고고학/건축 유물 메쉬를 정렬, 단면화, 전개(Flatten), 탁본 이미지화, 실측 SVG 출력까지 한 번에 처리하는 도구입니다.

## 현재 핵심 워크플로우

1. 메쉬 파일(`.obj`, `.ply`, `.off`, `.gltf`, `.glb`)을 연다.
2. 바닥면 지정/맞춤으로 자세(정치)를 잡는다.
3. 필요하면 단면선/ROI를 지정한다.
4. 목적에 맞는 출력 경로를 선택한다.
   - 탁본 이미지
   - 디지털 탁본(상/하면 2장)
   - 현재뷰 원통 이미지(초고속)
   - 2D 실측 도면(SVG)
   - 통합 SVG(실측+단면+탁본)

## 주요 기능

- 메쉬 정렬/표시
  - 바닥면 피킹 및 바닥면 맞춤
  - 6방향 정규 뷰(정면/후면/좌/우/상/하)
  - X-Ray, Flat Shading 등 시각화 보조

- 단면/2D 지정
  - 실시간 단면 슬라이싱
  - 2D 단면선 지정(상면)
  - ROI 지정 후 외곽 추출
  - 단면/레이어 저장

- 펼침(Flatten)
  - ARAP, LSCM, 면적 보존, 원통 펼침
  - 곡률 반경/방향/반복 횟수 설정

- 이미지/도면 내보내기
  - 탁본 이미지(PNG/TIFF)
  - 디지털 탁본(상/하면 2장): 자동 내/외면 분리 + 원통 펼침 + 곡률 제거
    - 저장 파일: `<이름>_top.<확장자>`, `<이름>_bottom.<확장자>`
  - 현재뷰 원통 이미지 내보내기(초고속)
  - 정사투영 이미지 내보내기
  - 2D 실측 도면(SVG) 및 6방향 패키지 내보내기
  - 통합 SVG
    - 실측+단면+내/외면 탁본
    - 디지털 탁본/원통 버전

- 프로젝트 파일
  - `.amr` 프로젝트 저장/불러오기
  - 객체 상태/카메라/레이어/옵션 복원

## 설치

```bash
pip install -r requirements.txt
pip install -r requirements-optional.txt  # 선택
```

## 실행

GUI 실행:

```bash
python app_interactive.py
```

또는:

```bash
python main.py --gui
python main.py --open-project <project.amr>
```

CLI 도움말:

```bash
python main.py --help
```

## CLI 예시

```bash
python main.py <mesh_file>
python main.py --info <mesh_file>
python main.py --flatten <mesh_file> [output]
python main.py --project <mesh_file> [output]
python main.py --separate <mesh_file>
```

## 지원 포맷

- OBJ (`.obj`)
- PLY (`.ply`)
- OFF (`.off`)
- glTF (`.gltf`)
- glTF Binary (`.glb`)

## 주요 단축키

- `1`~`6`: 6방향 뷰
- `F`: 선택 객체 화면 맞춤
- `Ctrl+S`: 프로젝트 저장
- `Ctrl+Shift+O`: 프로젝트 열기

## 환경 변수

- `ARCHMESHRUBBING_EXPORT_DPI`
- `ARCHMESHRUBBING_RENDER_RESOLUTION`
- `ARCHMESHRUBBING_ARAP_MAX_ITERATIONS`
- `ARCHMESHRUBBING_GUI_MIN_RESOLUTION`
- `ARCHMESHRUBBING_GUI_MAX_RESOLUTION`
- `ARCHMESHRUBBING_PROFILE_EXPORT_SAFE`
- `ARCHMESHRUBBING_DISABLE_OPENCV`
- `ARCHMESHRUBBING_CV2_IMPORT_TIMEOUT`
- `ARCHMESHRUBBING_LOG_LEVEL`

## 참고 문서

- 레퍼런스 목록: `docs/REFERENCES.md`
- 기능-레퍼런스 매핑: `docs/FEATURE_REFERENCES.md`

## 라이선스

GNU General Public License v2.0 (GPLv2)
