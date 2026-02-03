# ArchMeshRubbing

고고학 유물의 3D 스캔 메쉬를 평면으로 펼쳐 탁본과 유사한 이미지를 생성하는 도구입니다.

## 주요 기능

- **메쉬 평면화**: ARAP 알고리즘으로 곡면을 왜곡 최소화하며 펼침
- **정사투영**: 3D 메쉬의 평면도 생성
- **표면 분리**: 내면/외면 자동 감지 및 개별 처리
- **영역 선택**: 미구 등 특정 부분만 추출
- **스케일 출력**: 실측 스케일이 맞춰진 이미지 내보내기

## 설치

```bash
pip install -r requirements.txt
```

## 실행

### GUI (추천)

```bash
python app_interactive.py
```

### CLI

```bash
python main.py --help
```

## 2D 실측 도면(SVG) 내보내기

- GUI에서 `상/하/전/후/좌/우` 뷰를 본 뒤 **2D 실측 도면 내보내기(SVG)** 메뉴로 저장할 수 있습니다.

## 단축키

- `[` / `]`: 회전 기즈모 크기 줄이기/키우기
- `F`: 선택 객체 화면 맞춤
- `R`: 카메라 리셋

## 환경 변수 (트러블슈팅)

- `ARCHMESHRUBBING_PROFILE_EXPORT_SAFE=1`: SVG 외곽선/가이드 투영을 보수적으로 처리 (Illustrator에서 `격자 + 긴 직선`만 보일 때 권장)
- `ARCHMESHRUBBING_DISABLE_OPENCV=1`: OpenCV 비활성화 (SciPy 기반 경로 사용)
- `ARCHMESHRUBBING_CV2_IMPORT_TIMEOUT=2.0`: OpenCV import smoke-test 타임아웃(초)

## 지원 포맷

- OBJ (Wavefront)
- PLY (Polygon File Format)
- STL (Stereolithography)
- OFF (Object File Format)

## 라이선스

MIT License
