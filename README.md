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

```bash
python main.py
```

## 지원 포맷

- OBJ (Wavefront)
- PLY (Polygon File Format)
- STL (Stereolithography)
- OFF (Object File Format)

## 라이선스

MIT License
