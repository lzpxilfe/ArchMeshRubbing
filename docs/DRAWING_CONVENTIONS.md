# Drawing Conventions

이 문서는 실측 도면의 선 종류 어휘와, 각 선 종류에 어떤 굵기·파선을 쓸지 정하는 preset을 다룬다. 구현은 [`src/core/drawing_style.py`](../src/core/drawing_style.py)에 있다.

---

## 왜 별도 계층인가

측정 결과와 표현은 다른 것이다. path의 좌표는 record가 소유하고 hash로 봉인되며 preset이 건드릴 수 없다. preset이 정하는 것은 선 굵기, 파선 간격, 채움뿐이고, 이 값들은 1:1 출력 기준의 **종이 밀리미터**다.

그래서 preset을 바꿔도 측정값은 절대 움직이지 않는다. 반대로 preset이 바뀌면 도면의 바이트는 바뀌므로, 도면은 자기가 그려질 때 쓴 preset의 canonical 해시를 sidecar에 기록한다. 나중에 preset 값을 고치면 예전 도면의 검증이 통과하지 않고 실패한다. 조용히 다른 굵기로 다시 그려지는 일은 없다.

---

## 선 종류 어휘

닫힌 집합이다. `LINE_KINDS`의 순서가 그리는 순서이자 SVG 레이어 순서이며, 이 순서 덕분에 같은 record는 언제 그려도 같은 바이트가 된다.

| 선 종류 | 뜻 | 지금 만드는 곳 |
|---|---|---|
| `section_cut` | 단면 절단선. 닫힌 경우 내부를 해칭한다 | `vector.cutline.v1`의 `section` role |
| `outline_visible` | 보이는 외형선 | `vector.outline.v1`의 `exterior` role |
| `outline_hole` | 유물 내부의 구멍 경계 | `vector.outline.v1`의 `hole` role |
| `center_axis` | 회전축·대칭축의 일점쇄선 | 아직 생산자 없음 |

새 선 종류는 **그것을 실제로 만들어내는 것이 생긴 뒤에** 추가한다. 만들 수 없는 종류를 먼저 이름 붙이면 모든 도면에 빈 레이어가 생기고, preset이 아무도 지키지 않는 관례를 서술하게 된다.

record의 role 이름(`section`, `exterior`, `hole`)은 payload 해시의 일부라 절대 바뀌지 않는다. role에서 선 종류로 가는 매핑은 표현이므로 `RECORD_ROLE_LINE_KINDS`에 있다.

---

## preset과 출처

`preset.source_id`가 `null`이면 **잠정값**이다. 공개된 지침이 뒷받침하지 않는, 그저 1:1 출력에서 읽히도록 고른 값이라는 뜻이다. 이 사실은 preset의 `provisional` 필드로 도면 sidecar에까지 그대로 실린다. 도면을 받은 사람이 그 값이 표준인지 아닌지 확인할 수 있어야 하기 때문이다.

지금 배포되는 preset은 `provisional/v1` 하나이고, 모든 값이 잠정값이다.

### 출처 있는 preset 추가하기

1. 공개 원문에서 수치를 옮겨 적는다. 예를 들어 국립문화유산연구원의 유물 실측 도면 작성 지침, 발굴조사 보고서 작성 기준, 한국고고학회의 도면 작성 기준, 또는 공개된 논문. **기억이나 상용 프로그램 화면에서 옮기지 않는다.**
2. `docs/REFERENCES.md`에 출처를 등재하고 ID를 받는다.
3. `src/core/drawing_style.py`의 `_PRESETS`에 새 항목을 추가한다. `preset_id`는 `기관-지침-연도/v1`처럼 무엇을 따르는지 알 수 있게 짓고, `source_id`에 2단계의 ID를 넣는다.
4. 커밋 메시지에 어느 원문의 어느 항목에서 각 수치를 옮겼는지 적는다.

기존 preset의 값은 고치지 않고 새 preset을 추가한다. 값을 고치면 그 preset으로 만든 기존 도면이 전부 검증에 실패한다.

### 클린룸 원칙

이 프로젝트는 상용 실측 프로그램의 코드나 디컴파일 결과를 보지 않는다. 도면 관례는 공개 지침·논문·공개 표준에서만 가져온다. 상용 제품 고유의 기능 명칭, 아이콘, 색 프로필 이름은 사용하지 않는다. 도면 관례는 특정 회사의 것이 아니라 학계가 공유하는 것이므로, 공개 원문을 인용하는 한 이 경계는 지켜진다.

---

## 현재 잠정값

`provisional/v1`. 모든 값이 종이 밀리미터, 1:1 기준이다.

| 선 종류 | 굵기 (mm) | 파선 (mm) | 채움 |
|---|---|---|---|
| `section_cut` | 0.35 | - | 해칭 |
| `outline_visible` | 0.25 | - | - |
| `outline_hole` | 0.25 | - | - |
| `center_axis` | 0.13 | 4 - 1 - 1 - 1 (일점쇄선) | - |

해칭: 45°, 간격 1.0 mm, 선 굵기 0.13 mm.

출처 ID: 없음. 위 4단계에 따라 채워야 한다.

---

## 사용법

```python
from src.core.artifact_vector_export import VectorSVGOptions, build_vector_export

bundle = build_vector_export(
    document,
    "record:outline-top",
    options=VectorSVGOptions(style_preset="provisional/v1"),
)
```

`style_preset`을 주지 않으면 preset 이전과 **바이트 단위로 같은** 단일 굵기 도면이 나온다. 이 기본값은 의도적이다. preset이 생기기 전에 만들어진 패키지는 자기 SVG를 검증 시점에 다시 렌더해 바이트를 대조하므로, 기본 렌더링이 바뀌면 그 패키지들이 한꺼번에 검증 불가가 된다.

preset을 주면 SVG는 선 종류별 `<g id="layer-...">` 레이어로 나뉘고, Illustrator나 Inkscape에서 레이어로 분리되어 열린다.

---

## 실측 도판 (sheet)

한 기록의 1:1 SVG는 측정 결과다. 도판은 독자가 받는 페이지다. 입면도와 단면도를 나란히 놓고, 축척에 맞게 줄이고, 축척바와 제목란을 붙인 것. 구현은 [`src/core/drawing_sheet.py`](../src/core/drawing_sheet.py)에 있다.

```python
from src.core.drawing_sheet import (
    DrawingSheetOptions, SheetPage, TitleBlock, compose_drawing_sheet,
)

bundle = compose_drawing_sheet(
    document,
    ["record:outline-front", "record:cutline-a"],
    options=DrawingSheetOptions(
        title_block=TitleBlock(artifact_label="청자 대접", rows=(("작성", "홍길동"),)),
        scale_denominator=3,
        page=SheetPage(size="A4", orientation="portrait"),
    ),
)
```

앱에서는 단면 패널의 "실측 도판" 항목에서 체크박스로 기록을 고르고, 용지와 축척을 정한 뒤 버튼을 누른다. **체크한 순서가 배치 순서**이므로, 입면도를 왼쪽에 두고 싶으면 입면도를 먼저 체크한다.

### 세 가지 성질

**축척은 항상 인쇄되고, 호출자가 덮어쓸 수 없다.** 제목란의 축척 행은 실제 배치에서 파생되므로 도면이 그려진 축척과 다른 값을 말할 수 없다. `TitleBlock.rows`에 `("축척", "1:99")`를 넣어도 파생된 행이 따로 들어간다. 줄인 도면이 얼마나 줄었는지 말하지 않으면 종이에서 잴 수가 없다.

**선 굵기는 어떤 축척에서도 종이 mm 그대로다.** 좌표만 축척 분모로 나누고, 선 굵기·파선 길이·해칭 간격은 나누지 않는다. 0.35 mm 절단선은 1:1에서도 1:4에서도 종이에서 0.35 mm다. preset의 수치를 종이 mm로 잡아둔 이유가 이것이다.

**들어가지 않으면 조용히 줄이지 않는다.** 요청한 축척으로 페이지에 안 들어가면 실패하고, 쓸 수 있는 축척 분모를 알려준다. 알려주는 값은 항상 올림이라 그대로 다시 시도하면 반드시 성공한다. 자동으로 줄였다면 "1:2"라고 적힌 종이가 실제로는 다른 비율로 인쇄된다.

### 페이지와 배치

용지는 A5 · A4 · A3 · A2 · A1, 세로·가로. 여백 기본값은 12 mm다.

배치는 준 순서대로 왼쪽에서 오른쪽, 한 줄이 차면 다음 줄이다. 페이지 아래쪽에는 축척바와 제목란을 위한 띠를 **배치 전에** 폭 전체로 확보한다. 제목란 너비에 따라 달라지는 규칙은 라벨이 길어지는 순간 깨지기 때문이다.

축척바는 1 · 2 · 5 × 10ⁿ mm 중에서 종이 위 길이가 90 mm를 넘지 않는 가장 긴 것을 고르고, 4칸으로 나눠 번갈아 칠한다. 라벨 단위는 크기에 따라 mm · cm · m로 바뀐다. 1 m를 "100 cm"라고 쓰지는 않는다.

### 도판은 측정값이 아니다

도판은 표현물이다. 각 도형은 자기가 그린 payload의 해시를 sidecar에 기록하므로 어떤 기록으로 만든 도판인지 확인할 수 있지만, 도판 자체가 측정 권위가 되지는 않는다. 그래서 원자 staging을 쓰는 export controller를 거치지 않고 바로 파일로 쓴다. 실질적인 관문은 그대로다. READY이고 FRESH가 아닌 기록은 도판에 오르지 못한다.

---

## 아직 없는 것

- 토기의 좌 반입면 · 우 반단면 미러 배치. 중심축이 필요하고, `center_axis`를 만들어내는 것이 아직 없다.
- 결실 · 복원 · 균열 같은 상태 표기. `annotation.condition.v1` record가 생긴 뒤의 일이다.
- DXF 내보내기.
