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
| `condition_missing` | 결실부 경계 | `annotation.condition.v1`, kind `missing` |
| `condition_restored` | 복원부 경계 | `annotation.condition.v1`, kind `restored` |
| `condition_worn` | 마모부 경계 | `annotation.condition.v1`, kind `worn` |
| `condition_crack` | 균열 경계 | `annotation.condition.v1`, kind `crack` |
| `technique_groove_edge` | 홈의 튀어나온 능선 (직선) | `measurement.profile_groove.v1` |
| `technique_groove_trough` | 홈의 들어간 골 (간선) | `measurement.profile_groove.v1` |
| `center_axis` | 회전축·대칭축의 일점쇄선 | `rotation_axis_from_circle_records/v1` Align (record가 아니라 정치의 결과) |

새 선 종류는 **그것을 실제로 만들어내는 것이 생긴 뒤에** 추가한다. 만들 수 없는 종류를 먼저 이름 붙이면 모든 도면에 빈 레이어가 생기고, preset이 아무도 지키지 않는 관례를 서술하게 된다.

record의 role 이름(`section`, `exterior`, `hole`)은 payload 해시의 일부라 절대 바뀌지 않는다. role에서 선 종류로 가는 매핑은 표현이므로 `RECORD_ROLE_LINE_KINDS`에 있다.

상태 표기는 role이 아니라 **record의 kind**로 선 종류가 정해진다(`CONDITION_LINE_KINDS`). 상태 경계도 outline payload로 저장되어 path의 role이 `exterior`·`hole`이기 때문에, role로 그리면 결실부가 유물의 외형선으로 인쇄된다.

기법 선 종류는 외형선 뒤, 상태 앞에 놓았다. 기법은 표면 위에 있으므로 외형선 위에 그려지고, 상태는 그 도면을 어디까지 읽을 수 있는지를 말하므로 기법 위에 그려진다. 복원된 부분에 기법 선이 가려지는 것은 옳은 결과다 — 현대의 복원면에서 제작 기법을 읽어서는 안 된다.

상태 선 종류는 외형선 뒤, 중심축 앞에 놓았다. 자기가 설명하는 형태 위에 그려져야 하고, 구조선인 중심축은 그 모두 위에서 읽혀야 한다. 상태 넷 안에서는 면 성격의 셋(`missing` · `restored` · `worn`)이 먼저이고 `condition_crack`이 마지막이다. 균열은 유물 위의 선이고, 면 밑에 깔린 선은 독자가 잃어버린다.

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
| `condition_missing` | 0.25 | 1.5 - 1.5 | - |
| `condition_restored` | 0.25 | 3 - 1 - 0.5 - 1 | - |
| `condition_worn` | 0.18 | 0.5 - 0.5 | - |
| `condition_crack` | 0.3 | - | - |
| `technique_groove_edge` | 0.18 | - | - |
| `technique_groove_trough` | 0.18 | - (기하로 끊는다) | - |
| `center_axis` | 0.13 | 4 - 1 - 1 - 1 (일점쇄선) | - |

해칭: 45°, 간격 1.0 mm, 선 굵기 0.13 mm.

상태 넷에는 해칭을 쓰지 않았다. 지금 렌더러는 해칭되는 모든 선 종류에 **같은 해칭 기하**를 쓰므로, 둘 이상을 해칭하면 종이에서 구별되지 않는다. 각도나 간격을 선 종류별로 가지려면 `HatchStyle`이 preset 안에서 선 종류별이 되어야 하고, 그건 출처 있는 수치를 옮길 때 함께 할 일이다.

**간선의 끊김은 파선 패턴이 아니라 기하다.** 파선 패턴은 선이 얼마나 자주 끊기는지를 정하지 몇 번 끊기는지를 정하지 못하는데, 간선은 "두세 번 끊어 긋는" 것이므로 횟수가 관례다. 그래서 골 선은 별개의 일직선 조각들로 나와 그린다. `GROOVE_TROUGH_BREAK_COUNT = 2`, `GROOVE_TROUGH_BREAK_MM = 1.6`이고 둘 다 잠정값이다.

끊김은 **중심축을 기준으로 좌우 각각** 넣는다. 좌 반입면은 폭의 절반만 그리므로, 현 전체를 끊으면 그 절반이 잘려나가는 쪽에 남는다. 이렇게 하면 반입면이든 전체 입면이든 그려진 선이 제 횟수를 갖고, 전체 입면에서는 중심축에 대해 좌우 대칭으로 끊긴다.

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

### 탁본 붙이기

토기 외면 탁본은 종이 띠로 쳐서 도면 옆에 붙인다. 도판도 그렇게 한다. `record_ids`에 탁본 record(`raster.developed_rubbing.v1` 또는 `raster.digital_rubbing.v1`)를 벡터 기록과 **같은 목록에 섞어** 넣으면, 준 순서 그대로 자기 도형이 된다.

탁본 record는 픽셀이 아니라 receipt를 저장하므로, 픽셀은 호출자가 recipe로 다시 계산해 넘긴다.

```python
bundle = compose_drawing_sheet(
    document,
    ["record:cutline-front", "record:rubbing-strip"],
    options=options,
    rasters={"record:rubbing-strip": recomputed_raster},
)
```

넘긴 raster가 record의 receipt와 다르면 거부한다. 픽셀 없이 탁본 record만 올려도 거부한다 — 조용히 빈 자리를 남기지 않는다. 도판에 없는 record의 raster를 넘겨도 거부한다.

도판에 붙는 탁본은 **종이 전체가 옅은 먹을 먹은 상태**여야 실제 탁본으로 읽힌다. 솜방망이는 들어간 부분에도 먹을 조금 남기기 때문이다. 기본 권장값과 그 근거는 [`docs/POTTERY_STRIP_UNWRAP.md`](POTTERY_STRIP_UNWRAP.md)의 "솜방망이가 남기는 것"에 있다.

토기 띠 탁본은 **직사각형**으로 붙는다. 종이 탁본이 견부나 경부에서 저부까지 한 폭으로 내려간 한 장이기 때문이고, 전개 탁본의 기본 artboard 정책(`largest_covered_rectangle/v1`)이 전개 안에 완전히 들어가는 가장 큰 직사각형만 남기기 때문이다. 조각의 깨진 윤곽 자체가 정보인 기와·전에서는 전개 경계 정책을 골라 윤곽을 살린다. 자세한 것은 [`docs/POTTERY_STRIP_UNWRAP.md`](POTTERY_STRIP_UNWRAP.md)의 "종이의 모양"에 있다.

크기는 **물리 크기**다. receipt의 픽셀 수와 px/mm에서 mm를 계산하고, 시트 축척으로 함께 줄인다. 1:1이면 종이에서 실제 크기, 1:2면 절반이다. 선 굵기와 달리 탁본은 그림이므로 축척과 함께 줄어드는 것이 맞다.

SVG에는 canonical GA8 PNG를 base64 data URI로 박는다. 그래서 도판 한 장이 자기 완결적이고, 바이트도 결정적이다. 탁본이 실린 도판만 `xmlns:xlink`를 선언하므로, 선만 있는 도판은 예전과 **바이트까지 같다.** sidecar의 도형 항목에는 `vector_payload_sha256` 대신 `raster_sha256`·`raster_pixels_per_meter`·픽셀 수가 들어간다.

앱에서는 탁본 기록이 도판 체크 목록에 함께 나온다. 체크하고 도판을 만들면 recipe로 다시 계산해 검증한 뒤 배치한다.

### 도판은 측정값이 아니다

도판은 표현물이다. 각 도형은 자기가 그린 payload의 해시를 sidecar에 기록하므로 어떤 기록으로 만든 도판인지 확인할 수 있지만, 도판 자체가 측정 권위가 되지는 않는다. 그래서 원자 staging을 쓰는 export controller를 거치지 않고 바로 파일로 쓴다. 실질적인 관문은 그대로다. READY이고 FRESH가 아닌 기록은 도판에 오르지 못한다.

---

## 중심축선

중심축은 측정값이 아니라 **정치의 결과**다. 그래서 record type이 없다. record로 만들면 사용자가 축을 따로 "생성"해야 하고 3/6/6 완료 게이트까지 건드리는데, 이 선이 실제로 무엇인지와 맞지 않는다.

회전축으로 정치한 문서에서는 회전축이 월드 +Z이고 원점을 지난다. 도면 계층이 그 선을 각 record의 평면에 투영하고 그려진 내용 범위로 잘라 그린다.

```python
VectorSVGOptions(style_preset="provisional/v1", show_center_axis=True)
DrawingSheetOptions(title_block=..., show_center_axis=True)
```

세 가지 규칙이 있다.

**활성 Align이 회전축으로 만들어졌을 때만 그린다.** 수동 정치 상태에서 켜면 조용히 생략하고, 도판 sidecar의 `center_axis`에 요청은 있었으나 그리지 않았다는 사실과 그때의 Align recipe 종류를 남긴다. 근거 없는 축선은 종이 위에서 아무것도 뒷받침하지 않는 주장이 된다.

**평면도에는 나오지 않는다.** 위에서 본 축은 점으로 투영되므로 그 자리에 선을 그으면 거짓이 된다. top·bottom 뷰는 자동으로 생략된다.

**기본은 꺼짐이고, preset 없이는 켤 수 없다.** 모든 선이 같은 굵기면 중심축선을 외형선과 구분할 수 없으므로 preset 없이 요청하면 거부한다.

---

## 좌 반입면 · 우 반단면

실측 도면은 입면과 단면의 조화로 만들어진다. 토기 도면의 관례는 둘을 나란히 놓는 것이 아니라 **한 도형 안에서** 왼쪽 절반을 외형(입면), 오른쪽 절반을 단면으로 그리는 것이다. 독자가 기형과 두께를 두 그림을 대조하지 않고 한 번에 본다.

```python
DrawingSheetOptions(
    title_block=...,
    mirror_sections=(("record:elevation-front", "record:section-front"),),
)
```

입면 record는 `record_ids`에 그대로 두고, 단면 record는 그 도형 안에 들어가므로 `record_ids`에 넣지 않는다. **기본값은 빈 튜플이고, 주지 않으면 도판 바이트가 이전과 같다.**

앱에서는 도판 항목의 "좌 반입면 · 우 반단면으로 합치기"를 켜고 입면과 단면 기록을 고른다. 입면은 위 체크 목록에도 체크되어 있어야 하고, 단면은 체크하지 않는다.

### 확인하는 것

**회전축으로 정치되어 있어야 한다.** 절반이라는 말은 유물이 도는 축을 기준으로만 뜻이 있다. 수동 정치 상태에서 요청하면 조용히 생략하지 않고 거부한다 — 중심축선과 다른 점이다. 중심축선은 부가 표시라 없으면 그냥 안 그리면 되지만, 미러 도형은 요청한 그림 자체가 성립하지 않는다.

**두 record가 같은 평면이어야 한다.** 뷰 이름이 아니라 frame 자체를 대조한다. 같은 평면이 아니면 한 도형의 두 절반이 될 수 없다.

**한쪽이 비면 거부한다.** 축 왼쪽에 입면이 없거나 오른쪽에 단면이 없으면, 나오는 그림은 절반이 빈 도면이다.

**닫힌 path가 축에서 세 조각 이상으로 갈라지면 거부한다.** Sutherland–Hodgman은 축 위에 놓인 변으로 조각들을 이어 하나의 고리로 답하는데, 그 변은 유물에 없는 경계로 인쇄된다. 어느 쪽이 도면인지는 이 계층이 정할 수 있는 문제가 아니다.

### 접힌 자리는 선이 아니다

절반으로 자르면 각 고리는 축 위의 현(弦)으로 닫힌다. **그 현은 유물의 경계가 아니라 도면을 접은 자리다.** 그래서 획을 긋지 않는다.

- 입면 절반은 열린 path가 된다. 구연에서 저부까지 바깥을 도는 선 하나.
- 단면 절반은 해칭이 필요하므로 닫힌 도형을 유지하되 **채움 전용(`stroke="none"`)** 으로 한 번, 축 위의 변을 뺀 열린 선을 한 번, 두 요소로 나눠 그린다.
- 두 절반이 만나는 자리에는 중심축선만 남는다.

**미러 도형은 `show_center_axis`와 무관하게 자기 중심축선을 그린다.** 그 선이 이 관례의 이음매이기 때문이다. 없으면 독자가 접어 붙인 도형과 비대칭 유물의 도면을 구별할 수 없다.

상태 표기는 입면 절반에 함께 잘려 들어간다. 단면 절반의 상태 표기는 아직 없다 — 아래 "아직 없는 것"을 보라.

sidecar의 `mirrored_figures`에 어느 record가 어느 쪽 절반이었는지 남는다.

---

## 유물 상태 표기

결실 · 복원 · 균열 · 마모는 `annotation.condition.v1` record가 담는다. 자세한 내용은 [`docs/CONDITION_ANNOTATION.md`](CONDITION_ANNOTATION.md)에 있다. 도면 쪽에서 알아야 할 것은 이것뿐이다.

```python
DrawingSheetOptions(
    title_block=...,
    condition_records=("record:condition-1", "record:condition-2"),
)
```

**기본값은 빈 튜플이고, 주지 않으면 도판 바이트가 이전과 같다.** sidecar에 `condition` 블록도 생기지 않는다.

**투영 도면에만, 같은 평면에만 그린다.** 상태 경계는 한 방향에서 본 영역의 실루엣이다. 단면도는 평면이 자른 것을 보여주는 다른 종류의 그림이므로, 단면의 평면이 어떤 뷰와 우연히 같더라도 거기에는 그리지 않는다. 투영 도면끼리도 평면이 같은지는 뷰 이름이 아니라 **frame 자체**로 대조한다.

**READY이고 FRESH여야 한다.** 정치가 바뀌면 상태 record는 STALE_ALIGNMENT가 되고 도판에 오르지 못한다. 유물이 더 이상 있지 않은 자리에 결실부를 그리게 되기 때문이다.

sidecar의 `condition.records`에는 그리도록 지정된 record와 그 면 집합 해시가, `condition.drawn`에는 어느 record가 어느 도형에 어느 뷰로 실제로 그려졌는지가 남는다.

---

## 제작 기법 — 한 바퀴 도는 홈

토기 기벽을 한 바퀴 도는 홈(횡침선, 돌대 사이 홈)은 `measurement.profile_groove.v1` record가 담는다. 찾는 방법과 실측 수치는 [`docs/POTTERY_STRIP_UNWRAP.md`](POTTERY_STRIP_UNWRAP.md)의 "한 바퀴 도는 홈"에 있다. 도면 쪽에서 알아야 할 것은 이것뿐이다.

```python
DrawingSheetOptions(
    title_block=...,
    groove_records=("record:groove-body",),
)
```

**기본값은 빈 튜플이고, 주지 않으면 도판 바이트가 이전과 같다.** sidecar에 `groove` 블록도 생기지 않는다.

**홈 하나가 선 셋이다.** 들어간 골은 간선(`technique_groove_trough`), 튀어나온 두 능선은 직선(`technique_groove_edge`)으로 나온다. 홈은 결국 들어간 곳 하나와 나온 곳 둘로 이루어지므로, 도면이 표면과 같은 개수를 갖는다.

**회전축이 그 평면 안에 있어야 그린다.** 입면이든 단면이든 축을 품은 평면에는 그리고, 평면도에는 그리지 않는다 — 위에서 보면 한 바퀴 도는 홈은 선이 아니라 원이다. 축이 평면에 비스듬히 걸친 경우도 같은 이유로 그리지 않는다. 그 선은 홈의 단축 투영이라 유물이 그 높이에서 갖지 않는 폭을 주장하게 된다.

이 점이 상태 표기와 다르다. 상태 경계는 한 방향에서 본 실루엣이라 뷰마다 다른 모양이지만, 홈은 유물의 축에 대한 사실이라 축을 품은 어느 평면에서나 같은 선이다.

**READY이고 FRESH여야 한다.** 정치가 바뀌면 홈 record는 STALE_ALIGNMENT가 되고 도판에 오르지 못한다. 다른 곳에 서 있는 유물의 높이를 부르게 되기 때문이다.

sidecar의 `groove.records`에는 그리도록 지정된 record와 골의 높이들이, `groove.drawn`에는 어느 record가 어느 도형에 그려졌는지가 남는다.

---

## 아직 없는 것

- **단면 절반의 상태 표기.** 상태 record는 여섯 뷰의 투영 경계를 담는데, 단면은 사용자가 정한 임의의 평면이라 커밋 시점에 미리 계산해 둘 수가 없다. 절단면 위의 결실·복원을 그리려면 상태 record와 단면 record 둘 다에 의존하는 별도의 record가 필요하다.
- **1:1 vector export(`.amr-vector`)의 상태 표기.** 지금은 도판에만 그린다. 1:1 패키지는 검증할 때 sidecar만 가지고 SVG를 다시 렌더해 바이트를 대조하므로, 상태 경계를 그리려면 상태 payload 전체가 sidecar에 실리고 오프라인 검증기가 그것까지 검사해야 한다. 이 저장소에서 가장 엄격한 검증 경로라 따로 한 번에 한다.
- **외곽선의 격자 바늘구멍.** 외곽선은 삼각형 투영을 고정 격자에 스냅한 뒤 합집합한다. 실루엣 접선 부근에서는 어떤 매끄러운 면이든 투영 폭이 격자보다 좁은 삼각형이 생기고, 그런 삼각형은 스냅에서 면적 0이 되어 합집합에서 빠진다. 그러면 양옆 삼각형이 점으로만 붙어 실루엣 경계에 격자 한 칸짜리 바늘구멍이 남고, 경계에 닿은 구멍이므로 `hole_not_strictly_inside`로 거부된다. 거부 자체는 옳다 — 그건 유물의 구멍이 아니다 — 그리고 이제 메시지가 격자 몇 칸인지와 무엇을 바꿔야 하는지 말한다.

  합성 토기(74,112면, 승문 0.25 mm)로 재보면 격자에 따라 이렇게 갈린다.

  | 정밀도 격자 | 무늬 없음 | 승문 0.25 mm |
  |---|---|---|
  | 1.0 · 0.5 · 0.1 mm | 거부 | 거부 |
  | 0.05 mm | 통과 | 거부 |
  | 0.01 mm (기본값) | 통과 | 통과 (정면 22초, 평면 25초) |

  즉 **기본값 0.01 mm는 촘촘한 유물에서도 통과한다.** 격자를 굵게 잡으면 막히고, 메쉬가 촘촘하거나 표면에 무늬가 있으면 더 굵게 잡을 수 없다. 근본 해결은 스냅 전에 합집합을 하는 것인데, 그건 해시가 고정된 측정 경로를 바꾸는 일이라 따로 다룬다.
- 메쉬 위에서 영역을 칠하는 선택 도구. 지금은 면 인덱스를 직접 주어야 한다.
- **탁본 도형의 캡션과 위치 지정.** 지금은 다른 도형과 같은 규칙으로 왼쪽에서 오른쪽으로 놓인다. 도면 옆 정해진 자리에 두거나 "동체부 우측 20 mm 띠" 같은 캡션을 붙이려면 도형별 배치 지시가 필요하다.
- 선 종류별 해칭 기하.
- DXF 내보내기.
