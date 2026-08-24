# FLIP 답안 인식 파이프라인

페이지 사진 1장을 받아 손글씨 답·객관식 마킹을 인식해 자동 채점하는 **인식·채점 경로**의 설계.

> 시스템 경계·콘텐츠 트랙(자체 시험지 / 쎈)은 [architecture.md](architecture.md) 참고.
> 초기엔 로컬 YOLO-det 인식기를 실험했으나(`infer.py`/`synth_data.py`/`training-log.md`),
> 실사진에서 정렬·크롭·필체 한계로 폐기하고 **VLM 판독**으로 전환했다.

---

## 1. 핵심 결정

| 항목 | 결정 | 이유 |
|---|---|---|
| 인식 방법 | **블록 통크롭 → VLM API 판독** | 실사진에서 CV/YOLO는 12문제 중 1정답으로 실패. 블록을 통째로 VLM에 넘기면 11~12정답·오채점 0 |
| 앞단(위치) | **OCR anchor로 블록 절단까지만** | 쪽수·문제번호(큰 인쇄 숫자)는 OCR이 안정적. 이걸로 문제 블록 경계만 자르고, 마커/손글씨 위치는 VLM에 맡긴다 |
| 객관식 | `vlm.read_mcq` — 동그라미 친 번호 (1회) | 출력이 수 토큰·reasoning 0이라 재질의 불필요 |
| 주관식 | `vlm.read_math` — 손글씨 값 (2회검증) | 2회 불일치는 보류 → 자신있게 틀리는 것 방지 |
| 판정 원칙 | **보류 우선(HOLD-first)** | 확신 없으면 X가 아니라 보류. 오채점보다 사람 확인이 낫다 |
| 이미지 해상도 | `detail=high` 고정 | low는 손글씨를 오독(85→14, 측정 확인). 크롭이 작아 high여도 저렴 |
| 모델 | 기본 OpenAI(env로 `gpt-5.6-luna`), gemini/anthropic 전환 가능 | luna 페이지당 ~2.7원. **gpt-4o-mini 금지**(고해상 타일링으로 입력토큰 30배) |

### 왜 로컬 인식(YOLO)을 접었나
- 실사진은 정렬·조명·필체 편차가 커서 고정 좌표 크롭·글자별 detection이 깨진다.
- 객관식은 손동그라미가 인쇄와 얽혀 CV 특징(잉크비·폐곡선)이 형제 비교를 무너뜨렸다.
- VLM은 "블록을 보고 답만 말해"라는 지시 하나로 정렬·필체·혼재를 흡수한다. 비용은 페이지당 원 단위로 충분히 싸다.

---

## 2. 인식·채점 파이프라인 (`flip/`)

```
페이지 이미지 1장
  → preprocess : 그레이·정면화 보정
  → ocr(PaddleOCR) : 인쇄 텍스트 박스 (OCR_SCALE=0.5로 축소 실행 후 좌표 원복)
  → structure : 쪽수 인식 → 문제번호(anchor) LIS 매칭 → 문제별 블록 사각형 절단
  → 각 문제 블록을 통째로 크롭 → 유형 분기
       · 객관식 → vlm.read_mcq  → 동그라미 번호 집합 == 정답 집합 ? O : X
       · 주관식 → vlm.read_math → equivalence(SymPy) 동치 비교 ? O : X
  → 인식 실패/불확실/블록 없음 → 보류
```

- OCR·structure는 **블록을 자르는 데까지만** 쓴다. 실제 판정은 전부 VLM.
- DB(`db.*.json`)가 기대 문제번호 목록을 줘서 anchor 오독을 걸러낸다(structure의 LIS 매칭).

---

## 3. VLM 판독 계약 (`flip/vlm.py`)

- **절대 크래시하지 않는다.** 키 없음·타임아웃·응답 형식 불일치·안전거절은 전부 `None` → 호출부가 보류. 이 방어를 깨는 변경 금지.
- `read_mcq(block_crop)` → 정렬된 번호 리스트[int] 또는 `None`. `NONE`/거절/파싱실패는 None.
- `read_math(block_crop)` → 답 문자열 또는 `None`. 같은 크롭 2회 호출이 불일치하거나 `UNSURE`/`PRINTED`면 None.
- 답 표기는 LaTeX가 아니라 **선형 표기**(분수 `a/b`, 제곱근 `sqrt(x)`, 거듭제곱 `x^2`) — `flip/equivalence.py`가 그대로 파싱한다(antlr 의존 회피).

환경변수: `FLIP_VLM_PROVIDER`(openai|gemini|anthropic), `FLIP_VLM_API_KEY`, `FLIP_VLM_MODEL`, `FLIP_VLM_REASONING`(reasoning 모델일 때 low 권장).

---

## 4. 비용 (gpt-5.6-luna, $0.20/1M in · $1.20/1M out)

sim_p14/15 실측(2페이지=12문제, 객관식 1회+주관식 2회 = 17콜):

| 단위 | 비용 |
|---|---|
| 페이지(6문제) | ~$0.0019 (~2.7원) |
| 100페이지 | ~$0.19 (~270원) |
| 1만 페이지(100명×100장) | ~$19 (~2.7만원) |

- 비용의 ~70%가 입력(이미지) 토큰. reasoning 토큰은 개수는 커도 금액은 미미(출력 $1.2/1M).
- 입력토큰이 지배적이라 **더 비싼 모델(gpt-4o ~9배)·타일링 모델(mini 30배)로 갈수록 손해**. luna 유지가 최적.

---

## 5. 상태 / 남은 일

| 단계 | 상태 |
|---|---|
| preprocess / ocr / structure(블록 절단) | ✅ 동작 (블록 6/6) |
| 객관식 VLM 판독 (`read_mcq`) | ✅ 실측 7/7 |
| 주관식 VLM 판독 (`read_math`, 2회검증) | ✅ 실측 4~5/5 (경계 케이스는 보류) |
| 동치 비교 (`equivalence`, SymPy) | ✅ |
| 구 YOLO 인식기 (`infer.py`/`synth_data.py`) | 🗑 미사용 (초기 실험 잔재, 삭제 검토) |
| 구 객관식 CV (`mcq.py`)·손글씨 격리 (`handwriting`) | 🗑 미사용 (selftest용으로만 잔존) |
