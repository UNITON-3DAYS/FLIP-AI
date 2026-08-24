# FLIP-AI

문제집 페이지 사진을 자동 채점하는 CV/ML 프로젝트. 파이프라인:
보정(preprocess) → 인쇄체 OCR(PaddleOCR) → 페이지 구조·문제 블록 절단(structure) →
문제 블록 통크롭을 VLM에 판독(vlm) → SymPy 동치 비교(equivalence) → O/X/보류.

- 채점의 핵심은 VLM 판독(`flip/vlm.py`, 기본 OpenAI, gemini/anthropic 전환 가능).
  OCR·structure는 쪽수·문제번호(anchor)로 **블록을 자르는 데까지만** 쓰고, 판정은
  블록 통크롭을 통째로 VLM에 넘겨서 한다:
  - 객관식 `read_mcq` — "동그라미 친 번호"를 읽어 정답 집합과 비교 (1회 호출).
  - 주관식 `read_math` — 손글씨 값을 읽어 SymPy 동치 비교 (2회검증, 불일치는 보류).
  실사진 실측 12문제 11~12정답·오채점 0 (구 방식은 1/12). 비용 gpt-5.6-luna
  기준 페이지당 ~2.7원. `detail=high` 필수(low는 손글씨 오독), gpt-4o-mini는
  고해상 타일링으로 입력토큰 30배라 금지.
- 인쇄 마커 CV(`flip/mcq.py`)와 손글씨 격리(`handwriting.extract_crops`)는 실사진에서
  실패해 현재 파이프라인에서 **쓰지 않는다**(selftest용으로만 잔존). YOLO 실험 잔재
  (`infer.py`/`synth_data.py`)도 마찬가지 — `grade.py`는 위 VLM 경로만 탄다.
- Linear: Team `AKR` (akran) / Project `FLIP-AI`
- 브랜치 모델: main-only
- QA 방법: 로컬 실행 (`--selftest` 및 실제 이미지 채점)

## 검사 명령

- 집중 테스트: `python grade.py --selftest` — 이미지·OCR·API 키 없이
  스키마/채점/동치/구조 로직 검증
- 실제 이미지: `python grade.py --image <사진> --db db.ssen_2-1.json`
  (OCR은 PaddleOCR 설치, VLM은 `FLIP_VLM_API_KEY` 필요)
- lint/typecheck: 없음
- 빌드: 없음 (스크립트 실행형)

## 프로젝트 규칙

- 모델 가중치(`*.pt`), 학습 산출물(`runs/`, `data/`), 개인 사진(`test_images/`),
  추론/미리보기 이미지는 `.gitignore` 대상. 커밋에 넣지 않는다.
- VLM 호출은 절대 크래시하지 않는다: 키 없음/타임아웃/불일치는 전부 None →
  보류. 이 방어를 깨는 변경 금지.
