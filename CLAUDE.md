# FLIP-AI

문제집 페이지 사진을 자동 채점하는 CV/ML 프로젝트. 파이프라인:
보정(preprocess) → 인쇄체 OCR(PaddleOCR) → 페이지 구조 파악(structure) →
객관식 마킹 검출(mcq) | 손글씨 답 크롭(handwriting) → VLM 판독(vlm) →
SymPy 동치 비교(equivalence) → O/X/보류.

- 손글씨 답 판독은 VLM API 호출(`flip/vlm.py`, 기본 OpenAI gpt-4o-mini,
  gemini/anthropic 전환 가능). YOLO 아님 — `infer.py`/`synth_data.py`는
  초기 YOLO 실험의 잔재로, 현재 파이프라인(`grade.py`)에서 쓰지 않는다.
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
