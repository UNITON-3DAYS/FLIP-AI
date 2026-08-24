# FLIP-AI

YOLO-det 모델로 손글씨 답안 크롭을 인식하고 채점하는 CV/ML 프로젝트.

- Linear: Team `AKR` (akran) / Project `FLIP-AI`
- 브랜치 모델: main-only
- QA 방법: 로컬 실행 (`--selftest` 및 실제 이미지 추론)

## 검사 명령

- 집중 테스트: `python infer.py --selftest` (조립/채점 로직), `python synth_data.py --selftest` (합성 로직) — 모델·데이터셋 없이 검증
- lint/typecheck: 없음
- 빌드: 없음 (스크립트 실행형, 학습은 `flip_train_colab.ipynb` Colab)

## 프로젝트 규칙

- 클래스 순서는 `synth_data.py`의 `CLASSES`가 단일 소스. `infer.py`가 이를 import해
  id→문자 매핑을 공유하므로, 클래스를 바꾸면 양쪽이 함께 바뀐다.
- 모델 가중치(`*.pt`), 학습 산출물(`runs/`, `data/`), 개인 사진(`test_images/`),
  추론/합성 미리보기 이미지는 `.gitignore` 대상. 커밋에 넣지 않는다.
