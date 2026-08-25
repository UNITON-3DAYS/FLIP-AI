# archive/ — 폐기·미사용 잔재 보관

현재 채점 파이프라인(보정 → OCR로 블록 절단 → **VLM 블록크롭 판독** → SymPy 동치)과
무관한 구 실험·미사용 코드를 여기 모아둔다. 삭제하지 않고 참고용으로만 보관한다.
활성 코드는 `flip/`·`api/`에 있으며, 이 폴더의 어떤 것도 그쪽에서
import하지 않는다.

| 파일 | 무엇 | 폐기 사유 |
|---|---|---|
| `infer.py` | 구 YOLO 손글씨 인식 CLI | YOLO 인식 실험 폐기 — VLM 판독으로 대체 |
| `synth_data.py` | YOLO 학습용 합성 데이터 생성 | 위와 동일 (YOLO 트랙 폐기) |
| `flip_train_colab.ipynb` | YOLO 학습 Colab 노트북 | 위와 동일 |
| `spike_vlm.py` | VLM 판독 타당성 스파이크 | 검증 끝나 `flip/vlm.py`로 구현 이관 |
| `mcq.py` | 인쇄 마커 기준 객관식 마킹 CV | 실사진에서 실패(12중 1정답) — VLM `read_mcq`로 대체 |
| `handwriting.py` | OCR 마스킹 기반 손글씨 격리 | 실사진 0/5 실패 — 블록 통크롭 VLM 판독으로 대체 |

되살릴 일이 생기면 git 이력에 원위치가 남아 있으니 그대로 되돌리면 된다.
