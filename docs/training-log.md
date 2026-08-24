# 학습 기록 (Training Log) — 🗑 폐기(historical)

> **현재 파이프라인은 모델을 학습하지 않는다.** 답 판독은 VLM API 호출(`flip/vlm.py`)로 대체됐다.
> 아래는 초기 로컬 YOLO-det 인식기 실험용 기록 양식으로, 지금은 쓰지 않는다.
> 현행 인식·채점은 [recognition-pipeline.md](recognition-pipeline.md) 참고.

---

## (구) 기록 양식 — YOLO-det 실험

Colab 노트북이 출력하던 `metrics.txt`를 매 학습마다 붙여넣던 표.
**핵심 지표:** 문자열 완전일치율(답을 통째로 맞게 읽은 비율). mAP는 박스 검출 정확도(참고).

| 날짜 | 문자열 완전일치 | 글자 정확도 | mAP50 | mAP50-95 | 설정 | 비고 |
|---|---|---|---|---|---|---|
| — | — | — | — | — | yolov8n, imgsz=128 | 실제 학습 미수행, 접음 |
