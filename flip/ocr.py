"""PaddleOCR 래퍼. 인쇄체 구조(쪽수·문제번호·선택지 마커)를 잡는 용도.

- 지연 import: PaddleOCR 미설치 환경에서도 selftest가 돌게 한다.
- 결과는 OcrBox 리스트로 통일. 이후 모든 구조 분석 단계가 이 리스트를 재사용한다
  (OCR은 페이지당 1회만 실행).
"""
from dataclasses import dataclass

_engine = None  # PaddleOCR 인스턴스 캐시 (초기화가 느려서 재사용)

# 검출·인식만 축소본으로 돌려 속도를 얻고, 좌표는 원본 스케일로 되돌린다.
# 쪽수·문제번호 anchor는 크고 진해서 0.5배(≈826x1168)에서도 안전 (블록 6/6 유지).
# ponytail: 고정 0.5배. 더 작은 글자를 놓치면 상향, 더 빠르게는 mobile det 교체.
OCR_SCALE = 0.5


@dataclass
class OcrBox:
    text: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self):
        return (self.x1 + self.x2) / 2

    @property
    def cy(self):
        return (self.y1 + self.y2) / 2

    @property
    def h(self):
        return self.y2 - self.y1


def available():
    try:
        import paddleocr  # noqa: F401
        return True
    except ImportError:
        return False


def run_ocr(gray_img, lang="korean"):
    """보정된 그레이 이미지 -> [OcrBox]. PaddleOCR 미설치면 ImportError."""
    global _engine
    import os
    import cv2  # 지연 import: paddleocr와 함께 실제 실행 시에만
    from paddleocr import PaddleOCR
    if _engine is None:
        # 프로세스마다 도는 원격 모델소스 체크 생략 (수십 초 낭비)
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        _engine = PaddleOCR(
            lang=lang,
            use_textline_orientation=True,
            # 입력은 preprocess가 정면화한다. 문서 방향/왜곡 모델은 CPU에서 페이지당
            # 수십 초를 먹는 순수 낭비. 90도 눕은 사진이 들어오면 다시 켤 것.
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
    if gray_img.ndim == 2:  # PaddleOCR 전처리가 3채널을 요구
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    if OCR_SCALE != 1.0:
        small = cv2.resize(gray_img, None, fx=OCR_SCALE, fy=OCR_SCALE,
                           interpolation=cv2.INTER_AREA)
    else:
        small = gray_img
    result = _engine.predict(small)
    inv = 1.0 / OCR_SCALE  # 축소본 좌표 -> 원본 좌표 복원
    boxes = []
    for page in result:  # predict는 페이지 리스트를 반환 (입력 1장이면 1개)
        texts = page.get("rec_texts", [])
        scores = page.get("rec_scores", [])
        polys = page.get("rec_polys", page.get("dt_polys", []))
        for text, score, poly in zip(texts, scores, polys):
            xs = [p[0] * inv for p in poly]
            ys = [p[1] * inv for p in poly]
            boxes.append(OcrBox(text=text, conf=float(score),
                                x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys)))
    return boxes
