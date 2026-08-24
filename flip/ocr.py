"""PaddleOCR 래퍼. 인쇄체 구조(쪽수·문제번호·선택지 마커)를 잡는 용도.

- 지연 import: PaddleOCR 미설치 환경에서도 selftest가 돌게 한다.
- 결과는 OcrBox 리스트로 통일. 이후 모든 구조 분석 단계가 이 리스트를 재사용한다
  (OCR은 페이지당 1회만 실행).
"""
from dataclasses import dataclass

_engine = None  # PaddleOCR 인스턴스 캐시 (초기화가 느려서 재사용)


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
    from paddleocr import PaddleOCR  # 지연 import
    if _engine is None:
        _engine = PaddleOCR(lang=lang, use_textline_orientation=True)
    result = _engine.predict(gray_img)
    boxes = []
    for page in result:  # predict는 페이지 리스트를 반환 (입력 1장이면 1개)
        texts = page.get("rec_texts", [])
        scores = page.get("rec_scores", [])
        polys = page.get("rec_polys", page.get("dt_polys", []))
        for text, score, poly in zip(texts, scores, polys):
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            boxes.append(OcrBox(text=text, conf=float(score),
                                x1=min(xs), y1=min(ys), x2=max(xs), y2=max(ys)))
    return boxes
