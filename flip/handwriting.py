"""손글씨 답 후보 추출: 인쇄(OCR 박스) 마스킹 후 남은 잉크를 크롭.

읽기는 VLM(flip/vlm.py) 몫이고, 여기서는 "블록 안 어디에 손글씨가 있는가"만 찾는다.
- OCR이 검출한 영역 = 인쇄체로 보고 하얗게 지운다 (텍스트를 읽을 필요는 없음).
- 남은 잉크를 adaptive threshold → connected components로 뭉치고, 가까운 성분끼리
  병합해(같은 답의 글자들) 답 후보 크롭을 만든다.
- 후보 없음 = 빈 리스트: 답을 안 썼거나 인쇄 위에 겹쳐 쓴 경우 → 호출부가 보류.
"""
import cv2
import numpy as np

# ── 튜닝 포인트 ──────────────────────────────────────────────────────────
MASK_PAD = 4          # OCR 박스 마스킹 여유 (px)
THRESH_BLOCK = 31     # adaptive threshold 블록 크기 (홀수)
THRESH_C = 15         # adaptive threshold 보정 상수
MIN_AREA = 60         # 이 미만 픽셀 수 성분은 노이즈로 버림
MERGE_GAP = 30        # 이 거리(px) 이내 성분은 같은 답으로 병합
MIN_SIDE = 12         # 병합 후에도 이보다 작은 후보는 버림
CROP_PAD = 10         # 크롭 패딩 (px)
STROKE_STD_MIN = 0.15  # 획 굵기(distance transform) 표준편차 하한. 인쇄 잔재는
                       # 굵기가 균일해 분산이 낮다 — 보조 필터라 느슨하게 건다.


def _mask_printed(gray, boxes, block):
    """block 영역을 잘라내고 OCR 박스(인쇄)를 하얗게 마스킹한 사본을 반환."""
    h, w = gray.shape[:2]
    x1 = max(0, int(block[0]))
    y1 = max(0, int(block[1]))
    x2 = min(w, int(round(block[2])))
    y2 = min(h, int(round(block[3])))
    region = gray[y1:y2, x1:x2].copy()
    for b in boxes:
        if b.x2 <= x1 or b.x1 >= x2 or b.y2 <= y1 or b.y1 >= y2:
            continue  # block과 겹치지 않는 박스
        bx1 = max(0, int(b.x1) - x1 - MASK_PAD)
        by1 = max(0, int(b.y1) - y1 - MASK_PAD)
        bx2 = min(region.shape[1], int(round(b.x2)) - x1 + MASK_PAD)
        by2 = min(region.shape[0], int(round(b.y2)) - y1 + MASK_PAD)
        if bx2 > bx1 and by2 > by1:
            region[by1:by2, bx1:bx2] = 255
    return region


def _ink_components(region):
    """마스킹된 영역에서 잉크 이진화 + 성분 박스 목록. 영역이 너무 작으면 빈 결과."""
    if region.shape[0] < THRESH_BLOCK or region.shape[1] < THRESH_BLOCK:
        return None, []
    binary = cv2.adaptiveThreshold(region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, THRESH_BLOCK, THRESH_C)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    rects = []
    for i in range(1, n):  # 0번은 배경
        x, y, cw, ch, area = stats[i]
        if area < MIN_AREA:
            continue
        rects.append((int(x), int(y), int(x + cw), int(y + ch)))
    return binary, rects


def _near(a, b):
    """두 rect가 MERGE_GAP 이내인지 (a를 GAP만큼 넓혀 교차 검사)."""
    return (a[0] - MERGE_GAP < b[2] and b[0] < a[2] + MERGE_GAP and
            a[1] - MERGE_GAP < b[3] and b[1] < a[3] + MERGE_GAP)


def _merge_rects(rects):
    """가까운 rect들을 합집합으로 반복 병합 (같은 답의 글자 묶기)."""
    rects = list(rects)
    changed = True
    while changed:
        changed = False
        out = []
        while rects:
            cur = rects.pop()
            i = 0
            while i < len(rects):
                if _near(cur, rects[i]):
                    o = rects.pop(i)
                    cur = (min(cur[0], o[0]), min(cur[1], o[1]),
                           max(cur[2], o[2]), max(cur[3], o[3]))
                    changed = True
                else:
                    i += 1
            out.append(cur)
        rects = out
    return rects


def _stroke_std(binary, rect):
    """rect 내부 잉크의 획 굵기 분산 (distance transform 표준편차)."""
    x1, y1, x2, y2 = rect
    dist = cv2.distanceTransform(binary[y1:y2, x1:x2], cv2.DIST_L2, 3)
    vals = dist[dist > 0]
    if vals.size == 0:
        return 0.0
    return float(vals.std())


def _candidates(gray, boxes, block):
    """(마스킹된 영역, 영역 좌표계 후보 rect 목록). 셀프테스트에서 위치 검증용."""
    region = _mask_printed(gray, boxes, block)
    binary, rects = _ink_components(region)
    if binary is None:
        return region, []
    keep = []
    for r in _merge_rects(rects):
        if r[2] - r[0] < MIN_SIDE or r[3] - r[1] < MIN_SIDE:
            continue
        if _stroke_std(binary, r) < STROKE_STD_MIN:
            continue  # 굵기 분산이 지나치게 낮으면 인쇄 잔재로 본다
        keep.append(r)
    if not keep:
        return region, []
    # 블록당 답은 하나라는 전제로, 살아남은 성분 전체를 합집합 하나로 묶는다.
    # 세로 분수(분자/분수선/분모)처럼 MERGE_GAP보다 벌어져 조각난 답도
    # 통째로 VLM에 전달된다 — 조각 하나만 읽고 오판하는 사고 방지.
    union = (min(r[0] for r in keep), min(r[1] for r in keep),
             max(r[2] for r in keep), max(r[3] for r in keep))
    return region, [union]


def extract_crops(gray, boxes, block):
    """block에서 손글씨 답 크롭. 합집합 1개([crop]) 또는 빈 리스트."""
    region, rects = _candidates(gray, boxes, block)
    crops = []
    for x1, y1, x2, y2 in rects:
        cx1 = max(0, x1 - CROP_PAD)
        cy1 = max(0, y1 - CROP_PAD)
        cx2 = min(region.shape[1], x2 + CROP_PAD)
        cy2 = min(region.shape[0], y2 + CROP_PAD)
        crops.append(region[cy1:cy2, cx1:cx2])
    return crops


def _selftest():
    from flip.ocr import OcrBox

    # 흰 캔버스에 인쇄 텍스트 흉내(OCR 등록) + 등록 안 된 손글씨 흉내 획
    canvas = np.full((400, 600), 255, np.uint8)
    cv2.putText(canvas, "7-1. x^2-x-6=0", (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    boxes = [OcrBox("7-1. x^2-x-6=0", 0.9, 30, 35, 330, 75)]
    cv2.line(canvas, (380, 250), (420, 300), 0, 3)   # 손글씨 "x" 흉내
    cv2.line(canvas, (420, 250), (380, 300), 0, 3)
    cv2.line(canvas, (440, 275), (480, 275), 0, 3)   # 붙어 있는 획 → 병합 대상
    block = (0, 0, 600, 400)

    region, rects = _candidates(canvas, boxes, block)
    assert rects, "손글씨 후보가 검출돼야 한다"
    for x1, y1, x2, y2 in rects:
        # 마스킹 후 크롭은 인쇄 영역이 아니라 손글씨 쪽에서만 나와야 한다
        assert x1 > 330 and y1 > 100, (x1, y1, x2, y2)
    crops = extract_crops(canvas, boxes, block)
    assert len(crops) == len(rects) and all(c.size > 0 for c in crops)

    # 답을 안 쓴 블록(인쇄만 있음) → 빈 리스트 (호출부가 보류)
    blank = np.full((400, 600), 255, np.uint8)
    cv2.putText(blank, "1. abc", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    assert extract_crops(blank, [OcrBox("1. abc", 0.9, 30, 35, 170, 75)], block) == []

    # 세로 분수: 분자/분수선/분모가 MERGE_GAP보다 벌어져도 크롭 하나로 합쳐진다
    frac = np.full((400, 600), 255, np.uint8)
    cv2.putText(frac, "1", (300, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 3)  # 분자
    cv2.line(frac, (280, 200), (350, 200), 0, 3)                             # 분수선
    cv2.putText(frac, "2", (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 3)  # 분모
    _, frac_rects = _candidates(frac, [], block)
    assert len(frac_rects) == 1, f"세로 분수는 후보 1개로 합쳐져야: {frac_rects}"
    fx1, fy1, fx2, fy2 = frac_rects[0]
    assert fy1 < 130 and fy2 > 270, "분자~분모 전체를 덮어야 한다"
