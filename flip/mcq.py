"""객관식 마킹 검출: 인쇄 마커 기준 형제 ROI 비교.

손글씨(체크·동그라미) 자체를 인식하려 하지 않는다. 인쇄된 선택지 마커(①~⑤)를
기준점으로 잡고, 각 마커 주변 ROI의 특징(잉크 비율·큰 폐곡선·대각선 획·템플릿
잔여 잉크)을 같은 문제의 형제 선택지끼리 비교해, 형제 대비 튀는(z-score) 선택지
하나를 학생 마킹으로 판정한다.

경로:
  마커 검출 -> ROI 특징 -> 형제 robust z-score -> 이상치 1개면 O/X
  이상치 0개 + ROI 밖 잉크 덩어리 -> "숫자 작성 추정" 보류 (VLM 티켓이 연결)
  그 외(다중 마킹, 마커 미검출, 애매) -> 보류
"""
import re

import cv2
import numpy as np

from flip.results import HOLD, O, QuestionResult, X

# ── 튜닝 포인트 ──────────────────────────────────────────────────────────
CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩"          # 유니코드 선택지 마커 (번호 = 인덱스+1)
FALLBACK_RE = re.compile(r"\(?([1-9])\)")   # OCR이 ①을 "1)" "(1)"로 깨는 경우
ROI_PAD = 1.5          # ROI 패딩 = 마커 높이 x 이 배수 (학생 동그라미가 들어올 범위)
ADAPT_BLOCK = 31       # adaptive threshold 블록 크기 (홀수)
ADAPT_C = 10           # adaptive threshold 보정값
CONTOUR_SCALE = 1.3    # 마커 높이의 이 배수보다 큰 외곽 contour = 학생 동그라미
DIAG_MIN_DEG = 20      # 대각선 획(체크)으로 인정하는 기울기 범위 (도)
DIAG_MAX_DEG = 70
W_CONTOUR = 0.5        # 특징 가중치: 큰 폐곡선
W_DIAG = 0.3           # 특징 가중치: 대각선 획
W_RESIDUAL = 1.0       # 특징 가중치: median 템플릿 차영상 잔여 잉크
STD_FLOOR = 0.02       # z-score 분모 하한 (무마킹 페이지의 노이즈 증폭 방지)
Z_MARK = 2.5           # 마킹 이상치로 인정하는 z-score 임계
Z_GAP = 1.0            # 1위-2위 z-score 최소 격차 (미달이면 애매 -> 보류)
TEMPLATE_MIN = 5       # median 템플릿을 만들 최소 마커 출현 횟수 (페이지 전체)
TEMPLATE_SIZE = 24     # 템플릿 정규화 크기 (px)
RESIDUAL_DIFF = 40     # 템플릿 차영상에서 잉크로 보는 밝기 차
BLOB_AREA_SCALE = 0.5  # 숫자 작성 추정 덩어리 최소 면적 = 마커 높이^2 x 이 값


# ── 마커 검출 ────────────────────────────────────────────────────────────

def _marker_spans(text, num_choices):
    """텍스트 안의 선택지 마커 위치. 반환 [(번호, 시작 글자 idx, 끝 글자 idx)].

    유니코드 ①~ 우선("① 3x+1"처럼 섞여 있어도 인정). 하나도 없으면
    OCR 깨짐 대비 "1)" "(1)" 패턴 fallback.
    """
    spans = []
    for i, ch in enumerate(text):
        k = CIRCLED.find(ch)
        if 0 <= k < num_choices:
            spans.append((k + 1, i, i + 1))
    if spans:
        return spans
    for m in FALLBACK_RE.finditer(text):
        n = int(m.group(1))
        if n <= num_choices:
            spans.append((n, m.start(), m.end()))
    return spans


def _span_rect(box, i, j):
    """OcrBox 안 글자 구간 [i, j) -> 글자당 폭 비례로 근사한 사각형."""
    cw = (box.x2 - box.x1) / max(1, len(box.text))
    return (box.x1 + cw * i, box.y1, box.x1 + cw * j, box.y2)


def _find_markers(boxes, block, num_choices):
    """block 안 OcrBox에서 선택지 마커 사각형 추정. 반환 {번호: (x1,y1,x2,y2)}."""
    bx1, by1, bx2, by2 = block
    markers = {}
    for b in boxes:
        if not b.text or not (bx1 <= b.cx < bx2 and by1 <= b.cy < by2):
            continue
        for n, i, j in _marker_spans(b.text, num_choices):
            markers.setdefault(n, _span_rect(b, i, j))  # 첫 검출 우선
    _extrapolate_markers(markers, num_choices, block)
    return markers


def _extrapolate_markers(markers, num_choices, block):
    """미검출 마커 위치를 격자 외삽으로 복원 (in-place).

    학생이 마커 위에 동그라미를 치면 OCR이 그 마커를 못 읽는다. 검출된
    마커들로 행(cy 클러스터)·열 간격(dx)을 추정해 빠진 번호 위치를 채운다.
    행 전체가 사라진 경우 등 추정이 블록을 벗어나면 채우지 않는다.
    """
    missing = [n for n in range(1, num_choices + 1) if n not in markers]
    if not missing or len(markers) < 2:
        return
    h = float(np.median([m[3] - m[1] for m in markers.values()]))
    w = float(np.median([m[2] - m[0] for m in markers.values()]))

    rows = []  # [[번호, ...] 위 행부터], 같은 행 = y1 차이 < 0.8h
    for n in sorted(markers, key=lambda k: markers[k][1]):
        if rows and abs(markers[rows[-1][0]][1] - markers[n][1]) < h * 0.8:
            rows[-1].append(n)
        else:
            rows.append([n])
    for r in rows:
        r.sort()

    dxs = [(markers[b][0] - markers[a][0]) / (b - a)
           for row in rows for a, b in zip(row, row[1:])]
    dx = float(np.median(dxs)) if dxs else None
    x0 = min(m[0] for m in markers.values())

    if dx is None:
        # 세로 1열 배치: 이웃 중점 보간만
        for n in missing:
            if (n - 1) in markers and (n + 1) in markers:
                a, b = markers[n - 1], markers[n + 1]
                markers[n] = tuple((p + q) / 2 for p, q in zip(a, b))
        return

    row_start, row_y = {}, {}
    for ri, row in enumerate(rows):
        f = markers[row[0]]
        row_start[ri] = row[0] - round((f[0] - x0) / dx)
        row_y[ri] = (f[1], f[3])
    for n in missing:
        cand = [ri for ri, s in row_start.items() if s <= n]
        if not cand:
            continue
        ri = max(cand, key=lambda r: row_start[r])
        slot = n - row_start[ri]
        x1 = x0 + slot * dx
        if slot < 0 or x1 + w > block[2]:  # 행 전체 소실 등 추정 불가
            continue
        y1, y2 = row_y[ri]
        markers[n] = (x1, y1, x1 + w, y2)


# ── ROI 특징 ─────────────────────────────────────────────────────────────

def _binarize(gray):
    """어두운 픽셀(잉크) = 255 인 이진 이미지."""
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, ADAPT_BLOCK, ADAPT_C)


def _roi_rect(marker, img_h, img_w):
    """마커 사각형을 높이 x ROI_PAD 만큼 사방으로 확장 (이미지 경계 클램프)."""
    x1, y1, x2, y2 = marker
    pad = (y2 - y1) * ROI_PAD
    return (int(max(0, x1 - pad)), int(max(0, y1 - pad)),
            int(min(img_w, x2 + pad)), int(min(img_h, y2 + pad)))


def _features(binary, roi, marker_h):
    """ROI 하나의 마킹 특징 점수 (형제끼리 비교하는 상대값이므로 절대 의미 없음)."""
    x1, y1, x2, y2 = roi
    patch = binary[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0

    # 1) 잉크 비율
    ink = float(np.count_nonzero(patch)) / patch.size

    # 2) 큰 폐곡선: 마커보다 뚜렷이 큰 외곽 contour = 마커를 감싼 학생 동그라미
    contours, _ = cv2.findContours(patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = 0.0
    for c in contours:
        _, _, w, h = cv2.boundingRect(c)
        if w > marker_h * CONTOUR_SCALE and h > marker_h * CONTOUR_SCALE:
            big = 1.0
            break

    # 3) 긴 대각선 획 = 체크 표시
    diag = 0.0
    lines = cv2.HoughLinesP(patch, 1, np.pi / 180, threshold=20,
                            minLineLength=int(marker_h), maxLineGap=3)
    if lines is not None:
        for lx1, ly1, lx2, ly2 in lines.reshape(-1, 4):  # 빌드에 따라 (N,1,4)/(N,4) 혼재
            ang = abs(np.degrees(np.arctan2(int(ly2) - int(ly1), int(lx2) - int(lx1)))) % 180
            ang = min(ang, 180 - ang)
            if DIAG_MIN_DEG <= ang <= DIAG_MAX_DEG:
                diag = 1.0
                break

    return ink + W_CONTOUR * big + W_DIAG * diag


# ── median 템플릿 (보조 특징) ────────────────────────────────────────────

def _marker_crops(boxes, gray, num_choices):
    """페이지 전체에서 마커별 크롭 수집 (동일 크기 리사이즈). 반환 {번호: [crop]}."""
    crops = {}
    for b in boxes:
        if not b.text:
            continue
        for n, i, j in _marker_spans(b.text, num_choices):
            x1, y1, x2, y2 = [int(v) for v in _span_rect(b, i, j)]
            if x2 - x1 < 3 or y2 - y1 < 3:
                continue
            crop = gray[max(0, y1):y2, max(0, x1):x2]
            if crop.size:
                crops.setdefault(n, []).append(
                    cv2.resize(crop, (TEMPLATE_SIZE, TEMPLATE_SIZE)))
    return crops


def _residual(sibling_crops, my_crop):
    """동일 마커 median 템플릿("깨끗한 마커") 대비 잔여 잉크 비율.

    표본이 TEMPLATE_MIN 미만이면 median이 학생 잉크에 오염될 수 있어 생략(0).
    """
    if my_crop is None or len(sibling_crops) < TEMPLATE_MIN:
        return 0.0
    template = np.median(np.stack(sibling_crops).astype(np.int16), axis=0)
    diff = template - my_crop.astype(np.int16)  # 추가 잉크 = 템플릿보다 어두움
    return float(np.count_nonzero(diff > RESIDUAL_DIFF)) / diff.size


# ── 형제 비교 ────────────────────────────────────────────────────────────

def _zscores(scores):
    """자기 제외 형제 대비 robust z-score.

    mean/std 대신 median/MAD를 쓴다: 다중 마킹(이상치 2개)에서도 형제 기준선이
    오염되지 않아 두 마킹 모두 높은 z로 잡힌다. MAD가 0에 가까우면(형제가 거의
    동일) STD_FLOOR로 하한을 둬 노이즈 증폭을 막는다.
    """
    z = []
    for i in range(len(scores)):
        rest = np.array(scores[:i] + scores[i + 1:])
        med = float(np.median(rest))
        mad = float(np.median(np.abs(rest - med)))
        z.append((scores[i] - med) / max(mad, STD_FLOOR))
    return z


# ── 숫자 직접 작성 경로 ──────────────────────────────────────────────────

def _stray_ink(binary, block, rois, boxes, marker_h):
    """마커 ROI와 OCR 박스를 지운 뒤 남는 큰 잉크 덩어리 유무.

    True면 학생이 답을 숫자로 직접 썼을 가능성 (OCR이 손글씨를 못 잡은 영역).
    """
    bx1, by1, bx2, by2 = [int(v) for v in block]
    mask = binary[by1:by2, bx1:bx2].copy()

    def wipe(x1, y1, x2, y2):
        xs, ys = max(0, int(x1) - bx1 - 2), max(0, int(y1) - by1 - 2)
        xe, ye = max(0, int(x2) - bx1 + 2), max(0, int(y2) - by1 + 2)
        mask[ys:ye, xs:xe] = 0

    for r in rois:
        wipe(*r)
    for b in boxes:
        if b.x2 < bx1 or b.x1 > bx2 or b.y2 < by1 or b.y1 > by2:
            continue
        wipe(b.x1, b.y1, b.x2, b.y2)

    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    min_area = marker_h * marker_h * BLOB_AREA_SCALE
    return any(stats[i, cv2.CC_STAT_AREA] >= min_area for i in range(1, n))


# ── 판정 ─────────────────────────────────────────────────────────────────

def grade(color, gray, boxes, block, question):
    """객관식 한 문제 채점. 반환 QuestionResult (O/X/보류)."""
    img_h, img_w = gray.shape[:2]

    markers = _find_markers(boxes, block, question.num_choices)
    if len(markers) < question.num_choices:
        missing = sorted(set(range(1, question.num_choices + 1)) - set(markers))
        return QuestionResult(question.question_no, HOLD,
                              detail=f"선택지 마커 미검출: {missing}")

    binary = _binarize(gray)
    page_crops = _marker_crops(boxes, gray, question.num_choices)
    marker_h = float(np.mean([m[3] - m[1] for m in markers.values()]))

    rois, scores = [], []
    for n in range(1, question.num_choices + 1):
        mx1, my1, mx2, my2 = [int(v) for v in markers[n]]
        roi = _roi_rect(markers[n], img_h, img_w)
        rois.append(roi)
        score = _features(binary, roi, marker_h)
        crop = gray[max(0, my1):my2, max(0, mx1):mx2]
        if crop.size and min(crop.shape) >= 3:
            crop = cv2.resize(crop, (TEMPLATE_SIZE, TEMPLATE_SIZE))
            score += W_RESIDUAL * _residual(page_crops.get(n, []), crop)
        scores.append(score)

    z = _zscores(scores)
    order = sorted(range(len(z)), key=lambda i: z[i], reverse=True)
    top, second = order[0], order[1]
    high = [i for i in order if z[i] >= Z_MARK]

    # 복수정답("모두 고르면")은 answer가 리스트. 마킹 집합과 정답 집합을 비교한다.
    answers = question.answer if isinstance(question.answer, list) else [question.answer]

    if len(high) >= 2:
        marked = sorted(i + 1 for i in high)
        marked_s = ",".join(str(m) for m in marked)
        if len(answers) >= 2:
            verdict = O if marked == sorted(answers) else X
            return QuestionResult(question.question_no, verdict,
                                  student_answer=marked_s,
                                  detail="복수 마킹 검출")
        return QuestionResult(question.question_no, HOLD,
                              detail=f"다중 마킹 추정: {marked_s}번")
    if len(high) == 1:
        if z[top] - z[second] < Z_GAP:
            return QuestionResult(question.question_no, HOLD,
                                  detail=f"판정 애매: {top + 1}번 우세하나 격차 부족 "
                                         f"(z {z[top]:.1f} vs {z[second]:.1f})")
        choice = top + 1
        verdict = O if [choice] == sorted(answers) else X
        return QuestionResult(question.question_no, verdict,
                              student_answer=str(choice),
                              detail=f"마킹 검출 (z={z[top]:.1f})")

    # 마킹 이상치 없음 -> 숫자 직접 작성 여부 확인
    if _stray_ink(binary, block, rois, boxes, marker_h):
        return QuestionResult(question.question_no, HOLD,
                              detail="숫자 작성 추정 (VLM 인식 필요)")
    return QuestionResult(question.question_no, HOLD, detail="마킹 없음")


# ── selftest ─────────────────────────────────────────────────────────────

def _selftest():
    """이미지 파일·OCR 없이: 합성 캔버스에 인쇄 마커를 그리고 판정을 검증."""
    from flip.db import MULTIPLE_CHOICE, Question
    from flip.ocr import OcrBox

    W, H = 400, 720
    block = (0, 0, W, H)
    centers = [(60, 90 + i * 130) for i in range(5)]

    def base_page():
        g = np.full((H, W), 255, np.uint8)
        boxes = []
        for i, (cx, cy) in enumerate(centers):
            cv2.circle(g, (cx, cy), 14, 0, 2)  # 인쇄 마커: 원 + 숫자 흉내
            cv2.putText(g, str(i + 1), (cx - 6, cy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
            boxes.append(OcrBox(chr(0x2460 + i), 0.9,
                                cx - 16, cy - 16, cx + 16, cy + 16))
        return g, boxes

    def run(g, boxes, answer):
        color = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        q = Question("1", MULTIPLE_CHOICE, answer, num_choices=5)
        return grade(color, g, boxes, block, q)

    # 1) 3번 마커에 굵은 학생 동그라미 -> 3번 검출. 정답 3이면 O, 1이면 X.
    g, boxes = base_page()
    cv2.circle(g, centers[2], 26, 0, 3)
    r = run(g, boxes, 3)
    assert r.verdict == O and r.student_answer == "3", r
    r = run(g, boxes, 1)
    assert r.verdict == X and r.student_answer == "3", r

    # 2) 무마킹 -> 보류
    g, boxes = base_page()
    r = run(g, boxes, 3)
    assert r.verdict == HOLD and "마킹 없음" in r.detail, r

    # 3) 다중 마킹(2, 4번) -> 보류
    g, boxes = base_page()
    cv2.circle(g, centers[1], 26, 0, 3)
    cv2.circle(g, centers[3], 26, 0, 3)
    r = run(g, boxes, 3)
    assert r.verdict == HOLD and "다중 마킹" in r.detail, r

    # 3b) 복수정답: 2·4번 마킹, 정답 [2,4] -> O / [2,3] -> X / 단일 마킹 -> X
    r = run(g, boxes, [2, 4])
    assert r.verdict == O and r.student_answer == "2,4", r
    r = run(g, boxes, [2, 3])
    assert r.verdict == X and r.student_answer == "2,4", r
    g2, boxes2 = base_page()
    cv2.circle(g2, centers[1], 26, 0, 3)
    r = run(g2, boxes2, [2, 4])
    assert r.verdict == X and r.student_answer == "2", r

    # 4) 마커 일부 미검출 -> 보류
    g, boxes = base_page()
    r = run(g, boxes[:4], 3)
    assert r.verdict == HOLD and "마커 미검출" in r.detail, r

    # 4b) 마킹에 가려 마커 하나 미검출 -> 이웃 보간으로 복원해 채점
    g, boxes = base_page()
    cv2.circle(g, centers[2], 26, 0, 3)
    r = run(g, [b for b in boxes if b.text != chr(0x2460 + 2)], 3)
    assert r.verdict == O and r.student_answer == "3", r

    # 5) 무마킹 + ROI 밖 잉크 덩어리 -> 숫자 작성 추정 보류
    g, boxes = base_page()
    cv2.circle(g, (280, 350), 15, 0, -1)
    r = run(g, boxes, 3)
    assert r.verdict == HOLD and "숫자 작성" in r.detail, r

    # 마커 패턴: 유니코드 혼합 텍스트, fallback
    assert [s[0] for s in _marker_spans("① 3x+1", 5)] == [1]
    assert [s[0] for s in _marker_spans("(3)", 5)] == [3]
    assert [s[0] for s in _marker_spans("5)", 5)] == [5]
    assert _marker_spans("[3점]", 5) == []

    print("mcq selftest OK")


if __name__ == "__main__":
    _selftest()
