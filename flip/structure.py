"""인쇄 구조 분석: 쪽수 -> 단 -> 문제번호 anchor -> Question Block.

파이프라인 2~7번. OCR 결과([OcrBox])만 입력으로 받으므로 OCR 없이도 테스트 가능.

핵심 방어:
- anchor는 자유 검출이 아니라 DB의 기대 문제번호 목록에 대한 제약 매칭 (LIS).
- 검출 anchor 수 != 기대 문제 수면 페이지 전체 보류 (한 칸 밀림 = 전부 오배치 방지).
"""
import re

import cv2

# "12." "12)" "7-1." "7-1" 등. 인식 잡음으로 붙는 공백 허용.
ANCHOR_RE = re.compile(r"^\s*(\d{1,3}(?:-\d{1,2})?)\s*[.)]")
ANCHOR_BARE_RE = re.compile(r"^\s*(\d{1,3}-\d{1,2})\s*$")  # 소문항은 점 없이도 인정
PAGE_NO_RE = re.compile(r"^\s*(\d{1,3})\s*$")


def read_page_number(boxes, img_h, img_w, valid_pages):
    """하단 좌/우 모서리 영역에서 쪽수를 읽는다. 실패 시 ""."""
    candidates = []
    for b in boxes:
        if b.cy < img_h * 0.9:          # 하단 10% 띠만
            continue
        m = PAGE_NO_RE.match(b.text)
        if not m:
            continue
        page = m.group(1)
        if page not in valid_pages:      # DB 유효 쪽수로 오인식 필터
            continue
        # 모서리(좌우 25%)에 가까울수록 쪽수답다
        edge_dist = min(b.cx, img_w - b.cx)
        if edge_dist > img_w * 0.25:
            continue
        candidates.append((edge_dist, page))
    if not candidates:
        return ""
    return min(candidates)[1]


def split_columns(boxes, img_w):
    """텍스트 박스 x 분포의 세로 여백 띠로 1단/2단 판별.

    반환: [(x_start, x_end)] 단 목록 (왼쪽부터).
    """
    if not boxes:
        return [(0, img_w)]
    # 페이지 중앙 40~60% 구간에 박스가 걸치지 않는 세로 띠가 있으면 2단
    mid_lo, mid_hi = img_w * 0.42, img_w * 0.58
    gap_hits = [b for b in boxes if b.x1 < mid_hi and b.x2 > mid_lo]
    # 걸친 박스가 전체의 10% 미만이면 중앙 여백으로 본다 (제목 등 소수 예외 허용)
    if len(gap_hits) < max(2, len(boxes) * 0.10):
        mid = img_w / 2
        return [(0, mid), (mid, img_w)]
    return [(0, img_w)]


def _anchor_text(text):
    m = ANCHOR_RE.match(text)
    if m:
        return m.group(1)
    m = ANCHOR_BARE_RE.match(text)
    return m.group(1) if m else None


def find_anchors(boxes, columns, expected_nos):
    """anchor 3단 필터: 패턴 -> 단 좌측 정렬 -> 기대 목록 LIS 매칭.

    반환: [(question_no, box)] — 읽기 순서 (왼쪽 단 위->아래, 다음 단).
    """
    per_column = []
    for cx1, cx2 in columns:
        # 1) 패턴: 문제번호 꼴 텍스트만
        cands = []
        for b in boxes:
            if not (cx1 <= b.cx < cx2):
                continue
            no = _anchor_text(b.text)
            if no is not None:
                cands.append((no, b))
        if not cands:
            per_column.append([])
            continue
        # 2) 좌측 정렬: 단 시작점 근처(단 너비 20% 이내)에 붙은 것만.
        #    선택지 번호·배점·지문 속 숫자는 들여써져 있어 걸러진다.
        col_w = cx2 - cx1
        left_edge = min(b.x1 for _, b in cands)
        cands = [(no, b) for no, b in cands if b.x1 - left_edge < col_w * 0.20]
        cands.sort(key=lambda item: item[1].cy)  # 위 -> 아래
        per_column.append(cands)

    ordered = [item for col in per_column for item in col]

    # 3) 기대 목록 제약: 기대 순서(expected_nos)의 부분수열 중 가장 긴 것(LIS)만 남긴다.
    #    오검출(기대에 없는 번호, 순서가 어긋난 번호)이 떨어져 나간다.
    expected_index = {no: i for i, no in enumerate(expected_nos)}
    seq = [(i, item) for i, item in enumerate(ordered) if item[0] in expected_index]
    best = []
    for i, item in seq:  # O(n^2) LIS — anchor는 페이지당 수십 개라 충분
        best_prev = []
        for j in range(len(best)):
            chain = best[j]
            if expected_index[chain[-1][0]] < expected_index[item[0]] and len(chain) > len(best_prev):
                best_prev = chain
        best.append(best_prev + [item])
    if not best:
        return []
    chain = max(best, key=len)
    return chain


def cut_blocks(anchors, columns, img_h):
    """anchor -> Question Block 좌표. 각 단에서 anchor부터 다음 anchor 직전까지.

    반환: {question_no: (x1, y1, x2, y2)}
    """
    blocks = {}
    for cx1, cx2 in columns:
        col_anchors = [(no, b) for no, b in anchors if cx1 <= b.cx < cx2]
        for i, (no, b) in enumerate(col_anchors):
            y1 = b.y1
            y2 = col_anchors[i + 1][1].y1 if i + 1 < len(col_anchors) else img_h
            blocks[no] = (int(cx1), int(y1), int(cx2), int(y2))
    return blocks


def analyze(boxes, img_h, img_w, db, page_hint=None):
    """파이프라인 2~7번 조립. 반환: (page_no, blocks, hold_reason)."""
    page_no = page_hint or read_page_number(boxes, img_h, img_w, db.valid_pages())
    if not page_no:
        return "", {}, "쪽수를 읽지 못함 (--page 로 지정 가능)"
    questions = db.questions_for(page_no)
    if questions is None:
        return page_no, {}, f"DB에 없는 쪽수: {page_no}"

    expected_nos = [q.question_no for q in questions]
    columns = split_columns(boxes, img_w)
    anchors = find_anchors(boxes, columns, expected_nos)

    # 블록 수 검증: 하나라도 어긋나면 페이지 전체 보류 (오배치가 미검출보다 나쁘다)
    if len(anchors) != len(expected_nos):
        found = [no for no, _ in anchors]
        return page_no, {}, (f"anchor 수 불일치: 기대 {len(expected_nos)}개 {expected_nos}, "
                             f"검출 {len(found)}개 {found}")

    blocks = cut_blocks(anchors, columns, img_h)
    return page_no, blocks, ""


def draw_debug(color_img, blocks, path):
    """블록 경계 오버레이 이미지를 저장한다 (--debug)."""
    vis = color_img.copy()
    for no, (x1, y1, x2, y2) in blocks.items():
        cv2.rectangle(vis, (x1, y1), (x2 - 2, y2 - 2), (0, 0, 255), 2)
        cv2.putText(vis, no, (x1 + 5, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imwrite(str(path), vis)


# ── selftest ─────────────────────────────────────────────────────────────

def _selftest():
    from flip.db import AnswerDB
    from flip.ocr import OcrBox

    def box(text, x, y, w=40, h=20):
        return OcrBox(text, 0.95, x, y, x + w, y + h)

    db = AnswerDB.from_dict({"book": "t", "pages": {
        "12": [{"question_no": str(n), "type": "subjective", "answer": "1"} for n in (1, 2, 3, 4)]}})

    # 2단 페이지: 왼쪽 단 1,2 / 오른쪽 단 3,4. 잡음: 선택지 번호(들여씀), 배점, 쪽수.
    img_w, img_h = 1000, 1400
    boxes = [
        box("1.", 60, 100), box("어떤 문제 지문", 110, 100, 250),
        box("3", 200, 160),            # 지문 속 숫자 (패턴 불일치)
        box("2.", 60, 600), box("[3점]", 380, 600),   # 배점 (패턴 불일치)
        box("3.", 620, 100), box("4.", 620, 700),
        box("1)", 750, 200),           # 선택지 번호 (들여씀 -> 좌측정렬 필터)
        box("12", 80, 1350),           # 쪽수 (하단 좌측)
    ]
    # 쪽수
    assert read_page_number(boxes, img_h, img_w, db.valid_pages()) == "12"
    assert read_page_number(boxes, img_h, img_w, {"99"}) == ""  # DB 필터
    # 단
    cols = split_columns(boxes, img_w)
    assert len(cols) == 2, cols
    # 전체 분석
    page_no, blocks, hold = analyze(boxes, img_h, img_w, db)
    assert page_no == "12" and hold == "", hold
    assert set(blocks) == {"1", "2", "3", "4"}
    # 블록 경계: 1번은 2번 anchor에서 끝난다
    assert blocks["1"][3] == 600 and blocks["2"][3] == img_h
    # 오른쪽 단 블록은 오른쪽 절반에 있다
    assert blocks["3"][0] >= 500

    # anchor 누락 시 페이지 보류
    boxes_missing = [b for b in boxes if b.text != "2."]
    _, blocks2, hold2 = analyze(boxes_missing, img_h, img_w, db)
    assert blocks2 == {} and "anchor 수 불일치" in hold2

    # 오검출(기대에 없는 번호)은 LIS가 걸러낸다
    boxes_noise = boxes + [box("9.", 60, 900)]
    _, blocks3, hold3 = analyze(boxes_noise, img_h, img_w, db)
    assert hold3 == "" and set(blocks3) == {"1", "2", "3", "4"}

    # 소문항 패턴
    assert _anchor_text("7-1.") == "7-1" and _anchor_text("7-1") == "7-1"
    assert _anchor_text("(3)") is None

    print("structure selftest OK")


if __name__ == "__main__":
    _selftest()
