"""인쇄 구조 분석: 쪽수 -> 단 -> 문제번호 anchor -> Question Block.

파이프라인 2~7번. OCR 결과([OcrBox])만 입력으로 받으므로 OCR 없이도 테스트 가능.

핵심 방어:
- anchor는 자유 검출이 아니라 DB의 기대 문제번호 목록에 대한 제약 매칭 (LIS).
- 검출 anchor 수 != 기대 문제 수면 페이지 전체 보류 (한 칸 밀림 = 전부 오배치 방지).
"""
import re

import cv2
import numpy as np

# "12." "12)" "7-1." "7-1" 등. 인식 잡음으로 붙는 공백 허용.
ANCHOR_RE = re.compile(r"^\s*(\d{1,3}(?:-\d{1,2})?)\s*[.)]")
ANCHOR_BARE_RE = re.compile(r"^\s*(\d{1,3}-\d{1,2})\s*$")  # 소문항은 점 없이도 인정
# 쎈 등 4자리 연번은 점 없음. 난이도 아이콘이 '00596>'처럼 붙어도 앞 4자리만 취한다.
# 오검출은 기대목록 LIS가 거른다.
ANCHOR_BARE4_RE = re.compile(r"^\s*(\d{4})")
# 쪽수는 "14·Ⅰ.수와 식"처럼 장 제목과 한 박스로 붙어 읽히기도 한다. 접두 숫자만 취하고
# 오검출은 DB valid_pages 필터에 맡긴다.
PAGE_NO_RE = re.compile(r"^\s*(\d{1,3})(?!\d)")
PAGE_NO_TAIL_RE = re.compile(r"(?<!\d)(\d{1,3})\s*$")  # "01 유리수와 소수·15" 꼴은 끝 숫자


def read_page_number(boxes, img_h, img_w, valid_pages):
    """하단 좌/우 모서리 영역에서 쪽수를 읽는다. 실패 시 ""."""
    candidates = []
    for b in boxes:
        if b.cy < img_h * 0.9:          # 하단 10% 띠만
            continue
        matches = (PAGE_NO_RE.match(b.text), PAGE_NO_TAIL_RE.search(b.text))
        # 접두("14·Ⅰ...")·접미("01 유리수...·15") 후보 중 DB 유효 쪽수만 인정
        pages = [m.group(1) for m in matches if m and m.group(1) in valid_pages]
        if not pages:
            continue
        page = pages[0]
        # 모서리(좌우 25%)에 가까울수록 쪽수답다
        edge_dist = min(b.cx, img_w - b.cx)
        if edge_dist > img_w * 0.25:
            continue
        candidates.append((edge_dist, page))
    if not candidates:
        return ""
    return min(candidates)[1]


def split_columns(boxes, img_w):
    """텍스트 박스 x-커버리지로 1단/2단 판별.

    페이지 중앙 30~70%에서 박스가 거의 안 덮는 x의 최장 연속 구간(단 여백 띠)을
    찾아 그 중앙에서 나눈다. 고정 중앙 띠 방식은 본문 줄이 길어 여백이 좁은
    문제집(쎈 등)에서 2단을 놓친다.

    반환: [(x_start, x_end)] 단 목록 (왼쪽부터).
    """
    if not boxes:
        return [(0, img_w)]
    w = int(img_w)
    cov = np.zeros(w + 1, dtype=int)
    for b in boxes:
        x1, x2 = max(0, int(b.x1)), min(w, int(b.x2))
        if x2 > x1:
            cov[x1] += 1
            cov[x2] -= 1
    cov = np.cumsum(cov)[:w]
    lo, hi = int(w * 0.30), int(w * 0.70)
    # 단을 가로지르는 제목 등 소수 예외 허용 (전체의 3%)
    open_ = cov[lo:hi] < max(1, len(boxes) * 0.03)
    best = run = 0
    best_end = -1
    for i, v in enumerate(open_):
        run = run + 1 if v else 0
        if run > best:
            best, best_end = run, i
    if best >= w * 0.015:  # 유의미한 폭의 여백 띠만 인정
        mid = lo + best_end - best // 2
        return [(0, mid), (mid, img_w)]
    return [(0, img_w)]


def _anchor_text(text):
    m = ANCHOR_RE.match(text)
    if m:
        return m.group(1)
    m = ANCHOR_BARE_RE.match(text) or ANCHOR_BARE4_RE.match(text)
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
            if expected_index[chain[-1][1][0]] < expected_index[item[0]] and len(chain) > len(best_prev):
                best_prev = chain
        best.append(best_prev + [(i, item)])
    chain = max(best, key=len) if best else []

    # 4) 오독 보정: 기대 번호 하나가 빠진 자리(체인 이웃 사이 읽기 순서 창)에
    #    anchor 패턴이지만 기대에 없는 후보('0060'->'0900' 오독 등)가 정확히
    #    하나면 그 번호로 인정. 애매하면(0개/2개 이상) 보정하지 않는다.
    chain_by_no = {item[0]: (i, item[1]) for i, item in chain}
    used_idx = {i for i, _ in chain}
    result = []
    for k, no in enumerate(expected_nos):
        if no in chain_by_no:
            result.append((no, chain_by_no[no][1]))
            continue
        lo = max((chain_by_no[p][0] for p in expected_nos[:k] if p in chain_by_no), default=-1)
        hi = min((chain_by_no[p][0] for p in expected_nos[k + 1:] if p in chain_by_no),
                 default=len(ordered))
        window = [ordered[j] for j in range(lo + 1, hi)
                  if j not in used_idx and ordered[j][0] not in expected_index
                  and len(ordered[j][0]) == len(no)]  # 오독은 자릿수를 보존한다
        if len(window) == 1:
            result.append((no, window[0][1]))
    return result


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

    # 오독 보정: '2.'가 '8.'로 읽혀도 그 자리 유일 후보면 '2'로 복원
    boxes_misread = [box("8.", 60, 600) if b.text == "2." else b for b in boxes]
    _, blocks4, hold4 = analyze(boxes_misread, img_h, img_w, db)
    assert hold4 == "" and set(blocks4) == {"1", "2", "3", "4"}, (hold4, blocks4)
    assert blocks4["2"][1] == 600

    # 소문항 패턴
    assert _anchor_text("7-1.") == "7-1" and _anchor_text("7-1") == "7-1"
    assert _anchor_text("(3)") is None
    # 쎈 스타일 4자리 연번 (난이도 아이콘 잡음 포함)
    assert _anchor_text("0062") == "0062" and _anchor_text("00596>") == "0059"
    assert _anchor_text("0063$>") == "0063" and _anchor_text("123") is None

    print("structure selftest OK")


if __name__ == "__main__":
    _selftest()
