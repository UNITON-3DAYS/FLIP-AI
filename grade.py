"""FLIP 문제집 페이지 사진 자동 채점 CLI.

  python grade.py --image samples/p12.jpg --db samples/db.json
  python grade.py --batch samples/ --db samples/db.json
  python grade.py --selftest        # 이미지·OCR·API 없이 스키마/집계 로직 검증

파이프라인 (부모 이슈 참고):
  보정 -> OCR -> 쪽수 -> 단/anchor/블록 -> 유형 분기 -> 객관식|주관식 -> SymPy -> O/X/보류
아직 stub인 단계는 보류를 반환한다. 후속 티켓이 하나씩 실제 구현으로 교체한다.
"""
import argparse
import sys
from pathlib import Path

from flip import ocr
from flip.db import AnswerDB, MULTIPLE_CHOICE, SUBJECTIVE
from flip.preprocess import preprocess
from flip.results import HOLD, O, PageResult, QuestionResult, X, format_page

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


# ── stub 단계 (후속 티켓이 교체) ─────────────────────────────────────────

def analyze_structure(color, gray, boxes, db, page_hint=None):
    """파이프라인 2~7번 stub: 쪽수/단/anchor/블록. AKR 구조 티켓이 교체.

    반환: (page_no, blocks, hold_reason)
    blocks: {question_no: (x1, y1, x2, y2)}
    """
    page_no = page_hint or ""
    if not page_no:
        return "", {}, "쪽수 인식 미구현 (--page 로 지정 가능)"
    if db.questions_for(page_no) is None:
        return page_no, {}, f"DB에 없는 쪽수: {page_no}"
    return page_no, {}, "블록 분할 미구현"


def grade_mcq(color, gray, boxes, block, question):
    """객관식 stub. 마킹 검출 티켓이 교체."""
    return QuestionResult(question.question_no, HOLD, detail="객관식 미구현")


def grade_subjective(color, gray, boxes, block, question):
    """주관식 stub. VLM 인식 티켓이 교체."""
    return QuestionResult(question.question_no, HOLD, detail="주관식 미구현")


# ── 파이프라인 조립 ──────────────────────────────────────────────────────

def grade_page(image_path, db, page_hint=None, debug=False):
    """페이지 사진 1장 -> PageResult."""
    color, gray = preprocess(image_path)

    boxes = []
    if ocr.available():
        boxes = ocr.run_ocr(gray)

    page_no, blocks, hold = analyze_structure(color, gray, boxes, db, page_hint)
    questions = db.questions_for(page_no) or []

    if hold:
        return PageResult(image=str(image_path), page_no=page_no,
                          results=[QuestionResult(q.question_no, HOLD) for q in questions],
                          hold_reason=hold)

    results = []
    for q in questions:
        block = blocks.get(q.question_no)
        if block is None:
            results.append(QuestionResult(q.question_no, HOLD, detail="블록 없음"))
        elif q.qtype == MULTIPLE_CHOICE:
            results.append(grade_mcq(color, gray, boxes, block, q))
        else:
            results.append(grade_subjective(color, gray, boxes, block, q))
    return PageResult(image=str(image_path), page_no=page_no, results=results)


def run_batch(folder, db, debug=False):
    images = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        sys.exit(f"이미지 없음: {folder}")
    pages = [grade_page(p, db, debug=debug) for p in images]
    for pr in pages:
        print(format_page(pr))
    total = {O: 0, X: 0, HOLD: 0}
    for pr in pages:
        for k, v in pr.counts().items():
            total[k] += v
    print(f"== 합계: O {total[O]} / X {total[X]} / 보류 {total[HOLD]}")
    return pages


# ── selftest ─────────────────────────────────────────────────────────────

def selftest():
    """이미지·OCR·API 없이 스키마 파싱, 유형 분기, 집계를 검증."""
    db = AnswerDB.from_dict({
        "book": "테스트북",
        "pages": {
            "12": [
                {"question_no": "1", "type": "multiple_choice", "answer": 3, "num_choices": 5},
                {"question_no": "2", "type": "subjective", "answer": "-367"},
                {"question_no": "7-1", "type": "subjective", "answer": ["-2", "3"]},
            ],
        },
    })
    # 스키마 파싱
    qs = db.questions_for("12")
    assert [q.question_no for q in qs] == ["1", "2", "7-1"]
    assert qs[0].qtype == MULTIPLE_CHOICE and qs[0].answer == 3
    assert qs[2].qtype == SUBJECTIVE and qs[2].answer == ["-2", "3"]
    assert db.questions_for("99") is None
    assert db.valid_pages() == {"12"}

    # 페이지 결과 집계 (전체 보류 페이지)
    pr = PageResult(image="x.jpg", page_no="12",
                    results=[QuestionResult(q.question_no, HOLD) for q in qs],
                    hold_reason="테스트")
    assert pr.counts() == {O: 0, X: 0, HOLD: 3}

    # 정상 페이지 집계
    pr2 = PageResult(image="x.jpg", page_no="12", results=[
        QuestionResult("1", O), QuestionResult("2", X), QuestionResult("7-1", HOLD)])
    assert pr2.counts() == {O: 1, X: 1, HOLD: 1}
    assert "1번" in format_page(pr2).replace(" ", "")

    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", help="페이지 사진 1장")
    ap.add_argument("--batch", help="사진 폴더 (일괄 처리)")
    ap.add_argument("--db", help="정답 mock DB JSON")
    ap.add_argument("--page", help="쪽수 수동 지정 (쪽수 인식 대신)")
    ap.add_argument("--debug", action="store_true", help="블록 오버레이 등 디버그 출력")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.db:
        ap.error("--db 필요")
    db = AnswerDB.load(args.db)
    if args.image:
        print(format_page(grade_page(args.image, db, page_hint=args.page, debug=args.debug)))
    elif args.batch:
        run_batch(args.batch, db, debug=args.debug)
    else:
        ap.error("--image 또는 --batch 필요")


if __name__ == "__main__":
    main()
