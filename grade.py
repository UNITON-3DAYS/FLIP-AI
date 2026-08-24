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

from flip import ocr, structure
from flip import equivalence, handwriting, vlm
from flip.db import AnswerDB, MULTIPLE_CHOICE, SUBJECTIVE
from flip.preprocess import preprocess
from flip.results import HOLD, O, PageResult, QuestionResult, X, format_page

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


# ── stub 단계 (후속 티켓이 교체) ─────────────────────────────────────────

def grade_mcq(color, gray, boxes, block, question):
    """객관식: 인쇄 선택지 마커 주변 마킹 검출 (flip/mcq.py)."""
    from flip import mcq  # 지역 import: stub 교체 시 상단 import 충돌 최소화
    return mcq.grade(color, gray, boxes, block, question)


def grade_subjective(color, gray, boxes, block, question):
    """주관식: 손글씨 크롭 추출 → VLM 인식 → SymPy 동치 비교."""
    crops = handwriting.extract_crops(gray, boxes, block)
    if not crops:
        return QuestionResult(question.question_no, HOLD,
                              detail="손글씨 답 없음(또는 인쇄 위 겹쳐씀)")
    if not vlm.available():
        return QuestionResult(question.question_no, HOLD, detail="VLM API 키 없음")
    student = vlm.read_math(crops[0])  # 후보 중 가장 큰 크롭
    if student is None:
        return QuestionResult(question.question_no, HOLD, detail="인식 불확실")
    verdict = equivalence.equivalent(student, question.answer)
    if verdict is None:
        return QuestionResult(question.question_no, HOLD,
                              student_answer=student, detail="파싱 실패")
    return QuestionResult(question.question_no, O if verdict else X,
                          student_answer=student)


# ── 파이프라인 조립 ──────────────────────────────────────────────────────

def grade_page(image_path, db, page_hint=None, debug=False):
    """페이지 사진 1장 -> PageResult."""
    color, gray = preprocess(image_path)

    boxes = []
    if ocr.available():
        boxes = ocr.run_ocr(gray)

    h, w = gray.shape[:2]
    page_no, blocks, hold = structure.analyze(boxes, h, w, db, page_hint)
    questions = db.questions_for(page_no) or []

    if debug and blocks:
        out = Path(str(image_path)).with_suffix("") .name
        structure.draw_debug(color, blocks, f"pred_blocks_{out}.png")

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

    from flip.equivalence import _selftest as equivalence_selftest
    equivalence_selftest()
    from flip.handwriting import _selftest as handwriting_selftest
    handwriting_selftest()
    from flip.vlm import _selftest as vlm_selftest
    vlm_selftest()

    from flip.structure import _selftest as structure_selftest
    structure_selftest()
    from flip.mcq import _selftest as mcq_selftest; mcq_selftest()

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
