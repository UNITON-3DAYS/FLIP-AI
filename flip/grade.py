"""FLIP 문제집 페이지 사진 자동 채점 CLI.

  python -m flip.grade --image samples/p12.jpg --db samples/db.json
  python -m flip.grade --batch samples/ --db samples/db.json
  python -m flip.grade --selftest        # 이미지·OCR·API 없이 스키마/집계 로직 검증

파이프라인 (부모 이슈 참고):
  보정 -> OCR -> 쪽수 -> 단/anchor/블록 -> 유형 분기 -> 객관식|주관식 -> SymPy -> O/X/보류
아직 stub인 단계는 보류를 반환한다. 후속 티켓이 하나씩 실제 구현으로 교체한다.
"""
import argparse
import concurrent.futures
import os
import re
import sys
from pathlib import Path

from flip import ocr, structure
from flip import equivalence, vlm
from flip.db import AnswerDB, MULTIPLE_CHOICE, SUBJECTIVE
from flip.preprocess import preprocess
from flip.results import HOLD, O, PageResult, QuestionResult, X, format_page

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


# ── 판정 로직 (블록/전체페이지 그레이더가 공유) ──────────────────────────

def _verdict_mcq(question, picked):
    """마킹 번호 리스트 → O/X. 정답 집합과 순서 무관 일치면 O."""
    answer = question.answer if isinstance(question.answer, list) else [question.answer]
    ok = set(picked) == {int(a) for a in answer}
    return QuestionResult(question.question_no, O if ok else X,
                          student_answer=",".join(map(str, picked)))


def _verdict_subjective(question, student):
    """손글씨 답 문자열 → O/X/보류(파싱 실패). SymPy 동치 비교."""
    verdict = equivalence.equivalent(student, question.answer)
    if verdict is None:
        return QuestionResult(question.question_no, HOLD,
                              student_answer=student, detail="파싱 실패")
    return QuestionResult(question.question_no, O if verdict else X,
                          student_answer=student)


# ── 블록 그레이더: OCR로 블록 절단 → 문제별 크롭을 VLM에 판독 ────────────

def grade_mcq(color, gray, boxes, block, question):
    """객관식: 문제 블록을 통째로 VLM에 넘겨 동그라미 친 번호를 읽는다.

    구 flip/mcq.py CV 방식(인쇄 마커 기준 형제 비교)은 실사진에서 실패(12중 1정답)라
    블록크롭 VLM 판독으로 대체. 마킹 집합이 정답 집합과 같으면 O.
    """
    if not vlm.available():
        return QuestionResult(question.question_no, HOLD, detail="VLM API 키 없음")
    x1, y1, x2, y2 = block
    picked = vlm.read_mcq(color[y1:y2, x1:x2])
    if picked is None:
        return QuestionResult(question.question_no, HOLD, detail="마킹 인식 불확실")
    return _verdict_mcq(question, picked)


def grade_subjective(color, gray, boxes, block, question):
    """주관식: 문제 블록을 통째로 VLM에 넘겨 손글씨 답 인식 → SymPy 동치 비교.

    구 손글씨격리(OCR 마스킹) 방식은 실사진에서 0/5로 실패 — 격리 크롭이 인쇄
    잔재를 답으로 오독하거나 손글씨를 통째로 놓쳤다. 블록크롭을 통으로 넘기고
    프롬프트로 "손글씨만 읽어라"를 지시하는 편이 5/5로 안정적(측정). 2회검증 유지.
    """
    if not vlm.available():
        return QuestionResult(question.question_no, HOLD, detail="VLM API 키 없음")
    x1, y1, x2, y2 = block
    student = vlm.read_math(color[y1:y2, x1:x2])
    if student is None:
        return QuestionResult(question.question_no, HOLD, detail="인식 불확실")
    return _verdict_subjective(question, student)


# ── 파이프라인 조립 ──────────────────────────────────────────────────────

def grade_page(image_path, db, page_hint=None, debug=False):
    """페이지 사진 1장(파일 경로) -> PageResult."""
    color, gray = preprocess(image_path)
    return grade_prepared(color, gray, db, page_hint=page_hint,
                          label=str(image_path), debug=debug)


def grade_prepared(color, gray, db, page_hint=None, label="", debug=False):
    """보정 끝난 (컬러, 그레이) 이미지 -> PageResult. 서버(api)·CLI 공용 진입점.

    두 그레이더를 FLIP_GRADER로 갈아끼운다:
      - "blocks"(기본): OCR로 쪽수·문제 블록을 절단해 문제별 크롭을 VLM에 판독.
      - "fullpage": OCR 없이 페이지 전체를 VLM 1콜로 판독(RAM 0·비용↓, 쪽수도 VLM이 읽음).
    반환 PageResult 계약은 동일해서 소비자(api)는 어느 쪽인지 몰라도 된다.
    """
    if os.environ.get("FLIP_GRADER", "blocks") == "fullpage":
        return _grade_fullpage(color, db, page_hint, label)
    return _grade_blocks(color, gray, db, page_hint, label, debug)


def _qkey(k):
    """문제번호 키 정규화: 순수 숫자면 앞자리 0 제거 ("0021"→"21"). "7-1"·"0"은 유지."""
    return re.sub(r"^0+(\d)", r"\1", str(k))


def _infer_page(db, answers):
    """읽힌 문제번호가 가장 많이 속한 db 페이지. 매칭 0이면 None.

    VLM이 말하는 쪽수는 footer를 못 보면 프롬프트 예시("12")를 베껴 환각한다
    (luna·gemini·qwen 공통 실측). 반면 인쇄 문제번호는 안정적으로 읽히고 db가
    문제→페이지를 아니까, 쪽수는 문제번호로 역추론하는 쪽이 견고하다.
    """
    keys = {_qkey(k) for k, v in answers.items()}
    best, best_n = None, 0
    for page in db.valid_pages():
        hit = keys & {_qkey(q.question_no) for q in db.questions_for(page)}
        if len(hit) > best_n:
            best, best_n = page, len(hit)
    return best


def _grade_fullpage(color, db, page_hint, label):
    """페이지 전체를 VLM 1콜로 채점 (OCR·블록절단 없음).

    bbox 크롭 재판독은 시도 후 폐기(2026-08-26): bbox가 이웃 문제 영역을 잘라와
    오염이 페이지 판독보다 컸다. 콜별 흔들림은 FLIP_PAGE_VOTES 다수결로 잡는다.
    """
    if not vlm.available():
        return PageResult(image=label, hold_reason="VLM API 키 없음")
    vlm_page, answers = vlm.read_page(color)
    # VLM은 문제번호를 프롬프트 예시("0046")를 따라 0패딩하기도 한다("21"→"0021").
    # db 키 포맷과 안 맞으면 전부 미매칭되므로, 조회 전에 양쪽 번호를 정규화한다.
    answers = {_qkey(k): v for k, v in answers.items()}
    # 쪽수 우선순위: 수동 힌트 > 문제번호 역추론 > VLM이 말한 쪽수(환각 위험, 최후순위)
    page_no = page_hint or _infer_page(db, answers) or vlm_page
    if not page_no or page_no not in db.valid_pages():
        # 어느 페이지 문제도 못 읽음 → 페이지 전체 보류.
        return PageResult(image=label, page_no=page_no or "",
                          hold_reason="쪽수 인식 실패")
    results = []
    for q in db.questions_for(page_no) or []:
        raw = (answers.get(_qkey(q.question_no)) or "").strip()
        if not raw:  # 표시/필기 없음 → 보류(오답 처리 금지)
            results.append(QuestionResult(q.question_no, HOLD, detail="인식 불확실"))
        elif q.qtype == MULTIPLE_CHOICE:
            picked = sorted({int(n) for n in re.findall(r"[1-9]", raw)})
            results.append(_verdict_mcq(q, picked) if picked
                           else QuestionResult(q.question_no, HOLD, detail="마킹 인식 불확실"))
        else:
            results.append(_verdict_subjective(q, raw))
    return PageResult(image=label, page_no=page_no, results=results)


def _grade_blocks(color, gray, db, page_hint=None, label="", debug=False):
    """OCR로 블록을 절단해 문제별 크롭을 VLM에 판독하는 기존 그레이더."""
    boxes = []
    if ocr.available():
        boxes = ocr.run_ocr(gray)

    h, w = gray.shape[:2]
    page_no, blocks, hold = structure.analyze(boxes, h, w, db, page_hint)
    questions = db.questions_for(page_no) or []

    if debug and blocks:
        out = Path(label or "page").with_suffix("").name
        structure.draw_debug(color, blocks, f"pred_blocks_{out}.png")

    if hold:
        return PageResult(image=label, page_no=page_no,
                          results=[QuestionResult(q.question_no, HOLD) for q in questions],
                          hold_reason=hold)

    def grade_one(q):
        """문제 1개 채점. 예외는 보류로 흡수해 한 문제 실패가 페이지를 죽이지 않게."""
        try:
            block = blocks.get(q.question_no)
            if block is None:
                return QuestionResult(q.question_no, HOLD, detail="블록 없음")
            if q.qtype == MULTIPLE_CHOICE:
                return grade_mcq(color, gray, boxes, block, q)
            return grade_subjective(color, gray, boxes, block, q)
        except Exception as e:
            return QuestionResult(q.question_no, HOLD, detail=f"처리 오류: {e}")

    # 문제들은 서로 독립이고 전부 VLM 네트워크 대기라 동시에 채점한다. 실제 동시
    # API 호출 수는 vlm 모듈의 전역 공유 풀이 캡하므로, 여기선 문제 수만큼 스레드를
    # 열어도 안전하다(이 스레드들은 vlm 풀 슬롯을 기다리기만 하는 오케스트레이터).
    # ex.map은 입력 순서를 보존하므로 결과 순서는 순차 버전과 동일하다.
    if len(questions) <= 1:
        results = [grade_one(q) for q in questions]
    else:
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(questions), thread_name_prefix="q") as ex:
            results = list(ex.map(grade_one, questions))
    return PageResult(image=label, page_no=page_no, results=results)


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

    # 문제번호 정규화: VLM이 0패딩("0021")해도 db 키("21")와 매칭돼야 한다.
    assert _qkey("0021") == "21" and _qkey("21") == "21"
    assert _qkey("0") == "0" and _qkey("7-1") == "7-1"

    # 쪽수 역추론: 읽힌 문제번호가 속한 페이지 채택 (VLM 쪽수 환각 무력화)
    assert _infer_page(db, {"1": "2", "2": "-367"}) == "12"
    assert _infer_page(db, {"0001": "3"}) == "12"   # 0패딩도 매칭
    assert _infer_page(db, {"999": "5"}) is None

    from flip.answers import _selftest as answers_selftest
    answers_selftest()
    from flip.equivalence import _selftest as equivalence_selftest
    equivalence_selftest()
    from flip.vlm import _selftest as vlm_selftest
    vlm_selftest()

    from flip.structure import _selftest as structure_selftest
    structure_selftest()
    from flip.session import _selftest as session_selftest
    session_selftest()

    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", help="페이지 사진 1장")
    ap.add_argument("--batch", help="사진 폴더 (일괄 처리)")
    ap.add_argument("--simulate", help="사진 폴더를 스캔 세션처럼 간격 투입 (비동기 채점 데모)")
    ap.add_argument("--interval", type=float, default=2.0, help="--simulate 투입 간격(초)")
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
    elif args.simulate:
        from flip.session import simulate
        simulate(args.simulate, lambda p: grade_page(p, db, debug=args.debug),
                 interval=args.interval)
    else:
        ap.error("--image, --batch 또는 --simulate 필요")


if __name__ == "__main__":
    main()
