"""채점 결과 모델과 집계/출력.

판정은 세 가지뿐이다: O(정답) / X(오답) / HOLD(보류).
확신이 없으면 X가 아니라 HOLD로 낸다 — 오답 처리보다 사람 확인이 낫다.
"""
from dataclasses import dataclass

O, X, HOLD = "O", "X", "보류"


@dataclass
class QuestionResult:
    question_no: str
    verdict: str            # O | X | HOLD
    student_answer: str = ""  # 인식된 학생 답 (없으면 "")
    detail: str = ""          # 판정 근거 / 보류 사유


@dataclass
class PageResult:
    image: str
    page_no: str = ""         # 읽은 쪽수 ("" = 못 읽음)
    results: list = None      # [QuestionResult]
    hold_reason: str = ""     # 페이지 전체 보류 사유 (있으면 results 무시)

    def counts(self):
        if self.hold_reason:
            return {O: 0, X: 0, HOLD: len(self.results or [])}
        c = {O: 0, X: 0, HOLD: 0}
        for r in self.results or []:
            c[r.verdict] += 1
        return c


def format_page(pr):
    """페이지 결과를 사람이 보는 테이블 문자열로."""
    lines = [f"== {pr.image}  (p.{pr.page_no or '?'})"]
    if pr.hold_reason:
        lines.append(f"  !! 페이지 전체 보류: {pr.hold_reason}")
        return "\n".join(lines)
    for r in pr.results or []:
        ans = f"  [{r.student_answer}]" if r.student_answer else ""
        note = f"  ({r.detail})" if r.detail else ""
        lines.append(f"  {r.question_no:>4}번  {r.verdict}{ans}{note}")
    c = pr.counts()
    lines.append(f"  -- O {c[O]} / X {c[X]} / 보류 {c[HOLD]}")
    return "\n".join(lines)
