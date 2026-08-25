"""SymPy 동치 비교: 학생 답 문자열 vs DB 정답.

"1/2" ≡ "0.5", "x=-2,3" ≡ ["-2","3"] (순서 무관) 같은 수학적 동치를 판정한다.
반환은 3값: True(동치) / False(불일치) / None(파싱 실패 → 호출부가 보류).
X 처리보다 사람 확인이 낫다는 원칙(flip/results.py)에 따라, 못 읽으면 None이다.
"""
import re

import sympy

# "x=", "y=" 같은 변수 접두 (해집합 표기). 공백 제거 후 원소별로 적용.
VAR_PREFIX_RE = re.compile(r"^[a-zA-Z]\w*=")

# 손글씨 인식 결과에 섞이는 수학 기호 → sympy 선형 표기. 순서대로 치환.
REPLACEMENTS = [
    ("×", "*"),
    ("÷", "/"),
    ("−", "-"),   # 유니코드 마이너스
    ("π", "pi"),
    ("^", "**"),
    (" ", ""),
]


def _parse_set(value):
    """str 또는 [str] → [sympy 식] 해집합. 원소 하나라도 파싱 실패면 None."""
    raw = list(value) if isinstance(value, (list, tuple)) else [value]
    parts = []
    for item in raw:
        s = str(item)
        # 쉼표 없이 공백으로만 나열된 해집합("a=90 b=40") → 쉼표 구분으로 정규화.
        # 아래에서 공백이 전부 제거되면 "a=90b=40"이 되어 파싱이 죽는다(VLM 실측 출력).
        s = re.sub(r"\s+(?=[a-zA-Z]\w*=)", ",", s)
        for old, new in REPLACEMENTS:
            s = s.replace(old, new)
        # 쉼표로 해 분리 후 원소별 "변수=" 접두 제거 ("x=-2,3" → ["-2", "3"])
        for p in s.split(","):
            p = VAR_PREFIX_RE.sub("", p)
            if p:
                parts.append(p)
    if not parts:
        return None
    exprs = []
    for p in parts:
        try:
            exprs.append(sympy.sympify(p))
        except Exception:
            return None
    return exprs


def equivalent(student_str, answer):
    """학생 답 문자열과 DB 정답(str 또는 [str])의 동치 여부.

    True/False/None(파싱 실패). 해집합끼리 크기가 같고 원소가 순서 무관으로
    simplify(a-b)==0 매칭되면 동치.
    """
    got = _parse_set(student_str)
    want = _parse_set(answer)
    if got is None or want is None:
        return None
    if len(got) != len(want):
        return False
    remaining = list(want)
    for g in got:
        for i, w in enumerate(remaining):
            try:
                if sympy.simplify(g - w) == 0:
                    del remaining[i]
                    break
            except Exception:
                continue
        else:
            return False
    return True


def _selftest():
    # 해 여러 개, 순서 무관, 변수 접두
    assert equivalent("x=-2,3", ["-2", "3"]) is True
    assert equivalent("x=3,-2", ["-2", "3"]) is True
    # 분수/소수 동치
    assert equivalent("1/2", "0.5") is True
    assert equivalent("0.5", "3") is False
    # 기호식 동치
    assert equivalent("sqrt(4)", "2") is True
    assert equivalent("2^3", "8") is True
    # 해 개수 불일치는 오답
    assert equivalent("3", ["-2", "3"]) is False
    # 파싱 실패 → None (보류)
    assert equivalent("@#$!", "3") is None
