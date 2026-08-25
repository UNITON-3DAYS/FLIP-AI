"""VLM API 클라이언트 (provider 중립): 손글씨 수학 답 크롭 → 문자열.

환경변수:
  FLIP_VLM_PROVIDER  openai | gemini | anthropic (기본 gemini)
  FLIP_VLM_API_KEY   API 키. 없으면 available()이 False, read_math()는 None.
  FLIP_VLM_MODEL     모델명 (없으면 provider별 기본값)

방어:
- 같은 크롭을 2회 인식해 결과가 다르거나 UNSURE면 None → 호출부가 보류.
- 키 없음/네트워크 오류/타임아웃도 전부 None. 절대 크래시하지 않는다.
- SDK 없이 requests로 provider별 REST를 직접 호출한다.

동시성:
- 모든 API 왕복은 이 모듈의 전역 공유 풀(_executor)을 거친다. 동시 호출 수를
  FLIP_VLM_CONCURRENCY 하나로 묶어 rate limit을 넘지 않게 한다. 페이지 병렬
  (session.py)과 문제 병렬(flip/grade.py)은 이 풀의 슬롯을 두고 경쟁할 뿐이다.
- 풀에는 leaf HTTP 호출(_call)만 제출한다. read_math/read_mcq 자체를 풀에
  제출하면 안 된다 — 풀 안에서 다시 풀을 기다려 교착이 날 수 있다.
"""
import base64
import collections
import concurrent.futures
import json
import logging
import os
import re
import threading
import time

import cv2
import requests

log = logging.getLogger("flip.vlm")

TIMEOUT = 40      # 초. 페이지 콜이 15~20초대 꼬리를 밟는 실측(20s에선 부분 실패)
MAX_TOKENS = 100  # 답은 한 줄이라 짧게 (reasoning 모델은 아래에서 여유를 더 준다)
REASONING_MAX_TOKENS = 2000  # reasoning 모델은 사고 토큰이 한도를 먼저 소진하므로 크게
# OpenAI 이미지 해상도. low는 손글씨·동그라미를 놓쳐 오답을 낸다(측정: 85를 14로 오독).
# 크롭이 작아 high여도 문제당 ~750토큰($0.00015)로 저렴 — 정확도를 산다.
DETAIL = "high"

# LaTeX가 아니라 선형 표기를 강제하는 이유: sympy parse_latex는 antlr 의존성이
# 필요해서 피한다. flip/equivalence.py가 이 표기를 그대로 파싱한다.
PROMPT = (
    "손글씨 수학 답 이미지다. 답을 한 줄 선형 표기로만 출력해라: "
    "분수는 a/b, 제곱근은 sqrt(x), 거듭제곱은 x^2, "
    "해 여러 개는 쉼표 구분 (예: x=-2,3). 다른 말 금지. "
    "읽을 수 없거나 불확실하면 UNSURE만 출력. "
    "손글씨가 없고 인쇄된 활자만 보이면 PRINTED만 출력. "
    "인쇄와 손글씨가 섞여 있으면 손글씨 부분만 읽어라."
)

# 객관식: 문제 블록을 통째로 넘겨 학생이 친 마킹 번호만 받는다 (인쇄 마커 CV 대체).
MCQ_PROMPT = (
    "객관식 문제 이미지다. 학생이 손으로 그 번호를 동그라미로 감싸거나 번호 위에 직접 "
    "겹쳐 표시한 선택지만 출력해라. 획이 근처를 스쳐 지나가기만 한 번호는 제외한다. "
    "여러 개면 쉼표로 구분 (예: 2,4). 표시가 없으면 NONE. 숫자만, 다른 말 금지."
)

# 전체페이지 1콜 채점(FLIP_GRADER=fullpage)용. OCR·블록절단 없이 페이지 전체를 넘겨
# 쪽수와 문제별 학생 답을 한 번에 읽는다. 실측: 블록 방식과 동률(7/7)에 토큰 ~1/5.
# 역할·규칙은 시스템 프롬프트로 고정(암시적 캐싱 접두사 겸), 유저 턴은 이미지+한 줄.
PAGE_SYSTEM = (
    "[역할] 너는 수학 문제집 페이지 사진에서 학생의 손 마킹·필기만 판독하는 채점 판독기다. "
    "문제를 풀지 않는다 — 학생이 무엇을 표시했는지만 읽는다.\n"
    "[지침]\n"
    "1. 페이지에 보이는 모든 문제번호를 인쇄된 표기 그대로(예: 0046) answers의 키로 빠짐없이 넣는다. "
    "문제 누락 금지.\n"
    "2. 객관식 마킹은 두 방식이 있다:\n"
    "   (a) 인쇄된 선택지 번호에 동그라미를 감싸거나 체크(✓)를 얹음 — 체크는 V의 꼭짓점이 "
    "놓인 번호의 마킹이고, 긴 꼬리가 이웃 번호를 스쳐 지나간 것은 그 번호의 마킹이 아니다.\n"
    "   번호를 감싸거나 얹은 마킹만 인정한다 — 획이 스쳐 지난 것뿐이면 그 문제는 마킹 없음으로 본다.\n"
    "   (b) 문제 여백에 답 번호를 손글씨로 적음(동그라미로 감싸기도 함) — 그 손글씨 숫자를 "
    "그대로 읽는다.\n"
    "   복수 마킹이면 전부 쉼표로 나열한다(예: \"2,4\").\n"
    "3. 주관식: 손글씨 답을 선형 표기로 적는다 — 분수 a/b, 제곱근 sqrt(x), 거듭제곱 x^2, "
    "복수 값은 쉼표. 인쇄된 활자는 답이 아니다.\n"
    "4. 손 마킹이 보이면 흐릿해도 반드시 기록한다. 마킹·필기가 전혀 없는 문제만 빈 문자열 \"\".\n"
    "5. page는 페이지 모서리에 인쇄된 페이지 번호. 보이지 않으면 빈 문자열로 두고 절대 추측하지 않는다.\n"
    "[출력] 다른 말 없이 JSON 하나만:\n"
    '{"page":"12","answers":{"0046":"2","0047":"4"}}'
)
PAGE_PROMPT = "이 페이지에서 학생이 표시/필기한 답을 전부 판독해라."

# 읽기 거부 응답 (둘 다 None 처리 → 호출부 보류). PRINTED는 마스킹을 빠져나온
# 인쇄 수식이 답 후보로 잘못 올라온 경우의 마지막 방어선이다.
REFUSALS = {"UNSURE", "PRINTED"}

DEFAULT_MODELS = {
    "openai": "gpt-5.6-luna",
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-opus-5",
}


def _provider():
    # 기본 gemini — 실사진 A/B에서 gemini-2.5-flash가 박스 손글씨 판독 최상(2026-08-26)
    return os.environ.get("FLIP_VLM_PROVIDER", "gemini").strip().lower()


def _model():
    return os.environ.get("FLIP_VLM_MODEL") or DEFAULT_MODELS.get(_provider(), "")


def available():
    """VLM 사용 가능 여부 (API 키 존재)."""
    return bool(os.environ.get("FLIP_VLM_API_KEY"))


# ── 전역 공유 풀: 동시 API 호출 수의 유일한 상한 ──────────────────────────
DEFAULT_CONCURRENCY = 8  # 동시 API 왕복 상한 (rate limit 여유 있게, 조정 가능)
_EXECUTOR = None
_EXECUTOR_LOCK = threading.Lock()


def _concurrency():
    try:
        return max(1, int(os.environ.get("FLIP_VLM_CONCURRENCY", DEFAULT_CONCURRENCY)))
    except ValueError:
        return DEFAULT_CONCURRENCY


def _executor():
    """프로세스 전역 단일 ThreadPoolExecutor (지연 생성). 모든 _call이 이걸 거친다."""
    global _EXECUTOR
    if _EXECUTOR is None:
        with _EXECUTOR_LOCK:
            if _EXECUTOR is None:
                _EXECUTOR = concurrent.futures.ThreadPoolExecutor(
                    max_workers=_concurrency(), thread_name_prefix="vlm")
    return _EXECUTOR


def _encode_jpeg_b64(img):
    """크롭(그레이 또는 BGR) → base64 JPEG 문자열. 실패 시 None."""
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ── provider별 REST 호출 ─────────────────────────────────────────────────

def _extract_responses_text(data):
    """Responses API 응답 JSON에서 output_text를 꺼낸다. 없으면 None."""
    for item in data.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    return c.get("text")
    return None


def _call_openai(b64, key, model, prompt, max_out=REASONING_MAX_TOKENS, system=None):
    """OpenAI Responses API 주경로 (GPT-5 세대 권장 방식).

    reasoning effort 기본 low — 크롭/페이지 읽기는 단순 작업이라 low가 빠르고 싸다.
    미설정 시 OpenAI 기본(medium)이 적용돼 서버 콜이 12~13초까지 늘었던 실측이 근거.
    reasoning 미지원 모델이면 FLIP_VLM_REASONING을 빈 값으로 두면 파라미터를 안 보낸다.
    Responses 미지원 구모델이면 Chat Completions로 폴백.

    FLIP_VLM_BASE_URL로 OpenAI 호환 엔드포인트(OpenRouter/Together 등)를 갈아끼울 수
    있다 — Qwen VL 등 타사 모델도 이 경로 그대로 쓴다(Responses 미지원이면 폴백이 흡수).
    """
    base = os.environ.get("FLIP_VLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    headers = {"Authorization": f"Bearer {key}"}
    payload = {
        "model": model,
        "max_output_tokens": max_out,  # 사고 토큰이 한도를 먼저 먹는다
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image",
             "image_url": f"data:image/jpeg;base64,{b64}", "detail": DETAIL},
        ]}],
    }
    effort = os.environ.get("FLIP_VLM_REASONING", "low")
    if effort:
        payload["reasoning"] = {"effort": effort}
    if system:
        payload["instructions"] = system
    r = requests.post(f"{base}/responses",
                      headers=headers, json=payload, timeout=TIMEOUT)
    if r.status_code < 400:
        return _extract_responses_text(r.json())

    # 폴백: Responses를 모르는 구모델/구계정 → Chat Completions (구파라미터)
    legacy = {"model": model, "max_tokens": max(MAX_TOKENS, max_out),
              "messages": ([{"role": "system", "content": system}] if system else []) +
                          [{"role": "user", "content": [
                  {"type": "text", "text": prompt},
                  {"type": "image_url",
                   "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": DETAIL}},
              ]}]}
    r = requests.post(f"{base}/chat/completions",
                      headers=headers, json=legacy, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _call_gemini(b64, key, model, prompt, max_out=REASONING_MAX_TOKENS, system=None):
    # maxOutputTokens에 thinking 토큰도 포함되므로(2.5 계열) max_out 여유를 그대로 쓴다.
    gen = {"maxOutputTokens": max_out}
    # 사고 제한 (기본 0 = 끔): 6페이지 A/B에서 사고를 꺼도 판독 정확도 동일, 속도는
    # 평균 14.3s→6.6s(편차 최대 22.6s→7.9s), 사고토큰 비용도 0. 숫자면 thinkingBudget
    # (2.5 계열), 단어(low|high)면 thinkingLevel(3.x 계열). 빈 값이면 파라미터 생략
    # (thinkingConfig 미지원 모델용 — 지원 안 하는 모델에 보내면 400 → 보류가 나므로).
    thinking = os.environ.get("FLIP_VLM_THINKING", "0").strip()
    if thinking:
        gen["thinkingConfig"] = ({"thinkingBudget": int(thinking)}
                                 if thinking.lstrip("-").isdigit()
                                 else {"thinkingLevel": thinking})
    body = {"contents": [{"parts": [
        {"text": prompt},
        {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
    ]}],
        "generationConfig": gen}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        params={"key": key}, json=body, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    u = data.get("usageMetadata") or {}
    # 비용 실측용: 입력/출력/사고/캐시히트 토큰. cached는 암시적 캐싱(입력 75% 할인) 확인용.
    log.info("VLM usage in=%s out=%s think=%s cached=%s",
             u.get("promptTokenCount"), u.get("candidatesTokenCount"),
             u.get("thoughtsTokenCount", 0), u.get("cachedContentTokenCount", 0))
    return data["candidates"][0]["content"]["parts"][0]["text"]


def _call_anthropic(b64, key, model, prompt, max_out=REASONING_MAX_TOKENS, system=None):
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": max(MAX_TOKENS, max_out),
            **({"system": system} if system else {}),
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": prompt},
            ]}],
        },
        timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if data.get("stop_reason") == "refusal":  # 안전 분류기 거절 → 인식 실패 취급
        return None
    for block in data.get("content", []):
        if block.get("type") == "text":
            return block["text"]
    return None


_CALLS = {"openai": _call_openai, "gemini": _call_gemini, "anthropic": _call_anthropic}


def _call(b64, prompt=PROMPT, max_out=REASONING_MAX_TOKENS, system=None):
    """1회 호출 → 정리된 응답 문자열. 어떤 실패든 None (실패 사유는 로그로 남긴다)."""
    key = os.environ.get("FLIP_VLM_API_KEY")
    provider, model = _provider(), _model()
    call = _CALLS.get(provider)
    if not key or call is None:
        log.warning("VLM 호출 스킵(보류): %s",
                    "API 키 없음" if not key else f"미지원 provider={provider!r}")
        return None
    t0 = time.monotonic()
    text = None
    for attempt in (1, 2):  # 일시 장애(5xx/타임아웃)는 1회 재시도 — 페이지 통보류 방지
        try:
            text = call(b64, key, model, prompt, max_out, system)
            break
        except Exception as e:
            log.warning("VLM %s/%s 호출 실패(%d차, %.1fs): %s", provider, model,
                        attempt, time.monotonic() - t0, e)
            if attempt == 2:
                return None
            time.sleep(1.5)
    dt = time.monotonic() - t0
    if not text:
        log.warning("VLM %s/%s 응답 비어있음(보류, %.1fs)", provider, model, dt)
        return None
    text = text.strip().strip("`").strip()
    log.info("VLM %s/%s ok %.1fs → %r", provider, model, dt, text)
    return text


def read_math(crop):
    """크롭 이미지 → 인식된 답 문자열. 불확실/실패는 None (호출부가 보류).

    같은 크롭을 2회 호출해 결과가 일치할 때만 신뢰한다. 두 호출은 서로 독립이라
    전역 풀에 동시에 던져 왕복 시간을 반으로 줄인다 (실제 동시 호출 수는 풀이 캡).
    """
    if not available():
        return None  # 네트워크 호출 없이 종료
    b64 = _encode_jpeg_b64(crop)
    if b64 is None:
        return None
    pool = _executor()
    f1 = pool.submit(_call, b64)
    f2 = pool.submit(_call, b64)
    first, second = f1.result(), f2.result()
    if first is None or first.upper() in REFUSALS:
        return None
    if second is None or second.upper() in REFUSALS:
        return None
    if first.replace(" ", "") != second.replace(" ", ""):
        log.info("read_math 2회 불일치(보류): %r vs %r", first, second)
        return None  # 2회 불일치 → 인식 불확실
    return first


def read_mcq(block_crop):
    """객관식 문제 블록 크롭 → 학생이 친 선택지 번호 [정렬된 int]. 불확실/무마킹은 None.

    블록을 통째로 넘겨 VLM이 동그라미 친 번호를 읽는다 (구 mcq.py CV 대체).
    출력이 수 토큰·reasoning 0이라 read_math와 달리 1회면 충분 — detail=high가
    동그라미 판독의 핵심(저해상도는 마킹을 놓쳐 오답을 냄, 측정으로 확인).
    """
    if not available():
        return None
    b64 = _encode_jpeg_b64(block_crop)
    if b64 is None:
        return None
    text = _executor().submit(_call, b64, MCQ_PROMPT).result()  # 풀 경유로 전역 캡 반영
    if not text or text.upper() in REFUSALS or "NONE" in text.upper():
        return None  # 인식 실패/무마킹 → 호출부가 보류
    nums = sorted({int(n) for n in re.findall(r"[1-9]", text)})
    return nums or None


def _parse_page(text):
    """read_page 1콜 응답 → (쪽수, answers) 또는 None(파싱 실패)."""
    if not text:
        return None
    try:
        data = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
    except (AttributeError, ValueError):
        log.warning("read_page: 응답에서 JSON 파싱 실패")
        return None
    page = data.get("page")
    answers = {str(k): str(v).strip() for k, v in (data.get("answers") or {}).items()}
    return (str(page) if page else None), answers


def _merge_page_votes(reads):
    """N회 판독 다수결 병합. 문제별로 과반 값을 채택, 과반 없으면 ""(→보류).

    같은 페이지를 여러 번 읽으면 콜마다 마킹 판독이 흔들린다(실측: 사고0·동적 공통).
    무작위 오독은 콜 간 상관이 없어 다수결로 소거된다. 체계적 오독(체크 꼬리 겹침)은
    다수결로 못 잡는다 — 그건 픽셀의 한계.
    """
    norm = lambda k: re.sub(r"^0+(\d)", r"\1", str(k))
    keys = {norm(k) for _, answers in reads for k in answers}
    merged = {}
    for k in keys:
        # 키를 아예 안 낸 콜은 기권(불참) — 빈 문자열 투표로 세지 않는다. 모델이
        # 문제를 누락하는 것과 "마킹 없음"이라고 말하는 것은 다르다(누락을 ""로
        # 세면 과반 ""가 실제 판독을 덮어 보류가 샌다, 실측).
        votes = collections.Counter(
            v.strip() for _, answers in reads
            for kk, v in answers.items() if norm(kk) == k)
        val, n = votes.most_common(1)[0]
        if n > sum(votes.values()) // 2:
            merged[k] = val  # 참여 콜 중 과반 일치
        else:
            # 과반 없음 → 비어있지 않은 값 중 최빈값. 전부 보류시키면 의도 오답까지
            # 보류로 새서 손해 — 어떤 판독이든 있으면 db 대조가 O/X를 가른다.
            nonempty = collections.Counter(v for v in votes.elements() if v)
            merged[k] = nonempty.most_common(1)[0][0] if nonempty else ""
    pages = collections.Counter(p for p, _ in reads if p)
    page = pages.most_common(1)[0][0] if pages else None
    return page, merged


def read_page(page_img):
    """전체 페이지 이미지 → (쪽수 str|None, {문제번호: 학생답 str}). 실패는 (None, {}).

    OCR·블록절단 없이 페이지 전체를 한 번에 읽는 fullpage 그레이더용. 크래시 금지
    계약은 read_math/read_mcq와 동일 — 키 없음/타임아웃/JSON 깨짐은 전부 (None, {}).
    FLIP_PAGE_VOTES=N(기본 1)이면 같은 페이지를 N콜 병렬 판독해 문제별 다수결.
    """
    if not available():
        return None, {}
    b64 = _encode_jpeg_b64(page_img)
    if b64 is None:
        return None, {}
    try:
        n_votes = max(1, int(os.environ.get("FLIP_PAGE_VOTES", "1") or 1))
    except ValueError:
        n_votes = 1
    # 페이지 전체라 출력이 read_math보다 길다(문제 수만큼). gemini는 사고 토큰도
    # 이 한도를 나눠 쓰므로 넉넉히(동적 사고가 수천 토큰을 먼저 먹은 잘림 실측).
    futs = [_executor().submit(_call, b64, PAGE_PROMPT, 12000, PAGE_SYSTEM)
            for _ in range(n_votes)]
    reads = [r for r in (_parse_page(f.result()) for f in futs) if r is not None]
    if not reads:
        return None, {}
    page, answers = reads[0] if len(reads) == 1 else _merge_page_votes(reads)
    log.info("read_page: 쪽수=%s, 답 %d개 읽음 (%d/%d콜)",
             page or "?", len(answers), len(reads), n_votes)
    return page, answers


def _selftest():
    import numpy as np

    # 키 관련 환경변수를 잠시 비워서, 네트워크 없이 방어 동작을 검증
    saved = {k: os.environ.pop(k, None)
             for k in ("FLIP_VLM_API_KEY", "FLIP_VLM_PROVIDER", "FLIP_VLM_MODEL")}
    try:
        assert available() is False
        crop = np.full((30, 60), 255, np.uint8)
        assert read_math(crop) is None  # 키 없음 → 호출 없이 None
        assert read_mcq(crop) is None   # 객관식도 키 없으면 호출 없이 None

        os.environ["FLIP_VLM_API_KEY"] = "test-key"
        assert available() is True
        # provider 오설정이어도 크래시 없이 None
        os.environ["FLIP_VLM_PROVIDER"] = "unknown"
        assert _call("") is None
        assert read_mcq(crop) is None   # 응답 None → 파싱 안 하고 보류

        # 다수결 병합: 과반 채택, 과반 없으면 비어있지 않은 최빈값, 키 0패딩 무시,
        # 쪽수도 다수결. 전원 빈값이면 ""(보류).
        p, a = _merge_page_votes([("12", {"0046": "2", "47": "3", "48": ""}),
                                  ("12", {"0046": "2", "0047": "5", "48": ""}),
                                  ("13", {"0046": "4", "0047": "5", "48": ""})])
        assert p == "12" and a["46"] == "2" and a["47"] == "5" and a["48"] == ""
        p2, a2 = _merge_page_votes([("9", {"1": "3"}), ("9", {"1": "4"}), ("9", {"1": ""})])
        assert a2["1"] in ("3", "4")  # 과반 없음 → 비어있지 않은 값 채택(보류 아님)
        # 키 누락 콜은 기권: 1콜만 "5"를 냈고 나머지가 그 문제를 빠뜨렸으면 "5" 채택
        _, a3 = _merge_page_votes([("9", {"2": "5"}), ("9", {}), ("9", {})])
        assert a3["2"] == "5"
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
