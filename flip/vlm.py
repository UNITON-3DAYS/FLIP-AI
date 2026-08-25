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
PAGE_PROMPT = (
    "수학 문제집 한 페이지 사진이다. 페이지 번호와, 각 문제에서 학생이 손으로 표시/필기한 답을 읽어라.\n"
    "- 문제번호는 인쇄된 번호 그대로(예: 0046).\n"
    "- 객관식: 학생이 펜으로 동그라미로 감싸거나 번호 위에 겹쳐 표시한 선택지 번호. 획이 스쳐 지나간 "
    "것·인쇄된 원문자는 제외. 여러 개면 쉼표.\n"
    "- 주관식: 손글씨 값을 선형표기로(분수 a/b, 제곱근 sqrt(x), 거듭제곱 x^2).\n"
    "- 표시가 전혀 없는 문제는 빈 문자열.\n"
    'JSON만 출력: {"page":"12","answers":{"0046":"2","0047":"4"}}'
)

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


def _call_openai(b64, key, model, prompt, max_out=REASONING_MAX_TOKENS):
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
    r = requests.post(f"{base}/responses",
                      headers=headers, json=payload, timeout=TIMEOUT)
    if r.status_code < 400:
        return _extract_responses_text(r.json())

    # 폴백: Responses를 모르는 구모델/구계정 → Chat Completions (구파라미터)
    legacy = {"model": model, "max_tokens": max(MAX_TOKENS, max_out),
              "messages": [{"role": "user", "content": [
                  {"type": "text", "text": prompt},
                  {"type": "image_url",
                   "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": DETAIL}},
              ]}]}
    r = requests.post(f"{base}/chat/completions",
                      headers=headers, json=legacy, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _call_gemini(b64, key, model, prompt, max_out=REASONING_MAX_TOKENS):
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
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        params={"key": key},
        json={"contents": [{"parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
        ]}],
            "generationConfig": gen},
        timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def _call_anthropic(b64, key, model, prompt, max_out=REASONING_MAX_TOKENS):
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


def _call(b64, prompt=PROMPT, max_out=REASONING_MAX_TOKENS):
    """1회 호출 → 정리된 응답 문자열. 어떤 실패든 None (실패 사유는 로그로 남긴다)."""
    key = os.environ.get("FLIP_VLM_API_KEY")
    provider, model = _provider(), _model()
    call = _CALLS.get(provider)
    if not key or call is None:
        log.warning("VLM 호출 스킵(보류): %s",
                    "API 키 없음" if not key else f"미지원 provider={provider!r}")
        return None
    t0 = time.monotonic()
    try:
        text = call(b64, key, model, prompt, max_out)
    except Exception as e:  # 네트워크 오류/타임아웃/응답 형식 불일치 전부 보류로
        log.warning("VLM %s/%s 호출 실패(보류, %.1fs): %s", provider, model,
                    time.monotonic() - t0, e)
        return None
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


def read_page(page_img):
    """전체 페이지 이미지 → (쪽수 str|None, {문제번호: 학생답 str}). 실패는 (None, {}).

    OCR·블록절단 없이 페이지 전체를 한 번에 읽는 fullpage 그레이더용. 크래시 금지
    계약은 read_math/read_mcq와 동일 — 키 없음/타임아웃/JSON 깨짐은 전부 (None, {}).
    """
    if not available():
        return None, {}
    b64 = _encode_jpeg_b64(page_img)
    if b64 is None:
        return None, {}
    # 페이지 전체라 출력이 read_math보다 길다(문제 수만큼). reasoning 토큰이 한도를
    # 먼저 먹으므로 여유를 더 준다.
    text = _executor().submit(_call, b64, PAGE_PROMPT, 4000).result()
    if not text:
        return None, {}
    try:
        data = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
    except (AttributeError, ValueError):
        log.warning("read_page: 응답에서 JSON 파싱 실패(페이지 보류)")
        return None, {}  # JSON 없음/깨짐 → 페이지 보류
    page = data.get("page")
    answers = {str(k): str(v) for k, v in (data.get("answers") or {}).items()}
    log.info("read_page: 쪽수=%s, 답 %d개 읽음", page or "?", len(answers))
    return (str(page) if page else None), answers


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
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
