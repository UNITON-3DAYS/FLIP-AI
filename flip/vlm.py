"""VLM API 클라이언트 (provider 중립): 손글씨 수학 답 크롭 → 문자열.

환경변수:
  FLIP_VLM_PROVIDER  openai | gemini | anthropic (기본 openai)
  FLIP_VLM_API_KEY   API 키. 없으면 available()이 False, read_math()는 None.
  FLIP_VLM_MODEL     모델명 (없으면 provider별 기본값)

방어:
- 같은 크롭을 2회 인식해 결과가 다르거나 UNSURE면 None → 호출부가 보류.
- 키 없음/네트워크 오류/타임아웃도 전부 None. 절대 크래시하지 않는다.
- SDK 없이 requests로 provider별 REST를 직접 호출한다.
"""
import base64
import os

import cv2
import requests

TIMEOUT = 20      # 초 (reasoning 모델은 사고 시간이 있어 여유 있게)
MAX_TOKENS = 100  # 답은 한 줄이라 짧게 (reasoning 모델은 아래에서 여유를 더 준다)
REASONING_MAX_TOKENS = 2000  # reasoning 모델은 사고 토큰이 한도를 먼저 소진하므로 크게

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

# 읽기 거부 응답 (둘 다 None 처리 → 호출부 보류). PRINTED는 마스킹을 빠져나온
# 인쇄 수식이 답 후보로 잘못 올라온 경우의 마지막 방어선이다.
REFUSALS = {"UNSURE", "PRINTED"}

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "anthropic": "claude-opus-5",
}


def _provider():
    return os.environ.get("FLIP_VLM_PROVIDER", "openai").strip().lower()


def _model():
    return os.environ.get("FLIP_VLM_MODEL") or DEFAULT_MODELS.get(_provider(), "")


def available():
    """VLM 사용 가능 여부 (API 키 존재)."""
    return bool(os.environ.get("FLIP_VLM_API_KEY"))


def _encode_jpeg_b64(img):
    """크롭(그레이 또는 BGR) → base64 JPEG 문자열. 실패 시 None."""
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ── provider별 REST 호출 ─────────────────────────────────────────────────

def _call_openai(b64, key, model):
    messages = [{"role": "user", "content": [
        {"type": "text", "text": PROMPT},
        {"type": "image_url",
         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
    ]}]
    # GPT-5 세대는 max_tokens를 거부하고 max_completion_tokens를 요구한다.
    # 크롭 읽기는 단순 작업이라 reasoning effort는 낮출수록 빠르다
    # (FLIP_VLM_REASONING 미설정 시 파라미터 자체를 안 보낸다 — 구모델 호환).
    payload = {"model": model, "max_completion_tokens": REASONING_MAX_TOKENS,
               "messages": messages}
    effort = os.environ.get("FLIP_VLM_REASONING")
    if effort:
        payload["reasoning_effort"] = effort
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json=payload, timeout=TIMEOUT)
    if r.status_code == 400:
        # 구세대 모델(gpt-4o 등)이 max_completion_tokens를 모르면 구파라미터로 재시도
        legacy = {"model": model, "max_tokens": MAX_TOKENS, "messages": messages}
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json=legacy, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _call_gemini(b64, key, model):
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        params={"key": key},
        json={"contents": [{"parts": [
            {"text": PROMPT},
            {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
        ]}]},
        timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"]


def _call_anthropic(b64, key, model):
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": PROMPT},
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


def _call(b64):
    """1회 호출 → 정리된 응답 문자열. 어떤 실패든 None."""
    key = os.environ.get("FLIP_VLM_API_KEY")
    call = _CALLS.get(_provider())
    if not key or call is None:
        return None
    try:
        text = call(b64, key, _model())
    except Exception:
        return None  # 네트워크 오류/타임아웃/응답 형식 불일치 전부 보류로
    if not text:
        return None
    return text.strip().strip("`").strip()


def read_math(crop):
    """크롭 이미지 → 인식된 답 문자열. 불확실/실패는 None (호출부가 보류).

    같은 크롭을 2회 호출해 결과가 일치할 때만 신뢰한다.
    """
    if not available():
        return None  # 네트워크 호출 없이 종료
    b64 = _encode_jpeg_b64(crop)
    if b64 is None:
        return None
    first = _call(b64)
    if first is None or first.upper() in REFUSALS:
        return None
    second = _call(b64)
    if second is None or second.upper() in REFUSALS:
        return None
    if first.replace(" ", "") != second.replace(" ", ""):
        return None  # 2회 불일치 → 인식 불확실
    return first


def _selftest():
    import numpy as np

    # 키 관련 환경변수를 잠시 비워서, 네트워크 없이 방어 동작을 검증
    saved = {k: os.environ.pop(k, None)
             for k in ("FLIP_VLM_API_KEY", "FLIP_VLM_PROVIDER", "FLIP_VLM_MODEL")}
    try:
        assert available() is False
        crop = np.full((30, 60), 255, np.uint8)
        assert read_math(crop) is None  # 키 없음 → 호출 없이 None

        os.environ["FLIP_VLM_API_KEY"] = "test-key"
        assert available() is True
        # provider 오설정이어도 크래시 없이 None
        os.environ["FLIP_VLM_PROVIDER"] = "unknown"
        assert _call("") is None
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
