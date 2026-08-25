"""AKR-9 spike: VLM API가 손글씨 수학 답 크롭을 읽는지 검증하고 API를 고른다.

사용법:
  1) 샘플 사진에서 손글씨 답 부분만 크롭해 폴더에 모은다 (5~10장).
     파일명을 정답으로 쓰면 채점까지 된다: "x=-2,3.jpg" -> 기대값 x=-2,3
     (파일명에 못 쓰는 문자가 있으면 expected.json {"파일명": "기대값"} 으로 대체)
  2) 환경변수 설정:
     FLIP_VLM_PROVIDER=openai|gemini|anthropic
     FLIP_VLM_API_KEY=...
     FLIP_VLM_MODEL=(생략 시 provider 기본값)
  3) python spike_vlm.py --crops <폴더> [--repeat 2]

출력: 크롭별 인식 결과/일치 여부/지연시간 표 + 요약.
일회성 spike 스크립트 — 파이프라인 코드와 독립.
"""
import argparse
import base64
import json
import os
import statistics
import time
from pathlib import Path

import requests

PROMPT = (
    "이 이미지는 학생이 손으로 쓴 수학 답이다. 답을 한 줄 선형 표기로만 출력해라: "
    "분수는 a/b, 제곱근은 sqrt(x), 거듭제곱은 x^2, 해가 여러 개면 쉼표로 구분 (예: x=-2,3). "
    "설명 등 다른 말은 절대 쓰지 마라. 읽을 수 없거나 불확실하면 UNSURE 라고만 출력해라."
)

DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "gemini": "gemini-2.0-flash",
    "anthropic": "claude-sonnet-5",
}


def encode_image(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


def call_openai(key, model, b64):
    # Responses API 주경로 (GPT-5 세대 권장). 실패 시 Chat Completions 폴백.
    headers = {"Authorization": f"Bearer {key}"}
    payload = {
        "model": model,
        "max_output_tokens": 2000,  # reasoning 모델은 사고 토큰이 한도를 먼저 먹는다
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": PROMPT},
            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"},
        ]}],
    }
    effort = os.environ.get("FLIP_VLM_REASONING")
    if effort:
        payload["reasoning"] = {"effort": effort}
    r = requests.post("https://api.openai.com/v1/responses",
                      headers=headers, json=payload, timeout=30)
    if r.status_code < 400:
        for item in r.json().get("output", []):
            if item.get("type") == "message":
                for c in item.get("content", []):
                    if c.get("type") == "output_text":
                        return c["text"].strip()
        return "ERROR: 응답에서 텍스트 추출 실패"
    r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers,
                      json={"model": model, "max_tokens": 100, "messages": [{
                          "role": "user", "content": [
                              {"type": "text", "text": PROMPT},
                              {"type": "image_url",
                               "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                          ]}]},
                      timeout=30)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def call_gemini(key, model, b64):
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": key},
        json={"contents": [{"parts": [
            {"text": PROMPT},
            {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
        ]}]},
        timeout=30)
    r.raise_for_status()
    return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


def call_anthropic(key, model, b64):
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
        json={"model": model, "max_tokens": 100, "messages": [{
            "role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": PROMPT},
            ]}]},
        timeout=30)
    r.raise_for_status()
    return r.json()["content"][0]["text"].strip()


CALLERS = {"openai": call_openai, "gemini": call_gemini, "anthropic": call_anthropic}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crops", required=True, help="손글씨 답 크롭 폴더")
    ap.add_argument("--repeat", type=int, default=2, help="크롭당 반복 질의 수 (일관성 확인)")
    args = ap.parse_args()

    provider = os.environ.get("FLIP_VLM_PROVIDER", "")
    key = os.environ.get("FLIP_VLM_API_KEY", "")
    if provider not in CALLERS or not key:
        raise SystemExit("FLIP_VLM_PROVIDER (openai|gemini|anthropic) 와 FLIP_VLM_API_KEY 를 설정해라")
    model = os.environ.get("FLIP_VLM_MODEL", DEFAULT_MODELS[provider])
    call = CALLERS[provider]

    folder = Path(args.crops)
    expected_map = {}
    exp_file = folder / "expected.json"
    if exp_file.exists():
        expected_map = json.loads(exp_file.read_text(encoding="utf-8"))

    crops = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not crops:
        raise SystemExit(f"크롭 이미지 없음: {folder}")

    print(f"provider={provider} model={model} crops={len(crops)} repeat={args.repeat}\n")
    latencies, agree_cnt, correct_cnt, scored_cnt = [], 0, 0, 0
    for p in crops:
        b64 = encode_image(p)
        answers, times = [], []
        for _ in range(args.repeat):
            t0 = time.time()
            try:
                ans = call(key, model, b64)
            except Exception as e:  # 네트워크/API 오류도 결과의 일부
                ans = f"ERROR: {e}"
            times.append(time.time() - t0)
            answers.append(ans)
        latencies += times
        consistent = len(set(answers)) == 1
        agree_cnt += consistent

        expected = expected_map.get(p.name, p.stem)  # 파일명 = 기대값 규약
        got = answers[0]
        mark = ""
        if expected:
            scored_cnt += 1
            ok = got.replace(" ", "") == expected.replace(" ", "")
            correct_cnt += ok
            mark = "O" if ok else "X"
        print(f"  {p.name:30s} -> {got:20s} {'일치' if consistent else '불일치 ' + str(answers)}"
              f"  {mark}  ({statistics.mean(times):.1f}s)")

    print(f"\n요약: 재질의 일치 {agree_cnt}/{len(crops)}"
          + (f", 문자열 일치 {correct_cnt}/{scored_cnt}" if scored_cnt else "")
          + f", 평균 지연 {statistics.mean(latencies):.1f}s"
          f" (페이지당 병렬 전송 시 ≈ 최댓값 {max(latencies):.1f}s)")
    print("주의: 문자열 불일치여도 수학적 동치(0.5 vs 1/2)면 실제 파이프라인(SymPy)에선 정답 처리된다.")


if __name__ == "__main__":
    main()
