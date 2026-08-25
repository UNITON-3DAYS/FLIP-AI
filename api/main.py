"""FLIP 채점 서버 (FastAPI).

Spring이 base64 페이지 사진과 교재 이름(name)을 POST /grade로 보내면, 문제별
O/X/보류 채점 결과를 같은 요청의 응답으로 돌려준다.

핵심 계약:
- 페이지 번호는 받지 않는다. 정답 DB 전체를 로드해 valid_pages를 얻고, OCR로
  쪽수를 식별한다(DB 로드 → OCR 쪽수 → 채점).
- track=workbook만 구현. exam은 값만 받고 501(트랙 2에서 구현).
- 채점 실패는 500이 아니라 보류로 흡수한다(VLM 키 없음/OCR 미설치/인식 실패).
  없는 교재는 404, 깨진 base64/이미지는 400.

실행: uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
import base64
import binascii
import logging
import os

from fastapi import FastAPI, HTTPException

import grade
from flip import ocr, vlm
from flip.answers import get_source
from flip.preprocess import decode_image, preprocess_array
from flip.results import HOLD, O, PageResult, X
from api.schemas import (
    Counts, GradeRequest, GradeResponse, HealthResponse,
    QuestionResultOut, Track, Verdict,
)

log = logging.getLogger("flip.api")

# OpenAPI info.version. 계약이 바뀌면 올린다 — Java SDK(AKR-20)의 패키지 버전이
# 여기서 파생되므로, 이 값을 안 올리면 소비자가 변경을 인지하지 못한다.
API_VERSION = "0.1.0"

app = FastAPI(
    title="FLIP 채점 서버",
    version=API_VERSION,
    description="페이지 사진 1장 → 문제별 O/X/보류 채점 결과.",
)

# 내부 한글 판정('보류') → 응답 Enum 매핑.
_VERDICT = {O: Verdict.O, X: Verdict.X, HOLD: Verdict.HOLD}


@app.on_event("startup")
def _warmup():
    """느린 초기화(OCR 엔진, 정답 소스)를 첫 요청 전에 끝낸다. 실패해도 서버는 뜬다.

    OCR 워밍업은 첫 실행 시 PaddleOCR 모델을 내려받아 수십 초 걸릴 수 있다. CI
    헬스체크나 빠른 기동이 필요할 때는 FLIP_WARMUP_OCR=0으로 건너뛴다(첫 요청에서
    지연 로드된다).
    """
    get_source()  # 정답 소스 싱글턴 생성(환경변수 읽기)
    if os.environ.get("FLIP_WARMUP_OCR", "1") != "1":
        log.info("OCR 워밍업 건너뜀 (FLIP_WARMUP_OCR=0) — 첫 요청에서 지연 로드")
    elif ocr.available():
        try:
            import numpy as np
            ocr.run_ocr(np.full((32, 32), 255, np.uint8))  # 엔진 로드 트리거
            log.info("OCR 엔진 워밍업 완료")
        except Exception as e:  # 워밍업 실패는 치명적이지 않다(요청 때 재시도)
            log.warning("OCR 워밍업 실패(무시): %s", e)
    else:
        log.warning("PaddleOCR 미설치 — 쪽수 인식 불가, 페이지 보류로 응답")


def _to_response(pr: PageResult, track: Track, name: str) -> GradeResponse:
    c = pr.counts()
    return GradeResponse(
        track=track,
        name=name,
        page_no=pr.page_no,
        results=[
            QuestionResultOut(
                question_no=r.question_no,
                verdict=_VERDICT[r.verdict],
                student_answer=r.student_answer,
                detail=r.detail,
            )
            for r in (pr.results or [])
        ],
        counts=Counts(O=c[O], X=c[X], HOLD=c[HOLD]),
        hold_reason=pr.hold_reason,
    )


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(ocr=ocr.available(), vlm=vlm.available())


@app.post("/grade", response_model=GradeResponse)
def grade_endpoint(req: GradeRequest):
    """페이지 사진 1장을 동기로 채점. 블로킹 파이프라인이라 FastAPI가 스레드풀에서 돈다."""
    if req.track is not Track.exam:
        pass  # workbook 경로
    else:
        # 자체 시험지 트랙은 앞단(마커+고정좌표)이 아직 없다 → 명시적 미구현.
        raise HTTPException(status_code=501, detail="exam 트랙은 아직 미구현 (트랙 2)")

    # 1) base64 → 이미지 바이트 → BGR
    try:
        raw = base64.b64decode(req.image_base64, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="image_base64 디코드 실패 (올바른 base64가 아님)")
    if not raw:
        raise HTTPException(status_code=400, detail="image_base64가 비어 있음")
    try:
        img = decode_image(raw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2) 정답 DB 조회 (없는 교재 → 404). 쪽수 식별이 valid_pages에 의존하므로 먼저.
    db = get_source().get(req.name)
    if db is None:
        raise HTTPException(status_code=404, detail=f"등록되지 않은 교재: {req.name!r}")

    # 3) 보정 → 채점. 예기치 못한 예외도 페이지 전체 보류로 흡수(500 금지).
    try:
        color, gray = preprocess_array(img)
        pr = grade.grade_prepared(color, gray, db, page_hint=None, label=req.name)
    except Exception as e:
        log.exception("채점 중 예외 — 페이지 보류로 흡수")
        pr = PageResult(image=req.name, page_no="", results=[],
                        hold_reason=f"처리 오류: {e}")

    return _to_response(pr, req.track, req.name)
