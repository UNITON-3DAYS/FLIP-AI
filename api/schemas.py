"""FastAPI 요청/응답 스키마.

Spring 등 소비자는 이 스키마에서 생성된 OpenAPI(`/openapi.json`)로 타입 있는
클라이언트를 뽑는다. 그래서 필드는 codegen 친화적으로 둔다:
- verdict/track은 Enum → Java 쪽에 enum이 생겨 switch로 받는다.
- image_base64는 str(byte[] 아님) → 소비자가 이중 인코딩하지 않는다.
"""
from enum import Enum

from pydantic import BaseModel, Field


class Track(str, Enum):
    """채점 트랙. 문제집(쎈 등 기성 학습지) vs 자체 시험지."""
    workbook = "workbook"
    exam = "exam"


class Verdict(str, Enum):
    """문제 판정. 내부 한글 값('보류')을 ASCII로 직렬화한 것."""
    O = "O"        # 정답
    X = "X"        # 오답
    HOLD = "HOLD"  # 보류(사람 확인)


class GradeRequest(BaseModel):
    track: Track = Field(default=Track.workbook,
                         description="채점 트랙. 기본 workbook(문제집).")
    name: str = Field(description="책·시험지 이름. 정답 DB 조회 키 (예: '쎈 2-1').")
    image_base64: str = Field(description="페이지 사진 1장의 base64 (JPEG/PNG 바이트).")


class QuestionResultOut(BaseModel):
    question_no: str = Field(description="문제 번호 (예: '1', '7-1').")
    verdict: Verdict
    student_answer: str = Field(default="", description="인식된 학생 답 (없으면 빈 문자열).")
    detail: str = Field(default="", description="판정 근거 또는 보류 사유.")


class Counts(BaseModel):
    O: int = 0
    X: int = 0
    HOLD: int = 0


class GradeResponse(BaseModel):
    track: Track
    name: str
    page_no: str = Field(default="", description="인식된 쪽수 (못 읽으면 빈 문자열).")
    results: list[QuestionResultOut] = Field(default_factory=list)
    counts: Counts
    hold_reason: str = Field(default="", description="페이지 전체 보류 사유 (있으면 results는 전부 보류).")


class HealthResponse(BaseModel):
    status: str = "ok"
    ocr: bool = Field(description="PaddleOCR 사용 가능 여부. false면 쪽수 인식 불가 → 페이지 보류.")
    vlm: bool = Field(description="VLM API 키 설정 여부. false면 채점 보류.")
