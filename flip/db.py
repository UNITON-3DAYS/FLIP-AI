"""정답 mock DB 로드/조회.

스키마 (JSON):
{
  "book": "문제집 식별자",
  "pages": {
    "12": [
      {"question_no": "1",   "type": "multiple_choice", "answer": 3, "num_choices": 5},
      {"question_no": "2",   "type": "subjective",      "answer": "-367"},
      {"question_no": "7-1", "type": "subjective",      "answer": ["-2", "3"]}
    ]
  }
}

- 쪽수(pages 키)는 문자열. 페이지 사진에서 읽은 쪽수로 그 페이지의 문제 목록을 조회한다.
- subjective answer: 문자열 하나 또는 해가 여러 개면 리스트.
- multiple_choice answer: 1~num_choices 정수. 복수정답("모두 고르면")이면 정수 리스트.
"""
import json
from dataclasses import dataclass, field

MULTIPLE_CHOICE = "multiple_choice"
SUBJECTIVE = "subjective"


@dataclass
class Question:
    question_no: str          # "1", "12", "7-1"
    qtype: str                # MULTIPLE_CHOICE | SUBJECTIVE
    answer: object            # 객관식 int, 주관식 str 또는 [str]
    num_choices: int = 5      # 객관식만 의미


@dataclass
class AnswerDB:
    book: str
    pages: dict = field(default_factory=dict)  # 쪽수(str) -> [Question]

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw):
        pages = {}
        for page_no, questions in raw["pages"].items():
            qs = []
            for q in questions:
                qtype = q["type"]
                if qtype not in (MULTIPLE_CHOICE, SUBJECTIVE):
                    raise ValueError(f"알 수 없는 문제 유형: {qtype}")
                qs.append(Question(
                    question_no=str(q["question_no"]),
                    qtype=qtype,
                    answer=q["answer"],
                    num_choices=int(q.get("num_choices", 5)),
                ))
            pages[str(page_no)] = qs
        return cls(book=raw.get("book", ""), pages=pages)

    def questions_for(self, page_no):
        """쪽수 -> 문제 목록. 없는 쪽수면 None (빈 페이지와 구분)."""
        return self.pages.get(str(page_no))

    def valid_pages(self):
        """DB에 등록된 쪽수 집합. 쪽수 오인식 필터에 쓴다."""
        return set(self.pages.keys())
