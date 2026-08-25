"""정답 소스 추상화: 교재 이름(book) -> AnswerDB.

지금까지 `AnswerDB.load(path)`로 로컬 파일만 읽던 정답 조회를, "교재 이름을 주면
정답을 돌려주는" 인터페이스 뒤로 감춘다. 서버(api)는 저장소가 로컬 JSON인지
Supabase인지 몰라도 되고, 환경변수 하나로 백엔드를 바꾼다.

환경변수:
  FLIP_ANSWER_BACKEND   json | supabase (기본 json)
  FLIP_ANSWER_DIR       json 백엔드: 정답 JSON들이 있는 디렉터리 (기본 현재 디렉터리)
  FLIP_SUPABASE_URL     supabase 백엔드: 프로젝트 URL (예: https://xxx.supabase.co)
  FLIP_SUPABASE_KEY     supabase 백엔드: service_role 또는 anon 키
  FLIP_SUPABASE_TABLE   supabase 백엔드: 정답 테이블명 (기본 answers)

방어 원칙(vlm 모듈과 동일):
- 없는 교재는 예외가 아니라 None. 서버가 이걸 404로 바꾼다.
- Supabase 네트워크 오류·키 없음·스키마 불일치도 전부 None + 로그. 크래시 금지.
- 저작권 자료라 정답 JSON은 repo에 넣지 않는다(gitignore). 디렉터리로 주입한다.
"""
import json
import logging
import os
import threading
from pathlib import Path

import requests

from flip.db import AnswerDB

log = logging.getLogger(__name__)

# Supabase 테이블 컬럼명. 스키마 확정 전이라 한 곳(_rows_to_answerdb)에 모아
# 나중에 바꾸기 쉽게 둔다.
SUPABASE_TIMEOUT = 10


def _normalize_book(name):
    """교재 이름 정규화: 앞뒤·중복 공백 제거 + 대소문자 무시.

    Spring이 "쎈 2-1", " 쎈  2-1 ", "쎈 2-1"을 섞어 보내도 같은 교재로 맞춘다.
    파일명이 아니라 JSON 안의 book 필드를 기준으로 매칭하므로 파일명 규칙과 무관.
    """
    return " ".join((name or "").split()).casefold()


class JsonAnswerSource:
    """디렉터리 안의 정답 JSON들에서 book 이름으로 조회.

    디렉터리를 한 번 스캔해 {정규화된 book -> 파일경로} 색인을 만들고(파일마다 book
    필드만 읽음), get() 시 매칭 파일을 AnswerDB로 로드해 캐시한다.
    """

    def __init__(self, directory=None):
        self.directory = Path(directory or os.environ.get("FLIP_ANSWER_DIR") or ".")
        self._index = None            # {normalized book -> Path}
        self._cache = {}              # normalized book -> AnswerDB
        self._lock = threading.Lock()

    def _build_index(self):
        index = {}
        if not self.directory.is_dir():
            log.warning("정답 디렉터리 없음: %s", self.directory)
            return index
        for path in sorted(self.directory.glob("*.json")):
            try:
                with open(path, encoding="utf-8") as f:
                    book = json.load(f).get("book", "")
            except (OSError, ValueError) as e:
                log.warning("정답 JSON 읽기 실패 %s: %s", path, e)
                continue
            key = _normalize_book(book)
            if not key:
                continue
            if key in index:
                log.warning("중복 교재 '%s': %s 무시(%s 사용)", book, path, index[key])
                continue
            index[key] = path
        return index

    def get(self, book):
        key = _normalize_book(book)
        if not key:
            return None
        with self._lock:
            if self._index is None:
                self._index = self._build_index()
            if key in self._cache:
                return self._cache[key]
            path = self._index.get(key)
            if path is None:
                return None
            try:
                db = AnswerDB.load(path)
            except (OSError, ValueError, KeyError) as e:
                log.warning("정답 DB 로드 실패 %s: %s", path, e)
                return None
            self._cache[key] = db
            return db


def _rows_to_answerdb(book, rows):
    """Supabase row 리스트 -> AnswerDB.

    Supabase 테이블 스키마 확정 전이므로 변환은 이 함수 하나에 모은다. 기대 컬럼:
      book, page, question_no, type(multiple_choice|subjective), answer, num_choices
    answer는 문자열/정수/리스트 어느 쪽이든 기존 AnswerDB.from_dict가 흡수한다.
    row가 하나도 없으면(= 없는 교재) None.
    """
    pages = {}
    for r in rows:
        page = str(r["page"])
        q = {
            "question_no": r["question_no"],
            "type": r["type"],
            "answer": r["answer"],
        }
        if r.get("num_choices") is not None:
            q["num_choices"] = r["num_choices"]
        pages.setdefault(page, []).append(q)
    if not pages:
        return None
    return AnswerDB.from_dict({"book": book, "pages": pages})


class SupabaseAnswerSource:
    """Supabase(PostgREST)에서 book으로 정답 행을 받아 AnswerDB로 변환.

    키·URL 없음, 네트워크 오류, 스키마 불일치는 전부 None + 로그(크래시 금지).
    """

    def __init__(self, url=None, key=None, table=None):
        self.url = (url or os.environ.get("FLIP_SUPABASE_URL") or "").rstrip("/")
        self.key = key or os.environ.get("FLIP_SUPABASE_KEY") or ""
        self.table = table or os.environ.get("FLIP_SUPABASE_TABLE") or "answers"
        self._cache = {}
        self._lock = threading.Lock()

    def available(self):
        return bool(self.url and self.key)

    def get(self, book):
        key = _normalize_book(book)
        if not key:
            return None
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        if not self.available():
            log.warning("Supabase 미설정(FLIP_SUPABASE_URL/KEY) — None 반환")
            return None
        rows = self._fetch(book)
        if rows is None:
            return None  # 조회 실패는 캐시하지 않는다 (재시도 여지)
        try:
            db = _rows_to_answerdb(book, rows)
        except (KeyError, ValueError, TypeError) as e:
            log.warning("Supabase 응답 스키마 불일치(%s): %s", book, e)
            return None
        with self._lock:
            self._cache[key] = db   # None(없는 교재)도 캐시 — 반복 조회 방지
        return db

    def _fetch(self, book):
        """PostgREST GET. 성공 시 row 리스트, 실패 시 None."""
        try:
            resp = requests.get(
                f"{self.url}/rest/v1/{self.table}",
                params={"book": f"eq.{book}", "select": "*"},
                headers={"apikey": self.key, "Authorization": f"Bearer {self.key}"},
                timeout=SUPABASE_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as e:
            log.warning("Supabase 조회 실패(%s): %s", book, e)
            return None


# ── 팩토리 ───────────────────────────────────────────────────────────────

_source = None
_source_lock = threading.Lock()


def _build_source():
    backend = (os.environ.get("FLIP_ANSWER_BACKEND") or "json").strip().lower()
    if backend == "supabase":
        return SupabaseAnswerSource()
    if backend != "json":
        log.warning("알 수 없는 FLIP_ANSWER_BACKEND=%r — json으로 대체", backend)
    return JsonAnswerSource()


def get_source():
    """환경변수로 선택된 정답 소스 싱글턴. 서버 startup에서 한 번 만든다."""
    global _source
    with _source_lock:
        if _source is None:
            _source = _build_source()
        return _source


def reset_source():
    """테스트용: 다음 get_source()가 환경변수를 다시 읽게 한다."""
    global _source
    with _source_lock:
        _source = None


# ── selftest ───────────────────────────────────────────────────────────────

def _selftest():
    import tempfile

    # 정규화
    assert _normalize_book(" 쎈  2-1 ") == _normalize_book("쎈 2-1")
    assert _normalize_book("") == ""
    assert _normalize_book(None) == ""

    sample = {
        "book": "쎈 2-1",
        "pages": {
            "12": [
                {"question_no": "1", "type": "multiple_choice", "answer": 3, "num_choices": 5},
                {"question_no": "2", "type": "subjective", "answer": "-367"},
            ],
        },
    }
    with tempfile.TemporaryDirectory() as d:
        with open(Path(d) / "db.ssen_2-1.json", "w", encoding="utf-8") as f:
            json.dump(sample, f, ensure_ascii=False)

        src = JsonAnswerSource(d)
        # 파일명이 아니라 book 필드로 매칭 + 공백/대소문자 무시
        db = src.get(" 쎈 2-1 ")
        assert db is not None and db.valid_pages() == {"12"}
        assert db.questions_for("12")[0].answer == 3
        assert src.get("없는교재") is None       # 없는 교재 → None (예외 아님)
        assert src.get("") is None
        assert src.get("쎈 2-1") is db           # 캐시 동일 인스턴스

        # 빈 디렉터리도 크래시 없이 None
        with tempfile.TemporaryDirectory() as empty:
            assert JsonAnswerSource(empty).get("쎈 2-1") is None

    # Supabase row -> AnswerDB 변환 (네트워크 없이 단위 검증)
    rows = [
        {"page": 12, "question_no": "1", "type": "multiple_choice", "answer": 3, "num_choices": 5},
        {"page": 12, "question_no": "7-1", "type": "subjective", "answer": ["-2", "3"]},
        {"page": 13, "question_no": "2", "type": "subjective", "answer": "1/2", "num_choices": None},
    ]
    db = _rows_to_answerdb("쎈 2-1", rows)
    assert db.valid_pages() == {"12", "13"}
    assert db.questions_for("12")[1].answer == ["-2", "3"]
    assert db.questions_for("13")[0].qtype == "subjective"
    assert _rows_to_answerdb("빈교재", []) is None

    # Supabase 미설정이면 조회 없이 None (크래시 금지)
    supa = SupabaseAnswerSource(url="", key="")
    assert supa.available() is False
    assert supa.get("쎈 2-1") is None

    # 팩토리: 기본 json, 알 수 없는 값도 json으로 대체
    saved = os.environ.pop("FLIP_ANSWER_BACKEND", None)
    try:
        reset_source()
        assert isinstance(get_source(), JsonAnswerSource)
        os.environ["FLIP_ANSWER_BACKEND"] = "supabase"
        reset_source()
        assert isinstance(get_source(), SupabaseAnswerSource)
    finally:
        reset_source()
        if saved is None:
            os.environ.pop("FLIP_ANSWER_BACKEND", None)
        else:
            os.environ["FLIP_ANSWER_BACKEND"] = saved
