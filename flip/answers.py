"""정답 소스 추상화: 교재 이름(book) -> AnswerDB.

지금까지 `AnswerDB.load(path)`로 로컬 파일만 읽던 정답 조회를, "교재 이름을 주면
정답을 돌려주는" 인터페이스 뒤로 감춘다. 서버(api)는 저장소가 로컬 JSON인지
MySQL인지 몰라도 되고, 환경변수 하나로 백엔드를 바꾼다.

환경변수:
  FLIP_ANSWER_BACKEND   json | mysql (기본 json)
  FLIP_ANSWER_DIR       json 백엔드: 정답 JSON들이 있는 디렉터리 (기본 현재 디렉터리)
  FLIP_MYSQL_HOST       mysql 백엔드: 호스트 (예: 34.64.227.104)
  FLIP_MYSQL_PORT       mysql 백엔드: 포트 (기본 3306)
  FLIP_MYSQL_DB         mysql 백엔드: 데이터베이스명 (예: flip)
  FLIP_MYSQL_USER       mysql 백엔드: 사용자
  FLIP_MYSQL_PASSWORD   mysql 백엔드: 비밀번호
                        (테이블은 question⨝worksheet 고정, title=교재명으로 조회)

방어 원칙(vlm 모듈과 동일):
- 없는 교재는 예외가 아니라 None. 서버가 이걸 404로 바꾼다.
- 네트워크 오류·접속정보 없음·스키마 불일치도 전부 None + 로그. 크래시 금지.
- 저작권 자료라 정답 JSON은 repo에 넣지 않는다(gitignore). DB나 디렉터리로 주입한다.
"""
import json
import logging
import os
import re
import threading
from pathlib import Path

from flip.db import AnswerDB

log = logging.getLogger(__name__)

MYSQL_TIMEOUT = 10


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
    """DB row 리스트 -> AnswerDB (MySQL 백엔드 공용 변환기).

    row → AnswerDB 변환을 이 함수 하나에 모아, 스키마가 바뀌면 여기만 고친다. 기대 컬럼:
      page, question_no, type(multiple_choice|subjective), answer, num_choices
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


def _map_question_row(row):
    """MySQL question 조인 row → 표준 문제 dict (_rows_to_answerdb 계약).

    - type: Backend ENUM('MULTIPLE_CHOICE','SUBJECTIVE')을 소문자화 → db.py 상수와 일치.
    - 객관식 correct_answer: '3'·'2,4' → 정수 리스트([3]·[2,4]). _verdict_mcq가 int로
      비교하므로 문자열 '2,4'를 그대로 두면 int() 크래시. 숫자를 못 뽑으면 원문 유지
      (from_dict/verdict 단계에서 스키마 불일치로 걸러짐).
    - 주관식 correct_answer: raw 문자열 그대로(SymPy가 파싱).
    - num_choices: 컬럼값이 있으면 쓰고, NULL/없으면 5(Question 기본값)에 맡긴다.
    """
    qtype = str(row.get("type") or "").strip().lower()
    raw = str(row.get("correct_answer") or "").strip()
    q = {"page": row["page"], "question_no": row["question_number"], "type": qtype}
    if qtype == "multiple_choice":
        nums = [int(n) for n in re.findall(r"[1-9]", raw)]
        q["answer"] = nums or raw
    else:
        q["answer"] = raw
    if row.get("num_choices") is not None:
        q["num_choices"] = row["num_choices"]
    return q


class MySqlAnswerSource:
    """MySQL에서 book으로 정답 행을 받아 AnswerDB로 변환.

    Backend(Spring)와 같은 GCP MySQL(flip DB)을 읽는다 — 정답의 단일 출처.
    접속정보 없음/네트워크 오류/스키마 불일치는 전부 None + 로그(크래시 금지).
    row → AnswerDB 변환은 _rows_to_answerdb에 위임한다.
    """

    def __init__(self, host=None, port=None, db=None, user=None, password=None):
        self.host = host or os.environ.get("FLIP_MYSQL_HOST") or ""
        self.port = int(port or os.environ.get("FLIP_MYSQL_PORT") or 3306)
        self.db = db or os.environ.get("FLIP_MYSQL_DB") or ""
        self.user = user or os.environ.get("FLIP_MYSQL_USER") or ""
        self.password = password if password is not None else (os.environ.get("FLIP_MYSQL_PASSWORD") or "")
        self._cache = {}
        self._lock = threading.Lock()

    def available(self):
        return bool(self.host and self.db and self.user)

    def get(self, book):
        key = _normalize_book(book)
        if not key:
            return None
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        if not self.available():
            log.warning("MySQL 미설정(FLIP_MYSQL_HOST/DB/USER) — None 반환")
            return None
        raw_rows = self._fetch(book)
        if raw_rows is None:
            return None  # 조회 실패는 캐시하지 않는다 (재시도 여지)
        try:
            rows = [_map_question_row(r) for r in raw_rows]
            db = _rows_to_answerdb(book, rows)
        except (KeyError, ValueError, TypeError) as e:
            log.warning("MySQL 응답 스키마 불일치(%s): %s", book, e)
            return None
        with self._lock:
            self._cache[key] = db   # None(없는 교재)도 캐시 — 반복 조회 방지
        return db

    def _fetch(self, book):
        """worksheet.title=book인 question 행 조회. 성공 시 dict row 리스트, 실패 시 None.

        question(문제)을 worksheet(문제지)에 조인해 교재명(title)으로 필터한다.
        테이블·컬럼은 Backend 스키마 고정이라 식별자 인젝션 없음. book은 파라미터 바인딩.
        반환 컬럼: page, question_number, type, correct_answer → _map_question_row가 표준화.
        """
        try:
            import pymysql
            from pymysql.cursors import DictCursor
        except ImportError:
            log.warning("pymysql 미설치 — MySQL 백엔드 사용 불가")
            return None
        conn = None
        try:
            conn = pymysql.connect(
                host=self.host, port=self.port, database=self.db,
                user=self.user, password=self.password,
                connect_timeout=MYSQL_TIMEOUT, read_timeout=MYSQL_TIMEOUT,
                charset="utf8mb4", cursorclass=DictCursor,
            )
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT q.page AS page, q.question_number AS question_number, "
                    "q.type AS type, q.correct_answer AS correct_answer, "
                    "q.num_choices AS num_choices "
                    "FROM question q JOIN worksheet w ON w.id = q.worksheet_id "
                    "WHERE w.title = %s", (book,))
                return cur.fetchall()
        except Exception as e:  # 접속/쿼리/타임아웃 전부 보류
            log.warning("MySQL 조회 실패(%s): %s", book, e)
            return None
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass


# ── 팩토리 ───────────────────────────────────────────────────────────────

_source = None
_source_lock = threading.Lock()


def _build_source():
    backend = (os.environ.get("FLIP_ANSWER_BACKEND") or "json").strip().lower()
    if backend == "mysql":
        return MySqlAnswerSource()
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

    # DB row -> AnswerDB 변환 (네트워크 없이 단위 검증, MySQL 백엔드 공용)
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

    # MySQL 미설정이면 접속 없이 None (크래시 금지, pymysql 없어도 통과)
    my = MySqlAnswerSource(host="", db="", user="")
    assert my.available() is False
    assert my.get("쎈 2-1") is None
    assert my.get("") is None

    # question row 매핑: type 소문자화 + 객관식 문자열 정답 → 정수 리스트 + num_choices 전달
    m = _map_question_row({"page": 12, "question_number": 46, "type": "MULTIPLE_CHOICE",
                           "correct_answer": "2,4", "num_choices": 4})
    assert m == {"page": 12, "question_no": 46, "type": "multiple_choice",
                 "answer": [2, 4], "num_choices": 4}
    s = _map_question_row({"page": 13, "question_number": 47,
                           "type": "SUBJECTIVE", "correct_answer": "1/2", "num_choices": None})
    assert s["type"] == "subjective" and s["answer"] == "1/2" and "num_choices" not in s
    # 매핑 결과가 AnswerDB로 정상 변환 + 채점기 계약(객관식 int 비교·num_choices) 충족
    db = _rows_to_answerdb("교재", [m, s])
    assert db.questions_for("12")[0].answer == [2, 4]  # _verdict_mcq int() OK
    assert db.questions_for("12")[0].num_choices == 4  # NULL 아니면 컬럼값 사용

    # 팩토리: 기본 json, 알 수 없는 값도 json으로 대체
    saved = os.environ.pop("FLIP_ANSWER_BACKEND", None)
    try:
        reset_source()
        assert isinstance(get_source(), JsonAnswerSource)
        os.environ["FLIP_ANSWER_BACKEND"] = "mysql"
        reset_source()
        assert isinstance(get_source(), MySqlAnswerSource)
    finally:
        reset_source()
        if saved is None:
            os.environ.pop("FLIP_ANSWER_BACKEND", None)
        else:
            os.environ["FLIP_ANSWER_BACKEND"] = saved
