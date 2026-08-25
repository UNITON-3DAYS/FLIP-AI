# FLIP 채점 서버 API

Spring이 페이지 사진과 교재 이름을 보내면 문제별 O/X/보류 채점 결과를 **동기**로
돌려주는 HTTP API. 스펙의 정본은 서버가 뱉는 `/openapi.json`(repo 루트에도 커밋)이며,
이 문서는 소비자용 요약이다.

## 실행

### 로컬 (개발)

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
# 대화형 문서: http://localhost:8000/docs   (OpenAPI: /openapi.json)
```

### Docker (로컬·서버 공통)

```bash
cp .env.example .env                 # 값 채우기 (VLM 키 등)
mkdir -p answers                     # 정답 JSON을 여기 둔다 (저작권 자료라 이미지엔 안 넣음)
docker compose up --build            # http://localhost:8000
curl -sf http://localhost:8000/health
```

- 정답 디렉터리는 `./answers`가 기본값이며 컨테이너의 `/answers`(읽기전용)로 마운트된다.
  다른 경로면 `FLIP_ANSWER_HOST_DIR`로 지정한다.
- `.env`가 없어도 컨테이너는 뜬다(키 없으면 채점이 보류로 나올 뿐).
- 이미지에는 비밀정보(`.env`)·저작권 자료(`db.s/sen*.json`, `samples/`, `*.pdf`)가
  들어가지 않는다(`.dockerignore`).
- 빠른 기동이 필요하면 `FLIP_WARMUP_OCR=0`으로 OCR 모델 사전 로드를 건너뛴다
  (첫 요청에서 지연 로드).

필요 환경변수는 `.env.example` 참고. 최소: 정답 소스(`FLIP_ANSWER_BACKEND`)와
VLM 키(`FLIP_VLM_API_KEY`). 쪽수 인식을 위해 PaddleOCR도 설치돼 있어야 한다(도커
이미지에는 포함).

## POST /grade

페이지 사진 1장을 채점한다. **페이지 번호는 보내지 않는다** — 서버가 교재 정답
DB를 로드해 OCR로 쪽수를 스스로 식별한다.

### 요청

#### Request Header

| 변수명 | 타입 | 필수여부 | 설명 | 예시 |
| --- | --- | --- | --- | --- |
| Content-Type | string | O | 요청 본문 형식 | `application/json` |

#### Path Variable

없음.

#### Query String

없음. (쪽수도 보내지 않는다 — 서버가 OCR로 식별한다.)

#### Request Body (application/json)

| 변수명 | 타입 | 필수여부 | 설명 | 예시 |
| --- | --- | --- | --- | --- |
| workSheetSource | string(enum) | X (기본 `WORKBOOK`) | `WORKBOOK`(문제집) \| `EXAM`(자체 시험지). 둘 다 채점, 파이프라인 동일 | `WORKBOOK` |
| name | string | O | 책·시험지 이름. 정답 DB 조회 키 | `쎈 2-1` |
| image_base64 | string | O | 페이지 사진(JPEG/PNG) 바이트의 base64. `data:image/...` 접두사 없이 순수 base64만 | `iVBORw0KGgo...` |

```json
{ "workSheetSource": "WORKBOOK", "name": "쎈 2-1", "image_base64": "iVBORw0KGgo..." }
```

### 응답

`200 OK` (application/json)

| 변수명 | 타입 | 설명 |
|---|---|---|
| `workSheetSource` | enum | 요청 workSheetSource 반향. |
| `name` | string | 요청 name 반향. |
| `page_no` | string | 인식된 쪽수. 못 읽으면 `""`. |
| `results` | array | 문제별 결과. `hold_reason`이 있으면 전부 보류. |
| `results[].question_no` | string | 문제 번호 (예: `"1"`, `"7-1"`). |
| `results[].verdict` | `"O"` \| `"X"` \| `"HOLD"` | 정답 / 오답 / 보류. |
| `results[].student_answer` | string | 인식된 학생 답 (없으면 `""`). |
| `results[].detail` | string | 판정 근거 또는 보류 사유. |
| `counts` | object | `{ "O": n, "X": n, "HOLD": n }`. |
| `hold_reason` | string | 페이지 전체 보류 사유(쪽수 미인식 등). 있으면 개별 결과는 전부 보류. |

```json
{
  "workSheetSource": "WORKBOOK", "name": "쎈 2-1", "page_no": "12",
  "results": [
    { "question_no": "1", "verdict": "O", "student_answer": "3", "detail": "" },
    { "question_no": "2", "verdict": "X", "student_answer": "-368", "detail": "" },
    { "question_no": "3", "verdict": "HOLD", "student_answer": "", "detail": "인식 불확실" }
  ],
  "counts": { "O": 1, "X": 1, "HOLD": 1 },
  "hold_reason": ""
}
```

### 상태 코드

| 코드 | 언제 |
|---|---|
| 200 | 채점 완료. **인식 실패·VLM 키 없음·OCR 미설치도 200**이며 결과가 보류로 나온다. |
| 400 | `image_base64`가 올바른 base64가 아니거나 이미지로 디코드되지 않음. |
| 404 | `name`이 등록되지 않은 교재. |

> **판정 철학**: 확신이 없으면 X가 아니라 HOLD. 오답 처리보다 사람 확인이 낫다.
> 그래서 서버는 웬만한 실패를 500이 아니라 200+보류로 흡수한다.

### 타임아웃 (소비자 주의)

한 페이지 채점은 문제 수만큼 VLM을 호출하며 **수 초~십수 초** 걸릴 수 있다.
Spring의 read timeout을 넉넉히(≈60s) 잡을 것.

## GET /health

배포 점검용. 요청 파라미터 없음(Request Header / Path Variable / Query String 모두 없음).

### 응답

`200 OK` (application/json)

| 변수명 | 타입 | 설명 |
|---|---|---|
| `status` | string | 항상 `"ok"`. |
| `ocr` | boolean | PaddleOCR 사용 가능 여부. `false`면 쪽수 인식이 안 돼 페이지가 보류로만 나온다. |
| `vlm` | boolean | VLM API 키 설정 여부. `false`면 채점이 전부 보류다. |

```json
{ "status": "ok", "ocr": true, "vlm": true }
```

## 소비자 클라이언트 (SDK)

Spring은 이 API의 DTO를 손으로 쓰지 않고 `/openapi.json`에서 타입 있는 Java
클라이언트를 생성해 쓴다. 생성·배포 방식과 소비 스니펫은 **AKR-20**에서 다룬다.
핵심 주의: FastAPI는 **OpenAPI 3.1**을 내므로 openapi-generator는 **7.x 이상**을
써야 한다(6.x는 3.1을 못 먹는다).
