# FLIP 답안 인식 파이프라인

손으로 쓴 답(숫자·기호)을 인식해 자동 채점하는 **인식기(YOLO-det)의 설계·학습·사용법.**

> 시스템 경계·콘텐츠 트랙(자체 시험지 / 쎈)·구현 순서는 [architecture.md](architecture.md) 참고.
> 아래의 "OMR 답안지"는 그 문서의 **자체 시험지 트랙**에 해당하며, 인식기(YOLO)는 두 트랙 공통이다.

---

## 1. 핵심 결정

| 항목 | 결정 | 이유 |
|---|---|---|
| 촬영 방식 | **책장 넘김 자동촬영 → 버림** | OMR 답안지로 바뀌면 넘길 페이지가 없음. 모션감지·페이지인식·정렬 리스크 통째 제거 |
| 답 쓰는 위치 | **별도 OMR 답안지** (풀이는 원래 문제집에) | 원본 문제집은 답 위치 고정 불가 → 답만 정해진 종이에 |
| 답 칸 형태 | **칸 하나에 답 통째로** (글자별 칸 X) | 학생 편의. 대신 인식이 조금 어려워지는 건 Confidence→사람이 흡수 |
| 표기 방식 | 버블 X, **숫자·기호를 손으로 씀** (객관식도 번호를 숫자로) | 인식 경로 하나로 통일 |
| 인식 방법 | **로컬 YOLO-det** (API 호출 X) | 비용 0. 입력이 깨끗해서(고정 칸, 낙서 없음) 로컬로 충분 |
| 모델 | **yolo26n** (nano, 2026.1 최신) | 태스크 쉬워 정확도는 어느 버전이든 최대치. 26 선택은 CPU 43%↑·NMS-free·edge 최적화(서빙/폰) 때문. s는 과함 |
| 배포 | 지금은 **`.pt`**, 온디바이스(폰) export는 **확장** | 확장 시 CoreML/TFLite + int8 |

### 왜 YOLO-cls 아니고 detection인가
- cls(박스 통째 → 클래스 1개)는 답이 무한(`-3`, `2/5`, `12`…)이라 클래스화 불가.
- **det**는 글자마다 박스+클래스 → x좌표 순 정렬 → 문자열 조립. 붙여 쓴 글자·분수에 강함.

---

## 2. 인식 파이프라인

```
답안지 사진 1장
  → 고정 템플릿 좌표로 답 칸 N개 crop
  → 각 crop을 YOLO-det로 추론 (글자별 박스+클래스+conf)
  → x좌표 순 정렬 → 문자열 조립   (예: -,3,/,4 → "-3/4")
  → 정답 키와 비교 → O/X
  → Confidence 낮은 답만 선생님 확인 (Human-in-the-loop)
```

- **답 신뢰도 = 그 답의 글자들 중 최저 conf** (자릿수 하나만 흔들려도 답이 틀리므로 최저값이 타당).
- 신뢰도 < 임계값(기본 0.5) → 사람 확인 큐로.

---

## 3. 클래스 / 데이터셋

- **클래스 (15개):** `0 1 2 3 4 5 6 7 8 9 - / = pi`
  - `-` 음수, `/` 분수, `pi` → 표시·채점 시 `π`
- **데이터셋:** [xainano/handwrittenmathsymbols](https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols)
  - **단일 문자** 이미지(45×45)들의 클래스별 폴더 → detection용 "여러 글자+박스"는 **합성으로 생성**.

### YOLO-det 학습 데이터가 없는 이유와 해결
기성 "손글씨 숫자+기호 detection" 데이터셋은 사실상 없음. 대신 단일 문자 데이터셋은 많음 →
**글자를 캔버스에 붙여 문자열 이미지를 만들면 붙인 위치가 곧 박스 라벨**(자동 생성, 라벨링 0초).

---

## 4. ⚠️ 데이터셋 준비 (함정 주의)

Kaggle에서 받은 `archive.zip` 구조:

```
archive.zip
├── data.rar              ← 전체 데이터(430MB). 진짜는 여기 있음
└── extracted_images/     ← 일부만 풀림 (! ( ) + , - 0 까지만). 불완전!
```

`extracted_images/`만 보면 **숫자가 0밖에 없어 보임** — 나머지(`1~9`, `=`, `pi`, `forward_slash` 등)는 전부 `data.rar` 안.

**해결 (로컬):**
```bash
brew install unar                 # rar 도구 (unrar/7z 없을 때)
unar data.rar                     # 전체 클래스 폴더 추출
```
**해결 (Colab):** 노트북에서 `data.rar`를 감지해 추출하도록 처리 (아래 5절).

추출된 실제 폴더명 (확인 완료, 전체 376,058장 / 82클래스):
- `0`~`9`, `-`, `=`, `pi` → 그대로 존재
- **분수 `/` = `forward_slash`** (÷는 `div`, ×는 `times` 로 따로 있으니 혼동 주의)
- → `synth_data.py`의 `CLASS_FOLDERS` 기본값이 이미 다 맞음. 수정 불필요.

---

## 5. 스크립트

### `synth_data.py` — 합성기
단일 문자 데이터셋 → YOLO-det 학습 데이터(이미지 + 박스 라벨) 자동 생성.
```bash
python synth_data.py --selftest                      # 데이터셋 없이 로직 검증
python synth_data.py --dataset <클래스폴더_경로> --out data --n 4000
```
- 회전·간격·겹침·노이즈로 촬영/붙여쓰기 상황 흉내.
- `CLASS_FOLDERS` dict가 우리 클래스 → 실제 폴더명 매핑. 폴더명 다르면 여기만 수정.

### `infer.py` — 추론 + 채점
답 칸 crop 하나 → 문자열 + 신뢰도 + O/X.
```bash
python infer.py --selftest                                        # 조립 로직 검증
python infer.py --model best.pt --image crop.jpg --answer -3/4    # 인식+채점
```
- 입력은 **답 칸 하나 crop** (전체 페이지 X — 모델이 그 형태로 학습됨).
- `agnostic_nms`로 같은 자리 중복 클래스 제거.

### `flip_train_colab.ipynb` — Colab 학습
`synth_data.py` 업로드 → xainano 다운로드 → 폴더명 자동 매핑 → 합성 → `yolov8n` 학습 → `best.pt` 다운로드.
- 런타임 GPU(T4) 필요.
- 학습 명령: `yolo detect train model=yolo26n.pt data=data/data.yaml imgsz=128 epochs=60 fliplr=0.0 mosaic=0.0`

**증강(augmentation) 주의:** 데이터 다양성은 합성기가 이미 만든다(회전·크기·간격모드·노이즈·블러). YOLO 학습에선 **좌우반전 `fliplr`을 반드시 꺼야 함** — `2/5`→`5/2`로 뒤집히고 숫자가 깨진다. `mosaic`도 끔(추론은 답 하나 크롭이라 4장 콜라주가 무의미). scale/translate/hsv 등 나머지 기본값은 무해하거나 도움.

---

## 6. 상태 / 남은 일

| 단계 | 상태 |
|---|---|
| 합성기 (`synth_data.py`) | ✅ 셀프테스트 통과 |
| 추론 (`infer.py`, `.pt` 소비) | ✅ 셀프테스트 통과 |
| Colab 학습 노트북 | ✅ (data.rar 자동 추출 반영됨) |
| xainano 데이터 준비 | ✅ 폴더명 확인 완료 (로컬 전체추출은 선택) |
| 모델 학습 (`best.pt`) | ⬜ ← 이게 되면 실제 손글씨 사진 테스트 가능 |
| 온디바이스 export (CoreML/TFLite) | ⏭ 확장 |

### 학습 시 주의
- 셀프테스트용 **폰트 글리프로 학습한 모델은 진짜 손글씨를 못 읽음** → 반드시 xainano(실제 손글씨)로 학습.
- 넣을 테스트 사진은 **답 칸 하나 crop** 형태여야 함.
