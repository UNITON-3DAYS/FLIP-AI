# FLIP 채점 서버 이미지.
# 의존성 레이어를 코드 레이어와 분리해, 코드만 바뀌면 무거운
# paddlepaddle/paddleocr 설치 레이어가 캐시로 재사용되게 한다.
FROM python:3.12-slim

# opencv-python-headless / paddle 런타임 시스템 의존.
#   libglib2.0-0 : opencv, libgomp1 : paddle/opencv OpenMP, libgl1 : 일부 opencv 경로
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgomp1 libgl1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

# 1) 의존성만 먼저 — requirements.txt가 안 바뀌면 이 레이어는 캐시된다.
COPY requirements.txt .
RUN pip install -r requirements.txt

# 2) 그 다음 코드. 코드 변경은 여기부터만 재빌드된다.
COPY . .

EXPOSE 8000

# 헬스체크: /health 200이면 healthy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
