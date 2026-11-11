# ---- frontend build -----------------------------------------------------------------------
FROM node:22-alpine AS ui
WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
# vite.config.ts writes to ../static, i.e. /static inside this stage
RUN npm run build

# ---- backend ------------------------------------------------------------------------------
FROM python:3.12-slim AS app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken \
    QDRANT_PATH=/data/qdrant \
    STATIC_DIR=/app/static
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install . \
    # Bake the tokenizer tables into the image so startup never needs an outbound download.
    && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

COPY data ./data
COPY --from=ui /static ./static

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data/qdrant && chown -R appuser:appuser /data /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request,os;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/health')" || exit 1

CMD ["sh", "-c", "uvicorn policypilot.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
