FROM python:3.12-slim@sha256:57cd7c3a7a273101a6485ba99423ee568157882804b1124b4dd04266317710de AS base
COPY --from=ghcr.io/astral-sh/uv:0.7.17@sha256:68a26194ea8da0dbb014e8ae1d8ab08a469ee3ba0f4e2ac07b8bb66c0f8185c1 /uv /bin/uv
ENV UV_PYTHON_DOWNLOADS=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
RUN useradd --create-home --system app

FROM base AS app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-group training --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-group training \
    && mkdir -p /app/artifacts \
    && chown -R app:app /app/artifacts
ENV PATH="/app/.venv/bin:$PATH" COURT_ARTIFACTS_DIR=/app/artifacts
USER app
HEALTHCHECK --start-period=40s --interval=30s --timeout=5s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/health/live', timeout=3)"]
CMD ["python", "-m", "court"]

FROM base AS train
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --group training --no-install-project
COPY src ./src
COPY training ./training
COPY data ./data
RUN uv sync --frozen --no-dev --group training \
    && mkdir -p /app/artifacts \
    && chown -R app:app /app/artifacts
ENV PATH="/app/.venv/bin:$PATH"
USER app
CMD ["python", "-m", "training.train", "--out", "/app/artifacts"]
