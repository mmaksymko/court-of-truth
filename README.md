# Court Criminalist

Standalone criminalist (forensic) layer of Court of Truth. A FastAPI application
that runs four local, key-free ML detectors over Ukrainian news and returns
explainable signals. No OpenAI, no tribunal. The tribunal (LLM prosecutor,
advocate, and judge) is a separate layer that will be re-integrated on top of
this service later.

Detectors: `ai_generated`, `clickbait`, `jeansa`, `mt_translation`.

## Run locally

Python 3.13, `uv`, and trained artifacts under `artifacts/` are required:

```bash
uv sync --group dev --group training
uv run python -m court
```

The key-free forensic endpoint:

```bash
curl -X POST http://localhost:8000/v1/analyze \
  -H 'content-type: application/json' \
  -d '{"title":"Заголовок","text":"Текст матеріалу"}'
```

`GET /v1/health`, `/v1/health/live`, `/v1/health/ready`, and `/v1/detectors`
report status and detector metadata. `/v1/analyze` and `/v1/detectors` need only
the loaded registry.

## Docker

```bash
docker compose up --build app
```

## Security boundary

Place the app behind an authenticated, rate-limited gateway before exposing it
publicly. `COURT_OPERATION_RATE_PER_MINUTE` and `COURT_OPERATION_CONCURRENCY`
bound the compute-intensive endpoint; in-process limits also cap request bytes
and concurrent operations.

## Quality gate

```bash
bash scripts/check.sh
```

Runs Ruff formatting/linting, mypy, branch coverage, and the full test suite.
