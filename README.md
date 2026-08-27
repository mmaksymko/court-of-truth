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
uv sync --locked --group dev --group training --group ui
uv run python -m court
```

In a second terminal, start the product UI:

```bash
uv run --no-sync streamlit run streamlit_app.py
```

Open <http://127.0.0.1:8501>. The UI calls the local API at
`http://127.0.0.1:8000` by default; override it with `COURT_API_URL` when needed.

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
docker compose up --build app ui
```

The compose stack bind-mounts `./artifacts`, so it uses the same reviewed model
artifacts as the local runtime. If that directory has not been provisioned yet,
train the models into it first:

```bash
docker compose --profile train run --rm train
docker compose up --build app ui
```

The API is available at <http://127.0.0.1:8000> and the UI at
<http://127.0.0.1:8501>.

## Security boundary

Place the app behind an authenticated, rate-limited gateway before exposing it
publicly. `COURT_OPERATION_RATE_PER_MINUTE` and `COURT_OPERATION_CONCURRENCY`
bound the compute-intensive endpoint; in-process limits also cap request bytes
and concurrent operations.

## Quality gate

```bash
bash scripts/check.sh
```

Reproduces the development, training, and UI dependency groups, then runs Ruff
formatting/linting, mypy, a Streamlit availability check, branch coverage, and
the full test suite.
