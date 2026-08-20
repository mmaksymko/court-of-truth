import uvicorn

from court.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "court.api.app:app",
        host="0.0.0.0",
        port=settings.port,
        limit_concurrency=settings.uvicorn_limit_concurrency,
    )


if __name__ == "__main__":
    main()
