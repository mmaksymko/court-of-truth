from starlette.testclient import TestClient

LONG_BODY = "речення про новини та важливі події дня " * 80


def test_health_reports_components(client: TestClient):
    body = client.get("/v1/health").json()
    assert body["status"] == "ok"
    assert body["components"]["forensics"]["ready"]
    assert "tribunal" not in body["components"]
    assert set(body["detectors"]) == {"jeansa", "clickbait"}


def test_live_and_ready(client: TestClient):
    assert client.get("/v1/health/live").json() == {"status": "ok"}
    assert client.get("/v1/health/ready").json() == {"status": "ready"}


def test_detectors_metadata(client: TestClient):
    ids = {item["id"] for item in client.get("/v1/detectors").json()}
    assert ids == {"jeansa", "clickbait"}


def test_analyze_echoes_request_id(client: TestClient):
    response = client.post(
        "/v1/analyze",
        json={"title": "Названо звичку", "text": LONG_BODY},
        headers={"x-request-id": "trace-42"},
    )
    assert response.status_code == 200
    assert response.headers["x-request-id"] == "trace-42"
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["risk"]["kind"] == "heuristic"


def test_analyze_supports_title_only(client: TestClient):
    body = client.post("/v1/analyze", json={"title": "один два три чотири п'ять"}).json()
    by_id = {result["id"]: result for result in body["results"]}
    assert by_id["jeansa"]["status"] == "skipped"
    assert by_id["clickbait"]["status"] == "ok"
