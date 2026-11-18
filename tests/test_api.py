import io
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api import main


@pytest.fixture(autouse=True)
def capture_api_logs(monkeypatch):
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    monkeypatch.setattr(main.logger, "handlers", [handler])
    return stream


def _write_trip(eval_dir: Path, trip_id: str = "trip_1", route=None) -> Path:
    trip_dir = eval_dir / trip_id
    trip_dir.mkdir(parents=True)
    (trip_dir / f"{trip_id}.track").write_text(
        "0.0001 0.0000 0\n0.0011 0.0000 1\n",
        encoding="utf-8",
    )
    (trip_dir / f"{trip_id}.nodes").write_text(
        "0.0000 0.0000\n0.0010 0.0000\n0.0020 0.0000\n",
        encoding="utf-8",
    )
    (trip_dir / f"{trip_id}.arcs").write_text(
        "0 1\n1 2\n",
        encoding="utf-8",
    )
    if route is not None:
        (trip_dir / f"{trip_id}.route").write_text(
            "".join(f"{arc_id}\n" for arc_id in route),
            encoding="utf-8",
        )
    return trip_dir


@pytest.fixture
def eval_dir_with_route(tmp_path, monkeypatch):
    eval_dir = tmp_path / "evaluation"
    eval_dir.mkdir()
    _write_trip(eval_dir, route=[0, 1])
    monkeypatch.setenv("EVAL_DIR", str(eval_dir))
    return eval_dir


@pytest.fixture
def eval_dir_without_route(tmp_path, monkeypatch):
    eval_dir = tmp_path / "evaluation"
    eval_dir.mkdir()
    _write_trip(eval_dir)
    monkeypatch.setenv("EVAL_DIR", str(eval_dir))
    return eval_dir


def _prediction(pred_arcs=(0, 1), gps=((0.0, 0.0001), (0.0, 0.0011))):
    return {
        "pred_arcs": list(pred_arcs),
        "n_points": len(pred_arcs),
        "gps": [list(point) for point in gps],
    }


def _set_pipeline(monkeypatch, predictor_type, score_trip):
    monkeypatch.setattr(
        main,
        "_pipeline_module",
        SimpleNamespace(Predictor=predictor_type, score_trip=score_trip),
    )
    monkeypatch.setattr(main, "_predictor", None)


class _FakePredictor:
    def predict_trip(self, trip_dir):
        return _prediction()


def test_get_trips_returns_only_valid_trip_ids(eval_dir_with_route):
    invalid_dir = eval_dir_with_route / "not.valid"
    invalid_dir.mkdir()
    (invalid_dir / "not.valid.track").write_text("", encoding="utf-8")

    response = TestClient(main.app).get("/trips")

    assert response.status_code == 200
    assert response.json() == {"trip_ids": ["trip_1"]}


def test_post_predict_response_contract_and_no_ground_truth_leakage(
    eval_dir_with_route, monkeypatch
):
    score_calls = []

    def score_trip(trip_dir, pred_arcs):
        score_calls.append((trip_dir, pred_arcs.tolist()))
        return {"top1_clean": 0.75, "top1_all": 0.5}

    _set_pipeline(monkeypatch, _FakePredictor, score_trip)
    response = TestClient(main.app).post("/predict", json={"trip_id": "trip_1"})

    assert response.status_code == 200
    payload = response.json()
    assert {
        "gps_points",
        "predicted_polylines",
        "top1",
        "top1_all",
        "continuity",
        "n_points",
        "latency_ms",
        "route_precision",
        "route_recall",
    } == payload.keys()
    assert payload["gps_points"] == [
        {"lon": 0.0001, "lat": 0.0},
        {"lon": 0.0011, "lat": 0.0},
    ]
    assert payload["top1"] == 0.75
    assert payload["top1_all"] == 0.5
    assert payload["continuity"] == 1.0
    assert payload["n_points"] == 2
    assert payload["latency_ms"] >= 0.0
    assert payload["route_precision"] == pytest.approx(1.0)
    assert payload["route_recall"] == pytest.approx(1.0)
    assert score_calls == [(str(eval_dir_with_route / "trip_1"), [0, 1])]
    assert not {
        "route",
        "reference_route",
        "reference_arcs",
        "labels",
        "label_arc",
        "reference_coordinates",
    }.intersection(payload)


def test_invalid_trip_id_and_path_traversal_return_404(eval_dir_with_route):
    client = TestClient(main.app)

    for trip_id in ("../trip_1", "..\\trip_1", "bad id", "missing_trip"):
        response = client.post("/predict", json={"trip_id": trip_id})
        assert response.status_code == 404


def test_missing_model_artifact_returns_503_without_fallback(
    eval_dir_with_route, monkeypatch
):
    class MissingArtifactPredictor:
        def __init__(self):
            raise FileNotFoundError("synthetic model artifact missing")

    _set_pipeline(monkeypatch, MissingArtifactPredictor, lambda *_: None)
    response = TestClient(main.app).post("/predict", json={"trip_id": "trip_1"})

    assert response.status_code == 503
    assert "model initialization failed" in response.json()["detail"].lower()
    assert "synthetic model artifact missing" not in response.text


def test_missing_reference_route_returns_null_metrics(
    eval_dir_without_route, monkeypatch
):
    _set_pipeline(monkeypatch, _FakePredictor, lambda *_: None)
    response = TestClient(main.app).post("/predict", json={"trip_id": "trip_1"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["top1"] is None
    assert payload["top1_all"] is None
    assert payload["route_precision"] is None
    assert payload["route_recall"] is None


def test_predictor_initialization_is_cached(eval_dir_without_route, monkeypatch):
    instances = []

    class CountingPredictor(_FakePredictor):
        def __init__(self):
            instances.append(self)

    _set_pipeline(monkeypatch, CountingPredictor, lambda *_: None)
    client = TestClient(main.app)
    first = client.post("/predict", json={"trip_id": "trip_1"})
    second = client.post("/predict", json={"trip_id": "trip_1"})

    assert first.status_code == second.status_code == 200
    assert len(instances) == 1


def test_symlinked_eval_dir_is_rejected(eval_dir_with_route, monkeypatch):
    original_is_symlink = Path.is_symlink

    def is_symlink(path):
        return path == eval_dir_with_route or original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", is_symlink)
    assert main._resolved_evaluator_dir() is None
    assert TestClient(main.app).get("/trips").status_code == 503


def test_symlinked_trip_directory_is_rejected(eval_dir_with_route, monkeypatch):
    original_is_symlink = Path.is_symlink
    symlinked_trip = eval_dir_with_route / "trip_1"

    def is_symlink(path):
        return path == symlinked_trip or original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", is_symlink)

    assert main._trip_folder("trip_1") is None
    assert main._available_trip_ids() == []


def test_junctioned_trip_directory_is_rejected(
    eval_dir_with_route, monkeypatch
):
    if not hasattr(Path, "is_junction"):
        pytest.skip("Path.is_junction is unavailable on this Python version")

    original_is_junction = Path.is_junction
    junction = eval_dir_with_route / "trip_1"

    def is_junction(path):
        return path == junction or original_is_junction(path)

    monkeypatch.setattr(Path, "is_junction", is_junction)

    assert main._trip_folder("trip_1") is None
    assert main._available_trip_ids() == []


def test_cors_allows_configured_origin_and_rejects_unapproved_origin():
    cors_middleware = next(
        middleware
        for middleware in main.app.user_middleware
        if middleware.cls is CORSMiddleware
    )
    allowed_origin = cors_middleware.kwargs["allow_origins"][0]
    client = TestClient(main.app)

    allowed = client.options(
        "/predict",
        headers={
            "Origin": allowed_origin,
            "Access-Control-Request-Method": "POST",
        },
    )
    rejected = client.options(
        "/predict",
        headers={
            "Origin": "http://unapproved.invalid:8501",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert allowed.headers.get("access-control-allow-origin") == allowed_origin
    assert "access-control-allow-origin" not in rejected.headers


def test_crlf_request_path_cannot_create_extra_log_entries(capture_api_logs):
    response = TestClient(main.app).get("/unexpected%0D%0Aforged-entry")

    assert response.status_code == 404
    lines = capture_api_logs.getvalue().splitlines()
    assert len(lines) == 1
    assert "path=<unmatched>" in lines[0]
    assert "forged-entry" not in capture_api_logs.getvalue()
