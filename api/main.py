"""FastAPI entry point for the PathSnap route-prediction pipeline."""

from __future__ import annotations

import json
import importlib
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from io_utils import load_trip  # noqa: E402


LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "api.log"

logger = logging.getLogger("pathsnap.api")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


app = FastAPI(title="PathSnap API", version="0.1.0")

# Explicit origins keep frontend access predictable; credentials are not required.
def _cors_origins() -> list[str]:
    configured = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:8501,http://127.0.0.1:8501",
    )
    origins: list[str] = []
    for entry in configured.split(","):
        origin = entry.strip().rstrip("/")
        if origin and "*" not in origin and origin.startswith(("http://", "https://")):
            origins.append(origin)
    return origins


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


TRIP_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
TRIP_ID_MAX_LENGTH = 64
_pipeline_lock = threading.Lock()
_pipeline_module: Any = None
_predictor: Any = None


class PredictRequest(BaseModel):
    trip_id: str = Field(..., min_length=1, max_length=TRIP_ID_MAX_LENGTH)


def _evaluator_dir() -> Optional[Path]:
    value = os.getenv("EVAL_DIR")
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT_DIR / path


def _is_symlink_or_junction(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        return bool(is_junction()) if callable(is_junction) else False
    except (OSError, RuntimeError):
        return True


def _resolved_evaluator_dir() -> Optional[Path]:
    eval_dir = _evaluator_dir()
    if eval_dir is None:
        return None
    try:
        if _is_symlink_or_junction(eval_dir):
            return None
        resolved = eval_dir.resolve(strict=True)
    except (OSError, RuntimeError):
        return None
    return resolved if resolved.is_dir() else None


def _safe_log_trip_id(trip_id: str) -> str:
    """Render an untrusted trip ID as a bounded, single-line log representation."""
    text = "".join(
        character if character.isprintable() and character not in "\r\n" else " "
        for character in trip_id
    )
    return repr(text[:TRIP_ID_MAX_LENGTH])


def _safe_log_path(request: Request) -> str:
    """Log only known route templates; never copy an untrusted request path."""
    path = request.scope.get("path")
    if path == "/trips":
        return "/trips"
    if path == "/predict":
        return "/predict"
    return "<unmatched>"


def _safe_log_method(method: str) -> str:
    """Keep the request method useful while excluding control characters."""
    return "".join(
        character if character.isprintable() and character not in "\r\n" else "?"
        for character in method[:32]
    ) or "-"


def _trip_folder(trip_id: str) -> Optional[Path]:
    eval_dir = _resolved_evaluator_dir()
    if eval_dir is None:
        return None

    requested = trip_id.strip()
    if (
        not requested
        or len(requested) > TRIP_ID_MAX_LENGTH
        or not TRIP_ID_PATTERN.fullmatch(requested)
        or "/" in requested
        or "\\" in requested
        or ".." in requested
    ):
        return None

    try:
        folder = eval_dir / requested
        if _is_symlink_or_junction(folder):
            return None

        folder_resolved = folder.resolve(strict=True)
        if folder_resolved.parent != eval_dir:
            return None
        if not folder_resolved.is_dir() or _is_symlink_or_junction(folder_resolved):
            return None

        for suffix in ("track", "nodes", "arcs"):
            file_path = folder_resolved / f"{requested}.{suffix}"
            if _is_symlink_or_junction(file_path) or not file_path.is_file():
                return None
            file_resolved = file_path.resolve(strict=True)
            if file_resolved.parent != folder_resolved or not file_resolved.is_file():
                return None

        route_path = folder_resolved / f"{requested}.route"
        if route_path.exists():
            if _is_symlink_or_junction(route_path) or not route_path.is_file():
                return None
            route_resolved = route_path.resolve(strict=True)
            if route_resolved.parent != folder_resolved or not route_resolved.is_file():
                return None
    except (OSError, RuntimeError):
        return None
    return folder_resolved


def _available_trip_ids() -> list[str]:
    eval_dir = _evaluator_dir()
    if eval_dir is None:
        raise HTTPException(status_code=503, detail="EVAL_DIR is not configured")
    eval_dir_resolved = _resolved_evaluator_dir()
    if eval_dir_resolved is None:
        raise HTTPException(
            status_code=503,
            detail="EVAL_DIR does not exist or is not an allowed directory",
        )

    ids: list[str] = []
    for folder in sorted(eval_dir_resolved.iterdir(), key=lambda item: item.name):
        if not folder.is_dir() or _is_symlink_or_junction(folder):
            continue
        trip_id = folder.name
        if (
            len(trip_id) > TRIP_ID_MAX_LENGTH
            or not TRIP_ID_PATTERN.fullmatch(trip_id)
        ):
            continue
        required = (folder / f"{trip_id}.track", folder / f"{trip_id}.nodes", folder / f"{trip_id}.arcs")
        try:
            folder_resolved = folder.resolve(strict=True)
            eval_dir_resolved = eval_dir.resolve(strict=True)
            if folder_resolved.parent != eval_dir_resolved:
                continue
        except (OSError, RuntimeError):
            continue
        valid_files = True
        for path in required:
            try:
                if _is_symlink_or_junction(path) or not path.is_file():
                    valid_files = False
                    break
                if path.resolve(strict=True).parent != folder_resolved:
                    valid_files = False
                    break
            except (OSError, RuntimeError):
                valid_files = False
                break
        if valid_files:
            ids.append(trip_id)
    return ids


def _route_continuity(pred_arcs: np.ndarray, arcs: np.ndarray) -> float:
    """Fraction of consecutive route pairs connected by directed endpoints."""
    predicted = np.asarray(pred_arcs)
    arc_values = np.asarray(arcs)
    if predicted.ndim != 1 or arc_values.ndim != 2 or arc_values.shape[1] != 2:
        return 0.0
    if len(predicted) == 0:
        return 0.0

    valid_ids: list[Optional[int]] = []
    for arc_id in predicted:
        if (
            isinstance(arc_id, (bool, np.bool_))
            or not isinstance(arc_id, (int, np.integer))
            or arc_id < 0
            or arc_id >= len(arc_values)
        ):
            valid_ids.append(None)
        else:
            valid_ids.append(int(arc_id))

    if len(valid_ids) == 1:
        return 1.0 if valid_ids[0] is not None else 0.0

    connected = 0
    for left, right in zip(valid_ids, valid_ids[1:]):
        if (
            left is not None
            and right is not None
            and arc_values[left, 1] == arc_values[right, 0]
        ):
            connected += 1
    return float(connected / (len(valid_ids) - 1))


def _predicted_polylines(trip: Dict[str, Any], pred_arcs: np.ndarray) -> list[list[list[float]]]:
    """Convert predicted arc ids to coordinate polylines for the UI."""
    nodes = trip["nodes"]
    arcs = trip["arcs"]
    polylines: list[list[list[float]]] = []

    for arc_id in pred_arcs:
        if int(arc_id) < 0:
            continue
        from_node, to_node = arcs[int(arc_id)]
        start = nodes[int(from_node)]
        end = nodes[int(to_node)]
        polylines.append(
            [
                [float(start[0]), float(start[1])],
                [float(end[0]), float(end[1])],
            ]
        )
    return polylines


def _get_pipeline() -> tuple[Any, Any]:
    """Load pipeline code and model artifacts once per API process."""
    global _pipeline_module, _predictor
    with _pipeline_lock:
        try:
            if _pipeline_module is None:
                _pipeline_module = importlib.import_module("pipeline")
            if _predictor is None:
                _predictor = _pipeline_module.Predictor()
        except Exception as exc:
            logger.exception("PathSnap predictor initialization failed")
            raise HTTPException(
                status_code=503,
                detail="Prediction model initialization failed; check server logs and model artifacts.",
            ) from exc
    return _pipeline_module, _predictor


def predict_trip(trip_dir: str) -> Dict[str, Any]:
    """Run the deployed model and score its unbridged per-observation predictions."""
    pipeline, predictor = _get_pipeline()
    try:
        prediction = predictor.predict_trip(trip_dir)
    except Exception as exc:
        logger.exception("PathSnap prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed; check server logs for details.",
        ) from exc

    if not isinstance(prediction, dict) or "pred_arcs" not in prediction:
        logger.error("Pipeline returned an invalid prediction payload")
        raise HTTPException(status_code=500, detail="Prediction pipeline returned an invalid result.")

    pred_arcs = np.asarray(prediction["pred_arcs"])
    n_points = prediction.get("n_points")
    if (
        pred_arcs.ndim != 1
        or isinstance(n_points, (bool, np.bool_))
        or not isinstance(n_points, (int, np.integer))
        or n_points < 0
        or len(pred_arcs) != n_points
    ):
        logger.error("Pipeline returned inconsistent per-observation predictions")
        raise HTTPException(status_code=500, detail="Prediction pipeline returned an invalid result.")

    try:
        score = pipeline.score_trip(trip_dir, pred_arcs)
    except Exception as exc:
        logger.exception("Server-side prediction scoring failed")
        raise HTTPException(
            status_code=500,
            detail="Server-side scoring failed; check server logs for details.",
        ) from exc

    try:
        trip = load_trip(trip_dir, with_route=True)
        from bridge import bridge_route, route_precision_recall  # noqa: E402

        bridged_arcs = bridge_route(pred_arcs, trip["arcs"], trip["nodes"])
        route_precision = None
        route_recall = None
        if trip["route"] is not None:
            route_precision, route_recall = route_precision_recall(
                bridged_arcs,
                trip["route"],
                trip["arcs"],
                trip["nodes"],
            )
    except Exception as exc:
        logger.exception("N7 route bridging or server-side route scoring failed")
        raise HTTPException(
            status_code=500,
            detail="Route reconstruction failed; check server logs for details.",
        ) from exc

    gps = prediction.get("gps")
    if not isinstance(gps, (list, np.ndarray)) or len(gps) != n_points:
        logger.error("Pipeline returned invalid GPS coordinates")
        raise HTTPException(status_code=500, detail="Prediction pipeline returned an invalid result.")
    try:
        gps_points = [
            {"lon": float(point[1]), "lat": float(point[0])}
            for point in gps
            if len(point) == 2
        ]
    except (TypeError, ValueError, IndexError):
        logger.exception("Pipeline returned malformed GPS coordinates")
        raise HTTPException(status_code=500, detail="Prediction pipeline returned invalid coordinates.")
    if len(gps_points) != n_points:
        logger.error("Pipeline returned an incomplete GPS coordinate sequence")
        raise HTTPException(status_code=500, detail="Prediction pipeline returned invalid coordinates.")

    return {
        "gps_points": gps_points,
        "predicted_polylines": _predicted_polylines(trip, np.asarray(bridged_arcs, dtype=np.int64)),
        "top1": score.get("top1_clean") if score is not None else None,
        "top1_all": score.get("top1_all") if score is not None else None,
        "continuity": _route_continuity(np.asarray(bridged_arcs, dtype=np.int64), trip["arcs"]),
        "n_points": int(n_points),
        "route_precision": route_precision,
        "route_recall": route_recall,
    }


@app.middleware("http")
async def log_requests(request: Request, call_next: Any) -> JSONResponse:
    started = time.perf_counter()
    trip_id = "-"
    safe_method = _safe_log_method(request.method)
    safe_path = _safe_log_path(request)

    if request.method == "POST" and request.scope.get("path") == "/predict":
        try:
            body = await request.body()
            parsed = json.loads(body.decode("utf-8"))
            if isinstance(parsed, dict) and isinstance(parsed.get("trip_id"), str):
                trip_id = parsed["trip_id"].strip() or "-"
        except (UnicodeDecodeError, json.JSONDecodeError):
            trip_id = "-"

    try:
        response = await call_next(request)
    except Exception:
        latency_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "method=%s path=%s trip_id=%s n_points=0 top1=null latency_ms=%.3f status=500",
            safe_method,
            safe_path,
            _safe_log_trip_id(trip_id),
            latency_ms,
        )
        raise

    latency_ms = (time.perf_counter() - started) * 1000.0
    n_points = response.headers.get("X-PathSnap-N-Points", "0")
    top1 = response.headers.get("X-PathSnap-Top1", "null")
    trip_header = response.headers.get("X-PathSnap-Trip-ID")
    if trip_header:
        trip_id = trip_header

    logger.info(
        "method=%s path=%s trip_id=%s n_points=%s top1=%s latency_ms=%.3f status=%d",
        safe_method,
        safe_path,
        _safe_log_trip_id(trip_id),
        n_points,
        top1,
        latency_ms,
        response.status_code,
    )
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, str) else "Request failed"
    return JSONResponse(status_code=exc.status_code, content={"detail": detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred"},
    )


@app.get("/trips")
def get_trips() -> Dict[str, list[str]]:
    """Return only evaluator trip IDs; never return routes or labels."""
    return {"trip_ids": _available_trip_ids()}


@app.post("/predict")
def post_predict(payload: PredictRequest) -> JSONResponse:
    """Validate and score a requested evaluator trip on the server."""
    trip_id = payload.trip_id.strip()
    folder = _trip_folder(trip_id)
    if folder is None:
        eval_dir = _evaluator_dir()
        if eval_dir is None:
            raise HTTPException(status_code=503, detail="EVAL_DIR is not configured")
        if _resolved_evaluator_dir() is None:
            raise HTTPException(
                status_code=503,
                detail="EVAL_DIR does not exist or is not an allowed directory",
            )
        raise HTTPException(status_code=404, detail="Trip ID not found")

    started = time.perf_counter()
    result = predict_trip(str(folder))
    latency_ms = (time.perf_counter() - started) * 1000.0
    result["latency_ms"] = latency_ms

    headers = {
        "X-PathSnap-Trip-ID": trip_id,
        "X-PathSnap-N-Points": str(result["n_points"]),
        "X-PathSnap-Top1": (
            "null" if result["top1"] is None else f"{result['top1']:.6f}"
        ),
    }
    return JSONResponse(content=result, headers=headers)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
