"""Streamlit evaluator dashboard for the PathSnap FastAPI backend."""

from __future__ import annotations

from typing import Any, Iterable
import math
import os
from urllib.parse import urlparse

import folium
import requests
import streamlit as st
from streamlit_folium import st_folium


DEFAULT_API_URL = os.getenv("PATHSNAP_API_URL", "http://localhost:8000")
REQUEST_TIMEOUT_SECONDS = 10


def normalize_api_url(url: str) -> str:
    """Trim whitespace and a trailing slash while preserving the configured host."""
    return url.strip().rstrip("/")


def get_connection(api_url: str, timeout: float = REQUEST_TIMEOUT_SECONDS) -> tuple[bool, str]:
    """Test the backend and fetch its available evaluator trip IDs."""
    try:
        response = requests.get(f"{normalize_api_url(api_url)}/trips", timeout=timeout)
        response.raise_for_status()
    except requests.Timeout:
        return False, "Connection timed out. Check that the backend is running and reachable."
    except requests.ConnectionError as exc:
        return False, f"Could not connect to the backend: {exc}"
    except requests.HTTPError as exc:
        detail = _response_detail(response)
        return False, f"Backend returned HTTP {response.status_code}: {detail}"
    except requests.RequestException as exc:
        return False, f"Backend request failed: {exc}"

    try:
        payload = response.json()
    except requests.exceptions.JSONDecodeError:
        return False, "The backend returned invalid JSON."

    trip_ids = payload.get("trip_ids") if isinstance(payload, dict) else None
    if (
        not isinstance(trip_ids, list)
        or not all(isinstance(trip_id, str) and bool(trip_id.strip()) for trip_id in trip_ids)
    ):
        return False, "The /trips response did not contain a valid trip_ids list."
    return True, "Connected"


def fetch_trip_ids(api_url: str, timeout: float = REQUEST_TIMEOUT_SECONDS) -> list[str]:
    """Fetch and validate trip IDs from the backend."""
    response = requests.get(f"{normalize_api_url(api_url)}/trips", timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    trip_ids = payload.get("trip_ids")
    if (
        not isinstance(trip_ids, list)
        or not all(isinstance(trip_id, str) and bool(trip_id.strip()) for trip_id in trip_ids)
    ):
        raise ValueError("The /trips response did not contain a valid trip_ids list.")
    return [trip_id.strip() for trip_id in trip_ids]


def predict_trip(api_url: str, trip_id: str, timeout: float = REQUEST_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Send a prediction request and validate the locked API response fields."""
    response = requests.post(
        f"{normalize_api_url(api_url)}/predict",
        json={"trip_id": trip_id},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("The /predict response was not a JSON object.")

    required = (
        "gps_points",
        "predicted_polylines",
        "top1",
        "continuity",
        "n_points",
        "latency_ms",
    )
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"The /predict response is missing fields: {', '.join(missing)}")
    return payload


def _response_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
    except requests.exceptions.JSONDecodeError:
        return response.text.strip() or "No response detail"
    if isinstance(payload, dict) and isinstance(payload.get("detail"), str):
        return payload["detail"]
    return str(payload)


def format_api_error(exc: Exception) -> str:
    """Convert requests and validation exceptions into useful UI messages."""
    if isinstance(exc, requests.Timeout):
        return "The prediction request timed out. Check the backend and try again."
    if isinstance(exc, requests.ConnectionError):
        return f"Could not reach the backend. Verify the laptop IP and port: {exc}"
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return (
            f"Backend returned HTTP {exc.response.status_code}: "
            f"{_response_detail(exc.response)}"
        )
    if isinstance(exc, requests.RequestException):
        return f"Backend request failed: {exc}"
    return str(exc)


def valid_gps_points(gps_points: Any) -> list[tuple[float, float]]:
    """Accept only finite, in-range lon/lat coordinate pairs."""
    if not isinstance(gps_points, list):
        return []

    coordinates: list[tuple[float, float]] = []
    for point in gps_points:
        if not isinstance(point, dict):
            continue
        lon = point.get("lon")
        lat = point.get("lat")
        if isinstance(lon, bool) or isinstance(lat, bool):
            continue
        try:
            lon_value = float(lon)
            lat_value = float(lat)
        except (TypeError, ValueError):
            continue
        if (
            -180.0 <= lon_value <= 180.0
            and -90.0 <= lat_value <= 90.0
        ):
            if math.isfinite(lon_value) and math.isfinite(lat_value):
                coordinates.append((lat_value, lon_value))
    return coordinates


def valid_polylines(polylines: Any) -> list[list[tuple[float, float]]]:
    """Return finite lon/lat polyline segments in the Folium lat/lon format."""
    if not isinstance(polylines, list):
        return []

    result: list[list[tuple[float, float]]] = []
    for polyline in polylines:
        if not isinstance(polyline, list):
            continue
        segment: list[tuple[float, float]] = []
        for coordinate in polyline:
            if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 2:
                continue
            if isinstance(coordinate[0], bool) or isinstance(coordinate[1], bool):
                continue
            try:
                lon = float(coordinate[0])
                lat = float(coordinate[1])
            except (TypeError, ValueError):
                continue
            if (
                math.isfinite(lon)
                and math.isfinite(lat)
                and -180.0 <= lon <= 180.0
                and -90.0 <= lat <= 90.0
            ):
                segment.append((lat, lon))
        if len(segment) >= 2:
            result.append(segment)
    return result


def build_map(
    gps_coordinates: Iterable[tuple[float, float]],
    predicted_polylines: Iterable[list[tuple[float, float]]],
) -> folium.Map:
    """Build a map that fits available coordinates and clearly separates data layers."""
    gps = list(gps_coordinates)
    polylines = list(predicted_polylines)
    all_coordinates = gps + [point for polyline in polylines for point in polyline]

    if all_coordinates:
        lats = [point[0] for point in all_coordinates]
        lons = [point[1] for point in all_coordinates]
        center = ((min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2)
    else:
        center = (20.5937, 78.9629)

    fmap = folium.Map(location=center, zoom_start=12, tiles="OpenStreetMap")

    gps_layer = folium.FeatureGroup(name="GPS observations", show=True)
    for lat, lon in gps:
        folium.CircleMarker(
            location=(lat, lon),
            radius=2,
            color="#1E88E5",
            fill=True,
            fill_color="#42A5F5",
            fill_opacity=0.65,
            weight=1,
        ).add_to(gps_layer)
    gps_layer.add_to(fmap)

    route_layer = folium.FeatureGroup(name="Predicted route", show=True)
    for polyline in polylines:
        folium.PolyLine(
            polyline,
            color="#D32F2F",
            weight=3,
            opacity=0.9,
        ).add_to(route_layer)
    route_layer.add_to(fmap)

    if all_coordinates:
        fmap.fit_bounds(
            [
                [min(point[0] for point in all_coordinates), min(point[1] for point in all_coordinates)],
                [max(point[0] for point in all_coordinates), max(point[1] for point in all_coordinates)],
            ]
        )

    folium.LayerControl(collapsed=False).add_to(fmap)
    legend = """
    <div style="
        position: fixed; bottom: 24px; left: 24px; z-index: 9999;
        background: white; padding: 10px 12px; border-radius: 8px;
        border: 1px solid #d0d5dd; box-shadow: 0 2px 8px rgba(0,0,0,0.12);
        font-family: sans-serif; font-size: 13px;">
      <div style="font-weight: 700; margin-bottom: 6px;">Legend</div>
      <div><span style="color:#1E88E5;">●</span>&nbsp; GPS observations</div>
      <div><span style="color:#D32F2F;">━</span>&nbsp; Predicted route</div>
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend))
    return fmap


def metric_value(payload: dict[str, Any], key: str) -> float | None:
    """Safely coerce an API metric to a finite, protocol-valid float."""
    raw_value = payload.get(key)
    if isinstance(raw_value, bool):
        return None
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value < 0.0:
        return None
    if key in ("top1", "continuity") and value > 1.0:
        return None
    return value


def observation_count(payload: dict[str, Any]) -> str:
    """Safely format n_points for the metric card."""
    value = payload.get("n_points")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "Unavailable"
    if isinstance(value, float) and not value.is_integer():
        return "Unavailable"
    if not math.isfinite(float(value)) or value < 0:
        return "Unavailable"
    return f"{int(value):,}"


st.set_page_config(
    page_title="PathSnap",
    page_icon="🗺️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1500px; }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 14px 16px;
        box-shadow: 0 1px 4px rgba(16,24,40,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("PathSnap")
st.subheader("Distributed Map-Matching & Route Reconstruction")

with st.sidebar:
    st.header("Backend")
    api_url = normalize_api_url(
        st.text_input(
            "API URL",
            value=DEFAULT_API_URL,
            help="Use the backend laptop IP, for example http://192.168.1.20:8000",
        )
    )

    parsed_url = urlparse(api_url)
    valid_url = bool(parsed_url.scheme in ("http", "https") and parsed_url.netloc)
    connection_ok = False
    connection_message = ""

    if st.button("Test connection", use_container_width=True):
        if not valid_url:
            st.error("Enter a valid HTTP or HTTPS backend URL.")
        else:
            connection_ok, connection_message = get_connection(api_url)
            if connection_ok:
                st.success(connection_message)
                try:
                    st.session_state["trip_ids"] = fetch_trip_ids(api_url)
                    st.session_state["trip_ids_api_url"] = api_url
                except (requests.RequestException, ValueError) as exc:
                    st.session_state["trip_ids"] = []
                    st.error(format_api_error(exc))
            else:
                st.error(connection_message)

    if not valid_url:
        st.warning("Waiting for a valid backend URL.")
    elif st.session_state.get("trip_ids_api_url") != api_url:
        try:
            st.session_state["trip_ids"] = fetch_trip_ids(api_url)
            st.session_state["trip_ids_api_url"] = api_url
        except (requests.RequestException, ValueError) as exc:
            st.session_state["trip_ids"] = []
            st.caption(f"Trip discovery unavailable: {format_api_error(exc)}")

    trip_ids = st.session_state.get("trip_ids", [])
    selected_trip = None
    if trip_ids:
        selected_trip = st.selectbox("Evaluator trip", trip_ids)
    else:
        st.selectbox("Evaluator trip", [], placeholder="No trips available")

    predict_clicked = st.button(
        "Predict",
        type="primary",
        use_container_width=True,
        disabled=not valid_url or not selected_trip,
    )

header_left, header_right = st.columns([3, 1])
with header_left:
    st.caption("Live evaluator dashboard")
with header_right:
    if valid_url:
        ok, message = get_connection(api_url)
        if ok:
            st.success("Backend connected")
        else:
            st.error("Backend unavailable")

if predict_clicked:
    if not selected_trip:
        st.warning("Select an evaluator trip before running prediction.")
    else:
        with st.spinner("Requesting server-side prediction and scoring..."):
            try:
                payload = predict_trip(api_url, selected_trip)
            except Exception as exc:
                st.error(format_api_error(exc))
            else:
                st.session_state["prediction"] = payload
else:
    if "prediction" not in st.session_state:
        st.session_state["prediction"] = None

payload = st.session_state.get("prediction")
if payload is None:
    st.info("Configure the backend, select a trip, and select **Predict**.")
else:
    top1 = metric_value(payload, "top1")
    continuity = metric_value(payload, "continuity")
    latency = metric_value(payload, "latency_ms")
    n_points_display = observation_count(payload)

    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Top-1 Accuracy",
        "Unavailable" if top1 is None else f"{top1:.2%}",
    )
    metric_columns[1].metric(
        "Route Continuity",
        "Unavailable" if continuity is None else f"{continuity:.2%}",
    )
    metric_columns[2].metric("GPS Observations", n_points_display)
    metric_columns[3].metric(
        "Inference Latency",
        "Unavailable" if latency is None else f"{latency:.1f} ms",
    )

    gps_coordinates = valid_gps_points(payload.get("gps_points"))
    predicted_polylines = valid_polylines(payload.get("predicted_polylines"))

    if not gps_coordinates and not predicted_polylines:
        st.warning("The response contained no valid GPS or predicted-route coordinates.")
    elif not gps_coordinates:
        st.warning("No valid GPS observations were available; showing the predicted route if possible.")

    fmap = build_map(gps_coordinates, predicted_polylines)
    st_folium(fmap, width=None, height=620, returned_objects=[])
