"""Interactive Folium maps for the report (§2 EDA and §6 geographical visualisation).

Each map has separately toggleable layers (LayerControl) and a fixed legend:
  * Road network   : grey thin lines, arcs within NET_BUFFER_M of the GPS track (the relevant network geometry)
  * Raw GPS        : blue dots, the original .track observations
  * Reference route: green thick line, arcs of the supplied .route file
  * Predicted route: red line, HMM (learned transitions) decoded arcs, the deployed model's output
  * Classifier-only: orange dashed line, M7 argmax per point before HMM (hidden by default)
  * Errors         : black dots, clean points where the HMM prediction != reference label (hidden by default)
Predictions are read from results/preds_student_test_hmm.parquet (produced by src/run_hmm.py), so the
maps show the real experiment output. Evaluator-test data is refused on purpose.

Usage:
  python src/make_maps.py                                   # default report set (below)
  python src/make_maps.py --trips 00000040 00000077         # student-test trips with predictions
  python src/make_maps.py --eda train 00000054              # EDA map (no predictions) for any train/student-test trip
Outputs: maps/route_<partition>_<trip>.html, maps/eda_<partition>_<trip>.html
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import folium
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from io_utils import load_trip          # noqa: E402
from geo import to_xy                   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a)
NET_BUFFER_M = 150.0
MAX_GPS_MARKERS = 4000
DEFAULT_ROUTE_TRIPS = ["00000040", "00000077", "00000056"]   # best-matched, hardest (direction), parallel/junction cases
DEFAULT_EDA = [("train", "00000054"), ("train", "00000001")]  # reference-route anomaly, typical trip

COLORS = {"network": "#9e9e9e", "gps": "#1f77b4", "reference": "#2ca02c", "predicted": "#d62728",
          "classifier": "#ff7f0e", "errors": "#000000"}


def segs(arc_ids, arcs, nodes):
    """Arc ids -> list of [[lat, lon], [lat, lon]] segments (consecutive duplicates removed)."""
    out, prev = [], None
    for a in arc_ids:
        a = int(a)
        if a == prev:
            continue
        prev = a
        u, v = arcs[a]
        out.append([[nodes[u, 1], nodes[u, 0]], [nodes[v, 1], nodes[v, 0]]])
    return out


def nearby_arcs(trip):
    tr, nodes, arcs = trip["track"], trip["nodes"], trip["arcs"]
    lon0, lat0 = tr.lon.mean(), tr.lat.mean()
    gx, gy = to_xy(tr.lon.values, tr.lat.values, lon0, lat0)
    nx, ny = to_xy(nodes[:, 0], nodes[:, 1], lon0, lat0)
    d, _ = cKDTree(np.c_[gx, gy]).query(np.c_[nx, ny], distance_upper_bound=NET_BUFFER_M)
    near = np.isfinite(d)
    return np.flatnonzero(near[arcs[:, 0]] | near[arcs[:, 1]])


def legend(title, items):
    rows = "".join(
        f'<div style="margin:2px 0"><span style="display:inline-block;width:22px;height:{h}px;'
        f'background:{c};vertical-align:middle;margin-right:6px;{extra}"></span>{name}</div>'
        for name, c, h, extra in items)
    return folium.Element(
        f'<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:10px 12px;'
        f'border:1px solid #888;border-radius:6px;font:13px Arial;box-shadow:0 1px 4px rgba(0,0,0,.3)">'
        f'<b>{title}</b>{rows}<div style="margin-top:4px;color:#555">Toggle layers: top-right control</div></div>')


def base_map(trip):
    tr = trip["track"]
    m = folium.Map(tiles="cartodbpositron", control_scale=True)
    m.fit_bounds([[tr.lat.min(), tr.lon.min()], [tr.lat.max(), tr.lon.max()]])
    return m


def add_network(m, trip):
    ids = nearby_arcs(trip)
    fg = folium.FeatureGroup(name=f"Road network ({len(ids)} arcs within {int(NET_BUFFER_M)} m)", show=True)
    folium.PolyLine(segs(ids, trip["arcs"], trip["nodes"]), color=COLORS["network"], weight=1, opacity=0.6).add_to(fg)
    fg.add_to(m)


def add_gps(m, trip):
    tr = trip["track"]
    step = int(np.ceil(len(tr) / MAX_GPS_MARKERS))
    sub = tr.iloc[::step]
    label = f"Raw GPS observations ({len(tr)} points" + (f", every {step}th drawn)" if step > 1 else ")")
    fg = folium.FeatureGroup(name=label, show=True)
    for lat, lon, t in zip(sub.lat, sub.lon, sub.t):
        folium.CircleMarker([lat, lon], radius=2, color=COLORS["gps"], fill=True, fill_opacity=0.8,
                            weight=0, tooltip=f"t={t:.0f}s").add_to(fg)
    fg.add_to(m)


def add_reference(m, trip):
    fg = folium.FeatureGroup(name=f"Reference route ({len(trip['route'])} arcs)", show=True)
    folium.PolyLine(segs(trip["route"], trip["arcs"], trip["nodes"]), color=COLORS["reference"],
                    weight=7, opacity=0.55).add_to(fg)
    fg.add_to(m)


def route_map(tid, preds):
    trip = load_trip(P("data", "raw", "student_test", tid))
    p = preds[preds.trip_id == tid].sort_values("point_idx")
    clean = ~p.ref_uncertain.values
    acc_h = (p.pred_hmm_learned.values[clean] == p.label_arc.values[clean]).mean()
    acc_c = (p.pred_classifier.values[clean] == p.label_arc.values[clean]).mean()
    m = base_map(trip)
    add_network(m, trip)
    add_reference(m, trip)
    fg = folium.FeatureGroup(name=f"Predicted route: M7 + HMM learned (Top-1 {acc_h:.1%})", show=True)
    folium.PolyLine(segs(p.pred_hmm_learned, trip["arcs"], trip["nodes"]), color=COLORS["predicted"],
                    weight=3, opacity=0.9).add_to(fg)
    fg.add_to(m)
    fg = folium.FeatureGroup(name=f"Classifier-only M7 (Top-1 {acc_c:.1%})", show=False)
    folium.PolyLine(segs(p.pred_classifier, trip["arcs"], trip["nodes"]), color=COLORS["classifier"],
                    weight=3, opacity=0.9, dash_array="6,6").add_to(fg)
    fg.add_to(m)
    add_gps(m, trip)
    err = p[clean & (p.pred_hmm_learned.values != p.label_arc.values)]
    fg = folium.FeatureGroup(name=f"HMM errors ({len(err)} clean points)", show=False)
    tr = trip["track"]
    for i in err.point_idx.values:
        folium.CircleMarker([tr.lat.iat[i], tr.lon.iat[i]], radius=3, color=COLORS["errors"], fill=True,
                            weight=0, fill_opacity=0.9, tooltip=f"point {i}").add_to(fg)
    fg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(legend(f"Student-test trip {tid}", [
        ("Road network", COLORS["network"], 2, ""), ("Raw GPS observations", COLORS["gps"], 6, "border-radius:3px;"),
        ("Reference route", COLORS["reference"], 7, "opacity:.55;"), ("Predicted route (HMM)", COLORS["predicted"], 3, ""),
        ("Classifier-only (hidden)", COLORS["classifier"], 3, ""), ("HMM errors (hidden)", COLORS["errors"], 6, "border-radius:3px;")]))
    out = P("maps", f"route_student_test_{tid}.html")
    m.save(out)
    return out, acc_c, acc_h


def eda_map(partition, tid):
    if partition not in ("train", "student_test"):
        raise SystemExit("Only train / student_test are allowed (evaluator data stays untouched).")
    trip = load_trip(P("data", "raw", partition, tid))
    m = base_map(trip)
    add_network(m, trip)
    add_reference(m, trip)
    add_gps(m, trip)
    items = [("Road network", COLORS["network"], 2, ""), ("Raw GPS observations", COLORS["gps"], 6, "border-radius:3px;"),
             ("Reference route", COLORS["reference"], 7, "opacity:.55;")]
    pts_file = P("data", "processed", f"{partition}_points.parquet")
    if os.path.exists(pts_file):  # highlight GPS points the reference route does not cover (ref_uncertain, > 50 m)
        pts = pd.read_parquet(pts_file, columns=["trip_id", "point_idx", "lat", "lon", "ref_uncertain"])
        unc = pts[(pts.trip_id == tid) & pts.ref_uncertain]
        if len(unc):
            step = int(np.ceil(len(unc) / MAX_GPS_MARKERS))
            fg = folium.FeatureGroup(name=f"GPS not covered by reference (>50 m): {len(unc)} points", show=True)
            for lat, lon in zip(unc.lat.values[::step], unc.lon.values[::step]):
                folium.CircleMarker([lat, lon], radius=2, color="#9467bd", fill=True, weight=0,
                                    fill_opacity=0.9).add_to(fg)
            fg.add_to(m)
            items.append(("GPS not covered by reference", "#9467bd", 6, "border-radius:3px;"))
    folium.LayerControl(collapsed=False).add_to(m)
    m.get_root().html.add_child(legend(f"EDA: {partition} trip {tid}", items))
    out = P("maps", f"eda_{partition}_{tid}.html")
    m.save(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trips", nargs="*", help="student-test trip ids for prediction maps")
    ap.add_argument("--eda", nargs=2, action="append", metavar=("PARTITION", "TRIP"), help="EDA map without predictions")
    a = ap.parse_args()
    os.makedirs(P("maps"), exist_ok=True)
    route_trips = a.trips if a.trips is not None else ([] if a.eda else DEFAULT_ROUTE_TRIPS)
    eda = a.eda if a.eda else ([] if a.trips is not None else DEFAULT_EDA)
    if route_trips:
        preds = pd.read_parquet(P("results", "preds_student_test_hmm.parquet"))
        for tid in route_trips:
            out, ac, ah = route_map(tid, preds)
            print(f"{out}  (classifier Top-1 {ac:.4f}, HMM Top-1 {ah:.4f})")
    for part, tid in eda:
        print(eda_map(part, tid))


if __name__ == "__main__":
    main()
