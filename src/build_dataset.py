"""Build the candidate-level training tables.

Usage:  python src/build_dataset.py            (both partitions)
Outputs:
  data/processed/{train,student_test}_cands.parquet   one row per (GPS point, candidate arc)
  data/processed/{train,student_test}_points.parquet  one row per GPS point (labels, coords)
  results/dataset_summary.csv                        per-trip quality stats + naive baselines
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from io_utils import list_trips, load_trip          # noqa: E402
from geo import TripGeometry                          # noqa: E402
from candidates import generate                       # noqa: E402
from features import build_features, FEATURES         # noqa: E402
from labels import make_labels                        # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "processed")


def featurize_trip(trip):
    """GPS + network only (no route) -> (geometry, candidate feature DataFrame). Used by the API too."""
    g = TripGeometry(trip)
    c = generate(g)
    return g, build_features(g, c)


def process(args):
    partition, tid = args
    t0 = time.time()
    trip = load_trip(os.path.join(RAW, partition, tid))
    g, df = featurize_trip(trip)
    label_arc, label_pos, label_dist, unc = make_labels(g, trip["route"])

    pi = df["point_idx"].values
    df.insert(0, "trip_id", tid)
    df["t"] = g.t[pi]
    df["label_arc"] = label_arc[pi]
    df["ref_uncertain"] = unc[pi]
    df["label"] = (df["cand_arc"].values == df["label_arc"].values).astype(np.int8)

    pts = trip["track"].copy()
    pts.insert(0, "point_idx", np.arange(len(pts)))
    pts.insert(0, "trip_id", tid)
    pts["label_arc"], pts["label_route_pos"] = label_arc, label_pos
    pts["label_dist_m"], pts["ref_uncertain"] = label_dist, unc

    # per-trip stats on clean points (reference route covers the point)
    hit = df.groupby("point_idx")["label"].max().reindex(range(len(pts)), fill_value=0).values
    first = df.groupby("point_idx").head(1).set_index("point_idx")
    near = first["cand_arc"].reindex(range(len(pts))).values
    twin_of_near = g.twin[np.nan_to_num(near, nan=0).astype(int)]
    clean = ~unc
    stats = dict(
        partition=partition, trip_id=tid, n_points=len(pts), n_nodes=len(trip["nodes"]),
        n_arcs=len(trip["arcs"]), route_len=len(trip["route"]),
        n_cand_rows=len(df), cands_per_point=len(df) / len(pts),
        ref_uncertain_pct=100 * unc.mean(),
        miss_rate_clean_pct=100 * (1 - hit[clean].mean()) if clean.any() else np.nan,
        nearest_directed_top1_clean_pct=100 * (near[clean] == label_arc[clean]).mean() if clean.any() else np.nan,
        nearest_undirected_top1_clean_pct=100 * ((near[clean] == label_arc[clean]) |
                                                 (twin_of_near[clean] == label_arc[clean])).mean() if clean.any() else np.nan,
        max_gap_s=float(np.max(np.diff(g.t))) if len(g.t) > 1 else 0.0,
        seconds=time.time() - t0,
    )
    return df, pts, stats


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    all_stats = []
    for partition in ["train", "student_test"]:
        jobs = [(partition, t) for t in list_trips(os.path.join(RAW, partition))]
        t0 = time.time()
        with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
            res = pool.map(process, jobs, chunksize=1)
        cands = pd.concat([r[0] for r in res], ignore_index=True)
        pts = pd.concat([r[1] for r in res], ignore_index=True)
        cands.to_parquet(os.path.join(OUT, f"{partition}_cands.parquet"), index=False)
        pts.to_parquet(os.path.join(OUT, f"{partition}_points.parquet"), index=False)
        st = pd.DataFrame([r[2] for r in res])
        all_stats.append(st)
        clean = ~pts["ref_uncertain"].values
        w = st["n_points"] * (100 - st["ref_uncertain_pct"]) / 100
        print(f"[{partition}] trips={len(jobs)} points={len(pts)} cand_rows={len(cands)} "
              f"({len(cands)/len(pts):.1f}/pt) time={time.time()-t0:.0f}s")
        print(f"   ref_uncertain={100*(~clean).mean():.2f}% | on clean points: "
              f"miss={np.average(st.miss_rate_clean_pct, weights=w):.2f}% "
              f"nearest-directed={np.average(st.nearest_directed_top1_clean_pct, weights=w):.2f}% "
              f"nearest-undirected={np.average(st.nearest_undirected_top1_clean_pct, weights=w):.2f}%")
    pd.concat(all_stats).to_csv(os.path.join(ROOT, "results", "dataset_summary.csv"), index=False)
    print("features:", len(FEATURES), FEATURES)


if __name__ == "__main__":
    main()
