"""Reproducible exploratory data analysis (report §2). Every EDA number/figure comes from this script.

Partitions analysed: train (80 trips) and student_test (10 trips). Evaluator data is never read.
Usage:  python src/eda.py          (~1-2 min; needs data/raw and, for the label-based parts,
                                     data/processed/*_points.parquet from build_dataset.py)
Outputs (prefix eda_ so nothing else is overwritten):
  results/eda_inventory.csv      per trip: points, duration, nodes, arcs, route arcs, file sizes, extent, gaps, speeds
  results/eda_quality.csv        per trip: missing / invalid values, timestamp order, invalid arc / route references
  results/eda_network.csv        per trip: weak components, largest-component share, isolated nodes, two-way share
  results/eda_route.csv          per trip: route connectivity, repeated arcs, route length vs GPS length, coverage
  results/eda_heading_by_speed.csv  GPS heading vs true arc direction by speed bin (train, clean points)
  results/eda_summary.json       partition-level totals quoted in the report
  figures/eda_*.png              sampling gaps, speeds, trip locations, network connectivity, route vs GPS length,
                                 reference coverage, heading reliability vs speed
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from io_utils import list_trips, load_trip          # noqa: E402
from geo import to_xy, ang_diff                       # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a)
PARTS = ["train", "student_test"]


def analyse(args):
    part, tid = args
    d = P("data", "raw", part, tid)
    trip = load_trip(d)
    tr, nodes, arcs, route = trip["track"], trip["nodes"], trip["arcs"], trip["route"]
    sizes = {f"size_{e}_kb": os.path.getsize(os.path.join(d, f"{tid}.{e}")) / 1024 for e in ["track", "nodes", "arcs", "route"]}
    lon0, lat0 = tr.lon.mean(), tr.lat.mean()
    x, y = to_xy(tr.lon.values, tr.lat.values, lon0, lat0)
    dt = np.diff(tr.t.values)
    step = np.hypot(np.diff(x), np.diff(y))
    speed = step / np.where(dt > 0, dt, np.nan)
    nx_, ny_ = to_xy(nodes[:, 0], nodes[:, 1], lon0, lat0)
    alen = np.hypot(nx_[arcs[:, 1]] - nx_[arcs[:, 0]], ny_[arcs[:, 1]] - ny_[arcs[:, 0]])

    inv = dict(partition=part, trip_id=tid, n_points=len(tr), duration_min=(tr.t.iat[-1] - tr.t.iat[0]) / 60,
               n_nodes=len(nodes), n_arcs=len(arcs), route_arcs=len(route), centroid_lon=lon0, centroid_lat=lat0,
               extent_km_x=np.ptp(x) / 1000, extent_km_y=np.ptp(y) / 1000,
               pct_dt_exactly_1s=100 * np.mean(dt == 1), n_gaps_gt5s=int(np.sum(dt > 5)), n_gaps_gt60s=int(np.sum(dt > 60)),
               max_gap_s=float(dt.max()) if len(dt) else 0.0, median_speed_mps=float(np.nanmedian(speed)),
               p99_speed_mps=float(np.nanpercentile(speed, 99)), n_speed_gt50=int(np.nansum(speed > 50)),
               gps_length_km=float(step.sum() / 1000), **sizes)
    q = dict(partition=part, trip_id=tid,
             nan_track=int(tr.isna().sum().sum()), nan_nodes=int(np.isnan(nodes).sum()),
             lon_out_of_range=int(((tr.lon < -180) | (tr.lon > 180)).sum()), lat_out_of_range=int(((tr.lat < -90) | (tr.lat > 90)).sum()),
             non_increasing_t=int(np.sum(dt <= 0)), duplicate_t=int(np.sum(dt == 0)),
             invalid_arc_node_refs=int(np.sum((arcs < 0) | (arcs >= len(nodes)))),
             invalid_route_arc_ids=int(np.sum((route < 0) | (route >= len(arcs)))),
             zero_length_arcs=int(np.sum(alen == 0)), duplicate_arcs=int(len(arcs) - len(np.unique(arcs, axis=0))))
    n = len(nodes)
    g = coo_matrix((np.ones(len(arcs)), (arcs[:, 0], arcs[:, 1])), shape=(n, n))
    ncomp, lab = connected_components(g, directed=True, connection="weak")
    comp_sizes = np.bincount(lab)
    deg = np.bincount(arcs[:, 0], minlength=n) + np.bincount(arcs[:, 1], minlength=n)
    key = set(map(tuple, arcs.tolist()))
    twin = np.array([(v, u) in key for u, v in arcs.tolist()])
    ncomp_s, _ = connected_components(g, directed=True, connection="strong")
    net = dict(partition=part, trip_id=tid, weak_components=int(ncomp), strong_components=int(ncomp_s),
               largest_weak_component_pct=100 * comp_sizes.max() / n, isolated_nodes=int(np.sum(deg == 0)),
               components_lt10_nodes=int(np.sum(comp_sizes < 10)), pct_arcs_two_way=100 * twin.mean(),
               median_arc_len_m=float(np.median(alen)))
    conn = arcs[route[:-1], 1] == arcs[route[1:], 0]
    rt = dict(partition=part, trip_id=tid, route_arcs=len(route), route_transitions=len(route) - 1,
              route_disconnected=int((~conn).sum()), route_repeated_arcs=int(len(route) - len(np.unique(route))),
              route_length_km=float(alen[route].sum() / 1000), gps_length_km=inv["gps_length_km"],
              route_arcs_two_way_pct=100 * twin[route].mean())
    return inv, q, net, rt


def heading_by_speed():
    """Measured on TRAIN clean points: how often GPS heading agrees with the reference arc's direction."""
    pts = pd.read_parquet(P("data", "processed", "train_points.parquet"))
    rows = []
    for tid, p in pts.groupby("trip_id"):
        trip = load_trip(P("data", "raw", "train", tid), with_route=False)
        p = p.sort_values("point_idx")
        lon0, lat0 = p.lon.mean(), p.lat.mean()
        x, y = to_xy(p.lon.values, p.lat.values, lon0, lat0)
        nx_, ny_ = to_xy(trip["nodes"][:, 0], trip["nodes"][:, 1], lon0, lat0)
        a = trip["arcs"][p.label_arc.values]
        bearing = np.arctan2(ny_[a[:, 1]] - ny_[a[:, 0]], nx_[a[:, 1]] - nx_[a[:, 0]])
        n = len(x)
        im, ip = np.clip(np.arange(n) - 2, 0, n - 1), np.clip(np.arange(n) + 2, 0, n - 1)
        hx, hy = x[ip] - x[im], y[ip] - y[im]
        dtt = np.maximum(p.t.values[ip] - p.t.values[im], 1e-6)
        spd = np.hypot(hx, hy) / dtt
        agree = ang_diff(np.arctan2(hy, hx), bearing) < np.pi / 2
        keep = ~p.ref_uncertain.values
        rows.append(pd.DataFrame({"speed": spd[keep], "agree": agree[keep]}))
    df = pd.concat(rows)
    bins = [0, 1, 3, 6, 9, 12, 15, 20, 30, 100]
    df["bin"] = pd.cut(df.speed, bins, right=False)
    out = df.groupby("bin", observed=True).agg(n_points=("agree", "size"), pct_agree=("agree", "mean")).reset_index()
    out["pct_agree"] *= 100
    out["bin"] = out["bin"].astype(str)
    return out


def main():
    for d in ["results", "figures"]:
        os.makedirs(P(d), exist_ok=True)
    jobs = [(p, t) for p in PARTS for t in list_trips(P("data", "raw", p))]
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        res = pool.map(analyse, jobs, chunksize=1)
    inv, q, net, rt = (pd.DataFrame([r[i] for r in res]) for i in range(4))
    pts = {p: pd.read_parquet(P("data", "processed", f"{p}_points.parquet")) for p in PARTS
           if os.path.exists(P("data", "processed", f"{p}_points.parquet"))}
    if pts:
        cov = pd.concat(pts.values()).groupby("trip_id").ref_uncertain.mean().mul(100).rename("pct_points_not_covered_by_reference")
        rt = rt.merge(cov, left_on="trip_id", right_index=True, how="left")
    inv.to_csv(P("results", "eda_inventory.csv"), index=False)
    q.to_csv(P("results", "eda_quality.csv"), index=False)
    net.to_csv(P("results", "eda_network.csv"), index=False)
    rt.to_csv(P("results", "eda_route.csv"), index=False)

    summary = {}
    for p in PARTS:
        i, qq, nn, rr = (t[t.partition == p] for t in (inv, q, net, rt))
        s = dict(trips=len(i), gps_points=int(i.n_points.sum()), nodes=int(i.n_nodes.sum()), arcs=int(i.n_arcs.sum()),
                 route_arcs=int(i.route_arcs.sum()), points_per_trip_min=int(i.n_points.min()),
                 points_per_trip_median=float(i.n_points.median()), points_per_trip_max=int(i.n_points.max()),
                 pct_dt_exactly_1s=float(np.average(i.pct_dt_exactly_1s, weights=i.n_points)),
                 gaps_gt5s=int(i.n_gaps_gt5s.sum()), gaps_gt60s=int(i.n_gaps_gt60s.sum()),
                 max_gap_s=float(i.max_gap_s.max()), max_gap_trip=i.loc[i.max_gap_s.idxmax(), "trip_id"],
                 missing_or_invalid_gps_values=int(qq[["nan_track", "lon_out_of_range", "lat_out_of_range",
                                                       "non_increasing_t", "duplicate_t"]].sum().sum()),
                 invalid_network_or_route_refs=int(qq[["nan_nodes", "invalid_arc_node_refs",
                                                       "invalid_route_arc_ids"]].sum().sum()),
                 duplicate_arcs=int(qq.duplicate_arcs.sum()), zero_length_arcs=int(qq.zero_length_arcs.sum()),
                 isolated_nodes=int(nn.isolated_nodes.sum()), weak_components_total=int(nn.weak_components.sum()),
                 largest_component_pct_mean=float(nn.largest_weak_component_pct.mean()),
                 largest_component_pct_min=float(nn.largest_weak_component_pct.min()),
                 pct_arcs_two_way_mean=float(nn.pct_arcs_two_way.mean()),
                 route_disconnected_transitions=int(rr.route_disconnected.sum()),
                 route_repeated_arcs=int(rr.route_repeated_arcs.sum()))
        if p in pts:
            s["pct_points_not_covered_by_reference"] = float(100 * pts[p].ref_uncertain.mean())
        summary[p] = s
    hb = heading_by_speed() if "train" in pts else None
    if hb is not None:
        hb.to_csv(P("results", "eda_heading_by_speed.csv"), index=False)
        summary["train"]["heading_agreement_by_speed"] = dict(zip(hb.bin, hb.pct_agree.round(1)))
    json.dump(summary, open(P("results", "eda_summary.json"), "w"), indent=2, default=str)

    # ---------------- figures (all from the tables above) ----------------
    colors = {"train": "C0", "student_test": "C1"}
    allgaps = {p: np.concatenate([np.diff(load_trip(P("data", "raw", p, t), with_route=False)["track"].t.values)
                                  for t in list_trips(P("data", "raw", p))]) for p in PARTS}
    plt.figure(figsize=(7, 4))
    for p, g in allgaps.items():
        plt.hist(np.clip(g, 0, 30), bins=60, alpha=.6, label=f"{p} (n={len(g)})", color=colors[p], log=True)
    plt.xlabel("time between consecutive GPS points (s, clipped at 30)"); plt.ylabel("count (log)")
    plt.title("GPS sampling interval"); plt.legend(); plt.tight_layout(); plt.savefig(P("figures", "eda_sampling_gaps.png"), dpi=150); plt.close()

    plt.figure(figsize=(7, 4))
    for p in PARTS:
        sp = []
        for t in list_trips(P("data", "raw", p)):
            tr = load_trip(P("data", "raw", p, t), with_route=False)["track"]
            x, y = to_xy(tr.lon.values, tr.lat.values, tr.lon.mean(), tr.lat.mean())
            dtt = np.diff(tr.t.values)
            sp.append(np.hypot(np.diff(x), np.diff(y))[dtt > 0] / dtt[dtt > 0])
        plt.hist(np.clip(np.concatenate(sp), 0, 50), bins=100, alpha=.6, density=True, label=p, color=colors[p])
    plt.xlabel("speed between consecutive points (m/s, clipped at 50)"); plt.ylabel("density")
    plt.title("GPS speed distribution"); plt.legend(); plt.tight_layout(); plt.savefig(P("figures", "eda_speed.png"), dpi=150); plt.close()

    plt.figure(figsize=(8, 4))
    for p in PARTS:
        i = inv[inv.partition == p]
        plt.scatter(i.centroid_lon, i.centroid_lat, s=np.sqrt(i.n_points) * 2, alpha=.6, label=p, color=colors[p])
    plt.xlabel("longitude"); plt.ylabel("latitude"); plt.title("Trip locations (marker size ~ number of GPS points)")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(P("figures", "eda_trip_locations.png"), dpi=150); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for p in PARTS:
        nn = net[net.partition == p]
        ax[0].hist(nn.largest_weak_component_pct, bins=20, alpha=.6, label=p, color=colors[p])
        ax[1].scatter(nn.merge(inv, on=["partition", "trip_id"]).n_nodes, nn.weak_components, alpha=.6, label=p, color=colors[p])
    ax[0].set_xlabel("% of nodes in largest weakly connected component"); ax[0].set_ylabel("trips"); ax[0].legend()
    ax[1].set_xlabel("nodes in trip network"); ax[1].set_ylabel("weakly connected components"); ax[1].set_xscale("log")
    fig.suptitle("Road-network connectivity per trip"); fig.tight_layout(); fig.savefig(P("figures", "eda_network_connectivity.png"), dpi=150); plt.close(fig)

    plt.figure(figsize=(5, 5))
    for p in PARTS:
        r = rt[rt.partition == p]
        plt.scatter(r.gps_length_km, r.route_length_km, alpha=.7, label=p, color=colors[p])
    m = max(rt.gps_length_km.max(), rt.route_length_km.max())
    plt.plot([0, m], [0, m], "k--", lw=1, label="y = x")
    plt.xlabel("GPS trajectory length (km)"); plt.ylabel("reference route length (km)"); plt.title("Reference route vs GPS length")
    plt.legend(); plt.tight_layout(); plt.savefig(P("figures", "eda_route_vs_gps_length.png"), dpi=150); plt.close()

    if "pct_points_not_covered_by_reference" in rt:
        r = rt.sort_values("pct_points_not_covered_by_reference", ascending=False).head(15)
        plt.figure(figsize=(8, 4))
        plt.bar(r.trip_id, r.pct_points_not_covered_by_reference, color=[colors[p] for p in r.partition])
        plt.xticks(rotation=60, fontsize=8); plt.ylabel("% GPS points > 50 m from aligned\nreference arc (ref_uncertain)")
        plt.title("Reference-route coverage: 15 worst trips (blue = train, orange = student_test)")
        plt.tight_layout(); plt.savefig(P("figures", "eda_reference_coverage.png"), dpi=150); plt.close()

    if hb is not None:
        plt.figure(figsize=(7, 4))
        plt.bar(hb.bin, hb.pct_agree)
        for k, (b, v, nn_) in enumerate(zip(hb.bin, hb.pct_agree, hb.n_points)):
            plt.text(k, v + 1, f"{v:.0f}%\n(n={nn_})", ha="center", fontsize=7)
        plt.ylim(0, 115); plt.xlabel("GPS speed bin (m/s)"); plt.ylabel("% heading agrees with true arc direction")
        plt.title("Train, clean points: GPS heading is reliable only when moving"); plt.xticks(rotation=30)
        plt.tight_layout(); plt.savefig(P("figures", "eda_heading_vs_speed.png"), dpi=150); plt.close()

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
