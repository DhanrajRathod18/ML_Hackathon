"""Feature engineering for (GPS point, candidate arc) pairs.

Design is driven by measured facts (PATHSNAP_PLAN.md section 0.5):
  * 75 % of true arcs have a reverse twin (two-way road) at IDENTICAL distance,
    so distance alone cannot pick the direction -> heading features (N3).
  * GPS heading agrees with the true direction 59-67 % of the time below 3 m/s
    but 98-100 % above 6 m/s -> give the model speed / heading-confidence and a
    heading "carried" from the nearest fast-moving point (through stops).
  * a single point is ambiguous near parallel roads; the trajectory is not ->
    temporal-context features over +-CTX points (N2).
All features use only the GPS track + road network (never the route), so the same
code runs at inference time in the API.
"""
import numpy as np
import pandas as pd
from geo import point_segment, ang_diff
from candidates import R_M

CTX = 5            # context half-window (points)
FAST_MPS = 6.0     # above this speed GPS heading is ~99 % reliable (measured)
GAP_S = 5.0

FEATURES = [
    "dist_m", "dist_rank", "dist_minus_min", "dist_ratio_min", "n_cands", "n_close",
    "heading_diff", "speed_mps", "heading_conf",
    "carried_heading_diff", "carried_next_heading_diff", "has_carried",
    "has_twin", "dt_prev", "dt_next", "gap_flag",
    "arc_len_m", "proj_frac", "out_deg", "in_deg",
    "ctx_share", "ctx_mean_dist", "ctx_heading_diff", "ctx_disp",
]


def _shift_idx(n, k):
    return np.clip(np.arange(n) + k, 0, n - 1)


def point_kinematics(g):
    n = len(g.px)
    px, py, t = g.px, g.py, g.t
    i_m1, i_p1 = _shift_idx(n, -1), _shift_idx(n, 1)
    dt_c = np.maximum(t[i_p1] - t[i_m1], 1e-6)
    speed = np.hypot(px[i_p1] - px[i_m1], py[i_p1] - py[i_m1]) / dt_c
    i_m2, i_p2 = _shift_idx(n, -2), _shift_idx(n, 2)
    hx, hy = px[i_p2] - px[i_m2], py[i_p2] - py[i_m2]
    heading = np.arctan2(hy, hx)
    heading_conf = np.hypot(hx, hy)                       # metres moved over +-2 points
    i_mc, i_pc = _shift_idx(n, -CTX), _shift_idx(n, CTX)
    wx, wy = px[i_pc] - px[i_mc], py[i_pc] - py[i_mc]
    win_heading, win_disp = np.arctan2(wy, wx), np.hypot(wx, wy)

    fast = speed > FAST_MPS
    idx = np.where(fast, np.arange(n), -1)
    prev_fast = np.maximum.accumulate(idx)                # last fast point at or before i
    idx2 = np.where(fast, np.arange(n), n)
    next_fast = np.minimum.accumulate(idx2[::-1])[::-1]   # next fast point at or after i
    dt_prev = np.r_[np.nan, np.diff(t)]
    dt_next = np.r_[np.diff(t), np.nan]
    return {
        "speed": speed, "heading": heading, "heading_conf": heading_conf,
        "win_heading": win_heading, "win_disp": win_disp,
        "prev_fast": prev_fast, "next_fast": next_fast,
        "dt_prev": np.nan_to_num(dt_prev, nan=1.0), "dt_next": np.nan_to_num(dt_next, nan=1.0),
    }


def build_features(g, c):
    """g: TripGeometry, c: dict from candidates.generate -> DataFrame with FEATURES."""
    k = point_kinematics(g)
    n = len(g.px)
    pi, arc, d = c["point_idx"], c["cand_arc"], c["dist_m"]
    df = pd.DataFrame({"point_idx": pi, "cand_arc": arc, "dist_m": d,
                       "dist_rank": c["dist_rank"], "proj_frac": c["proj_frac"]})
    first = np.r_[0, np.flatnonzero(np.diff(pi)) + 1]
    counts = np.diff(np.r_[first, len(pi)])
    dmin = np.repeat(d[first], counts)                    # rows sorted by (point, dist)
    df["dist_minus_min"] = d - dmin
    df["dist_ratio_min"] = d / (dmin + 1.0)
    df["n_cands"] = np.repeat(counts, counts)
    close = (d - dmin) <= 10.0
    df["n_close"] = np.repeat(np.add.reduceat(close.astype(int), first), counts)

    bearing = g.arc_bearing[arc]
    df["heading_diff"] = ang_diff(k["heading"][pi], bearing)
    df["speed_mps"] = k["speed"][pi]
    df["heading_conf"] = k["heading_conf"][pi]

    pf, nf = k["prev_fast"][pi], k["next_fast"][pi]
    has_pf, has_nf = pf >= 0, nf < n
    neutral = np.pi / 2                                   # uninformative angle when no fast point exists
    df["carried_heading_diff"] = np.where(
        has_pf, ang_diff(k["heading"][np.clip(pf, 0, n - 1)], bearing), neutral)
    df["carried_next_heading_diff"] = np.where(
        has_nf, ang_diff(k["heading"][np.clip(nf, 0, n - 1)], bearing), neutral)
    df["has_carried"] = (has_pf | has_nf).astype(np.int8)

    df["has_twin"] = (g.twin[arc] >= 0).astype(np.int8)
    df["dt_prev"] = k["dt_prev"][pi]
    df["dt_next"] = k["dt_next"][pi]
    df["gap_flag"] = (np.maximum(df["dt_prev"], df["dt_next"]) > GAP_S).astype(np.int8)
    df["arc_len_m"] = g.arc_len[arc]
    df["out_deg"] = g.out_deg[arc]
    df["in_deg"] = g.in_deg[arc]

    # temporal context: how well does this arc explain the neighbouring GPS points?
    ax, ay, bx, by = g.ax[arc], g.ay[arc], g.bx[arc], g.by[arc]
    tot = np.zeros(len(pi))
    within = np.zeros(len(pi))
    for off in range(-CTX, CTX + 1):
        if off == 0:
            continue
        j = np.clip(pi + off, 0, n - 1)
        dj, _ = point_segment(g.px[j], g.py[j], ax, ay, bx, by)
        tot += np.minimum(dj, 200.0)
        within += dj <= R_M
    df["ctx_mean_dist"] = tot / (2 * CTX)
    df["ctx_share"] = within / (2 * CTX)
    df["ctx_heading_diff"] = ang_diff(k["win_heading"][pi], bearing)
    df["ctx_disp"] = k["win_disp"][pi]

    for col in FEATURES:
        df[col] = df[col].astype(np.float32)
    return df
