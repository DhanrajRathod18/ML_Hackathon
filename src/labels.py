"""Per-observation labels from the reference route (novelty N1).

The .route file is an ORDERED sequence of arc ids, not one label per GPS point.
We align the GPS sequence to the route sequence with dynamic programming:
    C[i, j] = D[i, j] + min_{j-S <= k <= j} C[i-1, k]
where D[i, j] is the metric distance from GPS point i to route arc j. This is the
globally optimal monotone alignment (time only moves forward along the route).
Measured on student_test: 0.14 % of labels > 50 m away, versus 3.8 % for a greedy
window search and 23 % for "nearest route arc" (routes revisit arcs in loops).
"""
import numpy as np
from geo import point_segment

STEP = 8           # max route arcs advanced between consecutive GPS points
UNCERTAIN_M = 50.0  # label farther than this from the GPS point -> reference doesn't cover it


def route_distance_matrix(g, route, chunk=4000):
    ax, ay, bx, by = g.ax[route], g.ay[route], g.bx[route], g.by[route]
    out = np.empty((len(g.px), len(route)), dtype=np.float32)
    for s in range(0, len(g.px), chunk):
        e = s + chunk
        d, _ = point_segment(g.px[s:e, None], g.py[s:e, None], ax[None], ay[None], bx[None], by[None])
        out[s:e] = d
    return out


def dp_align(D, S=STEP):
    N, M = D.shape
    INF = np.float32(1e30)
    cost = np.full(M, INF, dtype=np.float32)
    cost[:S] = D[0, :S]
    back = np.zeros((N, M), dtype=np.int32)
    ar = np.arange(M)
    for i in range(1, N):
        best, arg = cost.copy(), ar.copy()
        for s in range(1, S + 1):
            sh = np.full(M, INF, dtype=np.float32)
            sh[s:] = cost[:-s]
            m = sh < best
            best[m] = sh[m]
            arg[m] = ar[m] - s
        cost = best + D[i]
        back[i] = arg
    lo = max(0, M - S)
    j = int(np.argmin(cost[lo:]) + lo)
    lab = np.empty(N, dtype=np.int64)
    for i in range(N - 1, -1, -1):
        lab[i] = j
        j = back[i, j]
    return lab


def make_labels(g, route):
    """Return (label_arc, label_route_pos, label_dist_m, ref_uncertain) per GPS point."""
    D = route_distance_matrix(g, route)
    pos = dp_align(D)
    dist = D[np.arange(len(pos)), pos].astype(float)
    return route[pos], pos, dist, dist > UNCERTAIN_M
