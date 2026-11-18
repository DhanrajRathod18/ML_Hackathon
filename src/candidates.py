"""Candidate road segments for every GPS point.

Arcs are densified into points every STEP_M metres and indexed in a KD-tree, so a
radius query returns every arc passing near a GPS point regardless of how long the
arc is. Exact point-to-segment distances then filter to R_M.
Measured: R=50 m -> candidate miss rate 0.14 % on student_test, ~11 candidates/point.
"""
import numpy as np
from scipy.spatial import cKDTree
from geo import point_segment

R_M = 50.0      # candidate radius
K_MAX = 16      # cap per point (keeps the nearest K_MAX)
K_MIN = 3       # points with nothing inside R_M still get their nearest K_MIN arcs
STEP_M = 10.0   # densification spacing


def _densify(g):
    n = np.maximum(1, np.ceil(g.arc_len / STEP_M).astype(np.int64))
    cnt = n + 1                                     # include both endpoints
    arc_idx = np.repeat(np.arange(len(n)), cnt)
    starts = np.repeat(np.cumsum(cnt) - cnt, cnt)
    frac = (np.arange(cnt.sum()) - starts) / np.repeat(n, cnt)
    x = g.ax[arc_idx] + frac * (g.bx - g.ax)[arc_idx]
    y = g.ay[arc_idx] + frac * (g.by - g.ay)[arc_idx]
    return cKDTree(np.c_[x, y]), arc_idx


def _pairs_to_cands(g, pi, arc):
    key = np.unique(pi.astype(np.int64) * len(g.arcs) + arc)
    pi, arc = key // len(g.arcs), key % len(g.arcs)
    d, t = point_segment(g.px[pi], g.py[pi], g.ax[arc], g.ay[arc], g.bx[arc], g.by[arc])
    return pi, arc, d, t


def generate(g):
    """Return dict of flat arrays: point_idx, cand_arc, dist_m, proj_frac (sorted by point, then distance)."""
    tree, arc_idx = _densify(g)
    pts = np.c_[g.px, g.py]
    hits = tree.query_ball_point(pts, R_M + STEP_M / 2 + 1.0)
    lens = np.fromiter((len(h) for h in hits), dtype=np.int64, count=len(hits))
    pi = np.repeat(np.arange(len(pts)), lens)
    arc = arc_idx[np.fromiter((j for h in hits for j in h), dtype=np.int64, count=lens.sum())]
    pi, arc, d, t = _pairs_to_cands(g, pi, arc)
    keep = d <= R_M
    pi, arc, d, t = pi[keep], arc[keep], d[keep], t[keep]

    # fallback: points with no candidate inside R_M get their nearest K_MIN arcs
    empty = np.setdiff1d(np.arange(len(pts)), pi)
    if len(empty):
        k = min(80, len(arc_idx))
        _, nn = tree.query(pts[empty], k=k)
        nn = np.atleast_2d(nn)
        fpi = np.repeat(empty, nn.shape[1])
        fpi, farc, fd, ft = _pairs_to_cands(g, fpi, arc_idx[nn.ravel()])
        o = np.lexsort((fd, fpi))
        fpi, farc, fd, ft = fpi[o], farc[o], fd[o], ft[o]
        first = np.r_[0, np.flatnonzero(np.diff(fpi)) + 1]
        rank = np.arange(len(fpi)) - np.repeat(first, np.diff(np.r_[first, len(fpi)]))
        m = rank < K_MIN
        pi, arc, d, t = (np.r_[pi, fpi[m]], np.r_[arc, farc[m]], np.r_[d, fd[m]], np.r_[t, ft[m]])

    o = np.lexsort((d, pi))
    pi, arc, d, t = pi[o], arc[o], d[o], t[o]
    first = np.r_[0, np.flatnonzero(np.diff(pi)) + 1]
    rank = np.arange(len(pi)) - np.repeat(first, np.diff(np.r_[first, len(pi)]))
    m = rank < K_MAX
    return {"point_idx": pi[m], "cand_arc": arc[m], "dist_m": d[m], "proj_frac": t[m], "dist_rank": rank[m]}
