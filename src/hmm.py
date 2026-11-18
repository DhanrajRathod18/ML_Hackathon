"""HMM + Viterbi route reconstruction.

Hidden states  : at GPS point i, the candidate arcs of that point (from candidates.py), so the
                 candidate -> road-segment correspondence is preserved by construction.
Emission       : alpha * log P(arc | point), where P is M7's per-point softmax probability.
Transition     : depends on the route distance d_net between the two candidate positions
                 versus the straight GPS distance d_gps between the two observations.
  classic      : log T = -|d_net - d_gps| / (beta * sqrt(dt))   (Newson & Krumm 2009 style;
                 sqrt(dt) loosens the model across GPS gaps)
  learned (N6) : logistic regression on transition features, fitted on training trips,
                 normalised with log-softmax over the next point's candidates.
d_net          : rest of arc a + shortest network path to_node(a) -> from_node(b) + the start of arc b
                 (bounded Dijkstra, cached per source node). Moving along the same arc uses the
                 projection difference. Unreachable pairs get a finite floor so the chain never breaks.
Every point always has >= 3 candidates (candidates.K_MIN), so there are no empty states.
"""
import heapq
import numpy as np
from geo import ang_diff

BACK_TOL_M = 3.0      # tolerate this much backward jitter along the same arc
CUTOFF_MIN_M = 300.0  # Dijkstra search radius = 3 * d_gps + CUTOFF_MIN_M (capped)
CUTOFF_MAX_M = 20000.0
UNREACH = np.float32(np.inf)
TRANS_FEATURES = ["abs_diff", "rel_diff", "reachable", "same_arc", "connected", "uturn",
                  "turn_angle", "log_dgps", "log_dt"]


class Router:
    """Bounded Dijkstra on the directed road graph (node -> node, weight = arc length)."""

    def __init__(self, g):
        a = g.arcs
        order = np.argsort(a[:, 0], kind="stable")
        n_nodes = int(a.max()) + 1
        self.indptr = np.searchsorted(a[order, 0], np.arange(n_nodes + 1)).tolist()
        self.to = a[order, 1].tolist()
        self.w = g.arc_len[order].tolist()
        self.cache = {}

    def dist_from(self, s, cutoff):
        hit = self.cache.get(s)
        if hit is not None and hit[0] >= cutoff:
            return hit[1]
        dist = {s: 0.0}
        heap = [(0.0, s)]
        indptr, to, w = self.indptr, self.to, self.w
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for k in range(indptr[u], indptr[u + 1]):
                v, nd = to[k], d + w[k]
                if nd <= cutoff and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        self.cache[s] = (cutoff, dist)
        return dist


class Lattice:
    """Candidate states + pairwise transition quantities for one trip (built once, decoded many times)."""

    def __init__(self, g, cands):
        """cands: DataFrame of ONE trip sorted by (point_idx, dist), columns point_idx, cand_arc, proj_frac, prob."""
        pi = cands["point_idx"].values
        self.n = len(g.px)
        starts = np.searchsorted(pi, np.arange(self.n + 1))
        self.arcs = [cands["cand_arc"].values[starts[i]:starts[i + 1]] for i in range(self.n)]
        frac = cands["proj_frac"].values
        self.frac = [frac[starts[i]:starts[i + 1]] for i in range(self.n)]
        prob = np.clip(cands["prob"].values, 1e-6, 1.0)
        self.logp = [np.log(prob[starts[i]:starts[i + 1]]) for i in range(self.n)]
        self.dgps = np.hypot(np.diff(g.px), np.diff(g.py))
        self.dt = np.maximum(np.diff(g.t), 1.0)
        self.dnet, self.feats = [], []
        router = Router(g)
        A = g.arcs
        for i in range(1, self.n):
            a, ta = self.arcs[i - 1], self.frac[i - 1]
            b, tb = self.arcs[i], self.frac[i]
            la, lb = g.arc_len[a], g.arc_len[b]
            cutoff = min(3 * self.dgps[i - 1] + CUTOFF_MIN_M, CUTOFF_MAX_M)
            D = np.full((len(a), len(b)), UNREACH, dtype=np.float32)
            from_b = A[b, 0]
            for r in range(len(a)):
                dist = router.dist_from(int(A[a[r], 1]), cutoff)
                rest = (1 - ta[r]) * la[r]
                for c in range(len(b)):
                    if a[r] == b[c] and (tb[c] - ta[r]) * la[r] >= -BACK_TOL_M:
                        D[r, c] = max(0.0, (tb[c] - ta[r]) * la[r])
                        continue
                    sp = dist.get(int(from_b[c]))
                    if sp is not None:
                        D[r, c] = rest + sp + tb[c] * lb[c]
            self.dnet.append(D)
            same = a[:, None] == b[None, :]
            self.feats.append(dict(
                same=same,
                connected=A[a, 1][:, None] == A[b, 0][None, :],
                uturn=(g.twin[a][:, None] == b[None, :]),
                turn=ang_diff(g.arc_bearing[a][:, None], g.arc_bearing[b][None, :]).astype(np.float32),
            ))

    def trans_matrix_features(self, i):
        """Feature tensor (k_prev, k_cur, F) for the transition into point i (i >= 1)."""
        D, f = self.dnet[i - 1], self.feats[i - 1]
        dg, dt = self.dgps[i - 1], self.dt[i - 1]
        reach = np.isfinite(D)
        absd = np.where(reach, np.abs(D - dg), 500.0)
        absd = np.minimum(absd, 500.0)
        return np.stack([absd, absd / (dg + 5.0), reach, f["same"], f["connected"], f["uturn"], f["turn"],
                         np.full(D.shape, np.log1p(dg)), np.full(D.shape, np.log1p(dt))], -1).astype(np.float32)


def classic_trans(lat, i, beta, floor):
    D = lat.dnet[i - 1]
    scale = beta * np.sqrt(lat.dt[i - 1])
    T = -np.abs(D - lat.dgps[i - 1]) / scale
    return np.where(np.isfinite(D), np.maximum(T, floor), floor)


def learned_trans(lat, i, model):
    """model: (scaler, logreg) fitted on transition features; returns log-softmax over the next candidates."""
    X = lat.trans_matrix_features(i)
    k1, k2, F = X.shape
    scaler, lr = model
    z = lr.decision_function(scaler.transform(X.reshape(-1, F))).reshape(k1, k2)
    z = z - z.max(1, keepdims=True)
    return z - np.log(np.exp(z).sum(1, keepdims=True))


def viterbi(lat, alpha, trans_fn):
    """Return the index of the chosen candidate at each point."""
    score = alpha * lat.logp[0]
    back = []
    for i in range(1, lat.n):
        cand = score[:, None] + trans_fn(lat, i)
        arg = cand.argmax(0)
        score = cand[arg, np.arange(cand.shape[1])] + alpha * lat.logp[i]
        back.append(arg)
    path = np.empty(lat.n, dtype=np.int64)
    path[-1] = int(np.argmax(score))
    for i in range(lat.n - 1, 0, -1):
        path[i - 1] = back[i - 1][path[i]]
    return path


def decode_arcs(lat, path):
    return np.array([lat.arcs[i][path[i]] for i in range(lat.n)])


def classifier_only_arcs(lat):
    return np.array([lat.arcs[i][int(np.argmax(lat.logp[i]))] for i in range(lat.n)])
