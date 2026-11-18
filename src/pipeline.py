"""End-to-end inference used by the API: raw trip folder -> predicted arc per GPS point.

predict_trip() reads ONLY .track/.nodes/.arcs (never .route). Scoring against the reference
route is a separate server-side function, score_trip(), so ground truth never touches the
prediction path or the UI.
"""
import os
import sys
import json
import time
import joblib
import numpy as np
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from io_utils import load_trip                                     # noqa: E402
from build_dataset import featurize_trip                           # noqa: E402
from metrics import normalize_per_point                            # noqa: E402
from hmm import Lattice, viterbi, classic_trans, learned_trans, decode_arcs, classifier_only_arcs  # noqa: E402
from labels import make_labels                                     # noqa: E402
from geo import TripGeometry                                       # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Predictor:
    def __init__(self, models_dir=os.path.join(ROOT, "models"), model_file="M7_lgbm_rank.joblib"):
        b = joblib.load(os.path.join(models_dir, model_file))
        self.model, self.features, self.mode = b["model"], b["features"], b.get("score_mode", "prob")
        self.cfg = json.load(open(os.path.join(models_dir, "hmm_config.json")))
        if self.cfg["deploy"] == "learned":
            tm = joblib.load(os.path.join(models_dir, "hmm_learned_transition.joblib"))
            self.alpha, self.trans = self.cfg["learned"]["alpha"], partial(learned_trans, model=tm)
        else:
            c = self.cfg["classic"]
            self.alpha, self.trans = c["alpha"], partial(classic_trans, beta=c["beta"], floor=c["floor"])

    def _scores(self, df):
        if self.mode == "softmax":
            raw = self.model.predict(df[self.features])
        else:
            raw = self.model.predict_proba(df[self.features])[:, 1]
        return normalize_per_point(df.assign(trip_id="x"), raw, mode=self.mode)

    def predict_trip(self, trip_dir):
        t0 = time.time()
        trip = load_trip(trip_dir, with_route=False)
        g, df = featurize_trip(trip)
        df = df.sort_values(["point_idx", "dist_m"]).reset_index(drop=True)
        df["prob"] = self._scores(df)
        lat = Lattice(g, df)
        path = viterbi(lat, self.alpha, self.trans)
        pred = decode_arcs(lat, path)
        nodes, arcs = trip["nodes"], trip["arcs"]
        # ordered list of distinct consecutive arcs -> polyline segments [[lat, lon], [lat, lon]]
        seq = [int(pred[0])] + [int(a) for p, a in zip(pred[:-1], pred[1:]) if a != p]
        segs = [[[nodes[arcs[a, 0], 1], nodes[arcs[a, 0], 0]], [nodes[arcs[a, 1], 1], nodes[arcs[a, 1], 0]]] for a in seq]
        tr = trip["track"]
        return {
            "trip_id": trip["trip_id"],
            "pred_arcs": pred.tolist(),
            "classifier_arcs": classifier_only_arcs(lat).tolist(),
            "state_prob": [float(np.exp(lat.logp[i][path[i]])) for i in range(lat.n)],
            "gps": tr[["lat", "lon"]].values.tolist(),
            "route_segments": [[[float(x) for x in p] for p in s] for s in segs],
            "route_arc_sequence": seq,
            "n_points": int(lat.n),
            "latency_ms": round(1000 * (time.time() - t0), 1),
        }


def score_trip(trip_dir, pred_arcs):
    """SERVER-SIDE ONLY: compare predictions with the reference route (DP-aligned per-point labels)."""
    trip = load_trip(trip_dir, with_route=True)
    if trip["route"] is None:
        return None
    g = TripGeometry(trip)
    label_arc, _, _, unc = make_labels(g, trip["route"])
    pred = np.asarray(pred_arcs)
    clean = ~unc
    A = trip["arcs"]
    s = pred[clean]
    cont = float(((s[1:] == s[:-1]) | (A[s[:-1], 1] == A[s[1:], 0])).mean()) if len(s) > 1 else 1.0
    ref, got = set(trip["route"].tolist()), set(pred.tolist())
    return {
        "top1_clean": float((pred[clean] == label_arc[clean]).mean()) if clean.any() else None,
        "top1_all": float((pred == label_arc).mean()),
        "continuity": cont,
        "route_arc_precision": len(ref & got) / max(len(got), 1),
        "route_arc_recall": len(ref & got) / max(len(ref), 1),
        "n_clean_points": int(clean.sum()),
        "n_ref_uncertain_points": int(unc.sum()),
    }
