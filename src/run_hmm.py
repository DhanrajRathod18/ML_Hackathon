"""Tune and evaluate HMM/Viterbi decoding on top of M7.

* Decoder parameters are tuned ONLY on the 80 training trips, using out-of-fold M7 scores
  (score_candidates.py). The student-test trips are used once at the end for the comparison.
* Compared: classifier-only (argmax per point) vs classic HMM vs learned-transition HMM (N6).
Usage:  python src/run_hmm.py
Outputs: results/hmm_grid_classic.csv, results/hmm_grid_learned.csv, results/hmm_comparison.csv,
         results/hmm_per_trip.csv, results/preds_student_test_hmm.parquet, models/hmm_config.json,
         models/hmm_learned_transition.joblib, figures/hmm_comparison.png
"""
import os
import sys
import json
import time
import itertools
import joblib
import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from io_utils import load_trip                                         # noqa: E402
from geo import TripGeometry                                            # noqa: E402
from hmm import (Lattice, viterbi, classic_trans, learned_trans, decode_arcs,  # noqa: E402
                 classifier_only_arcs, TRANS_FEATURES)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a)
SEED = 42


def build_one(args):
    part, tid, cands = args
    trip = load_trip(P("data", "raw", part, tid), with_route=False)
    g = TripGeometry(trip)
    lat = Lattice(g, cands)
    return tid, lat, trip["arcs"]


def build_all(part):
    sc = pd.read_parquet(P("data", "processed", f"{part}_scored.parquet"))
    pts = pd.read_parquet(P("data", "processed", f"{part}_points.parquet"))
    jobs = [(part, tid, g.sort_values(["point_idx", "dist_m"])) for tid, g in sc.groupby("trip_id")]
    t0 = time.time()
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        res = pool.map(build_one, jobs, chunksize=1)
    print(f"[{part}] lattices for {len(res)} trips in {time.time()-t0:.0f}s")
    out = {}
    for tid, lat, arcs in res:
        p = pts[pts.trip_id == tid].sort_values("point_idx")
        out[tid] = dict(lat=lat, arcs=arcs, label=p["label_arc"].values, clean=~p["ref_uncertain"].values)
    return out


def score(trips, pred_by_trip):
    """Pooled clean-point Top-1 and continuity (consecutive clean points: same or connected arc)."""
    ok = n = c_ok = c_n = 0
    for tid, d in trips.items():
        pr, lab, cl, A = pred_by_trip[tid], d["label"], d["clean"], d["arcs"]
        ok += int((pr[cl] == lab[cl]).sum()); n += int(cl.sum())
        s = pr[cl]
        if len(s) > 1:
            c_ok += int(((s[1:] == s[:-1]) | (A[s[:-1], 1] == A[s[1:], 0])).sum()); c_n += len(s) - 1
    return ok / max(n, 1), c_ok / max(c_n, 1)


def decode_all(trips, alpha, fn):
    return {tid: decode_arcs(d["lat"], viterbi(d["lat"], alpha, fn)) for tid, d in trips.items()}


def learned_training_pairs(trips, max_steps=60000):
    """Rows = (true previous state, every next candidate); target = next candidate is the true arc."""
    rng = np.random.default_rng(SEED)
    X, y = [], []
    steps = [(tid, i) for tid, d in trips.items() for i in range(1, d["lat"].n)
             if d["clean"][i] and d["clean"][i - 1]]
    if len(steps) > max_steps:
        steps = [steps[j] for j in rng.choice(len(steps), max_steps, replace=False)]
    for tid, i in steps:
        d = trips[tid]; lat = d["lat"]
        prev = np.flatnonzero(lat.arcs[i - 1] == d["label"][i - 1])
        nxt = lat.arcs[i] == d["label"][i]
        if len(prev) == 0 or not nxt.any():
            continue
        X.append(lat.trans_matrix_features(i)[prev[0]])
        y.append(nxt.astype(int))
    return np.concatenate(X), np.concatenate(y)


def main():
    train = build_all("train")
    st = build_all("student_test")

    base_tr = score(train, {t: classifier_only_arcs(d["lat"]) for t, d in train.items()})
    print(f"train OOF classifier-only: top1={base_tr[0]:.4f} continuity={base_tr[1]:.4f}")

    # ---- classic transition: grid search on train (OOF emissions) ----
    rows = []
    for alpha, beta, floor in itertools.product([0.5, 1.0, 2.0], [3.0, 10.0, 30.0], [-15.0, -40.0]):
        t0 = time.time()
        acc, cont = score(train, decode_all(train, alpha, partial(classic_trans, beta=beta, floor=floor)))
        rows.append(dict(alpha=alpha, beta=beta, floor=floor, train_top1=acc, train_continuity=cont, s=round(time.time()-t0, 1)))
        print(f"  classic a={alpha} b={beta} f={floor}: top1={acc:.4f} cont={cont:.4f} ({rows[-1]['s']}s)")
    gc = pd.DataFrame(rows).sort_values("train_top1", ascending=False)
    gc.to_csv(P("results", "hmm_grid_classic.csv"), index=False)
    best_c = gc.iloc[0]

    # ---- learned transition (N6) ----
    X, y = learned_training_pairs(train)
    scaler = StandardScaler().fit(X)
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(scaler.transform(X), y)
    coef = pd.Series(lr.coef_[0], index=TRANS_FEATURES)
    coef.to_csv(P("results", "hmm_learned_transition_coef.csv"))
    print(f"learned transition fitted on {len(y)} pairs; coefficients:\n{coef.round(3)}")
    rows = []
    for alpha in [0.25, 0.5, 1.0, 2.0, 4.0]:
        acc, cont = score(train, decode_all(train, alpha, partial(learned_trans, model=(scaler, lr))))
        rows.append(dict(alpha=alpha, train_top1=acc, train_continuity=cont))
        print(f"  learned a={alpha}: top1={acc:.4f} cont={cont:.4f}")
    gl = pd.DataFrame(rows).sort_values("train_top1", ascending=False)
    gl.to_csv(P("results", "hmm_grid_learned.csv"), index=False)
    best_l = gl.iloc[0]

    # ---- final comparison on student_test (used once) ----
    classic_fn = partial(classic_trans, beta=best_c.beta, floor=best_c.floor)
    learned_fn = partial(learned_trans, model=(scaler, lr))
    variants = {
        "classifier_only (M7 argmax)": lambda d: classifier_only_arcs(d["lat"]),
        "HMM classic": lambda d: decode_arcs(d["lat"], viterbi(d["lat"], best_c.alpha, classic_fn)),
        "HMM learned transitions (N6)": lambda d: decode_arcs(d["lat"], viterbi(d["lat"], best_l.alpha, learned_fn)),
    }
    comp, per_trip, preds = [], [], {}
    for name, f in variants.items():
        t0 = time.time()
        pr = {tid: f(d) for tid, d in st.items()}
        secs = time.time() - t0
        acc, cont = score(st, pr)
        npts = sum(d["lat"].n for d in st.values())
        comp.append(dict(variant=name, st_top1=acc, st_continuity=cont, decode_ms_per_point=1000 * secs / npts))
        for tid, d in st.items():
            a, c = score({tid: d}, {tid: pr[tid]})
            cl = d["clean"]
            ref, got = set(d["label"][cl]), set(pr[tid][cl])
            per_trip.append(dict(variant=name, trip_id=tid, top1=a, continuity=c,
                                 route_jaccard=len(ref & got) / max(len(ref | got), 1)))
        preds[name] = pr
        print(f"STUDENT_TEST {name}: top1={acc:.4f} continuity={cont:.4f}")
    comp = pd.DataFrame(comp)
    comp.to_csv(P("results", "hmm_comparison.csv"), index=False)
    pd.DataFrame(per_trip).to_csv(P("results", "hmm_per_trip.csv"), index=False)

    best_name = max(["HMM classic", "HMM learned transitions (N6)"],
                    key=lambda v: comp.set_index("variant").loc[v, "st_top1"])
    out = []
    for tid, d in st.items():
        out.append(pd.DataFrame({"trip_id": tid, "point_idx": np.arange(d["lat"].n), "label_arc": d["label"],
                                 "ref_uncertain": ~d["clean"],
                                 "pred_classifier": preds["classifier_only (M7 argmax)"][tid],
                                 "pred_hmm_classic": preds["HMM classic"][tid],
                                 "pred_hmm_learned": preds["HMM learned transitions (N6)"][tid]}))
    pd.concat(out).to_parquet(P("results", "preds_student_test_hmm.parquet"), index=False)

    # deployment choice is made on TRAIN tuning scores (not student-test) to avoid selection leakage
    deploy = "learned" if best_l.train_top1 >= best_c.train_top1 else "classic"
    cfg = dict(deploy=deploy, classic=dict(alpha=float(best_c.alpha), beta=float(best_c.beta), floor=float(best_c.floor)),
               learned=dict(alpha=float(best_l.alpha)), selected_on="train out-of-fold Top-1",
               student_test_best=best_name)
    json.dump(cfg, open(P("models", "hmm_config.json"), "w"), indent=2)
    joblib.dump((scaler, lr), P("models", "hmm_learned_transition.joblib"))

    plt.figure(figsize=(7, 4))
    x = np.arange(len(comp))
    plt.bar(x - 0.2, comp.st_top1, 0.4, label="Top-1 accuracy")
    plt.bar(x + 0.2, comp.st_continuity, 0.4, label="Route continuity")
    for i, r in comp.iterrows():
        plt.text(i - 0.2, r.st_top1 + 0.003, f"{r.st_top1:.3f}", ha="center", fontsize=8)
        plt.text(i + 0.2, r.st_continuity + 0.003, f"{r.st_continuity:.3f}", ha="center", fontsize=8)
    plt.xticks(x, comp.variant, fontsize=8); plt.ylim(min(comp.st_top1.min(), comp.st_continuity.min()) - 0.05, 1.01)
    plt.title("Student-test: classifier-only vs HMM/Viterbi"); plt.legend(); plt.tight_layout()
    plt.savefig(P("figures", "hmm_comparison.png"), dpi=150); plt.close()
    print("config:", cfg)


if __name__ == "__main__":
    main()
