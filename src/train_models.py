"""M1-M4 (P3): Logistic Regression, Gaussian NB, distance-weighted KNN, Random Forest.

Same protocol as train_rank.py (M7) so the rows in results/model_results.csv are comparable:
- ref_uncertain points are excluded (LOCKED policy)
- 5-fold GroupKFold by trip_id on the 80 train trips (no trip is in both fit and validation)
- fit on all positives + <= 4 random negatives per GPS point (negative subsampling, fitting only)
- evaluation ALWAYS uses all candidates; scores are normalised per point with mode="prob"

Usage:  python src/train_models.py --models M1,M2,M3,M4
"""
import os
import sys
import json
import time
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import FEATURES                                                       # noqa: E402
from metrics import (normalize_per_point, evaluate, predicted_sequence, continuity,  # noqa: E402
                     append_result, argmax_flags)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a)
SEED = 42
KEYS = ["trip_id", "point_idx", "dist_rank"]


def load(part):
    df = pd.read_parquet(P("data", "processed", f"{part}_cands.parquet"))
    df = df[~df["ref_uncertain"]]          # reference route does not cover these points -> not trusted
    return df.sort_values(KEYS).reset_index(drop=True)


def drop_queries_without_positive(df):
    has_pos = df.groupby(["trip_id", "point_idx"])["label"].transform("max") > 0
    return df[has_pos].reset_index(drop=True)


def subsample_negatives(df, rng, k=4):
    """Keep every positive row and at most k random negative rows per GPS point (fitting only)."""
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0].copy()
    neg["_r"] = rng.random(len(neg))
    neg["_rk"] = neg.groupby(["trip_id", "point_idx"])["_r"].rank(method="first")
    neg = neg[neg["_rk"] <= k].drop(columns=["_r", "_rk"])
    return pd.concat([pos, neg]).sort_values(KEYS).reset_index(drop=True)


def trip_subsample(df, frac, rng):
    trips = df["trip_id"].unique()
    keep = rng.choice(trips, size=max(1, int(round(frac * len(trips)))), replace=False)
    return df[df["trip_id"].isin(keep)]


def point_subsample(df, frac, seed):
    """Random subset of GPS points (all candidates of a chosen point are kept)."""
    if frac >= 1:
        return df
    keys = df[["trip_id", "point_idx"]].drop_duplicates().sample(frac=frac, random_state=seed)
    return df.merge(keys, on=["trip_id", "point_idx"]).sort_values(KEYS).reset_index(drop=True)


def X(df):
    return df[FEATURES].astype("float32").to_numpy()


def scores(model, df, chunk=50000):
    Xm = X(df)
    return np.concatenate([model.predict_proba(Xm[i:i + chunk])[:, 1] for i in range(0, len(Xm), chunk)])


def build(mid, a):
    """returns (fresh model, family, config dict). Imputer = median fitted on the fit rows only."""
    imp = SimpleImputer(strategy="median")
    if mid == "M1":
        cfg = dict(C=1.0, max_iter=500, scaler="StandardScaler", neg_per_point=4)
        return (make_pipeline(imp, StandardScaler(), LogisticRegression(C=cfg["C"], max_iter=cfg["max_iter"],
                random_state=SEED)), "Linear (Logistic Regression)", cfg)
    if mid == "M2":
        cfg = dict(var_smoothing=1e-9, neg_per_point=4)
        return make_pipeline(imp, GaussianNB(var_smoothing=cfg["var_smoothing"])), "Probabilistic (Gaussian NB)", cfg
    if mid == "M3":
        cfg = dict(n_neighbors=a.knn_k, weights="distance", scaler="StandardScaler", train_trip_frac=a.knn_frac,
                   cv_val_point_frac=a.knn_cv_val_frac, neg_per_point=4)
        return (make_pipeline(imp, StandardScaler(), KNeighborsClassifier(n_neighbors=cfg["n_neighbors"],
                weights="distance", n_jobs=-1)), "Instance-based (distance-weighted KNN)", cfg)
    if mid == "M4":
        cfg = dict(n_estimators=a.rf_trees, min_samples_leaf=3, max_samples=0.3, neg_per_point=4)
        return (make_pipeline(imp, RandomForestClassifier(n_estimators=cfg["n_estimators"],
                min_samples_leaf=cfg["min_samples_leaf"], max_samples=cfg["max_samples"], n_jobs=-1,
                random_state=SEED)), "Ensemble (Random Forest, bagging)", cfg)
    raise ValueError(mid)


def fit_rows(df, mid, a, rng):
    if mid == "M3":                       # KNN is slow at predict time -> fit on a trip subsample
        df = trip_subsample(df, a.knn_frac, rng)
    return subsample_negatives(df, rng)


def run(mid, train, train_fit, st, a):
    print(f"\n===== {mid} =====")
    rng = np.random.default_rng(SEED)
    folds = []
    for k, (tr_i, va_i) in enumerate(GroupKFold(n_splits=a.folds).split(train, groups=train["trip_id"])):
        tr = fit_rows(train_fit[train_fit["trip_id"].isin(set(train.loc[tr_i, "trip_id"]))], mid, a, rng)
        va = train.loc[va_i].reset_index(drop=True)
        if mid == "M3":
            va = point_subsample(va, a.knn_cv_val_frac, SEED + k)
        model, family, cfg = build(mid, a)
        t0 = time.time()
        model.fit(X(tr), tr["label"].values)
        fit_s = time.time() - t0
        prob = normalize_per_point(va, scores(model, va), mode="prob")
        r = evaluate(va, prob)
        r.update(fold=k, n_val_trips=va["trip_id"].nunique(), fit_s=round(fit_s, 1))
        folds.append(r)
        print(f"  fold {k}: top1={r['top1']:.4f} macroF1={r['macro_f1']:.4f} logloss={r['logloss']:.4f} ({fit_s:.0f}s)", flush=True)
    fdf = pd.DataFrame(folds)
    fdf.to_csv(P("results", f"{mid}_cv_folds.csv"), index=False)

    # final fit on all 80 train trips -> evaluate on the 10 student_test trips (all candidates)
    tr = fit_rows(train_fit, mid, a, rng)
    model, family, cfg = build(mid, a)
    t0 = time.time()
    model.fit(X(tr), tr["label"].values)
    train_time = time.time() - t0
    t0 = time.time()
    raw = scores(model, st)
    infer_ms = 1000 * (time.time() - t0) / st.groupby(["trip_id", "point_idx"]).ngroups
    prob = normalize_per_point(st, raw, mode="prob")
    r = evaluate(st, prob)
    pred = predicted_sequence(st, prob)
    cont = continuity(pred, P("data", "raw", "student_test"))
    pred.to_parquet(P("results", f"preds_student_test_{mid}.parquet"), index=False)
    joblib.dump({"model": model, "features": FEATURES, "score_mode": "prob"}, P("models", f"{mid}.joblib"))

    # confusion matrix at candidate level: was the top-1 selected candidate the true arc or not
    flags = argmax_flags(st, prob).astype(int)
    cm = confusion_matrix(st["label"].values, flags, labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=["not true arc", "true arc"]).plot(values_format="d")
    plt.title(f"{mid} student_test: true label vs top-1 selected"); plt.tight_layout()
    plt.savefig(P("figures", f"{mid}_confusion.png"), dpi=150); plt.close()

    append_result(dict(
        model_id=mid, family=family, config=json.dumps(cfg),
        cv_macro_f1_mean=fdf.macro_f1.mean(), cv_macro_f1_std=fdf.macro_f1.std(), cv_top1_mean=fdf.top1.mean(),
        st_top1=r["top1"], st_macro_f1=r["macro_f1"], st_precision=r["precision"], st_recall=r["recall"],
        st_logloss=r["logloss"], st_continuity=cont, train_time_s=round(train_time, 1),
        infer_time_ms_per_point=round(infer_ms, 4), timestamp=pd.Timestamp.now().isoformat(timespec="seconds"),
    ), P("results", "model_results.csv"))
    print(f"CV top1={fdf.top1.mean():.4f} +/- {fdf.top1.std():.4f}  macroF1={fdf.macro_f1.mean():.4f} +/- {fdf.macro_f1.std():.4f}")
    print(f"STUDENT_TEST top1={r['top1']:.4f} macroF1={r['macro_f1']:.4f} P={r['precision']:.4f} R={r['recall']:.4f} "
          f"logloss={r['logloss']:.4f} continuity={cont:.4f} train={train_time:.0f}s infer={infer_ms:.3f}ms/pt", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="M1,M2,M3,M4")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--rf_trees", type=int, default=300)
    ap.add_argument("--knn_k", type=int, default=15)
    ap.add_argument("--knn_frac", type=float, default=0.10, help="share of train trips KNN is fitted on")
    ap.add_argument("--knn_cv_val_frac", type=float, default=0.20, help="share of validation points KNN scores in CV")
    a = ap.parse_args()
    for d in ["models", "results", "figures"]:
        os.makedirs(P(d), exist_ok=True)
    train, st = load("train"), load("student_test")
    train_fit = drop_queries_without_positive(train)
    print(f"train rows={len(train)} (fit rows={len(train_fit)}), student_test rows={len(st)}")
    for mid in [m.strip() for m in a.models.split(",")]:
        run(mid, train, train_fit, st, a)


if __name__ == "__main__":
    main()