"""M7: LightGBM LambdaRank, the learning-to-rank model family (novelty N4).

Why ranking: at every GPS point exactly ONE candidate is correct, and the task is to
put it first. LambdaRank optimises NDCG@1 per query (query = one GPS point), which is
exactly observation-level Top-1, while binary classifiers optimise per-row log loss
and spend capacity on easy far-away negatives.

Usage:  python src/train_rank.py
Outputs: models/M7_lgbm_rank.joblib, results/model_results.csv (one row),
         results/M7_cv_folds.csv, results/preds_student_test_M7.parquet,
         figures/M7_learning_curve.png
"""
import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import FEATURES                                            # noqa: E402
from metrics import normalize_per_point, evaluate, predicted_sequence, continuity, append_result  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a)
SEED = 42
PARAMS = dict(objective="lambdarank", metric="ndcg", eval_at=[1], n_estimators=400, learning_rate=0.05,
              num_leaves=63, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbose=-1)


def load(part):
    df = pd.read_parquet(P("data", "processed", f"{part}_cands.parquet"))
    df = df[~df["ref_uncertain"]]
    return df.sort_values(["trip_id", "point_idx", "dist_rank"]).reset_index(drop=True)


def drop_queries_without_positive(df):
    """A point whose true arc is not among its candidates teaches the ranker nothing (0.47 % of points)."""
    has_pos = df.groupby(["trip_id", "point_idx"])["label"].transform("max") > 0
    return df[has_pos].reset_index(drop=True)


def group_sizes(df):
    return df.groupby(["trip_id", "point_idx"], sort=False).size().values


def fit(df, eval_df=None):
    m = lgb.LGBMRanker(**PARAMS)
    kw = {}
    if eval_df is not None:
        kw = dict(eval_set=[(df[FEATURES], df["label"]), (eval_df[FEATURES], eval_df["label"])],
                  eval_group=[group_sizes(df), group_sizes(eval_df)], eval_names=["train", "valid"])
    m.fit(df[FEATURES], df["label"], group=group_sizes(df), **kw)
    return m


def main():
    for d in ["models", "results", "figures"]:
        os.makedirs(P(d), exist_ok=True)
    train = load("train")
    st = load("student_test")
    train_fit = drop_queries_without_positive(train)
    print(f"train rows={len(train)} (fit rows={len(train_fit)}), student_test rows={len(st)}")

    # 5-fold GroupKFold by trip: no trip is in both fit and validation
    folds = []
    gkf = GroupKFold(n_splits=5)
    for k, (tr_i, va_i) in enumerate(gkf.split(train, groups=train["trip_id"])):
        tr_trips = set(train.loc[tr_i, "trip_id"])
        tr = train_fit[train_fit["trip_id"].isin(tr_trips)].reset_index(drop=True)
        va = train.loc[va_i].reset_index(drop=True)
        t0 = time.time()
        m = fit(tr, eval_df=drop_queries_without_positive(va) if k == 0 else None)
        prob = normalize_per_point(va, m.predict(va[FEATURES]), mode="softmax")
        r = evaluate(va, prob)
        r.update(fold=k, n_val_trips=va["trip_id"].nunique(), fit_s=round(time.time() - t0, 1))
        folds.append(r)
        print(f"  fold {k}: top1={r['top1']:.4f} macroF1={r['macro_f1']:.4f} logloss={r['logloss']:.4f} ({r['fit_s']}s)")
        if k == 0:  # boosting-iteration history (real, not artificial) for the learning-curve plot
            hist = m.evals_result_
            plt.figure(figsize=(6, 4))
            for name in ["train", "valid"]:
                plt.plot(hist[name]["ndcg@1"], label=f"{name} (fold 0)")
            plt.xlabel("boosting iteration"); plt.ylabel("NDCG@1 (= Top-1 on points with a true candidate)")
            plt.title("M7 LambdaRank: training vs validation"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
            plt.savefig(P("figures", "M7_learning_curve.png"), dpi=150); plt.close()
    fdf = pd.DataFrame(folds)
    fdf.to_csv(P("results", "M7_cv_folds.csv"), index=False)

    # final fit on all 80 training trips, evaluate on the 10 student-test trips
    t0 = time.time()
    model = fit(train_fit)
    train_time = time.time() - t0
    t0 = time.time()
    raw = model.predict(st[FEATURES])
    infer_ms = 1000 * (time.time() - t0) / st.groupby(["trip_id", "point_idx"]).ngroups
    prob = normalize_per_point(st, raw, mode="softmax")
    r = evaluate(st, prob)
    pred = predicted_sequence(st, prob)
    cont = continuity(pred, P("data", "raw", "student_test"))
    pred.to_parquet(P("results", "preds_student_test_M7.parquet"), index=False)
    joblib.dump({"model": model, "features": FEATURES, "score_mode": "softmax"}, P("models", "M7_lgbm_rank.joblib"))

    cfg = {k: v for k, v in PARAMS.items() if k not in ("n_jobs", "verbose")}
    append_result(dict(
        model_id="M7", family="Learning-to-rank (LightGBM LambdaRank)", config=json.dumps(cfg),
        cv_macro_f1_mean=fdf.macro_f1.mean(), cv_macro_f1_std=fdf.macro_f1.std(), cv_top1_mean=fdf.top1.mean(),
        st_top1=r["top1"], st_macro_f1=r["macro_f1"], st_precision=r["precision"], st_recall=r["recall"],
        st_logloss=r["logloss"], st_continuity=cont, train_time_s=round(train_time, 1),
        infer_time_ms_per_point=round(infer_ms, 4), timestamp=pd.Timestamp.now().isoformat(timespec="seconds"),
    ), P("results", "model_results.csv"))

    imp = pd.Series(model.booster_.feature_importance("gain"), index=FEATURES).sort_values()
    imp.to_csv(P("results", "M7_feature_importance.csv"))
    plt.figure(figsize=(6, 6)); imp.plot.barh(); plt.title("M7 feature importance (gain)"); plt.tight_layout()
    plt.savefig(P("figures", "M7_feature_importance.png"), dpi=150); plt.close()

    print(f"CV top1={fdf.top1.mean():.4f}±{fdf.top1.std():.4f}  macroF1={fdf.macro_f1.mean():.4f}±{fdf.macro_f1.std():.4f}")
    print(f"STUDENT_TEST top1={r['top1']:.4f} macroF1={r['macro_f1']:.4f} P={r['precision']:.4f} R={r['recall']:.4f} "
          f"logloss={r['logloss']:.4f} continuity={cont:.4f} train={train_time:.0f}s infer={infer_ms:.3f}ms/pt")


if __name__ == "__main__":
    main()
