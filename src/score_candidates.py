"""Score every candidate with M7 so the HMM can be tuned without leakage.

* train: OUT-OF-FOLD scores, using the same 5 GroupKFold trip splits as train_rank.py. Each
  trip is scored by a model that never saw it, so the HMM parameters are tuned on realistic,
  not over-confident, emissions.
* student_test: scored by the final M7 model (models/M7_lgbm_rank.joblib).
All points are scored, including ref_uncertain ones: the HMM needs the full sequence, and those
points are only excluded when computing metrics.
Usage: python src/score_candidates.py  -> data/processed/{train,student_test}_scored.parquet
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import FEATURES                                   # noqa: E402
from metrics import normalize_per_point                         # noqa: E402
from train_rank import load, drop_queries_without_positive, fit, P   # noqa: E402

KEEP = ["trip_id", "point_idx", "cand_arc", "proj_frac", "dist_m", "label", "label_arc", "ref_uncertain"]


def load_all(part):
    df = pd.read_parquet(P("data", "processed", f"{part}_cands.parquet"))
    return df.sort_values(["trip_id", "point_idx", "dist_rank"]).reset_index(drop=True)


def main():
    clean = load("train")
    fit_rows = drop_queries_without_positive(clean)
    full = load_all("train")
    full["score"] = np.nan
    for k, (tr_i, va_i) in enumerate(GroupKFold(n_splits=5).split(clean, groups=clean["trip_id"])):
        tr_trips = set(clean.loc[tr_i, "trip_id"])
        va_trips = set(clean.loc[va_i, "trip_id"])
        m = fit(fit_rows[fit_rows["trip_id"].isin(tr_trips)].reset_index(drop=True))
        sel = full["trip_id"].isin(va_trips)
        full.loc[sel, "score"] = m.predict(full.loc[sel, FEATURES])
        print(f"fold {k}: scored {sel.sum()} rows of {len(va_trips)} held-out trips")
    # trips that have no clean points never appear in the clean split -> score them with a model that excludes them
    rest = full["score"].isna()
    if rest.any():
        m = fit(fit_rows[~fit_rows["trip_id"].isin(set(full.loc[rest, "trip_id"]))].reset_index(drop=True))
        full.loc[rest, "score"] = m.predict(full.loc[rest, FEATURES])
    full["prob"] = normalize_per_point(full, full["score"].values, mode="softmax")
    full[KEEP + ["prob"]].to_parquet(P("data", "processed", "train_scored.parquet"), index=False)

    st = load_all("student_test")
    bundle = joblib.load(P("models", "M7_lgbm_rank.joblib"))
    st["prob"] = normalize_per_point(st, bundle["model"].predict(st[FEATURES]), mode="softmax")
    st[KEEP + ["prob"]].to_parquet(P("data", "processed", "student_test_scored.parquet"), index=False)
    print("saved train_scored / student_test_scored")


if __name__ == "__main__":
    main()
