"""Shared evaluation metrics: EVERY model must be scored with these functions so the comparison is fair.

Conventions (state them in the report):
  * Evaluation unit = clean GPS points (ref_uncertain == False); all candidates of a point are scored.
  * Scores are normalised PER POINT so candidate probabilities sum to 1
    (classifiers: p / sum(p); rankers: softmax of raw scores). This is novelty N5.
  * Top-1 accuracy: argmax candidate per point == label_arc.
  * Candidate-level Macro-F1 / precision / recall: the argmax candidate of each point is
    predicted positive, the rest negative (one decision per point, threshold-free, so
    classifiers and rankers are comparable). Macro = mean F1 of class 0 and class 1.
  * Log loss: binary log loss of the normalised candidate probabilities vs label.
  * Route continuity: share of consecutive point pairs whose predicted arcs are the
    same arc or connected (arc_a.to_node == arc_b.from_node).
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, log_loss

RESULT_COLUMNS = ["model_id", "family", "config", "cv_macro_f1_mean", "cv_macro_f1_std", "cv_top1_mean",
                  "st_top1", "st_macro_f1", "st_precision", "st_recall", "st_logloss", "st_continuity",
                  "train_time_s", "infer_time_ms_per_point", "timestamp"]


def point_keys(df):
    return df["trip_id"].astype(str).values, df["point_idx"].values


def normalize_per_point(df, scores, mode="prob"):
    """Return per-point normalised probabilities. mode='prob' (p/sum p) or 'softmax' (raw scores)."""
    s = pd.Series(np.asarray(scores, dtype=float), index=df.index)
    g = [df["trip_id"].values, df["point_idx"].values]
    if mode == "softmax":
        s = s - s.groupby(g).transform("max")
        e = np.exp(s)
        return (e / e.groupby(g).transform("sum")).values
    s = s.clip(lower=1e-12)
    return (s / s.groupby(g).transform("sum")).values


def argmax_flags(df, prob):
    """1 for the highest-probability candidate of each point, else 0."""
    tmp = pd.DataFrame({"t": df["trip_id"].values, "p": df["point_idx"].values, "s": prob})
    idx = tmp.groupby(["t", "p"])["s"].idxmax().values
    flag = np.zeros(len(df), dtype=np.int8)
    flag[idx] = 1
    return flag


def evaluate(df, prob):
    """df must have trip_id, point_idx, cand_arc, label, label_arc. prob = normalised probabilities."""
    flag = argmax_flags(df, prob)
    y = df["label"].values
    n_points = df.groupby(["trip_id", "point_idx"]).ngroups
    top1 = (flag & y).sum() / n_points
    return {
        "top1": float(top1),
        "macro_f1": float(f1_score(y, flag, average="macro")),
        "precision": float(precision_score(y, flag, zero_division=0)),
        "recall": float(recall_score(y, flag, zero_division=0)),
        "logloss": float(log_loss(y, np.clip(prob, 1e-7, 1 - 1e-7), labels=[0, 1])),
    }


def predicted_sequence(df, prob):
    """Point-level prediction table: trip_id, point_idx, pred_arc, pred_prob, label_arc."""
    flag = argmax_flags(df, prob).astype(bool)
    out = df.loc[flag, ["trip_id", "point_idx", "cand_arc", "label_arc"]].copy()
    out["pred_prob"] = np.asarray(prob)[flag]
    return out.rename(columns={"cand_arc": "pred_arc"}).sort_values(["trip_id", "point_idx"]).reset_index(drop=True)


def continuity(pred, raw_partition_dir):
    """pred: output of predicted_sequence. Share of consecutive pairs that are same/connected arcs."""
    ok = tot = 0
    for tid, grp in pred.groupby("trip_id"):
        arcs = np.loadtxt(os.path.join(raw_partition_dir, tid, f"{tid}.arcs"), dtype=np.int64, ndmin=2)
        a = grp.sort_values("point_idx")["pred_arc"].values
        if len(a) < 2:
            continue
        same = a[1:] == a[:-1]
        conn = arcs[a[:-1], 1] == arcs[a[1:], 0]
        ok += int((same | conn).sum())
        tot += len(a) - 1
    return ok / max(tot, 1)


def append_result(row, path):
    row = {k: row.get(k, np.nan) for k in RESULT_COLUMNS}
    df = pd.DataFrame([row], columns=RESULT_COLUMNS)
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
