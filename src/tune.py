"""M7 tuning: Optuna TPE (Bayesian) on LightGBM LambdaRank.

Search: 50 trials, objective = mean Top-1 over 5-fold GroupKFold (by trip_id) on the 80 TRAIN trips.
Trial 0 is the untuned M7 configuration, so the baseline is scored under the exact same protocol.
Final: refit best params on all 80 train trips, evaluate once on the 10 student_test trips.
The evaluator partition is never loaded here.

Storage: Optuna file journal (results/optuna_journal.log) with the open-file lock, not SQLite
(SQLAlchemy's DLL is blocked by Application Control) and not the default symlink lock (Windows
needs a privilege for symlinks). Same effect: every trial is persisted and the run is resumable.

Usage:
    python src/tune.py --smoke      # 2 trials, in-memory, writes nothing (sanity check)
    python src/tune.py              # full run, resumable

Outputs: results/optuna_trials.csv, results/optuna_summary.json, results/optuna_journal.log,
         figures/optuna_history.png, figures/optuna_param_importance.png,
         models/M7_tuned_lgbm_rank.joblib, results/preds_student_test_M7_tuned.parquet,
         one row (model_id=M7_tuned) in results/model_results.csv
"""
import os
import sys
import json
import time
import argparse
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend, JournalFileOpenLock
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import FEATURES                                            # noqa: E402
from metrics import normalize_per_point, evaluate, predicted_sequence, continuity, append_result  # noqa: E402
from train_rank import PARAMS, load, drop_queries_without_positive, group_sizes  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a)
SEED = 42
N_TRIALS = 50
STUDY_NAME = "m7_lambdarank_tpe"


def fit_with(params, df):
    m = lgb.LGBMRanker(**params)
    m.fit(df[FEATURES], df["label"], group=group_sizes(df))
    return m


def make_folds(train):
    """Same splitter as train_rank.py (5-fold GroupKFold by trip, no shuffle) -> identical folds."""
    out = []
    for tr_i, va_i in GroupKFold(n_splits=5).split(train, groups=train["trip_id"]):
        out.append((set(train.loc[tr_i, "trip_id"]), va_i))
    return out


def suggest(trial):
    p = dict(PARAMS)
    p.update(
        n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 127),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 200),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    )
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    ap.add_argument("--smoke", action="store_true", help="2 trials, in-memory study, write nothing")
    args = ap.parse_args()
    n_trials = 2 if args.smoke else args.trials

    for d in ["models", "results", "figures"]:
        os.makedirs(P(d), exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.INFO)

    train = load("train")
    train_fit = drop_queries_without_positive(train)
    folds = make_folds(train)
    print(f"train rows={len(train)} (fit rows={len(train_fit)}), folds={len(folds)}")

    def objective(trial):
        params = suggest(trial)
        top1s, f1s, lls = [], [], []
        t0 = time.time()
        for k, (tr_trips, va_i) in enumerate(folds):
            tr = train_fit[train_fit["trip_id"].isin(tr_trips)].reset_index(drop=True)
            va = train.loc[va_i].reset_index(drop=True)
            m = fit_with(params, tr)
            prob = normalize_per_point(va, m.predict(va[FEATURES]), mode="softmax")
            r = evaluate(va, prob)
            top1s.append(r["top1"]); f1s.append(r["macro_f1"]); lls.append(r["logloss"])
            trial.set_user_attr("fold_top1", [round(x, 5) for x in top1s])
            trial.report(float(np.mean(top1s)), step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("cv_macro_f1_mean", float(np.mean(f1s)))
        trial.set_user_attr("cv_macro_f1_std", float(pd.Series(f1s).std()))
        trial.set_user_attr("cv_logloss_mean", float(np.mean(lls)))
        trial.set_user_attr("trial_seconds", round(time.time() - t0, 1))
        return float(np.mean(top1s))

    if args.smoke:
        storage = None  # in-memory
    else:
        jpath = P("results", "optuna_journal.log")
        storage = JournalStorage(JournalFileBackend(jpath, lock_obj=JournalFileOpenLock(jpath)))
    study = optuna.create_study(
        study_name=STUDY_NAME, storage=storage, load_if_exists=True, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=10),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
    )
    if len(study.trials) == 0:  # trial 0 = untuned M7 config, scored under the same CV protocol
        study.enqueue_trial(dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=50,
                                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0))

    S = optuna.trial.TrialState
    done = sum(t.state in (S.COMPLETE, S.PRUNED) for t in study.trials)
    remaining = max(0, n_trials - done)
    print(f"trials done={done}, running {remaining} more (target {n_trials})")

    def save_csv(study_, trial_):
        if not args.smoke:
            study_.trials_dataframe().to_csv(P("results", "optuna_trials.csv"), index=False)

    if remaining:
        study.optimize(objective, n_trials=remaining, n_jobs=1, callbacks=[save_csv])

    trials = [t for t in study.trials if t.state in (S.COMPLETE, S.PRUNED)]
    complete = [t for t in trials if t.state == S.COMPLETE]
    pruned = [t for t in trials if t.state == S.PRUNED]
    search_s = sum(t.duration.total_seconds() for t in trials if t.duration is not None)
    best = study.best_trial
    base = study.trials[0]
    print(f"\ncomplete={len(complete)} pruned={len(pruned)} search time={search_s/60:.1f} min")
    print(f"baseline (trial 0) CV top1={base.value}")
    print(f"best trial #{best.number} CV top1={best.value:.4f}")
    print("best params:", json.dumps(best.params))

    if args.smoke:
        print("smoke test OK, nothing written")
        return

    study.trials_dataframe().to_csv(P("results", "optuna_trials.csv"), index=False)

    # optimisation history (real trial values) + best-so-far
    plt.figure(figsize=(7, 4))
    xs = [t.number for t in complete]
    ys = [t.value for t in complete]
    plt.scatter(xs, ys, s=20, label="completed trial")
    if pruned:
        plt.scatter([t.number for t in pruned], [np.mean(t.user_attrs["fold_top1"]) for t in pruned],
                    s=20, marker="x", c="grey", label="pruned (partial CV mean)")
    plt.plot(xs, np.maximum.accumulate(ys), c="red", label="best so far")
    plt.xlabel("trial"); plt.ylabel("CV Top-1 (mean of 5 folds)")
    plt.title("M7 Optuna TPE: optimisation history"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(P("figures", "optuna_history.png"), dpi=150); plt.close()

    # parameter importance (Optuna's default evaluator, completed trials only)
    try:
        imp = optuna.importance.get_param_importances(study)
        s = pd.Series(imp).sort_values()
        plt.figure(figsize=(6, 4)); s.plot.barh()
        plt.xlabel("importance"); plt.title("M7 tuning: hyperparameter importance")
        plt.tight_layout(); plt.savefig(P("figures", "optuna_param_importance.png"), dpi=150); plt.close()
        s.to_csv(P("results", "optuna_param_importance.csv"))
    except Exception as e:
        print("param importance skipped:", e)

    # refit best params on all 80 train trips, evaluate on student_test
    params = dict(PARAMS)
    params.update(best.params)
    st = load("student_test")
    t0 = time.time()
    model = fit_with(params, train_fit)
    train_time = time.time() - t0
    t0 = time.time()
    raw = model.predict(st[FEATURES])
    infer_ms = 1000 * (time.time() - t0) / st.groupby(["trip_id", "point_idx"]).ngroups
    prob = normalize_per_point(st, raw, mode="softmax")
    r = evaluate(st, prob)
    pred = predicted_sequence(st, prob)
    cont = continuity(pred, P("data", "raw", "student_test"))
    pred.to_parquet(P("results", "preds_student_test_M7_tuned.parquet"), index=False)
    joblib.dump({"model": model, "features": FEATURES, "score_mode": "softmax"},
                P("models", "M7_tuned_lgbm_rank.joblib"))

    cfg = {k: v for k, v in params.items() if k not in ("n_jobs", "verbose")}
    append_result(dict(
        model_id="M7_tuned", family="Learning-to-rank (LightGBM LambdaRank), Optuna-tuned", config=json.dumps(cfg),
        cv_macro_f1_mean=best.user_attrs["cv_macro_f1_mean"], cv_macro_f1_std=best.user_attrs["cv_macro_f1_std"],
        cv_top1_mean=best.value,
        st_top1=r["top1"], st_macro_f1=r["macro_f1"], st_precision=r["precision"], st_recall=r["recall"],
        st_logloss=r["logloss"], st_continuity=cont, train_time_s=round(train_time, 1),
        infer_time_ms_per_point=round(infer_ms, 4), timestamp=pd.Timestamp.now().isoformat(timespec="seconds"),
    ), P("results", "model_results.csv"))

    summary = dict(
        study=STUDY_NAME, sampler="TPE (seed 42, 10 startup trials)", pruner="Median (10 startup, warmup 1 fold)",
        objective="mean Top-1, 5-fold GroupKFold by trip_id, 80 train trips",
        n_trials_total=len(trials), n_complete=len(complete), n_pruned=len(pruned),
        search_time_s=round(search_s, 1), baseline_cv_top1=base.value, best_trial=best.number,
        best_cv_top1=best.value, best_params=best.params, final_refit_time_s=round(train_time, 1),
    )
    with open(P("results", "optuna_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    res = pd.read_csv(P("results", "model_results.csv"))
    cols = ["model_id", "cv_macro_f1_mean", "cv_top1_mean", "st_top1", "st_macro_f1", "st_logloss",
            "st_continuity", "train_time_s", "infer_time_ms_per_point"]
    cmp_ = res[res["model_id"].isin(["M7", "M7_tuned"])].drop_duplicates("model_id", keep="last")
    print("\nUNTUNED vs TUNED")
    print(cmp_[cols].to_string(index=False))


if __name__ == "__main__":
    main()