"""Comparison plots from results/model_results.csv (real numbers only, nothing typed in by hand).

Usage: python src/plot_comparison.py
Outputs: figures/cmp_*.png and results/comparison_table.csv
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a)

df = pd.read_csv(P("results", "model_results.csv"))
# if a model was re-run, keep only its latest row
df = df.sort_values("timestamp").drop_duplicates("model_id", keep="last")
df = df.sort_values("model_id").reset_index(drop=True)
df.to_csv(P("results", "comparison_table.csv"), index=False)
os.makedirs(P("figures"), exist_ok=True)
labels = df["model_id"].tolist()
best = df["st_top1"].idxmax()


def bar(col, title, ylabel, fname, err=None, log=False, lower_better=False):
    plt.figure(figsize=(7, 4))
    bars = plt.bar(labels, df[col], yerr=None if err is None else df[err], capsize=4)
    pick = df[col].idxmin() if lower_better else df[col].idxmax()
    bars[pick].set_edgecolor("black"); bars[pick].set_linewidth(2)
    for b, v in zip(bars, df[col]):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.3f}" if v < 100 else f"{v:.0f}",
                 ha="center", va="bottom", fontsize=8)
    if log:
        plt.yscale("log")
    plt.title(title); plt.ylabel(ylabel); plt.xlabel("model"); plt.grid(axis="y", alpha=.3)
    plt.tight_layout(); plt.savefig(P("figures", fname), dpi=150); plt.close()


bar("cv_macro_f1_mean", "5-fold GroupKFold CV macro-F1 (error bar = std over folds)", "macro-F1", "cmp_cv_macro_f1.png", err="cv_macro_f1_std")
bar("st_top1", "Student-test observation-level Top-1 accuracy", "Top-1", "cmp_st_top1.png")
bar("st_macro_f1", "Student-test macro-F1", "macro-F1", "cmp_st_macro_f1.png")
bar("st_logloss", "Student-test log loss (lower is better)", "log loss", "cmp_logloss.png", lower_better=True)
bar("infer_time_ms_per_point", "Inference time per GPS point (log scale)", "ms / point", "cmp_infer_time.png", log=True, lower_better=True)
bar("train_time_s", "Final-fit training time (log scale)", "seconds", "cmp_train_time.png", log=True, lower_better=True)

# CV vs student-test Top-1 side by side: a big gap would indicate overfitting to the training trips
plt.figure(figsize=(7, 4))
x = range(len(labels)); w = 0.38
plt.bar([i - w / 2 for i in x], df["cv_top1_mean"], w, label="CV Top-1 (train trips)")
plt.bar([i + w / 2 for i in x], df["st_top1"], w, label="Student-test Top-1")
plt.xticks(list(x), labels); plt.ylim(df[["cv_top1_mean", "st_top1"]].min().min() - 0.03, 1)
plt.legend(); plt.title("Generalisation check: CV vs student-test Top-1"); plt.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig(P("figures", "cmp_cv_vs_test_top1.png"), dpi=150); plt.close()

cols = ["model_id", "family", "cv_macro_f1_mean", "cv_top1_mean", "st_top1", "st_macro_f1", "st_logloss",
        "st_continuity", "train_time_s", "infer_time_ms_per_point"]
print(df[cols].round(4).to_string(index=False))
print(f"\nbest student-test Top-1: {df.loc[best, 'model_id']}")