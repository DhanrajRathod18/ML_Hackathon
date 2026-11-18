"""Confusion matrix for M7 on the student-test clean points (same convention as M1-M6 and src/metrics.py:
the highest-scoring candidate of each GPS point is the predicted positive, all other candidates negative).
Usage: python src/m7_confusion.py  -> figures/M7_confusion.png
"""
import os
import sys
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import normalize_per_point, argmax_flags   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

b = joblib.load(os.path.join(ROOT, "models", "M7_lgbm_rank.joblib"))
st = pd.read_parquet(os.path.join(ROOT, "data", "processed", "student_test_cands.parquet"))
st = st[~st.ref_uncertain].reset_index(drop=True)
flag = argmax_flags(st, normalize_per_point(st, b["model"].predict(st[b["features"]]), "softmax"))
cm = confusion_matrix(st.label, flag)
print(cm)
ConfusionMatrixDisplay(cm, display_labels=["not true arc", "true arc"]).plot(values_format="d", cmap="Blues")
plt.title("M7 LambdaRank: student-test candidate-level\n(argmax per point = predicted positive)")
plt.tight_layout()
plt.savefig(os.path.join(ROOT, "figures", "M7_confusion.png"), dpi=150)
