"""Build the technical-report SKELETON (report structure from the Deliverables guideline).

What this script fills in automatically, from files in the repo (so every number is verifiable):
  * all tables (dataset inventory, 7-model comparison, per-model metrics, Optuna, HMM, error types)
  * all figures (figures/*.png)
What it does NOT write: explanations. Every place that needs the team's own reasoning is a yellow
"[TEAM WRITES ...]" box listing the questions the guideline requires. Replace each box with your own words.

Usage:  python report/build_report.py   -> report/PathSnap_Report_skeleton.docx
Then:   fill the yellow boxes in Word, add the UI/API screenshots, File > Save As > PDF.
"""
import os
import json
import pandas as pd
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_COLOR_INDEX

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = lambda *a: os.path.join(ROOT, *a)
R = lambda f: pd.read_csv(P("results", f))
doc = Document()
doc.styles["Normal"].font.size = Pt(10)


def team(text):
    p = doc.add_paragraph()
    run = p.add_run(f"[TEAM WRITES, in your own words] {text}")
    run.font.highlight_color = WD_COLOR_INDEX.YELLOW
    run.italic = True


def h(text, level):
    doc.add_heading(text, level=level)


def table(df, caption, fmt=None, font=8):
    fmt = fmt or {}
    t = doc.add_table(rows=1, cols=len(df.columns))
    t.style = "Light Grid Accent 1"
    for i, c in enumerate(df.columns):
        t.rows[0].cells[i].text = str(c)
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for i, c in enumerate(df.columns):
            v = row[c]
            cells[i].text = fmt[c].format(v) if c in fmt and pd.notna(v) else ("N/A" if pd.isna(v) else str(v))
    for r in t.rows:
        for c in r.cells:
            for p in c.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(font)
    cap = doc.add_paragraph(caption)
    cap.runs[0].italic = True
    cap.runs[0].font.size = Pt(8)


def fig(name, caption, width=15):
    path = P("figures", name)
    if not os.path.exists(path):
        team(f"(figure {name} missing: generate it or remove this placeholder)")
        return
    doc.add_picture(path, width=Cm(width))
    cap = doc.add_paragraph(f"{caption}  [source: figures/{name}]")
    cap.runs[0].italic = True
    cap.runs[0].font.size = Pt(8)


# ------------------------------------------------------------------ title
doc.add_heading("PathSnap: Distributed Map-Matching & Route Reconstruction", 0)
doc.add_paragraph("UE24CS352A Machine Learning: Hackathon Technical Report")
team("Team name, member names and SRNs, section, GitHub repository URL (public): https://github.com/DhanrajRathod18/ML_Hackathon")
team("1-paragraph summary: problem, approach (DP labels → candidate generation → M7 LambdaRank → learned-transition HMM → FastAPI/UI), headline student-test result.")

# ------------------------------------------------------------------ §2
h("2. Dataset Understanding and Exploratory Data Analysis", 1)
s = json.load(open(P("results", "eda_summary.json")))
inv = pd.DataFrame({p: {k: v for k, v in s[p].items() if not isinstance(v, dict)} for p in ["train", "student_test"]})
inv.index.name = "statistic"
table(inv.reset_index(), "Table 2.1: Dataset inventory and data-quality summary per partition (results/eda_summary.json, src/eda.py). "
      "Evaluator-test data was not used.", font=7)
team("2.2 How did you preprocess the data? Cover: loading per trip, metric projection (local equirectangular, why), "
     "DP alignment of GPS to the ordered .route to obtain per-point labels (greedy 3.8% vs DP 0.14% bad labels), "
     "candidate generation (50 m KD-tree, miss rate), ref_uncertain flag and why those points are excluded from training/headline metrics.")
fig("eda_sampling_gaps.png", "Figure 2.1: GPS sampling interval (train vs student-test).", 13)
fig("eda_speed.png", "Figure 2.2: GPS speed distribution.", 13)
fig("eda_trip_locations.png", "Figure 2.3: Trip locations: trips span several countries, hence a per-trip metric projection.", 13)
fig("eda_network_connectivity.png", "Figure 2.4: Road-network connectivity per trip.", 15)
fig("eda_route_vs_gps_length.png", "Figure 2.5: Reference-route length vs GPS trajectory length per trip.", 9)
fig("eda_reference_coverage.png", "Figure 2.6: Share of GPS points the reference route does not cover (ref_uncertain), 15 worst trips.", 14)
fig("eda_heading_vs_speed.png", "Figure 2.7: GPS heading vs true arc direction by speed (train, clean points).", 13)
team("2.3 Patterns observed and features extracted, and WHY (guideline Q2). Use the measured facts: 75% two-way twins (direction problem), "
     "heading reliable only above ~6 m/s (Fig 2.7), routes revisit arcs, 11.16% of train points not covered by the reference "
     "(trip 00000054: vehicle circled a 256 x 143 m site for 4.3 h). Then list the 24 features in groups (distance / direction / context / network / timing) with the reason for each.")
team("2.4 Folium map in EDA: insert a screenshot of maps/eda_train_00000054.html (GPS, network, reference, uncovered points) and maps/eda_train_00000001.html. State the partition (train).")
team("2.5 Missing values: state clearly that there were 0 missing/invalid GPS values and 0 invalid network/route references; "
     "network quirks = duplicate arcs and zero-length arcs (Table 2.1); 13 disconnected reference-route transitions in train, 1 in student-test.")

# ------------------------------------------------------------------ §3
h("3. Model Experimentation and Comparison", 1)
mr = R("model_results.csv")
base = mr[mr.model_id.isin([f"M{i}" for i in range(1, 8)])].copy()
base["order"] = base.model_id.str[1].astype(int)
base = base.sort_values("order")
cfg = base.config.astype(str).str.replace('"', "").str.slice(0, 70)
t31 = pd.DataFrame({"Model": base.model_id, "Algorithm family": base.family, "Key configuration": cfg,
                    "CV Macro-F1": base.cv_macro_f1_mean, "CV Top-1": base.cv_top1_mean, "ST Top-1": base.st_top1,
                    "Route continuity (ST)": base.st_continuity, "ST Macro-F1": base.st_macro_f1,
                    "Log loss (ST)": base.st_logloss, "Inference ms/point": base.infer_time_ms_per_point})
f4 = "{:.4f}"
table(t31, "Table 3.1: Overall model comparison (results/model_results.csv). CV = 5-fold GroupKFold by trip on 80 train trips; "
      "ST = 10 student-test trips; clean points; all models scored with src/metrics.py.",
      {c: f4 for c in ["CV Macro-F1", "CV Top-1", "ST Top-1", "Route continuity (ST)", "ST Macro-F1", "Log loss (ST)"]}, font=7)
team("State the primary selection metric (observation-level Top-1) and why; define Macro-F1 as used here (argmax candidate per point = predicted positive).")

h("3.2 Compulsory questions for every model", 2)
owner = {"M1": "P3", "M2": "P3", "M3": "P3", "M4": "P3", "M5": "P2", "M6": "P2", "M7": "P1"}
for _, r in base.iterrows():
    h(f"{r.model_id}: {r.family}  (owner {owner.get(r.model_id, '?')})", 3)
    m = pd.DataFrame([{"CV Top-1": r.cv_top1_mean, "CV Macro-F1 (mean ± sd)": f"{r.cv_macro_f1_mean:.4f} ± {r.cv_macro_f1_std:.4f}",
                       "ST Top-1": r.st_top1, "ST Macro-F1": r.st_macro_f1, "ST Precision": r.st_precision,
                       "ST Recall": r.st_recall, "ST Log loss": r.st_logloss, "Train s": r.train_time_s}])
    table(m, f"Table: {r.model_id} measured results (results/model_results.csv). Configuration: {r.config}",
          {c: f4 for c in ["CV Top-1", "ST Top-1", "ST Macro-F1", "ST Precision", "ST Recall", "ST Log loss"]})
    team("A1 Why is this algorithm suitable for PathSnap?  A2 Major strengths and limitations FOR THIS TASK (link to your results).  "
         "B5 Important hyperparameters and initial values.  B6 Loss function / optimisation objective.  "
         "C7-C8 Student-test performance (use the table above).  C9 Over/underfitting: what evidence?")
    if r.model_id == "M7":
        fig("M7_learning_curve.png", "M7 boosting-iteration history, fold 0 (real training history).", 12)
        fig("M7_feature_importance.png", "M7 feature importance (gain).", 10)
    if r.model_id == "M6":
        fig("M6_loss_curve.png", "M6 MLP training/validation loss per epoch.", 12)

h("3.3 Model visualisations", 2)
for f_, c in [("cmp_cv_macro_f1.png", "CV Macro-F1 across all models"), ("cmp_st_top1.png", "Student-test Top-1 across models"),
              ("cmp_st_macro_f1.png", "Student-test Macro-F1 across models"), ("cmp_cv_vs_test_top1.png", "CV vs student-test Top-1"),
              ("cmp_logloss.png", "Log loss across models"), ("cmp_infer_time.png", "Inference time across models"),
              ("cmp_train_time.png", "Training time across models")]:
    fig(f_, f"Figure: {c}.", 12)
h("Confusion matrices (optional diagnostics)", 3)
for i in range(1, 8):
    fig(f"M{i}_confusion.png", f"M{i} confusion matrix, student-test.", 7)
team("Justify the model selection from these plots (generalisation, fold stability, probability quality, speed, compatibility with the HMM).")

# ------------------------------------------------------------------ §4
h("4. Bayesian Hyperparameter Optimisation (M7)", 1)
o = json.load(open(P("results", "optuna_summary.json")))
table(pd.DataFrame([{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in o.items() if k != "best_params"}]).T.reset_index()
      .rename(columns={"index": "item", 0: "value"}), "Table 4.1: Optuna study summary (results/optuna_summary.json).", font=7)
table(pd.DataFrame([o["best_params"]]).T.reset_index().rename(columns={"index": "hyperparameter", 0: "best value"}),
      "Table 4.2: Best hyperparameters (trial %s)." % o.get("best_trial"))
tu = mr[mr.model_id.isin(["M7", "M7_tuned"])][["model_id", "cv_macro_f1_mean", "cv_top1_mean", "st_top1", "st_macro_f1",
                                                "st_logloss", "infer_time_ms_per_point"]]
table(tu.rename(columns={"model_id": "Model", "cv_macro_f1_mean": "CV Macro-F1", "cv_top1_mean": "CV Top-1", "st_top1": "ST Top-1",
                         "st_macro_f1": "ST Macro-F1", "st_logloss": "Log loss", "infer_time_ms_per_point": "Inference ms/pt"}),
      "Table 4.3: Untuned vs tuned M7, same CV protocol and same student-test partition (results/model_results.csv).",
      {c: f4 for c in ["CV Macro-F1", "CV Top-1", "ST Top-1", "ST Macro-F1", "Log loss"]})
fig("optuna_history.png", "Figure 4.1: Optimisation history.", 12)
fig("optuna_param_importance.png", "Figure 4.2: Hyperparameter importance.", 11)
team("Answer: 1) which hyperparameters, 2) ranges/distributions (see src/tune.py), 3) objective, 4) number of trials, 5) best values, "
     "6) did tuned beat untuned? (CV +0.0014, far below the fold s.d. 0.0275; ST equal), so we kept the untuned M7 and why.")

# ------------------------------------------------------------------ §5
h("5. HMM and Viterbi Route Reconstruction", 1)
hc = R("hmm_comparison.csv")
table(hc, "Table 5.1: Classifier-only vs HMM decoding on student-test (results/hmm_comparison.csv). Decoder tuned on train out-of-fold scores only.",
      {"st_top1": f4, "st_continuity": f4, "decode_ms_per_point": "{:.4f}"})
table(R("hmm_grid_learned.csv"), "Table 5.2: Learned-transition HMM tuning on TRAIN out-of-fold scores (results/hmm_grid_learned.csv).",
      {"train_top1": f4, "train_continuity": f4})
table(R("hmm_grid_classic.csv").head(6), "Table 5.3: Classic HMM grid, best 6 of 18 settings on TRAIN (results/hmm_grid_classic.csv).",
      {"train_top1": f4, "train_continuity": f4})
coef = R("hmm_learned_transition_coef.csv")
coef.columns = ["transition feature", "coefficient"]
table(coef, "Table 5.4: Learned transition model coefficients (standardised features).", {"coefficient": "{:.3f}"})
table(R("hmm_error_types.csv"), "Table 5.5: Error types, % of clean student-test points (results/hmm_error_types.csv).")
fig("hmm_comparison.png", "Figure 5.1: Classifier-only vs HMM on student-test.", 12)
team("Answer: 1) hidden states, 2) emission probabilities, 3) transition probabilities (classic vs learned), 4) missing candidates / large gaps, "
     "5) disconnected segments, 6) did the HMM improve the route? Include where it did NOT help (trip 00000077) and the Viterbi recurrence.")

# ------------------------------------------------------------------ §6
h("6. Geographical Visualisation Using Folium", 1)
team("Insert screenshots of maps/route_student_test_00000040.html (and 00000077 / 00000056): raw GPS, road network, reference route, "
     "predicted route, legend and layer control visible. Generated by src/make_maps.py. Add P2's error-taxonomy maps (maps/N8_*.html) and discussion.")
fig("N8_errors_by_trip.png", "Figure 6.1: Errors by student-test trip (P2 error taxonomy).", 12)

# ------------------------------------------------------------------ §7
h("7. API, UI and Deployment", 1)
team("Screenshot of the working UI (Laptop 2) with a predicted route and metrics; screenshot of the API request logs on the CMD of "
     "Laptop 1 (receiving laptop). Briefly describe /trips and /predict, server-side scoring and that ground truth never reaches the UI.")

# ------------------------------------------------------------------ §8
h("8. Academic Integrity and AI-Use Disclosure", 1)
team("Disclose how AI tools were used (e.g. code assistance), confirm all results come from the code in the repository and that the text was written by the team.")

os.makedirs(P("report"), exist_ok=True)
out = P("report", "PathSnap_Report_skeleton.docx")
doc.save(out)
print("saved", out)
