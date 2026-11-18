# PathSnap: Team Plan + LLM Handoff (read this first)

> **For any LLM picking this up:** this file is the single source of truth. Read it fully, check the STATUS
> checklist at the bottom, and continue from the first unchecked item. Do NOT redesign anything marked LOCKED.
> Do NOT touch evaluator-test data or ground truth except inside the API's scoring step.
> Interfaces in §4 are a contract: never rename columns or functions.

**Hard deadline: ALL models trained by 12:30 (before the break).** It is 10:50 at the time of writing.

---

## 0. The task (from the 5 PDFs)

Map-matching: snap noisy 1 Hz GPS points to road segments (arcs), clean up the sequence with HMM + Viterbi, and serve it through
**FastAPI on Laptop 1**, called by a **UI on Laptop 2** (the API console must show request logs; ground truth never reaches the UI).

| File | Columns (no header) | Facts we verified |
|---|---|---|
| `.track` | lon, lat, t(s) | 1 Hz (~97%), 1.3k–7k points per trip |
| `.nodes` | lon, lat | row = node_id. **Each trip has its OWN road network** (8k–89k nodes) |
| `.arcs` | from_node, to_node | row = arc_id, directed (a two-way road = 2 arcs) |
| `.route` | arc_id | **Ordered arc sequence, NOT one label per point** (1315 points vs 525 arcs) |

train = 80 trips (~208k points), student_test = 10 trips (~19k points), evaluator test = 10 trips (API only).
Trips span countries (8°E, 54°E, ...), so **project each trip to meters** (local equirectangular around the trip centroid).

Mandatory: EDA + Folium · ≥5 models · Optuna on GroupKFold with the trials log · HMM vs classifier-only · API + UI on 2 laptops ·
public GitHub + PDF report · no hardcoding/leakage · report in own words · everyone can explain everything (viva).

---

## 0.5 MEASURED facts that drive the design (probe run on student_test, 19,160 points, 11:05)

| Finding | Number | Consequence |
|---|---|---|
| Consecutive route arcs are connected | 3466 / 3467 | The route is a valid network path, so the HMM's connectivity assumption holds |
| Routes revisit arcs (loops/U-turns) | 255 repeats in 10 trips | Labelling with the nearest route arc breaks (23% of points > 50 m off) |
| Greedy monotonic labels (window 15) | 3.8% of labels > 50 m away | Too fragile. Window 5 gives 0.27%, window 40 gives 8.2% (very sensitive) |
| **DP (Viterbi-style) optimal alignment of GPS to route** | **0.14% > 50 m, mean 6.7 m, identical for step 3/8/20** | **LOCKED label method**: robust and globally optimal |
| Candidate miss rate, R = 50 m (with DP labels) | 0.14%, ~11 candidates/point | R = 50 m is enough, so topology expansion is NOT needed (dropped) |
| True arc has a reverse twin (two-way road) | 75% of points | **Direction is the #1 difficulty** |
| Baseline: nearest directed arc | 53.5% Top-1 | The naive baseline every team starts from |
| Nearest arc, ignoring direction | 89.1% | Geometry alone is ~89%, and the rest of the problem is mostly direction + parallel roads |
| GPS heading agrees with the true arc's direction | 59% at 0–3 m/s, 86% at 3–6, **98–100% above 6 m/s** | Heading must be weighed by speed. Slow points need the context/HMM (carried heading) |
| Hand rule "nearest + fixed heading penalty" | 84.9% (worse than undirected 89%) | Hand rules fail and a learned model is required. Good report evidence for "why ML" |

**Confirmed on train (80 trips, 208k points):** nearest-directed 55.5%, undirected 88.0%; heading agreement 67% (0–3 m/s), 91% (3–6), 98.7–99.9% (> 6 m/s). Same pattern.
**Reference-route anomaly (train):** 11.2% of train points lie > 50 m from EVERY arc of their own reference route, so the route doesn't
cover them (this is not a labelling bug). It's concentrated in 5 trips: **00000054** (87% of its 19.3k points; this trip alone is ~9% of train points),
**00000033** (59%), **00000002** (12%), 00000066 (9%), 00000043 (7%). Large time gaps also occur (up to 536 s in trip 81).
**Policy (LOCKED):** `ref_uncertain` points are **excluded from training and from the headline metrics**, but kept as features/context and
counted separately. Report metrics both ways ("clean points" = primary, "all points" = secondary). The API also reports both.
This goes into report §2 as the "reference route limitations" discussion the PDF asks for, with a Folium map of trip 00000054.

## 1. Our edge: what most teams (and their LLMs) will NOT do

Most teams will ask an LLM for "nearest road + RandomForest + textbook HMM". They'll get the same code, the same features and
the same numbers. Our pipeline differs in 8 places. Each one has an owner, and each one must show up in the report with
**our own measured before/after numbers** (only claim gains we actually measured).

| # | Novelty | Why it matters | Owner |
|---|---|---|---|
| N1 | **DP-optimal label alignment** (GPS sequence ↔ route sequence, monotone, minimises total distance) + a `ref_uncertain` flag (> 50 m) | Measured: 0.14% bad labels vs 3.8% for greedy and 23% for nearest-route-arc. Every other team's models train on noisier labels | P1 |
| N2 | **Temporal-context features** ("arc persistence"): over the ±5 neighbouring points, the share that also have this arc as a candidate, the mean distance of those neighbours to the arc, and heading agreement | A single GPS point is ambiguous and the trajectory is not. This gives the classifier sequence awareness before the HMM | P1 |
| N3 | **Direction-aware features** for the two-way-road problem (75% of points): `heading_diff`, `speed`, `heading_conf` (heading from displacement over ±2/±5 s), **`carried_heading_diff`** (heading from the last point moving > 6 m/s, carried through stops), `has_twin`, `twin_dist_gap` | Measured: heading is 59% reliable when slow vs 99%+ when moving. The model must learn when to trust it | P1 |
| N4 | **Learning-to-rank model** (LightGBM LambdaRank, query = GPS point) as an extra model family | Directly optimises "pick the right candidate per point" = Top-1, which is the metric that matters. Binary classifiers optimise the wrong thing | P3 |
| N5 | **Per-point softmax normalisation + calibration check** (reliability curve) of candidate scores | Probabilities that sum to 1 per point give a proper HMM emission and better log loss | P3 |
| N6 | **Learned (data-driven) HMM transitions**: a logistic regression on transition features (network distance − GPS distance, turn angle, same arc, connected, dt) fitted on train trips, compared against the classic exponential(β) transition | Turns a hand-tuned prior into a trained component. This is a strong viva talking point | P1 (after break) |
| N7 | **Gap bridging**: fill the gaps between consecutive predicted arcs with the network shortest path, so the output is a continuous drivable route, plus a **length-weighted route precision/recall** metric | Route continuity → ~100%, a realistic map, an extra metric | P4 |
| N8 | **Error taxonomy** on student_test: wrong direction / parallel road / junction / reference-uncertain, with counts and a Folium map per type | Shows understanding instead of one accuracy number | P2 |

Primary metric = **observation-level Top-1 accuracy** (what the API reports). Also report Macro-F1 (binary candidate level, defined in the report),
log loss, continuity, miss rate, and time.

---

## 2. Technical design (LOCKED)

**Labels (N1):** DP alignment. D[i,j] = point-to-segment distance of GPS i to route arc j. Cost recurrence
C[i,j] = D[i,j] + min_{j-S ≤ k ≤ j} C[i-1,k] with S = 8, start in the first S route arcs, end in the last S, then backtrack.
Save `label_arc`, `label_dist_m`, `ref_uncertain = label_dist_m > 50`. (A tested reference implementation exists: `probe3.py::dp_align`.)

**Candidates:** densify arcs into points every 10 m and put them in a scipy `cKDTree`. Query radius R+8, then filter by exact
point-to-segment distance ≤ **R = 50 m**, always keeping at least the nearest 3. Also record the projection fraction along the arc.

**Features per (point, candidate) row:** `dist_m, dist_rank, dist_minus_min, dist_ratio_min, heading_diff, speed_mps, heading_conf,
carried_heading_diff, has_twin, twin_dist_gap, dt_prev, dt_next, gap_flag, arc_len_m, proj_frac, n_cands, out_deg, in_deg,
ctx_share, ctx_mean_dist, ctx_heading_diff`. target = `label` (cand_arc == label_arc). Projection = local equirectangular (meters) around the trip centroid
(error < 0.1% at trip scale, so it's fine; justify this in the report).

**Models (7, so we comfortably clear the ≥5 rule):**
| ID | Model | Family | Runs on |
|---|---|---|---|
| M1 | Logistic Regression (StandardScaler) | Linear | P3 laptop |
| M2 | Gaussian NB | Probabilistic | P3 laptop |
| M3 | KNN distance-weighted (scaled, train on a 25% trip subsample) | Instance | P3 laptop |
| M4 | Random Forest (300 trees, n_jobs=-1) | Bagging | P3 laptop |
| M5 | HistGradientBoosting | Boosting | P2 laptop |
| M6 | MLP (64-32, early_stopping → loss curve) | Neural | P2 laptop |
| M7 | LightGBM LambdaRank (N4) | Learning-to-rank | P1 laptop |

Training rows: all positives + up to 4 negatives per point (negative subsampling). **Evaluation always uses all candidates.**
CV = 5-fold `GroupKFold(groups=trip_id)` on train. Dev comparison = the 10 student_test trips. Seed = 42 everywhere.
Every run appends one row to `results/model_results.csv` (columns in §4).

**HPO (after the break):** Optuna TPE with 50 trials on the best model, objective = GroupKFold mean Top-1, saving `results/optuna_trials.csv`.

**HMM/Viterbi (after the break):** states = candidates. Emission = log softmax-normalised prob. Transition = (a) exponential on |d_net − d_gps| with β,
(b) learned LR (N6). d_net = bounded Dijkstra on the arc graph (cutoff 3·d_gps+200 m), cached per source arc. Unreachable → floor −20.
No candidates → carry the previous state. Gap > 30 s → loosen β. Tune on train/student_test only.

**API/UI:** as in §4. Use `http://<laptop1 IPv4>:8000` from the UI (**never localhost**). Allow port 8000 in the Windows firewall. Both laptops on one phone hotspot.

---

## 3. Timeline (real clock)

| Time | P1: Dhanush + Claude | P2 | P3 | P4 |
|---|---|---|---|---|
| **10:50–11:00** | Read §0–2. Install packages (below) on ALL laptops | same | same | Create the **public GitHub repo** `pathsnap`, add 3 collaborators, push the skeleton (§4) |
| **11:00–11:40** | `build_dataset.py`: labels, candidates, features → parquet (N1–N3). Push code at 11:40 | EDA notebook on raw data (inventory, gaps, connectivity, 2 Folium maps) | `train_models.py` harness on **fake data** (`make_fake.py`, §4): CV, Top-1, F1, logloss, timing, CSV logging, plots | FastAPI `/trips` + `/predict` with a **dummy nearest-arc predictor** + Streamlit UI; 2-laptop call working |
| **11:40–11:50** | Everyone `git pull` and runs `python src/build_dataset.py` locally (~3–5 min) | | | |
| **11:50–12:25** | Train **M7 LambdaRank** | Train **M5, M6** | Train **M1–M4** | Gap bridging (N7) + route metric; screenshots |
| **12:25–12:30** | P3 merges all rows into `results/model_results.csv` + the comparison bar plots → **push. ALL MODELS DONE.** | | | |
| After break | HMM (classic + learned N6), classifier vs HMM table | Error taxonomy (N8) + Folium predicted-vs-reference maps, report §2 & §6 | Optuna 50 trials, tuned vs untuned, report §3–4 | Plug the real `predict_trip` into the API, test all 10 student_test trips, README, report §7 |
| Last hour | Freeze, export artifacts | Assemble the PDF | | Final rehearsal on 2 laptops |

**If P1 is late at 11:40:** P3/P2 keep running their models on the fake data so their code is proven. The real run starts the moment the parquet lands.
Cut order if short on time: KNN → 10% subsample, MLP → 1 hidden layer, RF → 150 trees. **Never miss 12:30 with fewer than 5 real models.**

Install (every laptop):
`pip install numpy pandas scikit-learn scipy networkx pyarrow joblib lightgbm optuna folium streamlit streamlit-folium fastapi uvicorn matplotlib requests`

---

## 4. Interfaces (contract: don't change them)

```
pathsnap/
  data/raw/train/<id>/...   data/raw/student_test/<id>/...   (unzip "Student folder" here; gitignore data/)
  data/processed/train_cands.parquet  student_test_cands.parquet      (gitignored, regenerate with build_dataset.py)
  data/eval_test/            (evaluator trips, on Laptop 1 only, gitignored)
  src/io_utils.py   load_trip(trip_dir) -> dict(track: DataFrame[lon,lat,t], nodes: ndarray, arcs: ndarray, route: ndarray|None)
  src/geo.py        to_xy(lon, lat, lon0, lat0) -> (x_m, y_m)
  src/candidates.py, labels.py, features.py
  src/build_dataset.py   -> writes the two parquets
  src/metrics.py    top1(df, score_col), macro_f1, logloss, continuity(pred_arcs, arcs)
  src/make_fake.py  -> fake parquet with the exact columns below (for P3/P2 before 11:40)
  src/train_models.py --models M1,M2,M3,M4   (each person runs their own subset)
  src/train_rank.py (M7)  src/tune.py  src/hmm.py  src/pipeline.py  src/make_maps.py
  notebooks/01_eda.ipynb  02_error_analysis.ipynb
  models/  results/  figures/  maps/  logs/
  api/main.py   ui/app.py   README.md  requirements.txt  PATHSNAP_PLAN.md
```

**Parquet columns:** `trip_id (str), point_idx (int), t (float), cand_arc (int), label (0/1), label_arc (int), ref_uncertain (bool)` + the feature columns in §2.

**results/model_results.csv columns:** `model_id, family, config, cv_macro_f1_mean, cv_macro_f1_std, cv_top1_mean, st_top1, st_macro_f1,
st_precision, st_recall, st_logloss, st_continuity, train_time_s, infer_time_ms_per_point, timestamp`

**pipeline:** `predict_trip(trip_dir) -> {"pred_arcs": list[int] per point, "probs": list[float], "track": ...}`. This is the ONLY thing the API calls.
**API:** `GET /trips` → ids in `EVAL_DIR`. `POST /predict {"trip_id"}` → GPS points, predicted polylines, top1, continuity, n_points, latency_ms
(scored server-side; the reference route is never returned). Logs every request to the console + `logs/api.log`.

---

## 5. Prompts for each person's LLM (paste the prompt AND this whole file)

Generic prefix for everyone:
> You are a senior ML engineer on our hackathon team. The attached PATHSNAP_PLAN.md is the binding spec: follow its interfaces, column
> names, file paths and LOCKED decisions exactly and do not redesign anything. Write complete, runnable Python 3.12 (no placeholders, no
> "..."), vectorised with numpy/pandas, seed 42, with tqdm-free plain prints of progress and timing. Never use evaluator-test data or
> student_test labels for training or tuning. Never hardcode results. After the code, list exactly the commands to run and which output
> files/plots get created. Keep the explanation short. I must be able to explain every line in a viva, so add brief comments on the WHY.

**P1 (if Claude is unavailable):**
> Implement src/io_utils.py, geo.py, labels.py, candidates.py, features.py and build_dataset.py per §2 and §4, including novelties N1, N2, N3.
> Data is in data/raw/{train,student_test}/<id>/<id>.{track,nodes,arcs,route}. Densify arcs every 10 m into a cKDTree. Compute exact
> point-to-segment distance and projection. Labels by the DP alignment in §2 (S=8), NOT greedy. Direction features (N3) and context
> features over ±5 points (N2). Must process all 90 trips in < 5 minutes. Print: rows, positives per point, Candidate Miss Rate,
> % ref_uncertain, and the baselines (nearest directed / nearest undirected Top-1). Expected on student_test: miss ≈ 0.14%, nearest-directed ≈ 53.5%.

**P2:**
> (a) notebooks/01_eda.ipynb on data/raw: inventory table (trips, points, nodes, arcs, route length per trip, file sizes), missing/invalid
> values, timestamp gaps histogram, speed distribution, trajectory extent, road-network connectivity (weakly connected components,
> isolated nodes via networkx), route-length vs track-length, and 2 Folium maps (GPS blue dots, network grey thin lines clipped to the
> track bbox + 300 m, reference route green, LayerControl, legend). Save all figures to figures/ and maps to maps/. State which partition
> each analysis uses. (b) At 11:50: run `python src/train_models.py --models M5,M6`. (c) After the break: notebooks/02_error_analysis.ipynb
> doing novelty N8 on student_test predictions.

**P3:**
> (a) src/make_fake.py producing a fake parquet with the exact §4 columns. (b) src/metrics.py and src/train_models.py: models M1–M6 per
> §2 table, negative subsampling (all positives + ≤4 negatives per point) for fitting only, 5-fold GroupKFold by trip_id, per-point
> softmax normalisation of scores (N5) before Top-1/logloss, metrics exactly as in the results CSV columns, append one row per model,
> save each fitted model to models/<id>.joblib, MLP loss curve png, confusion matrix png. CLI `--models`. (c) src/plot_comparison.py:
> bar charts of CV macro-F1, ST Top-1, ST macro-F1, logloss, inference time from results/model_results.csv. (d) After the break:
> src/tune.py with Optuna TPE, 50 trials, GroupKFold Top-1 objective, trials → results/optuna_trials.csv, history + param-importance plots,
> tuned vs untuned row in the CSV.

**P4:**
> (a) api/main.py (FastAPI) per §4 with a temporary dummy predictor (nearest arc per point) that you swap later for
> src.pipeline.predict_trip. Read EVAL_DIR from an env var. Log every request (trip_id, n_points, top1, latency) to the console and
> logs/api.log. Bind 0.0.0.0:8000. Add CORS. (b) ui/app.py in Streamlit: sidebar API URL box, trip dropdown from /trips, Predict button,
> folium map (GPS blue CircleMarkers, predicted route red PolyLine, LayerControl, legend, fit bounds) via streamlit-folium, metric cards.
> The UI must never read local data files. (c) Novelty N7: src/bridge.py with shortest-path gap filling between consecutive predicted
> arcs (networkx on the arc graph, length-weighted) + length-weighted route precision/recall vs the reference (server-side only).
> (d) README: install, build dataset, train, tune, run API, run UI, the 2-laptop network steps (ipconfig, firewall rule for port 8000).

---

## 6. Integrity rules (they check)
AI writes code. Humans run it and write the report prose **in their own words from real outputs**. Never put an unmeasured number in the
report. Disclose AI use. Before the viva, each person explains their part to the other three.

## 7. STATUS (tick + name + time)
- [ ] Repo public + skeleton (P4)
- [x] build_dataset.py → parquets (P1, done ~11:06, 25 s for 90 trips). Train: 208,417 points, 1,567,410 cand rows (7.5/pt),
      ref_uncertain 11.16%, clean miss 0.47%, nearest-directed 55.45%, undirected 87.95%. Student_test: 19,160 points,
      169,468 rows, ref_uncertain 0.14%, miss 0.03%, nearest-directed 53.44%, undirected 89.10%.
      Sanity check (untuned HistGB, 200 iters, not a submitted model): student_test clean Top-1 = 90.16%.
      → **P3/P2: skip make_fake.py, use the real parquet now** (README quick start).
- [ ] make_fake + train_models harness tested on fake data (P3)
- [ ] EDA notebook + 2 Folium maps (P2)
- [ ] API + UI dummy, 2-laptop call works (P4)
- [ ] M1–M4 real (P3)   - [ ] M5–M6 real (P2)
- [x] M7 LambdaRank (P1, `python src/train_rank.py`, ~2 min): CV Top-1 0.8997±0.0275, CV macro-F1 0.9440±0.0154;
      student_test Top-1 0.9047, macro-F1 0.9464, logloss 0.0493, continuity 0.9555, 0.03 ms/pt.
      M7 student_test errors: wrong direction (twin) 4.53% of points, adjacent/junction 1.78%, other road 3.22%; 28.7% of errors at < 3 m/s.
      → the HMM target is the direction errors (a wrong-direction arc doesn't connect to its neighbours).
- **ALL models MUST use `src/metrics.py`** (normalize_per_point → evaluate → predicted_sequence → continuity → append_result)
  and evaluate on `~ref_uncertain` rows, so that results/model_results.csv is comparable. Classifiers: mode="prob"; rankers: mode="softmax".
- [x] All 7 models in results/model_results.csv + comparison plots (P2/P3). Student-test Top-1: M7 0.9047, M5 0.9022, M6 0.8926,
      M4 0.8919, M1 0.8713, M2 0.8619, M3 0.8483. M7 is best on both CV and student-test, so it's the final model.
- [x] Optuna on M7 (P3, `src/tune.py`, results/optuna_summary.json): TPE seed 42, 25 trials (20 complete, 5 pruned), objective = mean
      5-fold GroupKFold Top-1 on the 80 train trips, 39.7 min. Best trial 13: CV Top-1 0.9011 vs untuned 0.8997 (+0.0014, far below the
      fold s.d. 0.0275); student-test 0.9046 vs 0.9047. → **The deployed model stays untuned M7** (gain not meaningful; the HMM is built on it).
      Best params favour smaller trees (num_leaves 20 vs 63), which is consistent with the train-vs-CV gap (in-sample Top-1 0.975 vs CV 0.900).
- [x] HMM classic + learned (N6) (P1). Run order: `python src/score_candidates.py` (out-of-fold M7 scores, ~2 min) → `python src/run_hmm.py` (~4 min).
      Tuned ONLY on train OOF scores; student_test used once. Results (Top-1 / continuity, clean points):
      | variant | train OOF | student_test |
      | classifier-only M7 | 0.9007 / 0.9616 | 0.9047 / 0.9555 |
      | HMM classic (best α=2, β=30, floor=−15, 18-point grid) | 0.8991 / 0.9785 | 0.9175 / 0.9776 |
      | **HMM learned transitions (α=1), DEPLOYED** | **0.9186 / 0.9874** | **0.9186 / 0.9854** |
      The classic HMM never beat classifier-only on train Top-1 (it only raised continuity), while the learned transition did on both partitions. This is the key N6 evidence.
      Learned LR coefficients (results/hmm_learned_transition_coef.csv): same_arc +2.05, connected +1.30, turn_angle −2.10, abs_diff −1.24, uturn −0.83.
      Error types (% of clean student_test points; direction / junction / other road): classifier 4.53/1.78/3.22 → learned HMM 3.93/1.52/2.69.
      Remaining direction errors: 590 of 752 are in trip 00000077 (runs up to 192 points, slow). On those points, even when moving > 4 m/s, GPS heading
      agrees with the reference direction only 45.6% (vs 88.3% on that trip's correct points), so the GPS is genuinely ambiguous there. Report this as a limitation; do NOT tune on it.
- [x] src/pipeline.py: `Predictor().predict_trip(trip_dir)` (no route read) + `score_trip(trip_dir, pred_arcs)` (server-side, DP labels).
      Tested on student_test 00000040: 2995 points in 1.6 s, top1 0.964, continuity 0.996. **P4: the API calls exactly these two.**
- [x] Gap bridging + route metric (N7, P4; repeated-arc loop bug fixed 15:45)   - [x] Error taxonomy + Folium maps (N8, P2)
- [x] API with the real pipeline: all 10 student_test trips OK through /trips + /predict (tested 15:32, re-tested after the bridge fix)
- [x] README incl. two-laptop deployment (P4)
- [ ] notebooks/01_eda.ipynb is EMPTY (0 bytes) on GitHub (P2 must re-save + push)
- [ ] Two-laptop rehearsal + screenshots (UI, Laptop-1 cmd log)   - [ ] Code freeze → evaluator crash-only check
- [ ] Report PDF (AI-use disclosure included)   - [ ] Viva run-through

## 8. Decisions log
- 10:50 plan v2: 12:30 training deadline, 7 models split across 3 laptops, novelties N1–N8 assigned.
- 11:10 plan v3 (evidence-based, §0.5): DP label alignment replaces greedy; topology expansion dropped (miss already 0.14%);
  N3 = direction-aware features (75% two-way twins, heading reliable only when > 6 m/s). Probe scripts kept in the scratchpad (copy them into repo `eda/probes/`).
