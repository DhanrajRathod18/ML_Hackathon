# PathSnap: Map-Matching & Route Reconstruction

Noisy 1 Hz GPS → road segments (arcs) via a learned candidate scorer + HMM/Viterbi, served by FastAPI (Laptop 1) to a UI (Laptop 2).
Full design, measured data facts and team plan: [PATHSNAP_PLAN.md](PATHSNAP_PLAN.md).

## Quick start (every team member)
```bash
pip install -r requirements.txt
```
1. Unzip `Student folder.zip` and copy its `train/` and `student_test/` folders into `data/raw/`
   (so you have `data/raw/train/00000001/00000001.track` etc.). `data/` is git-ignored: never commit data.
2. Build the candidate tables (~30 s):
```bash
python src/build_dataset.py
```
This creates `data/processed/{train,student_test}_cands.parquet` (one row per GPS point × candidate arc),
`{train,student_test}_points.parquet` (one row per GPS point) and `results/dataset_summary.csv`.

## Training table contract
- Keys: `trip_id, point_idx, cand_arc`. Target: `label` (1 = candidate is the reference arc).
- Feature list: `from features import FEATURES` (24 columns, defined in `src/features.py`). Never add label columns to it.
- **Train only on `~ref_uncertain` rows** (the reference route does not cover those points; see the plan, §0.5).
- Top-1 accuracy = per `(trip_id, point_idx)`, argmax score among candidates == `label_arc`.
- CV: `GroupKFold(5)` with `groups=trip_id`. Student-test is for model comparison only. Seed 42.

## Reproduce the results (in this order)
```bash
python src/build_dataset.py      # candidate tables (~30 s)
python src/train_rank.py         # M7 LambdaRank: CV + student-test, appends results/model_results.csv (~2 min)
python src/score_candidates.py   # out-of-fold M7 scores for HMM tuning (~2 min)
python src/run_hmm.py            # HMM classic vs learned transitions, writes models/hmm_config.json (~4 min)
```

## Pipeline modules (`src/`)
| File | Purpose |
|---|---|
| `io_utils.py` | load raw trip files |
| `geo.py` | metric projection, point-to-segment distance, arc geometry, two-way twins |
| `labels.py` | DP alignment of GPS to the reference route (per-point labels) |
| `candidates.py` | KD-tree candidate arcs within 50 m |
| `features.py` | 24 distance / direction / context features |
| `build_dataset.py` | runs everything in parallel, writes parquet + summary |
| `metrics.py` | shared metrics: every model is scored with these |
| `train_rank.py` | M7 LightGBM LambdaRank |
| `score_candidates.py` | out-of-fold scores for leakage-free HMM tuning |
| `hmm.py` | lattice, network distances (bounded Dijkstra), classic + learned transitions, Viterbi |
| `run_hmm.py` | HMM tuning + classifier-vs-HMM comparison |
| `pipeline.py` | `Predictor.predict_trip` (inference, no route) + `score_trip` (server-side scoring) |
