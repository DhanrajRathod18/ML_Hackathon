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
python src/make_maps.py          # Folium maps: GPS + road network + reference + predicted route, legend + layer control
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
| `make_maps.py` | report maps (`maps/route_*.html`, `maps/eda_*.html`); evaluator data refused |
| `pipeline.py` | `Predictor.predict_trip` (inference, no route) + `score_trip` (server-side scoring) |

## Run the evaluator service on two laptops

Use the same private Wi-Fi or phone hotspot for both laptops. The API belongs on
Laptop 1; the Streamlit UI belongs on Laptop 2. Run commands from the repository
root. Replace angle-bracket placeholders with values from your own setup; do not
copy evaluator data into this repository.

### Laptop 1: FastAPI backend

In PowerShell:

```powershell
python -m pip install -r requirements.txt

# Set this to the absolute path of the authorized evaluator-trip directory.
# Its children must be <trip-id>\<trip-id>.track, .nodes, .arcs and optional .route.
$env:EVAL_DIR = "<absolute-path-to-evaluator-trip-directory>"

# Use Laptop 2's IPv4 address and the Streamlit port. Add localhost origins only
# if you also use a browser on Laptop 1. Keep this as an explicit comma-separated list.
$env:CORS_ORIGINS = "http://<Laptop2-IPv4>:8501,http://localhost:8501,http://127.0.0.1:8501"

python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Keep that terminal open. API request logs are also written to `logs\api.log`;
to follow the file from another PowerShell window:

```powershell
Get-Content .\logs\api.log -Wait
```

### Laptop 2: Streamlit frontend

In PowerShell:

```powershell
python -m pip install -r requirements.txt
$env:PATHSNAP_API_URL = "http://<Laptop1-IPv4>:8000"
python -m streamlit run ui/app.py --server.port 8501
```

The sidebar URL can also be edited in the UI. For a second laptop, use
Laptop 1's IPv4 address, not `localhost` or `127.0.0.1`.

### Find the backend IPv4 address and allow the API port

On Laptop 1, run:

```powershell
ipconfig
```

Find the **IPv4 Address** for the adapter connected to the shared Wi-Fi/hotspot.
Do not use a VPN, virtual adapter, or loopback address. Enter that address in
Laptop 2's `PATHSNAP_API_URL` and include Laptop 2's browser origin in
`CORS_ORIGINS`. Restart the API after changing either environment variable.

If Windows Firewall blocks access, open PowerShell **as Administrator** on
Laptop 1 and add a private-network inbound rule:

```powershell
New-NetFirewallRule -DisplayName "PathSnap API 8000" `
  -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Private
```

Only allow the port on a trusted private network. The Streamlit UI does not
require an inbound firewall rule on Laptop 2 for this setup.

### Test and troubleshoot the connection

On Laptop 2, test the backend's trip-list endpoint using the configured URL:

```powershell
$apiUrl = $env:PATHSNAP_API_URL.TrimEnd("/")
Invoke-RestMethod -Uri "$apiUrl/trips"
```

The Streamlit **Test connection** button performs the same check. If it fails:

1. Confirm the API process is still running and Laptop 1 has a valid `EVAL_DIR`.
2. Confirm both laptops are on the same network and use Laptop 1's current
   adapter IPv4 address.
3. Check Laptop 1's firewall rule and that the API is bound to `0.0.0.0:8000`.
4. Confirm `CORS_ORIGINS` contains the exact UI origin (scheme, Laptop 2 address,
   and port); restart the API after changing it.
5. Inspect `logs\api.log` and the API terminal for server-side errors.

For local unit tests, install the test runner separately and use only the
synthetic fixtures in the test suite:

```powershell
python -m pip install pytest
pytest -p no:cacheprovider tests/ -v
```

### Model artifact compatibility

The deployed `Predictor` loads `M7_lgbm_rank.joblib` and, with the current
`hmm_config.json`, `hmm_learned_transition.joblib`; it does not load M1, M2 or M3.
The checked environment uses scikit-learn 1.9.0. M1–M3 were saved with 1.9.1
and may warn if loaded under 1.9.0. Do not load those artifacts in the API or
change the scikit-learn version without checking the artifact owner and testing
that artifact in its intended environment. The requirements pin scikit-learn
1.9.0 for the deployed learned-transition artifact and the currently installed
LightGBM 4.7.0. The M7 artifact does not record its LightGBM build version, so
run an approved inference smoke test before release. Other dependencies remain
unpinned and should be locked together for a fully reproducible release.
