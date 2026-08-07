# ABCDnn — per-event transfer-factor QCD background estimation

A **6-way `(region × sample)` MLP classifier** that produces a **per-event data-driven QCD
background** estimate for the boosted HH→4b analysis, plus the downstream **XGBoost analysis BDT**
trained on it. The classifier learns to separate `{data, ttbar} × {B, C, D}` from a *strict*
feature set that deliberately excludes the two ABCD axes (TXbb, mass), so its softmax probabilities
can be turned into a per-event transfer factor that predicts the QCD shape in the blinded signal
region (A).

## Regions & transfer factor

Regions are defined on `(TXbb, mass)` of one fatjet:

| | mass high | mass low |
|---|---|---|
| **TXbb high** | **A** — signal region (blinded, *never trained*) | **C** |
| **TXbb low** | **B** | **D** |

The classifier trains only on `{B, C, D}`. `apply.py` then computes, per Region-B data event:

```
TF(x)     = (P(data,C|x) − P(ttbar,C|x)) / (P(data,D|x) − P(ttbar,D|x))
purity(x) = (P(data,B|x) − P(ttbar,B|x)) / P(data,B|x)
w_QCD_A(x) = TF(x) · purity(x)          # the QCD-in-A per-event weight
```

**Current (v2) setup:** regions are keyed on **H2Xbb** (the *subleading* fatjet TXbb,
`--txbb-jet-index 1`, bins `0.0 0.65 1.0`). Because the split no longer uses H1's TXbb, **`H1Xbb`
is kept** as a BDT input (it no longer trivially separates the regions). Default BDT config:
`v13_glopartv3` (the retired `_noxbb` variant was the old jet0-TXbb "v1" setup).

## Pipeline (run order)

```
skimmer parquet
  0│ prep_bdt_inference_pickles.py   per-era HLT-OR + preselection → combined year-tag pickles
   ▼   <cache>/<year>/<sample>.pkl
  1│ features.py (lib)               strict 13-feature extraction
  2│ dataset.py (lib, via train.py)  region masks, 6-way labels, 70/15/15 split, standardize (μ,σ)
   ▼   <run>/processed_data.npz + feature_stats.json
  3│ model.py (lib)                  ABCDClassifier MLP (6 logits)
  4│ train.py                        train the ABCDnn MLP  → best_model.pt
  5│ apply.py                        softmax → TF·purity   → apply/per_event_weights.parquet
  6│ prepare_bdt_data.py             Region-A BDT datasets → <version>/{train,val,test}.parquet
  7│ train_bdt.py                    XGBoost 4-class BDT   → trained_bdt.model + ROC metrics
```

**Two different models, in sequence:**
- **`train.py`** trains the **ABCDnn MLP** (PyTorch, 6-way region×sample, strict features) — the
  *background-estimation engine*; `apply.py` turns it into the per-event QCD weight.
- **`train_bdt.py`** trains the **analysis discriminant** (XGBoost, 4-class `hh4b`,
  `vbfhh4b-k2v0`, `ttbar`, `qcd`) on Region-A data, delegating the fit to `HH4b.boosted.TrainBDT`.
  Its QCD class comes *from* the ABCDnn prediction.

## Files (committed core)

| file | what it does | run |
|---|---|---|
| **`prep_bdt_inference_pickles.py`** | Combine data-taking eras into one year-tag pickle, applying each era's own HLT-OR + preselection. Produces the caches every other stage reads. | **CLI** — `python -m HH4b.boosted.abcd_classifier.prep_bdt_inference_pickles` |
| **`train.py`** | Runs `dataset.prepare_dataset` (build NPZ + standardize) then trains the 6-way `ABCDClassifier` (weighted CE, Adam, `ReduceLROnPlateau`, early stop). `--prepare-only` stops after dataset prep (torch-less). | **CLI** — `python -m HH4b.boosted.abcd_classifier.train` |
| **`apply.py`** | Loads `best_model.pt`, softmaxes over standardized features, computes `TF·purity` per Region-B data event + Region-A QCD shape histograms. | **CLI** — `python -m HH4b.boosted.abcd_classifier.apply` |
| **`prepare_bdt_data.py`** | Build Region-A `{train,val,test}.parquet` for the analysis BDT; QCD source set by `--version`. | **CLI** — `python -m HH4b.boosted.abcd_classifier.prepare_bdt_data` |
| **`train_bdt.py`** | Train the XGBoost 4-class analysis BDT on Region-A datasets (dir mode or `--version-config` composition), with cross-eval. | **CLI** — `python -m HH4b.boosted.abcd_classifier.train_bdt` |
| `features.py` | `STRICT_FEATURES` (13 features, no TXbb/mass), `extract_features`, `*_era` (one-hot era) variants. Hard-codes `v13_glopartv3` as the BDT config. | library |
| `model.py` | `ABCDClassifier` — MLP `Linear→BN→ReLU→Dropout ×N` → 6 logits (default `hidden=512, layers=5, ≈1.1M params`). | library |
| `dataset.py` | Region masks, 6-way label `2*region+sample`, stratified 70/15/15 split, μ/σ standardize (train split only), NPZ/JSON persistence. | library |
| `_argparse_utils.py` | Local `add_bool_arg` (`--flag`/`--no-flag`), kept local to avoid the xgboost import chain. | library |
| `__init__.py` | Package docstring. | library |

## How to run (concrete)

```bash
# 0) build the per-year inference-pickle caches (combine eras)
python -m HH4b.boosted.abcd_classifier.prep_bdt_inference_pickles \
    --eras 2022 2022EE --year 2022 \
    --out-cache /ceph/.../bdt_inference/nanov15_v15_glopartv3_rawmass \
    --skimmer-dir /ceph/.../skimmer/nanov15_20251202_v15_signal

# 2+4) dataset prep + train the ABCDnn MLP (v2: regions on H2Xbb → --txbb-jet-index 1)
python -m HH4b.boosted.abcd_classifier.train \
    --bdt-inference-dir /ceph/.../bdt_inference \
    --model-name nanov15_v15_glopartv3_rawmass \
    --year 2022 --feature-set strict \
    --txbb-jet-index 1 --txbb-bins 0.0 0.65 1.0 --mass-bins 50 100 150 \
    --run-name 26Jun02_v2_jet1txbb_2022 --out-dir /ceph/.../abcd_classifier/ \
    --epochs 100 --patience 10 --lr 1e-3 --batch 1024
#   (add --prepare-only to build the NPZ on a CPU/torch-less node, then train on GPU)

# 5) apply → per-event QCD weights
python -m HH4b.boosted.abcd_classifier.apply \
    --run-dir /ceph/.../abcd_classifier/26Jun02_v2_jet1txbb_2022 \
    --model-name nanov15_v15_glopartv3_rawmass --year 2022 \
    --mode purity --negative-weight-policy clip --plot-vars bdt_score

# 6) Region-A BDT datasets (abcd = data-driven QCD, mc = MC-QCD baseline)
python -m HH4b.boosted.abcd_classifier.prepare_bdt_data \
    --model-name nanov15_v15_glopartv3_rawmass \
    --run-dir /ceph/.../abcd_classifier/26Jun02_v2_jet1txbb_2022 --year 2022 \
    --version abcd mc --txbb-jet-index 1 --txbb-bins 0.0 0.65 1.0 --mass-bins 50 100 150 \
    --bdt-config v13_glopartv3

# 7) train the analysis XGBoost BDT (dir mode, with cross-eval)
python -m HH4b.boosted.abcd_classifier.train_bdt \
    --dataset-dir <run>/bdt_datasets/abcd --model-name ABCD_<date>_abcd_xbb \
    --eval-dataset-dirs <run>/bdt_datasets/mc <run>/bdt_datasets/full \
    --txbb glopart-v3 --max-depth 3 --learning-rate 0.1 --n-estimators 5000
```

## `--version` semantics (`prepare_bdt_data.py`)

All versions restrict to Region A and take signal (`hh4b`, `vbfhh4b-k2v0`) + `ttbar` from Region-A
MC; they differ only in the **QCD source**:

| version | Region-A QCD | note |
|---|---|---|
| **`abcd`** | Region-**B data**, weighted by `w_QCD_A = TF·purity` | data-driven (the ABCDnn estimate) |
| **`mc`** | Region-**A MC** QCD (`finalWeight`) | MC baseline / reference |
| `full` | all-MC, no region cut (canonical `TrainBDT` baseline) | |
| `fullqcd` | full-region MC QCD + Region-A ttbar | pair with `--balance-bg` |
| `hybrid` | ABCDnn B-data→A + MC QCD for B/C/D | |
| `base_regionA`, `base_full` | "stream library" bases composed at train time via `train_bdt.py --version-config versions/*.yaml` | |

## Outputs

- `prep_bdt_inference_pickles` → `<cache>/<year>/<sample>.pkl` (kinematics + `finalWeight` + `bdt_score[_vbf]` + `era`).
- `train` → `<run>/{processed_data.npz, feature_stats.json, best_model.pt, train_log.csv, metrics.json}` + loss/ROC plots.
- `apply` → `<run>/apply/per_event_weights.parquet` (`TF`, `purity`, `w_QCD_A`, `kept`, …) + `apply/h_QCD_A_<var>.pkl`.
- `prepare_bdt_data` → `<out>/<version>/{train,val,test}.parquet` (BDT features + `label`, `sample`, `weight`, `region`, `year`).
- `train_bdt` → `<out>/<model-name>/{trained_bdt.model, metrics.json, test_predictions[_<label>].parquet, train_test_plots/}`.

## Orchestration

Multi-year runs are chained by convenience wrapper scripts in this dir (`run_multiyear_prep.sh`,
`run_multiyear_pipeline.sh`, `run_compose_yr.sh`, `train_bdt.sh`, `run_grid_compare.sh`, …) plus
`nrp/` for the GPU (Kubernetes) training of the MLP, and `versions/*.yaml` composition configs.
The diagnostic/comparison plotting scripts (`plot*.py`, `eval_3year_compare.py`) are also here.
These are workflow helpers around the committed CLIs above.

See `notes/ABCDnn.md` for the full method, conventions, and the original Task-0–7 plan.
