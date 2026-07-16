# Neural Network — Transformer Jet Classifier

A transformer-based multi-class classifier used as an alternative discriminant to the XGBoost
BDT. Each input group (event-level, AK8 Higgs jets, AK4 away-jets, VBF jets) is embedded
separately into tokens; a learned `[CLS]` token is prepended; a `TransformerEncoder` processes
the sequence; the pooled representation (`CLS ⊕ masked-mean`) is classified by an MLP head into
per-class softmax scores. Training is config-driven (YAML), with equalize-class weights consistent
with `TrainBDT.py`.

## Pipeline (run order)

```
skimmer parquet
   │  prepare_data.py   (cuts + trigger + BDT-input vars, 80/10/10 split, class weights)
   ▼
{train,val,test,all}/<year>_<class>.parquet  +  class_weights_<year>.json
   │  train.py --config configs/<x>.yaml
   ▼
checkpoints/*.pt  +  roc_curves/*  +  scores/epoch_<N>/<split>/<class>.npz
   │  train.py ... --evaluation-only --eval-epoch best   (inference-only re-eval)
   ▼
scores/*.npz  →  ROC / downstream comparison
```

## Files

| file | what it does | how to run |
|---|---|---|
| **`prepare_data.py`** | Loads skimmer parquet, applies the same kinematic + trigger cuts as `TrainBDT.py`, computes BDT input features (`--bdt-config`), cleans NaN/outliers → `-99999`, splits 80/10/10, writes per-class parquet + `class_weights_<year>.json`. | **CLI** — `python prepare_data.py` (all args defaulted). |
| **`train.py`** | Main entry point. Builds dataloaders/model/loss/optimizer/scheduler from a YAML config; runs the train/val loop with early stopping + checkpointing; computes ROC; writes per-split softmax scores. Also a library (`build_model`, `run_epoch`, … reused for extra-sample eval). | **CLI** — `python train.py --config configs/<x>.yaml` |
| `models/classifier.py` | `JetClassifier`: per-group embedding → `[CLS]` + tokens → `TransformerEncoder` → `CLS ⊕ masked-mean` pool → MLP head → logits. | library (built in `train.py:build_model`) |
| `models/transformer.py` | `TransformerEncoder` / pre-norm `TransformerEncoderLayer` + `RMSNorm`, `SwiGLUFFN`, `LayerScale`, `DropPath`, register tokens. | library |
| `models/embedding.py` | `InputEmbedding` (linear-projects continuous + sums discrete `nn.Embedding` + LayerNorm, zeroes invalid tokens). | library |
| `utils/dataset.py` | `JetDataset` (numpy-backed, low worker memory), `jet_collate_fn` (batches raw tensors + applies feature transforms in one vectorized pass), `GroupConfig`. | library |
| `utils/roc.py` | `compute_roc_metrics` / `compute_roc_per_signal` (weighted AUC + sig-eff@bkg-eff), `save_roc_plot`, `ScoreAccumulator`. | library |
| `utils/feature_transform.py` | `FeatureTransform`: `normal` (z-score), `log_normal`, `min_max`, `digitize` (→ discrete embedding indices). | library |
| `utils/scheduler.py` | `get_scheduler` — StepLR / Cosine / SequentialLR (+ warmup), LR-resync on resume. | library |
| `utils/ckpt.py` | Checkpoint save/load + `resolve_output_dir` / `get_checkpoints_path` with `${name}`/`${epoch_num}` expansion. | library |
| `utils/logger.py` | Colored console + file logger (`LOGGER`). | library |
| `utils/__init__.py` | Re-exports the public utils API (`from utils import …`). | library |
| `configs/*.yaml` | Run configs (dataset, training, inputs, architecture). See below. | — |

## How to run

```bash
# 1) prepare data (defaults: --tag nanov15_20251202_v15_signal, --years 2024, --bdt-config v13_glopartv3)
python prepare_data.py --years 2022 2022EE 2023 2023BPix 2024

# 2) train
python train.py --config configs/transformer.yaml            # add --use-wandb for W&B logging

# multi-GPU (torchrun)
torchrun --nproc_per_node 4 train.py --config configs/transformer.yaml --log-file logs/run.log

# 3) inference-only (regenerate scores + ROC from the best checkpoint)
python train.py --config configs/transformer.yaml --evaluation-only --eval-epoch best
```

Key `train.py` flags: `--config/-c` (**required**), `--use-wandb`, `--evaluation-only`,
`--eval-epoch <int|best>`, `--log-file`, `--log-level`.

## Config YAML

A config defines four blocks:
- **`dataset`** — `dir`, `file_pattern` (e.g. `${split}/${year}_${class}.parquet`), `years`,
  `weight_key: finalWeight`, and `classes` with per-class `cls_weights`.
- **`training`** — `loss` (`focal`/`cross_entropy`), `use_weights` (per loss/acc/roc), `batch_size`,
  `num_epochs`, `patience`, `max_grad_norm`, `optimizer`, `scheduler`, `roc_groups`.
- **`inputs`** — the input groups (`event`, `bbFatJet1`, `bbFatJet2`, `AK4JetAway`, `VBFJet`), each
  with `idx`, `mask`, and `continuous_features` / `discrete_features` (+ per-feature `transform`).
- **`architecture`** — `encoder` (`dim`, `num_layers`, `num_heads`, `activation`, `norm`, …) + `head`.

`configs/config.yaml` is the committed template. The base experiment `transformer.yaml`
(4 classes: `hh4b`, `vbfhh4b-k2v0`, `qcd`, `ttbar`) uses `dim=128, num_layers=4, num_heads=8`,
SwiGLU/RMSNorm, a 2-layer MLP head, `AdamW`, `batch_size=50000`, cosine warmup→decay over 500 epochs.

## Conventions & notes

- **LR scaling:** `lr = 1e-4` at `batch_size = 1000`, scaled **linearly** with batch size (e.g.
  `batch_size=50000` → `lr=5e-3`). Encoded per-config in `optimizer.kwargs.lr`, not auto-computed.
- **Class weights:** signal `cls_weights` are large (≈130k `hh4b`, ≈23k `vbfhh4b-k2v0`) so total
  signal weight balances total background; `qcd`/`ttbar` = 1.0 (matches `TrainBDT.py`).
- **Outputs** (under the config's `output_dir`): `checkpoints/checkpoint_epoch_{N,best}.pt`,
  `roc_curves/roc_<split>_epoch_<N>.{png,pdf}` + `eval_metrics_*.json`,
  `scores/epoch_<N>/<split>/<class>.npz` (softmax scores + weights).
- **Cadence:** during training, scores/ROC are collected only on new-best epochs; a full
  train/val/test evaluation runs at end-of-training and in `--evaluation-only` mode.
- Multi-GPU uses HuggingFace `accelerate` (`dispatch_batches=True`) + bf16 when enabled.
