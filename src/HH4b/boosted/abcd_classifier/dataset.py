"""
Dataset assembly: load cached pickles, apply region masks, build features
+ labels + weights, train/val/test split, standardize, persist NPZ.

Pipeline (called from train.py via :func:`prepare_dataset`):

    cached pickles ──▶ load_events() ──▶ {'data': df, 'ttbar': df}
                                                │
                                                ▼
                                       build_dataset()
        ┌──────────────────────────────────────────┐
        │  for sample in (data, ttbar):            │
        │     for region in (B, C, D):             │
        │        select rows by region mask        │
        │        extract features (Task 1)         │
        │        replace pad sentinels with 0.0    │
        │        assign 6-way label, weight        │
        │  concatenate, stratified 70/15/15 split  │
        └──────────────────────────────────────────┘
                                │
                                ▼
                    standardize_and_save()
        compute (μ, σ) on train split only
        write processed_data.npz + feature_stats.json

See notes/ABCDnn.md Task 2.
"""  # noqa: RUF002

from __future__ import annotations

import json
import logging
import logging.config
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from HH4b.hh_vars import txbb_strings
from HH4b.log_utils import log_config

from . import features

log_config["root"]["level"] = "INFO"
logging.config.dictConfig(log_config)
logger = logging.getLogger("ABCDnn.dataset")


# §1.1 region masks ---------------------------------------------------------

REGIONS = ("A", "B", "C", "D")


def build_region_masks(
    events: pd.DataFrame,
    txbb_col: str,
    mass_col: str,
    txbb_bins: tuple[float, float, float],
    mass_bins: tuple[float, float, float],
    txbb_jet_index: int = 0,
    mass_jet_index: int = 0,
) -> dict[str, np.ndarray]:
    """A/B/C/D region masks on (TXbb[txbb_jet_index], mass[mass_jet_index]).

    ``txbb_jet_index`` and ``mass_jet_index`` choose which FatJet column
    to use for each axis (0 = leading, 1 = subleading).  The original
    v1 setup uses (0, 0); v2 uses (1, 0).

    See notes/abcd_runs.md for the per-tag conventions.  Region
    encoding (used by both the ABCDnn classifier and BDT prep)::

        A: TXbb high × mass high  (SR, blinded — not in classifier training)
        B: TXbb low  × mass high
        C: TXbb high × mass low
        D: TXbb low  × mass low
    """  # noqa: RUF002
    txbb_arr = events[txbb_col].to_numpy()[:, txbb_jet_index]
    mass_arr = events[mass_col].to_numpy()[:, mass_jet_index]

    txbb_lo, txbb_split, txbb_hi = txbb_bins
    mass_lo, mass_split, mass_hi = mass_bins

    txbb_high = (txbb_arr >= txbb_split) & (txbb_arr < txbb_hi)
    txbb_low = (txbb_arr >= txbb_lo) & (txbb_arr < txbb_split)
    mass_high = (mass_arr >= mass_split) & (mass_arr < mass_hi)
    mass_low = (mass_arr >= mass_lo) & (mass_arr < mass_split)

    return {
        "A": txbb_high & mass_high,
        "B": txbb_low & mass_high,
        "C": txbb_high & mass_low,
        "D": txbb_low & mass_low,
    }


# §1.2 conventions ----------------------------------------------------------

SAMPLES = ("data", "ttbar")
TRAINING_REGIONS = ("B", "C", "D")  # A is the SR, never used in training

SAMPLE_TO_ID = {s: i for i, s in enumerate(SAMPLES)}
REGION_TO_ID = {r: i for i, r in enumerate(TRAINING_REGIONS)}

# Flat 6-way label encoding: label = 2 * region_idx + sample_idx
#   0 = (B, data)   1 = (B, ttbar)
#   2 = (C, data)   3 = (C, ttbar)
#   4 = (D, data)   5 = (D, ttbar)
LABEL_NAMES = [f"{r}_{s}" for r in TRAINING_REGIONS for s in SAMPLES]


def label_for(region: str, sample: str) -> int:
    return 2 * REGION_TO_ID[region] + SAMPLE_TO_ID[sample]


# Loading -------------------------------------------------------------------


def cache_path(bdt_inference_dir: str | Path, model_name: str, year: str, sample: str) -> Path:
    return Path(bdt_inference_dir) / model_name / year / f"{sample}.pkl"


def load_events(
    bdt_inference_dir: str | Path,
    model_name: str,
    years: list[str],
    samples: tuple[str, ...] = SAMPLES,
) -> dict[str, pd.DataFrame]:
    """Load cached post-inference pickles, concatenate across years."""
    out: dict[str, pd.DataFrame] = {}
    for sample in samples:
        per_year = []
        for year in years:
            p = cache_path(bdt_inference_dir, model_name, year, sample)
            if not p.exists():
                raise FileNotFoundError(f"missing cache: {p}")
            logger.info(f"loading {p}")
            with p.open("rb") as f:
                per_year.append(pickle.load(f))
        out[sample] = (
            per_year[0] if len(per_year) == 1 else pd.concat(per_year, axis=0, ignore_index=True)
        )
        logger.info(f"  {sample}: {len(out[sample])} events")
    return out


# Per-region build ----------------------------------------------------------


def _per_sample_arrays(
    df: pd.DataFrame,
    sample: str,
    txbb_bins: tuple[float, float, float],
    mass_bins: tuple[float, float, float],
    txbb_str: str,
    mass_str: str,
    feature_set: str,
    txbb_jet_index: int = 0,
    mass_jet_index: int = 0,
) -> dict[str, np.ndarray] | None:
    """For one sample DataFrame, build per-event arrays for events landing in
    one of the three training regions {B, C, D}.  A and 'none' are dropped.

    Returns a dict of (N_kept,)-shape arrays, or None if no events landed
    in any of the training regions.
    """
    masks = build_region_masks(
        df,
        txbb_str,
        mass_str,
        txbb_bins,
        mass_bins,
        txbb_jet_index=txbb_jet_index,
        mass_jet_index=mass_jet_index,
    )

    Xs, ys, ws, region_ids, orig_indices = [], [], [], [], []

    for region in TRAINING_REGIONS:
        mask = masks[region]
        n = int(mask.sum())
        logger.info(f"  {sample}: region {region}: {n} events")
        if n == 0:
            continue
        sub = df[mask]
        X_raw = features.extract_features(sub, feature_set)
        X = features.replace_pad_vals(X_raw, replacement=0.0)

        if sample == "data":
            w = np.ones(n, dtype=np.float32)
        else:
            w = sub["finalWeight"].to_numpy().astype(np.float32).reshape(-1)

        y = np.full(n, label_for(region, sample), dtype=np.int64)
        rid = np.full(n, REGION_TO_ID[region], dtype=np.int64)
        oi = np.where(mask)[0].astype(np.int64)

        Xs.append(X)
        ys.append(y)
        ws.append(w)
        region_ids.append(rid)
        orig_indices.append(oi)

    if not Xs:
        return None

    return {
        "X": np.concatenate(Xs, axis=0),
        "y": np.concatenate(ys, axis=0),
        "w": np.concatenate(ws, axis=0),
        "region_id": np.concatenate(region_ids, axis=0),
        "orig_idx": np.concatenate(orig_indices, axis=0),
    }


# Stratified split ----------------------------------------------------------


def _stratified_three_way_split(
    y: np.ndarray, train_frac: float, val_frac: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified 3-way split that preserves per-class fractions."""
    test_frac = 1.0 - train_frac - val_frac

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    idx_trainval, idx_test = next(sss1.split(np.zeros(len(y)), y))

    val_within = val_frac / (train_frac + val_frac)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_within, random_state=seed + 1)
    y_tv = y[idx_trainval]
    rel_train, rel_val = next(sss2.split(np.zeros(len(y_tv)), y_tv))
    return idx_trainval[rel_train], idx_trainval[rel_val], idx_test


# Top-level builder ---------------------------------------------------------


def build_dataset(
    events: dict[str, pd.DataFrame],
    txbb_bins: tuple[float, float, float],
    mass_bins: tuple[float, float, float],
    txbb_str: str,
    mass_str: str,
    feature_set: str,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
    txbb_jet_index: int = 0,
    mass_jet_index: int = 0,
) -> dict[str, np.ndarray]:
    """End-to-end build of the per-event arrays + split indices for the
    training set ``{data, ttbar} × {B, C, D}``.

    Returns a dict with keys:
        X          (N, d) float32  features after replace_pad_vals
        y          (N,)   int64    6-way class label per §1.2
        w          (N,)   float32  per-event training weight per §1.5
        sample_id  (N,)   int64    0=data, 1=ttbar
        region_id  (N,)   int64    0=B, 1=C, 2=D
        event_id   (N,)   int64    sequential index within the assembled dataset
        idx_train  (N_t,) int64    indices into the above arrays
        idx_val    (N_v,) int64
        idx_test   (N_e,) int64
    """  # noqa: RUF002
    arrays_by_sample: list[dict[str, np.ndarray]] = []
    sample_ids: list[np.ndarray] = []

    for sample, df in events.items():
        if sample not in SAMPLES:
            logger.warning(f"unexpected sample {sample!r}, skipping")
            continue
        per = _per_sample_arrays(
            df,
            sample,
            txbb_bins,
            mass_bins,
            txbb_str,
            mass_str,
            feature_set,
            txbb_jet_index=txbb_jet_index,
            mass_jet_index=mass_jet_index,
        )
        if per is None:
            continue
        arrays_by_sample.append(per)
        sample_ids.append(np.full(len(per["X"]), SAMPLE_TO_ID[sample], dtype=np.int64))

    if not arrays_by_sample:
        raise RuntimeError("no events landed in any of B/C/D — check region cuts and inputs")

    X = np.concatenate([a["X"] for a in arrays_by_sample], axis=0)
    y = np.concatenate([a["y"] for a in arrays_by_sample], axis=0)
    w = np.concatenate([a["w"] for a in arrays_by_sample], axis=0)
    region_id = np.concatenate([a["region_id"] for a in arrays_by_sample], axis=0)
    orig_idx = np.concatenate([a["orig_idx"] for a in arrays_by_sample], axis=0)
    sample_id = np.concatenate(sample_ids, axis=0)
    event_id = np.arange(len(X), dtype=np.int64)

    # Sanity: every label in [0, 6) and every label populated
    assert (  # noqa: PT018
        y.min() >= 0 and y.max() < 6
    ), f"unexpected label range: [{y.min()}, {y.max()}]"
    populated = set(np.unique(y).tolist())
    if populated != set(range(6)):
        missing = set(range(6)) - populated
        logger.warning(f"missing labels in dataset: {sorted(missing)}")

    idx_train, idx_val, idx_test = _stratified_three_way_split(y, train_frac, val_frac, seed)

    # Sanity: every class is represented in every split
    for split_name, split_idx in (
        ("train", idx_train),
        ("val", idx_val),
        ("test", idx_test),
    ):
        present = set(np.unique(y[split_idx]).tolist())
        missing = set(range(6)) - present
        if missing:
            raise RuntimeError(
                f"split {split_name!r} missing classes {sorted(missing)} — "
                "needs more events or smaller train/val/test fractions"
            )

    logger.info(
        f"dataset built: N={len(X)}  train/val/test = "
        f"{len(idx_train)}/{len(idx_val)}/{len(idx_test)}"
    )
    for c in range(6):
        n = int((y == c).sum())
        logger.info(f"  class {c} ({LABEL_NAMES[c]:>10s}): {n} events  Σw = {w[y == c].sum():.2f}")

    return {
        "X": X.astype(np.float32),
        "y": y.astype(np.int64),
        "w": w.astype(np.float32),
        "sample_id": sample_id.astype(np.int64),
        "region_id": region_id.astype(np.int64),
        "orig_idx": orig_idx.astype(np.int64),
        "event_id": event_id,
        "idx_train": idx_train.astype(np.int64),
        "idx_val": idx_val.astype(np.int64),
        "idx_test": idx_test.astype(np.int64),
    }


# Standardize + persist ------------------------------------------------------


def standardize_and_save(
    dataset: dict[str, np.ndarray],
    out_dir: str | Path,
    feature_set: str,
    txbb_bins: tuple[float, float, float],
    mass_bins: tuple[float, float, float],
    txbb_str: str,
    mass_str: str,
    txbb_jet_index: int = 0,
    mass_jet_index: int = 0,
) -> Path:
    """Compute (μ, σ) on the train split only, persist NPZ + JSON."""  # noqa: RUF002
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = dataset["X"]
    idx_train = dataset["idx_train"]

    mu = X[idx_train].mean(axis=0).astype(np.float32)
    sigma = X[idx_train].std(axis=0).astype(np.float32)
    sigma = np.where(sigma == 0, 1.0, sigma).astype(np.float32)  # avoid /0

    feature_names = features.feature_columns(feature_set)

    stats = {
        "feature_set": feature_set,
        "feature_names": feature_names,
        "mu": mu.tolist(),
        "sigma": sigma.tolist(),
        "txbb_bins": list(txbb_bins),
        "mass_bins": list(mass_bins),
        "txbb_str": txbb_str,
        "mass_str": mass_str,
        "txbb_jet_index": int(txbb_jet_index),
        "mass_jet_index": int(mass_jet_index),
        "label_names": LABEL_NAMES,
    }
    json_path = out_dir / "feature_stats.json"
    json_path.write_text(json.dumps(stats, indent=2))
    logger.info(f"saved {json_path}")

    npz_path = out_dir / "processed_data.npz"
    np.savez_compressed(
        npz_path,
        X=dataset["X"],
        y=dataset["y"],
        w=dataset["w"],
        sample_id=dataset["sample_id"],
        region_id=dataset["region_id"],
        orig_idx=dataset["orig_idx"],
        event_id=dataset["event_id"],
        idx_train=dataset["idx_train"],
        idx_val=dataset["idx_val"],
        idx_test=dataset["idx_test"],
    )
    logger.info(f"saved {npz_path}")
    return npz_path


# Orchestrator (called from train.py) ---------------------------------------


def prepare_dataset(args) -> Path:
    """End-to-end: load, build, standardize, save.  Returns the run dir.

    Skips the build if ``processed_data.npz`` already exists and
    ``args.force_rebuild`` is False.
    """
    # Year-tags are literal cache subdir names; multi-era combining is done
    # upstream in prep_bdt_inference_pickles.py.  No era expansion here.
    years = list(args.year)

    txbb_str = txbb_strings[args.txbb]
    mass_str = args.mass

    run_dir = Path(args.out_dir) / args.run_name
    npz_path = run_dir / "processed_data.npz"
    if npz_path.exists() and not args.force_rebuild:
        logger.info(f"NPZ already exists: {npz_path} (use --force-rebuild to redo)")
        return run_dir

    events = load_events(args.bdt_inference_dir, args.model_name, years)

    txbb_jet_index = getattr(args, "txbb_jet_index", 0)
    mass_jet_index = getattr(args, "mass_jet_index", 0)

    dataset = build_dataset(
        events,
        txbb_bins=tuple(args.txbb_bins),
        mass_bins=tuple(args.mass_bins),
        txbb_str=txbb_str,
        mass_str=mass_str,
        feature_set=args.feature_set,
        seed=args.seed,
        txbb_jet_index=txbb_jet_index,
        mass_jet_index=mass_jet_index,
    )

    standardize_and_save(
        dataset,
        out_dir=run_dir,
        feature_set=args.feature_set,
        txbb_bins=tuple(args.txbb_bins),
        mass_bins=tuple(args.mass_bins),
        txbb_str=txbb_str,
        mass_str=mass_str,
        txbb_jet_index=txbb_jet_index,
        mass_jet_index=mass_jet_index,
    )
    return run_dir
