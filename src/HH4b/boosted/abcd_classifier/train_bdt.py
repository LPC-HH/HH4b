"""
Train an XGBoost BDT on the Region-A-only datasets produced by
``prepare_bdt_data.py``.

Reads ``<dataset-dir>/{train,val,test}.parquet`` (already preselected,
region-A-filtered, BDT-features computed) and delegates the actual fit,
loss plotting, and train/test ROC plotting to
``HH4b.boosted.TrainBDT`` so the trained model is directly comparable to
the canonical training pipeline.  Only the data-loading front-end
differs: we read pre-split, feature-extracted parquets instead of raw
skimmer pickles, so :func:`TrainBDT.preprocess_data` (which calls
``bdt_dataframe`` on raw events) is replaced by a small multi-index
constructor here.

Outputs (under ``<out-dir>/<model-name>/``)::

    trained_bdt.model            # written by TrainBDT.train_model
    evals_result.txt             # written by TrainBDT.train_model
    losses.pdf                   # written by TrainBDT.plot_losses
    train_test_plots/            # written by TrainBDT.plot_train_test
    test_predictions.parquet     # held-out test predictions (this script)
    metrics.json                 # test-set per-class AUCs + config (this script)
"""

from __future__ import annotations

import argparse
import json
import logging
import logging.config
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

from HH4b.boosted import TrainBDT
from HH4b.hh_vars import txbb_strings
from HH4b.log_utils import log_config
from HH4b.run_utils import add_bool_arg

log_config["root"]["level"] = "INFO"
logging.config.dictConfig(log_config)
logger = logging.getLogger("ABCDnn.train_bdt")


# Match prepare_bdt_data.DEFAULT_LABEL_MAP.  Following the canonical
# TrainBDT setup, the VBF signal class is the BSM κ_{2V}=0 point.
LABEL_NAMES = {0: "hh4b", 1: "vbfhh4b-k2v0", 2: "ttbar", 3: "qcd"}
NAME_TO_LABEL = {v: k for k, v in LABEL_NAMES.items()}
# Non-feature columns.  "year" is a bookkeeping string column added by
# combine_bdt_datasets.py; the numeric one-hot "year_<tag>" columns it
# optionally adds are NOT listed here, so they ARE picked up as features.
META_COLS = {"label", "sample", "weight", "region", "year"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost BDT on Region-A datasets.")
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="dir containing {train,val,test}.parquet (single-year). "
        "Mutually exclusive with --dataset-dirs.",
    )
    parser.add_argument(
        "--dataset-dirs",
        nargs="+",
        default=None,
        help="Multiple per-year version dirs (each <run>/bdt_datasets/<version>) "
        "to concatenate in memory.  The year is read from each run-dir name; "
        "no combined copy is written to disk.",
    )
    # --- composition mode: build versions from base datasets via a YAML ---
    parser.add_argument(
        "--version-config",
        default=None,
        help="YAML composing the trained version from base datasets (each "
        "class -> {base, sample}).  Replaces --dataset-dir(s); requires "
        "--base-dirs.  No per-version parquet on disk.",
    )
    parser.add_argument(
        "--eval-version-configs",
        nargs="*",
        default=[],
        help="Version YAMLs to cross-evaluate on (composed from the same "
        "--base-dirs).  Predictions saved as test_predictions_<name>.parquet.",
    )
    parser.add_argument(
        "--base-dirs",
        nargs="+",
        default=None,
        help="Per-year base-dataset parent dirs (each holding base_regionA/ and "
        "base_full/).  One per year for a year-aware combined training.",
    )
    add_bool_arg(
        # defined later in the file; forward-declared default here.
        parser,
        "add-year-feature",
        "Add one-hot year_<tag> feature columns at load time (year-aware BDT). "
        "Only meaningful with multiple --dataset-dirs / --base-dirs.",
        default=False,
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).parent.parent / "bdt_trainings_run3"),
    )
    parser.add_argument("--sig-keys", nargs="+", default=["hh4b", "vbfhh4b-k2v0"])
    parser.add_argument("--bg-keys", nargs="+", default=["ttbar", "qcd"])
    parser.add_argument(
        "--eval-dataset-dirs",
        nargs="*",
        default=[],
        help="Extra dataset dirs (each containing test.parquet) to evaluate the "
        "trained BDT on, in addition to --dataset-dir.  Predictions saved as "
        "test_predictions_<dir-basename>.parquet for cross-evaluation.",
    )
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--n-estimators", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--txbb",
        choices=["pnet-v12", "pnet-legacy", "glopart-v2", "glopart-v3"],
        default="glopart-v3",
        help="TXbb version (only used for the legend title in train/test plots).",
    )
    add_bool_arg(parser, "equalize-weights", "Equalise total sig and bg weights", default=True)
    add_bool_arg(
        parser,
        "balance-bg",
        "Also balance the background classes among themselves (each bg class "
        "rescaled to an equal share of the total bg weight) before the "
        "signal/bg equalisation.  Use when one bg class would otherwise swamp "
        "another (e.g. Region-A ttbar + full-region QCD).",
        default=False,
    )
    add_bool_arg(parser, "plot-train-test", "Make train/test ROC plots", default=True)
    add_bool_arg(
        parser,
        "evaluate-only",
        "Skip training: load existing trained_bdt.model and only run evaluation.",
        default=False,
    )
    return parser.parse_args()


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def _year_of_dir(version_dir: str | Path) -> str:
    """Year-tag for a version dir <run>/bdt_datasets/<version> = run name's
    trailing ``_<year>`` token (e.g. ``..._2023`` -> ``2023``)."""
    return Path(version_dir).parent.parent.name.rsplit("_", 1)[-1]


def _load_split(
    version_dirs: list[str],
    split: str,
    add_year_feature: bool,
    all_years: list[str],
) -> pd.DataFrame:
    """Read ``<dir>/<split>.parquet`` for every version dir, tag each with its
    year, optionally append one-hot ``year_<tag>`` columns, and concatenate.

    Keeping the per-year parquets as the only on-disk copy (no combined
    duplicate); the join + year feature happen here in memory.
    """
    multi = len(version_dirs) > 1 or add_year_feature
    frames = []
    for d in version_dirs:
        p = Path(d) / f"{split}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing")
        df = pd.read_parquet(p)
        if multi:
            y = _year_of_dir(d)
            df["year"] = y  # bookkeeping (excluded from features via META_COLS)
            if add_year_feature:
                for yy in all_years:
                    df[f"year_{yy}"] = np.float32(1.0 if yy == y else 0.0)
        frames.append(df)
    return pd.concat(frames, axis=0, ignore_index=True) if len(frames) > 1 else frames[0]


def _year_of_base_dir(base_dir: str | Path) -> str:
    """Year for a base-parent dir ``<run>/bdt_datasets_base`` = run name's
    trailing ``_<year>`` token."""
    return Path(base_dir).parent.name.rsplit("_", 1)[-1]


def _load_version_yaml(path: str | Path) -> dict:
    """Load a version-composition YAML:  name + classes mapping each training
    class to a ``{base: <regionA|full>, sample: <stream>}`` source."""
    import yaml  # noqa: PLC0415

    cfg = yaml.safe_load(Path(path).read_text())
    if not isinstance(cfg, dict) or "name" not in cfg or "classes" not in cfg:
        raise SystemExit(f"version config {path} must have 'name' and 'classes'")
    return cfg


def _load_split_composed(
    base_dirs: list[str],
    split: str,
    version_cfg: dict,
    add_year_feature: bool,
    all_years: list[str],
) -> pd.DataFrame:
    """Compose a version's split from the base datasets: for each training
    class, take the ``(base, sample)`` stream named in ``version_cfg`` from each
    year's ``<base_dir>/base_<base>/<split>.parquet``, relabel it to the class,
    and concatenate.  No per-version copy on disk — the bases hold each event
    once; the same stream can feed different versions."""
    multi = len(base_dirs) > 1 or add_year_feature
    frames: list[pd.DataFrame] = []
    for d in base_dirs:
        y = _year_of_base_dir(d)
        base_cache: dict[str, pd.DataFrame] = {}
        for cls, spec in version_cfg["classes"].items():
            base_name, sample = spec["base"], spec["sample"]
            if base_name not in base_cache:
                p = Path(d) / f"base_{base_name}" / f"{split}.parquet"
                if not p.exists():
                    raise FileNotFoundError(f"{p} missing")
                base_cache[base_name] = pd.read_parquet(p)
            bdf = base_cache[base_name]
            sub = bdf[bdf["sample"] == sample].copy()
            if sub.empty:
                logger.warning(
                    f"  compose {cls}: sample {sample!r} absent from "
                    f"base_{base_name}/{split} ({d})"
                )
            sub["sample"] = cls
            sub["label"] = NAME_TO_LABEL[cls]
            if multi:
                # Prefer the base's PER-EVENT 'year' (set in prepare_bdt_data
                # from the cache 'era' column) so a single combined-cache base
                # still gets real categorical year; fall back to the dir year.
                if "year" not in sub.columns:
                    sub["year"] = y
                if add_year_feature:
                    yr = sub["year"].to_numpy()
                    for yy in all_years:
                        sub[f"year_{yy}"] = (yr == yy).astype(np.float32)
            frames.append(sub)
    return pd.concat(frames, axis=0, ignore_index=True)


def _build_indexed(
    df: pd.DataFrame, feat_cols: list[str], training_keys: list[str]
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Build multi-index (training_key, row) frame + weight series."""
    pieces_x: list[pd.DataFrame] = []
    pieces_w: list[pd.Series] = []
    keys_present: list[str] = []
    for key in training_keys:
        lab = NAME_TO_LABEL[key]
        m = df["label"].to_numpy() == lab
        if not m.any():
            continue
        pieces_x.append(df.loc[m, feat_cols].reset_index(drop=True))
        pieces_w.append(df.loc[m, "weight"].reset_index(drop=True).astype(np.float32))
        keys_present.append(key)
    if not pieces_x:
        raise RuntimeError("no training events found in any class")
    X = pd.concat(pieces_x, keys=keys_present)
    w = pd.concat(pieces_w, keys=keys_present)
    return X, w, keys_present


def _equalize_weights(
    weights: pd.Series,
    sig_keys: list[str],
    bg_keys: list[str],
    training_keys: list[str],
    split_name: str,
    balance_bg: bool = False,
) -> pd.Series:
    """Mirror of ``TrainBDT.preprocess_data``'s equalize-weights branch.

    Background weights are kept as ``|w|``; signal weights are rescaled so
    Σw_sig (per signal) = Σw_bg / num_sigs.

    If ``balance_bg`` is set, each background class is first rescaled to an
    equal share of the (preserved) total background weight, so one bg class
    can't swamp another — needed e.g. for Region-A ttbar + full-region QCD.

    Implementation note: assignment via ``w_out.loc[level0] = series`` on a
    multi-index Series does index-aligned assignment between a multi-index
    slice and a single-level Series, which can silently introduce NaNs.
    We instead build a per-key scale vector and multiply once.
    """
    abs_w = np.abs(weights).astype(np.float64)
    bg_present = [k for k in bg_keys if k in training_keys]
    sig_present = [k for k in sig_keys if k in training_keys]

    scales: dict[str, float] = dict.fromkeys(training_keys, 1.0)

    # Optionally balance the bg classes among themselves first.  Total bg
    # weight is preserved (each class -> bg_total/n_bg), so the downstream
    # signal/bg equalisation is unaffected in aggregate.
    if balance_bg and len(bg_present) > 1:
        bg_totals = {k: float(abs_w.loc[k].sum()) for k in bg_present}
        bg_total_pre = sum(bg_totals.values())
        target = bg_total_pre / len(bg_present)
        for k in bg_present:
            if bg_totals[k] > 0:
                scales[k] = target / bg_totals[k]
                logger.info(
                    f"  {split_name}: balance-bg scaling {k} by {scales[k]:.4g} "
                    f"(Σw {bg_totals[k]:.3f} -> {target:.3f})"
                )

    bg_total = float(sum(scales[k] * abs_w.loc[k].sum() for k in bg_present))
    for sk in sig_present:
        sig_total = float(abs_w.loc[sk].sum())
        if sig_total <= 0:
            logger.warning(f"  {split_name}: {sk} has Σ|w| ≤ 0; leaving its weights unscaled")
            continue
        scale = (bg_total / sig_total) / len(sig_present)
        scales[sk] = scale
        logger.info(f"  {split_name}: scaling {sk} by {scale:.4f}")

    level0 = abs_w.index.get_level_values(0)
    scale_arr = np.fromiter(
        (scales.get(k, 1.0) for k in level0), dtype=np.float64, count=len(level0)
    )
    w_out = abs_w * scale_arr  # element-wise; preserves multi-index

    for key in training_keys:
        if key in level0:
            post = float(w_out.loc[key].sum())
            logger.info(f"  {split_name}: total {key} after equalize: {post:.3f}")
    return w_out.astype(np.float32)


def _build_targets(
    X_indexed: pd.DataFrame,
    sig_keys: list[str],
    label_encoder: LabelEncoder,
) -> tuple[np.ndarray, np.ndarray]:
    """Build multiclass-encoded ``y`` (label-encoded) + binary ``yt`` (sig vs bg)."""
    level0 = X_indexed.index.get_level_values(0)
    y = label_encoder.transform(list(level0))
    yt = np.isin(level0, sig_keys).astype(np.int64)
    return y, yt


def _drop_bad_weights(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Drop rows whose ``|weight|`` is 0 or non-finite.

    Run before any other processing so equalize-weights sees only clean
    inputs (otherwise a near-zero ``bg_total`` can collapse the per-class
    scale and zero out an entire signal class downstream).

    Negative weights are kept here — they are flipped to positive at
    training time via ``|w|``.  Only exact 0 / NaN / inf are dropped,
    since XGBoost requires ``weight > 0`` strictly.
    """
    arr = np.abs(df["weight"].to_numpy())
    keep = np.isfinite(arr) & (arr > 0)
    n_drop = int((~keep).sum())
    if n_drop == 0:
        return df
    per_sample = df.loc[~keep].groupby("sample").size().sort_values(ascending=False)
    breakdown = ", ".join(f"{s}={int(n)}" for s, n in per_sample.items())
    logger.warning(
        f"  {split_name}: dropping {n_drop} events with |w| = 0 or "
        f"non-finite ({n_drop / len(df):.3%} of split) — {breakdown}"
    )
    return df.loc[keep].reset_index(drop=True)


def _summarize_weight_signs(df: pd.DataFrame, split_name: str) -> None:
    n_neg = int((df["weight"] < 0).sum())
    n_zero = int((df["weight"] == 0).sum())
    if n_neg or n_zero:
        per_sample = df.loc[df["weight"] <= 0].groupby("sample").size().sort_values(ascending=False)
        breakdown = ", ".join(f"{s}={n}" for s, n in per_sample.items())
        logger.info(
            f"  {split_name}: |weight| applied to {n_neg} negative + {n_zero} zero "
            f"events ({(n_neg + n_zero) / len(df):.3%}) — {breakdown}"
        )


def _evaluate_held_out(
    model,
    test_df: pd.DataFrame,
    feat_cols: list[str],
    out_dir: Path,
    label: str = "",
) -> dict:
    """Evaluate on a held-out test split and save predictions + per-class AUC.

    The output parquet preserves the raw signed ``finalWeight`` (so
    downstream ROC plotting can pick its own weighting convention).

    AUC is computed with ``|w|``: ``sklearn.roc_auc_score`` calls
    ``auc(fpr, tpr)`` internally and rejects the non-monotonic ``fpr``
    that signed weights produce (NLO ttbar @ POWHEG, ~3-10% negative).
    The difference between signed-weight and ``|w|`` AUC is well under
    1%, so this stays consistent with the canonical signed-weight ROC
    curve plotted by ``plot_roc_compare``.
    """
    suffix = f"_{label}" if label else ""
    X_test = test_df[feat_cols].to_numpy(np.float32)
    y_test = test_df["label"].to_numpy(np.int64)
    w_signed = test_df["weight"].to_numpy(np.float32)
    w_abs = np.abs(w_signed)
    probs = model.predict_proba(X_test)

    test_pred_df = test_df.copy()
    for c in range(probs.shape[1]):
        test_pred_df[f"prob_{LABEL_NAMES[c]}"] = probs[:, c]
    out_path = out_dir / f"test_predictions{suffix}.parquet"
    test_pred_df.to_parquet(out_path)
    logger.info(f"saved {out_path}")

    # Only signal-vs-rest AUCs are meaningful here — the BDT is a
    # signal/background discriminator, so OvR-AUC for the bg classes
    # (ttbar, qcd) doesn't correspond to anything analysis-relevant.
    sig_labels = {0, 1}  # hh4b, vbfhh4b-k2v0
    aucs: dict[str, float] = {}
    for c, name in LABEL_NAMES.items():
        if c not in sig_labels:
            continue
        y_bin = (y_test == c).astype(np.int64)
        if 0 < y_bin.sum() < len(y_bin):
            aucs[name] = float(roc_auc_score(y_bin, probs[:, c], sample_weight=w_abs))
    return aucs


def _run_cross_eval(
    model,
    own_test_df: pd.DataFrame,
    feat_cols: list[str],
    model_dir: Path,
    train_label: str,
    eval_test_dfs: dict[str, pd.DataFrame],
    extra_metrics: dict | None = None,
) -> None:
    """Evaluate ``model`` on ``own_test_df`` and on each cross-eval version's
    pre-loaded test set in ``eval_test_dfs`` (version name -> DataFrame); save
    per-version predictions and merge AUCs into ``model_dir/metrics.json``.
    """
    all_aucs: dict[str, dict[str, float]] = {}
    all_aucs[train_label] = _evaluate_held_out(
        model, own_test_df, feat_cols, model_dir, label=train_label
    )
    logger.info(f"test AUCs on {train_label} (own test): {all_aucs[train_label]}")

    for version, extra_test_df in eval_test_dfs.items():
        if version == train_label:
            continue
        extra_test_df = _drop_bad_weights(extra_test_df, f"test/{version}")  # noqa: PLW2901
        all_aucs[version] = _evaluate_held_out(
            model, extra_test_df, feat_cols, model_dir, label=version
        )
        logger.info(f"test AUCs on {version} (cross): {all_aucs[version]}")

    # Merge with any existing metrics.json so eval-only mode preserves
    # the training-time fields (classifier_params, training_keys, ...).
    metrics_path = model_dir / "metrics.json"
    existing = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics = {
        **existing,
        "train_dataset": train_label,
        "test_OvR_AUCs_weighted_by_dataset": all_aucs,
        "features": feat_cols,
    }
    if extra_metrics:
        metrics.update(extra_metrics)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    logger.info(f"saved {metrics_path}")


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)  # noqa: NPY002
    args.txbb_str = txbb_strings[args.txbb]

    add_year = args.add_year_feature

    # Two ways to source splits:
    #   (A) composition mode (--version-config): build the trained version (and
    #       cross-eval versions) from base datasets via YAML — no per-version
    #       parquet on disk.
    #   (B) dir mode (--dataset-dir / --dataset-dirs): read per-version parquets.
    # Both expose the same closures: load_split(split) for the trained version
    # and load_eval_test_dfs() -> {version: test_df} for cross-eval.
    if args.version_config:
        if not args.base_dirs:
            raise SystemExit("--version-config requires --base-dirs")
        base_dirs = list(args.base_dirs)
        train_cfg = _load_version_yaml(args.version_config)
        train_label = train_cfg["name"]
        eval_cfgs = {}
        for p in args.eval_version_configs:
            c = _load_version_yaml(p)
            eval_cfgs[c["name"]] = c
        # Component years: split each base-dir tag on '-' so a single combined
        # base ('2022-2023-2024') still yields per-year one-hots {2022,2023,2024}
        # matching the base's per-event 'year' column.
        all_years = sorted({y for d in base_dirs for y in _year_of_base_dir(d).split("-")})

        def load_split(split):
            return _load_split_composed(base_dirs, split, train_cfg, add_year, all_years)

        def load_eval_test_dfs():
            out: dict[str, pd.DataFrame] = {}
            for name, c in eval_cfgs.items():
                if name == train_label:
                    continue
                try:
                    out[name] = _load_split_composed(base_dirs, "test", c, add_year, all_years)
                except FileNotFoundError as e:
                    logger.warning(f"  cross-eval {name}: {e}; skipping")
            return out

    else:
        if args.dataset_dirs:
            dataset_dirs = list(args.dataset_dirs)
        elif args.dataset_dir:
            dataset_dirs = [args.dataset_dir]
        else:
            raise SystemExit("one of --dataset-dir / --dataset-dirs / --version-config is required")
        train_label = Path(dataset_dirs[0]).name
        eval_groups: dict[str, list[str]] = {}
        for d in args.eval_dataset_dirs:
            eval_groups.setdefault(Path(d).name, []).append(d)
        all_years = sorted(
            {_year_of_dir(d) for d in dataset_dirs}
            | {_year_of_dir(d) for dirs in eval_groups.values() for d in dirs}
        )

        def load_split(split):
            return _load_split(dataset_dirs, split, add_year, all_years)

        def load_eval_test_dfs():
            out: dict[str, pd.DataFrame] = {}
            for version, dirs in eval_groups.items():
                if version == train_label:
                    continue
                try:
                    out[version] = _load_split(dirs, "test", add_year, all_years)
                except FileNotFoundError as e:
                    logger.warning(f"  cross-eval {version}: {e}; skipping")
            return out

    if add_year:
        logger.info(f"year-aware: one-hot year columns for {all_years}")

    model_dir = Path(args.out_dir) / args.model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # ----- eval-only fast path: load model + cross-eval, skip everything else
    if args.evaluate_only:
        model_path = model_dir / "trained_bdt.model"
        if not model_path.exists():
            raise FileNotFoundError(f"--evaluate-only requested but {model_path} doesn't exist.")
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        logger.info(f"loaded existing model: {model_path}")

        test_df = load_split("test")
        _summarize_weight_signs(test_df, "test")
        test_df = _drop_bad_weights(test_df, "test")
        feat_cols = _feature_columns(test_df)
        logger.info(f"eval features ({len(feat_cols)}): {feat_cols}")

        _run_cross_eval(
            model,
            test_df,
            feat_cols,
            model_dir,
            train_label,
            load_eval_test_dfs(),
        )
        return

    train_df = load_split("train")
    val_df = load_split("val")
    test_df = load_split("test")

    for name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        _summarize_weight_signs(df, name)

    # Drop pathological-weight rows up front so equalize-weights sees only
    # clean inputs.  Without this, a class with all-zero weights would
    # survive into the multi-index but then disappear entirely after
    # XGBoost-side weight checks, causing KeyError on ``X.loc[key]``.
    train_df = _drop_bad_weights(train_df, "train")
    val_df = _drop_bad_weights(val_df, "val")
    test_df = _drop_bad_weights(test_df, "test")

    feat_cols = _feature_columns(train_df)
    logger.info(f"features ({len(feat_cols)}): {feat_cols}")
    logger.info(f"sizes  train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    training_keys_req = list(args.sig_keys) + list(args.bg_keys)
    X_train, w_train_raw, keys_train = _build_indexed(train_df, feat_cols, training_keys_req)
    X_val, w_val_raw, keys_val = _build_indexed(  # noqa: RUF059
        val_df, feat_cols, training_keys_req
    )

    sig_keys = [k for k in args.sig_keys if k in keys_train]
    bg_keys = [k for k in args.bg_keys if k in keys_train]
    training_keys = sig_keys + bg_keys
    logger.info(f"training keys (in order): {training_keys}")

    if args.equalize_weights:
        logger.info("Equalize signal weights so total signal = total bg")
        if args.balance_bg:
            logger.info("Balance background classes among themselves (equal Σw)")
        w_train = _equalize_weights(
            w_train_raw, sig_keys, bg_keys, training_keys, "train", args.balance_bg
        )
        w_val = _equalize_weights(
            w_val_raw, sig_keys, bg_keys, training_keys, "val", args.balance_bg
        )
    else:
        w_train = np.abs(w_train_raw).astype(np.float32)
        w_val = np.abs(w_val_raw).astype(np.float32)

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(training_keys)
    y_train, yt_train = _build_targets(X_train, sig_keys, label_encoder)
    y_val, yt_val = _build_targets(X_val, sig_keys, label_encoder)

    # Defensive: XGBoost requires every weight > 0.  After pre-filter +
    # equalize this should always hold; if not, surface the breakdown
    # before XGBoost's opaque "Weights must be positive values" error.
    for split_name, w_s in (("train", w_train), ("val", w_val)):
        arr = np.asarray(w_s)
        bad = ~np.isfinite(arr) | (arr <= 0)
        if bad.any():
            per_key = (
                pd.Series(bad, index=w_s.index).groupby(level=0).sum()
                if isinstance(w_s, pd.Series)
                else None
            )
            raise RuntimeError(
                f"  {split_name}: {int(bad.sum())} weight values ≤ 0 or non-finite "
                f"after equalize (min={arr.min()!r}, "
                f"per-key bad={per_key.to_dict() if per_key is not None else 'n/a'})"
            )

    # Match TrainBDT.main exactly: no objective/num_class/eval_metric;
    # XGBClassifier auto-detects multinomial from y.
    classifier_params = {
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "n_estimators": args.n_estimators,
        "verbosity": 2,
        "reg_lambda": 1.0,
    }

    model, evals_result = TrainBDT.train_model(
        X_train,
        X_val,
        y_train,
        y_val,
        w_train,
        w_val,
        training_keys,
        model_dir,
        **classifier_params,
    )
    TrainBDT.plot_losses(evals_result, model_dir, multiclass=True)

    if args.plot_train_test:
        TrainBDT.plot_train_test(
            X_train,
            y_train,
            yt_train,
            w_train,
            X_val,
            y_val,
            yt_val,
            w_val,
            model,
            True,  # multiclass
            sig_keys,
            training_keys,
            model_dir,
            args.txbb_str,
        )

    _run_cross_eval(
        model,
        test_df,
        feat_cols,
        model_dir,
        train_label,
        load_eval_test_dfs(),
        extra_metrics={
            "training_keys": training_keys,
            "classifier_params": classifier_params,
        },
    )


if __name__ == "__main__":
    main()
