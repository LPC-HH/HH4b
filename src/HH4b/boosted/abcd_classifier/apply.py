"""
Apply the trained ABCDnn classifier to Region-B data events, compute the
per-event "data-in-B → QCD-in-A" weight, and build A-region QCD shape
histograms.  See notes/ABCDnn.md Task 5.

Per-event weight (canonical formula, §1.3 of the plan):

    TF(x)        = (P(data,C|x) - P(ttbar,C|x)) / (P(data,D|x) - P(ttbar,D|x))
    purity(x)    = (P(data,B|x) - P(ttbar,B|x)) /  P(data,B|x)
    w_QCD^A(x)   = TF(x) · purity(x)         # --mode purity (default)

Or in mc-subtract mode (--mode mc-subtract):

    For e ∈ B-data:    w(x_e) = +1            · TF(x_e)
    For e ∈ B-ttbar:   w(x_e) = -finalWeight  · TF(x_e)

Outputs land under ``<run-dir>/apply/``:

  * ``per_event_weights.parquet``  — per Region-B event:
        event_id, sample, P0..P5, TF, purity, w_QCD_A, w_clipped, kept
  * ``h_QCD_A_<safe_var>.pkl``     — predicted A-region QCD shape per plot var
        ``{'bins': ndarray, 'h': ndarray, 'h_err': ndarray, 'var': (name, idx)}``

k-fold ensemble (``--ensemble-dir <run-dir>/kfold<K>_s<S>``, members from
``train.py --n-folds K --fold i``; purity mode only).  The nominal columns above
still come from ``<run-dir>/best_model.pt`` (unchanged), and the outputs go to
``<ensemble-dir>/apply/`` instead of ``<run-dir>/apply/``:

  * ``per_event_weights.parquet`` — the nominal rows and columns, plus
        kfold                       fold of the row (-1 = test split: no member trained on it;
                                    fold i: only member i did not train on it)
        w_QCD_A_fixclip             nominal clip(clip(TF,0)·clip(purity,0), 0, w_max)
        <q>_{median,mean,std,q16,q84}   for q in w_QCD_A, w_QCD_A_fixclip, TF, purity:
                                    per-event statistics over the K members (median = central
                                    value as in HIG-24-010 boosted, mean as in its resolved
                                    channel; std with ddof=1; q16/q84 = linear-interpolated
                                    16%/84% quantiles); NaN outside Region-B data
        n_members_kept              members with a valid TF and purity (-1 outside Region-B data)
    Each member's weight is computed exactly like the nominal one (``compute_weights``).
  * ``member_weights.parquet``    — Region-B data rows (same order and event_id as above):
        kfold, w_QCD_A_f<ii>, w_QCD_A_fixclip_f<ii> per member ii (--no-save-member-weights skips)
  * ``h_QCD_A_<safe_var>.pkl``    — the nominal keys plus ``'ensemble'``: per weight, the K
        member templates, their per-bin median/mean/std/q16/q84 and halfwidth (q84-q16)/2
        (the HIG-24-010 per-bin uncertainty), and the templates of the per-event median and
        mean weights
  * ``ensemble_summary.json``     — partition hashes, training settings, per-member training and
        common-test-split metrics, and Σw ratios to the nominal model (no absolute Σw or event
        counts: they include the blinded m(H2) window)
    Only members that finished training (completion marker ``fold<ii>/metrics.json``, same
    attempt, matching best_model.pt sha256) with identical training settings are accepted.
"""

from __future__ import annotations

import argparse
import json
import logging
import logging.config
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from HH4b import hh_vars
from HH4b.log_utils import log_config

from . import dataset as ds_mod
from ._argparse_utils import add_bool_arg

log_config["root"]["level"] = "INFO"
logging.config.dictConfig(log_config)
logger = logging.getLogger("ABCDnn.apply")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the trained ABCDnn classifier: compute per-event "
        "weights w_QCD^A(x) on B-data events and build A-region QCD shape "
        "histograms.",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="run directory written by train.py (best_model.pt, feature_stats.json).",
    )
    parser.add_argument(
        "--bdt-inference-dir",
        default="/ceph/cms/store/user/zichun/bbbb/signal_processed/bdt_inference",
        help="base directory containing the cached post-inference pickles.",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="BDT training directory name; cache subdirectory.",
    )
    parser.add_argument(
        "--year",
        nargs="+",
        type=str,
        default=["2022"],
        choices=hh_vars.years + ["2022-2023", "2022-2023-2024", "2022-2025"],
    )
    parser.add_argument(
        "--mode",
        choices=["purity", "mc-subtract"],
        default="purity",
    )
    parser.add_argument(
        "--negative-weight-policy",
        choices=["keep", "clip"],
        default="clip",
        help="'clip' sets negative w_QCD^A to 0 (use for downstream ML); "
        "'keep' preserves negatives (use for unbiased histograms).",
    )
    parser.add_argument(
        "--plot-vars",
        nargs="+",
        default=["bdt_score"],
        help="variables to histogram in the A region. Format: 'name' or "
        "'name:idx' or 'name:idx:lo,hi,nbins'.",
    )
    parser.add_argument(
        "--tf-denom-eps",
        type=float,
        default=1e-6,
        help="drop events where |P(data,D|x) - P(ttbar,D|x)| < this.",
    )
    parser.add_argument(
        "--purity-denom-eps",
        type=float,
        default=1e-6,
        help="drop events where |P(data,B|x)| < this (--mode purity only).",
    )
    parser.add_argument(
        "--w-clamp",
        type=float,
        nargs=2,
        default=[-100.0, 100.0],
        metavar=("MIN", "MAX"),
        help="clamp w_QCD^A to this range. Under --negative-weight-policy "
        "clip, MIN is forced to 0.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4096,
        help="forward-pass batch size.",
    )
    parser.add_argument(
        "--ensemble-dir",
        default=None,
        help="k-fold ensemble dir (<run-dir>/kfold<K>_s<S>, from train.py --n-folds): also "
        "apply all K members and write per-event median/mean/std/q16/q84 columns of w_QCD_A, "
        "w_QCD_A_fixclip, TF and purity next to the nominal ones, into <ensemble-dir>/apply/.  "
        "Default: single model only.",
    )
    add_bool_arg(
        parser,
        "save-member-weights",
        "With --ensemble-dir, also write every member's per-event weights "
        "(apply/member_weights.parquet)",
        default=True,
    )
    args = parser.parse_args()
    if args.ensemble_dir is not None and args.mode != "purity":
        parser.error("--ensemble-dir needs --mode purity")
    return args


# ------------------------------------------------------------------
# Histogram helpers
# ------------------------------------------------------------------


_DEFAULT_PLOT_BINS = {
    "bdt_score": np.linspace(0, 1, 41),
    "bdt_score_vbf": np.linspace(0, 1, 41),
    "bbFatJetPt": np.linspace(250, 1500, 41),
    "bbFatJetParT3TXbb": np.linspace(0, 1, 41),
    "bbFatJetParT3massX2p": np.linspace(40, 250, 43),
    "bbFatJetMsd": np.linspace(0, 300, 31),
    "MET_pt": np.linspace(0, 600, 31),
}


def _resolve_plot_var(var_spec: str) -> tuple[tuple, str, np.ndarray]:
    """Parse 'name', 'name:idx', or 'name:idx:lo,hi,nbins' into a flat key,
    label, and bin edges.  Mirrors bdt_ABCD_study.resolve_plot_var.
    """
    parts = var_spec.split(":")
    name = parts[0]
    idx = int(parts[1]) if len(parts) > 1 and parts[1] != "" else 0
    key = (name, idx)
    if len(parts) > 2:
        lo, hi, nb = parts[2].split(",")
        bins = np.linspace(float(lo), float(hi), int(nb) + 1)
    else:
        bins = _DEFAULT_PLOT_BINS.get(name, np.linspace(0, 1, 41))
    label = name if idx == 0 else f"{name}[{idx}]"
    return key, label, bins


# ------------------------------------------------------------------
# Forward-pass batching
# ------------------------------------------------------------------


def _softmax_probs(model, X_std: np.ndarray, device, batch_size: int) -> np.ndarray:
    """Run ``model`` over ``X_std`` in mini-batches, return ``(N, 6)`` softmax
    probabilities as float64 numpy.
    """
    import torch  # noqa: PLC0415

    model.eval()
    out = np.empty((len(X_std), model.n_classes), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(X_std), batch_size):
            stop = min(start + batch_size, len(X_std))
            xb = torch.from_numpy(X_std[start:stop]).to(device, non_blocking=True)
            logits = model(xb)
            probs = torch.softmax(logits, dim=-1)
            out[start:stop] = probs.cpu().numpy()
    return out


# ------------------------------------------------------------------
# Per-event weights (the canonical formula)
# ------------------------------------------------------------------


def compute_weights(
    probs: np.ndarray,
    mode: str,
    tf_denom_eps: float,
    purity_denom_eps: float,
    w_clamp: tuple[float, float],
    negative_weight_policy: str,
) -> dict[str, np.ndarray]:
    """
    Per-event TF, purity, w_QCD^A and a `kept` mask.

    Class index encoding (§1.2 of the plan):
        0 = (B, data)   1 = (B, ttbar)
        2 = (C, data)   3 = (C, ttbar)
        4 = (D, data)   5 = (D, ttbar)
    """
    P_B_data = probs[:, 0]
    P_B_tt = probs[:, 1]
    P_C_data = probs[:, 2]
    P_C_tt = probs[:, 3]
    P_D_data = probs[:, 4]
    P_D_tt = probs[:, 5]

    qcd_C = P_C_data - P_C_tt
    qcd_D = P_D_data - P_D_tt
    qcd_B = P_B_data - P_B_tt

    # numerator/denominator guards
    keep_tf = np.abs(qcd_D) >= tf_denom_eps
    if mode == "purity":
        keep_purity = np.abs(P_B_data) >= purity_denom_eps
    else:
        keep_purity = np.ones_like(keep_tf)
    kept = keep_tf & keep_purity

    # safe division (sentinel for dropped events)
    safe_qcd_D = np.where(keep_tf, qcd_D, 1.0)
    TF = qcd_C / safe_qcd_D
    TF[~keep_tf] = 0.0

    if mode == "purity":
        safe_P_B_data = np.where(keep_purity, P_B_data, 1.0)
        purity = qcd_B / safe_P_B_data
        purity[~keep_purity] = 0.0
        w_raw = TF * purity
    else:
        # mc-subtract: w = TF (for B-data) / -finalWeight*TF (for B-ttbar)
        # The caller handles the sample split.  Here we just expose TF.
        purity = np.full_like(TF, np.nan)
        w_raw = TF

    # clamp + neg-weight policy
    w_min, w_max = w_clamp
    if negative_weight_policy == "clip":
        w_min = max(w_min, 0.0)
    w = np.clip(w_raw, w_min, w_max)

    out = {
        "P0": P_B_data,
        "P1": P_B_tt,
        "P2": P_C_data,
        "P3": P_C_tt,
        "P4": P_D_data,
        "P5": P_D_tt,
        "TF": TF.astype(np.float32),
        "purity": purity.astype(np.float32),
        "w_QCD_A_raw": w_raw.astype(np.float32),
        "w_QCD_A": w.astype(np.float32),
        "kept": kept,
    }
    if mode == "purity":
        # fixed clip (2026-09-23): each factor clipped at 0 before the product, so that
        # TF < 0 & purity < 0 events get 0 instead of a positive weight
        w_fix = np.clip(np.clip(TF, 0.0, None) * np.clip(purity, 0.0, None), 0.0, w_max)
        out["w_QCD_A_fixclip"] = w_fix.astype(np.float32)
    return out


# ------------------------------------------------------------------
# k-fold ensemble
# ------------------------------------------------------------------

ENSEMBLE_STATS = ("median", "mean", "std", "q16", "q84")
ENSEMBLE_QUANTITIES = ("w_QCD_A", "w_QCD_A_fixclip", "TF", "purity")
ENSEMBLE_WEIGHTS = ("w_QCD_A", "w_QCD_A_fixclip")


def ensemble_stats(a: np.ndarray) -> dict[str, np.ndarray]:
    """Statistics over the members (axis 0) of ``a`` (K members x N entries),
    in float64: median (mean of the two middle members for even K), mean,
    std (ddof=1), and the 16%/84% quantiles (numpy's default linear
    interpolation)."""
    a = np.asarray(a, dtype=np.float64)
    q16, q84 = np.quantile(a, [0.16, 0.84], axis=0)
    return {
        "median": np.median(a, axis=0),
        "mean": a.mean(axis=0),
        "std": a.std(axis=0, ddof=1) if len(a) > 1 else np.zeros(a.shape[1:]),
        "q16": q16,
        "q84": q84,
    }


def _load_ensemble(
    ens_dir: Path, stats: dict, npz, device
) -> tuple[list[dict], np.ndarray, dict, list[dict], dict]:
    """Load the K members of ``ens_dir`` and prove that they belong together.

    Checks: the stored fold assignment is the one ``dataset.kfold_partition``
    gives for this NPZ (labels, train/val/test split) and matches its hashes;
    every member ``fold<ii>`` exists for ii in [0, K) and finished training
    (its completion marker ``metrics.json`` exists, belongs to the same attempt
    as its ``kfold_member.json`` and records the sha256 of its
    ``best_model.pt``); it records the same hashes and its own fold, training
    settings identical to every other member's with init seed base + ii, and
    the run's features and region definition.
    Returns ``(members, fold, assignment, member_infos, train_config)``.
    """
    import torch  # noqa: PLC0415

    from .model import ABCDClassifier  # noqa: PLC0415

    assign = json.loads((ens_dir / f"{ds_mod.KFOLD_ASSIGNMENT}.json").read_text())
    k, fold_seed = assign["n_folds"], assign["fold_seed"]
    fold = np.load(ens_dir / f"{ds_mod.KFOLD_ASSIGNMENT}.npz")["fold"]
    y, idx_test = npz["y"], npz["idx_test"]
    expected = ds_mod.kfold_partition(y, npz["idx_train"], npz["idx_val"], idx_test, k, fold_seed)
    if not np.array_equal(fold, expected):
        raise RuntimeError(f"{ens_dir}: stored fold assignment != kfold_partition of this NPZ")
    fp = ds_mod.kfold_fingerprint(y, idx_test, fold, k, fold_seed)
    if fp != assign:
        raise RuntimeError(f"{ens_dir}: partition fingerprint mismatch: {fp} vs {assign}")

    missing = [
        i
        for i in range(k)
        if not (ens_dir / ds_mod.member_dirname(i) / "best_model.pt").exists()
        or not (ens_dir / ds_mod.member_dirname(i) / ds_mod.KFOLD_MEMBER_JSON).exists()
    ]
    if missing:
        raise FileNotFoundError(f"{ens_dir}: members {missing} of {k} are not trained")
    # completion: a preempted, failed or still-running member has no marker
    unfinished = [
        i
        for i in range(k)
        if not (ens_dir / ds_mod.member_dirname(i) / ds_mod.KFOLD_MEMBER_DONE).exists()
    ]
    if unfinished:
        raise RuntimeError(
            f"{ens_dir}: members {unfinished} of {k} did not finish training (no "
            f"{ds_mod.KFOLD_MEMBER_DONE}); relaunch them"
        )

    same_keys = ("feature_set", "feature_names", "txbb_bins", "mass_bins", "txbb_str")
    same_keys += ("mass_str", "txbb_jet_index", "mass_jet_index", "label_names")
    members, infos = [], []
    train_config = None
    for i in range(k):
        mdir = ens_dir / ds_mod.member_dirname(i)
        info = json.loads((mdir / ds_mod.KFOLD_MEMBER_JSON).read_text())
        bad = {key: (info.get(key), v) for key, v in assign.items() if info.get(key) != v}
        if bad or info["fold"] != i:
            raise RuntimeError(f"{mdir}: member partition does not match the ensemble: {bad}")
        metrics = json.loads((mdir / ds_mod.KFOLD_MEMBER_DONE).read_text())
        if "best_model_sha256" not in metrics:
            raise RuntimeError(
                f"{mdir}: {ds_mod.KFOLD_MEMBER_DONE} has no completion record (member trained "
                "by an older train.py); relaunch the member"
            )
        if metrics.get("kfold") != info:
            raise RuntimeError(
                f"{mdir}: {ds_mod.KFOLD_MEMBER_DONE} and {ds_mod.KFOLD_MEMBER_JSON} come from "
                "different training attempts; relaunch the member"
            )
        if metrics.get("best_model_sha256") != ds_mod.file_sha256(mdir / "best_model.pt"):
            raise RuntimeError(
                f"{mdir}: best_model.pt is not the model recorded in {ds_mod.KFOLD_MEMBER_DONE}"
            )
        cfg = info.get("train_config")
        if cfg is None or tuple(cfg) != ds_mod.KFOLD_TRAIN_CONFIG_KEYS:
            raise RuntimeError(f"{mdir}: {ds_mod.KFOLD_MEMBER_JSON} lacks the training settings")
        if train_config is None:
            train_config = cfg
        if cfg != train_config:
            diff = {
                key: (train_config[key], cfg[key]) for key in cfg if cfg[key] != train_config[key]
            }
            raise RuntimeError(f"{mdir}: training settings differ from fold00's: {diff}")
        if info["seed"] != cfg["base_seed"] + i:
            raise RuntimeError(f"{mdir}: init seed {info['seed']} != {cfg['base_seed']} + {i}")
        mstats = json.loads((mdir / "feature_stats.json").read_text())
        bad = [key for key in same_keys if mstats.get(key) != stats.get(key)]
        if bad:
            raise RuntimeError(f"{mdir}: feature_stats differ from the run's in {bad}")
        mc = mstats["model_config"]
        arch = {key: cfg[key] for key in ("hidden", "num_hidden_layers", "dropout")}
        if mc != arch:
            raise RuntimeError(f"{mdir}: model_config {mc} != training settings {arch}")
        model = ABCDClassifier(
            d_in=len(mstats["feature_names"]),
            hidden=mc["hidden"],
            num_hidden_layers=mc["num_hidden_layers"],
            dropout=mc["dropout"],
        ).to(device)
        sd = torch.load(mdir / "best_model.pt", map_location=device, weights_only=True)
        model.load_state_dict(sd)
        model.eval()
        members.append(
            {
                "fold": i,
                "model": model,
                "mu": np.asarray(mstats["mu"], dtype=np.float32),
                "sigma": np.asarray(mstats["sigma"], dtype=np.float32),
            }
        )
        infos.append(
            {
                "fold": i,
                "seed": info["seed"],
                "n_train": info["n_train"],
                "n_val": info["n_val"],
                **{
                    key: metrics[key]
                    for key in ("best_val_loss", "best_epoch", "epochs_run", "stopped_early")
                },
                **{key: metrics[key] for key in ("test_loss", "test_acc", "val_aucs", "test_aucs")},
                "provenance": info.get("provenance"),
            }
        )
    logger.info(
        f"loaded {k} finished ensemble members from {ens_dir} (fold seed {fold_seed}; "
        f"training settings {train_config})"
    )
    return members, fold, assign, infos, train_config


def _ensemble_b_data(
    members: list[dict], X_B: np.ndarray, args, device, nominal_sum: dict[str, float]
) -> dict[str, np.ndarray]:
    """Per-member TF, purity, w_QCD_A, w_QCD_A_fixclip and kept of the Region-B
    data events (``X_B``, raw features), each stacked to shape (K, N).  The log
    gives each member's Σw relative to the nominal model's (``nominal_sum``),
    never absolute sums (blinding: they include m(H2) in [110, 140] GeV)."""
    per: dict[str, list[np.ndarray]] = {q: [] for q in (*ENSEMBLE_QUANTITIES, "kept")}
    for m in members:
        X_std = ((X_B - m["mu"]) / m["sigma"]).astype(np.float32)
        probs = _softmax_probs(m["model"], X_std, device, args.batch)
        cw = compute_weights(
            probs,
            "purity",
            args.tf_denom_eps,
            args.purity_denom_eps,
            tuple(args.w_clamp),
            args.negative_weight_policy,
        )
        for q, arrays in per.items():
            arrays.append(cw[q])
        rel = {q: cw[q].sum(dtype=np.float64) / nominal_sum[q] for q in ENSEMBLE_WEIGHTS}
        logger.info(
            f"    member fold{m['fold']:02d}: sum w / nominal = {rel['w_QCD_A']:.4f} (legacy), "
            f"{rel['w_QCD_A_fixclip']:.4f} (fixclip); kept fraction "
            f"{cw['kept'].mean():.5f}"
        )
    return {q: np.stack(v) for q, v in per.items()}


def _hist(x: np.ndarray, bins: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted histogram and its sqrt(Σw²), accumulated in float64 (with explicit
    edges numpy accumulates in the weights' dtype)."""
    w = np.asarray(w, dtype=np.float64)
    h, _ = np.histogram(x, bins=bins, weights=w)
    h2, _ = np.histogram(x, bins=bins, weights=w**2)
    return h, np.sqrt(h2)


def _ensemble_templates(
    x: np.ndarray, bins: np.ndarray, members_w: dict[str, np.ndarray], ens_cols: dict
) -> dict:
    """Member templates of each ensemble weight and their per-bin statistics.

    All Region-B data events are histogrammed (a member's weight is exactly 0
    where its TF or purity is invalid, so this equals the kept-only template),
    with the weights in float64.
    ``halfwidth`` = (q84 - q16)/2 per bin is the HIG-24-010 uncertainty.
    """
    out = {}
    for wname in ENSEMBLE_WEIGHTS:
        h_members = np.stack(
            [
                np.histogram(x, bins=bins, weights=np.asarray(w, dtype=np.float64))[0]
                for w in members_w[wname]
            ]
        )
        st = ensemble_stats(h_members)
        d = {"members": h_members}
        d.update({f"members_{k}": v for k, v in st.items()})
        d["members_halfwidth"] = 0.5 * (st["q84"] - st["q16"])
        for central in ("median", "mean"):
            h, h_err = _hist(x, bins, ens_cols[f"{wname}_{central}"])
            d[f"central_{central}_w"] = h
            d[f"central_{central}_w_err"] = h_err
        out[wname] = d
    return out


def _run_ensemble(
    loaded: tuple,
    X_all: np.ndarray,
    sample_id: np.ndarray,
    region_id: np.ndarray,
    streams: list[dict],
    args,
    device,
) -> dict:
    """Apply the K members (``loaded`` = the output of ``_load_ensemble``) to
    the Region-B data events and combine them.

    Returns ``{'fold', 'members_w', 'cols', 'assign', 'infos', 'train_config',
    'nominal_sum'}``: the fold per NPZ row, the (K, N_B) member arrays, the
    per-event columns for the Region-B data rows (nominal fixed-clip weight,
    ensemble statistics, n_members_kept), the partition / member metadata, and
    the nominal Σw of each weight (only used for ratios).
    """
    members, fold, assign, infos, train_config = loaded
    sel_B = (region_id == ds_mod.REGION_TO_ID["B"]) & (sample_id == ds_mod.SAMPLE_TO_ID["data"])
    comp = next(s for s in streams if s["sample"] == "data" and s["region"] == "B")["comp"]

    # compute_weights (used for the members) must reproduce the nominal columns bit for bit
    nominal = compute_weights(
        np.stack([comp[f"P{c}"] for c in range(6)], axis=1),
        "purity",
        args.tf_denom_eps,
        args.purity_denom_eps,
        tuple(args.w_clamp),
        args.negative_weight_policy,
    )
    for q in ("TF", "purity", "w_QCD_A_raw", "w_QCD_A", "kept"):
        if not np.array_equal(nominal[q], comp[q]):
            raise RuntimeError(f"compute_weights does not reproduce the nominal {q!r} column")

    nominal_sum = {q: nominal[q].sum(dtype=np.float64) for q in ENSEMBLE_WEIGHTS}
    logger.info(f"  ensemble: {len(members)} members on the Region-B data events")
    members_w = _ensemble_b_data(members, X_all[sel_B], args, device, nominal_sum)
    cols: dict[str, np.ndarray] = {"w_QCD_A_fixclip": nominal["w_QCD_A_fixclip"]}
    for q in ENSEMBLE_QUANTITIES:
        for st, v in ensemble_stats(members_w[q]).items():
            cols[f"{q}_{st}"] = v.astype(np.float32)
    cols["n_members_kept"] = members_w["kept"].sum(axis=0).astype(np.int8)

    for wname in ENSEMBLE_WEIGHTS:
        med = cols[f"{wname}_median"].astype(np.float64)
        half = 0.5 * (cols[f"{wname}_q84"] - cols[f"{wname}_q16"]).astype(np.float64)
        pos = med > 0
        rel = np.median(half[pos] / med[pos]) if pos.any() else float("nan")
        logger.info(
            f"    {wname}: sum median / sum nominal = {med.sum() / nominal_sum[wname]:.4f}; "
            f"median per-event (q84-q16)/2 / median = {rel:.4f}"
        )
    return {
        "fold": fold,
        "members_w": members_w,
        "cols": cols,
        "assign": assign,
        "infos": infos,
        "train_config": train_config,
        "nominal_sum": nominal_sum,
    }


def _write_ensemble_extras(ens: dict, pew: pd.DataFrame, apply_dir: Path, args) -> None:
    """member_weights.parquet (unless --no-save-member-weights) and ensemble_summary.json."""
    b_rows = pew[(pew["sample"] == "data") & (pew["region"] == "B")]
    members_w = ens["members_w"]
    if args.save_member_weights:
        mw = {"event_id": b_rows["event_id"].to_numpy(), "kfold": b_rows["kfold"].to_numpy()}
        for wname in ENSEMBLE_WEIGHTS:
            for info, w in zip(ens["infos"], members_w[wname]):
                mw[f"{wname}_f{info['fold']:02d}"] = w
        path = apply_dir / "member_weights.parquet"
        pd.DataFrame(mw).to_parquet(path)
        logger.info(f"saved {path}  ({len(b_rows)} rows)")

    # Blinding: Σw over Region-B data includes m(H2) in [110, 140] GeV, so the summary holds
    # only ratios to the nominal model's Σw (same weight definition) and fractions, never sums
    # or event counts (those stay recomputable from the parquet).
    nominal_sum = ens["nominal_sum"]

    def _rel(a, wname: str) -> float:
        return float(np.sum(np.asarray(a), dtype=np.float64) / nominal_sum[wname])

    kf, kf_n = np.unique(b_rows["kfold"], return_counts=True)
    summary = {
        "assignment": ens["assign"],
        "train_config": ens["train_config"],
        "region_b_data_kfold_fraction": {str(k): float(v / len(b_rows)) for k, v in zip(kf, kf_n)},
        "sum_w_over_nominal": {
            f"{wname}_{st}": _rel(b_rows[f"{wname}_{st}"], wname)
            for wname in ENSEMBLE_WEIGHTS
            for st in ("median", "mean")
        },
        "members": [
            {
                **info,
                **{
                    f"sum_{wname}_over_nominal": _rel(members_w[wname][m], wname)
                    for wname in ENSEMBLE_WEIGHTS
                },
            }
            for m, info in enumerate(ens["infos"])
        ],
    }
    path = apply_dir / "ensemble_summary.json"
    path.write_text(json.dumps(summary, indent=2))
    logger.info(f"saved {path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    import torch  # local — apply.py needs torch but train.py does too  # noqa: PLC0415

    from .model import ABCDClassifier  # noqa: PLC0415

    run_dir = Path(args.run_dir)
    ens_dir = Path(args.ensemble_dir) if args.ensemble_dir is not None else None
    if ens_dir is not None and not (ens_dir / f"{ds_mod.KFOLD_ASSIGNMENT}.json").exists():
        raise FileNotFoundError(f"{ens_dir} is not a k-fold ensemble dir (no kfold_assignment)")
    apply_dir = (run_dir if ens_dir is None else ens_dir) / "apply"
    apply_dir.mkdir(parents=True, exist_ok=True)

    # -- load model + standardization stats -------------------------------
    stats = json.loads((run_dir / "feature_stats.json").read_text())
    feature_names = stats["feature_names"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Recreate the architecture train.py wrote into feature_stats.json
    # (falls back to the constructor defaults for legacy runs without
    # model_config — those used hidden=256, num_hidden_layers=3).
    mc = stats.get("model_config", {})
    model = ABCDClassifier(
        d_in=len(feature_names),
        hidden=mc.get("hidden", 256),
        num_hidden_layers=mc.get("num_hidden_layers", 3),
        dropout=mc.get("dropout", 0.2),
    ).to(device)
    sd = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(sd)
    model.eval()
    logger.info(
        f"loaded model from {run_dir / 'best_model.pt'} on {device}  "
        f"(hidden={model.hidden}, n_layers={model.num_hidden_layers})"
    )

    # -- load processed_data.npz; filter to Region B -----------------------
    npz_path = run_dir / "processed_data.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"{npz_path} missing — run train.py --prepare-only first.")
    npz = np.load(npz_path)
    X_all = npz["X"]  # already standardized? -- NO, raw floats.
    sample_id = npz["sample_id"]
    region_id = npz["region_id"]
    orig_idx = npz["orig_idx"]
    if "orig_idx" not in npz.files:
        raise KeyError(
            "processed_data.npz lacks 'orig_idx' — re-run train.py "
            "--prepare-only --force-rebuild to refresh."
        )

    # k-fold ensemble: validate and load all members before the expensive steps
    ens_loaded = _load_ensemble(ens_dir, stats, npz, device) if ens_dir is not None else None

    mu = np.asarray(stats["mu"], dtype=np.float32)
    sigma = np.asarray(stats["sigma"], dtype=np.float32)
    X_std_all = ((X_all - mu) / sigma).astype(np.float32)

    SAMP_DATA = ds_mod.SAMPLE_TO_ID["data"]
    SAMP_TTBAR = ds_mod.SAMPLE_TO_ID["ttbar"]
    REG_TO_ID = {r: ds_mod.REGION_TO_ID[r] for r in ("B", "C", "D")}

    # Per-region (data, ttbar) class indices in the 6-class encoding §1.2.
    # 0=(B,data) 1=(B,ttbar)  2=(C,data) 3=(C,ttbar)  4=(D,data) 5=(D,ttbar)
    REG_TO_PROB_IDX = {"B": (0, 1), "C": (2, 3), "D": (4, 5)}
    PROBS_C_DATA = REG_TO_PROB_IDX["C"][0]
    PROBS_C_TT = REG_TO_PROB_IDX["C"][1]
    PROBS_D_DATA = REG_TO_PROB_IDX["D"][0]
    PROBS_D_TT = REG_TO_PROB_IDX["D"][1]

    # -- load original pickles for plot-var lookups ----------------------
    # Year-tags are literal cache subdir names (multi-era combining done
    # upstream in prep_bdt_inference_pickles.py).  No era expansion.
    years = list(args.year)
    samples_needed = ("data",) if args.mode == "purity" else ("data", "ttbar")
    pickles = ds_mod.load_events(args.bdt_inference_dir, args.model_name, years, samples_needed)

    # -- per-event weight stream -----------------------------------------
    # For each sample × region present in NPZ, compute per-event probs and  # noqa: RUF003
    # per-region purity.  For B-data events specifically, also compute TF
    # and w_QCD_A = TF · purity (the canonical "data-in-B → QCD-in-A" weight).
    streams: list[dict[str, np.ndarray]] = []

    for sample, samp_id in (("data", SAMP_DATA), ("ttbar", SAMP_TTBAR)):
        if sample not in pickles:
            continue

        for region, reg_id in REG_TO_ID.items():
            sel = (region_id == reg_id) & (sample_id == samp_id)
            n = int(sel.sum())
            if n == 0:
                continue
            logger.info(f"  {sample}: {n} events in Region {region}")

            X_std_R = X_std_all[sel]
            oidx_R = orig_idx[sel]

            probs = _softmax_probs(model, X_std_R, device, args.batch)
            P_R_data, P_R_tt = REG_TO_PROB_IDX[region]
            qcd_R = probs[:, P_R_data] - probs[:, P_R_tt]
            keep_purity = np.abs(probs[:, P_R_data]) >= args.purity_denom_eps
            safe_P_R_data = np.where(keep_purity, probs[:, P_R_data], 1.0)
            purity_R = qcd_R / safe_P_R_data
            purity_R[~keep_purity] = 0.0

            # TF and w_QCD_A only meaningful for B-data events
            if region == "B" and sample == "data":
                qcd_C = probs[:, PROBS_C_DATA] - probs[:, PROBS_C_TT]
                qcd_D = probs[:, PROBS_D_DATA] - probs[:, PROBS_D_TT]
                keep_tf = np.abs(qcd_D) >= args.tf_denom_eps
                safe_qcd_D = np.where(keep_tf, qcd_D, 1.0)
                TF = qcd_C / safe_qcd_D
                TF[~keep_tf] = 0.0

                w_raw = TF * purity_R
                w_min, w_max = args.w_clamp
                if args.negative_weight_policy == "clip":
                    w_min = max(w_min, 0.0)
                w = np.clip(w_raw, w_min, w_max)
                kept = keep_tf & keep_purity
            else:
                # For C/D-data and any-region ttbar, only purity is computed.
                # In mc-subtract mode, B-ttbar's "weight" is -finalWeight·TF;
                # we re-derive TF below (it doesn't depend on the event's own
                # region label, just its features).
                TF = np.full(n, np.nan, dtype=np.float32)
                if args.mode == "mc-subtract" and region == "B" and sample == "ttbar":
                    qcd_C = probs[:, PROBS_C_DATA] - probs[:, PROBS_C_TT]
                    qcd_D = probs[:, PROBS_D_DATA] - probs[:, PROBS_D_TT]
                    keep_tf = np.abs(qcd_D) >= args.tf_denom_eps
                    safe_qcd_D = np.where(keep_tf, qcd_D, 1.0)
                    TF = qcd_C / safe_qcd_D
                    TF[~keep_tf] = 0.0
                    rows_for_w = pickles[sample].iloc[oidx_R].reset_index(drop=True)
                    ftw = rows_for_w["finalWeight"].to_numpy().astype(np.float32).reshape(-1)
                    w_raw = (-ftw * TF).astype(np.float32)
                    w = w_raw.copy()  # don't clip the negative ttbar leg
                    kept = keep_tf
                else:
                    w_raw = np.full(n, np.nan, dtype=np.float32)
                    w = w_raw.copy()
                    kept = np.ones(n, dtype=bool)

            comp = {
                "P0": probs[:, 0],
                "P1": probs[:, 1],
                "P2": probs[:, 2],
                "P3": probs[:, 3],
                "P4": probs[:, 4],
                "P5": probs[:, 5],
                "purity": purity_R.astype(np.float32),
                "TF": TF.astype(np.float32),
                "w_QCD_A_raw": w_raw.astype(np.float32),
                "w_QCD_A": w.astype(np.float32),
                "kept": kept,
            }

            rows = pickles[sample].iloc[oidx_R].reset_index(drop=True)
            streams.append(
                {
                    "sample": sample,
                    "region": region,
                    "rows": rows,
                    "comp": comp,
                }
            )

            # Logging summary
            n_kept = int(kept.sum())
            wk = w[kept]
            if region == "B" and sample == "data":
                qs = np.quantile(wk, [0.01, 0.5, 0.99]) if len(wk) > 0 else (0, 0, 0)
                logger.info(
                    f"    B-data: w_QCD_A mean={wk.mean():.4f}  "
                    f"median={qs[1]:.4f}  q01={qs[0]:.4f}  q99={qs[2]:.4f}  "
                    f"sum={wk.sum():.2f}  kept={n_kept}/{n}"
                )
            else:
                pk = purity_R[keep_purity]
                if len(pk) > 0:
                    qs = np.quantile(pk, [0.01, 0.5, 0.99])
                    logger.info(
                        f"    {region}-{sample}: purity mean={pk.mean():.4f}  "
                        f"median={qs[1]:.4f}  q01={qs[0]:.4f}  q99={qs[2]:.4f}  "
                        f"kept={n_kept}/{n}"
                    )

    # -- k-fold ensemble: every member on the Region-B data events --------
    ens = None
    if ens_dir is not None:
        ens = _run_ensemble(ens_loaded, X_all, sample_id, region_id, streams, args, device)

    # -- write per_event_weights.parquet ----------------------------------
    rows = []
    for s in streams:
        n = len(s["rows"])
        df_out = pd.DataFrame(
            {
                "event_id": np.arange(n, dtype=np.int64),
                "sample": s["sample"],
                "region": s["region"],
                "P0": s["comp"]["P0"],
                "P1": s["comp"]["P1"],
                "P2": s["comp"]["P2"],
                "P3": s["comp"]["P3"],
                "P4": s["comp"]["P4"],
                "P5": s["comp"]["P5"],
                "TF": s["comp"]["TF"],
                "purity": s["comp"]["purity"],
                "w_QCD_A_raw": s["comp"]["w_QCD_A_raw"],
                "w_QCD_A": s["comp"]["w_QCD_A"],
                "kept": s["comp"]["kept"],
            }
        )
        if ens is not None:
            is_b_data = s["sample"] == "data" and s["region"] == "B"
            sel = (region_id == REG_TO_ID[s["region"]]) & (
                sample_id == ds_mod.SAMPLE_TO_ID[s["sample"]]
            )
            df_out["kfold"] = ens["fold"][sel]
            for col, v in ens["cols"].items():
                df_out[col] = (
                    v if is_b_data else np.full(n, -1 if v.dtype == np.int8 else np.nan, v.dtype)
                )
        rows.append(df_out)

    pew = pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame()
    pew_path = apply_dir / "per_event_weights.parquet"
    pew.to_parquet(pew_path)
    logger.info(f"saved {pew_path}  ({len(pew)} rows)")
    if ens is not None:
        _write_ensemble_extras(ens, pew, apply_dir, args)

    # -- per-variable A-region QCD prediction histograms (B-data only) ---
    # The A-region prediction h_QCD^A uses B-data events weighted by
    # w_QCD_A = TF · purity_B.  See notes/ABCDnn.md §1.3.
    b_data_streams = [s for s in streams if s["sample"] == "data" and s["region"] == "B"]
    for var_spec in args.plot_vars:
        key, label, bins = _resolve_plot_var(var_spec)
        h_total = np.zeros(len(bins) - 1, dtype=np.float64)
        h_w2 = np.zeros(len(bins) - 1, dtype=np.float64)

        for s in b_data_streams:
            df_R = s["rows"]
            comp = s["comp"]
            if key not in df_R.columns:
                logger.warning(f"  var {var_spec}: column {key} not found; skipping")
                continue
            x = df_R[key].to_numpy().reshape(-1)
            w = comp["w_QCD_A"]
            kept = comp["kept"]
            x_k = x[kept]
            w_k = w[kept]

            h, _ = np.histogram(x_k, bins=bins, weights=w_k)
            h_total += h
            h2, _ = np.histogram(x_k, bins=bins, weights=w_k**2)
            h_w2 += h2

        h_err = np.sqrt(h_w2)
        safe = var_spec.replace(":", "_").replace(",", "_").replace(".", "p")
        path = apply_dir / f"h_QCD_A_{safe}.pkl"
        out = {"var": key, "label": label, "bins": bins, "h": h_total, "h_err": h_err}
        if ens is not None and b_data_streams and key in b_data_streams[0]["rows"].columns:
            x = b_data_streams[0]["rows"][key].to_numpy().reshape(-1)
            out["ensemble"] = _ensemble_templates(x, bins, ens["members_w"], ens["cols"])
            h_fix, h_fix_err = _hist(x, bins, ens["cols"]["w_QCD_A_fixclip"])
            out["ensemble"]["w_QCD_A_fixclip"]["nominal"] = h_fix
            out["ensemble"]["w_QCD_A_fixclip"]["nominal_err"] = h_fix_err
            out["ensemble"]["n_members"] = len(ens["infos"])
        with path.open("wb") as f:
            pickle.dump(out, f)
        logger.info(f"  saved {path}  Σh={h_total.sum():.2f}  ±{h_err.sum():.2f}")

    logger.info("done")


if __name__ == "__main__":
    main()
