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
from ._argparse_utils import add_bool_arg  # noqa: F401  (kept for future use)

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
        choices=hh_vars.years + ["2022-2023", "2022-2023-2024"],
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
    return parser.parse_args()


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

    return {
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


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    import torch  # local — apply.py needs torch but train.py does too  # noqa: PLC0415

    from .model import ABCDClassifier  # noqa: PLC0415

    run_dir = Path(args.run_dir)
    apply_dir = run_dir / "apply"
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
        rows.append(df_out)

    pew = pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame()
    pew_path = apply_dir / "per_event_weights.parquet"
    pew.to_parquet(pew_path)
    logger.info(f"saved {pew_path}  ({len(pew)} rows)")

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
        with path.open("wb") as f:
            pickle.dump(
                {"var": key, "label": label, "bins": bins, "h": h_total, "h_err": h_err},
                f,
            )
        logger.info(f"  saved {path}  Σh={h_total.sum():.2f}  ±{h_err.sum():.2f}")

    logger.info("done")


if __name__ == "__main__":
    main()
