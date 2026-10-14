"""
ABCD background-estimation study for QCD, using (TXbb[0], mass[0]) as the
two axes.

Four regions (index 0 = leading bb-tagged AK8 jet):
    SR     : TXbb in [txbb_hi_lo, txbb_hi_hi),  mass in [mass_hi_lo, mass_hi_hi)
    MassSB : TXbb in [txbb_hi_lo, txbb_hi_hi),  mass in [mass_lo_lo, mass_lo_hi)
    TXbbSB : TXbb in [txbb_lo_lo, txbb_lo_hi),  mass in [mass_hi_lo, mass_hi_hi)
    BothSB : TXbb in [txbb_lo_lo, txbb_lo_hi),  mass in [mass_lo_lo, mass_lo_hi)

Prediction under the ABCD assumption (TXbb and mass independent for QCD):
    N_SR_pred = N_MassSB * N_TXbbSB / N_BothSB

Two closures are reported:
  * MC-only   : uses QCD MC yields in all four regions
  * Data-driven: uses (data - ttbar) yields in the control regions, compared
    to QCD MC in SR

Example (from src/HH4b/boosted/):

    python bdt_ABCD_study.py \
        --data-path /ceph/cms/store/user/zichun/bbbb/skimmer/nanov15_20251202_v15_signal \
        --model-name 26Apr07_v15_2022_glopartv3_rawmass \
        --config-name v13_glopartv3 \
        --year 2022 --txbb glopart-v3 --mass bbFatJetParT3massX2p \
        --apply-cuts --plot --plot-vars bdt_score

Author(s): Zichun Hao
"""

from __future__ import annotations

import argparse
import importlib
import logging
import logging.config
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
import xgboost as xgb
from TrainBDT import apply_cuts

from HH4b import hh_vars
from HH4b.hh_vars import samples_run3, txbb_strings
from HH4b.log_utils import log_config
from HH4b.postprocessing import HLTs, load_run3_samples
from HH4b.postprocessing.bdt_inference import compute_scores
from HH4b.run_utils import add_bool_arg
from HH4b.utils import get_var_mapping

log_config["root"]["level"] = "INFO"
logging.config.dictConfig(log_config)
logger = logging.getLogger("ABCDStudy")

# matplotlib.mathtext logs INFO messages whenever it falls back to a different
# font for a glyph it doesn't carry (e.g. \in from STIXGeneral under CMS
# style). The substitution itself is harmless and renders correctly.
logging.getLogger("matplotlib.mathtext").setLevel(logging.WARNING)

plt.style.use(hep.style.CMS)


REGION_NAMES = ["SR", "MassSB", "TXbbSB", "BothSB"]
REGION_LABELS = {
    "SR": "A (SR)",
    "MassSB": "B (mass SB)",
    "TXbbSB": "C (TXbb SB)",
    "BothSB": "D (both SB)",
}

DEFAULT_SIGNAL_PREFIXES = ("hh4b", "vbfhh4b")

SAMPLE_COLORS = {
    "qcd": "#ffd166",
    "ttbar": "#ef476f",
    "vhtobb": "#7fb4ff",
    "novhhtobb": "#6b8cff",
    "tthtobb": "#118ab2",
    "zz": "#9b5de5",
    "nozzdiboson": "#c77dff",
    "vjets": "#73d2de",
    "hh4b": "#06d6a0",
    "vbfhh4b-k2v0": "#073b4c",
}


def classify_samples(
    sample_keys, signal_prefixes: tuple[str, ...] = DEFAULT_SIGNAL_PREFIXES
) -> dict[str, list[str]]:
    """Split sample keys into {qcd, data, signal, resonant}.  Everything that is
    neither QCD nor data nor signal-by-prefix is treated as a resonant BG."""
    groups = {"qcd": [], "data": [], "signal": [], "resonant": []}
    for k in sample_keys:
        if k == "qcd":
            groups["qcd"].append(k)
        elif k == "data":
            groups["data"].append(k)
        elif any(k.startswith(p) for p in signal_prefixes):
            groups["signal"].append(k)
        else:
            groups["resonant"].append(k)
    return groups


def build_region_masks(
    events: pd.DataFrame,
    txbb_col: str,
    mass_col: str,
    txbb_bins: tuple[float, float, float],
    mass_bins: tuple[float, float, float],
) -> dict[str, np.ndarray]:
    """Return boolean masks for the four ABCD regions, keyed by region name."""
    txbb0 = events[txbb_col].to_numpy()[:, 0]
    mass0 = events[mass_col].to_numpy()[:, 0]

    txbb_lo_lo, txbb_split, txbb_hi_hi = txbb_bins
    mass_lo_lo, mass_split, mass_hi_hi = mass_bins

    txbb_pass = (txbb0 >= txbb_split) & (txbb0 < txbb_hi_hi)
    txbb_fail = (txbb0 >= txbb_lo_lo) & (txbb0 < txbb_split)
    mass_pass = (mass0 >= mass_split) & (mass0 < mass_hi_hi)
    mass_fail = (mass0 >= mass_lo_lo) & (mass0 < mass_split)

    return {
        "SR": txbb_pass & mass_pass,
        "MassSB": txbb_pass & mass_fail,
        "TXbbSB": txbb_fail & mass_pass,
        "BothSB": txbb_fail & mass_fail,
    }


def assign_region_column(events: pd.DataFrame, masks: dict[str, np.ndarray]) -> pd.DataFrame:
    """Append a flat 'region' column (string) that labels each event or 'none'."""
    region = np.full(len(events), "none", dtype=object)
    for name in REGION_NAMES:
        region[masks[name]] = name
    events = events.copy()
    events[("region", 0)] = region
    return events


def weighted_yield(events: pd.DataFrame, mask: np.ndarray) -> tuple[float, float]:
    """Return (sum of weights, sum of weights^2) for the masked events."""
    if mask.sum() == 0:
        return 0.0, 0.0
    w = events.loc[mask, "finalWeight"].to_numpy()
    return float(w.sum()), float((w**2).sum())


def compute_region_yields(
    events_dict: dict[str, pd.DataFrame],
    masks_dict: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    """Build a (sample, region) -> (yield, stat_err) table."""
    rows = []
    for sample, events in events_dict.items():
        for region in REGION_NAMES:
            y, w2 = weighted_yield(events, masks_dict[sample][region])
            rows.append(
                {
                    "sample": sample,
                    "region": region,
                    "yield": y,
                    "stat_err": np.sqrt(w2),
                    "n_raw": int(masks_dict[sample][region].sum()),
                }
            )
    return pd.DataFrame(rows)


def abcd_predict(
    n_mass: float,
    n_txbb: float,
    n_both: float,
    e_mass: float,
    e_txbb: float,
    e_both: float,
) -> tuple[float, float]:
    """Return (predicted SR yield, propagated stat uncertainty)."""
    if n_both <= 0:
        return float("nan"), float("nan")
    pred = n_mass * n_txbb / n_both
    # relative uncertainties added in quadrature
    rel2 = 0.0
    for n, e in [(n_mass, e_mass), (n_txbb, e_txbb), (n_both, e_both)]:
        if n != 0:
            rel2 += (e / n) ** 2
    return pred, abs(pred) * np.sqrt(rel2)


def load_bdt_model(model_dir: Path) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier()
    model.load_model(fname=str(model_dir / "trained_bdt.model"))
    return model


def run_bdt_inference(
    events: pd.DataFrame,
    model: xgb.XGBClassifier,
    bdt_dataframe_fn,
    weight_ttbar_bdt: float,
    bdt_disc: bool,
) -> pd.DataFrame:
    """Append bdt_score (and bdt_score_vbf if multiclass) to events as flat cols."""
    df_in = bdt_dataframe_fn(events, get_var_mapping(""))
    preds = model.predict_proba(df_in)
    # write scores into a small frame then copy over
    scores = pd.DataFrame(index=events.index)
    compute_scores(scores, preds, jshift="", weight_ttbar=weight_ttbar_bdt, use_disc=bdt_disc)
    for col in scores.columns:
        events[(col, 0)] = scores[col].to_numpy()
    return events


def cache_path_for(args, year: str, sample: str) -> Path:
    return Path(args.bdt_inference_dir) / args.model_name / year / f"{sample}.pkl"


def load_and_prepare(
    args,
    samples_this_year: dict,
    year: str,
    model: xgb.XGBClassifier,
    bdt_dataframe_fn,
) -> dict[str, pd.DataFrame]:
    """
    Load samples for a year, apply preselection + trigger, run BDT inference.

    Per (year, sample) results are cached as pickle under
    ``<bdt_inference_dir>/<model_name>/<year>/<sample>.pkl`` — an existing cache
    file is reused unless ``--force-rerun`` is set.
    """
    events: dict[str, pd.DataFrame] = {}
    to_process: dict[str, list[str]] = {}

    for sample, subsamples in samples_this_year.items():
        cpath = cache_path_for(args, year, sample)
        if cpath.exists() and not args.force_rerun:
            logger.info(f"[{year}] loading cached {sample} from {cpath}")
            with cpath.open("rb") as f:
                events[sample] = pickle.load(f)
        else:
            to_process[sample] = subsamples

    if not to_process:
        return events

    raw = load_run3_samples(
        args.data_path,
        year,
        {year: to_process},
        reorder_txbb=True,
        load_systematics=False,
        txbb_version=args.txbb,
        scale_and_smear=False,
        mass_str=args.mass_str,
        bdt_version=args.config_name,
        load_bdt_scores=False,
    )

    if args.apply_cuts:
        raw = apply_cuts(raw, args.txbb_str, args.mass_str)

    hlt_cols = list(HLTs[year])
    for key in list(raw.keys()):
        n_before = len(raw[key])
        trigger_mask = raw[key][[c for c in raw[key].columns if c[0] in hlt_cols]].any(axis=1)
        raw[key] = raw[key][trigger_mask].copy()
        logger.info(f"[{year}] trigger: {key}: {len(raw[key])} / {n_before} pass")

    for key in list(raw.keys()):
        if len(raw[key]) == 0:
            logger.warning(f"[{year}] {key}: 0 events after selection, skipping BDT")
            continue
        raw[key] = run_bdt_inference(
            raw[key],
            model,
            bdt_dataframe_fn,
            weight_ttbar_bdt=args.weight_ttbar_bdt,
            bdt_disc=args.bdt_disc,
        )

        cpath = cache_path_for(args, year, key)
        cpath.parent.mkdir(parents=True, exist_ok=True)
        with cpath.open("wb") as f:
            pickle.dump(raw[key], f)
        logger.info(f"[{year}] cached {key} to {cpath}")

    events.update(raw)
    return events


def _hist_for(events_dict, masks_dict, key, region, var, bins) -> np.ndarray:
    """Weighted 1D histogram (or unweighted for 'data') — empty array if no events."""
    if key not in events_dict:
        return np.zeros(len(bins) - 1)
    ev = events_dict[key]
    mask = masks_dict[key][region]
    if mask.sum() == 0:
        return np.zeros(len(bins) - 1)
    x = ev.loc[mask, var].to_numpy().reshape(-1)
    if key == "data":
        h, _ = np.histogram(x, bins=bins)
    else:
        w = ev.loc[mask, "finalWeight"].to_numpy()
        h, _ = np.histogram(x, bins=bins, weights=w)
    return h


def _abcd_sr_qcd_template(events_dict, masks_dict, sample_groups, var, bins) -> np.ndarray | None:
    """
    Data-driven QCD template in the SR: take the ``data - (resonant + signal MC)``
    shape from the MassSB region (same TXbb as SR), then normalize so its
    integral equals the ABCD-predicted SR yield
    ``sum(residual_MassSB) * sum(residual_TXbbSB) / sum(residual_BothSB)``.

    Returns None if any of the control-region residuals sum to <= 0.
    """
    if "data" not in events_dict:
        return None
    nonqcd = sample_groups["resonant"] + sample_groups["signal"]

    def residual(region):
        h = _hist_for(events_dict, masks_dict, "data", region, var, bins).astype(float)
        for k in nonqcd:
            h -= _hist_for(events_dict, masks_dict, k, region, var, bins)
        return h

    shape = residual("MassSB")
    n_mass = shape.sum()
    n_txbb = residual("TXbbSB").sum()
    n_both = residual("BothSB").sum()
    if n_mass <= 0 or n_txbb <= 0 or n_both <= 0:
        return None
    pred = n_mass * n_txbb / n_both
    return shape * (pred / n_mass)


def plot_region_stack(
    events_dict: dict[str, pd.DataFrame],
    masks_dict: dict[str, dict[str, np.ndarray]],
    var: tuple,
    var_label: str,
    bins: np.ndarray,
    out_path: Path,
    txbb_bins: tuple[float, float, float],
    mass_bins: tuple[float, float, float],
    sample_groups: dict[str, list[str]],
    apply_abcd: bool,
    qcd_only: bool = False,
    signal_scale: float = 1.0,
    save_formats: tuple[str, ...] = ("pdf",),
):
    """
    2x2 panel plot over the four ABCD regions (TXbb up, mass right).

    apply_abcd=False (before): QCD comes from MC in every region.
    apply_abcd=True  (after):  QCD in the SR panel is the ABCD-predicted shape
        (``data - non-QCD MC`` from MassSB, normalized to the ABCD yield).
        The three CR panels are identical to "before" since they are the
        prediction's inputs.

    qcd_only=False: stack = QCD + all resonant MC; signals overlaid as step
        lines (scaled by ``signal_scale``); data overlaid as points.
    qcd_only=True:  stack = QCD only; ``data - non-QCD MC`` overlaid as
        points (this is the "implied QCD" from data and should line up with
        the ABCD-predicted QCD in the SR panel if the method works).
    """
    layout = [["MassSB", "SR"], ["BothSB", "TXbbSB"]]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
    centers = 0.5 * (bins[:-1] + bins[1:])

    qcd_keys = sample_groups["qcd"]
    resonant_keys = sample_groups["resonant"]
    signal_keys = sample_groups["signal"]
    has_data = bool(sample_groups["data"])
    nonqcd_mc = resonant_keys + signal_keys

    abcd_qcd_sr = None
    if apply_abcd:
        abcd_qcd_sr = _abcd_sr_qcd_template(events_dict, masks_dict, sample_groups, var, bins)
        if abcd_qcd_sr is None:
            logger.warning(
                "ABCD SR template could not be built (residual <= 0); "
                "falling back to QCD MC for this plot."
            )

    for r in range(2):
        for c in range(2):
            ax = axes[r, c]
            region = layout[r][c]

            stack_hists, stack_labels, stack_colors = [], [], []

            # QCD goes first so it sits on the bottom of the stack.
            if region == "SR" and apply_abcd and abcd_qcd_sr is not None:
                stack_hists.append(abcd_qcd_sr)
                stack_labels.append("QCD (ABCD pred.)")
                stack_colors.append(SAMPLE_COLORS.get("qcd", "#ffd166"))
            else:
                for k in qcd_keys:
                    h = _hist_for(events_dict, masks_dict, k, region, var, bins)
                    if h.sum() > 0:
                        stack_hists.append(h)
                        stack_labels.append(k)
                        stack_colors.append(SAMPLE_COLORS.get(k, "#cccccc"))

            if not qcd_only:
                for k in resonant_keys:
                    h = _hist_for(events_dict, masks_dict, k, region, var, bins)
                    if h.sum() > 0:
                        stack_hists.append(h)
                        stack_labels.append(k)
                        stack_colors.append(SAMPLE_COLORS.get(k, "#cccccc"))

            if stack_hists:
                hep.histplot(
                    stack_hists,
                    bins=bins,
                    stack=True,
                    histtype="fill",
                    label=stack_labels,
                    color=stack_colors,
                    ax=ax,
                )

            if not qcd_only:
                for k in signal_keys:
                    h = _hist_for(events_dict, masks_dict, k, region, var, bins)
                    if h.sum() > 0:
                        label = k if signal_scale == 1.0 else f"{k} x{signal_scale:g}"
                        hep.histplot(
                            h * signal_scale,
                            bins=bins,
                            histtype="step",
                            label=label,
                            color=SAMPLE_COLORS.get(k, "#000000"),
                            ax=ax,
                            linewidth=2,
                        )

            if has_data:
                h_data = _hist_for(events_dict, masks_dict, "data", region, var, bins)
                if qcd_only:
                    # data - sum(non-QCD MC) = implied QCD from data
                    h_imp = h_data.astype(float)
                    for k in nonqcd_mc:
                        h_imp -= _hist_for(events_dict, masks_dict, k, region, var, bins)
                    if np.any(h_data > 0):
                        err = np.sqrt(h_data)  # stat from data only
                        ax.errorbar(
                            centers,
                            h_imp,
                            yerr=err,
                            fmt="ko",
                            markersize=4,
                            label="data $-$ non-QCD MC",
                            capsize=0,
                        )
                else:
                    if h_data.sum() > 0:
                        err = np.sqrt(h_data)
                        ax.errorbar(
                            centers,
                            h_data,
                            yerr=err,
                            fmt="ko",
                            markersize=4,
                            label="data",
                            capsize=0,
                        )

            ax.set_ylabel("Events")
            ax.set_yscale("log")
            ax.legend(loc="best", fontsize=10)

    for ax in axes[-1, :]:
        ax.set_xlabel(var_label)

    fig.tight_layout(rect=(0.06, 0.05, 1.0, 0.97))

    label_fontsize = plt.rcParams["axes.labelsize"]

    mode = "after" if apply_abcd else "before"
    scope = "QCD-only" if qcd_only else "all samples"
    title = f"ABCD {mode} — {scope} " f"(QCD {'from ABCD pred. in SR' if apply_abcd else 'MC'})"
    fig.suptitle(title, fontsize=label_fontsize)

    # Row labels on the left (TXbb ranges, increasing upward)
    txbb_ranges = [
        f"TXbb $\\in$ [{txbb_bins[1]:g}, {txbb_bins[2]:g})",
        f"TXbb $\\in$ [{txbb_bins[0]:g}, {txbb_bins[1]:g})",
    ]
    for r, txt in enumerate(txbb_ranges):
        bbox = axes[r, 0].get_position()
        y = (bbox.y0 + bbox.y1) / 2
        fig.text(0.015, y, txt, rotation=90, va="center", ha="center", fontsize=label_fontsize)

    # Column labels at the bottom (mass ranges, increasing rightward)
    mass_ranges = [
        f"mass $\\in$ [{mass_bins[0]:g}, {mass_bins[1]:g}) GeV",
        f"mass $\\in$ [{mass_bins[1]:g}, {mass_bins[2]:g}) GeV",
    ]
    for c, txt in enumerate(mass_ranges):
        bbox = axes[-1, c].get_position()
        x = (bbox.x0 + bbox.x1) / 2
        fig.text(x, 0.01, txt, va="bottom", ha="center", fontsize=label_fontsize)

    for fmt in save_formats:
        path = out_path.parent / f"{out_path.name}.{fmt.lstrip('.')}"
        fig.savefig(path, bbox_inches="tight")
        logger.info(f"Saved plot {path}")
    plt.close(fig)


def write_summary(
    yields: pd.DataFrame,
    out_path: Path,
    txbb_bins: tuple[float, float, float],
    mass_bins: tuple[float, float, float],
    sample_groups: dict[str, list[str]],
):
    """Write human-readable ABCD summary text file.

    Data-driven ABCD subtracts all non-QCD, non-data MC (resonant BGs + signal)
    from data in each control region; the result is the implied QCD yield.
    """

    def get(sample, region, col="yield"):
        row = yields[(yields["sample"] == sample) & (yields["region"] == region)]
        if len(row) == 0:
            return 0.0
        return float(row[col].iloc[0])

    samples = sorted(yields["sample"].unique())
    nonqcd_mc = sample_groups["resonant"] + sample_groups["signal"]

    lines = []
    lines.append("ABCD study summary")
    lines.append("==================")
    lines.append(
        f"TXbb bins (low, split, high): {txbb_bins};  " f"mass bins (low, split, high): {mass_bins}"
    )
    lines.append(f"Resonant MC (subtracted from data in CRs): {sample_groups['resonant']}")
    lines.append(f"Signal MC (subtracted from data in CRs): {sample_groups['signal']}")
    lines.append("")
    lines.append("Per-region yields (weighted, stat-err):")
    lines.append("  " + "sample".ljust(22) + "  ".join(r.ljust(22) for r in REGION_NAMES))
    for sample in samples:
        row = [f"  {sample.ljust(22)}"]
        for region in REGION_NAMES:
            y = get(sample, region, "yield")
            e = get(sample, region, "stat_err")
            row.append(f"{y:10.2f} +/- {e:8.2f}")
        lines.append("  ".join(row))

    lines.append("")

    # MC-only closure using QCD MC
    if "qcd" in samples:
        mass_y = get("qcd", "MassSB")
        mass_e = get("qcd", "MassSB", "stat_err")
        txbb_y = get("qcd", "TXbbSB")
        txbb_e = get("qcd", "TXbbSB", "stat_err")
        both_y = get("qcd", "BothSB")
        both_e = get("qcd", "BothSB", "stat_err")
        sr_obs = get("qcd", "SR")
        sr_obs_e = get("qcd", "SR", "stat_err")
        pred, pred_e = abcd_predict(mass_y, txbb_y, both_y, mass_e, txbb_e, both_e)
        ratio = pred / sr_obs if sr_obs > 0 else float("nan")
        lines.append("MC-only closure (QCD MC):")
        lines.append(f"  N_SR observed    = {sr_obs:10.2f} +/- {sr_obs_e:6.2f}")
        lines.append(f"  N_SR predicted   = {pred:10.2f} +/- {pred_e:6.2f}")
        lines.append(f"  pred / obs       = {ratio:8.4f}")
        lines.append("")

    # Data-driven: (data - non-QCD MC) in all four regions, predict SR QCD
    if "data" in samples:

        def cr_minus_nonqcd(region):
            y_d = get("data", region)
            e_d = get("data", region, "stat_err")
            y_sub = 0.0
            e2_sub = 0.0
            for k in nonqcd_mc:
                if k in samples:
                    y_sub += get(k, region)
                    e2_sub += get(k, region, "stat_err") ** 2
            return y_d - y_sub, np.sqrt(e_d**2 + e2_sub)

        mass_y, mass_e = cr_minus_nonqcd("MassSB")
        txbb_y, txbb_e = cr_minus_nonqcd("TXbbSB")
        both_y, both_e = cr_minus_nonqcd("BothSB")
        pred, pred_e = abcd_predict(mass_y, txbb_y, both_y, mass_e, txbb_e, both_e)

        sr_minus, sr_minus_e = cr_minus_nonqcd("SR")
        sr_qcd_mc = get("qcd", "SR") if "qcd" in samples else float("nan")

        lines.append("Data-driven ABCD (data - resonant - signal MC in CRs, implied QCD):")
        lines.append(f"  N_MassSB (data-nonQCD) = {mass_y:10.2f} +/- {mass_e:6.2f}")
        lines.append(f"  N_TXbbSB (data-nonQCD) = {txbb_y:10.2f} +/- {txbb_e:6.2f}")
        lines.append(f"  N_BothSB (data-nonQCD) = {both_y:10.2f} +/- {both_e:6.2f}")
        lines.append(f"  N_SR    predicted      = {pred:10.2f} +/- {pred_e:6.2f}")
        lines.append(
            f"  N_SR (data-nonQCD) obs = {sr_minus:10.2f} +/- {sr_minus_e:6.2f}  "
            f"(blinded if SR is masked in data)"
        )
        lines.append(f"  N_SR    QCD MC         = {sr_qcd_mc:10.2f}")
        if sr_qcd_mc > 0:
            lines.append(f"  pred / QCD MC          = {pred / sr_qcd_mc:8.4f}")
        if sr_minus > 0:
            lines.append(f"  pred / (data-nonQCD)   = {pred / sr_minus:8.4f}")

    out_path.write_text("\n".join(lines) + "\n")
    logger.info(f"Wrote summary to {out_path}")


DEFAULT_PLOT_BINS = {
    "bdt_score": np.linspace(0, 1, 41),
    "bdt_score_vbf": np.linspace(0, 1, 41),
    "bbFatJetPt": np.linspace(250, 1500, 41),
    "bbFatJetParT3TXbb": np.linspace(0, 1, 41),
    "bbFatJetParT3massX2p": np.linspace(40, 250, 43),
    "bbFatJetMsd": np.linspace(0, 300, 31),
}


def resolve_plot_var(
    var_spec: str, events_dict: dict[str, pd.DataFrame]
) -> tuple[tuple, str, np.ndarray]:
    """
    Accepts either 'name' (index 0 assumed) or 'name:index' or 'name:index:lo,hi,nbins'.
    Returns (flat tuple key, axis label, bins array).
    """
    parts = var_spec.split(":")
    name = parts[0]
    idx = int(parts[1]) if len(parts) > 1 and parts[1] != "" else 0
    key = (name, idx)

    # any sample that has the key determines the bins
    bins = None
    if len(parts) > 2:
        lo, hi, nb = parts[2].split(",")
        bins = np.linspace(float(lo), float(hi), int(nb) + 1)
    else:
        bins = DEFAULT_PLOT_BINS.get(name)
        if bins is None:
            # infer range from any sample
            for ev in events_dict.values():
                if key in ev.columns and len(ev) > 0:
                    x = ev[key].to_numpy().reshape(-1)
                    lo, hi = np.nanquantile(x, [0.01, 0.99])
                    bins = np.linspace(lo, hi, 41)
                    break

    label = name if idx == 0 else f"{name}[{idx}]"
    return key, label, bins


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(args.model_dir) / args.model_name
    logger.info(f"Loading BDT model from {model_dir}")
    model = load_bdt_model(model_dir)
    bdt_dataframe_fn = importlib.import_module(
        f"HH4b.boosted.bdt_trainings_run3.{args.config_name}"
    ).bdt_dataframe

    if args.year == ["2022-2023"]:
        years = ["2022", "2022EE", "2023", "2023BPix"]
    else:
        years = args.year

    # restrict samples_run3 dict to requested samples & years
    all_years = list(samples_run3.keys())
    for year in all_years:
        for key in list(samples_run3[year].keys()):
            if year not in years or key not in args.samples:
                samples_run3[year].pop(key)

    logger.info(f"Samples to load: {samples_run3}")

    # accumulate per-sample events across years
    events_dict: dict[str, pd.DataFrame] = {}
    for year in years:
        per_year = load_and_prepare(args, samples_run3[year], year, model, bdt_dataframe_fn)
        for key, df in per_year.items():
            if len(df) == 0:
                continue
            if key in events_dict:
                events_dict[key] = pd.concat([events_dict[key], df], axis=0, ignore_index=True)
            else:
                events_dict[key] = df

    txbb_bins = tuple(args.txbb_bins)
    mass_bins = tuple(args.mass_bins)

    # region masks + region column
    masks_dict = {}
    for key, df in list(events_dict.items()):
        masks_dict[key] = build_region_masks(df, args.txbb_str, args.mass_str, txbb_bins, mass_bins)
        events_dict[key] = assign_region_column(df, masks_dict[key])

    # save per-sample events (all columns preserved via pickle)
    events_out = out_dir / "events.pkl"
    with events_out.open("wb") as f:
        pickle.dump(events_dict, f)
    logger.info(f"Saved events dict (all columns) to {events_out}")

    # compute yields
    yields = compute_region_yields(events_dict, masks_dict)
    yields_path = out_dir / "yields.csv"
    yields.to_csv(yields_path, index=False)
    logger.info(f"Saved yields to {yields_path}")
    logger.info(f"\n{yields.to_string(index=False)}")

    sample_groups = classify_samples(events_dict.keys())
    logger.info(f"Sample groups: {sample_groups}")

    # textual summary
    summary_path = out_dir / "abcd_summary.txt"
    write_summary(yields, summary_path, txbb_bins, mass_bins, sample_groups)
    print(summary_path.read_text())

    if args.plot:
        plots_dir = out_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        modes = [
            ("before_all", False, False),
            ("after_all", True, False),
            ("before_qcd_only", False, True),
            ("after_qcd_only", True, True),
        ]
        for var_spec in args.plot_vars:
            key, label, bins = resolve_plot_var(var_spec, events_dict)
            if bins is None:
                logger.warning(f"Could not resolve bins for {var_spec}, skipping")
                continue
            safe = var_spec.replace(":", "_").replace(",", "_")
            for suffix, apply_abcd, qcd_only in modes:
                plot_region_stack(
                    events_dict,
                    masks_dict,
                    key,
                    label,
                    bins,
                    plots_dir / f"{safe}_{suffix}",
                    txbb_bins,
                    mass_bins,
                    sample_groups,
                    apply_abcd=apply_abcd,
                    qcd_only=qcd_only,
                    signal_scale=args.signal_scale,
                    save_formats=tuple(args.save_formats),
                )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        required=True,
        help="path to skimmer output (e.g. .../nanov15_20251202_v15_signal)",
    )
    parser.add_argument(
        "--model-dir",
        default=str(Path(__file__).parent / "bdt_trainings_run3"),
        help="directory containing BDT training outputs",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="BDT training directory name under --model-dir",
    )
    parser.add_argument(
        "--config-name",
        default=None,
        help="BDT config module name under HH4b.boosted.bdt_trainings_run3 "
        "(defaults to --model-name)",
    )
    parser.add_argument(
        "--year",
        nargs="+",
        type=str,
        default=["2022"],
        choices=hh_vars.years + ["2022-2023"],
    )
    parser.add_argument(
        "--samples",
        nargs="+",
        default=["data", "qcd", "ttbar"],
        help="top-level sample keys to load",
    )
    parser.add_argument(
        "--txbb",
        choices=["pnet-v12", "pnet-legacy", "glopart-v2", "glopart-v3"],
        required=True,
    )
    parser.add_argument(
        "--mass",
        choices=[
            "bbFatJetPNetMass",
            "bbFatJetPNetMassLegacy",
            "bbFatJetMsd",
            "bbFatJetParTmassVis",
            "bbFatJetParT3massX2p",
            "bbFatJetParT3massGeneric",
        ],
        required=True,
    )
    parser.add_argument(
        "--txbb-bins",
        type=float,
        nargs=3,
        default=[0.3, 0.8, 1.0],
        metavar=("LOW", "SPLIT", "HIGH"),
    )
    parser.add_argument(
        "--mass-bins",
        type=float,
        nargs=3,
        default=[50.0, 100.0, 140.0],
        metavar=("LOW", "SPLIT", "HIGH"),
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="output directory (default: <model-dir>/<model-name>/abcd_study)",
    )
    parser.add_argument(
        "--plot-vars",
        nargs="+",
        default=[
            "bdt_score",
            "bbFatJetParT3TXbb:1",
            "bbFatJetParT3massX2p:1",
        ],
        help="variables to plot per region. Format: 'name' or 'name:idx' or "
        "'name:idx:lo,hi,nbins'. Index defaults to 0.",
    )
    parser.add_argument(
        "--signal-scale",
        type=float,
        default=1.0,
        help="multiplicative factor for signal step-line overlay in plots",
    )
    parser.add_argument(
        "--save-formats",
        nargs="+",
        default=["pdf"],
        help="image formats to save each plot in (e.g. pdf png svg)",
    )
    parser.add_argument("--weight-ttbar-bdt", type=float, default=1.0)
    parser.add_argument(
        "--bdt-inference-dir",
        default="/ceph/cms/store/user/zichun/bbbb/signal_processed/bdt_inference",
        help="base directory for cached post-inference events "
        "(per (year, sample) pickle under <dir>/<model-name>/<year>/<sample>.pkl)",
    )

    add_bool_arg(parser, "apply-cuts", "Apply preselection cuts", default=True)
    add_bool_arg(parser, "plot", "Produce per-region plots", default=False)
    add_bool_arg(parser, "bdt-disc", "Use P_sig/(P_sig+P_bkg) BDT discriminant", default=True)
    add_bool_arg(
        parser, "force-rerun", "Ignore cached inference and re-run from scratch", default=False
    )

    args = parser.parse_args()
    args.txbb_str = txbb_strings[args.txbb]
    args.mass_str = args.mass
    if args.config_name is None:
        args.config_name = args.model_name
    if args.out_dir is None:
        args.out_dir = str(Path(args.model_dir) / args.model_name / "abcd_study")
    return args


if __name__ == "__main__":
    main(parse_args())
