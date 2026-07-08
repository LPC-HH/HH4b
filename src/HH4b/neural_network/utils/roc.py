"""roc_utils.py — ROC curve computation and plotting for JetClassifier."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import torch
from accelerate import Accelerator
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve

_log = logging.getLogger(__name__)

mpl.use("Agg")

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf"]
_DASHES = [(6, 2), (3, 2), (1, 2)]  # long-dash, medium-dash, dotted


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_roc_metrics(
    all_logits: torch.Tensor,  # (N, num_classes)
    all_labels: torch.Tensor,  # (N,)
    all_weights: torch.Tensor | None,  # (N,) or None
    roc_group: dict[str, Any],
    class_to_idx: dict[str, int],
) -> dict[str, Any]:
    """Compute ROC AUC and signal efficiencies at given background efficiencies.

    Args:
        all_logits:   Raw model logits for the full split.
        all_labels:   Integer class labels.
        all_weights:  Per-sample weights (or None).
        roc_group:    One entry from config['training']['roc_groups'].
        class_to_idx: Mapping from class name -> integer index.

    Returns:
        Dict with keys:
            'auc'        : float
            'sig_effs'   : list of (eps_b_target, eps_s_achieved) tuples
            'fpr'        : np.ndarray   (background efficiency)
            'tpr'        : np.ndarray   (signal efficiency)
            'thresholds' : np.ndarray
    """
    signal_classes = roc_group["signal"]
    background_classes = roc_group["background"]
    target_bkg_effs: list[float] = roc_group.get("bkg_effs", [1e-1, 1e-2, 1e-3, 1e-4])

    sig_indices = [class_to_idx[c] for c in signal_classes]
    bkg_indices = [class_to_idx[c] for c in background_classes]

    probs = torch.softmax(all_logits, dim=-1).cpu().numpy()  # (N, C)
    labels_np = all_labels.cpu().numpy()  # (N,)
    weights_np = all_weights.cpu().numpy() if all_weights is not None else None

    is_sig = np.isin(labels_np, sig_indices)
    is_bkg = np.isin(labels_np, bkg_indices)
    keep = is_sig | is_bkg

    probs = probs[keep]
    is_sig = is_sig[keep]
    weights_np = weights_np[keep] if weights_np is not None else None

    n_sig = int(is_sig.sum())
    n_bkg = int((~is_sig).sum())

    # Weighted counts (what sklearn actually sees)
    if weights_np is not None:
        w_sig = float(weights_np[is_sig].sum())
        w_bkg = float(weights_np[~is_sig].sum())
    else:
        w_sig, w_bkg = float(n_sig), float(n_bkg)

    _log.debug(
        f"Computing ROC metrics for roc_group '{roc_group['name']}': "
        f"n_sig={n_sig:,} (w={w_sig:.3g})  n_bkg={n_bkg:,} (w={w_bkg:.3g})"
    )

    if n_sig == 0 or n_bkg == 0 or w_sig <= 0 or w_bkg <= 0:
        warnings.warn(
            f"roc_group '{roc_group['name']}': "
            f"n_sig={n_sig:,} (w={w_sig:.3g}), n_bkg={n_bkg:,} (w={w_bkg:.3g}) — "
            f"ROC is undefined. Signal classes: {signal_classes}.",
            stacklevel=2,
        )
        nan = float("nan")
        return {
            "auc": nan,
            "sig_effs": [(eps_b, nan) for eps_b in sorted(target_bkg_effs, reverse=True)],
            "fpr": np.array([0.0, 1.0]),
            "tpr": np.array([0.0, 1.0]),
            "thresholds": np.array([1.0, 0.0]),
        }

    # Score = sum of signal class softmax probabilities
    score = probs[:, sig_indices].sum(axis=1)
    binary_label = is_sig.astype(np.float32)

    fpr, tpr, thresholds = roc_curve(binary_label, score, sample_weight=weights_np)

    auc = float(abs(np.trapezoid(tpr, fpr)))

    # List of (eps_b_target, eps_s_achieved), sorted descending by eps_b
    sig_effs: list[tuple[float, float]] = []
    for eps_b in sorted(target_bkg_effs, reverse=True):
        valid = fpr <= eps_b + 1e-12
        eps_s = float(tpr[valid][-1]) if valid.any() else 0.0
        sig_effs.append((eps_b, eps_s))

    return {
        "auc": auc,
        "sig_effs": sig_effs,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }


def compute_roc_per_signal(
    all_logits: torch.Tensor,
    all_labels: torch.Tensor,
    all_weights: torch.Tensor | None,
    sig_keys: list[str],
    bg_keys: list[str],
    class_to_idx: dict[str, int],
    bkg_effs: list[float] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Compute ROC curves for each signal vs each background + merged background.

    Like TrainBDT.py: one figure per signal, with curves for each individual
    background and a "merged" (all backgrounds) curve.

    Returns:
        {sig_key: {bkg_label: roc_result_dict, ...}, ...}
        where bkg_label is each bg_key plus "merged".
    """
    if bkg_effs is None:
        bkg_effs = [1e-1, 1e-2, 1e-3, 1e-4]

    all_results: dict[str, dict[str, dict[str, Any]]] = {}

    for sig_key in sig_keys:
        sig_results: dict[str, dict[str, Any]] = {}

        # Individual backgrounds
        for bkg_key in bg_keys:
            roc_group = {
                "name": f"{sig_key}_vs_{bkg_key}",
                "signal": [sig_key],
                "background": [bkg_key],
                "bkg_effs": bkg_effs,
            }
            sig_results[bkg_key] = compute_roc_metrics(
                all_logits, all_labels, all_weights, roc_group, class_to_idx
            )

        # Merged (all backgrounds)
        roc_group_merged = {
            "name": f"{sig_key}_vs_merged",
            "signal": [sig_key],
            "background": bg_keys,
            "bkg_effs": bkg_effs,
        }
        sig_results["merged"] = compute_roc_metrics(
            all_logits, all_labels, all_weights, roc_group_merged, class_to_idx
        )

        all_results[sig_key] = sig_results

    return all_results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _format_exp(val: float) -> str:
    """Format e.g. 1e-2 as '10^{-2}' for matplotlib LaTeX."""
    exp = round(np.log10(val))
    return rf"10^{{{exp}}}"


def _make_roc_figure(
    roc_results: dict[str, dict[str, Any]],
    epoch: int | str,
    title: str | None = None,
) -> plt.Figure:
    """Build and return the matplotlib Figure.

    Style follows TrainBDT.py: x = signal eff (tpr), y = background eff (fpr),
    log-y scale, x-axis spans [0, 1.0], operating-point markers with labels for
    signal efficiency at key background efficiencies.
    """
    fig, ax = plt.subplots(figsize=(18, 12))

    th_colours = ["#9381FF", "#1f78b4", "#a6cee3", "cyan", "blue", "#ff7f00"]

    for i, (name, res) in enumerate(roc_results.items()):
        color = _COLORS[i % len(_COLORS)]

        # ROC curve: x = signal eff (tpr), y = background eff (fpr), log-y
        ax.plot(
            res["tpr"],
            res["fpr"],
            lw=2,
            color=color,
            label=rf"{name}  AUC $=$ {res['auc']:.4f}",
            zorder=3,
        )

        # Operating-point markers only for the last curve (merged)
        is_last = i == len(roc_results) - 1
        if is_last:
            for j, (eps_b, eps_s) in enumerate(res["sig_effs"]):
                if not eps_s or np.isnan(eps_s) or eps_s <= 0:
                    continue
                th_color = th_colours[j % len(th_colours)]
                op_label = (
                    rf"$\varepsilon_s = {eps_s:.4f}$  @  "
                    rf"$\varepsilon_b = {_format_exp(eps_b)}$"
                )
                ax.scatter(
                    eps_s,
                    eps_b,
                    marker="o",
                    s=40,
                    color=th_color,
                    label=op_label,
                    zorder=100,
                )
                ax.vlines(
                    x=eps_s,
                    ymin=0,
                    ymax=eps_b,
                    color=th_color,
                    linestyles="dashed",
                    alpha=0.5,
                )
                ax.hlines(
                    y=eps_b,
                    xmin=0,
                    xmax=eps_s,
                    color=th_color,
                    linestyles="dashed",
                    alpha=0.5,
                )

    ax.set_xlabel("Signal efficiency", fontsize=13)
    ax.set_ylabel("Background efficiency", fontsize=13)
    ax.set_yscale("log")
    ax.set_xlim(0.0, 1.0)

    all_bkg_effs = [
        eps_b for res in roc_results.values() for eps_b, _ in res["sig_effs"] if eps_b > 0
    ]
    ax.set_ylim(min(all_bkg_effs) * 0.3 if all_bkg_effs else 1e-5, 1.5)

    ax.xaxis.grid(True, which="major")
    ax.yaxis.grid(True, which="major")
    if title is not None:
        ax.set_title(title, fontsize=13, pad=10)
    else:
        epoch_str = f"{epoch:04d}" if isinstance(epoch, int) else str(epoch)
        ax.set_title(f"ROC Curves -- Epoch {epoch_str}", fontsize=13, pad=10)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.85, edgecolor="0.7")

    fig.tight_layout()
    return fig


def save_roc_plot(
    roc_results: dict[str, dict[str, Any]],
    output_dir: Path,
    epoch: int | str,
    title: str | None = None,
    filename: str | None = None,
) -> None:
    """Save ROC curves as PNG and PDF under ``output_dir/``.

    Args:
        filename: If provided, use this as the stem (without extension).
                  Otherwise defaults to ``roc_epoch_XXXX``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = _make_roc_figure(roc_results, epoch, title=title)
    if filename is not None:
        stem = output_dir / filename
    else:
        epoch_str = f"{epoch:04d}" if isinstance(epoch, int) else str(epoch)
        stem = output_dir / f"roc_epoch_{epoch_str}"
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_roc_per_signal(
    all_results: dict[str, dict[str, dict[str, Any]]],
    output_dir: Path,
    epoch: int | str,
    split: str = "val",
) -> None:
    """Save one ROC figure per signal class.

    Each figure has curves for each individual background + merged.
    Follows TrainBDT.py convention.

    Args:
        all_results: Output from ``compute_roc_per_signal``.
        output_dir:  Base directory for saving plots.
        epoch:       Current epoch (used for filenames).
        split:       Split name (e.g. "val", "test").
    """
    epoch_str = f"{epoch:04d}" if isinstance(epoch, int) else str(epoch)

    for sig_key, bkg_results in all_results.items():
        sig_dir = output_dir / sig_key
        sig_dir.mkdir(parents=True, exist_ok=True)

        title = f"{sig_key} ROC -- {split} -- Epoch {epoch_str}"
        filename = f"roc_{split}_epoch_{epoch_str}"
        save_roc_plot(bkg_results, sig_dir, epoch, title=title, filename=filename)
        _log.info(f"  ROC plot saved → {sig_dir}/{filename}.[png|pdf]")


# ---------------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------------


class ScoreAccumulator:
    """Collects (logits, labels, weights) across batches for later ROC computation."""

    def __init__(self) -> None:
        self._logits: list[torch.Tensor] = []
        self._labels: list[torch.Tensor] = []
        self._weights: list[torch.Tensor] = []
        self._has_weights = False

    def update(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> None:
        self._logits.append(logits.detach().cpu())
        self._labels.append(labels.detach().cpu())
        if weights is not None:
            self._weights.append(weights.detach().cpu())
            self._has_weights = True

    def finalize(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Returns (all_logits, all_labels, all_weights_or_None)."""
        logits = torch.cat(self._logits, dim=0)
        labels = torch.cat(self._labels, dim=0)
        weights = torch.cat(self._weights, dim=0) if self._has_weights else None

        _log.debug(
            f"Finalized ScoreAccumulator: {len(logits):,} samples, "
            f"with labels {torch.unique(labels)} "
            f"{'with' if weights is not None else 'without'} weights."
        )

        return logits, labels, weights

    def gather(self, accelerator: Accelerator) -> ScoreAccumulator:
        logits, labels, weights = self.finalize()

        logits = accelerator.gather(logits.to(accelerator.device))
        labels = accelerator.gather(labels.to(accelerator.device))
        if weights is not None:
            weights = accelerator.gather(weights.to(accelerator.device))

        _log.debug(
            f"Gathered ScoreAccumulator across processes: {len(logits):,} samples, "
            f"with labels {torch.unique(labels)} "
            f"{'with' if weights is not None else 'without'} weights."
        )

        new = ScoreAccumulator()
        new.update(logits.cpu(), labels.cpu(), weights.cpu() if weights is not None else None)
        return new
