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
from matplotlib import ticker as mticker
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
    target_bkg_effs: list[float] = roc_group.get("bkg_effs", [1e-1, 1e-2, 1e-3])

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
        f"roc_group '{roc_group['name']}': "
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

    # Normalize weights within signal and background separately so each class
    # group contributes equal total weight regardless of absolute cross-section.
    # Without this, a rare signal like vbfhh4b has near-zero total weight vs QCD
    # and sklearn declares "no positive samples".
    if weights_np is not None:
        w_norm = weights_np.copy()
        w_sig_sum = w_norm[is_sig].sum()
        w_bkg_sum = w_norm[~is_sig].sum()
        if w_sig_sum > 0:
            w_norm[is_sig] /= w_sig_sum
        if w_bkg_sum > 0:
            w_norm[~is_sig] /= w_bkg_sum
        weights_np = w_norm

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


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _format_exp(val: float) -> str:
    """Format e.g. 1e-2 as '10^{-2}' for matplotlib LaTeX."""
    exp = round(np.log10(val))
    return rf"10^{{{exp}}}"


def _make_roc_figure(
    roc_results: dict[str, dict[str, Any]],
    epoch: int,
) -> plt.Figure:
    """Build and return the matplotlib Figure."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for i, (name, res) in enumerate(roc_results.items()):
        color = _COLORS[i % len(_COLORS)]

        # ROC curve: x = signal eff (tpr), y = background eff (fpr), log-y
        ax.plot(
            res["tpr"],
            res["fpr"],
            lw=1.8,
            color=color,
            label=rf"{name}  AUC $=$ {res['auc']:.4f}",
            zorder=3,
        )

        # Dashed operating-point lines + markers
        for j, (eps_b, eps_s) in enumerate(res["sig_effs"]):
            if not eps_s or np.isnan(eps_s) or eps_s <= 0:
                continue
            dash = _DASHES[j % len(_DASHES)]
            op_label = rf"$\varepsilon_s = {eps_s:.3f},\;\varepsilon_b = {_format_exp(eps_b)}$"
            # Invisible line just to register the legend entry
            ax.plot([], [], color=color, lw=1.2, ls=(0, dash), label=op_label)
            # Actual dashed cross-hairs
            ax.axvline(eps_s, color=color, lw=0.9, ls=(0, dash), alpha=0.65, zorder=2)
            ax.axhline(eps_b, color=color, lw=0.9, ls=(0, dash), alpha=0.65, zorder=2)
            ax.plot(eps_s, eps_b, "o", color=color, ms=5, zorder=4)

    ax.set_xlabel(r"Signal efficiency $\varepsilon_s$", fontsize=13)
    ax.set_ylabel(r"Background efficiency $\varepsilon_b$", fontsize=13)
    ax.set_yscale("log")
    ax.set_xlim(0.0, 1.0)

    all_bkg_effs = [
        eps_b for res in roc_results.values() for eps_b, _ in res["sig_effs"] if eps_b > 0
    ]
    ax.set_ylim(min(all_bkg_effs) * 0.3 if all_bkg_effs else 1e-4, 1.5)

    ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext())
    ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.4)
    ax.set_title(rf"ROC Curves -- Epoch {epoch:04d}", fontsize=13, pad=10)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.85, edgecolor="0.7")

    fig.tight_layout()
    return fig


def save_roc_plot(
    roc_results: dict[str, dict[str, Any]],
    output_dir: Path,
    epoch: int,
) -> None:
    """Save ROC curves as PNG and PDF under ``output_dir/roc_epoch_XXXX.{png,pdf}``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = _make_roc_figure(roc_results, epoch)
    stem = output_dir / f"roc_epoch_{epoch:04d}"
    # fig.savefig(f"{stem}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


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
        return logits, labels, weights

    def gather(self, accelerator: Accelerator) -> ScoreAccumulator:
        logits, labels, weights = self.finalize()
        logits = accelerator.gather(logits)
        labels = accelerator.gather(labels)
        if weights is not None:
            weights = accelerator.gather(weights)
        new = ScoreAccumulator()
        new.update(logits, labels, weights)
        return new
