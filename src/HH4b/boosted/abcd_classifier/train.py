"""
Train the 6-way ABCDnn classifier.  See notes/ABCDnn.md Task 4.

Pipeline (after dataset prep from Task 2):

    processed_data.npz + feature_stats.json
                │
                ▼
        standardize X via saved μ/σ
                │
                ▼
       train/val DataLoader  (TensorDataset on CPU)
                │
                ▼
       ABCDClassifier MLP, weighted CE loss, Adam,
       ReduceLROnPlateau, early stopping
                │
                ▼
       best_model.pt + train_log.csv + train_loss.pdf
       + roc_per_class.pdf

Note: this file pulls in torch only inside ``train_classifier`` so the
``--prepare-only`` path runs in environments without torch (e.g. the
``hh4b`` micromamba env).  Training itself requires torch (use the
``hbb-tagger`` env on this cluster).
"""  # noqa: RUF002

from __future__ import annotations

import argparse
import json
import logging
import logging.config
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from HH4b import hh_vars
from HH4b.log_utils import log_config

from ._argparse_utils import add_bool_arg

log_config["root"]["level"] = "INFO"
logging.config.dictConfig(log_config)
logger = logging.getLogger("ABCDnn.train")

plt.style.use(hep.style.CMS)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the 6-way (region × sample) ABCDnn classifier.",  # noqa: RUF001
    )

    # Inputs
    parser.add_argument(
        "--bdt-inference-dir",
        default="/ceph/cms/store/user/zichun/bbbb/signal_processed/bdt_inference",
        help="base directory containing the cached post-inference pickles "
        "written by bdt_ABCD_study.py "
        "(<dir>/<model-name>/<year>/<sample>.pkl).",
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help="BDT training directory name; used as the cache subdirectory.",
    )
    parser.add_argument(
        "--year",
        nargs="+",
        type=str,
        default=["2022"],
        choices=hh_vars.years + ["2022-2023", "2022-2023-2024"],
    )

    # Region definition (must match bdt_ABCD_study)
    parser.add_argument(
        "--txbb",
        choices=["pnet-v12", "pnet-legacy", "glopart-v2", "glopart-v3"],
        default="glopart-v3",
        help="tagger version; selects the TXbb column for region splits.",
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
        default="bbFatJetParT3massX2p",
        help="mass column for region splits.",
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
        default=[50.0, 100.0, 150.0],
        metavar=("LOW", "SPLIT", "HIGH"),
    )
    parser.add_argument(
        "--txbb-jet-index",
        type=int,
        default=0,
        choices=[0, 1],
        help="FatJet index for the ABCD TXbb axis.  v1: 0 (leading). " "v2: 1 (subleading).",
    )
    parser.add_argument(
        "--mass-jet-index",
        type=int,
        default=0,
        choices=[0, 1],
        help="FatJet index for the ABCD mass axis.  Kept at 0 across tags.",
    )

    # Feature set
    parser.add_argument(
        "--feature-set",
        choices=["strict", "literal", "strict_era", "literal_era"],
        default="strict",
        help="strict (13, no jet-0 mass/TXbb leakage) or literal (also 13); "
        "append '_era' to add one-hot era columns (year-aware classifier, "
        "only useful for multi-era runs).",
    )

    # Run identification
    parser.add_argument(
        "--run-name",
        required=True,
        help="subdirectory name under --out-dir for this run.",
    )
    parser.add_argument(
        "--out-dir",
        default="/ceph/cms/store/user/zichun/bbbb/signal_processed/abcd_classifier",
        help="parent directory for run outputs.",
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader worker count (0 = synchronous).",
    )

    # Model architecture
    parser.add_argument(
        "--hidden",
        type=int,
        default=512,
        help="hidden layer width.",
    )
    parser.add_argument(
        "--num-hidden-layers",
        type=int,
        default=5,
        help="number of (Linear → BN → ReLU → Dropout) hidden blocks.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="dropout probability after each ReLU.",
    )

    # Misc
    add_bool_arg(
        parser, "force-rebuild", "Rebuild processed_data.npz even if cached", default=False
    )
    add_bool_arg(
        parser,
        "prepare-only",
        "Build the dataset (Task 2) and exit, skipping training",
        default=False,
    )

    return parser.parse_args()


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------


def _set_seeds(seed: int) -> None:
    import torch  # noqa: PLC0415

    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _epoch_pass(
    model,
    loader,
    optimizer,
    device,
    train: bool,
):
    """Run one epoch over ``loader``.  Returns (mean weighted loss, accuracy)."""
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    model.train(train)
    loss_sum = 0.0
    n_seen = 0
    n_correct = 0

    for X, y, w in loader:
        X = X.to(device, non_blocking=True)  # noqa: PLW2901
        y = y.to(device, non_blocking=True)  # noqa: PLW2901
        w = w.to(device, non_blocking=True)  # noqa: PLW2901

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = model(X)
            loss_per = F.cross_entropy(logits, y, reduction="none")
            loss = (loss_per * w).mean()

        if train:
            loss.backward()
            optimizer.step()

        bs = X.size(0)
        loss_sum += float(loss.item()) * bs
        n_seen += bs
        n_correct += int((logits.argmax(dim=-1) == y).sum().item())

    return loss_sum / max(n_seen, 1), n_correct / max(n_seen, 1)


def _evaluate_predictions(model, loader, device):
    """Return concatenated (probs, labels, weights) over the loader."""
    import torch  # noqa: PLC0415

    model.eval()
    probs_chunks, y_chunks, w_chunks = [], [], []
    with torch.no_grad():
        for X, y, w in loader:
            X = X.to(device, non_blocking=True)  # noqa: PLW2901
            logits = model(X)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            probs_chunks.append(probs)
            y_chunks.append(y.numpy())
            w_chunks.append(w.numpy())
    return (
        np.concatenate(probs_chunks, axis=0),
        np.concatenate(y_chunks, axis=0),
        np.concatenate(w_chunks, axis=0),
    )


def _plot_train_loss(log_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax0, ax1 = axes

    ax0.plot(log_df["epoch"], log_df["train_loss"], label="train", color="#118ab2")
    ax0.plot(log_df["epoch"], log_df["val_loss"], label="val", color="#ef476f")
    ax0.set_xlabel("epoch")
    ax0.set_ylabel("weighted CE loss")
    ax0.legend()
    ax0.grid(True, which="major", alpha=0.3)

    ax1.plot(log_df["epoch"], log_df["train_acc"], label="train", color="#118ab2")
    ax1.plot(log_df["epoch"], log_df["val_acc"], label="val", color="#ef476f")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("accuracy (unweighted, top-1)")
    ax1.legend()
    ax1.grid(True, which="major", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"saved {out_path}")


def _plot_roc_per_class(
    probs: np.ndarray,
    y: np.ndarray,
    label_names: list[str],
    out_path: Path,
) -> dict[str, float]:
    """One-vs-rest ROCs for all 6 classes + the three (R, data) vs
    (R, ttbar) pairs that the per-event TF formula relies on.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    ax_ovr, ax_pair = axes
    aucs: dict[str, float] = {}

    # One-vs-rest, six curves
    for c, name in enumerate(label_names):
        y_bin = (y == c).astype(np.int64)
        if y_bin.sum() == 0 or y_bin.sum() == len(y):
            continue
        fpr, tpr, _ = roc_curve(y_bin, probs[:, c])
        auc = roc_auc_score(y_bin, probs[:, c])
        aucs[f"OvR_{name}"] = float(auc)
        ax_ovr.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    ax_ovr.plot([0, 1], [0, 1], color="#888888", linestyle="--", linewidth=1)
    ax_ovr.set_xlabel("false positive rate")
    ax_ovr.set_ylabel("true positive rate")
    ax_ovr.set_title("One-vs-rest ROC, all 6 classes")
    ax_ovr.legend(loc="lower right", fontsize=11)
    ax_ovr.grid(True, alpha=0.3)

    # Pairs: data vs ttbar within each region.  Restrict to events in that
    # region, score = P(data, R | x) / (P(data, R | x) + P(ttbar, R | x)).
    pair_specs = [
        ("B", 0, 1),
        ("C", 2, 3),
        ("D", 4, 5),
    ]
    for region, c_data, c_tt in pair_specs:
        mask = (y == c_data) | (y == c_tt)
        if mask.sum() == 0:
            continue
        sub_probs = probs[mask][:, [c_data, c_tt]]
        norm = sub_probs.sum(axis=1, keepdims=True)
        norm = np.where(norm == 0, 1.0, norm)
        score = sub_probs[:, 0] / norm[:, 0]
        y_bin = (y[mask] == c_data).astype(np.int64)
        fpr, tpr, _ = roc_curve(y_bin, score)
        auc = roc_auc_score(y_bin, score)
        aucs[f"pair_{region}_data_vs_ttbar"] = float(auc)
        ax_pair.plot(fpr, tpr, label=f"{region}: data vs ttbar  (AUC={auc:.3f})")

    ax_pair.plot([0, 1], [0, 1], color="#888888", linestyle="--", linewidth=1)
    ax_pair.set_xlabel("false positive rate")
    ax_pair.set_ylabel("true positive rate")
    ax_pair.set_title("Within-region data vs ttbar separation")
    ax_pair.legend(loc="lower right", fontsize=11)
    ax_pair.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info(f"saved {out_path}")
    return aucs


def train_classifier(run_dir: Path, args) -> None:
    """Task 4 — train the MLP, log per-epoch metrics, save best model + plots."""
    import torch  # noqa: PLC0415
    from torch.optim.lr_scheduler import ReduceLROnPlateau  # noqa: PLC0415
    from torch.utils.data import DataLoader, TensorDataset  # noqa: PLC0415

    from .model import ABCDClassifier  # noqa: PLC0415

    _set_seeds(args.seed)

    # Load processed data + standardization stats
    npz = np.load(run_dir / "processed_data.npz")
    stats = json.loads((run_dir / "feature_stats.json").read_text())

    X_all = npz["X"]
    y_all = npz["y"]
    w_all = npz["w"]

    mu = np.asarray(stats["mu"], dtype=np.float32)
    sigma = np.asarray(stats["sigma"], dtype=np.float32)
    X_std = ((X_all - mu) / sigma).astype(np.float32)

    label_names = stats["label_names"]
    d_in = X_std.shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}; d_in={d_in}; N={len(X_std)}")

    def make_loader(idx, shuffle):
        ds = TensorDataset(
            torch.from_numpy(X_std[idx]),
            torch.from_numpy(y_all[idx]),
            torch.from_numpy(w_all[idx]),
        )
        return DataLoader(
            ds,
            batch_size=args.batch,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

    train_loader = make_loader(npz["idx_train"], shuffle=True)
    val_loader = make_loader(npz["idx_val"], shuffle=False)
    test_loader = make_loader(npz["idx_test"], shuffle=False)

    model = ABCDClassifier(
        d_in,
        hidden=args.hidden,
        num_hidden_layers=args.num_hidden_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    logger.info(
        f"model params: {model.num_parameters()}  "
        f"(hidden={args.hidden}, n_layers={args.num_hidden_layers}, "
        f"dropout={args.dropout})"
    )

    # Persist architecture to feature_stats.json so apply.py can recreate
    # the same model when loading best_model.pt.
    stats_path = run_dir / "feature_stats.json"
    stats = json.loads(stats_path.read_text())
    stats["model_config"] = {
        "hidden": args.hidden,
        "num_hidden_layers": args.num_hidden_layers,
        "dropout": args.dropout,
    }
    stats_path.write_text(json.dumps(stats, indent=2))

    history: list[dict] = []
    best_val_loss = float("inf")
    epochs_since_improvement = 0
    best_path = run_dir / "best_model.pt"

    for epoch in range(args.epochs):
        train_loss, train_acc = _epoch_pass(model, train_loader, optimizer, device, train=True)
        val_loss, val_acc = _epoch_pass(model, val_loader, optimizer, device, train=False)
        scheduler.step(val_loss)

        lr = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "lr": lr,
            }
        )
        logger.info(
            f"epoch {epoch:3d}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  "
            f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  lr={lr:.2e}"
        )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            epochs_since_improvement = 0
            torch.save(model.state_dict(), best_path)
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= args.patience:
                logger.info(
                    f"early stopping at epoch {epoch} (no improvement {args.patience} epochs)"
                )
                break

    log_df = pd.DataFrame(history)
    log_df.to_csv(run_dir / "train_log.csv", index=False)
    logger.info(f"saved {run_dir / 'train_log.csv'}")

    _plot_train_loss(log_df, run_dir / "train_loss.png")

    # Reload best model for the final ROC + metrics dump
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    val_probs, val_y, _ = _evaluate_predictions(model, val_loader, device)
    test_probs, test_y, _ = _evaluate_predictions(model, test_loader, device)  # noqa: RUF059

    val_aucs = _plot_roc_per_class(val_probs, val_y, label_names, run_dir / "roc_per_class.png")

    metrics = {
        "best_val_loss": best_val_loss,
        "epochs_run": len(history),
        "val_aucs": val_aucs,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info(f"final val AUCs: {val_aucs}")
    logger.info(f"test set size: {len(test_y)} (test ROC not plotted; eval via apply.py)")


def main() -> None:
    args = parse_args()

    from . import dataset  # noqa: PLC0415

    run_dir = dataset.prepare_dataset(args)
    print(f"\nrun directory: {run_dir}")

    if args.prepare_only:
        return

    train_classifier(run_dir, args)


if __name__ == "__main__":
    main()
