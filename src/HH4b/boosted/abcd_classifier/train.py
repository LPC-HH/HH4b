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

k-fold ensemble member (``--n-folds K --fold i [--fold-seed S]``; HIG-24-010
style ensemble of K networks, combined per event by ``apply.py --ensemble-dir``):

  * reads the run's existing ``processed_data.npz`` (never rebuilds it) and
    keeps its test split, identical for every member;
  * deals the train+val rows into K stratified folds with ``--fold-seed``
    (``dataset.kfold_partition``; independent of the init seed);
  * trains on the other K-1 folds and early-stops on fold i, with init/shuffle
    seed ``--seed + i`` and μ/σ from its own training rows;
  * writes everything into ``<run>/kfold<K>_s<S>/fold<ii>/`` (the run dir's
    own files are only read) and records the partition hashes in
    ``fold<ii>/kfold_member.json`` (with the training settings and provenance)
    and ``kfold<K>_s<S>/kfold_assignment.json``;
  * ``fold<ii>/metrics.json`` is the completion marker: deleted when the member
    starts, written last (atomically) with the sha256 of ``best_model.pt``, the
    best epoch, and the loss, accuracy and AUCs on the common test split (also
    plotted in ``roc_per_class_test.png``).  ``apply.py --ensemble-dir`` refuses
    members without it.

Without ``--n-folds`` (or with 0) the behaviour is the single-model one.

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
import os
import platform
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
        choices=hh_vars.years + ["2022-2023", "2022-2023-2024", "2022-2025"],
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
    add_bool_arg(
        parser,
        "fast-loader",
        "Fetch each mini-batch with one tensor index (same RandomSampler/BatchSampler, so the "
        "same batches in the same order) instead of per-event __getitem__ + collate; much faster "
        "on CPU; ignores --num-workers",
        default=False,
    )

    # k-fold ensemble member
    parser.add_argument(
        "--n-folds",
        type=int,
        default=0,
        help="k-fold ensemble member mode with K folds (0 = single model, the default).  Keeps "
        "the NPZ test split, partitions the train+val rows into K folds, trains on K-1 of them "
        "and early-stops on fold --fold; outputs go to <run>/kfold<K>_s<fold-seed>/fold<ii>/.  "
        "Needs an existing processed_data.npz (build it once with --prepare-only).",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="member index i in [0, K): validation fold of this member (with --n-folds).  "
        "The member's init/shuffle seed is --seed + i.",
    )
    parser.add_argument(
        "--fold-seed",
        type=int,
        default=2026,
        help="seed of the fold partition (with --n-folds); must be the same for all members.",
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

    args = parser.parse_args()
    if args.n_folds:
        if args.n_folds < 2:
            parser.error("--n-folds must be >= 2 (or 0 for a single model)")
        if args.fold is None or not 0 <= args.fold < args.n_folds:
            parser.error(f"--n-folds {args.n_folds} needs --fold in [0, {args.n_folds})")
        if args.prepare_only or args.force_rebuild:
            parser.error(
                "--n-folds never builds the NPZ (all members must share it); "
                "build it once without --n-folds"
            )
        npz_path = Path(args.out_dir) / args.run_name / "processed_data.npz"
        if not npz_path.exists():
            parser.error(f"--n-folds needs an existing {npz_path}")
    elif args.fold is not None:
        parser.error("--fold needs --n-folds")
    return args


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


def _member_train_config(args) -> dict:
    """Training settings of a k-fold member (``dataset.KFOLD_TRAIN_CONFIG_KEYS``);
    ``apply.py --ensemble-dir`` requires them to be identical for all members."""
    from . import dataset  # noqa: PLC0415

    cfg = {
        "hidden": args.hidden,
        "num_hidden_layers": args.num_hidden_layers,
        "dropout": args.dropout,
        "lr": args.lr,
        "batch": args.batch,
        "epochs": args.epochs,
        "patience": args.patience,
        "base_seed": args.seed,
        "fast_loader": bool(getattr(args, "fast_loader", False)),
    }
    assert tuple(cfg) == dataset.KFOLD_TRAIN_CONFIG_KEYS
    return cfg


def _setup_kfold_member(
    run_dir: Path,
    npz,
    X_all: np.ndarray,
    y_all: np.ndarray,
    stats: dict,
    args,
    seed: int,
    device,
) -> tuple[Path, np.ndarray, np.ndarray, dict, dict]:
    """k-fold member i: partition, (train, val) rows, own μ/σ and output dir.

    Deletes a completion marker (``metrics.json``) left in the member dir by an
    earlier attempt before writing anything, so an interrupted relaunch can
    never pass as a finished member.  Returns ``(out_dir, idx_train, idx_val,
    stats, member_info)``; ``stats`` is the run's feature_stats with μ/σ
    recomputed on the member's training rows (as
    ``dataset.standardize_and_save`` does for the single model).
    """  # noqa: RUF002
    import torch  # noqa: PLC0415

    from . import dataset  # noqa: PLC0415

    k, i = args.n_folds, args.fold
    idx_test = npz["idx_test"]
    fold = dataset.kfold_partition(
        y_all, npz["idx_train"], npz["idx_val"], idx_test, k, args.fold_seed
    )
    fingerprint = dataset.kfold_fingerprint(y_all, idx_test, fold, k, args.fold_seed)
    ens_dir = run_dir / dataset.kfold_dirname(k, args.fold_seed)
    dataset.write_or_check_assignment(ens_dir, fold, fingerprint)

    idx_train, idx_val = dataset.member_split(fold, i)
    mu, sigma = dataset.train_standardization(X_all, idx_train)
    n_cls = len(stats["label_names"])
    member = {
        **fingerprint,
        "fold": int(i),
        "seed": int(seed),
        "run_dir": str(run_dir),
        "n_train": len(idx_train),
        "n_val": len(idx_val),
        "class_counts_train": np.bincount(y_all[idx_train], minlength=n_cls).tolist(),
        "class_counts_val": np.bincount(y_all[idx_val], minlength=n_cls).tolist(),
        "train_config": _member_train_config(args),
        # informational only (not checked): where the member ran; NODE_NAME is set by the
        # NRP job YAML (downward API), GPU results are not bit-reproducible across GPU types
        "provenance": {
            "host": platform.node(),
            "node": os.environ.get("NODE_NAME"),
            "device": device.type,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    out_dir = ens_dir / dataset.member_dirname(i)
    out_dir.mkdir(parents=True, exist_ok=True)
    stale = out_dir / dataset.KFOLD_MEMBER_DONE
    if stale.exists():
        logger.info(f"removing the completion marker of an earlier attempt: {stale}")
        stale.unlink()
    (out_dir / dataset.KFOLD_MEMBER_JSON).write_text(json.dumps(member, indent=2))
    stats = {**stats, "mu": mu.tolist(), "sigma": sigma.tolist()}
    (out_dir / "feature_stats.json").write_text(json.dumps(stats, indent=2))
    logger.info(
        f"k-fold member {i}/{k} (fold seed {args.fold_seed}, init seed {seed}): "
        f"train/val/test = {len(idx_train)}/{len(idx_val)}/{len(idx_test)}; output {out_dir}"
    )
    return out_dir, idx_train, idx_val, stats, member


def train_classifier(run_dir: Path, args) -> None:
    """Task 4 — train the MLP, log per-epoch metrics, save best model + plots.

    With ``args.n_folds`` > 0, train k-fold member ``args.fold`` instead (see
    the module docstring); its outputs go to the member dir, not ``run_dir``.
    """
    import torch  # noqa: PLC0415
    from torch.optim.lr_scheduler import ReduceLROnPlateau  # noqa: PLC0415
    from torch.utils.data import DataLoader, TensorDataset  # noqa: PLC0415

    from .model import ABCDClassifier  # noqa: PLC0415

    n_folds = getattr(args, "n_folds", 0) or 0
    seed = args.seed + args.fold if n_folds else args.seed
    _set_seeds(seed)

    # Load processed data + standardization stats
    npz = np.load(run_dir / "processed_data.npz")
    stats = json.loads((run_dir / "feature_stats.json").read_text())

    X_all = npz["X"]
    y_all = npz["y"]
    w_all = npz["w"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    member = None
    if n_folds:
        out_dir, idx_train, idx_val, stats, member = _setup_kfold_member(
            run_dir, npz, X_all, y_all, stats, args, seed, device
        )
    else:
        out_dir, idx_train, idx_val = run_dir, npz["idx_train"], npz["idx_val"]

    mu = np.asarray(stats["mu"], dtype=np.float32)
    sigma = np.asarray(stats["sigma"], dtype=np.float32)
    X_std = ((X_all - mu) / sigma).astype(np.float32)

    label_names = stats["label_names"]
    d_in = X_std.shape[1]

    logger.info(f"device: {device}; d_in={d_in}; N={len(X_std)}")

    def make_loader(idx, shuffle):
        ds = TensorDataset(
            torch.from_numpy(X_std[idx]),
            torch.from_numpy(y_all[idx]),
            torch.from_numpy(w_all[idx]),
        )
        if getattr(args, "fast_loader", False):
            # Same samplers as DataLoader(batch_size=args.batch, shuffle=shuffle) builds
            # internally, so the batches and the global-RNG draws are identical; the batch
            # is fetched as ds[list_of_indices] (TensorDataset supports it) with no collate.
            from torch.utils.data import (  # noqa: PLC0415
                BatchSampler,
                RandomSampler,
                SequentialSampler,
            )

            base = RandomSampler(ds) if shuffle else SequentialSampler(ds)
            return DataLoader(
                ds,
                batch_size=None,
                sampler=BatchSampler(base, batch_size=args.batch, drop_last=False),
                num_workers=0,
                pin_memory=(device.type == "cuda"),
            )
        return DataLoader(
            ds,
            batch_size=args.batch,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

    train_loader = make_loader(idx_train, shuffle=True)
    val_loader = make_loader(idx_val, shuffle=False)
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
    stats_path = out_dir / "feature_stats.json"
    stats = json.loads(stats_path.read_text())
    stats["model_config"] = {
        "hidden": args.hidden,
        "num_hidden_layers": args.num_hidden_layers,
        "dropout": args.dropout,
    }
    stats_path.write_text(json.dumps(stats, indent=2))

    history: list[dict] = []
    best_val_loss = float("inf")
    best_epoch = -1
    stopped_early = False
    epochs_since_improvement = 0
    best_path = out_dir / "best_model.pt"

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
            best_epoch = epoch
            epochs_since_improvement = 0
            torch.save(model.state_dict(), best_path)
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= args.patience:
                logger.info(
                    f"early stopping at epoch {epoch} (no improvement {args.patience} epochs)"
                )
                stopped_early = True
                break

    log_df = pd.DataFrame(history)
    log_df.to_csv(out_dir / "train_log.csv", index=False)
    logger.info(f"saved {out_dir / 'train_log.csv'}")

    _plot_train_loss(log_df, out_dir / "train_loss.png")

    # Reload best model for the final ROC + metrics dump
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    val_probs, val_y, _ = _evaluate_predictions(model, val_loader, device)
    test_probs, test_y, _ = _evaluate_predictions(model, test_loader, device)

    val_aucs = _plot_roc_per_class(val_probs, val_y, label_names, out_dir / "roc_per_class.png")

    metrics = {
        "best_val_loss": best_val_loss,
        "epochs_run": len(history),
        "val_aucs": val_aucs,
    }
    if member is None:
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        logger.info(f"final val AUCs: {val_aucs}")
        logger.info(f"test set size: {len(test_y)} (test ROC not plotted; eval via apply.py)")
        return

    # k-fold member: metrics on the common test split (identical rows for every member), then
    # the completion marker, written last
    from . import dataset  # noqa: PLC0415

    test_loss, test_acc = _epoch_pass(model, test_loader, None, device, train=False)
    test_aucs = _plot_roc_per_class(
        test_probs, test_y, label_names, out_dir / "roc_per_class_test.png"
    )
    metrics.update(
        {
            "best_epoch": best_epoch,
            "stopped_early": stopped_early,
            "n_test": len(test_y),
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_aucs": test_aucs,
            "best_model_sha256": dataset.file_sha256(best_path),
            "kfold": member,
        }
    )
    dataset.write_json_atomic(out_dir / dataset.KFOLD_MEMBER_DONE, metrics)
    logger.info(f"final val AUCs: {val_aucs}")
    logger.info(f"test loss {test_loss:.5f}, test AUCs: {test_aucs}")
    logger.info(f"k-fold member {member['fold']} complete: {out_dir / dataset.KFOLD_MEMBER_DONE}")


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
