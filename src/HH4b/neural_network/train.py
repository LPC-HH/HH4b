"""
train.py — Training script for JetClassifier.

Usage:
    python train.py --config config.yaml [--use-wandb]
"""

from __future__ import annotations

import argparse
import contextlib
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import InitProcessGroupKwargs

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from models import (
    InputEmbedding,
    JetClassifier,
    TransformerEncoder,
    build_mlp_head,
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import (
    LOGGER,
    GroupConfig,
    JetDataset,
    ScoreAccumulator,
    compute_roc_metrics,
    configure_logger,
    get_scheduler,
    jet_collate_fn,
    load_checkpoint,
    resolve_output_dir,
    save_checkpoint,
    save_roc_plot,
)


def setup_training_precision(config: dict[str, Any]) -> bool:
    """Configure matmul precision and bf16. Returns resolved use_bf16."""
    matmul_precision = config.get("float32_matmul_precision", "highest")
    torch.set_float32_matmul_precision(matmul_precision)
    LOGGER.info(f"float32 matmul precision: {matmul_precision}")

    use_bf16 = config.get("use_bf16", False)
    if use_bf16:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            LOGGER.info("bf16 enabled")
        else:
            LOGGER.warning(
                "use_bf16=True but bf16 is not supported on this device — falling back to fp32"
            )
            use_bf16 = False
    return use_bf16


class FocalLoss(nn.Module):
    """Multi-class focal loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t).

    Args:
        weight:  Per-class weights (same as nn.CrossEntropyLoss `weight`).
        gamma:   Focusing parameter. 0 = standard cross-entropy.
        reduction: 'mean' or 'sum'.
    """

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.register_buffer("weight", weight)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 1. Get standard log probabilities
        log_prob = F.log_softmax(logits, dim=-1)

        # 2. Get UNWEIGHTED cross-entropy to properly calculate p_t
        ce_unweighted = F.nll_loss(log_prob, targets, reduction="none")

        # 3. Extract p_t (true probability of the correct class)
        pt = torch.exp(-ce_unweighted)

        # 4. Calculate the focal modifier
        focal_term = (1.0 - pt) ** self.gamma

        # 5. Apply class weights if provided
        if self.weight is not None:
            # Gather the specific weight for each target in the batch
            alpha = self.weight[targets]
            loss = alpha * focal_term * ce_unweighted
        else:
            loss = focal_term * ce_unweighted

        # 6. Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()

        return loss


def build_loss(config: dict[str, Any], device: torch.device) -> nn.Module:
    classes = config["dataset"]["classes"]
    weights = torch.tensor([c["cls_weights"] for c in classes], dtype=torch.float32)
    # weights = weights / weights.mean()  # normalize
    weights = weights.to(device)

    loss_cfg = config["training"].get("loss", {})
    loss_type = loss_cfg if isinstance(loss_cfg, str) else loss_cfg.get("type", "CrossEntropy")
    loss_type = loss_type.lower()

    if "focal" in loss_type:
        gamma = loss_cfg.get("gamma", 2.0) if isinstance(loss_cfg, dict) else 2.0
        LOGGER.info(f"Using FocalLoss (gamma={gamma})")
        return FocalLoss(weight=weights, gamma=gamma, reduction="none")  # always none
    else:
        LOGGER.info("Using CrossEntropyLoss")
        return nn.CrossEntropyLoss(weight=weights, reduction="none")


def resolve_file_pattern(config: dict[str, Any], split: str) -> list[Path]:
    """Expand file_pattern for all years and classes, return matching paths."""
    dataset_cfg = config["dataset"]
    base_dir = Path(dataset_cfg["dir"])
    pattern = dataset_cfg["file_pattern"]
    years = dataset_cfg.get("years", [""])
    name = config.get("name", "")
    classes = [c["name"] for c in dataset_cfg.get("classes", [{"name": ""}])]

    paths = []
    for year in years:
        for cls in classes:
            p = (
                pattern.replace("${split}", split)
                .replace("${year}", year)
                .replace("${name}", name)
                .replace("${class}", cls)
            )
            matched = sorted(base_dir.glob(p))
            if not matched:
                LOGGER.warning(f"No files matched pattern: {base_dir / p}")
            paths.extend(Path(m) for m in matched)

    if not paths:
        raise FileNotFoundError(
            f"No files found for split='{split}' under {base_dir} with pattern '{pattern}'"
        )
    return paths


def load_dataframe(paths: list[Path], config: dict) -> pd.DataFrame:
    classes = [c["name"] for c in config["dataset"]["classes"]]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    frames = []
    for p in paths:
        df = pd.read_parquet(p)
        # Infer class from filename stem, e.g. "2024_hh4b.parquet" → "hh4b"
        cls_name = next(
            (c for c in sorted(classes, key=len, reverse=True) if c in p.stem),
            None,
        )
        if cls_name is None:
            raise ValueError(f"Cannot infer class from filename: {p.name}")
        df["label"] = class_to_idx[cls_name]
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    counts = df["label"].value_counts().sort_index()
    idx_to_class = dict(enumerate(classes))
    count_str = "  ".join(f"{idx_to_class.get(i, i)}={n:,}" for i, n in counts.items())
    # Warn loudly for any class that ended up with zero rows
    loaded_classes = set(counts[counts > 0].index.tolist())
    for cls, idx in class_to_idx.items():
        if idx not in loaded_classes:
            LOGGER.warning(
                f"Class '{cls}' has ZERO rows in this split — check that the file exists!"
            )
    LOGGER.info(f"Loaded {len(df):,} rows from {len(paths)} file(s)  [{count_str}]")
    return df


def _pad_dataframe(df: pd.DataFrame, batch_size: int, weight_key: str | None) -> pd.DataFrame:
    remainder = len(df) % batch_size
    if remainder == 0:
        return df
    n_pad = batch_size - remainder
    pad = df.iloc[:n_pad].copy()
    if weight_key is not None:
        pad[weight_key] = 0.0
        return pd.concat([df, pad], ignore_index=True)
    else:
        # No weight column — just trim instead of padding with unmasked samples
        n_keep = len(df) - remainder
        LOGGER.warning(
            f"No weight_key set; trimming {remainder} samples to avoid partial last batch "
            f"({len(df):,} → {n_keep:,})"
        )
        return df.iloc[:n_keep].reset_index(drop=True)


def build_dataloaders(
    config: dict[str, Any],
    config_path: str,
    device: torch.device,
    is_main: bool = False,
    accelerator: Accelerator | None = None,
) -> tuple[DataLoader, DataLoader]:
    group_configs = JetDataset.build_group_configs(config)
    collate_fn = partial(jet_collate_fn, group_configs=group_configs, device=device)
    batch_size = config["training"]["batch_size"]

    if is_main:
        train_df = load_dataframe(resolve_file_pattern(config, "train"), config)
        val_df = load_dataframe(resolve_file_pattern(config, "val"), config)
        train_df = _pad_dataframe(train_df, batch_size, config["dataset"].get("weight_key"))
        val_df = _pad_dataframe(val_df, batch_size, config["dataset"].get("weight_key"))
        train_ds = JetDataset(train_df, config_path=config_path)
        val_ds = JetDataset(val_df, config_path=config_path)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=False, collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn
        )
    else:
        train_loader = DataLoader([], collate_fn=collate_fn)
        val_loader = DataLoader([], collate_fn=collate_fn)

    if accelerator is not None:
        accelerator.wait_for_everyone()

    return train_loader, val_loader


def build_model(config: dict[str, Any]) -> JetClassifier:
    """Build JetClassifier from config + dataset group configs.

    Steps:
        1. Read encoder dim and group configs to build InputEmbedding per group.
        2. Build TransformerEncoder from architecture.encoder.
        3. Build MLP head from architecture.head.
        4. Wrap in JetClassifier.
    """

    arch = config["architecture"]
    enc_cfg = arch["encoder"]
    head_cfg = arch["head"]

    dim = enc_cfg["dim"]
    num_classes = len(config["dataset"]["classes"])

    # --- InputEmbeddings ---
    group_configs: list[GroupConfig] = JetDataset.build_group_configs(config)
    embeddings = nn.ModuleDict()
    for grp in group_configs:
        embeddings[grp.name] = InputEmbedding(
            n_continuous=grp.n_continuous,
            dim=dim,
            discrete_num_bins=grp.discrete_num_bins(),
        )

    # --- Encoder ---
    encoder = TransformerEncoder(
        dim=dim,
        num_layers=enc_cfg["num_layers"],
        num_heads=enc_cfg["num_heads"],
        activation=enc_cfg.get("activation", "SwiGLU"),
        norm=enc_cfg.get("norm", "LayerNorm"),
        layer_scale_init=enc_cfg.get("layer_scale_init", 1e-4),
        num_registers=enc_cfg.get("num_registers", 0),
        mlp_ratio=enc_cfg.get("mlp_ratio", 4),
        qkv_bias=enc_cfg.get("qkv_bias", True),
        attention_dropout=enc_cfg.get("attention_dropout", 0.0),
        norm_eps=enc_cfg.get("norm_eps", 1e-5),
        apply_final_norm=enc_cfg.get("apply_final_norm", True),
        apply_embedding_norm=enc_cfg.get("apply_embedding_norm", False),
    )

    # --- Head ---
    # Pool output is CLS ⊕ masked mean → 2 * dim
    hidden_dim = head_cfg.get("mlp_hidden_dim")
    if isinstance(hidden_dim, int):
        hidden_dim = [hidden_dim] * head_cfg.get("num_hidden_layers", 1)

    head = build_mlp_head(
        in_dim=2 * dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_hidden_layers=head_cfg.get("num_hidden_layers", 1),
        batch_norm=head_cfg.get("batch_norm", False),
        dropout=head_cfg.get("dropout", 0.0),
    )

    model = JetClassifier(
        embeddings=embeddings,
        encoder=encoder,
        head=head,
        dim=dim,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info(f"Built model with {n_params:,} trainable parameters")
    return model


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def save_ckpt(
    config: dict[str, Any],
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    val_loss: float,
    is_best: bool,
    accelerator: Accelerator | None = None,
) -> None:
    unwrapped_model = accelerator.unwrap_model(model) if accelerator is not None else model
    ckpt = {
        "epoch": epoch,
        "model": unwrapped_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "val_loss": val_loss,
    }
    path = save_checkpoint(ckpt, config=config, epoch_num=epoch)
    LOGGER.info(f"Saved checkpoint → {path}")
    if is_best:
        best_path = save_checkpoint(ckpt, config=config, epoch_num="best")
        LOGGER.info(f"New best model → {best_path}")


def maybe_resume(
    config: dict[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
) -> tuple[int, float]:
    """Load checkpoint if training.load_epoch is set. Returns (start_epoch, best_val_loss)."""
    load_epoch = config["training"].get("load_epoch")
    if load_epoch is None:
        return 0, float("inf")

    LOGGER.info(f"Resuming from epoch: {load_epoch}")
    # Will determine device later
    ckpt = load_checkpoint(config=config, epoch_num=load_epoch, map_location="cpu")

    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])

    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])

    start_epoch = ckpt["epoch"] + 1
    best_val_loss = ckpt.get("val_loss", float("inf"))
    LOGGER.info(f"Resumed from epoch {ckpt['epoch']}, best val_loss={best_val_loss:.4f}")
    return start_epoch, best_val_loss


# ---------------------------------------------------------------------------
# Train / val loops
# ---------------------------------------------------------------------------
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    is_train: bool,
    use_bf16: bool = False,
    epoch: int = 0,
    accelerator: Accelerator | None = None,
    wandb_run: Any = None,
    global_step: int = 0,
    collect_scores: bool = False,
) -> tuple[float, float, int, ScoreAccumulator | None]:
    """Run one epoch. Returns (avg_loss, accuracy, global_step, score_accumulator|None).

    Args:
        collect_scores: If True, accumulate logits/labels/weights for ROC computation.
                        Only meaningful for validation (is_train=False).
    """
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    accumulator = ScoreAccumulator() if collect_scores else None

    amp_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if (use_bf16 and accelerator is None)
        else contextlib.nullcontext()
    )
    grad_ctx = torch.enable_grad() if is_train else torch.no_grad()

    phase = "train" if is_train else "val"
    is_main = (accelerator is None) or (accelerator.is_main_process)
    pbar = tqdm(
        loader,
        desc=f"Epoch {epoch:03d} [{phase}]",
        leave=False,
        dynamic_ncols=True,
        disable=not is_main,
    )

    with grad_ctx:
        for batch in pbar:
            labels = batch.pop("label")  # already on device
            sample_weights = batch.pop("sample_weight", None)  # already on device

            with amp_ctx:
                logits = model(batch)
                loss_per_sample = criterion(logits, labels)  # (B,)

                if sample_weights is not None:
                    loss = (loss_per_sample * sample_weights).sum() / sample_weights.sum().clamp(
                        min=1e-8
                    )
                else:
                    loss = loss_per_sample.mean()

            if is_train:
                optimizer.zero_grad()
                if accelerator is not None:
                    accelerator.backward(loss)
                else:
                    loss.backward()
                optimizer.step()

                # --- Batch-level W&B logging (train only) ---
                if wandb_run is not None and is_main:
                    if sample_weights is not None:
                        batch_acc = (
                            ((logits.argmax(dim=-1) == labels) * sample_weights).sum()
                            / sample_weights.sum()
                        ).item()
                    else:
                        batch_acc = (logits.argmax(dim=-1) == labels).float().mean().item()
                    wandb_run.log(
                        {
                            "batch/loss": loss.item(),
                            "batch/acc": batch_acc,
                            "batch/lr": optimizer.param_groups[0]["lr"],
                            "global_step": global_step,
                        },
                        step=global_step,
                    )

                global_step += 1

            # --- Score accumulation (val only) ---
            if accumulator is not None:
                accumulator.update(logits, labels, sample_weights)

            batch_size = labels.size(0)
            if sample_weights is not None:
                total_loss += loss.item() * sample_weights.sum().item()
                correct += ((logits.argmax(dim=-1) == labels) * sample_weights).sum().item()
                total += sample_weights.sum().item()
            else:
                total_loss += loss.item() * batch_size
                correct += (logits.argmax(dim=-1) == labels).sum().item()
                total += batch_size

            pbar.set_postfix(
                loss=f"{total_loss / total:.5e}",
                acc=f"{correct / total:.2%}",
            )

    avg_loss = total_loss / total
    avg_acc = correct / total

    if accelerator is not None:
        t = torch.tensor([avg_loss, avg_acc], device=accelerator.device)
        gathered = accelerator.gather(t)  # (num_gpus * 2,) or (2,)
        gathered = gathered.view(-1, 2).mean(dim=0)  # always (2,)
        avg_loss, avg_acc = gathered[0].item(), gathered[1].item()

        LOGGER.info(
            f"Epoch {epoch:03d} [{phase}] | "
            f"avg_loss={avg_loss:.5e} avg_acc={avg_acc:.2%} (gathered across {accelerator.num_processes} processes)"
        )

    return avg_loss, avg_acc, global_step, accumulator


# ---------------------------------------------------------------------------
# ROC evaluation
# ---------------------------------------------------------------------------


def run_roc_eval(
    accumulator: ScoreAccumulator,
    config: dict[str, Any],
    output_dir: Path,
    epoch: int,
    wandb_run: Any = None,
    global_step: int = 0,
    split: str = "val",
) -> dict[str, dict[str, Any]]:
    """Compute ROC metrics for all configured roc_groups and log/save results.

    Args:
        accumulator:  Filled ScoreAccumulator from the val loop.
        config:       Full training config dict.
        output_dir:   Experiment output directory (plots saved under roc_curves/).
        epoch:        Current epoch (used for filenames and logging).
        wandb_run:    Active W&B run or None.
        global_step:  Current global step for W&B x-axis.
        split:        Which split was evaluated (filters roc_groups by their 'splits' field).

    Returns:
        {group_name: result_dict} for all evaluated groups.
    """
    roc_groups = config["training"].get("roc_groups", [])
    if not roc_groups:
        return {}

    classes = [c["name"] for c in config["dataset"]["classes"]]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    all_logits, all_labels, all_weights = accumulator.finalize()

    results: dict[str, dict[str, Any]] = {}
    wandb_metrics: dict[str, float] = {}

    for roc_group in roc_groups:
        # Only evaluate groups configured for this split
        if split not in roc_group.get("splits", ["val"]):
            continue

        name = roc_group["name"]
        res = compute_roc_metrics(
            all_logits=all_logits,
            all_labels=all_labels,
            all_weights=all_weights,
            roc_group=roc_group,
            class_to_idx=class_to_idx,
        )
        results[name] = res

        # --- Log to console ---
        # sig_effs is a list of (eps_b_target, eps_s_achieved) tuples
        sig_eff_strs = "  ".join(
            f"eps_s={eps_s:.4f}@eps_b={eps_b:.0e}" for eps_b, eps_s in res["sig_effs"]
        )
        LOGGER.info(f"  ROC [{name}] | AUC={res['auc']:.4f}  {sig_eff_strs}")

        # --- Collect W&B metrics ---
        wandb_metrics[f"roc/{name}/auc"] = res["auc"]
        for eps_b, eps_s in res["sig_effs"]:
            wandb_metrics[f"roc/{name}/sig_eff@bkg{eps_b:.0e}"] = eps_s

    if not results:
        return results

    # --- Save ROC curve plots ---
    roc_plot_dir = output_dir / "roc_curves"
    save_roc_plot(results, roc_plot_dir, epoch)
    LOGGER.info(f"  ROC plots saved → {roc_plot_dir}/roc_epoch_{epoch:04d}.[png|pdf]")

    # --- Log to W&B ---
    if wandb_run is not None and wandb_metrics:
        wandb_metrics["epoch"] = epoch
        wandb_run.log(wandb_metrics, step=global_step)

        # Also log the ROC plot image
        try:
            import wandb  # noqa: PLC0415

            fig_path = roc_plot_dir / f"roc_epoch_{epoch:04d}.png"
            if fig_path.exists():
                wandb_run.log(
                    {"roc/curves": wandb.Image(str(fig_path), caption=f"Epoch {epoch:04d}")},
                    step=global_step,
                )
        except Exception:
            pass

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train JetClassifier")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file. Overrides the default <output_dir>/logs/train.log",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- Config ---
    with Path(args.config).open() as f:
        config = yaml.safe_load(f)

    # after device is resolved, before building data/model:
    use_bf16 = setup_training_precision(config)

    # optionally wrap with accelerate:
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    accelerator = None
    use_accelerate = config.get("use_accelerate", False)
    if use_accelerate:
        dataloader_config = DataLoaderConfiguration(
            dispatch_batches=True,  # Main process loads batches
            split_batches=False,
        )
        kwargs = InitProcessGroupKwargs(timeout=timedelta(days=365))
        accelerator = Accelerator(
            dataloader_config=dataloader_config,
            kwargs_handlers=[kwargs],
            mixed_precision="bf16" if use_bf16 else "no",
        )
        device = accelerator.device
        # only log main process info
        LOGGER.addFilter(lambda _: accelerator.is_main_process)
        LOGGER.info(
            f"Accelerate enabled (device={device}, mixed_precision={accelerator.mixed_precision})"
        )

    is_main = (accelerator is None) or accelerator.is_main_process

    # --- Output dir & logging ---
    output_dir = resolve_output_dir(config)
    if is_main:
        if args.log_file is not None:
            log_file = Path(args.log_file)
        else:
            log_file = output_dir / "logs" / "train.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        configure_logger(LOGGER, log_file=str(log_file), log_level=args.log_level)

    # --- W&B ---
    wandb_run = None
    if args.use_wandb and is_main:
        try:
            import wandb  # noqa: PLC0415

            wandb_run = wandb.init(
                project="HH4b-nn-trainings",
                name=config.get("name", "jet-classifier"),
                config=config,
                dir=str(output_dir),
            )
            LOGGER.info("W&B run initialized")
        except ImportError:
            LOGGER.warning("wandb not installed — skipping W&B logging")

    # --- Data ---
    LOGGER.info("Building dataloaders...")
    train_loader, val_loader = build_dataloaders(
        config, config_path=args.config, device=device, is_main=is_main, accelerator=accelerator
    )

    # --- Model ---
    LOGGER.info("Building model...")
    model = build_model(config)
    if accelerator is None:
        model = model.to(device)

    # --- Optimizer ---
    opt_cfg = config["training"]["optimizer"]
    optimizer = getattr(torch.optim, opt_cfg["type"])(
        model.parameters(), **opt_cfg.get("kwargs", {})
    )

    # --- Scheduler ---
    scheduler = get_scheduler(optimizer, config)

    # --- Loss ---
    criterion = build_loss(config, device)

    # --- Resume ---
    start_epoch, best_val_loss = maybe_resume(
        config=config, model=model, optimizer=optimizer, scheduler=scheduler
    )

    if accelerator is not None:
        model, optimizer, train_loader, val_loader = accelerator.prepare(
            model, optimizer, train_loader, val_loader
        )
        if scheduler is not None:
            scheduler = accelerator.prepare(scheduler)

    # --- ROC config ---
    roc_groups = config["training"].get("roc_groups", [])
    has_roc = bool(roc_groups) and is_main
    if has_roc:
        LOGGER.info(f"ROC evaluation enabled for {len(roc_groups)} group(s)")

    # --- Training loop ---
    training_cfg = config["training"]
    num_epochs = training_cfg["num_epochs"]
    patience = training_cfg.get("patience", float("inf"))
    epochs_no_improve = 0
    global_step = 0  # tracks total batches seen across all epochs

    LOGGER.info(f"Starting training from epoch {start_epoch}/{num_epochs}")

    for epoch in range(start_epoch, num_epochs):
        train_loss, train_acc, global_step, _ = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            is_train=True,
            use_bf16=use_bf16,
            epoch=epoch,
            accelerator=accelerator,
            wandb_run=wandb_run,
            global_step=global_step,
            collect_scores=False,
        )
        val_loss, val_acc, _, val_accumulator = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            is_train=False,
            use_bf16=use_bf16,
            epoch=epoch,
            accelerator=accelerator,
            wandb_run=None,  # no batch-level logging for val
            global_step=global_step,
            collect_scores=has_roc,  # accumulate scores only if ROC groups are configured
        )

        if val_accumulator is not None and accelerator is not None:
            val_accumulator = val_accumulator.gather(accelerator)

        if scheduler is not None:
            scheduler.step()

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if is_main:
            LOGGER.info(
                f"Epoch {epoch:04d} | "
                f"train_loss={train_loss:.5e} train_acc={train_acc:.2%} | "
                f"val_loss={val_loss:.5e} val_acc={val_acc:.2%}"
                + (" ← best" if is_best else f" (no improve {epochs_no_improve}/{patience})")
            )

            save_ckpt(
                config=config,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                val_loss=val_loss,
                is_best=is_best,
                accelerator=accelerator,
            )

            # --- ROC evaluation ---
            if has_roc and val_accumulator is not None:
                run_roc_eval(
                    accumulator=val_accumulator,
                    config=config,
                    output_dir=output_dir,
                    epoch=epoch,
                    wandb_run=wandb_run,
                    global_step=global_step,
                    split="val",
                )

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "epoch": epoch,
                        "train/loss": train_loss,
                        "train/acc": train_acc,
                        "val/loss": val_loss,
                        "val/acc": val_acc,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    step=global_step,
                )

        should_stop = epochs_no_improve >= patience
        if accelerator is not None:
            stop_tensor = torch.tensor(int(should_stop), device=device)
            torch.distributed.broadcast(stop_tensor, src=0)
            should_stop = bool(stop_tensor.item())

        if should_stop:
            if is_main:
                LOGGER.info(
                    f"Early stopping triggered after {patience} epochs without improvement."
                )
            break

    if is_main:
        LOGGER.info(f"Training complete. Best val_loss={best_val_loss:.4f}")
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
