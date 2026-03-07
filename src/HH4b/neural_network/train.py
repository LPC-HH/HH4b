"""
train.py — Training script for JetClassifier.

Usage:
    python train.py --config config.yaml [--use-wandb]
"""

from __future__ import annotations

import argparse
import contextlib
from datetime import timedelta
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
    configure_logger,
    get_scheduler,
    load_checkpoint,
    resolve_output_dir,
    save_checkpoint,
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
        # Standard CE per sample: (B,)
        log_prob = F.log_softmax(logits, dim=-1)
        ce = F.nll_loss(log_prob, targets, weight=self.weight, reduction="none")

        # p_t = exp(-ce) for the true class
        pt = torch.exp(-ce)
        focal = (1.0 - pt) ** self.gamma * ce

        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


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
        cls_name = next((c for c in classes if c in p.stem), None)
        if cls_name is None:
            raise ValueError(f"Cannot infer class from filename: {p.name}")
        df["label"] = class_to_idx[cls_name]
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    LOGGER.info(f"Loaded {len(df):,} rows from {len(paths)} file(s)")
    return df


def build_dataloaders(
    config: dict[str, Any],
    config_path: str,
    device: torch.device,
    is_main: bool = False,
) -> tuple[DataLoader, DataLoader]:
    train_paths = resolve_file_pattern(config, split="train")
    val_paths = resolve_file_pattern(config, split="val")

    LOGGER.info("Loading training dataset...")
    train_df = load_dataframe(train_paths, config)
    LOGGER.info("Loading validation dataset...")
    val_df = load_dataframe(val_paths, config)

    train_ds = JetDataset(train_df, config_path=config_path)
    val_ds = JetDataset(val_df, config_path=config_path)

    collate_fn = train_ds.make_collate_fn(device=device)

    batch_size = config["training"]["batch_size"]
    num_workers = 4 if is_main else 0

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader


def build_model(config: dict[str, Any], dataset: JetDataset) -> JetClassifier:
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
    group_configs: list[GroupConfig] = dataset.group_configs()
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
) -> tuple[float, float]:
    """Run one epoch. Returns (avg_loss, accuracy)."""
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

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
                    w = sample_weights / sample_weights.sum()
                    loss = (loss_per_sample * w).sum()
                else:
                    loss = loss_per_sample.mean()

            if is_train:
                optimizer.zero_grad()
                if accelerator is not None:
                    accelerator.backward(loss)
                else:
                    loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += batch_size

            pbar.set_postfix(
                loss=f"{total_loss / total:.4f}",
                acc=f"{correct / total:.4f}",
            )

    avg_loss = total_loss / total
    avg_acc = correct / total

    if accelerator is not None:
        t = torch.tensor([avg_loss, avg_acc], device=accelerator.device)
        t = accelerator.gather(t).mean(dim=0)
        avg_loss, avg_acc = t[0].item(), t[1].item()

    return avg_loss, avg_acc


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
                project=config.get("name", "jet-classifier"),
                config=config,
                dir=str(output_dir),
            )
            LOGGER.info("W&B run initialised")
        except ImportError:
            LOGGER.warning("wandb not installed — skipping W&B logging")

    # --- Data ---
    LOGGER.info("Building dataloaders...")
    if accelerator is not None and not accelerator.is_main_process:
        # Non-main processes wait; main process loads data
        accelerator.wait_for_everyone()
        train_loader, val_loader = build_dataloaders(config, config_path=args.config, device=device)
    else:
        train_loader, val_loader = build_dataloaders(config, config_path=args.config, device=device)
        if accelerator is not None:
            accelerator.wait_for_everyone()

    # --- Model ---
    LOGGER.info("Building model...")
    # Use train dataset to infer group configs
    train_ds: JetDataset = train_loader.dataset  # type: ignore[assignment]
    model = build_model(config, dataset=train_ds)
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

    # --- Training loop ---
    training_cfg = config["training"]
    num_epochs = training_cfg["num_epochs"]
    patience = training_cfg.get("patience", float("inf"))
    epochs_no_improve = 0

    LOGGER.info(f"Starting training from epoch {start_epoch} / {num_epochs}")

    for epoch in range(start_epoch, num_epochs):
        train_loss, train_acc = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            is_train=True,
            use_bf16=use_bf16,
            epoch=epoch,
            accelerator=accelerator,
        )
        val_loss, val_acc = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            is_train=False,
            use_bf16=use_bf16,
            epoch=epoch,
            accelerator=accelerator,
        )

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
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
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

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "epoch": epoch,
                        "train/loss": train_loss,
                        "train/acc": train_acc,
                        "val/loss": val_loss,
                        "val/acc": val_acc,
                        "lr": optimizer.param_groups[0]["lr"],
                    }
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
