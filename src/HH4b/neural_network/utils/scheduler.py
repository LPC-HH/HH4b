from __future__ import annotations

import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LambdaLR,
    LinearLR,
    ReduceLROnPlateau,
    SequentialLR,
    StepLR,
    _LRScheduler,
)

from .ckpt import get_checkpoints_path
from .logger import LOGGER


def get_cosine_schedule(
    iter: int,
    total_iters: int,
    base_value: float = 1e-3,
    final_value: float = 0.0,
):
    # Clamp iter to [0, total_iters] to avoid returning nonsense values if
    # last_epoch overshoots (e.g. due to a double-step or off-by-one on resume).
    iter = max(0, min(iter, total_iters))
    return final_value + 0.5 * (base_value - final_value) * (
        1 + math.cos(math.pi * iter / total_iters)
    )


def get_scheduler(
    optimizer: Optimizer,
    config: dict,
) -> _LRScheduler | ReduceLROnPlateau | None:
    """
    Get the scheduler. If not specified, will give None.
    """

    scheduler_config = config["training"].get("scheduler")
    if not scheduler_config:
        LOGGER.info("No scheduler specified in config. Returning None.")
        return None

    scheduler_name = scheduler_config.get("name", "").lower()
    scheduler_params = scheduler_config.get("params", {})

    if scheduler_name == "steplr":
        scheduler = StepLR(optimizer, **scheduler_params)
    elif scheduler_name == "reducelronplateau":
        scheduler = ReduceLROnPlateau(optimizer, **scheduler_params)
    elif scheduler_name == "cosineannealinglr":
        scheduler = CosineAnnealingLR(optimizer, **scheduler_params)
    elif scheduler_name == "cosineannealingwarmrestarts":
        scheduler = CosineAnnealingWarmRestarts(optimizer, **scheduler_params)
    elif scheduler_name == "linearlr":
        scheduler = LinearLR(optimizer, **scheduler_params)
    elif scheduler_name == "constantlr":
        scheduler = ConstantLR(optimizer, **scheduler_params)
    elif scheduler_name == "cosinescheduler":
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda it: get_cosine_schedule(iter=it, **scheduler_params),
        )
    elif scheduler_name == "sequentiallr":
        scheduler = _create_sequential_scheduler(optimizer, scheduler_params)
        if not scheduler:
            return None
    else:
        LOGGER.warning(f"Unknown scheduler type: {scheduler_name}. Returning None.")
        return None

    LOGGER.info(f"Created scheduler: {scheduler.__class__.__name__}")
    LOGGER.debug(f"Scheduler parameters: {scheduler_params}")

    # Load scheduler state if specified in config
    epoch = config["training"].get("load_epoch")
    if epoch:
        ckpt_path = get_checkpoints_path(config=config, epoch_num=epoch)
        try:
            state_dict = torch.load(ckpt_path)
        except RuntimeError as e:
            LOGGER.error(f"Failed to load checkpoint {ckpt_path}: {e}. Trying cpu.")
            state_dict = torch.load(ckpt_path, map_location="cpu")
        _load_scheduler_state(scheduler, state_dict)

    return scheduler


def _create_sequential_scheduler(optimizer: Optimizer, scheduler_params: dict):
    """Create a SequentialLR scheduler from config."""
    schedulers_config = scheduler_params.get("schedulers", [])
    milestones = scheduler_params.get("milestones", [])

    if not schedulers_config:
        LOGGER.error("SequentialLR requires 'schedulers' list in params")
        return None

    if not milestones:
        LOGGER.error("SequentialLR requires 'milestones' list in params")
        return None

    # Create individual schedulers
    schedulers = []
    for i, sched_config in enumerate(schedulers_config):
        sched_name = sched_config.get("name", "").lower()
        sched_params = sched_config.get("params", {})

        if sched_name == "steplr":
            scheduler = StepLR(optimizer, **sched_params)
        elif sched_name == "reducelronplateau":
            LOGGER.error("ReduceLROnPlateau cannot be used in SequentialLR")
            return None
        elif sched_name == "cosineannealinglr":
            scheduler = CosineAnnealingLR(optimizer, **sched_params)
        elif sched_name == "cosineannealingwarmrestarts":
            scheduler = CosineAnnealingWarmRestarts(optimizer, **sched_params)
        elif sched_name == "linearlr":
            scheduler = LinearLR(optimizer, **sched_params)
        elif sched_name == "constantlr":
            scheduler = ConstantLR(optimizer, **sched_params)
        elif sched_name == "cosinescheduler":
            scheduler = LambdaLR(
                optimizer,
                lr_lambda=lambda it, p=sched_params: get_cosine_schedule(it, **p),
            )
        else:
            LOGGER.error(f"Unknown scheduler type in SequentialLR: {sched_name}")
            return None

        schedulers.append(scheduler)
        LOGGER.info(f"Created scheduler {i} for SequentialLR: {scheduler.__class__.__name__}")

    return SequentialLR(optimizer, schedulers=schedulers, milestones=milestones)


def _load_scheduler_state(
    scheduler: _LRScheduler | ReduceLROnPlateau,
    state_dict: dict,
) -> None:
    if "scheduler" not in state_dict:
        LOGGER.warning(
            "Checkpoint file does not contain scheduler state. Starting with fresh scheduler."
        )
        return

    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.best = state_dict["scheduler"].get("best", scheduler.best)
        scheduler.num_bad_epochs = state_dict["scheduler"].get(
            "num_bad_epochs", scheduler.num_bad_epochs
        )
        LOGGER.info(
            f"Loaded ReduceLROnPlateau state: best={scheduler.best}, "
            f"num_bad_epochs={scheduler.num_bad_epochs}"
        )
    else:
        scheduler.load_state_dict(state_dict["scheduler"])
        LOGGER.info(f"Loaded scheduler state from checkpoint (last_epoch={scheduler.last_epoch})")

        # After restoring scheduler state, the optimizer's param_groups['lr'] will
        # be whatever was saved in the optimizer state dict (the raw decayed value).
        # We need to recompute LRs from the scheduler's restored last_epoch and
        # write them back into the optimizer, so the two are consistent.
        #
        # _get_closed_form_lr() is not available on all schedulers, so we use
        # the safer approach: temporarily step the scheduler back one tick and
        # re-step it forward to trigger a recompute.
        #
        # The cleanest portable approach: directly call get_lr() via the internal
        # mechanism by computing LRs and writing them to param groups.
        _sync_optimizer_lr_from_scheduler(scheduler)


def _sync_optimizer_lr_from_scheduler(scheduler: _LRScheduler) -> None:
    """
    After load_state_dict, the optimizer param_group LRs may be stale (from the
    optimizer state dict). This recomputes the correct LR from the scheduler's
    restored last_epoch and writes it back into the optimizer param groups.

    Strategy: decrement last_epoch by 1, call .step() to recompute and apply LRs,
    which increments last_epoch back to where it was. Net effect: LRs in optimizer
    are recomputed correctly from the restored scheduler state.
    """
    # Save the current last_epoch
    saved_last_epoch = scheduler.last_epoch

    # Decrement so that .step() brings us back to saved_last_epoch
    scheduler.last_epoch = saved_last_epoch - 1

    # Suppress the "Detected call of lr_scheduler.step() before optimizer.step()"
    # warning that PyTorch emits if step() is called out of the normal order.
    import warnings  # noqa: PLC0415

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scheduler.step()

    # Verify
    restored_lrs = [pg["lr"] for pg in scheduler.optimizer.param_groups]
    LOGGER.info(
        f"Resynced optimizer LRs from scheduler (last_epoch={scheduler.last_epoch}): {restored_lrs}"
    )
