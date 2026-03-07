from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


def process_placeholder(s: str, config: dict[str, Any], epoch_num: str | int | None) -> str:
    """
    Replace placeholders in a string with values from the config.
    - ${name} or $JOB_NAME: The name of the job, specified by "name" in the config.
    - $EPOCH_NUM:           The epoch number, given by the epoch_num argument.
    - $PROJECT_ROOT:        The root directory of the project.
    """
    job_name = config.get("name", "") or ""
    if epoch_num is not None:
        s = s.replace("${epoch_num}", str(epoch_num))
    # Support both ${name}
    s = s.replace("${name}", job_name)
    s = s.replace("${project_root}", str(PROJECT_ROOT))
    return s


def resolve_output_dir(config: dict[str, Any]) -> Path:
    """Resolve the output_dir from the config, expanding ${name} placeholders."""
    raw = config.get("output_dir", "outputs/${name}")
    return Path(process_placeholder(raw, config=config, epoch_num=None))


def get_checkpoints_path(config: dict[str, Any], epoch_num: str | int) -> Path:
    """
    Get the path to the checkpoints file based on the training config and epoch number.

    Args:
        config:    The training config.
        epoch_num: The epoch number, or 'best'.

    Returns:
        The path to the checkpoints file.
    """
    training_params = config["training"]
    checkpoints_filename = training_params.get(
        "checkpoints_filename", "checkpoint_epoch_${epoch_num}.pt"
    )
    checkpoints_filename = process_placeholder(
        s=checkpoints_filename, config=config, epoch_num=epoch_num
    )

    checkpoints_dir = training_params.get(
        "checkpoints_dir", str(resolve_output_dir(config) / "checkpoints")
    )
    checkpoints_dir = process_placeholder(s=checkpoints_dir, config=config, epoch_num=epoch_num)

    return Path(checkpoints_dir) / checkpoints_filename


def get_best_checkpoint_path(config: dict[str, Any]) -> Path:
    """Return the path for the best checkpoint (epoch_num='best')."""
    return get_checkpoints_path(config=config, epoch_num="best")


def save_checkpoint(
    checkpoint_dict: dict[str, Any],
    config: dict[str, Any],
    epoch_num: int | str,
) -> Path:
    """
    Save the checkpoint to the specified path.

    Args:
        checkpoint_dict: The checkpoint dictionary to save.
        config:          The training config.
        epoch_num:       The epoch number, or 'best'.

    Returns:
        The path to which the checkpoint was saved.
    """
    path = get_checkpoints_path(config=config, epoch_num=epoch_num)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_dict, path)
    return path


def load_checkpoint(
    config: dict[str, Any],
    epoch_num: int | str,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """
    Load a checkpoint from disk.

    Args:
        config:       The training config.
        epoch_num:    The epoch number, or 'best'.
        map_location: torch map_location for torch.load.

    Returns:
        The checkpoint dictionary.
    """
    path = get_checkpoints_path(config=config, epoch_num=epoch_num)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location)
