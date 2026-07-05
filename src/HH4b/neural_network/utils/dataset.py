from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset

from .feature_transform import FeatureTransform


def _apply_transform(x: torch.Tensor, transform_cfg: Any) -> torch.Tensor:
    """Apply a single transform config block to a tensor."""
    if transform_cfg is None:
        return x

    if isinstance(transform_cfg, str):
        t_type = transform_cfg
        kwargs = {}
    else:
        t_type = transform_cfg.get("type")
        kwargs = transform_cfg.get("kwargs", {}) or {}

    fn = getattr(FeatureTransform, t_type)
    return fn(x, **kwargs)


class GroupConfig:
    """Pre-parsed configuration for one input group."""

    def __init__(self, cfg: dict):
        self.name: str = cfg["name"]
        self.idx: list[int] = cfg["idx"]

        self.masks: list[dict] = cfg.get("mask") or []
        self.continuous: list[dict] = cfg.get("continuous_features") or []
        self.discrete: list[dict] = cfg.get("discrete_features") or []

    @property
    def n_continuous(self) -> int:
        return len(self.continuous)

    def discrete_num_bins(self) -> list[int]:
        result = []
        for feat in self.discrete:
            bins = feat["transform"]["kwargs"]["bins"]
            result.append(len(bins) + 1)
        return result


# ---------------------------------------------------------------------------
# Collate — runs in the DataLoader worker (or main process if num_workers=0)
# ---------------------------------------------------------------------------


def jet_collate_fn(
    batch: list[dict[str, Any]],
    group_configs: list[GroupConfig],
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Stack per-sample raw tensors into (B, N, D) batches and apply all
    feature transforms in one vectorised pass.

    PyTorch's DataLoader runs ``collate_fn`` inside each worker subprocess
    when ``num_workers > 0``.  Forked workers cannot initialise CUDA, so
    pass ``device=None`` (or a CPU device) when ``num_workers > 0`` and rely
    on ``pin_memory`` + downstream ``.to(device)`` (or ``accelerator.prepare``)
    for the H2D copy.  Only pass a CUDA device when ``num_workers == 0``.

    Use via ``functools.partial``::

        from functools import partial
        collate_fn = partial(jet_collate_fn, group_configs=ds.group_configs(), device=None)
        DataLoader(ds, ..., collate_fn=collate_fn, num_workers=2, pin_memory=True)

    Args:
        batch:         List of per-sample dicts from ``JetDataset.__getitem__``.
        group_configs: Group configs from ``JetDataset.group_configs()``.
        device:        Target device for the returned tensors.  ``None`` keeps
                       tensors on CPU (rely on ``pin_memory`` for fast H2D).
                       Must be ``None`` / CPU when used with ``num_workers > 0``.
    """
    result: dict[str, Any] = {}

    # --- Labels & sample weights (already long / float scalars) ---
    result["label"] = torch.stack([b["label"] for b in batch])  # (B,)
    if "sample_weight" in batch[0]:
        result["sample_weight"] = torch.stack([b["sample_weight"] for b in batch])  # (B,)

    # --- Groups ---
    for grp in group_configs:
        continuous_raw = torch.stack([b[grp.name]["continuous"] for b in batch])  # (B, N, D_cont)
        discrete_raw = torch.stack([b[grp.name]["discrete"] for b in batch])  # (B, N, D_disc)
        mask = torch.stack([b[grp.name]["mask"] for b in batch])  # (B, N)

        if device is not None:
            continuous_raw = continuous_raw.to(device, non_blocking=True)
            discrete_raw = discrete_raw.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)

        with torch.no_grad():
            # Continuous: transform each feature slice (B, N) independently
            cont_out: list[torch.Tensor] = []
            for f_idx, feat in enumerate(grp.continuous):
                col = continuous_raw[..., f_idx]  # (B, N)
                t_cfg = feat.get("transform")
                if isinstance(t_cfg, str):
                    kwargs = feat.get("kwargs", {}) or {}
                    col = getattr(FeatureTransform, t_cfg)(col, **kwargs)
                elif t_cfg is not None:
                    col = _apply_transform(col, t_cfg)
                cont_out.append(col)

            if cont_out:
                continuous = torch.stack(cont_out, dim=-1)  # (B, N, D_cont)
            else:
                B, N = continuous_raw.shape[:2]
                continuous = torch.zeros(B, N, 0, dtype=torch.float32, device=continuous_raw.device)

            # Discrete: transform then cast to long
            disc_out: list[torch.Tensor] = []
            for f_idx, feat in enumerate(grp.discrete):
                col = discrete_raw[..., f_idx]  # (B, N) float32
                t_cfg = feat.get("transform")
                col = _apply_transform(col, t_cfg).long()  # (B, N)
                disc_out.append(col)

            if disc_out:
                discrete = torch.stack(disc_out, dim=-1)  # (B, N, D_disc)
            else:
                B, N = discrete_raw.shape[:2]
                discrete = torch.zeros(B, N, 0, dtype=torch.long, device=discrete_raw.device)

        result[grp.name] = {
            "continuous": continuous,  # (B, N, D_cont) float32, transformed
            "discrete": discrete,  # (B, N, D_disc) long, bin indices
            "mask": mask,  # (B, N) bool
        }

    return result


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class JetDataset(Dataset):
    """PyTorch Dataset for jet classification backed by numpy arrays.

    The DataFrame is immediately decomposed into per-column numpy arrays at
    construction time.  After the DataLoader forks worker processes the arrays
    are shared via OS copy-on-write rather than being deep-copied, eliminating
    the x(num_workers+1) memory blow-up that arises when workers hold
    references to a live pandas DataFrame.

    ``__getitem__`` returns *raw* (un-transformed) tensors.  All feature
    transforms are applied in ``jet_collate_fn`` on the full batch in the main
    process, enabling vectorised ops and optional GPU execution.

    Each ``__getitem__`` returns::

        {
            group_name: {
                "continuous": Tensor(N, D_cont),   # raw float32
                "discrete":   Tensor(N, D_disc),   # raw float32 (long after collate)
                "mask":       Tensor(N,),           # bool, True = invalid
            },
            ...
            "label":         LongTensor scalar,
            "sample_weight": FloatTensor scalar,   # only if weight_key is set
        }

    Typical usage::

        ds = JetDataset(df, config_path="config.yaml")
        collate_fn = partial(jet_collate_fn, group_configs=ds.group_configs(), device=device)
        loader = DataLoader(ds, batch_size=512, num_workers=4,
                            pin_memory=True, collate_fn=collate_fn)
    """

    def __init__(
        self,
        df: pd.DataFrame | None = None,
        config_path: str | None = None,
        label_col: str | None = "label",
    ):
        super().__init__()
        self._len = 0

        if config_path is not None:
            with Path(config_path).open() as f:
                self.cfg = yaml.safe_load(f)
            self.groups: list[GroupConfig] = self.build_group_configs(self.cfg)
            self.classes = [c["name"] for c in self.cfg["dataset"]["classes"]]
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
            self.weight_key = self.cfg["dataset"].get("weight_key")
            self.label_col = label_col

        if df is None:
            return  # non-main rank: valid _len=0, groups set, no data

        # --- data loading (main process only) ---
        df = df.reset_index(drop=True)
        self._len = len(df)
        needed_cols = self._collect_needed_columns()
        self._cols = {}
        for col in needed_cols:
            if col not in df.columns:
                raise KeyError(f"Column {col} not found. " f"Available: {list(df.columns[:10])}...")
            self._cols[col] = df[col].to_numpy(dtype=np.float32, na_value=np.nan)

        if label_col is not None:
            self._labels: np.ndarray | None = df[label_col].to_numpy(dtype=np.int64)
        else:
            self._labels = None

        if self.weight_key is not None:
            self._weights: np.ndarray | None = df[self.weight_key].to_numpy(dtype=np.float32)
        else:
            self._weights = None

        # --- Pre-build contiguous arrays per group for fast batch slicing ---
        self._group_arrays: dict[str, dict[str, np.ndarray]] = {}
        for grp in self.groups:
            N = len(grp.idx)
            L = self._len

            # Continuous: (L, N, D_cont)
            if grp.continuous:
                cont = np.stack(
                    [
                        np.stack(
                            [self._cols[(feat["name"], str(idx))] for feat in grp.continuous],
                            axis=-1,
                        )
                        for idx in grp.idx
                    ],
                    axis=1,
                )  # (L, N, D_cont)
            else:
                cont = np.zeros((L, N, 0), dtype=np.float32)

            # Discrete: (L, N, D_disc)
            if grp.discrete:
                disc = np.stack(
                    [
                        np.stack(
                            [self._cols[(feat["name"], str(idx))] for feat in grp.discrete],
                            axis=-1,
                        )
                        for idx in grp.idx
                    ],
                    axis=1,
                )  # (L, N, D_disc)
            else:
                disc = np.zeros((L, N, 0), dtype=np.float32)

            # Mask: (L, N) — True = invalid
            mask = np.zeros((L, N), dtype=np.bool_)
            for m_cfg in grp.masks:
                lo = m_cfg.get("min", float("-inf"))
                hi = m_cfg.get("max", float("inf"))
                lo = float(lo) if lo != "inf" else float("inf")
                hi = float(hi) if hi != "-inf" else float("-inf")
                for n, idx in enumerate(grp.idx):
                    vals = self._cols[(m_cfg["name"], str(idx))]
                    mask[:, n] |= ~((vals > lo) & (vals < hi))

            self._group_arrays[grp.name] = {
                "continuous": cont,
                "discrete": disc,
                "mask": mask,
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_needed_columns(self) -> list[tuple[str, str]]:
        """Return all (feature, str_idx) keys referenced by any group."""
        seen: set[tuple[str, str]] = set()
        out: list[tuple[str, str]] = []
        for grp in self.groups:
            for idx in grp.idx:
                sidx = str(idx)
                for feat in grp.continuous:
                    k = (feat["name"], sidx)
                    if k not in seen:
                        seen.add(k)
                        out.append(k)
                for feat in grp.discrete:
                    k = (feat["name"], sidx)
                    if k not in seen:
                        seen.add(k)
                        out.append(k)
                for m in grp.masks:
                    k = (m["name"], sidx)
                    if k not in seen:
                        seen.add(k)
                        out.append(k)
        return out

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return getattr(self, "_len", 0)

    def __getitem__(self, i: int) -> dict[str, Any]:
        sample: dict[str, Any] = {}

        for grp in self.groups:
            arrs = self._group_arrays[grp.name]
            sample[grp.name] = {
                "continuous": torch.from_numpy(arrs["continuous"][i]),  # (N, D_cont)
                "discrete": torch.from_numpy(arrs["discrete"][i]),  # (N, D_disc)
                "mask": torch.from_numpy(arrs["mask"][i]),  # (N,)
            }

        if self._labels is not None:
            sample["label"] = torch.tensor(int(self._labels[i]), dtype=torch.long)

        if self._weights is not None:
            sample["sample_weight"] = torch.tensor(float(self._weights[i]), dtype=torch.float32)

        return sample

    # ------------------------------------------------------------------
    # Helpers for model / dataloader building
    # ------------------------------------------------------------------

    def group_configs(self) -> list[GroupConfig]:
        return self.groups

    def num_classes(self) -> int:
        return len(self.classes)

    def make_collate_fn(self, device: torch.device | None = None):
        """Convenience method — returns a ready-to-use collate_fn for this dataset."""
        return partial(jet_collate_fn, group_configs=self.groups, device=device)

    @staticmethod
    def build_group_configs(config: dict) -> list[GroupConfig]:
        """Helper to extract group configs from a full model config dict."""
        return [GroupConfig(g) for g in config["inputs"]]
