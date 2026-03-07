from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset

from .feature_transform import FeatureTransform


def _apply_transform(x: torch.Tensor, transform_cfg: Any) -> torch.Tensor:
    """Apply a single transform config block to a tensor."""
    if transform_cfg is None:
        return x

    # Transform config can be:
    #   transform: normal          (bare string, kwargs at sibling level)
    #   transform:
    # type: log_normal
    #     kwargs: {mean: ..., std: ...}
    if isinstance(transform_cfg, str):
        # bare string — no kwargs; caller must have merged kwargs separately
        t_type = transform_cfg
        kwargs = {}
    else:
        t_type = transform_cfg.get("type")
        kwargs = transform_cfg.get("kwargs", {}) or {}

    fn = getattr(FeatureTransform, t_type)
    return fn(x, **kwargs)


def _get_column(df: pd.DataFrame, feature: str, idx: int) -> pd.Series:
    """Fetch (feature, str(idx)) from a MultiIndex DataFrame, raising clearly if missing."""
    key = (feature, str(idx))
    if key not in df.columns:
        raise KeyError(
            f"Column {key} not found in DataFrame. "
            f"Available columns: {list(df.columns[:10])}..."
        )
    return df[key]


class GroupConfig:
    """Pre-parsed configuration for one input group."""

    def __init__(self, cfg: dict):
        self.name: str = cfg["name"]
        self.idx: list[int] = cfg["idx"]  # token indices, len = N

        # mask: list of {name, min, max}
        self.masks: list[dict] = cfg.get("mask") or []

        # continuous features: list of {name, transform (dict or str), kwargs (optional)}
        self.continuous: list[dict] = cfg.get("continuous_features") or []

        # discrete features: list of {name, transform: {type: digitize, kwargs: {bins}}}
        self.discrete: list[dict] = cfg.get("discrete_features") or []

    @property
    def n_continuous(self) -> int:
        return len(self.continuous)

    def discrete_num_bins(self) -> list[int]:
        """Number of embedding bins per discrete feature (len(bins) + 1)."""
        result = []
        for feat in self.discrete:
            bins = feat["transform"]["kwargs"]["bins"]
            result.append(len(bins) + 1)
        return result


class JetDataset(Dataset):
    """PyTorch Dataset for jet classification from a MultiIndex DataFrame.

    The DataFrame is expected to have pd.MultiIndex columns of the form
    (feature_name, str(idx)), e.g. ("bbFatJetPt", "0").

    Each __getitem__ returns a dict:
        {
            group_name: {
                "continuous": Tensor(N, D_cont),   # transformed
                "discrete":   Tensor(N, D_disc),   # raw bin indices (long)
                "mask":       Tensor(N,),           # bool, True = invalid
            },
            ...
            "label": Tensor scalar (long)
        }

    Args:
        df:          MultiIndex DataFrame.
        config_path: Path to the YAML config file.
        label_col:   Column name (or MultiIndex key) for the class label.
                     Pass None to omit labels (inference mode).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config_path: str,
        label_col: str | None = "label",
    ):
        super().__init__()

        with Path(config_path).open() as f:
            self.cfg = yaml.safe_load(f)

        self.df = df.reset_index(drop=True)
        self.label_col = label_col
        self.classes: list[str] = [
            c["name"] for c in self.cfg["dataset"]["classes"]
        ]  # fix: was cfg["dataset"]["classes"] as strings
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.groups: list[GroupConfig] = [GroupConfig(g) for g in self.cfg["inputs"]]

        # Sample weight key (optional)
        self.weight_key: str | None = self.cfg["dataset"].get("weight_key")

        self._validate_columns()

    def _validate_columns(self):
        for grp in self.groups:
            for idx in grp.idx:
                for feat in grp.continuous:
                    _get_column(self.df, feat["name"], idx)
                for feat in grp.discrete:
                    _get_column(self.df, feat["name"], idx)
                for m in grp.masks:
                    _get_column(self.df, m["name"], idx)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict[str, Any]:
        row = self.df.iloc[i]
        sample: dict[str, Any] = {}

        for grp in self.groups:
            sample[grp.name] = self._process_group(row, grp)

        if self.label_col is not None:
            label_val = row[self.label_col]
            if isinstance(label_val, str):
                label_val = self.class_to_idx[label_val]
            sample["label"] = torch.tensor(int(label_val), dtype=torch.long)

        # Per-sample physics weight
        if self.weight_key is not None:
            sample["sample_weight"] = torch.tensor(float(row[self.weight_key]), dtype=torch.float32)

        return sample

    def _process_group(self, row: pd.Series, grp: GroupConfig) -> dict[str, torch.Tensor]:
        """Build the continuous, discrete, and mask tensors for one group."""
        N = len(grp.idx)

        # --- Validity mask (B=1 squeezed out at row level) ---
        # mask[n] = True  →  token n is INVALID
        mask = torch.zeros(N, dtype=torch.bool)

        for m_cfg in grp.masks:
            for n, idx in enumerate(grp.idx):
                val = float(
                    _get_column(self.df, m_cfg["name"], idx).iloc[0]
                    if isinstance(row, pd.DataFrame)
                    else row[(m_cfg["name"], str(idx))]
                )
                lo = m_cfg.get("min", float("-inf"))
                hi = m_cfg.get("max", float("inf"))
                # Convert "inf" strings from YAML
                lo = float(lo) if lo != "inf" else float("inf")
                hi = float(hi) if hi != "-inf" else float("-inf")
                # Valid if lo < val < hi  →  invalid (mask=True) otherwise
                if not (lo < val < hi):
                    mask[n] = True

        # --- Continuous features ---
        cont_tensors = []  # each is (N,)
        for feat in grp.continuous:
            vals = torch.tensor(
                [float(row[(feat["name"], str(idx))]) for idx in grp.idx],
                dtype=torch.float32,
            )
            # Resolve transform (handle YAML quirk where kwargs sit at sibling level)
            t_cfg = feat.get("transform")
            if isinstance(t_cfg, str):
                # e.g.  transform: normal  with  kwargs: {mean: ..., std: ...}
                kwargs = feat.get("kwargs", {}) or {}
                vals = getattr(FeatureTransform, t_cfg)(vals, **kwargs)
            elif t_cfg is not None:
                vals = _apply_transform(vals, t_cfg)
            cont_tensors.append(vals)

        if cont_tensors:
            continuous = torch.stack(cont_tensors, dim=-1)  # (N, D_cont)
        else:
            continuous = torch.zeros(N, 0, dtype=torch.float32)

        # --- Discrete features ---
        disc_tensors = []  # each is (N,)
        for feat in grp.discrete:
            vals = torch.tensor(
                [float(row[(feat["name"], str(idx))]) for idx in grp.idx],
                dtype=torch.float32,
            )
            t_cfg = feat.get("transform")
            vals = _apply_transform(vals, t_cfg).long()  # (N,) long bin indices
            disc_tensors.append(vals)

        if disc_tensors:
            discrete = torch.stack(disc_tensors, dim=-1)  # (N, D_disc)
        else:
            discrete = torch.zeros(N, 0, dtype=torch.long)

        return {
            "continuous": continuous,  # (N, D_cont)
            "discrete": discrete,  # (N, D_disc)
            "mask": mask,  # (N,)  True = invalid
        }

    def group_configs(self) -> list[GroupConfig]:
        """Return parsed group configs (useful for building InputEmbedding modules)."""
        return self.groups

    def num_classes(self) -> int:
        return len(self.classes)
