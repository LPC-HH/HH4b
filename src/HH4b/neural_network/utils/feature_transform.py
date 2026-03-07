from __future__ import annotations

import torch


class FeatureTransform:
    @staticmethod
    def normal(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
        """Standardization (Z-score scaling). Best for Gaussian-like vars (e.g., eta)."""
        if std == 0:
            raise ValueError("Standard deviation cannot be zero for standardization.")
        return (x - mean) / std

    @staticmethod
    def log_normal(x: torch.Tensor, mean: float, std: float, clamp_min: float = 0) -> torch.Tensor:
        """Log-Normal scaling. Essential for high-pT tails and mass."""
        # Adding 1 to avoid log(0) if variable can be 0
        if std == 0:
            raise ValueError("Standard deviation cannot be zero for standardization.")
        clamp_min = max(clamp_min, 1e-12)  # Ensure clamp_min is non-negative for log scaling
        x = torch.clamp(x, min=clamp_min)
        return (torch.log(x) - mean) / std

    @staticmethod
    def min_max(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
        """Uniform scaling. Best for bounded variables (e.g., neural scores 0-1)."""
        if max_val == min_val:
            raise ValueError("Max and min values cannot be the same for min-max scaling.")
        return (x - min_val) / (max_val - min_val)

    @staticmethod
    def digitize(x: torch.Tensor, bins: list) -> torch.Tensor:
        """
        Categorical binning.
        Maps x to index based on [bins[i], bins[i+1]).
        Example: bins=[0.8, 0.9, 1.0] ->
        x < 0.8: 0 | 0.8 <= x < 0.9: 1 | 0.9 <= x < 1.0: 2 | x >= 1.0: 3
        """
        return torch.bucketize(x, torch.tensor(bins))
