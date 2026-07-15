from __future__ import annotations

import torch
import torch.nn as nn


class DiscreteFeatureEmbedding(nn.Module):
    """Embedding lookup for one discrete (digitized) feature.

    Args:
        num_bins:   Number of bins output by digitize (len(bins) + 1).
        dim:        Output embedding dimension.
    """

    def __init__(self, num_bins: int, dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_bins, dim)
        nn.init.trunc_normal_(self.embedding.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N) long tensor of bin indices.
        Returns:
            (B, N, dim)
        """
        return self.embedding(x)


class InputEmbedding(nn.Module):
    """Embeds one input group (e.g. bbFatJet0, AK4JetAway) into (B, N, dim).

    Processing:
        1. Project continuous features linearly: (B, N, D_cont) → (B, N, dim)
        2. For each discrete feature, look up embedding: (B, N) → (B, N, dim)
        3. Sum all embeddings (continuous projection + each discrete embedding)
        4. Apply LayerNorm
        5. Return tokens (B, N, dim) and validity mask (B, N)

    The mask (True = invalid) is derived from the raw continuous feature values
    BEFORE transformation, using the configured min/max bounds per mask field.

    Args:
        n_continuous:       Number of continuous input features (D_cont).
        dim:                Model embedding dimension.
        discrete_num_bins:  List of num_bins per discrete feature, in order.
        dropout:            Dropout applied after summing embeddings.
    """

    def __init__(
        self,
        n_continuous: int,
        dim: int,
        discrete_num_bins: list[int] | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.continuous_proj = nn.Linear(n_continuous, dim) if n_continuous > 0 else None

        discrete_num_bins = discrete_num_bins or []
        self.discrete_embeddings = nn.ModuleList(
            [DiscreteFeatureEmbedding(num_bins, dim) for num_bins in discrete_num_bins]
        )

        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        group: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            group: dict with keys:
                'continuous'  - (B, N, D_cont) float, already transformed
                'discrete'    - (B, N, D_disc) long, one column per discrete feature
                                (may be absent if no discrete features)
                'mask'        - (B, N) bool, True = invalid/padded token
                                (may be absent if no mask defined)

        Returns:
            tokens: (B, N, dim)
            mask:   (B, N) bool — True = invalid
        """
        B, N = None, None

        # --- Continuous projection ---
        out = None
        if "continuous" in group and self.continuous_proj is not None:
            cont = group["continuous"]  # (B, N, D_cont)
            B, N, _ = cont.shape
            out = self.continuous_proj(cont)  # (B, N, dim)

        # --- Discrete embeddings (additive) ---
        if "discrete" in group and len(self.discrete_embeddings) > 0:
            disc = group["discrete"]  # (B, N, D_disc)
            if B is None:
                B, N, _ = disc.shape
            for i, emb in enumerate(self.discrete_embeddings):
                d = emb(disc[..., i])  # (B, N, dim)
                out = d if out is None else out + d

        if out is None:
            raise ValueError("InputEmbedding received no continuous or discrete features.")

        out = self.drop(self.norm(out))  # (B, N, dim)

        # --- Mask ---
        if "mask" in group:
            mask = group["mask"]  # (B, N) bool, True = invalid
        else:
            mask = torch.zeros(B, N, dtype=torch.bool, device=out.device)

        # Zero out embeddings for invalid tokens so they don't carry signal
        out = out * (~mask).unsqueeze(-1).float()

        return out, mask
