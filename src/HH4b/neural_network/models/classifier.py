from __future__ import annotations

import torch
import torch.nn as nn

from .transformer import TransformerEncoder


class JetClassifier(nn.Module):
    """Jet classifier that combines per-group embeddings, a transformer encoder,
    and a classification head.

    Architecture:
        1. Embed each input group independently → (B, N_i, dim)
        2. Concatenate all groups along the token axis → (B, N_total, dim)
        3. Prepend a learned [CLS] token → (B, 1 + N_total, dim)
        4. Run through TransformerEncoder (with combined key-padding mask)
        5. Pool: CLS output  ⊕  masked mean over non-CLS tokens → (B, 2*dim)
        6. Head MLP → (B, num_classes)

    Args:
        embeddings:  nn.ModuleDict mapping input-group name → InputEmbedding.
                     Each InputEmbedding must accept the group's batch dict and
                     return (tokens: Tensor(B, N_i, dim), mask: Tensor(B, N_i))
                     where mask is True for *invalid* (padded) tokens.
        encoder:     TransformerEncoder instance.
        head:        MLP that maps (B, 2*dim) → (B, num_classes).
        dim:         Model dimension (must match encoder & embeddings).
    """

    def __init__(
        self,
        embeddings: nn.ModuleDict,
        encoder: TransformerEncoder,
        head: nn.Module,
        dim: int,
    ):
        super().__init__()
        self.embeddings = embeddings
        self.encoder = encoder
        self.head = head
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, batch: dict[str, dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Args:
            batch: dict mapping group name → dict with keys
                   'continuous' (B, N_i, D_cont), 'discrete' (B, N_i, D_disc),
                   and optionally 'mask' (B, N_i) bool (True = invalid).

        Returns:
            logits: (B, num_classes)
        """
        tokens_list = []
        masks_list = []

        for name, embedding in self.embeddings.items():
            tokens, mask = embedding(batch[name])  # (B, N_i, dim), (B, N_i)
            tokens_list.append(tokens)
            masks_list.append(mask)

        # (B, N_total, dim)  &  (B, N_total)
        x = torch.cat(tokens_list, dim=1)
        mask = torch.cat(masks_list, dim=1)

        B = x.size(0)

        # Prepend [CLS]: always valid → False in key_padding_mask
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, dim)
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)

        x = torch.cat([cls, x], dim=1)  # (B, 1+N_total, dim)
        mask = torch.cat([cls_mask, mask], dim=1)  # (B, 1+N_total)

        # Encode
        x = self.encoder(x, key_padding_mask=mask)  # (B, 1+N_total, dim)

        # Pool ----------------------------------------------------------------
        cls_out = x[:, 0]  # (B, dim)

        # Masked mean over non-CLS tokens (mask: True = invalid → exclude)
        token_out = x[:, 1:]  # (B, N_total, dim)
        valid_mask = ~mask[:, 1:]  # (B, N_total)  True = valid
        token_out = token_out * valid_mask.unsqueeze(-1).float()
        denom = valid_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        mean_out = token_out.sum(dim=1) / denom  # (B, dim)

        pooled = torch.cat([cls_out, mean_out], dim=-1)  # (B, 2*dim)

        return self.head(pooled)  # (B, num_classes)


def build_mlp_head(
    in_dim: int,
    num_classes: int,
    hidden_dim: list[int] | None = None,
    num_hidden_layers: int = 1,
    batch_norm: bool = False,
    dropout: float = 0.0,
) -> nn.Sequential:
    dims = hidden_dim if hidden_dim is not None else [in_dim] * num_hidden_layers
    layers: list[nn.Module] = []
    current_dim = in_dim
    use_dropout = dropout > 0.0
    for h in dims:
        layers.append(nn.Linear(current_dim, h))
        if batch_norm:
            layers.append(nn.BatchNorm1d(h))
        layers.append(nn.GELU())
        if use_dropout:
            layers.append(nn.Dropout(dropout))
        current_dim = h
    layers.append(nn.Linear(current_dim, num_classes))
    return nn.Sequential(*layers)
