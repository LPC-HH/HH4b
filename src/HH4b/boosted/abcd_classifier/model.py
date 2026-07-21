"""
ABCDClassifier MLP definition.  See notes/ABCDnn.md Task 3.

Tabular MLP, 6-way logit output:

    Linear(d_in    → hidden) → BatchNorm1d → ReLU → Dropout
    [Linear(hidden → hidden) → BatchNorm1d → ReLU → Dropout] × (num_hidden_layers - 1)
    Linear(hidden  → n_classes)

Architecture is configurable via constructor / CLI (--hidden,
--num-hidden-layers).  Default 5×512 ≈ 1.1M params.
"""  # noqa: RUF002

from __future__ import annotations

import torch
from torch import nn


class ABCDClassifier(nn.Module):
    """6-way classifier for {B, C, D} × {data, ttbar}.

    Inputs are standardized event-level features (Task 2's NPZ).  Outputs
    are raw logits; ``softmax`` is applied at inference time when computing
    the per-event transfer factor + purity correction.

    Parameters
    ----------
    d_in
        Input feature dimensionality (13 for the strict feature set).
    hidden
        Width of each hidden layer.
    n_classes
        Output dim; fixed at 6 by the §1.2 label encoding.
    dropout
        Dropout probability applied after each ReLU.
    num_hidden_layers
        Number of ``Linear → BN → ReLU → Dropout`` hidden blocks before the
        final classification layer.
    """  # noqa: RUF002

    def __init__(
        self,
        d_in: int,
        hidden: int = 512,
        n_classes: int = 6,
        dropout: float = 0.2,
        num_hidden_layers: int = 5,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.n_classes = n_classes
        self.hidden = hidden
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout

        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be >= 1")

        layers: list[nn.Module] = []
        in_dim = d_in
        for _ in range(num_hidden_layers):
            layers += [
                nn.Linear(in_dim, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            in_dim = hidden
        layers.append(nn.Linear(hidden, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape ``(batch, n_classes)``."""
        return self.net(x)

    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
