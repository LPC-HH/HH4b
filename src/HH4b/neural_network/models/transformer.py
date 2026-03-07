from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.scale * x / rms


def get_norm(norm: Literal["LayerNorm", "RMSNorm"], dim: int, eps: float = 1e-5) -> nn.Module:
    if norm == "RMSNorm":
        return RMSNorm(dim, eps=eps)
    return nn.LayerNorm(dim, eps=eps)


class FFN(nn.Module):
    """Standard two-layer FFN with GELU."""

    def __init__(self, dim: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN: (x W1) * SiLU(x W2) -> W3.

    ffn_dim is the inner gate dimension (typically 4/3 * dim to keep param count
    comparable to a standard 4*dim FFN).
    """

    def __init__(self, dim: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(dim, ffn_dim, bias=False)
        self.w3 = nn.Linear(ffn_dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w3(self.w1(x) * F.silu(self.w2(x))))


class LayerScale(nn.Module):
    """Per-channel learnable scalar applied to residual branch."""

    def __init__(self, dim: int, init_value: float = 1e-4):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * x


class TransformerEncoderLayer(nn.Module):
    """Pre-norm transformer encoder layer.

    Args:
        dim:              Model dimension.
        num_heads:        Number of attention heads.
        ffn_dim:          Inner FFN dimension.
        activation:       'GELU' or 'SwiGLU'.
        norm:             'LayerNorm' or 'RMSNorm'.
        layer_scale_init: Init value for LayerScale (None to disable).
        qkv_bias:         Whether to use bias in QKV projections.
        attention_dropout: Dropout on attention weights.
        norm_eps:         Epsilon for norm layers.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        activation: Literal["GELU", "SwiGLU"] = "SwiGLU",
        norm: Literal["LayerNorm", "RMSNorm"] = "LayerNorm",
        layer_scale_init: float | None = 1e-4,
        qkv_bias: bool = True,
        attention_dropout: float = 0.0,
        norm_eps: float = 1e-5,
    ):
        super().__init__()

        self.norm1 = get_norm(norm, dim, eps=norm_eps)
        self.norm2 = get_norm(norm, dim, eps=norm_eps)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attention_dropout,
            bias=qkv_bias,
            batch_first=True,
        )

        ffn_cls = SwiGLUFFN if activation == "SwiGLU" else FFN
        self.ffn = ffn_cls(dim, ffn_dim)

        self.ls1 = (
            LayerScale(dim, layer_scale_init) if layer_scale_init is not None else nn.Identity()
        )
        self.ls2 = (
            LayerScale(dim, layer_scale_init) if layer_scale_init is not None else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:                 (B, N, dim)
            key_padding_mask:  (B, N) bool — True means *ignore* that position.
        Returns:
            (B, N, dim)
        """
        # --- Self-attention (pre-norm) ---
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = residual + self.ls1(attn_out)

        # --- FFN (pre-norm) ---
        residual = x
        x = self.norm2(x)
        x = residual + self.ls2(self.ffn(x))

        return x


class TransformerEncoder(nn.Module):
    """Stack of TransformerEncoderLayers.

    Args:
        dim:                  Model dimension.
        num_layers:           Number of encoder layers.
        num_heads:            Number of attention heads.
        activation:           'GELU' or 'SwiGLU'. Default 'SwiGLU'.
        norm:                 'LayerNorm' or 'RMSNorm'. Default 'LayerNorm'.
        layer_scale_init:     LayerScale init value (None to disable). Default 1e-4.
        num_registers:        Number of learnable register tokens prepended before
                              the input tokens (à la DINOv2). Default 0.
        mlp_ratio:            FFN hidden dim multiplier (ffn_dim = mlp_ratio * dim).
                              Default 4. For SwiGLU, inner dim is scaled by 2/3 to
                              keep parameter count comparable.
        qkv_bias:             Bias in QKV projections. Default True.
        attention_dropout:    Dropout on attention weights. Default 0.0.
        norm_eps:             Epsilon for norm layers. Default 1e-5.
        apply_final_norm:     Apply a norm after the last layer. Default True.
        apply_embedding_norm: Apply a norm to the input embeddings before the first
                              layer. Default False.
    """

    def __init__(
        self,
        dim: int,
        num_layers: int,
        num_heads: int,
        activation: Literal["GELU", "SwiGLU"] = "SwiGLU",
        norm: Literal["LayerNorm", "RMSNorm"] = "LayerNorm",
        layer_scale_init: float | None = 1e-4,
        num_registers: int = 0,
        mlp_ratio: int = 4,
        qkv_bias: bool = True,
        attention_dropout: float = 0.0,
        norm_eps: float = 1e-5,
        apply_final_norm: bool = True,
        apply_embedding_norm: bool = False,
    ):
        super().__init__()

        # For SwiGLU, scale inner dim by 2/3 so parameter count matches a
        # standard 4*dim GELU FFN (SwiGLU has 3 matrices vs 2).
        if activation == "SwiGLU":
            ffn_dim = int(mlp_ratio * dim * 2 / 3)
        else:
            ffn_dim = mlp_ratio * dim

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    dim=dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    activation=activation,
                    norm=norm,
                    layer_scale_init=layer_scale_init,
                    qkv_bias=qkv_bias,
                    attention_dropout=attention_dropout,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = get_norm(norm, dim, eps=norm_eps) if apply_final_norm else nn.Identity()
        self.embedding_norm = (
            get_norm(norm, dim, eps=norm_eps) if apply_embedding_norm else nn.Identity()
        )

        # Register tokens: always valid, prepended before input tokens
        self.num_registers = num_registers
        if num_registers > 0:
            self.register_tokens = nn.Parameter(torch.zeros(1, num_registers, dim))
            nn.init.trunc_normal_(self.register_tokens, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:                (B, N, dim)
            key_padding_mask: (B, N) bool — True means ignore. Should NOT include
                              positions for register tokens; those are prepended here.
        Returns:
            (B, N, dim) — register positions are stripped, output length matches input.
        """
        B = x.size(0)
        x = self.embedding_norm(x)

        # Prepend register tokens (always valid → False in mask)
        if self.num_registers > 0:
            regs = self.register_tokens.expand(B, -1, -1)  # (B, R, dim)
            x = torch.cat([regs, x], dim=1)  # (B, R+N, dim)
            if key_padding_mask is not None:
                reg_mask = torch.zeros(B, self.num_registers, dtype=torch.bool, device=x.device)
                key_padding_mask = torch.cat([reg_mask, key_padding_mask], dim=1)

        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)

        x = self.final_norm(x)

        # Strip register tokens before returning
        if self.num_registers > 0:
            x = x[:, self.num_registers :]

        return x
