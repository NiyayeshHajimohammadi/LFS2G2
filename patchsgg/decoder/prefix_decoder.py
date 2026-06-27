"""Prefix decoder: Patch-ioner/DeCap style, extended from a single prefix token to a prefix *set*.

The conditioning tokens are projected and prepended as a (bidirectional) prefix; the graph tokens
attend to the whole prefix plus their causal past. A decoder-only baseline against the
cross-attention variant.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from patchsgg.decoder.base import GraphDecoder
from patchsgg.encoders.base import ConditioningSet


class PrefixDecoder(GraphDecoder):
    def __init__(self, n_layers=6, n_heads=8, dim_ff=2048, dropout=0.1, **kw):
        super().__init__(**kw)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def _attn_mask(self, n_prefix: int, t: int, device) -> torch.Tensor:
        """Boolean mask, True = *disallowed* (matches the boolean key-padding mask dtype)."""
        L = n_prefix + t
        mask = torch.ones((L, L), dtype=torch.bool, device=device)  # default disallow
        mask[:n_prefix, :n_prefix] = False           # prefix attends among prefix
        mask[n_prefix:, :n_prefix] = False           # tokens attend to all prefix
        causal = torch.triu(torch.ones((t, t), dtype=torch.bool, device=device), diagonal=1)
        mask[n_prefix:, n_prefix:] = causal          # tokens attend causally to tokens
        return mask

    def _hidden(self, cond: ConditioningSet, tokens: torch.Tensor) -> torch.Tensor:
        prefix = self.cond_proj(cond.tokens)  # [B, N, d] (no positional -> set)
        tgt = self._embed_tokens(tokens)      # [B, T, d]
        N, T = prefix.shape[1], tgt.shape[1]
        x = torch.cat([prefix, tgt], dim=1)
        mask = self._attn_mask(N, T, x.device)

        kpm = None
        if cond.mask is not None:
            B = x.shape[0]
            token_valid = torch.zeros(B, T, dtype=torch.bool, device=x.device)
            kpm = torch.cat([~cond.mask, token_valid], dim=1)  # True = ignore

        out = self.encoder(x, mask=mask, src_key_padding_mask=kpm)
        return out[:, N:, :]  # token positions only
