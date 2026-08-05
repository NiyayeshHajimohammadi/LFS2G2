"""Default decoder: a Transformer decoder that cross-attends to the conditioning token set.

This is the centrepiece for the bottleneck / instance-binding hypothesis -- the decoder attends to
the full ``[B, N, D]`` token set (image patches at inference) instead of a single pooled vector.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from patchsgg.decoder.base import GraphDecoder #My comment: It provides all shared decoder functionality
from patchsgg.encoders.base import ConditioningSet


class CrossAttnDecoder(GraphDecoder):
    def __init__(self, n_layers=6, n_heads=8, dim_ff=2048, dropout=0.1, **kw):
        super().__init__(**kw)
        layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)

    def _hidden(self, cond: ConditioningSet, tokens: torch.Tensor) -> torch.Tensor:
        memory = self.cond_proj(cond.tokens)  # [B, N, d]
        tgt = self._embed_tokens(tokens)      # [B, T, d]
        tgt_mask = self.causal_mask(tokens.shape[1], tokens.device)
        mem_kpm = None if cond.mask is None else ~cond.mask  # True = ignore
        return self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=mem_kpm,
        )
