"""Prefix decoder with a bidirectional conditioning set and causal graph tokens."""
from __future__ import annotations

import torch
import torch.nn as nn

from patchsgg.decoder.base import GraphDecoder
from patchsgg.encoders.base import ConditioningSet


class PrefixDecoder(GraphDecoder):
    def __init__(self, n_layers=6, n_heads=8, dim_ff=2048, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    @staticmethod
    def _attn_mask(n_prefix: int, token_count: int, device) -> torch.Tensor:
        """Boolean structural mask where ``True`` means attention is disallowed."""
        total = n_prefix + token_count
        mask = torch.ones((total, total), dtype=torch.bool, device=device)

        # Conditioning tokens attend bidirectionally to all conditioning tokens.
        mask[:n_prefix, :n_prefix] = False
        # Graph tokens attend to every conditioning token.
        mask[n_prefix:, :n_prefix] = False
        # Graph tokens attend causally to graph tokens.
        mask[n_prefix:, n_prefix:] = torch.triu(
            torch.ones((token_count, token_count), dtype=torch.bool, device=device),
            diagonal=1,
        )
        return mask

    def _hidden(self, cond: ConditioningSet, tokens: torch.Tensor) -> torch.Tensor:
        prefix = self.cond_proj(cond.tokens)
        target = self._embed_tokens(tokens)
        n_prefix, token_count = prefix.shape[1], target.shape[1]

        x = torch.cat([prefix, target], dim=1)
        attention_mask = self._attn_mask(n_prefix, token_count, x.device)

        key_padding_mask = None
        if cond.mask is not None:
            token_padding = torch.zeros(
                x.shape[0],
                token_count,
                dtype=torch.bool,
                device=x.device,
            )
            key_padding_mask = torch.cat([~cond.mask.bool(), token_padding], dim=1)

        hidden = self.encoder(
            x,
            mask=attention_mask,
            src_key_padding_mask=key_padding_mask,
        )
        return hidden[:, n_prefix:, :]
