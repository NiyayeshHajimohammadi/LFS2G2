"""Prefix decoder: Patch-ioner/DeCap style, extended from a single prefix token to a prefix *set*.

The conditioning tokens are projected and prepended as a (bidirectional) prefix; the graph tokens
attend to the whole prefix plus their causal past. A decoder-only baseline against the
cross-attention variant.
"""
#My comment: after encoder-> ConditioningSet.tokens [B,N,D_cond]-> LinearProjection [B,N,D_model] (concat) Graph tokens [B,T,D_model]: [B,N+T,D_model->Masked TransformerEncoder-> Keep graph positions only[B,T,D_model]->LayerNorm + vocabulary head [B, T, vocab_size] 
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

    def _attn_mask(self, n_prefix: int, t: int, device) -> torch.Tensor:#My comment: constructs the structural attention mask
        """Boolean mask, True = *disallowed* (matches the boolean key-padding mask dtype)."""
        L = n_prefix + t #My comment: Computes the length of the concatenated sequence L is total tokens, n_prefix is conditoining tokens, T is graph tokens
        mask = torch.ones((L, L), dtype=torch.bool, device=device)  # default disallow ->My comment: Nobody may attend to anything.
        mask[:n_prefix, :n_prefix] = False           # prefix attends among prefix -> My comment: This opens prefix-to-prefix attention.
        #My comment: Rows 0...N-1-> prefix queries Columns 0...N-1-> prefix keys=> Every prefix token can attend to every prefix token.
        mask[n_prefix:, :n_prefix] = False           # tokens attend to all prefix -> My comment: This opens graph-to-prefix attention.
        #My comment: Rows N...L-1-> Graph-token queries Columns 0...N-1 -> prefix keys=>every graph token can use the full conditioning set.
        causal = torch.triu(torch.ones((t, t), dtype=torch.bool, device=device), diagonal=1)#My comment: builds a standard upper-triangular causal mask for the graph-token region.
        #My comment: Positions above the diagonal are True, so a graph token cannot see later graph positions.
        mask[n_prefix:, n_prefix:] = causal          # tokens attend causally to tokens 
        #My comment: Places the causal matrix into the graph-to-graph section.
        return mask

    def _hidden(self, cond: ConditioningSet, tokens: torch.Tensor) -> torch.Tensor:#My comment: cond is shaped [B,N,D_cond], tokens is shaped [B,T]-output-> [B,T,D_model]
        prefix = self.cond_proj(cond.tokens)  # [B, N, d] (no positional -> set)
        tgt = self._embed_tokens(tokens)      # [B, T, d]
        N, T = prefix.shape[1], tgt.shape[1] #My commnt: prefix is shaped [B, N, d_model] tgt is shaped [B,T,d_model] and x is shaped [B, N+T,d_model]
        mask = self._attn_mask(N, T, x.device)#My comment: Builds the structural attention mask on the same device as x
        x = torch.cat([prefix, tgt], dim=1)

        kpm = None #My comment: key padding mask
        if cond.mask is not None:
            B = x.shape[0]
            token_valid = torch.zeros(B, T, dtype=torch.bool, device=x.device)
            kpm = torch.cat([~cond.mask, token_valid], dim=1)  # True = ignore

        out = self.encoder(x, mask=mask, src_key_padding_mask=kpm)
        return out[:, N:, :]  # token positions only 
        #My comment: The prefix hidden states are intermediate conditioning states. The project does not predict vocabulary tokens at those positions.
