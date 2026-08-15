"""Transformer encoder-decoder for Pix2SG-style graph generation."""
from __future__ import annotations

import torch
import torch.nn as nn

from patchsgg.decoder.base import GraphDecoder
from patchsgg.encoders.base import ConditioningSet


class CrossAttnDecoder(GraphDecoder):
    def __init__(
        self,
        n_layers=2,
        n_encoder_layers=0,
        n_heads=8,
        dim_ff=2048,
        dropout=0.1,
        norm_first=True,
        **kw,
    ):
        super().__init__(**kw)

        # --------------------------------------------------------------
        # Visual Transformer encoder
        # --------------------------------------------------------------
        if n_encoder_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=n_heads,
                dim_feedforward=dim_ff,
                dropout=dropout,
                activation="relu",
                batch_first=True,
                norm_first=norm_first,
            )

            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=n_encoder_layers,
            )
        else:
            # Preserve backward compatibility with old configs.
            self.encoder = None

        # --------------------------------------------------------------
        # Autoregressive graph Transformer decoder
        # --------------------------------------------------------------
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=norm_first,
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=n_layers,
        )

    def _hidden(
        self,
        cond: ConditioningSet,
        tokens: torch.Tensor,
    ) -> torch.Tensor:

        # CLIP feature dimension -> Pix2SG hidden size.
        memory = self.cond_proj(cond.tokens)

        # True means "ignore this position" for PyTorch Transformer masks.
        mem_kpm = (
            None
            if cond.mask is None
            else ~cond.mask
        )

        # --------------------------------------------------------------
        # Pix2SG visual encoder
        # --------------------------------------------------------------
        if self.encoder is not None:
            memory = self.encoder(
                memory,
                src_key_padding_mask=mem_kpm,
            )

        # --------------------------------------------------------------
        # Autoregressive graph decoder
        # --------------------------------------------------------------
        tgt = self._embed_tokens(tokens)

        tgt_mask = self.causal_mask(
            tokens.shape[1],
            tokens.device,
        )

        return self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=mem_kpm,
        )