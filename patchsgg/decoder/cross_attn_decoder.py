"""Transformer encoder-decoder for Pix2SG-style graph generation."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from patchsgg.decoder.base import GraphDecoder
from patchsgg.encoders.base import ConditioningSet


class _Pix2SGEncoderLayer(nn.Module):
    """DETR/Pix2SG-style encoder layer with positions added to Q/K."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_ff: int,
        dropout: float,
        norm_first: bool,
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.linear1 = nn.Linear(
            d_model,
            dim_ff,
        )

        self.linear2 = nn.Linear(
            dim_ff,
            d_model,
        )

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.activation = nn.ReLU()

        self.norm_first = bool(
            norm_first
        )

    @staticmethod
    def _with_pos(
        x: torch.Tensor,
        pos: torch.Tensor | None,
    ) -> torch.Tensor:

        if pos is None:
            return x

        return x + pos

    def forward(
        self,
        src: torch.Tensor,
        *,
        src_key_padding_mask: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
    ) -> torch.Tensor:

        # --------------------------------------------------------------
        # Pre-norm path retained for compatibility with other configs.
        # Pix2SG uses norm_first=False.
        # --------------------------------------------------------------
        if self.norm_first:
            src_norm = self.norm1(src)

            q = k = self._with_pos(
                src_norm,
                pos,
            )

            attn = self.self_attn(
                q,
                k,
                src_norm,
                key_padding_mask=src_key_padding_mask,
                need_weights=False,
            )[0]

            src = (
                src
                + self.dropout1(attn)
            )

            src_norm = self.norm2(src)

            ff = self.linear2(
                self.dropout(
                    self.activation(
                        self.linear1(src_norm)
                    )
                )
            )

            return (
                src
                + self.dropout2(ff)
            )

        # --------------------------------------------------------------
        # Pix2SG / DETR post-norm path.
        # --------------------------------------------------------------
        q = k = self._with_pos(
            src,
            pos,
        )

        attn = self.self_attn(
            q,
            k,
            src,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )[0]

        src = self.norm1(
            src
            + self.dropout1(attn)
        )

        ff = self.linear2(
            self.dropout(
                self.activation(
                    self.linear1(src)
                )
            )
        )

        src = self.norm2(
            src
            + self.dropout2(ff)
        )

        return src


class _Pix2SGDecoderLayer(nn.Module):
    """Pix2SG decoder layer with graph-query and image positions."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_ff: int,
        dropout: float,
        norm_first: bool,
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.cross_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.linear1 = nn.Linear(
            d_model,
            dim_ff,
        )

        self.linear2 = nn.Linear(
            dim_ff,
            d_model,
        )

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.activation = nn.ReLU()

        self.norm_first = bool(
            norm_first
        )

    @staticmethod
    def _with_pos(
        x: torch.Tensor,
        pos: torch.Tensor | None,
    ) -> torch.Tensor:

        if pos is None:
            return x

        return x + pos

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        *,
        tgt_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        pos: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
    ) -> torch.Tensor:

        # --------------------------------------------------------------
        # Pre-norm compatibility path.
        # --------------------------------------------------------------
        if self.norm_first:
            tgt_norm = self.norm1(tgt)

            q = k = self._with_pos(
                tgt_norm,
                query_pos,
            )

            attn = self.self_attn(
                q,
                k,
                tgt_norm,
                attn_mask=tgt_mask,
                need_weights=False,
            )[0]

            tgt = (
                tgt
                + self.dropout1(attn)
            )

            tgt_norm = self.norm2(tgt)

            attn = self.cross_attn(
                self._with_pos(
                    tgt_norm,
                    query_pos,
                ),
                self._with_pos(
                    memory,
                    pos,
                ),
                memory,
                key_padding_mask=memory_key_padding_mask,
                need_weights=False,
            )[0]

            tgt = (
                tgt
                + self.dropout2(attn)
            )

            tgt_norm = self.norm3(tgt)

            ff = self.linear2(
                self.dropout(
                    self.activation(
                        self.linear1(tgt_norm)
                    )
                )
            )

            return (
                tgt
                + self.dropout3(ff)
            )

        # --------------------------------------------------------------
        # Pix2SG / DETR post-norm decoder path.
        # --------------------------------------------------------------

        # Self-attention:
        #
        # q = graph features + graph positional embedding
        # k = graph features + graph positional embedding
        # v = graph features
        q = k = self._with_pos(
            tgt,
            query_pos,
        )

        attn = self.self_attn(
            q,
            k,
            tgt,
            attn_mask=tgt_mask,
            need_weights=False,
        )[0]

        tgt = self.norm1(
            tgt
            + self.dropout1(attn)
        )

        # Cross-attention:
        #
        # query = graph features + graph positions
        # key   = visual memory + image 2D positions
        # value = visual memory
        attn = self.cross_attn(
            self._with_pos(
                tgt,
                query_pos,
            ),
            self._with_pos(
                memory,
                pos,
            ),
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )[0]

        tgt = self.norm2(
            tgt
            + self.dropout2(attn)
        )

        ff = self.linear2(
            self.dropout(
                self.activation(
                    self.linear1(tgt)
                )
            )
        )

        tgt = self.norm3(
            tgt
            + self.dropout3(ff)
        )

        return tgt


class CrossAttnDecoder(GraphDecoder):
    def __init__(
        self,
        n_layers=2,
        n_encoder_layers=0,
        n_heads=8,
        dim_ff=2048,
        dropout=0.1,
        norm_first=True,
        use_2d_positional_encoding=False,
        **kw,
    ):
        super().__init__(**kw)

        self.use_2d_positional_encoding = bool(
            use_2d_positional_encoding
        )

        # --------------------------------------------------------------
        # Visual Transformer encoder.
        #
        # location_free_paper.yaml:
        #
        #   n_encoder_layers = 6
        # --------------------------------------------------------------
        self.encoder_layers = nn.ModuleList(
            [
                _Pix2SGEncoderLayer(
                    d_model=self.d_model,
                    n_heads=n_heads,
                    dim_ff=dim_ff,
                    dropout=dropout,
                    norm_first=norm_first,
                )
                for _ in range(
                    int(n_encoder_layers)
                )
            ]
        )

        # --------------------------------------------------------------
        # Autoregressive graph Transformer decoder.
        #
        # location_free_paper.yaml:
        #
        #   n_layers = 2
        # --------------------------------------------------------------
        self.decoder_layers = nn.ModuleList(
            [
                _Pix2SGDecoderLayer(
                    d_model=self.d_model,
                    n_heads=n_heads,
                    dim_ff=dim_ff,
                    dropout=dropout,
                    norm_first=norm_first,
                )
                for _ in range(
                    int(n_layers)
                )
            ]
        )

    def _image_position_encoding(
        self,
        cond: ConditioningSet,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """
        Build normalized DETR-style 2D sine positional embeddings.

        Output:
            [B, N_patches, d_model]
        """

        if not self.use_2d_positional_encoding:
            return None

        batch_size, n_tokens, _ = cond.tokens.shape

        # CLIP ViT-L/14 at 224x224 gives:
        #
        #   16 x 16 = 256 patch tokens
        side = int(
            math.isqrt(n_tokens)
        )

        if side * side != n_tokens:
            raise ValueError(
                "2D positional encoding requires a square "
                f"patch grid; got {n_tokens} visual tokens"
            )

        if self.d_model % 4 != 0:
            raise ValueError(
                "DETR-style 2D sine positional encoding "
                "requires d_model divisible by 4; "
                f"got {self.d_model}"
            )

        # ConditioningSet.mask:
        #
        #   True = valid
        #
        # DETR position construction wants valid locations.
        if cond.mask is None:
            valid = torch.ones(
                (
                    batch_size,
                    side,
                    side,
                ),
                dtype=torch.bool,
                device=cond.tokens.device,
            )

        else:
            valid = cond.mask.reshape(
                batch_size,
                side,
                side,
            )

        # --------------------------------------------------------------
        # Cumulative x/y coordinates.
        # --------------------------------------------------------------
        y_embed = valid.cumsum(
            1,
            dtype=torch.float32,
        )

        x_embed = valid.cumsum(
            2,
            dtype=torch.float32,
        )

        # --------------------------------------------------------------
        # Normalize coordinates to [0, 2*pi].
        # --------------------------------------------------------------
        eps = 1e-6
        scale = 2.0 * math.pi

        y_embed = (
            y_embed
            / (
                y_embed[:, -1:, :]
                + eps
            )
            * scale
        )

        x_embed = (
            x_embed
            / (
                x_embed[:, :, -1:]
                + eps
            )
            * scale
        )

        # Half the hidden dimension is allocated to x and half to y.
        num_pos_feats = (
            self.d_model // 2
        )

        dim_t = torch.arange(
            num_pos_feats,
            dtype=torch.float32,
            device=cond.tokens.device,
        )

        dim_t = 10000 ** (
            2
            * torch.div(
                dim_t,
                2,
                rounding_mode="floor",
            )
            / num_pos_feats
        )

        pos_x = (
            x_embed[..., None]
            / dim_t
        )

        pos_y = (
            y_embed[..., None]
            / dim_t
        )

        pos_x = torch.stack(
            (
                pos_x[..., 0::2].sin(),
                pos_x[..., 1::2].cos(),
            ),
            dim=-1,
        ).flatten(-2)

        pos_y = torch.stack(
            (
                pos_y[..., 0::2].sin(),
                pos_y[..., 1::2].cos(),
            ),
            dim=-1,
        ).flatten(-2)

        pos = torch.cat(
            (
                pos_y,
                pos_x,
            ),
            dim=-1,
        )

        pos = pos.reshape(
            batch_size,
            n_tokens,
            self.d_model,
        )

        return pos.to(
            dtype=dtype
        )

    def _hidden(
        self,
        cond: ConditioningSet,
        tokens: torch.Tensor,
    ) -> torch.Tensor:

        # --------------------------------------------------------------
        # Project visual features into Pix2SG hidden dimension.
        #
        # e.g.
        #
        # CLIP dimension -> 256
        # --------------------------------------------------------------
        memory = self.cond_proj(
            cond.tokens
        )

        # PyTorch convention:
        #
        # True = ignore this token.
        mem_kpm = (
            None
            if cond.mask is None
            else ~cond.mask
        )

        # --------------------------------------------------------------
        # DETR/Pix2SG 2D visual positional encoding.
        # --------------------------------------------------------------
        image_pos = self._image_position_encoding(
            cond,
            dtype=memory.dtype,
        )

        # --------------------------------------------------------------
        # Six-layer visual Transformer encoder.
        # --------------------------------------------------------------
        for layer in self.encoder_layers:
            memory = layer(
                memory,
                src_key_padding_mask=mem_kpm,
                pos=image_pos,
            )

        # --------------------------------------------------------------
        # Graph-token embeddings.
        #
        # _embed_tokens() already adds the learned sequence-position
        # embeddings to the token embeddings.
        # --------------------------------------------------------------
        tgt = self._embed_tokens(
            tokens
        )

        # --------------------------------------------------------------
        # Pix2SG also supplies those learned sequence positions explicitly
        # as query_pos during Transformer attention.
        # --------------------------------------------------------------
        positions = torch.arange(
            tokens.shape[1],
            device=tokens.device,
        )

        query_pos = self.pos_embed(
            positions
        )[None].expand(
            tokens.shape[0],
            -1,
            -1,
        )

        # --------------------------------------------------------------
        # Prevent token t from seeing future graph tokens.
        # --------------------------------------------------------------
        tgt_mask = self.causal_mask(
            tokens.shape[1],
            tokens.device,
        )

        # --------------------------------------------------------------
        # Two-layer autoregressive graph decoder.
        # --------------------------------------------------------------
        for layer in self.decoder_layers:
            tgt = layer(
                tgt,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=mem_kpm,
                pos=image_pos,
                query_pos=query_pos,
            )

        return tgt