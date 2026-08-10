"""Autoregressive graph decoders over the flat five-token relation vocabulary."""
from __future__ import annotations

import torch
import torch.nn as nn

from patchsgg.decoder.sampling import GenConfig, constrained_generate
from patchsgg.encoders.base import ConditioningSet
from patchsgg.graph_seq.vocab import TOKENS_PER_REL, GraphVocab


class GraphDecoder(nn.Module):
    """Shared token embeddings, output head, and generation for graph decoders."""

    def __init__(
        self,
        vocab: GraphVocab,
        cond_dim: int,
        d_model: int = 512,
        max_seq_len: int = 512,
    ):
        super().__init__()

        self.vocab = vocab
        self.d_model = int(d_model)
        self.max_seq_len = int(max_seq_len)

        # Project conditioning features from the encoder dimension into the
        # decoder hidden dimension.
        self.cond_proj = nn.Linear(cond_dim, self.d_model)

        # Scene-graph token embeddings.
        self.token_embed = nn.Embedding(vocab.vocab_size, self.d_model)

        # Learned positional embeddings used by the non-GPT-2 decoders.
        self.pos_embed = nn.Embedding(self.max_seq_len, self.d_model)

        # Final normalization before the graph-vocabulary prediction head.
        self.norm = nn.LayerNorm(self.d_model)

        # Predict one token from the complete scene-graph vocabulary.
        self.head = nn.Linear(self.d_model, vocab.vocab_size)

    def _hidden(
        self,
        cond: ConditioningSet,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Return decoder hidden states with shape ``[B, T, d_model]``."""
        raise NotImplementedError

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed graph tokens and add their positional embeddings."""
        if tokens.shape[1] > self.max_seq_len:
            raise ValueError(
                f"token sequence length {tokens.shape[1]} exceeds decoder context "
                f"{self.max_seq_len}"
            )

        positions = torch.arange(
            tokens.shape[1],
            device=tokens.device,
        )

        return (
            self.token_embed(tokens)
            + self.pos_embed(positions)[None]
        )

    def logits(
        self,
        cond: ConditioningSet,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Return graph-vocabulary logits with shape ``[B, T, V]``."""
        hidden = self._hidden(cond, tokens)
        hidden = self.norm(hidden)
        return self.head(hidden)

    def forward(
        self,
        cond: ConditioningSet,
        input_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced logits ``[B, T, V]`` aligned to target tokens."""
        return self.logits(cond, input_tokens)

    @torch.no_grad()
    def generate(
        self,
        cond: ConditioningSet,
        gen_cfg: GenConfig,
    ):
        """Generate with full-prefix recomputation for non-cached decoders."""
        required_positions = (
            1
            + int(gen_cfg.max_rels) * TOKENS_PER_REL
        )

        if required_positions > self.max_seq_len:
            raise ValueError(
                f"generation needs {required_positions} positions but decoder has "
                f"{self.max_seq_len}"
            )

        start_tokens = torch.full(
            (cond.batch_size, 1),
            self.vocab.start_token,
            dtype=torch.long,
            device=cond.tokens.device,
        )

        def step_fn(sequence: torch.Tensor) -> torch.Tensor:
            return self.logits(
                cond,
                sequence,
            )[:, -1]

        return constrained_generate(
            step_fn=step_fn,
            start_tokens=start_tokens,
            vocab=self.vocab,
            cfg=gen_cfg,
        )

    @staticmethod
    def causal_mask(
        length: int,
        device,
    ) -> torch.Tensor:
        """Create a standard additive causal attention mask."""
        return torch.triu(
            torch.full(
                (length, length),
                float("-inf"),
                device=device,
            ),
            diagonal=1,
        )


def build_decoder(
    cfg,
    vocab: GraphVocab,
    cond_dim: int,
) -> GraphDecoder:
    """Construct the decoder selected by ``cfg.decoder.type``.

    Supported decoder types:

    ``cross_attn``
        The project's custom Transformer cross-attention decoder.

    ``prefix``
        The project's prefix-conditioning Transformer decoder.

    ``gpt2_cross_attn``
        Hugging Face GPT-2 with image/text conditioning through
        cross-attention.

        GPT-2 additionally supports three training strategies:

        1. Full fine-tuning
           ``freeze_pretrained=False`` and ``lora.enabled=False``

        2. Frozen pretrained GPT-2
           ``freeze_pretrained=True`` and ``lora.enabled=False``

        3. LoRA adaptation
           ``lora.enabled=True``

        LoRA is completely optional. Configurations without a ``lora``
        section behave exactly as before, with LoRA disabled.
    """
    kind = str(cfg.decoder.type)

    # ------------------------------------------------------------------
    # Sequence-length requirements
    # ------------------------------------------------------------------
    #
    # One START token is followed by five tokens for every relation:
    #
    #   [subject, subject_instance, predicate, object, object_instance]
    #
    # The decoder must therefore be large enough for both training and
    # the requested evaluation-generation length.
    # ------------------------------------------------------------------
    minimum_seq_len = (
        1
        + max(
            vocab.max_num_rels,
            int(cfg.eval.get("max_rels", 100)),
        )
        * TOKENS_PER_REL
    )

    max_seq_len = int(
        cfg.decoder.get(
            "max_seq_len",
            minimum_seq_len,
        )
    )

    if max_seq_len < minimum_seq_len:
        raise ValueError(
            f"decoder.max_seq_len={max_seq_len} is too small; "
            f"at least {minimum_seq_len} positions are required"
        )

    # ==================================================================
    # Hugging Face GPT-2 cross-attention decoder
    # ==================================================================
    if kind == "gpt2_cross_attn":
        from patchsgg.decoder.gpt2_decoder import GPT2CrossAttnDecoder

        # --------------------------------------------------------------
        # Optional LoRA configuration
        # --------------------------------------------------------------
        #
        # Calling .get("lora", {}) keeps older configuration files fully
        # backward-compatible.
        #
        # For example, an old decoder configuration containing only:
        #
        #   decoder:
        #     type: gpt2_cross_attn
        #     freeze_pretrained: false
        #
        # results in:
        #
        #   lora_enabled = False
        #
        # and therefore behaves exactly as it did before LoRA support was
        # introduced.
        # --------------------------------------------------------------
        lora_cfg = cfg.decoder.get("lora", {})

        return GPT2CrossAttnDecoder(
            vocab=vocab,
            cond_dim=cond_dim,
            max_seq_len=max_seq_len,

            # ----------------------------------------------------------
            # Hugging Face GPT-2 source
            # ----------------------------------------------------------
            model_name=str(
                cfg.decoder.get(
                    "model_name",
                    "openai-community/gpt2",
                )
            ),
            revision=str(
                cfg.decoder.get(
                    "revision",
                    "main",
                )
            ),
            cache_dir=cfg.decoder.get(
                "cache_dir",
                None,
            ),
            local_files_only=bool(
                cfg.decoder.get(
                    "local_files_only",
                    False,
                )
            ),

            # ----------------------------------------------------------
            # GPT-2 training/runtime options
            # ----------------------------------------------------------
            gradient_checkpointing=bool(
                cfg.decoder.get(
                    "gradient_checkpointing",
                    True,
                )
            ),
            freeze_pretrained=bool(
                cfg.decoder.get(
                    "freeze_pretrained",
                    False,
                )
            ),
            tie_graph_embeddings=bool(
                cfg.decoder.get(
                    "tie_graph_embeddings",
                    True,
                )
            ),
            extend_positions=bool(
                cfg.decoder.get(
                    "extend_positions",
                    True,
                )
            ),
            dropout=float(
                cfg.decoder.get(
                    "dropout",
                    0.1,
                )
            ),

            # ----------------------------------------------------------
            # Optional LoRA configuration
            # ----------------------------------------------------------
            #
            # If decoder.lora is missing entirely, LoRA defaults to off.
            #
            # Exact target-module selection is intentionally handled
            # inside GPT2CrossAttnDecoder rather than here. The decoder
            # knows its GPT-2 module hierarchy and can therefore avoid
            # accidentally applying LoRA to:
            #
            #   - newly-created cross-attention
            #   - GPT-2 MLP layers
            #
            # unless support for those targets is explicitly added later.
            # ----------------------------------------------------------
            lora_enabled=bool(
                lora_cfg.get(
                    "enabled",
                    False,
                )
            ),
            lora_r=int(
                lora_cfg.get(
                    "r",
                    8,
                )
            ),
            lora_alpha=int(
                lora_cfg.get(
                    "alpha",
                    16,
                )
            ),
            lora_dropout=float(
                lora_cfg.get(
                    "dropout",
                    0.05,
                )
            ),
            lora_bias=str(
                lora_cfg.get(
                    "bias",
                    "none",
                )
            ),
        )

    # ==================================================================
    # Existing custom decoders
    # ==================================================================
    #
    # These decoders have nothing to do with GPT-2 or LoRA and therefore
    # remain completely unchanged.
    # ==================================================================

    common = {
        "vocab": vocab,
        "cond_dim": cond_dim,
        "d_model": int(cfg.decoder.d_model),
        "max_seq_len": max_seq_len,
    }

    # ------------------------------------------------------------------
    # Transformer cross-attention decoder
    # ------------------------------------------------------------------
    if kind == "cross_attn":
        from patchsgg.decoder.cross_attn_decoder import CrossAttnDecoder

        return CrossAttnDecoder(
            n_layers=int(cfg.decoder.n_layers),
            n_heads=int(cfg.decoder.n_heads),
            dim_ff=int(cfg.decoder.dim_ff),
            dropout=float(cfg.decoder.dropout),
            **common,
        )

    # ------------------------------------------------------------------
    # Prefix-conditioning decoder
    # ------------------------------------------------------------------
    if kind == "prefix":
        from patchsgg.decoder.prefix_decoder import PrefixDecoder

        return PrefixDecoder(
            n_layers=int(cfg.decoder.n_layers),
            n_heads=int(cfg.decoder.n_heads),
            dim_ff=int(cfg.decoder.dim_ff),
            dropout=float(cfg.decoder.dropout),
            **common,
        )

    raise ValueError(
        f"unknown decoder.type {kind!r}"
    )