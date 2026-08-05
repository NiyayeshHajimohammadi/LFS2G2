"""PatchSGG core model.

Pipeline:

    frozen encoder
        -> modality bridge
        -> autoregressive graph decoder
        -> graph-token loss / scene-graph predictions

The model is a plain ``nn.Module`` so it can be used by a bare PyTorch loop
or wrapped by PyTorch Lightning.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from patchsgg.bridge import build_bridge
from patchsgg.decoder import GenConfig, build_decoder
from patchsgg.encoders import build_encoders
from patchsgg.encoders.base import ConditioningSet
from patchsgg.graph_seq.vocab import (
    TOKENS_PER_REL,
    GraphVocab,
)
from patchsgg.losses import build_loss
from patchsgg.postprocess import sequences_to_predictions


def build_vocab(cfg) -> GraphVocab:
    """Build the structured scene-graph vocabulary from the config."""
    vocab_cfg = cfg.get("vocab", {}) if hasattr(cfg, "get") else {}

    return GraphVocab(
        n_preds=int(vocab_cfg.get("n_preds", 51)),
        n_entities=int(vocab_cfg.get("n_entities", 151)),
        max_instance_id=int(
            vocab_cfg.get("max_instance_id", 30)
        ),
        random_max_instance_id=int(
            vocab_cfg.get("random_max_instance_id", 10)
        ),
        max_num_rels=int(
            vocab_cfg.get("max_num_rels", 55)
        ),
    )


class PatchSGGModel(nn.Module):
    """Complete PatchSGG encoder-decoder model."""

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.vocab = build_vocab(cfg)

        # At the moment the project builds both encoders.
        #
        # For the LF-SGG configuration:
        #   train_modality = image
        #   eval_modality = image
        #
        # Therefore only image_encoder is actually used, although both are
        # still constructed by build_encoders().
        self.text_encoder, self.image_encoder = build_encoders(cfg)

        text_dim = int(self.text_encoder.embed_dim)
        image_dim = int(self.image_encoder.embed_dim)

        if text_dim != image_dim:
            raise ValueError(
                "Text and image encoder dimensions differ: "
                f"text={text_dim}, image={image_dim}. "
                "Use encoders that share one embedding space or add an "
                "explicit modality projection."
            )

        cond_dim = image_dim

        self.bridge = build_bridge(cfg)

        # build_decoder() selects:
        #   cross_attn
        #   prefix
        #   gpt2_cross_attn
        #
        # For location_free.yaml this constructs GPT2CrossAttnDecoder.
        self.decoder = build_decoder(
            cfg=cfg,
            vocab=self.vocab,
            cond_dim=cond_dim,
        )

        self.loss_fn = build_loss(
            cfg=cfg,
            vocab=self.vocab,
        )

        # Create the initial inference configuration.
        self.set_generation_config(cfg.eval)

    # -------------------------------------------------------------------------
    # Generation configuration
    # -------------------------------------------------------------------------

    def set_generation_config(self, eval_cfg) -> None:
        """Update autoregressive generation settings.

        This is useful after loading a checkpoint because CLI inference
        overrides are not automatically copied into the model's GenConfig.
        """
        max_rels = int(eval_cfg.get("max_rels", 100))

        if max_rels < 0:
            raise ValueError(
                f"eval.max_rels must be non-negative, got {max_rels}"
            )

        required_positions = (
            1 + max_rels * TOKENS_PER_REL
        )

        decoder_max_positions = int(
            self.decoder.max_seq_len
        )

        if required_positions > decoder_max_positions:
            raise ValueError(
                f"Generating {max_rels} relations requires "
                f"{required_positions} decoder positions: "
                f"1 START + {max_rels} × {TOKENS_PER_REL}. "
                f"The current decoder supports only "
                f"{decoder_max_positions} positions."
            )

        self.gen_cfg = GenConfig(
            max_rels=max_rels,
            temperature=float(
                eval_cfg.get("temperature", 1.75)
            ),
            top_p=float(
                eval_cfg.get("top_p", 0.95)
            ),
            top_k=int(
                eval_cfg.get("top_k", 0)
            ),
            entity_sampling=str(
                eval_cfg.get(
                    "entity_sampling",
                    "stochastic",
                )
            ),
            allow_end=bool(
                eval_cfg.get("allow_end", True)
            ),
        )

    # -------------------------------------------------------------------------
    # Conditioning
    # -------------------------------------------------------------------------

    def encode(
        self,
        batch: Dict,
        modality: str,
        training: bool,
    ) -> ConditioningSet:
        """Encode text or images into the common conditioning structure."""
        if modality == "text":
            if "texts" not in batch:
                raise KeyError(
                    "Text conditioning requires batch['texts']"
                )

            cond = self.text_encoder.encode(
                batch["texts"]
            )

        elif modality == "image":
            if "images" not in batch:
                raise KeyError(
                    "Image conditioning requires batch['images']"
                )

            cond = self.image_encoder.encode(
                batch["images"]
            )

        else:
            raise ValueError(
                "modality must be either 'text' or 'image', "
                f"got {modality!r}"
            )

        return self.bridge(
            cond,
            training=training,
            modality=modality,
        )

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    def compute_loss(
        self,
        batch: Dict,
        modality: str = "text",
    ) -> torch.Tensor:
        """Compute teacher-forced graph-token training loss."""
        cond = self.encode(
            batch,
            modality=modality,
            training=True,
        )

        device = cond.tokens.device

        input_tokens = batch["input_tokens"].to(
            device=device,
            dtype=torch.long,
        )

        target_tokens = batch["target_tokens"].to(
            device=device,
            dtype=torch.long,
        )

        # Shape:
        #
        # input_tokens: [B, T]
        # logits:       [B, T, graph_vocab_size]
        logits = self.decoder(
            cond,
            input_tokens,
        )

        return self.loss_fn(
            logits,
            target_tokens,
            input_tokens,
        )

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        batch: Dict,
        modality: str,
    ) -> List[List[tuple]]:
        """Generate graph-token sequences and convert them into relations."""
        cond = self.encode(
            batch,
            modality=modality,
            training=False,
        )

        sequences, scores = self.decoder.generate(
            cond,
            self.gen_cfg,
        )

        return sequences_to_predictions(
            sequences,
            scores,
            self.vocab,
        )

    # -------------------------------------------------------------------------
    # Optimization
    # -------------------------------------------------------------------------

    def trainable_parameters(self):
        """Return trainable decoder and bridge parameters.

        Text and image encoders remain frozen.
        """
        parameters = list(
            self.decoder.parameters()
        )

        parameters.extend(
            parameter
            for parameter in self.bridge.parameters()
            if parameter.requires_grad
        )

        return [
            parameter
            for parameter in parameters
            if parameter.requires_grad
        ]