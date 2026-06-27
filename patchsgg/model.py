"""PatchSGG core model: frozen encoders + modality bridge + AR graph decoder + loss.

A plain ``nn.Module`` so it can be driven by a bare torch loop or wrapped by Lightning. The encoder
used depends on the *phase*: text at train, image at inference (or text->text for the diagnostic).
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from patchsgg.bridge import build_bridge
from patchsgg.decoder import GenConfig, build_decoder
from patchsgg.encoders import build_encoders
from patchsgg.encoders.base import ConditioningSet
from patchsgg.graph_seq.vocab import GraphVocab
from patchsgg.losses import build_loss
from patchsgg.postprocess import sequences_to_predictions


def build_vocab(cfg) -> GraphVocab:
    v = cfg.get("vocab", {}) if hasattr(cfg, "get") else {}
    return GraphVocab(
        n_preds=int(v.get("n_preds", 51)),
        n_entities=int(v.get("n_entities", 151)),
        max_instance_id=int(v.get("max_instance_id", 30)),
        random_max_instance_id=int(v.get("random_max_instance_id", 10)),
        max_num_rels=int(v.get("max_num_rels", 55)),
    )


class PatchSGGModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.vocab = build_vocab(cfg)
        self.text_encoder, self.image_encoder = build_encoders(cfg)

        cond_dim = self.text_encoder.embed_dim
        if self.image_encoder.embed_dim != cond_dim:
            raise ValueError(
                f"text/image embed dims differ ({cond_dim} vs {self.image_encoder.embed_dim}); "
                "use a shared space (dinov2_talk2dino/clip) or add a projection."
            )
        self.bridge = build_bridge(cfg)
        self.decoder = build_decoder(cfg, self.vocab, cond_dim)
        self.loss_fn = build_loss(cfg, self.vocab)
        self.gen_cfg = GenConfig(
            max_rels=int(cfg.eval.get("max_rels", 100)),
            temperature=float(cfg.eval.get("temperature", 1.75)),
            top_p=float(cfg.eval.get("top_p", 0.95)),
            top_k=int(cfg.eval.get("top_k", 0)),
            entity_sampling=cfg.eval.get("entity_sampling", "stochastic"),
            allow_end=bool(cfg.eval.get("allow_end", True)),
        )

    # --- conditioning --------------------------------------------------------------------
    def encode(self, batch: Dict, modality: str, training: bool) -> ConditioningSet:
        if modality == "text":
            cond = self.text_encoder.encode(batch["texts"])
        elif modality == "image":
            cond = self.image_encoder.encode(batch["images"])
        else:
            raise ValueError(modality)
        return self.bridge(cond, training=training, modality=modality)

    # --- train / infer -------------------------------------------------------------------
    def compute_loss(self, batch: Dict, modality: str = "text") -> torch.Tensor:
        cond = self.encode(batch, modality=modality, training=True)
        logits = self.decoder(cond, batch["input_tokens"].to(cond.tokens.device))
        target = batch["target_tokens"].to(cond.tokens.device)
        return self.loss_fn(logits, target, batch["input_tokens"].to(cond.tokens.device))

    @torch.no_grad()
    def predict(self, batch: Dict, modality: str) -> List[List[tuple]]:
        cond = self.encode(batch, modality=modality, training=False)
        seq, scores = self.decoder.generate(cond, self.gen_cfg)
        return sequences_to_predictions(seq, scores, self.vocab)

    def trainable_parameters(self):
        """Decoder (+ any trainable bridge params); encoders are frozen."""
        params = list(self.decoder.parameters())
        params += [p for p in self.bridge.parameters() if p.requires_grad]
        return [p for p in params if p.requires_grad]
