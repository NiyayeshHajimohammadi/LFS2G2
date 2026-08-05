"""DeCap support-memory projection bridge (inference-time only).

At inference the image conditioning is projected onto a memory bank of text embeddings, pulling
image features toward the text manifold the decoder trained on. Applied per token so it works for
both pooled and patch-set conditioning.
"""
from __future__ import annotations

import torch

from patchsgg.bridge.base import Bridge
from patchsgg.bridge.decap_memory import DecapMemoryProjector
from patchsgg.encoders.base import ConditioningSet


class DecapProjectionBridge(Bridge):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.temperature = float(cfg.bridge.projection_temperature)
        self.apply_to_tokens = bool(cfg.bridge.get("project_tokens", True))
        self.memory_path = cfg.bridge.memory_path
        self._projector = None

    def _lazy_projector(self) -> DecapMemoryProjector:
        if self._projector is None:
            self._projector = DecapMemoryProjector(
                self.memory_path,
                device=self.cfg.device,
                hf_repo_id=self.cfg.bridge.get("memory_hf_repo_id", None),
                hf_filename=self.cfg.bridge.get("memory_hf_filename", None),
                cache_dir=self.cfg.encoders.get("hf_cache_dir", None),
            )
        return self._projector

    @torch.no_grad()
    def forward(self, cond, *, training, modality):
        if training or modality != "image":
            return cond
        proj = self._lazy_projector()
        pooled = proj.project(cond.pooled, temperature=self.temperature)
        if not self.apply_to_tokens:
            return ConditioningSet(tokens=cond.tokens, pooled=pooled, mask=cond.mask)
        B, N, D = cond.tokens.shape
        tokens = proj.project(cond.tokens.reshape(B * N, D), temperature=self.temperature).reshape(B, N, D)
        return ConditioningSet(tokens=tokens, pooled=pooled, mask=cond.mask)
