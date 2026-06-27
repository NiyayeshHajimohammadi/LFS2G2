"""CapDec-style Gaussian noise injection (train-time only).

Adds isotropic noise to the (text) conditioning during training so the decoder becomes robust to
the text->image embedding offset at inference. Re-normalises to keep features on the unit sphere,
matching Patch-ioner's recipe.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from patchsgg.bridge.base import Bridge
from patchsgg.encoders.base import ConditioningSet


class CapDecNoiseBridge(Bridge):
    def __init__(self, std: float = 0.08):
        super().__init__()
        self.std = std

    def forward(self, cond, *, training, modality):
        if not training or self.std <= 0:
            return cond
        tokens = cond.tokens + self.std * torch.randn_like(cond.tokens)
        tokens = F.normalize(tokens, dim=-1)
        pooled = F.normalize(cond.pooled + self.std * torch.randn_like(cond.pooled), dim=-1)
        return ConditioningSet(tokens=tokens, pooled=pooled, mask=cond.mask)
