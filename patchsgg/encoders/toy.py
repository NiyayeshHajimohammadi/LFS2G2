"""Lightweight encoders for smoke / overfit tests -- no external weights or network.

ToyTextEncoder maps each distinct text deterministically to a fixed vector (so distinct graphs get
distinct, separable conditioning -> the text->text diagnostic can actually overfit). ToyImageEncoder
turns an image into a small grid of patch tokens via a fixed random conv. Both are frozen.
"""
#My comment: If the rest of my pipeline is correct, can it learn at all?
from __future__ import annotations

import hashlib

import torch
import torch.nn as nn

from patchsgg.encoders.base import ConditioningSet


def _text_to_vec(text: str, dim: int) -> torch.Tensor:
    seed = int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16)
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=g)
    return v / v.norm().clamp_min(1e-6)


class ToyTextEncoder(nn.Module):
    modality = "text"

    def __init__(self, cfg):
        super().__init__()
        self.embed_dim = int(cfg.encoders.dim)
        self.n_tokens = int(cfg.encoders.get("toy_text_tokens", 1))
        self.device = cfg.device

    @torch.no_grad()
    def encode(self, texts) -> ConditioningSet:
        vecs = torch.stack([_text_to_vec(t, self.embed_dim) for t in texts]).to(self.device)#My comment: shape [B,D]
        tokens = vecs.unsqueeze(1).repeat(1, self.n_tokens, 1)
        return ConditioningSet(tokens=tokens, pooled=vecs)


class ToyImageEncoder(nn.Module):
    modality = "image"

    def __init__(self, cfg):
        super().__init__()
        self.embed_dim = int(cfg.encoders.dim)
        self.grid = int(cfg.encoders.get("toy_grid", 4))
        self.device = cfg.device
        self.proj = nn.Conv2d(3, self.embed_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(self.grid)
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> ConditioningSet:
        images = images.to(self.device)
        x = self.pool(self.proj(images))                 # [B, D, g, g] My comment-> [8,512,4,4]
        B, D, g, _ = x.shape
        tokens = x.reshape(B, D, g * g).permute(0, 2, 1)  # [B, N, D]
        pooled = tokens.mean(dim=1) #My commet: Averages all patch embeddings.
        return ConditioningSet(tokens=tokens, pooled=pooled)
#My comment:  What this module does, is detectine issues with the following factors:
# The decoder is capable of memorization
# Training loop is correct
# Tensor flow is correct
