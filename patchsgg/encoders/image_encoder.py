"""Frozen image encoders (inference path).

  * :class:`DinoV2ImageEncoder` -- DINOv2 patch tokens (same space as Talk2DINO-projected text).
  * :class:`ClipImageEncoder`   -- CLIP visual patch tokens + pooled (CLIP joint space).

Both emit a :class:`ConditioningSet` with ``tokens=[B,N_patch,D]`` and ``pooled=[B,D]`` so the
decoder is modality-agnostic. ``cfg.encoders.image_token_mode`` selects ``patch`` (full grid) or
``pooled`` (CLS only -> N==1, the bottleneck baseline).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from patchsgg.encoders.base import ConditioningSet


class DinoV2ImageEncoder(nn.Module):
    modality = "image"

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.token_mode = cfg.encoders.image_token_mode
        # My comment: "patch" → full spatial grid
        # My comment: "pooled" → CLS-only bottleneck
        self.model_name = cfg.encoders.dino_model  # e.g. 'dinov2_vitb14' / 'dinov2_vitl14_reg'
        self.model = torch.hub.load("facebookresearch/dinov2", self.model_name)
        self.model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.embed_dim = int(cfg.encoders.dino_embed_dim)

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> ConditioningSet:
        images = images.to(self.device)
        # DINOv2's forward_features already returns normalized CLS + patch tokens.
        out = self.model.forward_features(images)
        cls = out["x_norm_clstoken"].float()           # [B, D]
        patches = out["x_norm_patchtokens"].float()    # [B, N, D]
        if self.token_mode == "pooled":
            return ConditioningSet(tokens=cls.unsqueeze(1), pooled=cls)
        return ConditioningSet(tokens=patches, pooled=cls)


class ClipImageEncoder(nn.Module):
    modality = "image"

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.token_mode = cfg.encoders.image_token_mode
        import clip

        self.model, self.preprocess = clip.load(cfg.encoders.clip_model, device=self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.embed_dim = self.model.text_projection.shape[1]

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> ConditioningSet:
        """Patch tokens from CLIP ViT (projected to the joint space) + pooled image embedding."""
        images = images.to(self.device)
        pooled = self.model.encode_image(images).float()
        pooled = pooled / pooled.norm(dim=-1, keepdim=True)#My comment: L2 normalization
        if self.token_mode == "pooled":
            return ConditioningSet(tokens=pooled.unsqueeze(1), pooled=pooled)
        v = self.model.visual
        x = v.conv1(images.type(self.model.dtype))#My comment: Transforms image into patch grid.
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [B, grid^2, width]
        cls = v.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)#My comment:a learnable global token.
        x = torch.cat([cls, x], dim=1) + v.positional_embedding.to(x.dtype)
        x = v.ln_pre(x).permute(1, 0, 2)
        x = v.transformer(x).permute(1, 0, 2)
        x = v.ln_post(x)
        patches = x[:, 1:, :].float()                  # drop CLS -> [B, grid^2, width]
        if v.proj is not None:
            patches = patches @ v.proj.float()         # -> joint dim
        return ConditioningSet(tokens=patches, pooled=pooled)

#My comment:
# Manual CLIP internals (HIGH RISK)
# No guaranteed alignment check between encoders (MEDIUM)
# Token space mismatch risk (MEDIUM)
# Preprocessing pipeline inconsistency (MEDIUM)
# No positional regularization alignment across models (LOW-MEDIUM)
