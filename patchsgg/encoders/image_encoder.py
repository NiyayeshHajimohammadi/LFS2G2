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

        self.model, self.preprocess = clip.load(
            cfg.encoders.clip_model,
            device=self.device,
        )

        self.model.eval()

        # --------------------------------------------------------------
        # Freeze the complete CLIP model first.
        #
        # This preserves the old behaviour when:
        #
        #   train_image_encoder: false
        #
        # or when the field is absent from an older config.
        # --------------------------------------------------------------
        for p in self.model.parameters():
            p.requires_grad_(False)

        # --------------------------------------------------------------
        # Optional partial visual-backbone fine-tuning.
        # --------------------------------------------------------------
        self.train_image_encoder = bool(
            cfg.encoders.get(
                "train_image_encoder",
                False,
            )
        )

        if self.train_image_encoder:
            # OpenAI CLIP CUDA weights are fp16 by default.
            # Train the visual backbone in fp32 for numerical stability.
            self.model.visual.float()

            freeze_first_n = int(
                cfg.encoders.get(
                    "freeze_first_n",
                    16,
                )
            )

            blocks = self.model.visual.transformer.resblocks

            if not 0 <= freeze_first_n <= len(blocks):
                raise ValueError(
                    f"freeze_first_n must be in "
                    f"[0, {len(blocks)}], "
                    f"got {freeze_first_n}"
                )

            # Freeze blocks before freeze_first_n and train the rest.
            #
            # For ViT-L/14:
            #
            #   blocks 0-15  -> frozen
            #   blocks 16-23 -> trainable
            #
            # when freeze_first_n == 16.
            for block in blocks[freeze_first_n:]:
                for p in block.parameters():
                    p.requires_grad_(True)

        # IMPORTANT:
        # This must be set regardless of whether CLIP is frozen or trainable.
        self.embed_dim = int(
            self.model.text_projection.shape[1]
        )

    def encode(
        self,
        images: torch.Tensor,
    ) -> ConditioningSet:

        images = images.to(self.device)

        v = self.model.visual

        # --------------------------------------------------------------
        # 1. CLIP patch embedding
        #
        # [B, 3, H, W]
        #       ->
        # [B, N_patches, visual_width]
        # --------------------------------------------------------------
        x = v.conv1(
            images.type(self.model.dtype)
        )

        x = x.reshape(
            x.shape[0],
            x.shape[1],
            -1,
        ).permute(
            0,
            2,
            1,
        )

        # --------------------------------------------------------------
        # 2. Prepend CLS token.
        # --------------------------------------------------------------
        cls = (
            v.class_embedding.to(x.dtype)
            + torch.zeros(
                x.shape[0],
                1,
                x.shape[-1],
                dtype=x.dtype,
                device=x.device,
            )
        )

        x = torch.cat(
            [cls, x],
            dim=1,
        )

        # --------------------------------------------------------------
        # 3. CLIP learned image positional encoding.
        # --------------------------------------------------------------
        x = (
            x
            + v.positional_embedding.to(x.dtype)
        )

        x = v.ln_pre(x)

        # --------------------------------------------------------------
        # 4. OpenAI CLIP Transformer expects:
        #
        # [sequence, batch, channels]
        # --------------------------------------------------------------
        x = x.permute(
            1,
            0,
            2,
        )

        x = v.transformer(x)

        # Back to:
        #
        # [batch, sequence, channels]
        x = x.permute(
            1,
            0,
            2,
        )

        # --------------------------------------------------------------
        # 5. Final CLIP visual normalization.
        # --------------------------------------------------------------
        x = v.ln_post(x)

        # --------------------------------------------------------------
        # 6. Separate CLS and spatial patch features.
        # --------------------------------------------------------------
        pooled = x[:, 0, :]
        patches = x[:, 1:, :]

        # --------------------------------------------------------------
        # 7. Project into CLIP's joint embedding space.
        # --------------------------------------------------------------
        if v.proj is not None:
            proj = v.proj.to(x.dtype)

            pooled = pooled @ proj
            patches = patches @ proj

        pooled = pooled.float()
        patches = patches.float()

        pooled = pooled / pooled.norm(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-12)

        # --------------------------------------------------------------
        # 8. Return either pooled-only or full spatial patch conditioning.
        # --------------------------------------------------------------
        if self.token_mode == "pooled":
            return ConditioningSet(
                tokens=pooled.unsqueeze(1),
                pooled=pooled,
            )

        return ConditioningSet(
            tokens=patches,
            pooled=pooled,
        )