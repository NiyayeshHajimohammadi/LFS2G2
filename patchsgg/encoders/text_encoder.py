"""Frozen text encoders (training path).

Two flavours:
  * :class:`Talk2DinoTextEncoder` -- CLIP text -> Talk2DINO projection into DINOv2 patch space.
    This is the default: text (train) and image patches (inference) live in the *same* space.
  * :class:`ClipTextEncoder` -- plain CLIP joint space (baseline).

Token-set vs pooled is controlled by ``cfg.encoders.text_token_mode``:
  * ``pooled``      -> N==1 (the pooled-vector baseline)
  * ``clip_tokens`` -> N==L per-token CLIP hidden states (projected). NOTE: Talk2DINO was trained
                       on *pooled* text, so projecting per-token states is an explicit research
                       approximation, kept as a knob.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from patchsgg.encoders.base import ConditioningSet
from patchsgg.utils.huggingface import get_model_path_with_hf_fallback


def _load_clip(cfg, device):
    import clip  # openai CLIP

    model, _ = clip.load(cfg.encoders.clip_model, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, clip


def _clip_token_hidden_states(clip_model, tokens: torch.Tensor) -> torch.Tensor:
    """Per-token hidden states from CLIP's text transformer: ``[B, L, D_transformer]``.

    Mirrors ``CLIP.encode_text`` up to (but not including) the EOS pooling + projection.
    """
    x = clip_model.token_embedding(tokens).type(clip_model.dtype)
    x = x + clip_model.positional_embedding.type(clip_model.dtype)
    x = x.permute(1, 0, 2)
    x = clip_model.transformer(x)
    x = x.permute(1, 0, 2)
    x = clip_model.ln_final(x).type(clip_model.dtype)
    return x  # [B, L, D]


class _BaseTextEncoder(nn.Module):
    modality = "text"

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.token_mode = cfg.encoders.text_token_mode
        self.max_tokens = int(getattr(cfg.encoders, "max_text_tokens", 32))

    @torch.no_grad()
    def _tokenize(self, texts):
        return self._clip.tokenize(texts, truncate=True).to(self.device)


class ClipTextEncoder(_BaseTextEncoder):
    """Plain CLIP joint space."""

    embed_dim = 512

    def __init__(self, cfg):
        super().__init__(cfg)
        self.clip_model, self._clip = _load_clip(cfg, self.device)
        self.embed_dim = self.clip_model.text_projection.shape[1]

    @torch.no_grad()
    def encode(self, texts) -> ConditioningSet:
        tokens = self._tokenize(texts)
        pooled = self.clip_model.encode_text(tokens).float()
        pooled = pooled / pooled.norm(dim=-1, keepdim=True)
        if self.token_mode == "pooled":
            return ConditioningSet(tokens=pooled.unsqueeze(1), pooled=pooled)
        hidden = _clip_token_hidden_states(self.clip_model, tokens).float()
        # project per-token hidden states with the text projection (same map CLIP uses on EOS)
        hidden = hidden @ self.clip_model.text_projection.float()
        mask = (tokens != 0)[:, : hidden.shape[1]]
        return ConditioningSet(tokens=hidden, pooled=pooled, mask=mask)


class Talk2DinoTextEncoder(_BaseTextEncoder):
    """CLIP text projected into DINOv2 patch space via Talk2DINO's ProjectionLayer."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.clip_model, self._clip = _load_clip(cfg, self.device)
        from patchsgg.encoders.talk2dino import ProjectionLayer

        self.proj = ProjectionLayer.from_config(cfg.encoders.talk2dino_config)
        weights_path = get_model_path_with_hf_fallback(
            cfg.encoders.talk2dino_weights,
            hf_repo_id=cfg.encoders.get("talk2dino_hf_repo_id", None),
            filename=cfg.encoders.get("talk2dino_hf_filename", None),
            cache_dir=cfg.encoders.get("hf_cache_dir", None),
        )
        sd = torch.load(weights_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        self.proj.load_state_dict(sd)
        self.proj.eval().to(self.device)
        for p in self.proj.parameters():
            p.requires_grad_(False)
        self.embed_dim = int(cfg.encoders.dino_embed_dim)

    @torch.no_grad()
    def encode(self, texts) -> ConditioningSet:
        tokens = self._tokenize(texts)
        clip_pooled = self.clip_model.encode_text(tokens).float()
        clip_pooled = clip_pooled / clip_pooled.norm(dim=-1, keepdim=True)
        pooled = self.proj.project_clip_txt(clip_pooled).float()
        if self.token_mode == "pooled":
            return ConditioningSet(tokens=pooled.unsqueeze(1), pooled=pooled)
        # clip_tokens: project each per-token CLIP hidden state into DINO space
        hidden = _clip_token_hidden_states(self.clip_model, tokens).float()
        hidden = hidden @ self.clip_model.text_projection.float()  # -> CLIP joint dim
        B, L, D = hidden.shape
        proj_tokens = self.proj.project_clip_txt(hidden.reshape(B * L, D)).reshape(B, L, -1)
        mask = (tokens != 0)[:, :L]
        return ConditioningSet(tokens=proj_tokens.float(), pooled=pooled, mask=mask)
