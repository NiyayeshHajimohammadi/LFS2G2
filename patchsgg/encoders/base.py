"""Unified conditioning interface shared by the text (train) and image (inference) paths.

The decoder never sees an encoder directly -- it sees a :class:`ConditioningSet`. This is what
makes train(text)->infer(image) transfer a config switch rather than a code change, and what lets
us toggle pooled-vector vs patch-token-set conditioning uniformly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import torch


@dataclass
class ConditioningSet:
    """Modality-agnostic conditioning.

    tokens: ``[B, N, D]`` per-token features (text tokens, or image patch tokens). May be a
            single token (``N==1``) for the pooled baseline.
    pooled: ``[B, D]`` global feature (CLS / EOS / mean).
    mask:   ``[B, N]`` boolean, True = valid token (for variable-length text). None => all valid.
    """
    #My comment: stores the outputs of an encoder in a standardized format.
    tokens: torch.Tensor
    pooled: torch.Tensor
    mask: Optional[torch.Tensor] = None

    @property
    def dim(self) -> int:
        return self.tokens.shape[-1]

    @property
    def batch_size(self) -> int:
        return self.tokens.shape[0]

    def to(self, device) -> "ConditioningSet":
        return ConditioningSet(
            tokens=self.tokens.to(device),
            pooled=self.pooled.to(device),
            mask=None if self.mask is None else self.mask.to(device),
        )

    def as_pooled(self) -> "ConditioningSet":
        """Collapse to a single conditioning token (the pooled baseline)."""
        return ConditioningSet(tokens=self.pooled.unsqueeze(1), pooled=self.pooled, mask=None) #My comment: Treat this embedding as a sequence containing exactly one token.
        #My comment: Masks are needed when different samples have different numbers of tokens.


class Encoder(Protocol):
    """Frozen encoder producing a :class:`ConditioningSet`. ``modality`` is 'text' or 'image'."""
    #My comment: Anything implementing these members is considered an Encoder

    modality: str
    embed_dim: int

    def encode(self, batch) -> ConditioningSet:  # pragma: no cover - interface
        ...


def build_encoders(cfg) -> tuple["Encoder", "Encoder"]:
    """Build the (text_encoder, image_encoder) pair selected by ``cfg.encoders.space``.

    Imports are local so a missing heavy dependency (DINOv2/CLIP/Talk2DINO) only breaks the path
    that actually needs it, not the whole package.
    """
    #My comment: factory function:)))
    space = cfg.encoders.space
    if space == "toy":
        from patchsgg.encoders.toy import ToyImageEncoder, ToyTextEncoder

        return ToyTextEncoder(cfg), ToyImageEncoder(cfg)
    if space == "dinov2_talk2dino":
        from patchsgg.encoders.image_encoder import DinoV2ImageEncoder
        from patchsgg.encoders.text_encoder import Talk2DinoTextEncoder

        text = Talk2DinoTextEncoder(cfg)
        image = DinoV2ImageEncoder(cfg)
    elif space == "clip":
        from patchsgg.encoders.image_encoder import ClipImageEncoder
        from patchsgg.encoders.text_encoder import ClipTextEncoder

        text = ClipTextEncoder(cfg)
        image = ClipImageEncoder(cfg)
    else:
        raise ValueError(f"unknown encoders.space={space!r}")
    return text, image
