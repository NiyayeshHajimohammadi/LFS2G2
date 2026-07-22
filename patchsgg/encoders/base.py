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
            single token (``N==1``) for the pooled baseline. (N is the number of conditioning tokens)
    pooled: ``[B, D]`` global feature (CLS / EOS / mean).
    mask:   ``[B, N]`` boolean, True = valid token (for variable-length text). None => all valid.
    """
    #My comment: stores the outputs of an encoder in a standardized format.
    tokens: torch.Tensor #My comment: the main conditioning tensor shape [B, N, D]
    pooled: torch.Tensor #My comment: one global feature vector per input sample shape [B, D]
    mask: Optional[torch.Tensor] = None #My comment: which token is valid and which is not [B, N]

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
    #My comment: factory function:)))->selects and constructs the text/image encoder pair specified in the configuration.
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

# Problems:
# Missing shape validation (Medium severity)->Nothing checks these assumptions.A malformed encoder could silently return inconsistent shapes, and the error might only appear much later in the decoder.Adding validation in __post_init__() would make debugging much easier.
# Device consistency (Low severity)->to() moves every tensor, but there is no verification that all tensors originally reside on the same device. In practice this is usually fine, but explicit consistency checks can prevent subtle bugs.
# Modality string (Low severity)->The protocol documents modality as "text" or "image", but this is only a convention. An Enum instead of a free-form string would prevent accidental typos such as "images" or "img".
# Runtime protocol checking (Very low severity)-> Protocol improves static type checking but does not enforce the interface at runtime. If you wanted runtime validation, you could decorate it with @runtime_checkable and use isinstance(). For this project, static checking is probably sufficient.
