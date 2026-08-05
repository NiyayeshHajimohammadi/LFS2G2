"""Unified conditioning interface for text and image encoders."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import torch


@dataclass
class ConditioningSet:
    """Modality-agnostic conditioning tensors.

    ``tokens`` has shape ``[B, N, D]``; ``pooled`` has shape ``[B, D]``;
    ``mask`` is optional ``[B, N]`` with ``True`` for valid positions.
    """

    tokens: torch.Tensor
    pooled: torch.Tensor
    mask: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if self.tokens.ndim != 3:
            raise ValueError(
                f"tokens must have shape [B, N, D], got {tuple(self.tokens.shape)}"
            )
        if self.pooled.ndim != 2:
            raise ValueError(
                f"pooled must have shape [B, D], got {tuple(self.pooled.shape)}"
            )
        if self.tokens.shape[0] != self.pooled.shape[0]:
            raise ValueError("tokens and pooled must have the same batch size")
        if self.tokens.shape[-1] != self.pooled.shape[-1]:
            raise ValueError("tokens and pooled must have the same feature dimension")
        if self.mask is not None:
            if self.mask.shape != self.tokens.shape[:2]:
                raise ValueError(
                    f"mask must have shape {tuple(self.tokens.shape[:2])}, "
                    f"got {tuple(self.mask.shape)}"
                )
            self.mask = self.mask.bool()

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
        return ConditioningSet(
            tokens=self.pooled.unsqueeze(1),
            pooled=self.pooled,
            mask=None,
        )


class Encoder(Protocol):
    modality: str
    embed_dim: int

    def encode(self, batch) -> ConditioningSet:  # pragma: no cover - interface
        ...


def build_encoders(
    cfg,
    *,
    need_text: bool = True,
    need_image: bool = True,
) -> tuple[Optional["Encoder"], Optional["Encoder"]]:
    """Build only the encoder modalities required by the selected experiment."""
    if not need_text and not need_image:
        raise ValueError("at least one encoder modality must be requested")

    space = str(cfg.encoders.space)
    text = None
    image = None

    if space == "toy":
        if need_text:
            from patchsgg.encoders.toy import ToyTextEncoder

            text = ToyTextEncoder(cfg)
        if need_image:
            from patchsgg.encoders.toy import ToyImageEncoder

            image = ToyImageEncoder(cfg)
        return text, image

    if space == "dinov2_talk2dino":
        if need_text:
            from patchsgg.encoders.text_encoder import Talk2DinoTextEncoder

            text = Talk2DinoTextEncoder(cfg)
        if need_image:
            from patchsgg.encoders.image_encoder import DinoV2ImageEncoder

            image = DinoV2ImageEncoder(cfg)
    elif space == "clip":
        if need_text:
            from patchsgg.encoders.text_encoder import ClipTextEncoder

            text = ClipTextEncoder(cfg)
        if need_image:
            from patchsgg.encoders.image_encoder import ClipImageEncoder

            image = ClipImageEncoder(cfg)
    else:
        raise ValueError(f"unknown encoders.space={space!r}")

    return text, image
