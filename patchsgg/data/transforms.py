"""Encoder-aware image preprocessing without a hard torchvision dependency."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from PIL import Image


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass(frozen=True)
class SquareImageTransform:
    """Resize the short side, center-crop a square, convert to CHW, and normalize."""

    size: int
    mean: Sequence[float] | None = None
    std: Sequence[float] | None = None

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError(f"image size must be positive, got {self.size}")
        if (self.mean is None) != (self.std is None):
            raise ValueError("mean and std must either both be set or both be None")

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            raise TypeError(f"expected PIL.Image.Image, got {type(image)!r}")
        image = image.convert("RGB")
        width, height = image.size
        scale = self.size / min(width, height)
        new_width = max(self.size, int(round(width * scale)))
        new_height = max(self.size, int(round(height * scale)))
        image = image.resize((new_width, new_height), resample=Image.Resampling.BICUBIC)
        left = (new_width - self.size) // 2
        top = (new_height - self.size) // 2
        image = image.crop((left, top, left + self.size, top + self.size))

        array = np.asarray(image, dtype=np.float32).copy() / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        if self.mean is not None and self.std is not None:
            mean = torch.tensor(self.mean, dtype=tensor.dtype).view(3, 1, 1)
            std = torch.tensor(self.std, dtype=tensor.dtype).view(3, 1, 1)
            tensor = (tensor - mean) / std
        return tensor


def build_image_transform(cfg) -> SquareImageTransform:
    """Return preprocessing compatible with the configured frozen image encoder."""
    space = str(cfg.encoders.space)
    if space == "clip":
        size = int(cfg.encoders.get("clip_resize_dim", 224))
        return SquareImageTransform(size=size, mean=CLIP_MEAN, std=CLIP_STD)
    if space == "dinov2_talk2dino":
        size = int(cfg.encoders.get("resize_dim", 518))
        return SquareImageTransform(size=size, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if space == "toy":
        size = int(cfg.encoders.get("toy_image_size", 32))
        return SquareImageTransform(size=size)
    raise ValueError(f"unknown encoders.space={space!r}")
