"""Modality-gap bridges, applied to the :class:`ConditioningSet` before the decoder.

A bridge sees ``(cond, training, modality)`` and returns a (possibly modified) ConditioningSet.
Bridges are composable (config gives a list) so e.g. CapDec-noise (train-time) and DeCap-projection
(inference-time) can be stacked. Each is a no-op outside the phase it targets.
"""
from __future__ import annotations

from typing import List

import torch.nn as nn

from patchsgg.encoders.base import ConditioningSet


class Bridge(nn.Module):
    def forward(self, cond: ConditioningSet, *, training: bool, modality: str) -> ConditioningSet:  # noqa: D401
        raise NotImplementedError


class ComposedBridge(Bridge):
    def __init__(self, bridges: List[Bridge]):
        super().__init__()
        self.bridges = nn.ModuleList(bridges)

    def forward(self, cond, *, training, modality):
        for b in self.bridges:
            cond = b(cond, training=training, modality=modality)
        return cond


def build_bridge(cfg) -> Bridge:
    names = list(cfg.bridge.types) if getattr(cfg.bridge, "types", None) else [cfg.bridge.type]
    bridges: List[Bridge] = []
    for name in names:
        if name == "identity":
            from patchsgg.bridge.identity import IdentityBridge

            bridges.append(IdentityBridge())
        elif name == "capdec_noise":
            from patchsgg.bridge.capdec_noise import CapDecNoiseBridge

            bridges.append(CapDecNoiseBridge(std=float(cfg.bridge.noise_std)))
        elif name == "decap_projection":
            from patchsgg.bridge.decap_projection import DecapProjectionBridge

            bridges.append(DecapProjectionBridge(cfg))
        else:
            raise ValueError(f"unknown bridge type {name!r}")
    return ComposedBridge(bridges)
