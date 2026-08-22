"""Training losses over the flat 5-tuple vocabulary.

All losses are masked token-level cross-entropy at heart (NO_KNOWN positions ignored); they differ
in how they treat the two location-free pathologies:
  * ``ce``            -- plain CE (+ optional label smoothing).
  * ``ce_weighted``   -- frequency-weighted CE (rare predicates upweighted -> mean-recall).
  * ``matching_ce``   -- relabels target instance ids to the model's assignment (branched matcher)
                         before CE, so arbitrary instance-id permutations are not penalised.
  * ``order_agnostic``-- reorders GT tuples to best-match the model's tuple order before CE.

A :class:`Loss` is a callable: ``loss(logits[B,T,V], target[B,T], input_tokens[B,T]) -> scalar``.
"""
from __future__ import annotations

import json
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from patchsgg.graph_seq.vocab import GraphVocab

class Loss(nn.Module):
    def __init__(self, vocab: GraphVocab, weight: Optional[torch.Tensor] = None, label_smoothing: float = 0.0):
        super().__init__()
        self.vocab = vocab
        self.label_smoothing = label_smoothing
        self.register_buffer("weight", weight if weight is not None else torch.ones(vocab.vocab_size))

    def _ce(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1),
            weight=self.weight.to(logits.dtype),
            ignore_index=self.vocab.no_known_token,
            label_smoothing=self.label_smoothing,
        )

    def forward(self, logits, target, input_tokens=None):
        return self._ce(logits, target)


def _frequency_weights(vocab: GraphVocab, freq_path: Optional[str]) -> torch.Tensor:
    """LF-SGG-style log weighting ``1/(log f + 1)``; END/NOISE strongly downweighted."""
    weights = torch.ones(vocab.vocab_size)
    if freq_path:
        with open(freq_path) as f:
            freqs = json.load(f)  # {token_id(str): count}
        for k, v in freqs.items():
            weights[int(k)] = 1.0 / (math.log(max(v, 1)) + 1.0)
        min_w = float(min(weights[int(k)] for k in freqs))
        weights[vocab.end_token] = min_w / 100.0
        weights[vocab.noise_token] = min_w / 100.0
    else:
        weights[vocab.end_token] = 0.01
        weights[vocab.noise_token] = 0.01
    return weights


def build_loss(cfg, vocab: GraphVocab) -> Loss: 
    kind = cfg.loss.type
    ls = float(getattr(cfg.loss, "label_smoothing", 0.0))
    if kind == "ce":
        return Loss(vocab, label_smoothing=ls)
    if kind == "ce_weighted":
        w = _frequency_weights(vocab, getattr(cfg.loss, "freq_path", None))
        return Loss(vocab, weight=w, label_smoothing=ls)
    if kind == "matching_ce":
        from patchsgg.losses.matching_ce import MatchingCELoss

        w = _frequency_weights(vocab, getattr(cfg.loss, "freq_path", None)) if cfg.loss.get("weighted", False) else None
        return MatchingCELoss(vocab, weight=w, label_smoothing=ls,
                              matcher_n=int(cfg.loss.get("matcher_n", 3)),
                              matcher_depth=int(cfg.loss.get("matcher_depth", 10)))
    if kind == "order_agnostic":
        from patchsgg.losses.order_agnostic import OrderAgnosticCELoss

        return OrderAgnosticCELoss(vocab, label_smoothing=ls)
    raise ValueError(f"unknown loss.type {kind!r}")
