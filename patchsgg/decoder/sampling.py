"""Constrained, token-type-aware autoregressive generation.

Reproduces LF-SGG's decoding discipline: at each position only the logits for the role expected at
that position are considered (entity / instance / predicate). Entities are sampled with
temperature + nucleus (top-p); instances and predicates are greedy. END is only permitted at a
tuple boundary (start of a new subject), giving a natural stopping point.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import torch

from patchsgg.graph_seq.vocab import TOKENS_PER_REL, GraphVocab, TokenType


@dataclass
class GenConfig:
    max_rels: int = 100
    temperature: float = 1.75
    top_p: float = 0.95
    top_k: int = 0
    entity_sampling: str = "stochastic"  # 'stochastic' | 'greedy'
    allow_end: bool = True


def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0) -> torch.Tensor:
    """Mask logits outside the top-k / top-p (nucleus) set with -inf. ``logits``: [B, V]."""
    logits = logits.clone()
    if top_k > 0:
        kth = torch.topk(logits, top_k, dim=-1).values[:, -1, None]
        logits[logits < kth] = float("-inf")
    if top_p > 0.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cum > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        scatter_remove = remove.scatter(1, sorted_idx, remove)
        logits[scatter_remove] = float("-inf")
    return logits


def _select_range(logits: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """Return a full-vocab logit tensor with everything outside [lo, hi) masked to -inf."""
    masked = torch.full_like(logits, float("-inf"))
    masked[:, lo:hi] = logits[:, lo:hi]
    return masked


@torch.no_grad()
def constrained_generate(
    step_fn: Callable[[torch.Tensor], torch.Tensor],
    start_tokens: torch.Tensor,
    vocab: GraphVocab,
    cfg: GenConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate token sequences.

    ``step_fn(tokens[B,T]) -> logits[B,V]`` gives next-token logits for the current prefix.
    ``start_tokens``: [B,1] (the START token). Returns ``(seq[B,T], scores[B,T])`` where ``seq``
    excludes the START token. ``scores`` are the chosen-token probabilities (for top-K ranking).
    """
    device = start_tokens.device
    B = start_tokens.shape[0]
    seq = start_tokens
    out_tokens, out_scores = [], []
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    total_steps = cfg.max_rels * TOKENS_PER_REL

    for i in range(total_steps):
        role = vocab.role_at(i)
        logits = step_fn(seq)  # [B, V]
        lo, hi = vocab.range_for_role(role)
        masked = _select_range(logits, lo, hi)

        at_tuple_start = (i % TOKENS_PER_REL) == 0
        if cfg.allow_end and at_tuple_start:
            masked[:, vocab.end_token] = logits[:, vocab.end_token]

        if role is TokenType.ENTITY and cfg.entity_sampling == "stochastic":
            scaled = masked / cfg.temperature
            scaled = top_k_top_p_filter(scaled, top_k=cfg.top_k, top_p=cfg.top_p)
            probs = torch.softmax(scaled, dim=-1)
            nxt = torch.multinomial(probs, 1)
        else:
            probs = torch.softmax(masked, dim=-1)
            nxt = probs.argmax(dim=-1, keepdim=True)
        score = probs.gather(-1, nxt)

        nxt = nxt.masked_fill(finished.unsqueeze(1), vocab.no_known_token)
        finished = finished | (nxt.squeeze(1) == vocab.end_token)

        out_tokens.append(nxt)
        out_scores.append(score)
        seq = torch.cat([seq, nxt], dim=1)
        if bool(finished.all()):
            break

    return torch.cat(out_tokens, dim=1), torch.cat(out_scores, dim=1)
