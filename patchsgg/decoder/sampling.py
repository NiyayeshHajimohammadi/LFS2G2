"""Constrained, token-type-aware autoregressive generation.

The generation loop is decoder-agnostic. Decoder-specific code only produces
next-token logits; this module enforces the Location-Free SGG five-token grammar:

    subject entity, subject instance, object entity, object instance, predicate

Entity positions may use stochastic top-k/top-p sampling. Instance and predicate
positions are greedy. END is permitted only at a relation boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Tuple

import torch

from patchsgg.graph_seq.vocab import TOKENS_PER_REL, GraphVocab, TokenType


@dataclass
class GenConfig:
    """Inference-time generation settings."""

    max_rels: int = 100
    temperature: float = 1.75
    top_p: float = 0.95
    top_k: int = 0
    entity_sampling: str = "stochastic"  # stochastic | greedy
    allow_end: bool = True


FullPrefixStep = Callable[[torch.Tensor], torch.Tensor]
StatefulStep = Callable[[torch.Tensor, Any], Tuple[torch.Tensor, Any]]


def top_k_top_p_filter(
    logits: torch.Tensor,
    top_k: int = 0,
    top_p: float = 0.0,
) -> torch.Tensor:
    """Mask logits outside the top-k or nucleus set with ``-inf``.

    Parameters
    ----------
    logits:
        Full-vocabulary logits with shape ``[B, V]``.
    top_k:
        Number of highest-logit candidates to retain. Zero disables top-k.
    top_p:
        Nucleus probability threshold. Zero disables top-p.
    """
    if logits.ndim != 2:
        raise ValueError(f"Expected logits [B, V], got {tuple(logits.shape)}")
    if top_k < 0:
        raise ValueError(f"top_k must be non-negative, got {top_k}")
    if top_p < 0.0 or top_p > 1.0:
        raise ValueError(f"top_p must be in [0, 1], got {top_p}")

    filtered = logits.clone()

    if top_k > 0:
        k = min(int(top_k), filtered.shape[-1])
        threshold = torch.topk(filtered, k, dim=-1).values[:, -1, None]
        filtered = filtered.masked_fill(filtered < threshold, float("-inf"))

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        cumulative_probability = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

        remove_sorted = cumulative_probability > float(top_p)
        # Keep the first token that crosses the threshold.
        remove_sorted[:, 1:] = remove_sorted[:, :-1].clone()
        remove_sorted[:, 0] = False

        remove_original_order = torch.zeros_like(remove_sorted).scatter(
            dim=1,
            index=sorted_indices,
            src=remove_sorted,
        )
        filtered = filtered.masked_fill(remove_original_order, float("-inf"))

    return filtered


def _select_range(logits: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """Keep only logits in the half-open vocabulary interval ``[lo, hi)``."""
    masked = torch.full_like(logits, float("-inf"))
    masked[:, lo:hi] = logits[:, lo:hi]
    return masked


@torch.no_grad()
def sample_constrained_token(
    logits: torch.Tensor,
    step_index: int,
    vocab: GraphVocab,
    cfg: GenConfig,
    finished: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Choose one next token while enforcing the LF-SGG token grammar.

    Returns ``(next_token[B,1], selected_probability[B,1], finished[B])``.
    """
    if logits.ndim != 2:
        raise ValueError(f"Expected logits [B, V], got {tuple(logits.shape)}")
    if logits.shape[-1] != vocab.vocab_size:
        raise ValueError(
            f"Expected vocabulary dimension {vocab.vocab_size}, got {logits.shape[-1]}"
        )
    if finished.ndim != 1 or finished.shape[0] != logits.shape[0]:
        raise ValueError("finished must have shape [B] matching the logits batch")
    if cfg.entity_sampling not in {"stochastic", "greedy"}:
        raise ValueError(
            "entity_sampling must be 'stochastic' or 'greedy', "
            f"got {cfg.entity_sampling!r}"
        )

    role = vocab.role_at(step_index)
    lo, hi = vocab.range_for_role(role)
    masked_logits = _select_range(logits, lo, hi)

    at_relation_boundary = step_index % TOKENS_PER_REL == 0
    if cfg.allow_end and at_relation_boundary:
        masked_logits[:, vocab.end_token] = logits[:, vocab.end_token]

    if role is TokenType.ENTITY and cfg.entity_sampling == "stochastic":
        temperature = float(cfg.temperature)
        if temperature <= 0:
            raise ValueError(f"temperature must be greater than zero, got {temperature}")

        filtered_logits = top_k_top_p_filter(
            masked_logits / temperature,
            top_k=int(cfg.top_k),
            top_p=float(cfg.top_p),
        )
        probabilities = torch.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
    else:
        probabilities = torch.softmax(masked_logits, dim=-1)
        next_token = probabilities.argmax(dim=-1, keepdim=True)

    selected_probability = probabilities.gather(dim=-1, index=next_token)
    was_finished = finished
    newly_finished = next_token.squeeze(1) == vocab.end_token
    finished = was_finished | newly_finished

    # Keep tensor shapes aligned while other batch elements continue generating.
    next_token = next_token.masked_fill(
        was_finished.unsqueeze(1),
        vocab.no_known_token,
    )
    selected_probability = selected_probability.masked_fill(
        was_finished.unsqueeze(1),
        0.0,
    )

    return next_token, selected_probability, finished


@torch.no_grad()
def constrained_generate(
    step_fn: FullPrefixStep | StatefulStep,
    start_tokens: torch.Tensor,
    vocab: GraphVocab,
    cfg: GenConfig,
    *,
    initial_state: Any = None,
    stateful: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a structured scene-graph sequence.

    Two decoder styles are supported:

    * ``stateful=False`` (backward compatible):
      ``step_fn(full_sequence[B,T]) -> logits[B,V]``.
    * ``stateful=True`` (for GPT-2 caching):
      ``step_fn(full_sequence[B,T], state) -> (logits[B,V], new_state)``.

    Returned sequences exclude the START token. Scores are selected-token
    probabilities and have the same shape as the generated token tensor.
    """
    if start_tokens.ndim != 2 or start_tokens.shape[1] != 1:
        raise ValueError(
            "start_tokens must have shape [B, 1], "
            f"got {tuple(start_tokens.shape)}"
        )

    max_rels = int(cfg.max_rels)
    if max_rels < 0:
        raise ValueError(f"max_rels must be non-negative, got {max_rels}")

    batch_size = start_tokens.shape[0]
    device = start_tokens.device
    total_steps = max_rels * TOKENS_PER_REL

    if total_steps == 0:
        return (
            torch.empty(batch_size, 0, dtype=torch.long, device=device),
            torch.empty(batch_size, 0, dtype=torch.float, device=device),
        )

    sequence = start_tokens
    state = initial_state
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    output_tokens: list[torch.Tensor] = []
    output_scores: list[torch.Tensor] = []

    for step_index in range(total_steps):
        if stateful:
            logits, state = step_fn(sequence, state)  # type: ignore[misc]
        else:
            logits = step_fn(sequence)  # type: ignore[misc]

        next_token, score, finished = sample_constrained_token(
            logits=logits,
            step_index=step_index,
            vocab=vocab,
            cfg=cfg,
            finished=finished,
        )

        output_tokens.append(next_token)
        output_scores.append(score)
        sequence = torch.cat([sequence, next_token], dim=1)

        if bool(finished.all()):
            break

    return (
        torch.cat(output_tokens, dim=1),
        torch.cat(output_scores, dim=1),
    )
