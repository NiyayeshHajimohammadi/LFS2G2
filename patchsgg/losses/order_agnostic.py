"""Order-agnostic cross-entropy.

A scene graph is a *set*, but teacher forcing imposes the dataset's tuple order. Here we reorder the
GT tuples to best match the model's predicted tuples (role-restricted argmax) before CE, removing
the spurious penalty for predicting the right tuples in a different order.

Assignment is a small linear-assignment problem per sample (n = #tuples <= max_num_rels). We build
the cost matrix with vectorized tensor ops (no n^2 Python loop) and solve it with the Hungarian
algorithm (scipy if available, otherwise a vectorized greedy fallback). Only the per-sample loop
over the batch remains.
"""
from __future__ import annotations

from typing import List

import torch

from patchsgg.graph_seq.vocab import TOKENS_PER_REL, TokenType
from patchsgg.losses.base import Loss

try:  # optional, gives optimal assignment
    from scipy.optimize import linear_sum_assignment as _hungarian

    _HAS_SCIPY = True
except Exception:  # pragma: no cover
    _HAS_SCIPY = False


def _assign(cost: torch.Tensor) -> torch.Tensor:
    """Return ``order`` s.t. predicted block ``i`` is matched to GT block ``order[i]``.

    ``cost``: [n, n] (rows = predicted blocks, cols = GT blocks), lower is better.
    """
    n = cost.shape[0]
    if _HAS_SCIPY:
        _, col = _hungarian(cost.detach().cpu().numpy())
        return torch.as_tensor(col, device=cost.device, dtype=torch.long)
    # vectorized greedy fallback: each pred row picks its cheapest unused GT col
    order = torch.empty(n, dtype=torch.long, device=cost.device)
    used = torch.zeros(n, dtype=torch.bool, device=cost.device)
    big = cost.max() + 1.0
    for i in range(n):
        row = cost[i].masked_fill(used, big)
        j = int(row.argmin())
        order[i] = j
        used[j] = True
    return order


class OrderAgnosticCELoss(Loss):
    def _pred_blocks(self, logits_b: torch.Tensor, n_blocks: int) -> torch.Tensor:
        """Role-restricted argmax over the first ``n_blocks`` tuples -> [n_blocks, 5] (vectorized)."""
        pred = torch.empty((n_blocks, TOKENS_PER_REL), dtype=torch.long, device=logits_b.device)
        block_pos = torch.arange(n_blocks, device=logits_b.device) * TOKENS_PER_REL
        for j in range(TOKENS_PER_REL):
            lo, hi = self.vocab.range_for_role(self.vocab.role_at(j))
            sl = logits_b[block_pos + j, lo:hi]          # [n_blocks, range]
            pred[:, j] = sl.argmax(dim=-1) + lo
        return pred

    @torch.no_grad()
    def _reorder_target(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        new_target = target.clone()
        B = logits.shape[0]
        for b in range(B):
            tgt = target[b]
            row = tgt.tolist()
            n_blocks = (row.index(self.vocab.end_token) // TOKENS_PER_REL) if self.vocab.end_token in row else 0
            if n_blocks <= 1:
                continue
            gt_blocks = tgt[: n_blocks * TOKENS_PER_REL].view(n_blocks, TOKENS_PER_REL)
            pred_blocks = self._pred_blocks(logits[b], n_blocks)
            # cost[i, j] = #mismatched fields between pred block i and gt block j
            matches = (pred_blocks[:, None, :] == gt_blocks[None, :, :]).sum(-1)  # [n, n]
            cost = (TOKENS_PER_REL - matches).float()
            order = _assign(cost)
            new_target[b, : n_blocks * TOKENS_PER_REL] = gt_blocks[order].reshape(-1)
        return new_target

    def forward(self, logits, target, input_tokens=None):
        reordered = self._reorder_target(logits.detach(), target)
        return self._ce(logits, reordered)
