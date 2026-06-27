"""Instance-permutation-invariant cross-entropy (the novel loss).

Instance ids are arbitrary labels: ``(man,1,riding,horse,1)`` and ``(man,2,riding,horse,2)`` are the
same graph. Plain CE penalises the model for choosing a different-but-equivalent id. Here we run the
*same* branched matcher used at eval to align the target's instance ids to the model's predicted
assignment, then apply CE on the relabelled target. Only instance tokens are rewritten.

NOTE: the matcher runs per sample on CPU; for large-scale training consider a cheaper alignment
(e.g. first-occurrence canonicalization). Kept here as the principled reference loss.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from patchsgg.eval.matcher import InstanceMatcher
from patchsgg.graph_seq.vocab import TOKENS_PER_REL, GraphVocab, TokenType
from patchsgg.losses.base import Loss


class MatchingCELoss(Loss):
    def __init__(self, vocab, weight=None, label_smoothing=0.0, matcher_n=3, matcher_depth=10):
        super().__init__(vocab, weight=weight, label_smoothing=label_smoothing)
        self.matcher = InstanceMatcher(n=matcher_n, depth_limit=matcher_depth)

    def _role_argmax(self, logits_row: torch.Tensor, position: int) -> int:
        role = self.vocab.role_at(position)
        lo, hi = self.vocab.range_for_role(role)
        return lo + int(logits_row[lo:hi].argmax().item())

    def _blocks(self, tokens: List[int]) -> int:
        """Number of complete 5-token tuples before END/padding."""
        if self.vocab.end_token in tokens:
            tokens = tokens[: tokens.index(self.vocab.end_token)]
        return len(tokens) // TOKENS_PER_REL

    def _matcher_tuple(self, block: List[int]) -> Tuple[int, int, int, int, int]:
        sub_cls, sub_inst, obj_cls, obj_inst, pred = block
        return (sub_cls, self.vocab.instance_idx(sub_inst), self.vocab.predicate_idx(pred),
                obj_cls, self.vocab.instance_idx(obj_inst))

    @torch.no_grad()
    def _relabel_target(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        new_target = target.clone()
        B, T, _ = logits.shape
        for b in range(B):
            tgt = target[b].tolist()
            n_blocks = self._blocks(tgt)
            if n_blocks == 0:
                continue
            # build gt + predicted (role-restricted argmax) tuples, block-aligned
            gt_tuples, pred_tuples = [], []
            for k in range(n_blocks):
                s = k * TOKENS_PER_REL
                gt_block = tgt[s : s + TOKENS_PER_REL]
                pred_block = [self._role_argmax(logits[b, s + j], s + j) for j in range(TOKENS_PER_REL)]
                gt_tuples.append(self._matcher_tuple(gt_block))
                pred_tuples.append(self._matcher_tuple(pred_block))
            mapping = self.matcher.match(gt_tuples, pred_tuples)  # (entity, pred_inst) -> gt_inst
            # invert: (entity, gt_inst) -> pred_inst
            want: Dict[Tuple[int, int], int] = {}
            for (entity, pred_inst), gt_inst in mapping.items():
                if gt_inst is not None:
                    want[(entity, gt_inst)] = pred_inst
            # rewrite instance tokens in the target
            for k in range(n_blocks):
                s = k * TOKENS_PER_REL
                for j in (1, 3):  # subject / object instance positions
                    entity = tgt[s + j - 1]
                    gt_inst = self.vocab.instance_idx(tgt[s + j])
                    if (entity, gt_inst) in want:
                        new_target[b, s + j] = self.vocab.instance_token(want[(entity, gt_inst)])
        return new_target

    def forward(self, logits, target, input_tokens=None):
        relabeled = self._relabel_target(logits.detach(), target)
        return self._ce(logits, relabeled)
