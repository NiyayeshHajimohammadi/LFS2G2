"""End-to-end recall evaluation for a set of (gt, prediction) graphs.

Reproduces LF-SGG's evaluation: dedup + top-K predictions, branched instance matching, then
Recall@K / mean-Recall@K plus the instance-agnostic Triplet / Entity / Pred set recalls.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

from patchsgg.eval.matcher import InstanceMatcher
from patchsgg.eval.metrics import (
    apply_mapping_to_predictions,
    global_recall,
    mean_over_predicates,
    mean_recall_helper,
    set_recall,
)

MatcherTuple = Tuple[int, int, int, int, int]


def _dedup_keep_order(preds: Sequence[MatcherTuple]) -> List[MatcherTuple]:
    seen = set()
    out: List[MatcherTuple] = []
    for p in preds:
        tp = tuple(p)
        if tp not in seen:
            seen.add(tp)
            out.append(tp)
    return out


def evaluate_graphs(
    samples: Sequence[Tuple[Sequence[MatcherTuple], Sequence[MatcherTuple]]],
    ks: Sequence[int] = (20, 50, 100),
    matcher: InstanceMatcher | None = None,
) -> Dict[str, float]:
    """``samples`` = list of ``(gt_tuples, pred_tuples)`` ordered by descending pred score.

    Returns a flat dict of metric_name -> value.
    """
    matcher = matcher or InstanceMatcher()
    recalls: Dict[str, List[float]] = {f"R@{k}": [] for k in ks}
    mrecall_acc: Dict[int, Dict[int, List[float]]] = {k: defaultdict(list) for k in ks}
    for kind in ("triplet", "entity", "pred"):
        recalls[f"set/{kind}"] = []

    for gt, pred in samples:
        if len(gt) == 0:
            continue
        for kind in ("triplet", "entity", "pred"):
            recalls[f"set/{kind}"].append(set_recall(gt, pred, kind))

        unique_pred = _dedup_keep_order(pred)
        for k in ks:
            topk = (unique_pred[:k] if len(unique_pred) >= k else list(pred[:k]))[:k]
            mapping = matcher.match(list(gt), list(topk))
            mapped = apply_mapping_to_predictions(topk, mapping)
            recalls[f"R@{k}"].append(global_recall(gt, mapped))
            for pid, vals in mean_recall_helper(gt, mapped).items():
                mrecall_acc[k][pid].extend(vals)

    out: Dict[str, float] = {name: (float(np.mean(v)) if v else 0.0) for name, v in recalls.items()}
    for k in ks:
        out[f"mR@{k}"] = mean_over_predicates(mrecall_acc[k])
    return out
