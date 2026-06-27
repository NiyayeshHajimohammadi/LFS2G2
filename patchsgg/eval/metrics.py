"""Recall metrics for location-free scene graphs.

Direct ports of LF-SGG ``_calculate_global_recall`` / ``_mean_recall_helper`` /
``_apply_mapping_to_predictions`` (pure Python, no torch). Tuples are matcher tuples:
``(subj_entity_token, subj_instance, predicate_token, obj_entity_token, obj_instance)``.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

MatcherTuple = Tuple[int, int, int, int, int]


def global_recall(gts: Sequence[MatcherTuple], preds: Sequence[MatcherTuple]) -> float:
    """Exact-match recall over full 5-tuples, counting duplicates by multiplicity."""
    if len(gts) == 0:
        return 0.0
    gts_dict: Dict[tuple, int] = defaultdict(int)
    for g in gts:
        gts_dict[tuple(g)] += 1
    preds_dict: Dict[tuple, int] = defaultdict(int)
    for p in preds:
        preds_dict[tuple(p)] += 1
    correct = 0.0
    for gt_rel, gt_count in gts_dict.items():
        if gt_rel in preds_dict:
            correct += min(gt_count, preds_dict[gt_rel])
    return correct / len(gts)


def mean_recall_helper(
    gts: Sequence[MatcherTuple], preds: Sequence[MatcherTuple]
) -> Dict[int, List[float]]:
    """Per-predicate recall contributions (predicate id = tuple index 2).

    Returns ``{predicate: [recall]*count}`` so a later global mean weights predicates equally.
    """
    gts_dict: Dict[tuple, int] = defaultdict(int)
    for g in gts:
        gts_dict[tuple(g)] += 1
    preds_dict: Dict[tuple, int] = defaultdict(int)
    for p in preds:
        preds_dict[tuple(p)] += 1

    per_predicate: Dict[int, List[int]] = defaultdict(list)
    gt_predicate_count: Dict[int, int] = defaultdict(int)
    for gt_rel, gt_count in gts_dict.items():
        gt_predicate_count[gt_rel[2]] += gt_count
        if gt_rel in preds_dict:
            per_predicate[gt_rel[2]].append(min(gt_count, preds_dict[gt_rel]))
        else:
            per_predicate[gt_rel[2]].append(0)

    out: Dict[int, List[float]] = {}
    for pred_id, hits in per_predicate.items():
        recall = sum(hits) / gt_predicate_count[pred_id]
        out[pred_id] = [recall] * gt_predicate_count[pred_id]
    return out


def apply_mapping_to_predictions(
    predictions: Sequence[MatcherTuple],
    mapping: Dict[Tuple[int, int], Optional[int]],
) -> List[MatcherTuple]:
    """Remap predicted instance ids using a matcher mapping ``{(entity_token, inst): gt_inst}``.

    Unmatched (entity, instance) pairs get instance id ``None`` (never matches a GT tuple).
    """
    mapped: List[MatcherTuple] = []
    for sub_id, sub_inst, pred_id, obj_id, obj_inst in predictions:
        new_sub = mapping.get((sub_id, sub_inst), None)
        new_obj = mapping.get((obj_id, obj_inst), None)
        mapped.append((sub_id, new_sub, pred_id, obj_id, new_obj))
    return mapped


def set_recall(gts: Sequence[MatcherTuple], preds: Sequence[MatcherTuple], kind: str) -> float:
    """Location-free set recalls that ignore instance ids (LF-SGG 'Triplet'/'Entity'/'Pred').

    kind='triplet' -> (sub_cls, pred, obj_cls); 'entity' -> (sub_cls, obj_cls); 'pred' -> pred.
    """
    if len(gts) == 0:
        return 0.0
    if kind == "triplet":
        g = {(t[0], t[2], t[3]) for t in gts}
        p = {(t[0], t[2], t[3]) for t in preds}
    elif kind == "entity":
        g = {(t[0], t[3]) for t in gts}
        p = {(t[0], t[3]) for t in preds}
    elif kind == "pred":
        g = {t[2] for t in gts}
        p = {t[2] for t in preds}
    else:
        raise ValueError(kind)
    return len(g & p) / len(g)


def mean_over_predicates(per_predicate_recalls: Dict[int, List[float]]) -> float:
    """Mean over predicate classes (each class contributes its mean recall)."""
    import numpy as np

    if not per_predicate_recalls:
        return 0.0
    return float(np.mean([np.mean(v) for v in per_predicate_recalls.values()]))
