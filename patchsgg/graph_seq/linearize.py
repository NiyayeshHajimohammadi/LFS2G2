"""Graph <-> token-sequence conversion.

A clean *semantic* relation representation (0-based class / instance indices) plus explicit,
tested converters to (a) the AR token sequence and (b) the LF-SGG matcher/metrics tuple. This
reimplements LF-SGG ``preprocess_input`` / ``postprocess_output`` in a model-agnostic way.
"""
from __future__ import annotations

from typing import List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from patchsgg.graph_seq.vocab import VG_VOCAB, GraphVocab, TOKENS_PER_REL


class Relation(NamedTuple):
    """Semantic 5-tuple, all fields 0-based in dataset space."""

    subj_cls: int
    subj_inst: int
    predicate: int
    obj_cls: int
    obj_inst: int


Graph = List[Relation]
# Matcher/metrics tuple, mirroring LF-SGG eval format:
#   (subj_entity_token, subj_instance_idx, predicate_token, obj_entity_token, obj_instance_idx)
MatcherTuple = Tuple[int, int, int, int, int]


# --------------------------------------------------------------------------------------------
# single-relation conversions
# --------------------------------------------------------------------------------------------
def relation_to_tokens(rel: Relation, vocab: GraphVocab = VG_VOCAB) -> List[int]:
    """5 token ids in *sequence* order: sub_cls, sub_inst, obj_cls, obj_inst, predicate."""
    #My comment: This function converts one semantic relationship into five integer tokens.
    return [
        vocab.entity_token(rel.subj_cls),
        vocab.instance_token(rel.subj_inst),
        vocab.entity_token(rel.obj_cls),
        vocab.instance_token(rel.obj_inst),
        vocab.predicate_token(rel.predicate),
    ]


def tokens_to_relation(tok: Sequence[int], vocab: GraphVocab = VG_VOCAB) -> Relation:
    """Inverse of :func:`relation_to_tokens` for one 5-token block (sequence order)."""
    #My comment: converts those token IDs back into a meaningful semantic relation.
    sub_cls, sub_inst, obj_cls, obj_inst, pred = tok
    return Relation(
        subj_cls=vocab.entity_idx(int(sub_cls)),
        subj_inst=vocab.instance_idx(int(sub_inst)),
        predicate=vocab.predicate_idx(int(pred)),
        obj_cls=vocab.entity_idx(int(obj_cls)),
        obj_inst=vocab.instance_idx(int(obj_inst)),
    )


def canonical_tuple(rel: Relation, vocab: GraphVocab = VG_VOCAB) -> MatcherTuple:
    """LF-SGG matcher/metrics tuple (entities in token space, preds/instances 0-based)."""
    #My comment: This function performs that reordering without changing the semantic meaning.
    return (
        vocab.entity_token(rel.subj_cls),
        rel.subj_inst,
        vocab.predicate_token(rel.predicate),  # PRED_START == 0 -> already 0-based
        vocab.entity_token(rel.obj_cls),
        rel.obj_inst,
    )


def graph_to_matcher_tuples(graph: Graph, vocab: GraphVocab = VG_VOCAB) -> List[MatcherTuple]:
    return [canonical_tuple(r, vocab) for r in graph] #My comment: list comprehension.


# --------------------------------------------------------------------------------------------
# full-graph conversions
# --------------------------------------------------------------------------------------------
def graph_to_sequence(
    graph: Graph,
    vocab: GraphVocab = VG_VOCAB,
    add_start: bool = True,
    add_end: bool = True,
) -> List[int]:
    """Flatten a graph to a token sequence (no padding). Used for eval / quick round-trips."""
    #My commet: This converts an entire graph into a single autoregressive sequence.
    seq: List[int] = [vocab.start_token] if add_start else []
    for rel in graph[: vocab.max_num_rels]:
        seq.extend(relation_to_tokens(rel, vocab))
    if add_end:
        seq.append(vocab.end_token)
    return seq


def sequence_to_graph(tokens: Sequence[int], vocab: GraphVocab = VG_VOCAB) -> Graph:
    """Parse a (possibly generated) token sequence into relations.

    Robust to a leading START token and a trailing END / padding: stops at the first END token
    and ignores a trailing partial (<5 tokens) block.
    """
    #This converts a single autoregressive sequence into an entire graph.
    toks = list(int(t) for t in tokens)
    if toks and toks[0] == vocab.start_token:
        toks = toks[1:]
    if vocab.end_token in toks:
        toks = toks[: toks.index(vocab.end_token)]
    graph: Graph = []
    for i in range(0, len(toks) - TOKENS_PER_REL + 1, TOKENS_PER_REL):
        block = toks[i : i + TOKENS_PER_REL]
        graph.append(tokens_to_relation(block, vocab))
    return graph


# --------------------------------------------------------------------------------------------
# training pair (teacher forcing) -- faithful port of LF-SGG preprocess_input(train)
# --------------------------------------------------------------------------------------------
def build_train_pair(
    graph: Graph,
    vocab: GraphVocab = VG_VOCAB,
    rng: Optional[np.random.Generator] = None,
    pad_to_max: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(input_seq, target_seq)`` int arrays of equal length for teacher forcing.

    Mirrors LF-SGG: real tuples are followed by random "noise" tuples on the *input* side; the
    *target* side ends the real graph with END and fills padded positions with
    ``[NO_KNOWN, NO_KNOWN, NO_KNOWN, NOISE, END]`` so that NO_KNOWN positions can be masked out
    of the loss while the model still learns to emit END after the real graph.
    """
    #My comment: It creates the input and target sequences for teacher forcing.
    rng = rng or np.random.default_rng()
    real = graph[: vocab.max_num_rels]
    n_real = len(real)
    n_pad = vocab.max_num_rels - n_real if pad_to_max else 0

    real_tokens: List[int] = []
    for rel in real:
        real_tokens.extend(relation_to_tokens(rel, vocab))

    # random padding tuples on the input side (sequence order: sub_cls, sub_inst, obj_cls, obj_inst, pred)
    e_lo, e_hi = vocab.entity_range
    p_lo, p_hi = vocab.pred_range
    rand_tokens: List[int] = []
    for _ in range(n_pad):
        rand_tokens.extend(
            [
                int(rng.integers(e_lo, e_hi)),                                        # sub_cls
                vocab.instance_start + int(rng.integers(0, vocab.random_max_instance_id)),  # sub_inst
                int(rng.integers(e_lo, e_hi)),                                        # obj_cls
                vocab.instance_start + int(rng.integers(0, vocab.random_max_instance_id)),  # obj_inst
                int(rng.integers(p_lo, p_hi)),                                        # pred
            ]
        )

    input_seq = [vocab.start_token] + real_tokens + rand_tokens

    # target side
    target_seq = list(real_tokens) + [vocab.end_token]
    pad_block = [vocab.no_known_token, vocab.no_known_token, vocab.no_known_token,
                 vocab.noise_token, vocab.end_token]
    for _ in range(n_pad):
        target_seq.extend(pad_block)

    input_arr = np.asarray(input_seq, dtype=np.int64)
    target_arr = np.asarray(target_seq, dtype=np.int64)
    assert len(input_arr) == len(target_arr), (len(input_arr), len(target_arr))
    return input_arr, target_arr

# Problems: 
# No range validation when constructing or decoding relations. Invalid indices silently produce invalid tokens.
# Hard truncation to max_num_rels discards information for dense scenes without warning.
# Strict 5-token alignment means one decoding error can corrupt the parsing of every subsequent relation.
# Training depends on correct loss masking for NO_KNOWN. If the loss function later forgets to ignore these tokens, training quality could degrade significantly.
# Lack of explicit unit tests for round-trip guarantees such as sequence_to_graph(graph_to_sequence(g)) == g, which are critical invariants for this module.
